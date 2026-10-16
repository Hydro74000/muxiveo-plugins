"""Gardes du paquet GPU : refuser les attestations antérieures ou incomplètes."""
import copy
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("fel_package", Path(__file__).parents[1] / "scripts/package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.library = Path(self.temp.name) / "library.so"
        self.library.write_bytes(b"bibliotheque de test")
        self.validation = Path(self.temp.name) / "validation.json"
        def mode(name: str) -> dict:
            result = {"max_cpu_pq12": 0.0 if name == "exact" else 1.0, "max_reference_pq12": 1.1,
                      "max_fast_codes16": {"fast": 16, "standard": 1, "exact": 0}[name],
                      "producer": {"backend": "vulkan: test", "frame_sha256": ["b"*64]*3},
                      "yuv": [{"chroma": "topleft", "dither": "none", "crop": [2, 4, 6, 8], "max_code10": .5,
                               "frame_sha256": ["d"*64]*3},
                              {"chroma": "left", "dither": "blue", "crop": [0, 0, 0, 0], "max_code10": .9999,
                               "frame_sha256": ["e"*64]*3}]}
            if name == "exact":
                result["cpu_identical"] = True
                result["producer"].update(certified_rescued=12, certified_undecided=0)
                result["cpu_reference"] = {"certified_rescued": 30, "certified_undecided": 0}
                result["variant"] = "fixed"
                result["single_worker"] = {"backend": "vulkan: test", "identical": True}
            return result
        self.report = {
            "schema": 5, "passed": True, "tolerance_pq12": 2, "tolerance_codes16": {"standard": 1, "exact": 0},
            "library_sha256": package.digest(self.library),
            "engine_sources": package.engine_sources(),
            "dependencies": json.loads((package.ROOT / "dependencies.json").read_text()),
            "samples": [{"device": {"uuid": "a"*32, "float64": True}, "clip_sha256": str(index)*64, "frames": 3,
                         "modes": {name: mode(name) for name in ("fast", "standard", "exact")}}
                        for index in range(3)],
        }
        self.original = copy.deepcopy(self.report)
        lib = Mock()
        lib.mvo_fel_abi_version.return_value = 1
        lib.mvo_fel_capabilities.return_value = b'{"experimental":true}'
        patcher = patch.object(package.ctypes, "CDLL", return_value=lib)
        patcher.start()
        self.addCleanup(patcher.stop)

    def load(self):
        self.validation.write_text(json.dumps(self.report))
        return package.load_capabilities(self.library, self.validation)

    def test_valid_attestation(self):
        self.assertTrue(self.load()["experimental"])

    def test_old_attestation_is_rejected(self):
        self.report["schema"] = 4
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.load()

    def test_changed_engine_sources_are_rejected(self):
        self.report["engine_sources"]["src/shaders/fixed_render.glsl"] = "0"*64
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.load()

    def test_undecided_components_are_rejected(self):
        for key in ("producer", "cpu_reference"):
            with self.subTest(key=key):
                self.report = copy.deepcopy(self.original)
                self.report["samples"][1]["modes"]["exact"][key]["certified_undecided"] = 1
                with self.assertRaisesRegex(RuntimeError, "certifié"):
                    self.load()

    def test_exact_variant_and_single_worker_are_required(self):
        for key in ("variant", "single_worker", "cpu_reference"):
            with self.subTest(key=key):
                self.report = copy.deepcopy(self.original)
                del self.report["samples"][2]["modes"]["exact"][key]
                with self.assertRaisesRegex(RuntimeError, "certifié"):
                    self.load()

    def test_different_library_is_rejected(self):
        self.library.write_bytes(b"autre bibliotheque")
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.load()

    def test_missing_producer_pixels_are_rejected(self):
        del self.report["samples"][0]["modes"]["fast"]["producer"]
        with self.assertRaisesRegex(RuntimeError, "pixels"):
            self.load()

    def test_cpu_fallback_is_rejected(self):
        self.report["samples"][0]["modes"]["exact"]["producer"]["backend"] = "cpu"
        with self.assertRaisesRegex(RuntimeError, "pixels"):
            self.load()

    def test_pq_threshold_is_not_relaxed(self):
        self.report["samples"][0]["modes"]["fast"]["max_reference_pq12"] = 2.001
        with self.assertRaisesRegex(RuntimeError, "PQ12"):
            self.load()

    def test_precise_modes_are_required_on_float64_gpus(self):
        del self.report["samples"][2]["modes"]["standard"]
        with self.assertRaisesRegex(RuntimeError, "Précisions"):
            self.load()

    def test_exact_mode_must_match_cpu(self):
        self.report["samples"][0]["modes"]["exact"]["max_fast_codes16"] = 1
        with self.assertRaisesRegex(RuntimeError, "exact"):
            self.load()

    def test_standard_mode_bound_is_one_code(self):
        self.report["samples"][1]["modes"]["standard"]["max_fast_codes16"] = 2
        with self.assertRaisesRegex(RuntimeError, "standard"):
            self.load()

    def test_gpu_without_float64_needs_fast_mode_only(self):
        for sample in self.report["samples"]:
            sample["device"]["float64"] = False
            del sample["modes"]["standard"], sample["modes"]["exact"]
        self.load()

    def test_final_yuv_outputs_are_required(self):
        self.report["samples"][0]["modes"]["standard"]["yuv"].pop()
        with self.assertRaisesRegex(RuntimeError, "Y'CbCr"):
            self.load()

    def test_yuv_rounding_is_not_relaxed(self):
        self.report["samples"][1]["modes"]["fast"]["yuv"][0]["max_code10"] = .6
        with self.assertRaisesRegex(RuntimeError, "Y'CbCr"):
            self.load()

    def test_direct_binary_and_pixels_are_required(self):
        binary = self.library.with_name("ffmpeg-fel")
        binary.write_bytes(b"filtre de test")
        self.report["direct"] = {"binary_sha256": package.digest(binary),
            "filter_sources": {p.name: package.digest(p) for p in sorted((package.ROOT/'integrations/ffmpeg').glob('*.c'))}}
        for sample in self.report["samples"]:
            for result in sample["modes"].values():
                result["direct"] = {"transport": "vulkan", "frames_read": 17,
                                    "frame_sha256": result["producer"]["frame_sha256"],
                                    "yuv": [case["frame_sha256"] for case in result["yuv"]]}
        self.load()
        package.check_direct(binary, self.validation)
        exact = self.report["samples"][0]["modes"]["exact"]
        exact["direct"]["yuv"][1] = ["f"*64]*3
        self.load()
        with self.assertRaisesRegex(RuntimeError, "Pixels"):
            package.check_direct(binary, self.validation)
        exact["direct"]["yuv"][1] = exact["yuv"][1]["frame_sha256"]
        self.report["samples"][0]["modes"]["fast"]["direct"]["frame_sha256"] = ["c"*64]*3
        self.load()
        with self.assertRaisesRegex(RuntimeError, "Pixels"):
            package.check_direct(binary, self.validation)
        binary.write_bytes(b"autre filtre")
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            package.check_direct(binary, self.validation)

    def test_glibc_baseline_rejects_local_runtime(self):
        with patch.object(package.subprocess, "run", return_value=Mock(stdout="Name: GLIBC_2.28\nName: GLIBC_2.2.5")):
            package.check_glibc(self.library, "2.28")
        for version in ("GLIBC_2.43", "GLIBC_ABI_GNU2_TLS", "GLIBC_PRIVATE"):
            with patch.object(package.subprocess, "run", return_value=Mock(stdout="GLIBC_2.2.5\n"+version)):
                with self.assertRaisesRegex(RuntimeError, "ABI glibc"):
                    package.check_glibc(self.library, "2.28")

    def test_windows_dll_imports_only_system_libraries(self):
        imports = "DLL Name: KERNEL32.dll\nDLL Name: api-ms-win-crt-runtime-l1-1-0.dll\nDLL Name: bcrypt.dll\n"
        with patch.object(package.subprocess, "run", return_value=Mock(stdout=imports)):
            package.check_windows_imports(self.library)
        for foreign in ("libstdc++-6.dll", "vulkan-1.dll", "libwinpthread-1.dll"):
            with patch.object(package.subprocess, "run", return_value=Mock(stdout=imports + "DLL Name: " + foreign)):
                with self.assertRaisesRegex(RuntimeError, "non embarquées"):
                    package.check_windows_imports(self.library)


class WindowsPackageTests(unittest.TestCase):
    """Parcours complet de main() pour une DLL GPU Windows attestée, sur une racine factice."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        work = Path(self.temp.name)
        self.root = work / "mvo-fel"
        (self.root / "LICENSES").mkdir(parents=True)
        (self.root / "LICENSES/MIT.txt").write_text("MIT")
        (self.root / "CMakeLists.txt").write_text("project(mvo-fel VERSION 9.8.7 LANGUAGES C CXX)")
        (self.root / "NOTICES.md").write_text("avis")
        (self.root / "dependencies.json").write_text("{}")
        self.build = work / "build"
        (self.build / "plugin").mkdir(parents=True)
        (self.build / "sources").mkdir()
        (self.build / "ffmpeg").mkdir()
        (self.build / "ffmpeg/config.h").write_text("/* config */")
        (self.build / "plugin/mvo_fel.dll").write_bytes(b"MZ dll de test")
        self.validation = work / "validation.json"
        self.validation.write_text("{}")
        self.output = work / "dist"
        self.calls = []

    def run_main(self, imports):
        def run(command, **kwargs):
            self.calls.append(command[0])
            if command[0] == "readelf":
                raise AssertionError("readelf appelé sur une DLL PE")
            return Mock(stdout=imports)
        argv = ["package.py", "windows-x86_64", str(self.build), str(self.output), "--validation", str(self.validation)]
        with patch.object(package, "ROOT", self.root), patch.object(package.sys, "argv", argv), \
                patch.object(package, "load_capabilities", return_value={"experimental": True}), \
                patch.object(package.subprocess, "run", side_effect=run):
            package.main()

    def test_gpu_dll_is_packaged_without_elf_checks(self):
        self.run_main("DLL Name: KERNEL32.dll\nDLL Name: bcrypt.dll\n")
        self.assertEqual(self.calls[0], "objdump")
        self.assertNotIn("readelf", self.calls)
        name = "mvo-fel-9.8.7-windows-x86_64-prototype"
        with zipfile.ZipFile(self.output / f"{name}.zip") as archive:
            files = set(archive.namelist())
        for expected in ("mvo_fel.dll", "gpu-validation.json", "manifest.json", "sources.tar.gz"):
            self.assertIn(f"{name}/{expected}", files)

    def test_foreign_dll_import_stops_packaging(self):
        with self.assertRaisesRegex(RuntimeError, "non embarquées"):
            self.run_main("DLL Name: KERNEL32.dll\nDLL Name: vulkan-1.dll\n")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
