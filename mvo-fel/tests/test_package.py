"""Gardes du paquet GPU : refuser les attestations antérieures ou incomplètes."""
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
        self.report = {
            "schema": 3, "passed": True, "tolerance_pq12": 2,
            "library_sha256": package.digest(self.library),
            "backend_source_sha256": package.digest(package.ROOT / "src/gpu_backend.c"),
            "dependencies": json.loads((package.ROOT / "dependencies.json").read_text()),
            "samples": [{"device": {"uuid": "a"*32}, "clip_sha256": str(index)*64, "frames": 3,
                         "max_cpu_pq12": 1.0, "max_reference_pq12": 1.1,
                         "producer": {"backend": "vulkan: test", "frame_sha256": ["b"*64]*3},
                         "yuv": [{"chroma": "topleft", "dither": "none", "crop": [2, 4, 6, 8], "max_code10": .5,
                                  "frame_sha256": ["d"*64]*3},
                                 {"chroma": "left", "dither": "blue", "crop": [0, 0, 0, 0], "max_code10": .9999,
                                  "frame_sha256": ["e"*64]*3}]}
                        for index in range(3)],
        }
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
        self.report["schema"] = 1
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.load()

    def test_different_library_is_rejected(self):
        self.library.write_bytes(b"autre bibliotheque")
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.load()

    def test_missing_producer_pixels_are_rejected(self):
        del self.report["samples"][0]["producer"]
        with self.assertRaisesRegex(RuntimeError, "pixels"):
            self.load()

    def test_cpu_fallback_is_rejected(self):
        self.report["samples"][0]["producer"]["backend"] = "cpu"
        with self.assertRaisesRegex(RuntimeError, "pixels"):
            self.load()

    def test_pq_threshold_is_not_relaxed(self):
        self.report["samples"][0]["max_reference_pq12"] = 2.001
        with self.assertRaisesRegex(RuntimeError, "PQ12"):
            self.load()

    def test_final_yuv_outputs_are_required(self):
        self.report["samples"][0]["yuv"].pop()
        with self.assertRaisesRegex(RuntimeError, "Y'CbCr"):
            self.load()

    def test_yuv_rounding_is_not_relaxed(self):
        self.report["samples"][1]["yuv"][0]["max_code10"] = .6
        with self.assertRaisesRegex(RuntimeError, "Y'CbCr"):
            self.load()

    def test_direct_binary_and_pixels_are_required(self):
        binary = self.library.with_name("ffmpeg-fel")
        binary.write_bytes(b"filtre de test")
        self.report["direct"] = {"binary_sha256": package.digest(binary),
            "filter_sources": {p.name: package.digest(p) for p in sorted((package.ROOT/'integrations/ffmpeg').glob('*.c'))}}
        for sample in self.report["samples"]:
            sample["direct"] = {"transport": "vulkan", "frames_read": 17,
                                "frame_sha256": sample["producer"]["frame_sha256"],
                                "yuv": [case["frame_sha256"] for case in sample["yuv"]]}
        self.load()
        package.check_direct(binary, self.validation)
        self.report["samples"][0]["direct"]["yuv"][1] = ["f"*64]*3
        self.load()
        with self.assertRaisesRegex(RuntimeError, "Pixels"):
            package.check_direct(binary, self.validation)
        self.report["samples"][0]["direct"]["yuv"][1] = self.report["samples"][0]["yuv"][1]["frame_sha256"]
        self.report["samples"][0]["direct"]["frame_sha256"] = ["c"*64]*3
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
