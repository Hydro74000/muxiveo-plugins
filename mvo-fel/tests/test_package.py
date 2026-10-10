"""Gardes du paquet GPU : refuser les attestations antérieures ou incomplètes."""
import importlib.util
import json
import tempfile
import unittest
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
            "schema": 2, "passed": True, "tolerance_pq12": 2,
            "library_sha256": package.digest(self.library),
            "backend_source_sha256": package.digest(package.ROOT / "src/gpu_backend.c"),
            "dependencies": json.loads((package.ROOT / "dependencies.json").read_text()),
            "samples": [{"device": {"uuid": "a"*32}, "clip_sha256": str(index)*64, "frames": 3,
                         "max_cpu_pq12": 1.0, "max_reference_pq12": 1.1,
                         "producer": {"backend": "vulkan: test", "frame_sha256": ["b"*64]*3}}
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

    def test_direct_binary_and_pixels_are_required(self):
        binary = self.library.with_name("ffmpeg-fel")
        binary.write_bytes(b"filtre de test")
        self.report["direct"] = {"binary_sha256": package.digest(binary),
            "filter_sources": {p.name: package.digest(p) for p in sorted((package.ROOT/'integrations/ffmpeg').glob('*.c'))}}
        for sample in self.report["samples"]:
            sample["direct"] = {"transport": "vulkan", "frames_read": 17,
                                "frame_sha256": sample["producer"]["frame_sha256"]}
        self.load()
        package.check_direct(binary, self.validation)
        self.report["samples"][0]["direct"]["frame_sha256"] = ["c"*64]*3
        self.load()
        with self.assertRaisesRegex(RuntimeError, "Pixels"):
            package.check_direct(binary, self.validation)
        binary.write_bytes(b"autre filtre")
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            package.check_direct(binary, self.validation)


if __name__ == "__main__":
    unittest.main()
