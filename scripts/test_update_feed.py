"""Les variantes CPU/GPU FEL partagent le contrat, pas leurs capacités matérielles."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from update_feed import build_entry


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.assets = Path(self.temp.name)

    def archive(self, platform, *, name="mvo-fel", abi=1, gpu=False, suffix=""):
        manifest = {"name": name, "version": "0.1.0", "abi": abi, "platform": platform,
                    "library": "library", "files": {"library": "0"*64},
                    "dependencies": {"ffmpeg": {"revision": "pinned"}},
                    "capabilities": {"cpu": True, "gpu": gpu}}
        if gpu:
            manifest["direct_ffmpeg"] = "ffmpeg-fel"
        with zipfile.ZipFile(self.assets/f"{platform}{suffix}.zip", "w") as archive:
            archive.writestr("plugin/manifest.json", json.dumps(manifest))

    def test_fel_keeps_platform_specific_capabilities(self):
        self.archive("linux-x86_64", gpu=True)
        self.archive("windows-x86_64")
        entry = build_entry("mvo-fel", "mvo-fel-v0.1.0", self.assets)
        self.assertEqual(entry["abi"], 1)
        self.assertNotIn("capabilities", entry)
        by_platform = {p["platform"]: p for p in entry["platforms"]}
        self.assertTrue(by_platform["linux-x86_64"]["capabilities"]["gpu"])
        self.assertEqual(by_platform["linux-x86_64"]["direct_ffmpeg"], "ffmpeg-fel")
        self.assertFalse(by_platform["windows-x86_64"]["capabilities"]["gpu"])

    def test_contract_mismatch_remains_rejected(self):
        self.archive("linux-x86_64", gpu=True)
        self.archive("windows-x86_64", abi=2)
        with self.assertRaisesRegex(ValueError, "manifeste différent"):
            build_entry("mvo-fel", "mvo-fel-v0.1.0", self.assets)

    def test_duplicate_fel_platform_is_rejected(self):
        self.archive("linux-x86_64")
        self.archive("linux-x86_64", gpu=True, suffix="-gpu")
        with self.assertRaisesRegex(ValueError, "même plate-forme"):
            build_entry("mvo-fel", "mvo-fel-v0.1.0", self.assets)

    def test_other_extensions_keep_strict_common_fields(self):
        self.archive("linux-x86_64", name="mvo-rife", gpu=True)
        self.archive("windows-x86_64", name="mvo-rife")
        with self.assertRaisesRegex(ValueError, "manifeste différent"):
            build_entry("mvo-rife", "mvo-rife-v0.1.0", self.assets)


if __name__ == "__main__":
    unittest.main()
