"""Reproduit la séquence de transport, sans extrait de film ni validation perceptuelle."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fel_rpu", type=Path, help="assets/tests/fel_orig.bin du dépôt dovi_tool épinglé")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fel-fixture-") as temporary:
        work = Path(temporary)
        rpu = work / "original.bin"
        rpu.write_bytes(args.fel_rpu.read_bytes()*12)
        edit = work / "edit.json"
        # L5 du RPU UHD de référence ne convient pas aux dimensions synthétiques.
        edit.write_text(json.dumps({"mode": 0, "active_area": {
            "presets": [{"id": 0, "top": 0, "bottom": 0, "left": 0, "right": 0}],
            "edits": {"0-11": 0}}}))
        corrected = work / "rpu.bin"
        subprocess.run(["dovi_tool", "editor", "-i", str(rpu), "-j", str(edit), "-o", str(corrected)], check=True)
        for name, size in (("bl", "256x144"), ("el", "128x72")):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                f"color=gray:s={size}:r=24000/1001", "-frames:v", "12", "-pix_fmt", "yuv420p10le",
                "-c:v", "libx265", "-x265-params",
                "log-level=0:pools=1:frame-threads=1:keyint=12:min-keyint=12:scenecut=0:bframes=3:chromaloc=2",
                str(work / f"{name}.hevc")], check=True)
        subprocess.run(["dovi_tool", "inject-rpu", "-i", str(work / "el.hevc"),
            "-r", str(corrected), "-o", str(work / "el-rpu.hevc")], check=True)
        subprocess.run(["dovi_tool", "mux", "--bl", str(work / "bl.hevc"), "--el", str(work / "el-rpu.hevc"),
            "-o", str(args.output.resolve())], check=True)


if __name__ == "__main__":
    main()
