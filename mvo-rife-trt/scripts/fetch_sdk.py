"""Télécharge le SDK TensorRT for RTX épinglé (sdk.json), vérifie son SHA-256 et l'extrait.

Usage : python fetch_sdk.py <linux-x86_64|windows-x86_64> <dossier de destination>
Affiche le dossier du SDK extrait (à passer à CMake : -DTRTRTX_ROOT=...).
"""
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    platform, dest = sys.argv[1], Path(sys.argv[2])
    sdk = json.loads((HERE / "sdk.json").read_text(encoding="utf-8"))
    info = sdk[platform]
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / Path(info["url"]).name
    if not archive.is_file():
        tmp = archive.with_suffix(archive.suffix + ".part")
        # URL NVIDIA épinglée (redirection vers un lien signé) ; somme vérifiée ci-dessous.
        with urllib.request.urlopen(info["url"], timeout=600) as resp, tmp.open("wb") as out:  # nosec B310
            shutil.copyfileobj(resp, out)
        tmp.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != info["sha256"]:
        archive.unlink()
        raise SystemExit(f"SDK : SHA-256 inattendu ({digest})")
    root = dest / f"TensorRT-RTX-{sdk['tensorrt_rtx']}"
    if not (root / "include" / "NvInfer.h").is_file():
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        elif subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(dest)], check=False).returncode != 0:
            # tar sans zstd (manylinux_2_28) : décompression par zstd puis extraction
            with subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE) as z:
                subprocess.run(["tar", "-xf", "-", "-C", str(dest)], stdin=z.stdout, check=True)
            if z.returncode != 0:
                raise SystemExit("SDK : décompression zstd impossible")
    print(root)


if __name__ == "__main__":
    main()
