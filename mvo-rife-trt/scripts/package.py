"""Assemble le plugin mvo-rife-trt : bibliothèque, TensorRT for RTX, modèles ONNX, licences, manifeste.

Usage : python package.py <plate-forme> <bibliothèque du plugin> <SDK TensorRT for RTX> <dossier des modèles> <sortie>
Produit <sortie>/mvo-rife-trt-<version>-<plate-forme>/ et son archive (.tar.gz ou .zip). Le manifeste
(manifest.json) liste chaque fichier avec son SHA-256 : Muxiveo le vérifie avant d'activer le plugin.
"""
import hashlib
import json
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ABI_RE = re.compile(r"#define MVO_TRT_ABI_VERSION (\d+)u")
VERSION_RE = re.compile(r"project\(mvo-rife-trt VERSION ([0-9.]+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    platform, plugin_lib, sdk_root, models_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    version = VERSION_RE.search((HERE / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
    abi = int(ABI_RE.search((HERE.parent / "mvo-rife" / "src" / "trt_plugin_abi.h").read_text(encoding="utf-8")).group(1))
    sdk = json.loads((HERE / "sdk.json").read_text(encoding="utf-8"))
    name = f"mvo-rife-trt-{version}-{platform}"
    dest = out / name
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "models").mkdir(parents=True)
    (dest / "LICENSES").mkdir()

    shutil.copy2(plugin_lib, dest / plugin_lib.name)
    if platform.startswith("windows"):
        runtime = [p for p in (sdk_root / "bin").glob("*.dll")]
    else:
        runtime = [p for p in (sdk_root / "lib").glob("libtensorrt*_rtx.so.*") if p.name.count(".") == 2]
    if len(runtime) != 2:
        raise SystemExit(f"bibliothèques TensorRT for RTX inattendues : {[p.name for p in runtime]}")
    for lib in runtime:
        shutil.copy2(lib.resolve(), dest / lib.name)
    models = sorted(models_dir.glob("*.onnx"))
    if not models:
        raise SystemExit("aucun modèle ONNX")
    for model in models:
        shutil.copy2(model, dest / "models" / model.name)
    for lic in sorted((HERE / "LICENSES").glob("*.txt")):
        shutil.copy2(lic, dest / "LICENSES" / lic.name)
    shutil.copy2(HERE.parent / "LICENSE", dest / "LICENSES" / "MIT-mvo-rife-trt.txt")
    shutil.copy2(sdk_root / "doc" / "Acknowledgements.txt", dest / "LICENSES" / "NVIDIA-TensorRT-RTX-Acknowledgements.txt")

    files = {p.relative_to(dest).as_posix(): sha256(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    manifest = {
        "name": "mvo-rife-trt",
        "version": version,
        "abi": abi,
        "min_muxiveo_rife": sdk["abi"]["min_muxiveo_rife"],
        "platform": platform,
        "tensorrt_rtx": sdk["tensorrt_rtx"],
        "cuda": sdk["cuda"],
        "library": plugin_lib.name,
        "models": [m.stem for m in models],
        "files": files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if platform.startswith("windows"):
        archive = out / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(dest.rglob("*")):
                z.write(p, Path(name) / p.relative_to(dest))
    else:
        archive = out / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            t.add(dest, arcname=name)
    print(f"{archive} ({archive.stat().st_size / 1e6:.0f} Mo, {len(files)} fichiers)")


if __name__ == "__main__":
    main()
