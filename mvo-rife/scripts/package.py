"""Assemble l'extension mvo-rife : moteur muxiveo-rife, modèles, poids du sélecteur, préréglages, licences.

Usage : python package.py <plate-forme> <binaire muxiveo-rife> <sortie> [--moltenvk <libMoltenVK.dylib> <licence>]
Produit <sortie>/mvo-rife-<version>-<plate-forme>/ et son archive (.zip sous Windows, sinon .tar.gz). Le
manifeste (manifest.json) liste chaque fichier avec son SHA-256 : Muxiveo le vérifie avant d'activer
l'extension ; il annonce aussi le contrat avec Muxiveo (--capabilities) et les préréglages (presets.json).
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from fetch_models import fetch  # noqa: E402

VERSION_RE = re.compile(r"project\(muxiveo-rife VERSION ([0-9.]+)")
CONTRACT_RE = re.compile(r"#define MUXIVEO_RIFE_CONTRACT (\d+)")
ABI_RE = re.compile(r"#define MVO_TRT_ABI_VERSION (\d+)u")
FORMAT_RE = re.compile(r"constexpr int SELECTOR_WEIGHTS_FORMAT = (\d+);")
PLATFORMS = ("linux-x86_64", "windows-x86_64", "macos-arm64")
ENGINES = ("rife", "hybrid")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def search(regex: re.Pattern, path: Path) -> str:
    """Première capture de regex dans le fichier ; arrêt explicite si absente."""
    match = regex.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"{regex.pattern} introuvable dans {path}")
    return match.group(1)


def check_presets(presets: dict, models: dict) -> None:
    """Préréglages cohérents : moteur connu, modèle livré, arguments supplémentaires en liste de chaînes."""
    if presets.get("schema") != 1 or not isinstance(presets.get("presets"), dict) or not presets["presets"]:
        raise SystemExit("presets.json : schéma 1 et préréglages attendus")
    for name, preset in presets["presets"].items():
        if preset.get("engine") not in ENGINES:
            raise SystemExit(f"preset {name} : moteur {preset.get('engine')} inconnu")
        if preset.get("model") not in models:
            raise SystemExit(f"preset {name} : modèle {preset.get('model')} absent de models.json")
        args = preset.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise SystemExit(f"preset {name} : args doit être une liste de chaînes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble l'extension mvo-rife.")
    parser.add_argument("platform", choices=PLATFORMS)
    parser.add_argument("binary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--moltenvk", nargs=2, type=Path, metavar=("DYLIB", "LICENSE"))
    args = parser.parse_args()

    version = search(VERSION_RE, HERE / "CMakeLists.txt")
    contract = int(search(CONTRACT_RE, HERE / "src" / "main.cpp"))
    abi = int(search(ABI_RE, HERE / "src" / "trt_plugin_abi.h"))
    selector_format = int(search(FORMAT_RE, HERE / "src" / "selector_weights.h"))
    models = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    presets = json.loads((HERE / "presets.json").read_text(encoding="utf-8"))
    check_presets(presets, models["models"])

    name = f"mvo-rife-{version}-{args.platform}"
    dest = args.out / name
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "LICENSES").mkdir(parents=True)
    executable = args.binary.name
    shutil.copy2(args.binary, dest / executable)
    (dest / executable).chmod(0o755)
    fetch(dest / "rife-models")
    shutil.copy2(HERE / "selector" / "selector.txt", dest / "rife-models" / "selector.txt")
    for doc in ("README.md", "UPSTREAM.md", "models.json", "presets.json"):
        shutil.copy2(HERE / doc, dest / doc)
    if (HERE / "docs").is_dir():
        shutil.copytree(HERE / "docs", dest / "docs")
    for lic in sorted((HERE / "LICENSES").glob("*.txt")):
        shutil.copy2(lic, dest / "LICENSES" / lic.name)
    shutil.copy2(HERE.parent / "LICENSE", dest / "LICENSES" / "MIT-mvo-rife.txt")
    if args.moltenvk:
        dylib, license_file = args.moltenvk
        shutil.copy2(dylib, dest / "libMoltenVK.dylib")
        shutil.copy2(license_file, dest / "LICENSES" / "Apache-2.0-MoltenVK.txt")

    files = {p.relative_to(dest).as_posix(): sha256(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    manifest = {
        "name": "mvo-rife",
        "version": version,
        "contract": contract,
        "platform": args.platform,
        "executable": executable,
        "trt_abi": abi,
        "selector_format": selector_format,
        "models": sorted(models["models"]),
        "presets": sorted(presets["presets"]),
        "files": files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.platform.startswith("windows"):
        archive = args.out / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(dest.rglob("*")):
                z.write(p, Path(name) / p.relative_to(dest))
    else:
        archive = args.out / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            t.add(dest, arcname=name)
    print(f"{archive} ({archive.stat().st_size / 1e6:.0f} Mo, {len(files)} fichiers)")


if __name__ == "__main__":
    main()
