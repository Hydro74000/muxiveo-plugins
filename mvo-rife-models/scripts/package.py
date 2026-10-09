"""Assemble la release mvo-rife-models : fichiers des modèles (noms d'asset du catalogue), licences, manifeste.

Usage : python package.py <dossier source> <sortie>
<dossier source>/<modèle>/<fichier> (ex. rife-v4.15-mvo1/flownet.bin) ; chaque fichier est vérifié contre
catalog.json (taille, SHA-256) puis copié sous son nom d'asset. manifest.json reprend le catalogue avec les
SHA-256 de tous les fichiers publiés (licences comprises) : Muxiveo et mvo-rife-trt les vérifient.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    catalog = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for model, entry in catalog["models"].items():
        for name, meta in entry["files"].items():
            path = src / model / name
            if not path.is_file():
                raise SystemExit(f"{model}/{name} : fichier absent")
            digest = sha256(path)
            if path.stat().st_size != meta["size"] or digest != meta["sha256"]:
                raise SystemExit(f"{model}/{name} : taille ou SHA-256 différent du catalogue")
            shutil.copyfile(path, out / meta["asset"])
            files[meta["asset"]] = digest
    for lic in catalog["licenses"]:
        target = out / Path(lic).name
        shutil.copyfile(HERE / lic, target)
        files[target.name] = sha256(target)
    manifest = {"name": catalog["name"], "version": catalog["version"], "models": catalog["models"],
                "licenses": [Path(lic).name for lic in catalog["licenses"]], "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{catalog['name']} {catalog['version']} : {len(files)} fichier(s) dans {out}")


if __name__ == "__main__":
    main()
