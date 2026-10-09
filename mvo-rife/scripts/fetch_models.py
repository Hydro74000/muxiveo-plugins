#!/usr/bin/env python3
"""Télécharge les modèles RIFE épinglés dans models.json et vérifie leur sha256.

Usage :
    python3 mvo-rife/scripts/fetch_models.py <dossier-destination> [--model NOM ...]

Les modèles sont écrits sous ``<destination>/<nom>/flownet.{param,bin}``, disposition
attendue par ``muxiveo-rife`` dans ``<dossier de l'exécutable>/rife-models``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "models.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_url(source: str, commit: str, model: str, filename: str) -> str:
    repo = source.removeprefix("https://github.com/").rstrip("/")
    return f"https://raw.githubusercontent.com/{repo}/{commit}/models/{model}/{filename}"


def fetch(dest: Path, models: list[str] | None = None) -> int:
    """Télécharge (ou revalide) les modèles demandés ; retourne le nombre de fichiers écrits."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = models or list(manifest["models"])
    written = 0
    for name in wanted:
        entry = manifest["models"].get(name)
        if entry is None:
            raise SystemExit(f"modèle inconnu : {name}")
        model_dir = dest / name
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename, meta in entry["files"].items():
            target = model_dir / filename
            if target.is_file() and _sha256(target) == meta["sha256"]:
                continue
            # modèle affiné par Muxiveo : URL de release propre au fichier ; sinon dépôt amont épinglé
            url = meta.get("url") or _raw_url(manifest["source"], manifest["commit"], name, filename)
            print(f"téléchargement {name}/{filename}", file=sys.stderr)
            with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                try:
                    # URL construite avec le préfixe HTTPS raw.githubusercontent.com constant.
                    with urllib.request.urlopen(url, timeout=120) as resp:  # nosec B310
                        while chunk := resp.read(1 << 20):
                            tmp.write(chunk)
                except BaseException:
                    # téléchargement interrompu : pas de fichier partiel laissé
                    tmp.close()
                    tmp_path.unlink(missing_ok=True)
                    raise
            actual = _sha256(tmp_path)
            if actual != meta["sha256"]:
                tmp_path.unlink(missing_ok=True)
                raise SystemExit(f"sha256 invalide pour {name}/{filename} : {actual}")
            os.chmod(tmp_path, 0o644)
            tmp_path.replace(target)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharge les modèles RIFE épinglés (models.json).")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    count = fetch(args.dest, args.models)
    print(f"{count} fichier(s) écrit(s) dans {args.dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
