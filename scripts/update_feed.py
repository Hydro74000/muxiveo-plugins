"""Met à jour le flux d'une extension (`<extension>.json` de la release `extensions-feed`), lu par Muxiveo.

Usage : python scripts/update_feed.py --name <extension> --tag <extension>-vX.Y.Z --assets <dossier des archives>
                                      --out <fichier .json> [--feed <flux actuel>]

Appelé par la CI d'une extension une fois sa release publiée. Chaque entrée reprend le manifeste des archives
(version, contrat ou interface, modèles…) : Muxiveo choisit la version la plus récente compatible avec lui, sans
release de l'application. Entrées triées de la plus récente à la plus ancienne ; une version republiée remplace
son entrée.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

FEED_SCHEMA = 1
MAX_RELEASES = 100
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def read_manifest(archive: Path) -> dict:
    """manifest.json du dossier racine d'une archive d'extension."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("/manifest.json")]
            if len(names) != 1:
                raise ValueError(f"{archive.name} : manifeste introuvable")
            return json.loads(z.read(names[0]).decode("utf-8"))
    with tarfile.open(archive, "r:gz") as t:
        members = [m for m in t.getmembers() if m.isfile() and m.name.count("/") == 1 and m.name.endswith("/manifest.json")]
        if len(members) != 1:
            raise ValueError(f"{archive.name} : manifeste introuvable")
        handle = t.extractfile(members[0])
        if handle is None:
            raise ValueError(f"{archive.name} : manifeste illisible")
        return json.loads(handle.read().decode("utf-8"))


def build_entry(name: str, tag: str, assets: Path, published: str | None = None) -> dict:
    """Entrée du flux : champs communs des manifestes, plates-formes et archives publiées."""
    archives = sorted(p for p in assets.iterdir() if p.is_file() and p.name.endswith((".tar.gz", ".zip")))
    if not archives:
        raise ValueError(f"aucune archive dans {assets}")
    common: dict | None = None
    platforms: list[dict] = []
    for archive in archives:
        manifest = read_manifest(archive)
        if manifest.get("name") != name:
            raise ValueError(f"{archive.name} : extension {manifest.get('name')} au lieu de {name}")
        per_platform = ("capabilities", "direct_ffmpeg") if name == "mvo-fel" else ()
        fields = {k: v for k, v in manifest.items()
                  if k not in ("files", "platform", "executable", "library", *per_platform)}
        if common is None:
            common = fields
        elif fields != common:
            raise ValueError(f"{archive.name} : manifeste différent des autres plates-formes")
        if name == "mvo-fel" and any(p["platform"] == manifest["platform"] for p in platforms):
            raise ValueError(f"{archive.name} : plusieurs archives pour la même plate-forme")
        platforms.append({"platform": manifest["platform"], "asset": archive.name, "size": archive.stat().st_size,
                          **{k: manifest[k] for k in per_platform if k in manifest}})
    assert common is not None
    if tag != f"{name}-v{common['version']}":
        raise ValueError(f"tag {tag} différent de la version {common['version']}")
    return {
        **common,
        "tag": tag,
        "published": published or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": sorted(platforms, key=lambda p: p["platform"]),
    }


def update_feed(feed: dict | None, name: str, entry: dict) -> dict:
    """Flux avec l'entrée ajoutée (ou remplacée), trié par version décroissante et borné."""
    releases = [] if feed is None else [r for r in feed.get("releases", []) if r.get("version") != entry["version"]]
    if feed is not None and (feed.get("schema") != FEED_SCHEMA or feed.get("name") != name):
        raise ValueError("flux existant d'un autre schéma ou d'une autre extension")
    releases.append(entry)
    releases.sort(key=lambda r: _version_key(r["version"]), reverse=True)
    return {"schema": FEED_SCHEMA, "name": name, "updated": entry["published"], "releases": releases[:MAX_RELEASES]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Met à jour le flux d'une extension.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--feed", type=Path, help="flux actuel (absent : nouveau flux)")
    args = parser.parse_args(argv)
    try:
        if not _NAME_RE.match(args.name):
            raise ValueError(f"nom d'extension invalide : {args.name}")
        current = None
        if args.feed and args.feed.is_file():
            current = json.loads(args.feed.read_text(encoding="utf-8"))
        feed = update_feed(current, args.name, build_entry(args.name, args.tag, args.assets))
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(feed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"flux {args.name} : {feed['releases'][0]['version']} ({len(feed['releases'])} version(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
