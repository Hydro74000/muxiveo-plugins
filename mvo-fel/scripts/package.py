"""Archive autonome du moteur, avec manifeste vérifié et sources pour le reliage LGPL."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = {"linux-x86_64": "libmvo_fel.so", "windows-x86_64": "mvo_fel.dll", "macos-arm64": "libmvo_fel.dylib"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_capabilities(library: Path, validation: Path | None = None) -> dict:
    """Valide le moteur ; un prototype exige son attestation numérique exacte."""
    lib = ctypes.CDLL(str(library.resolve()))
    lib.mvo_fel_abi_version.restype = ctypes.c_uint32
    if lib.mvo_fel_abi_version() != 1:
        raise RuntimeError("ABI du moteur incompatible")
    lib.mvo_fel_capabilities.restype = ctypes.c_char_p
    capabilities = json.loads(lib.mvo_fel_capabilities())
    if capabilities.get("experimental"):
        if validation is None:
            raise RuntimeError("Le prototype Vulkan exige une attestation numérique de cette bibliothèque")
        report = json.loads(validation.read_text())
        if (report.get("schema") != 2 or report.get("passed") is not True or report.get("tolerance_pq12") != 2
                or report.get("library_sha256") != digest(library)
                or report.get("backend_source_sha256") != digest(ROOT / "src/gpu_backend.c")
                or report.get("dependencies") != json.loads((ROOT / "dependencies.json").read_text())):
            raise RuntimeError("Attestation GPU incompatible avec cette bibliothèque")
        scenes: dict[str, set[str]] = {}
        for sample in report.get("samples", []):
            for metric in ("max_cpu_pq12", "max_reference_pq12"):
                value = float(sample[metric])
                if not math.isfinite(value) or not 0 <= value <= 2:
                    raise RuntimeError("Seuil PQ12 dépassé dans l'attestation")
            if sample.get("frames", 0) < 3:
                raise RuntimeError("Attestation GPU incomplète")
            producer = sample.get("producer", {})
            hashes = producer.get("frame_sha256", [])
            if (len(hashes) != sample["frames"] or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
                    or not producer.get("backend", "").startswith("vulkan:")):
                raise RuntimeError("Comparaison des pixels de la bibliothèque livrée absente")
            scenes.setdefault(sample["device"]["uuid"], set()).add(sample["clip_sha256"])
        if not scenes or any(len(clips) < 3 for clips in scenes.values()):
            raise RuntimeError("Au moins neuf images de trois scènes par GPU sont nécessaires")
    return capabilities


def check_direct(binary: Path, validation: Path) -> None:
    """Le filtre livré doit avoir produit les mêmes pixels que le renderer attesté."""
    report = json.loads(validation.read_text())
    direct = report.get("direct", {})
    sources = {p.name: digest(p) for p in sorted((ROOT/'integrations/ffmpeg').glob('*.c'))}
    if direct.get("binary_sha256") != digest(binary) or direct.get("filter_sources") != sources:
        raise RuntimeError("Attestation du prototype direct incompatible")
    if not report.get("samples"):
        raise RuntimeError("Images directes absentes")
    for sample in report["samples"]:
        measured = sample.get("direct") or {}
        if (measured.get("transport") != "vulkan" or measured.get("frames_read") != 17
                or measured.get("frame_sha256") != sample["producer"]["frame_sha256"]):
            raise RuntimeError("Pixels du filtre direct absents ou différents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, help="Bibliothèque à empaqueter, sinon build/plugin")
    parser.add_argument("--validation", type=Path, help="Attestation validate_gpu.py (prototype Linux)")
    parser.add_argument("--direct-ffmpeg", type=Path, help="Filtre direct privé, exige son attestation numérique")
    parser.add_argument("platform", choices=LIBRARIES)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    library = args.library or args.build / "plugin" / LIBRARIES[args.platform]
    capabilities = load_capabilities(library, args.validation)
    if capabilities.get("experimental") and args.platform != "linux-x86_64":
        raise RuntimeError("Le prototype GPU empaqueté est actuellement validé sur Linux x86_64 uniquement")
    if args.direct_ffmpeg:
        if not capabilities.get("experimental") or args.platform != "linux-x86_64" or not args.validation:
            raise RuntimeError("Le transport direct exige un paquet GPU Linux attesté")
        check_direct(args.direct_ffmpeg, args.validation)
    version = re.search(r"project\(mvo-fel VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text()).group(1)
    name = f"mvo-fel-{version}-{args.platform}" + ("-prototype" if capabilities.get("experimental") else "")
    destination = args.output / name
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(library, destination / library.name)
    if args.direct_ffmpeg:
        shutil.copy2(args.direct_ffmpeg, destination / "ffmpeg-fel")
    shutil.copytree(ROOT / "LICENSES", destination / "LICENSES")
    shutil.copy2(ROOT / "NOTICES.md", destination)
    shutil.copy2(ROOT / "dependencies.json", destination)
    if capabilities.get("experimental"):
        shutil.copy2(args.validation, destination / "gpu-validation.json")
        # Refuser toute dépendance dynamique non système dans le prototype autonome.
        allowed = {"libc.so.6", "libm.so.6", "libgcc_s.so.1", "libdl.so.2", "libpthread.so.0", "ld-linux-x86-64.so.2"}
        for executable in (library, *([args.direct_ffmpeg] if args.direct_ffmpeg else [])):
            result = subprocess.run(["readelf", "-d", str(executable)], capture_output=True, text=True, check=True)
            needed = re.findall(r"Shared library: \[(.*?)\]", result.stdout)
            if set(needed)-allowed:
                raise RuntimeError(f"Dépendances non embarquées dans {executable.name} : {set(needed)-allowed}")
    # Le test s'exécute sur la copie livrée, sans bibliothèque FFmpeg système.
    subprocess.run([sys.executable, str(ROOT / "tests/smoke.py"), str(destination / library.name)], check=True)
    # Archive de sources complète, y compris dépendances transitives Rust.
    with tarfile.open(destination / "sources.tar.gz", "w:gz") as tar:
        tar.add(ROOT, arcname="mvo-fel", filter=lambda item: None if "/build/" in item.name else item)
        tar.add(args.build / "sources", arcname="dependencies")
        tar.add(args.build / "ffmpeg/config.h", arcname="configuration/ffmpeg-config.h")
        if args.direct_ffmpeg:
            tar.add(args.build / "ffmpeg-direct/config.h", arcname="configuration/ffmpeg-direct-config.h")
            tar.add(args.build / "ffmpeg-direct/ffbuild/config.mak", arcname="configuration/ffmpeg-direct-config.mak")
    manifest = {"name": "mvo-fel", "version": version, "abi": 1, "platform": args.platform,
        "library": library.name, "capabilities": capabilities,
        "dependencies": json.loads((ROOT / "dependencies.json").read_text()),
        "files": {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
    if args.direct_ffmpeg:
        manifest["direct_ffmpeg"] = "ffmpeg-fel"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.platform.startswith("windows"):
        with zipfile.ZipFile(args.output / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(name) / path.relative_to(destination))
    else:
        with tarfile.open(args.output / f"{name}.tar.gz", "w:gz") as archive:
            archive.add(destination, arcname=name)


if __name__ == "__main__":
    main()
