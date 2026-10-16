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


# Sources du rendu GPU et de la référence certifiée liées à l'attestation (schéma 5).
ENGINE_SOURCES = ("src/gpu_backend.c", "src/certified.cpp", "src/fixed_point.cpp", "src/reconstruction.cpp",
                  "src/shaders/*.glsl", "data/pq_thresholds.bin")


def engine_sources() -> dict[str, str]:
    """Empreintes des sources du moteur, par chemin relatif."""
    return {path.relative_to(ROOT).as_posix(): digest(path)
            for pattern in ENGINE_SOURCES for path in sorted(ROOT.glob(pattern))}


def check_mode(result: dict, mode: str) -> None:
    """Une précision attestée : écarts, pixels du producteur, sorties Y'CbCr."""
    for metric in ("max_cpu_pq12", "max_reference_pq12"):
        value = float(result[metric])
        if not math.isfinite(value) or not 0 <= value <= 2:
            raise RuntimeError("Seuil PQ12 dépassé dans l'attestation")
    codes = result.get("max_fast_codes16")
    if (not isinstance(codes, int) or codes < 0 or mode == "standard" and codes > 1
            or mode == "exact" and (codes or result.get("cpu_identical") is not True)):
        raise RuntimeError(f"Précision {mode} hors tolérance dans l'attestation")
    if mode == "exact":
        # Référence CPU et rendu GPU certifiés (aucune composante indécise), variante retenue,
        # mêmes pixels avec une seule image en vol.
        reference = result.get("cpu_reference") or {}
        counters = (result.get("producer", {}).get("certified_rescued"),
                    result.get("producer", {}).get("certified_undecided"),
                    reference.get("certified_rescued"), reference.get("certified_undecided"))
        if (not all(isinstance(value, int) and value >= 0 for value in counters) or counters[1] or counters[3]
                or result.get("variant") not in {"fixed", "double"}
                or (result.get("single_worker") or {}).get("identical") is not True):
            raise RuntimeError("Rendu exact non certifié dans l'attestation")
    producer = result.get("producer", {})
    hashes = producer.get("frame_sha256", [])
    if (len(hashes) != 3 or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
            or not producer.get("backend", "").startswith("vulkan:")):
        raise RuntimeError("Comparaison des pixels de la bibliothèque livrée absente")
    # Sorties finales Y'CbCr : arrondi exact recadré et tramage livré, face à la référence en double.
    yuv = result.get("yuv") or []
    if (not any(case.get("dither") == "none" and any(case.get("crop") or []) for case in yuv)
            or not any(case.get("dither") == "blue" for case in yuv)):
        raise RuntimeError("Sorties Y'CbCr finales absentes de l'attestation")
    for case in yuv:
        value = float(case["max_code10"])
        if (not math.isfinite(value) or not 0 <= value <= (.52 if case["dither"] == "none" else 1.02)
                or len(case.get("frame_sha256") or []) != 3):
            raise RuntimeError("Sortie Y'CbCr hors tolérance dans l'attestation")


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
        report = json.loads(validation.read_text(encoding="utf-8"))
        if (report.get("schema") != 5 or report.get("passed") is not True or report.get("tolerance_pq12") != 2
                or report.get("tolerance_codes16") != {"standard": 1, "exact": 0}
                or report.get("library_sha256") != digest(library)
                or report.get("engine_sources") != engine_sources()
                or report.get("dependencies") != json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8"))):
            raise RuntimeError("Attestation GPU incompatible avec cette bibliothèque")
        scenes: dict[str, set[str]] = {}
        for sample in report.get("samples", []):
            if sample.get("frames", 0) < 3:
                raise RuntimeError("Attestation GPU incomplète")
            # Rapide sur tout GPU ; standard et exacte sur un GPU calculant en double.
            modes = sample.get("modes") or {}
            required = ("fast", "standard", "exact") if sample["device"].get("float64") else ("fast",)
            if any(mode not in modes for mode in required):
                raise RuntimeError("Précisions absentes de l'attestation")
            for mode in required:
                check_mode(modes[mode], mode)
            scenes.setdefault(sample["device"]["uuid"], set()).add(sample["clip_sha256"])
        if not scenes or any(len(clips) < 3 for clips in scenes.values()):
            raise RuntimeError("Au moins neuf images de trois scènes par GPU sont nécessaires")
    return capabilities


def check_direct(binary: Path, validation: Path) -> None:
    """Le filtre livré doit avoir produit les mêmes pixels que le renderer attesté, dans chaque précision."""
    report = json.loads(validation.read_text(encoding="utf-8"))
    direct = report.get("direct", {})
    sources = {p.name: digest(p) for p in sorted((ROOT/'integrations/ffmpeg').glob('*.c'))}
    if direct.get("binary_sha256") != digest(binary) or direct.get("filter_sources") != sources:
        raise RuntimeError("Attestation du prototype direct incompatible")
    if not report.get("samples"):
        raise RuntimeError("Images directes absentes")
    for sample in report["samples"]:
        for result in (sample.get("modes") or {}).values():
            measured = result.get("direct") or {}
            if (measured.get("transport") != "vulkan" or measured.get("frames_read") != 17
                    or measured.get("frame_sha256") != result["producer"]["frame_sha256"]
                    or measured.get("yuv") != [case["frame_sha256"] for case in result.get("yuv") or []]):
                raise RuntimeError("Pixels du filtre direct absents ou différents")


def check_glibc(binary: Path, maximum: str) -> None:
    """Refuse une archive Linux exigeant une libc plus récente que sa cible."""
    if not re.fullmatch(r"\d+\.\d+", maximum):
        raise ValueError("Version glibc cible invalide")
    result = subprocess.run(["readelf", "--version-info", str(binary)],
                            capture_output=True, text=True, check=True)
    versions = re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b", result.stdout)
    if (not versions or "GLIBC_ABI_" in result.stdout or "GLIBC_PRIVATE" in result.stdout
            or max(tuple(map(int, version.split('.'))) for version in versions) > tuple(map(int, maximum.split('.')))):
        raise RuntimeError(f"{binary.name} exige une ABI glibc au-delà de {maximum}")


# Imports admis d'une DLL Windows autonome : système et UCRT (api-ms-win-*).
WINDOWS_SYSTEM_DLLS = {"kernel32.dll", "msvcrt.dll", "ucrtbase.dll", "bcrypt.dll", "bcryptprimitives.dll",
                       "ntdll.dll", "shlwapi.dll", "userenv.dll", "ws2_32.dll", "advapi32.dll", "ole32.dll",
                       "user32.dll", "gdi32.dll", "shell32.dll"}


def check_windows_imports(binary: Path) -> None:
    """Refuse une DLL Windows dépendant d'une bibliothèque non système (MinGW, Vulkan...)."""
    result = subprocess.run(["objdump", "-p", str(binary)], capture_output=True, text=True, check=True)
    names = {name.lower() for name in re.findall(r"DLL Name: (\S+)", result.stdout)}
    foreign = {n for n in names if n not in WINDOWS_SYSTEM_DLLS and not n.startswith("api-ms-win-")}
    if foreign:
        raise RuntimeError(f"Dépendances non embarquées dans {binary.name} : {sorted(foreign)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, help="Bibliothèque à empaqueter, sinon build/plugin")
    parser.add_argument("--validation", type=Path, help="Attestation validate_gpu.py (prototype Linux)")
    parser.add_argument("--direct-ffmpeg", type=Path, help="Filtre direct privé, exige son attestation numérique")
    parser.add_argument("--max-glibc", help="Version maximale requise pour le paquet Linux publié")
    parser.add_argument("platform", choices=LIBRARIES)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    library = args.library or args.build / "plugin" / LIBRARIES[args.platform]
    capabilities = load_capabilities(library, args.validation)
    if args.max_glibc:
        if args.platform != "linux-x86_64":
            raise RuntimeError("La vérification glibc concerne uniquement Linux")
        for binary in (library, *([args.direct_ffmpeg] if args.direct_ffmpeg else [])):
            check_glibc(binary, args.max_glibc)
    # macOS : aucune attestation sur contenu réel possible (runners de CI uniquement).
    if capabilities.get("experimental") and args.platform not in {"linux-x86_64", "windows-x86_64"}:
        raise RuntimeError("Le GPU empaqueté exige une attestation sur contenu réel (Linux, Windows)")
    # DLL autonome (CPU comme GPU, liées en -static) : imports PE système uniquement.
    if args.platform == "windows-x86_64":
        check_windows_imports(library)
    if args.direct_ffmpeg:
        if not capabilities.get("experimental") or args.platform != "linux-x86_64" or not args.validation:
            raise RuntimeError("Le transport direct exige un paquet GPU Linux attesté")
        check_direct(args.direct_ffmpeg, args.validation)
    version = re.search(r"project\(mvo-fel VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
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
    if capabilities.get("experimental") and args.platform == "linux-x86_64":
        # Refuser toute dépendance dynamique non système dans le prototype autonome (ELF ; PE contrôlé plus haut).
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
        "dependencies": json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8")),
        "files": {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
    if args.direct_ffmpeg:
        manifest["direct_ffmpeg"] = "ffmpeg-fel"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
