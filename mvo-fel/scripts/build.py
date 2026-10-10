"""Construit mvo-fel et ses dépendances privées épinglées (aucun téléchargement à l'exécution).

Exécuter avec Python >= 3.10, CMake, Ninja, make, pkg-config, NASM et Rust >= 1.88.
Windows : environnement MSYS2 UCRT64, Rust x86_64-pc-windows-gnu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def fetch(name: str, spec: dict, cache: Path) -> Path:
    archive = cache / f"{name}-{spec['revision']}.tar.gz"
    if not archive.exists():
        staging = archive.with_suffix(".part")
        with urllib.request.urlopen(spec["url"], timeout=120) as source, staging.open("wb") as target:
            shutil.copyfileobj(source, target)
        staging.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
        raise RuntimeError(f"Somme SHA-256 incorrecte : {archive}")
    with tarfile.open(archive) as tar:
        top = tar.getnames()[0].split("/")[0]
        destination = cache / top
        if not destination.exists():
            for member in tar.getmembers():
                path = (cache / member.name).resolve()
                if not path.is_relative_to(cache.resolve()) or member.issym() or member.islnk():
                    raise RuntimeError(f"Entrée d'archive refusée : {member.name}")
            tar.extractall(cache)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    parser.add_argument("--gpu", action="store_true", help="Prototype Vulkan avec dépendances statiques privées")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    cache = build / "sources"
    cache.mkdir(exist_ok=True)
    deps = json.loads((ROOT / "dependencies.json").read_text())
    env = dict(os.environ)
    ffmpeg = fetch("ffmpeg", deps["ffmpeg"], cache)
    dovi = fetch("libdovi", deps["libdovi"], cache) / "dolby_vision"
    prefix = build / "private"
    ffbuild = build / "ffmpeg"
    ffbuild.mkdir(exist_ok=True)
    configure = [str(ffmpeg / "configure"), f"--prefix={prefix}", "--disable-everything",
        "--disable-autodetect", "--disable-programs", "--disable-doc", "--disable-debug",
        "--disable-network", "--disable-avdevice", "--disable-avfilter", "--disable-swscale",
        "--disable-swresample", "--disable-shared", "--enable-static", "--enable-pic",
        "--enable-protocol=file", "--enable-demuxer=matroska,mov,mpegts,hevc", "--enable-muxer=nut",
        "--enable-decoder=hevc", "--enable-parser=hevc", "--enable-bsf=dovi_split,hevc_mp4toannexb",
        "--enable-encoder=rawvideo", "--extra-cflags=-fvisibility=hidden"]
    run(["bash", *configure], ffbuild, env)
    run(["make", f"-j{max(1,args.jobs)}"], ffbuild, env)
    run(["make", "install"], ffbuild, env)
    # cargo doit connaître le type de sortie avant de résoudre les dépendances.
    manifest = dovi / "Cargo.toml"
    original = manifest.read_text()
    if 'crate-type = ["rlib", "staticlib"]' not in original:
        manifest.write_text(original.replace("[lib]\n", '[lib]\ncrate-type = ["rlib", "staticlib"]\n'))
    env["CARGO_TARGET_DIR"] = str(build / "rust")
    # Sources de toutes les dépendances transitives, également redistribuées.
    vendor = cache / "cargo-vendor"
    offline = vendor.exists()
    if offline:
        config = dovi / ".cargo"
        config.mkdir(exist_ok=True)
        (config / "config.toml").write_text(
            '[source.crates-io]\nreplace-with = "vendored-sources"\n'
            '[source.vendored-sources]\ndirectory = ' + json.dumps(str(vendor)) + '\n'
        )
    run(["cargo", "build", "--locked", *(["--offline"] if offline else []),
         "--release", "--lib", "--features", "capi"], dovi, env)
    if not vendor.exists():
        run(["cargo", "vendor", "--locked", str(vendor)], dovi, env)
    env["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
    if args.gpu:
        from build_gpu import build_gpu
        gpu_prefix = build_gpu(build, env, max(1, args.jobs))
        env["PKG_CONFIG_PATH"] = os.pathsep.join((str(gpu_prefix / "lib/pkgconfig"), env["PKG_CONFIG_PATH"]))
    target = env.get("CARGO_BUILD_TARGET", "")
    release = build / "rust" / target / "release"
    library = release / "libdolby_vision.a"
    if not library.exists():
        library = release / "dolby_vision.lib"
    run(["cmake", "-S", str(ROOT), "-B", str(build / "plugin"), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", f"-DMVO_FEL_DOVI_LIBRARY={library}",
         f"-DMVO_FEL_VULKAN_PROTOTYPE={'ON' if args.gpu else 'OFF'}",
         f"-DMVO_FEL_REFERENCE_TESTS={'ON' if args.gpu else 'OFF'}"], ROOT, env)
    run(["cmake", "--build", str(build / "plugin"), "--parallel", str(max(1,args.jobs))], ROOT, env)
    run(["ctest", "--test-dir", str(build / "plugin"), "--output-on-failure"], ROOT, env)


if __name__ == "__main__":
    main()
