"""Construit la référence libplacebo épinglée et compare les pixels (Vulkan de test)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from build import ROOT, fetch, prepare_vulkan_headers, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    env = dict(os.environ)
    spec = json.loads((ROOT / "dependencies.json").read_text())["libplacebo_reference"]
    source = fetch("libplacebo", spec, build / "sources")
    prefix = build / "reference-install"
    prepare_vulkan_headers(build, source, prefix)
    reference = build / "reference"
    if not (reference / "build.ninja").exists():
        run(["meson", "setup", str(reference), str(source), f"--prefix={prefix}", "--libdir=lib",
             "-Ddemos=false", "-Dtests=false", "-Dopengl=disabled", "-Dvulkan=enabled",
             "-Dlibdovi=disabled", "-Dglslang=disabled", "-Dshaderc=enabled"], ROOT, env)
    run(["meson", "install", "-C", str(reference)], ROOT, env)
    env["PKG_CONFIG_PATH"] = os.pathsep.join((str(prefix / "lib/pkgconfig"), str(build / "private/lib/pkgconfig")))
    env["CFLAGS"] = f"-I{prefix / 'include'} " + env.get("CFLAGS", "")
    run(["cmake", "-S", str(ROOT), "-B", str(build / "oracle"), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DMVO_FEL_MATH_ONLY=ON", "-DMVO_FEL_REFERENCE_TESTS=ON"], ROOT, env)
    run(["cmake", "--build", str(build / "oracle")], ROOT, env)
    run(["ctest", "--test-dir", str(build / "oracle"), "--output-on-failure"], ROOT, env)


if __name__ == "__main__":
    main()
