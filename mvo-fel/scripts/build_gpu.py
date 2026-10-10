"""Dépendances GPU privées : libplacebo et compilateur GLSL liés statiquement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from build import ROOT, fetch, prepare_vulkan_headers, run


def build_gpu(build: Path, env: dict[str, str], jobs: int) -> Path:
    deps = json.loads((ROOT / "dependencies.json").read_text())
    cache = build / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    glslang = fetch("glslang", deps["glslang"], cache)
    placebo = fetch("libplacebo", deps["libplacebo_reference"], cache)
    prefix = build / "gpu-private"
    prepare_vulkan_headers(build, placebo, prefix)
    compiler = build / "glslang"
    run(["cmake", "-S", str(glslang), "-B", str(compiler), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_INSTALL_LIBDIR=lib",
         "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
         "-DBUILD_SHARED_LIBS=OFF", "-DBUILD_EXTERNAL=OFF", "-DENABLE_OPT=OFF",
         "-DENABLE_GLSLANG_BINARIES=OFF", "-DENABLE_HLSL=OFF", "-DGLSLANG_TESTS=OFF"], ROOT, env)
    run(["cmake", "--build", str(compiler), "--parallel", str(jobs)], ROOT, env)
    run(["cmake", "--install", str(compiler)], ROOT, env)
    local = dict(env)
    local["CFLAGS"] = f"-I{prefix}/include -fvisibility=hidden"
    local["CXXFLAGS"] = f"-I{prefix}/include -fvisibility=hidden"
    local["LDFLAGS"] = f"-L{prefix}/lib"
    local["LIBRARY_PATH"] = str(prefix / "lib")
    local["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
    meson = build / "placebo-gpu"
    run(["meson", "setup", *( ["--reconfigure"] if (meson / "build.ninja").exists() else []),
         str(meson), str(placebo), f"--prefix={prefix}", "--libdir=lib", "--buildtype=release",
         "--default-library=static", "-Dprefer_static=true", "-Ddemos=false", "-Dtests=false",
         "-Dopengl=disabled", "-Dd3d11=disabled", "-Dvulkan=enabled", "-Dvk-proc-addr=disabled",
         "-Dlibdovi=disabled", "-Dglslang=enabled", "-Dshaderc=disabled", "-Dlcms=disabled",
         "-Dunwind=disabled", "-Dxxhash=disabled"], ROOT, local)
    run(["meson", "compile", "-C", str(meson), "-j", str(jobs)], ROOT, local)
    run(["meson", "install", "-C", str(meson)], ROOT, local)
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    build_gpu(args.build.resolve(), dict(os.environ), max(1, args.jobs))
