"""Prototype FFmpeg privé avec source FEL Vulkan ; n'altère aucun outil installé."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import sys
from pathlib import Path

from build import ROOT, bash_executable, fetch, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Le prototype direct Vulkan/CUDA est actuellement réservé à Linux")
    build = args.build.resolve()
    env = dict(os.environ)
    deps = json.loads((ROOT / "dependencies.json").read_text())
    source = fetch("ffmpeg", deps["ffmpeg"], build / "sources")
    headers = fetch("ffnvcodec", deps["ffnvcodec"], build / "sources")
    prefix = build / "gpu-private"
    run(["make", "install", f"PREFIX={prefix}"], headers, env)
    direct = build / "ffmpeg-direct-source"
    if not direct.exists():
        shutil.copytree(source, direct)
    shutil.copy2(ROOT / "integrations/ffmpeg/vsrc_mvo_fel.c", direct / "libavfilter/vsrc_mvo_fel.c")
    shutil.copy2(ROOT / "integrations/ffmpeg/vf_mvo_fel_cuda.c", direct / "libavfilter/vf_mvo_fel_cuda.c")
    for name in ("fel_abi.h", "fel_direct_abi.h"):
        shutil.copy2(ROOT / "include" / name, direct / "libavfilter" / name)
    additions = {
        "libavfilter/Makefile": "OBJS-$(CONFIG_MVO_FEL_FILTER) += vsrc_mvo_fel.o\nOBJS-$(CONFIG_MVO_FEL_CUDA_FILTER) += vf_mvo_fel_cuda.o\n",
        "libavfilter/allfilters.c": "extern const FFFilter ff_vsrc_mvo_fel;\nextern const FFFilter ff_vf_mvo_fel_cuda;\n",
    }
    for name, line in additions.items():
        path = direct / name
        text = path.read_text()
        if name.endswith("allfilters.c"):
            for declaration in line.splitlines(keepends=True):
                text = text.replace(declaration, "")
            marker = '#include "libavfilter/filter_list.c"'
            if marker not in text:
                raise RuntimeError("Point de déclaration des filtres FFmpeg introuvable")
            path.write_text(text.replace(marker,line+marker))
        else:
            for declaration in line.splitlines(keepends=True):
                if declaration not in text:
                    text += "\n"+declaration
            path.write_text(text)
    directory = build / "ffmpeg-direct"
    directory.mkdir(exist_ok=True)
    env["PKG_CONFIG_PATH"] = str(build / "gpu-private/lib/pkgconfig")
    archives = shlex.join(str(p) for p in sorted((build / "gpu-private/lib").glob("*.a")))
    run([bash_executable(), str(direct / "configure"), "--disable-everything", "--disable-autodetect",
         "--disable-doc", "--disable-debug", "--disable-network", "--disable-programs", "--enable-ffmpeg",
         "--disable-shared", "--enable-static", "--enable-pic", "--enable-vulkan", "--enable-libplacebo",
         "--enable-ffnvcodec", "--enable-cuda", "--enable-nvenc", "--enable-protocol=file,pipe",
         "--enable-demuxer=matroska,mov,hevc,nut", "--enable-muxer=matroska,hevc,nut,null,rawvideo",
         "--enable-decoder=hevc,rawvideo", "--enable-encoder=hevc_nvenc,rawvideo", "--enable-parser=hevc",
         "--enable-filter=mvo_fel,mvo_fel_cuda,libplacebo,hwdownload,hwmap,format,setparams,crop,scale,setsar",
         "--pkg-config-flags=--static", f"--extra-cflags=-I{build / 'gpu-private/include'}",
         "--extra-ldflags=-static-libgcc -static-libstdc++",
         f"--extra-libs=-Wl,--start-group {archives} -Wl,--end-group -Wl,-Bstatic -lstdc++ -Wl,-Bdynamic -lm -ldl -lpthread"], directory, env)
    run(["make", f"-j{max(1,args.jobs)}"], directory, env)
    run([str(directory / "ffmpeg"), "-hide_banner", "-filters"], ROOT, env)
    (directory / "fel-direct-build.json").write_text(json.dumps({
        "ffmpeg_revision": deps["ffmpeg"]["revision"], "ffnvcodec_revision": deps["ffnvcodec"]["revision"],
        "platform": sys.platform,
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
