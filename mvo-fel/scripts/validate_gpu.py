"""Attestation locale : neuf images par GPU, tolérance stricte PQ12, aucun média redistribué."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
from functools import partial
from pathlib import Path

from validate_producer import producer_frames, validate_producer
from validate_direct import validate_direct
from validate_yuv import direct_options, p010_to_planar, validate_yuv

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Précisions attestées : rapide sur tout GPU ; standard et exacte sur GPU calculant en double.
MODES = ("fast", "standard", "exact")


def oracle_run(args, device: dict, index: int, clip: Path, mode: str) -> tuple[dict, list[tuple[str, str]], str]:
    """Renderer séparé : écarts au CPU (référence et chemin rapide exact), à kernel64, empreintes."""
    name = f"{args.report.stem}-{device['uuid']}-{index}-{mode}.log"
    log = args.report.with_name(name)
    env = dict(os.environ, MVO_FEL_TEST_DEVICE=device["uuid"], MVO_FEL_TEST_PRECISION=mode)
    with log.open("w") as output:
        subprocess.run([str(args.oracle.resolve()), str(clip.resolve()), "0", "3", "8", "validate"],
                       env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=600)
    text = log.read_text(encoding="utf-8", errors="replace")
    cpu = [float(v) for v in re.findall(r"max_delta_cpu_pq12=([0-9.e+-]+)", text)]
    oracle = [float(v) for v in re.findall(r"max_delta_kernel64_pq12=([0-9.e+-]+)", text)]
    fast = [int(v) for v in re.findall(r"max_delta_fast_codes16=(\d+)", text)]
    pixels = re.findall(r"gpu_frame_bytes=(\d+) gpu_frame_sha256=([0-9a-f]{64})", text)
    if (len(cpu) != 3 or len(oracle) != 3 or len(fast) != 3 or max(*cpu, *oracle) > 2
            or f"gpu_precision={mode}" not in text):
        raise RuntimeError(f"Validation numérique incomplète : {log}")
    # Exacte : identique au CPU ; standard : au plus 1 code 16 bits.
    if mode == "exact" and (max(fast) or max(cpu)) or mode == "standard" and max(fast) > 1:
        raise RuntimeError(f"Précision {mode} hors tolérance : {log}")
    if len(pixels) != 3 or len({size for size, _ in pixels}) != 1:
        raise RuntimeError(f"Empreintes des images de référence absentes : {log}")
    return {"max_cpu_pq12": max(cpu), "max_reference_pq12": max(oracle), "max_fast_codes16": max(fast),
            "log": name}, pixels, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--direct-ffmpeg", type=Path, help="Valider aussi le filtre Vulkan privé")
    args = parser.parse_args()
    if len(args.clips) < 3:
        parser.error("Au moins trois scènes FEL distinctes sont nécessaires")
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.mvo_fel_devices_features.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mvo_fel_devices_features.restype = ctypes.c_size_t
    size = lib.mvo_fel_devices_features(None, 0)
    if not 0 < size <= 65536:
        raise RuntimeError("Énumération GPU impossible")
    buffer = ctypes.create_string_buffer(size)
    if lib.mvo_fel_devices_features(buffer, size) != size:
        raise RuntimeError("Liste GPU modifiée")
    devices = json.loads(buffer.value)
    if not devices:
        raise RuntimeError("Aucun GPU à valider")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": 4, "library_sha256": digest(args.library), "oracle_sha256": digest(args.oracle),
              "backend_source_sha256": digest(ROOT / "src/gpu_backend.c"),
              "dependencies": json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8")),
              "tolerance_pq12": 2, "tolerance_codes16": {"standard": 1, "exact": 0}, "samples": [], "passed": False}
    if args.direct_ffmpeg:
        report["direct"] = {"binary_sha256": digest(args.direct_ffmpeg),
                            "filter_sources": {p.name: digest(p) for p in sorted((ROOT/'integrations/ffmpeg').glob('*.c'))}}
    cpu_frames: dict[Path, tuple[list[bytes], list[dict]]] = {}
    for device in devices:
        for index, clip in enumerate(args.clips):
            sample = {"device": device, "clip_sha256": digest(clip), "frames": 3, "modes": {}}
            for mode in MODES if device.get("float64") else MODES[:1]:
                print(f"Validation {device['name']} : scène {index+1}, précision {mode}", flush=True)
                result, pixels, _ = oracle_run(args, device, index, clip, mode)
                hashes = [sha for _, sha in pixels]
                result["producer"], rgb = validate_producer(lib, clip, device["uuid"], int(pixels[0][0]), hashes, mode)
                # Sorties finales Y'CbCr : conversion de référence en double à partir du RGB attesté.
                result["yuv"] = validate_yuv(lib, clip, device["uuid"], rgb, mode)
                if mode == "exact":
                    # Identique au CPU (double) octet pour octet, RGB et Y'CbCr.
                    if clip not in cpu_frames:
                        cpu_rgb = producer_frames(lib, clip, "cpu", int(pixels[0][0]), "gbrp16le")[1]
                        cpu_frames[clip] = (cpu_rgb, validate_yuv(lib, clip, "cpu", cpu_rgb, mode))
                    cpu_rgb, cpu_yuv = cpu_frames[clip]
                    if (rgb != cpu_rgb or [case["frame_sha256"] for case in result["yuv"]]
                            != [case["frame_sha256"] for case in cpu_yuv]):
                        raise RuntimeError(f"Précision exacte différente du CPU : {device['name']}, scène {index+1}")
                    result["cpu_identical"] = True
                if args.direct_ffmpeg:
                    option = f":precision={mode}"
                    direct = validate_direct(args.direct_ffmpeg, args.library, clip, device["uuid"], int(pixels[0][0]),
                                             hashes, option)
                    # P010 du filtre : mêmes octets que la sortie Y'CbCr 10 bits du producteur.
                    direct["yuv"] = [validate_direct(args.direct_ffmpeg, args.library, clip, device["uuid"],
                        case["width"]*case["height"]*3, case["frame_sha256"], direct_options(case) + option, "p010le",
                        partial(p010_to_planar, width=case["width"], height=case["height"]))["frame_sha256"]
                        for case in result["yuv"]]
                    result["direct"] = direct
                sample["modes"][mode] = result
            report["samples"].append(sample)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if (digest(args.library) != report["library_sha256"] or digest(args.oracle) != report["oracle_sha256"]
            or digest(ROOT / "src/gpu_backend.c") != report["backend_source_sha256"]
            or json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8")) != report["dependencies"]):
        raise RuntimeError("Le moteur ou ses sources ont changé pendant la validation")
    if args.direct_ffmpeg and (digest(args.direct_ffmpeg) != report["direct"]["binary_sha256"]
            or report["direct"]["filter_sources"] != {p.name: digest(p) for p in sorted((ROOT/'integrations/ffmpeg').glob('*.c'))}):
        raise RuntimeError("Le prototype direct a changé pendant la validation")
    report["passed"] = True
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Validation stricte terminée : {args.report}", flush=True)


if __name__ == "__main__":
    main()
