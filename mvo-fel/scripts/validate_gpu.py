"""Attestation locale : neuf images par GPU, tolérance stricte PQ12, aucun média redistribué."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from validate_producer import validate_producer
from validate_direct import validate_direct

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    lib.mvo_fel_devices.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mvo_fel_devices.restype = ctypes.c_size_t
    size = lib.mvo_fel_devices(None, 0)
    if not 0 < size <= 65536:
        raise RuntimeError("Énumération GPU impossible")
    buffer = ctypes.create_string_buffer(size)
    if lib.mvo_fel_devices(buffer, size) != size:
        raise RuntimeError("Liste GPU modifiée")
    devices = json.loads(buffer.value)
    if not devices:
        raise RuntimeError("Aucun GPU à valider")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": 2, "library_sha256": digest(args.library), "oracle_sha256": digest(args.oracle),
              "backend_source_sha256": digest(ROOT / "src/gpu_backend.c"),
              "dependencies": json.loads((ROOT / "dependencies.json").read_text()),
              "tolerance_pq12": 2, "samples": [], "passed": False}
    if args.direct_ffmpeg:
        report["direct"] = {"binary_sha256": digest(args.direct_ffmpeg),
                            "filter_sources": {p.name: digest(p) for p in sorted((ROOT/'integrations/ffmpeg').glob('*.c'))}}
    for device in devices:
        env = dict(os.environ, MVO_FEL_TEST_DEVICE=device["uuid"])
        for index, clip in enumerate(args.clips):
            name = f"{args.report.stem}-{device['uuid']}-{index}.log"
            log = args.report.with_name(name)
            print(f"Validation {device['name']} : scène {index+1}", flush=True)
            with log.open("w") as output:
                subprocess.run([str(args.oracle.resolve()), str(clip.resolve()), "0", "3", "8", "validate"],
                               env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
            text = log.read_text()
            cpu = [float(v) for v in re.findall(r"max_delta_cpu_pq12=([0-9.e+-]+)", text)]
            oracle = [float(v) for v in re.findall(r"max_delta_kernel64_pq12=([0-9.e+-]+)", text)]
            if len(cpu) != 3 or len(oracle) != 3 or max(*cpu, *oracle) > 2:
                raise RuntimeError(f"Validation numérique incomplète : {log}")
            pixels = re.findall(r"gpu_frame_bytes=(\d+) gpu_frame_sha256=([0-9a-f]{64})", text)
            if len(pixels) != 3 or len({size for size, _ in pixels}) != 1:
                raise RuntimeError(f"Empreintes des images de référence absentes : {log}")
            producer = validate_producer(lib, clip, device["uuid"], int(pixels[0][0]), [sha for _, sha in pixels])
            direct = (validate_direct(args.direct_ffmpeg, args.library, clip, device["uuid"], int(pixels[0][0]),
                                      [sha for _, sha in pixels]) if args.direct_ffmpeg else None)
            report["samples"].append({"device": device, "clip_sha256": digest(clip), "frames": 3,
                "max_cpu_pq12": max(cpu), "max_reference_pq12": max(oracle), "log": name,
                "producer": producer, "direct": direct})
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if (digest(args.library) != report["library_sha256"] or digest(args.oracle) != report["oracle_sha256"]
            or digest(ROOT / "src/gpu_backend.c") != report["backend_source_sha256"]
            or json.loads((ROOT / "dependencies.json").read_text()) != report["dependencies"]):
        raise RuntimeError("Le moteur ou ses sources ont changé pendant la validation")
    if args.direct_ffmpeg and (digest(args.direct_ffmpeg) != report["direct"]["binary_sha256"]
            or report["direct"]["filter_sources"] != {p.name: digest(p) for p in sorted((ROOT/'integrations/ffmpeg').glob('*.c'))}):
        raise RuntimeError("Le prototype direct a changé pendant la validation")
    report["passed"] = True
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"Validation stricte terminée : {args.report}", flush=True)


if __name__ == "__main__":
    main()
