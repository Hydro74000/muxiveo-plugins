"""Restauration (R2) de bout en bout dans le binaire : nombre de trames, déterminisme, ordre des niveaux, tramage 8 bits,
cumul avec l'interpolation.

Exige le binaire (``MUXIVEO_RIFE_BIN`` ou PATH), un GPU Vulkan et les graphes de restauration (``MUXIVEO_RESTORE_MODELS`` :
dossier contenant r2_flow.ncnn.* et r2_body.ncnn.*) ; ignorés sinon.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

RIFE_BIN = os.environ.get("MUXIVEO_RIFE_BIN") or shutil.which("muxiveo-rife") or ""
MODELS = os.environ.get("MUXIVEO_RESTORE_MODELS", "")
W, H, N = 320, 192, 30


def _ready() -> bool:
    if not RIFE_BIN or not Path(RIFE_BIN).is_file() or not MODELS or not (Path(MODELS) / "r2_body.ncnn.param").is_file():
        return False
    gpus = subprocess.run([RIFE_BIN, "--list-gpus"], capture_output=True, text=True, timeout=60)
    return gpus.returncode == 0 and bool(json.loads(gpus.stdout or "{}").get("gpus"))


pytestmark = pytest.mark.skipif(not _ready(), reason="binaire, GPU Vulkan et graphes de restauration requis")


def _source() -> bytes:
    """Dégradé lent qui défile + grain gaussien indépendant par image (10 bits, 4:2:0, plage limitée)."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W]
    out = [b"YUV4MPEG2 W%d H%d F24:1 Ip A1:1 C420p10 XYSCSS=420P10\n" % (W, H)]
    for t in range(N):
        base = 200 + 500 * (xx + 2 * t) / W + 60 * np.sin(yy / 17.0)
        y = np.clip(np.rint(base + rng.normal(0, 6, base.shape)), 64, 940).astype("<u2")
        c = np.full((H // 2, W // 2), 512, "<u2")
        out.append(b"FRAME\n" + y.tobytes() + c.tobytes() + c.tobytes())
    return b"".join(out)


def _run(src: bytes, *args: str) -> tuple[list[np.ndarray], str]:
    r = subprocess.run([RIFE_BIN, "--matrix", "bt709", "--chroma-loc", "left", "--restore-model", MODELS, *args],
                       input=src, capture_output=True, timeout=600)
    assert r.returncode == 0, r.stderr.decode()[-500:]
    data = r.stdout
    frames, i = [], data.index(b"\n") + 1
    size = (W * H + 2 * (W // 2) * (H // 2)) * 2
    while i < len(data):
        j = data.index(b"\n", i) + 1
        frames.append(np.frombuffer(data[j:j + size], "<u2").astype(np.int32))
        i = j + size
    return frames, r.stderr.decode()


def _grain(frames: list[np.ndarray], src: list[np.ndarray]) -> float:
    """Écart type du résidu image − moyenne temporelle de la source (grain restant), sur la luma."""
    mean = np.mean([f[: W * H] for f in src], 0)
    return float(np.mean([np.std(f[: W * H] - mean) for f in frames]))


def test_restore_alone_levels_and_determinism():
    src_bytes = _source()
    leger, log = _run(src_bytes, "--factor", "1", "--restore", "leger")
    fort, _ = _run(src_bytes, "--factor", "1", "--restore", "fort")
    fort2, _ = _run(src_bytes, "--factor", "1", "--restore", "fort")
    assert len(leger) == len(fort) == N
    assert all(np.array_equal(a, b) for a, b in zip(fort, fort2))  # déterministe, tramage compris
    raw = []
    data = src_bytes
    i = data.index(b"\n") + 1
    size = (W * H + 2 * (W // 2) * (H // 2)) * 2
    while i < len(data):
        j = data.index(b"\n", i) + 1
        raw.append(np.frombuffer(data[j:j + size], "<u2").astype(np.int32))
        i = j + size
    g_src, g_leger, g_fort = _grain(raw, raw), _grain(leger, raw), _grain(fort, raw)
    assert g_fort < g_leger <= g_src * 1.01  # Fort retire davantage de grain que Léger
    assert "restauration :" in log and "plan(s)" in log


def test_restore_dither_8bit_and_interpolation():
    src_bytes = _source()
    out8, _ = _run(src_bytes, "--factor", "1", "--restore", "auto", "--restore-dither", "8")
    assert all(np.all(f % 4 == 0) for f in out8)  # codes 8 bits exacts (× 4)
    alone, _ = _run(src_bytes, "--factor", "1", "--restore", "auto")
    both, log = _run(src_bytes, "--factor", "2", "--restore", "auto")
    assert len(both) == 2 * N
    assert all(np.array_equal(both[2 * i], alone[i]) for i in range(N))  # trames d'origine = trames restaurées
    assert "done in=%d out=%d" % (N, 2 * N) in log


TRT_PLUGIN = os.environ.get("MUXIVEO_RIFE_TRT_PLUGIN", "")


@pytest.mark.skipif(not TRT_PLUGIN or not (Path(MODELS) / "r2_body.onnx").is_file(),
                    reason="plugin TensorRT (MUXIVEO_RIFE_TRT_PLUGIN) et graphes ONNX de restauration requis")
def test_restore_tensorrt_matches_vulkan(tmp_path):
    src_bytes = _source()
    vk, _ = _run(src_bytes, "--factor", "1", "--restore", "auto", "--restore-dither", "off", "--backend", "vulkan")
    trt, log = _run(src_bytes, "--factor", "1", "--restore", "auto", "--restore-dither", "off", "--trt-plugin", TRT_PLUGIN,
                    "--trt-cache", str(tmp_path), "--backend", "tensorrt")
    assert "TensorRT" in log and len(trt) == N
    y = np.stack([f[: W * H] for f in trt]).astype(np.float64) - np.stack([f[: W * H] for f in vk])
    assert 10 * np.log10(1023 ** 2 / max(np.mean(y ** 2), 1e-9)) > 60  # mêmes graphes : écarts d'arrondi fp16 seulement


def _two_shots() -> bytes:
    """Deux plans : le dégradé granuleux, puis le même nettement plus clair et inversé (coupe à l'image N // 2)."""
    src = _source()
    i = src.index(b"\n") + 1
    size = (W * H + 2 * (W // 2) * (H // 2)) * 2
    out = [src[:i]]
    for t in range(N):
        j = i + t * (size + 6)
        frame = np.frombuffer(src[j + 6:j + 6 + size], "<u2").copy()
        if t >= N // 2:
            frame[: W * H] = np.clip(1004 - frame[: W * H] + 150, 64, 940)
        out.append(b"FRAME\n" + frame.tobytes())
    return b"".join(out)


def test_frame_statistics_follow_restoration_cuts(tmp_path):
    src = _two_shots()
    for factor, cut in (("1", N // 2), ("2", N)):
        stats = tmp_path / f"s{factor}.jsonl"
        frames, log = _run(src, "--factor", factor, "--restore", "auto", "--transfer", "pq", "--frame-statistics", str(stats))
        lines = [json.loads(line) for line in stats.read_text().splitlines()]
        records = [line for line in lines if "frame" in line]
        assert len(records) == len(frames) == N * int(factor)
        assert [r["frame"] for r in records if r["scene_start"]] == [0, cut]
        assert [(s["scene_first"], s["frames"]) for s in lines if "scene_first" in s] == [(0, cut), (cut, len(frames) - cut)]
        assert "2 plan(s)" in log


def test_source_statistics_precede_restoration(tmp_path):
    src = _source()
    restored, source, plain = tmp_path / "r.jsonl", tmp_path / "s.jsonl", tmp_path / "p.jsonl"
    _run(src, "--factor", "1", "--restore", "fort", "--transfer", "pq", "--frame-statistics", str(restored),
         "--source-statistics", str(source))
    subprocess.run([RIFE_BIN, "--factor", "1", "--matrix", "bt709", "--chroma-loc", "left", "--transfer", "pq", "--quiet",
                    "--frame-statistics", str(plain)], input=src, capture_output=True, check=True, timeout=120)
    frames = lambda path: [json.loads(line) for line in path.read_text().splitlines() if '"frame"' in line]  # noqa: E731
    assert [f["max_nits"] for f in frames(source)] == [f["max_nits"] for f in frames(plain)]
    assert max(max(f["max_nits"]) for f in frames(restored)) < max(max(f["max_nits"]) for f in frames(source))
