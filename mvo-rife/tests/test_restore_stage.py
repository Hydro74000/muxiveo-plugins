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
