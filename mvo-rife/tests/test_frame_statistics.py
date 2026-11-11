"""Statistiques des trames produites (``--frame-statistics``) : format mvo-fel-frame-statistics, valeurs, plans.

Sans GPU (facteur 1 sans Restauration : recopie). Binaire : ``MUXIVEO_RIFE_BIN`` ou ``muxiveo-rife`` dans le PATH ;
ignorés sinon. Les plans issus des coupes de la Restauration sont vérifiés par ``test_restore_stage.py``.
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


def _supported() -> bool:
    if not RIFE_BIN or not Path(RIFE_BIN).is_file():
        return False
    caps = subprocess.run([RIFE_BIN, "--capabilities"], capture_output=True, text=True, timeout=60)
    return "frame-statistics" in json.loads(caps.stdout or "{}").get("options", [])


pytestmark = pytest.mark.skipif(not _supported(), reason="muxiveo-rife avec --frame-statistics requis")
W, H = 64, 32
M1, M2 = 2610 / 16384, 2523 / 4096 * 128
C1, C2, C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def _eotf(code: int) -> float:
    p = (code / 65535) ** (1 / M2)
    return float(np.float32(10000 * (max(p - C1, 0) / (C2 - C3 * p)) ** (1 / M1)))


def _y4m(lumas: list[int]) -> bytes:
    """Trames grises 10 bits (plage limitée, chroma neutre), une valeur de luma par trame."""
    out = [b"YUV4MPEG2 W%d H%d F24:1 Ip A1:1 C420p10 XYSCSS=420P10\n" % (W, H)]
    chroma = np.full((H // 2, W // 2), 512, "<u2").tobytes()
    for y in lumas:
        out.append(b"FRAME\n" + np.full((H, W), y, "<u2").tobytes() + chroma + chroma)
    return b"".join(out)


def _run(src: bytes, path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([RIFE_BIN, "--factor", "1", "--matrix", "bt2020nc", "--quiet", "--frame-statistics", str(path),
                           *args], input=src, capture_output=True, timeout=120)


def test_values_and_format(tmp_path):
    stats = tmp_path / "s.jsonl"
    lumas = [64, 512, 940]
    r = _run(_y4m(lumas), stats, "--transfer", "pq")
    assert r.returncode == 0, r.stderr.decode()
    lines = [json.loads(line) for line in stats.read_text().splitlines()]
    assert lines[0] == {"format": "mvo-fel-frame-statistics", "version": 1, "bins": 1024, "width": W, "height": H}
    frames = [line for line in lines if "frame" in line]
    scenes = [line for line in lines if "scene_first" in line]
    assert [f["frame"] for f in frames] == [0, 1, 2]
    assert [f["scene_start"] for f in frames] == [True, False, False]
    assert scenes == [{"scene_first": 0, "frames": 3, "maxrgb_hist": scenes[0]["maxrgb_hist"]}]
    for f, y in zip(frames, lumas):
        code = int(np.floor((y - 64) / 876 * 65535 + 0.5))  # R' = G' = B' = Y' (chroma neutre)
        nits = _eotf(code)
        assert f["max_nits"] == pytest.approx([nits] * 3, rel=1e-6)
        assert f["avg_nits"] == pytest.approx(nits, rel=1e-6)
        assert f["y100"] == (1.0 if nits <= 100 else 0.0)
        assert f["maxrgb_percentiles"] == pytest.approx([_eotf((code >> 6) * 64 + 32)] * 7, rel=1e-6)
        assert f["window"] == [0, 0, W, H]
    assert frames[2]["max_nits"][0] == pytest.approx(10000, rel=1e-4)
    assert sum(n for _, n in scenes[0]["maxrgb_hist"]) == 3 * W * H
    assert r.stdout.count(b"FRAME") == 3  # trames recopiées telles quelles


def test_requires_pq(tmp_path):
    r = _run(_y4m([512]), tmp_path / "s.jsonl")
    assert r.returncode != 0 and b"--transfer pq" in r.stderr


def test_source_statistics_measure_the_frames_read(tmp_path):
    produced, source = tmp_path / "out.jsonl", tmp_path / "in.jsonl"
    r = _run(_y4m([100, 600, 900, 300]), produced, "--transfer", "pq", "--source-statistics", str(source))
    assert r.returncode == 0, r.stderr.decode()
    assert source.read_text() == produced.read_text()  # recopie : trames lues = trames produites
    gpus = subprocess.run([RIFE_BIN, "--list-gpus"], capture_output=True, text=True, timeout=60)
    if gpus.returncode != 0 or not json.loads(gpus.stdout or "{}").get("gpus"):
        pytest.skip("interpolation : aucun périphérique Vulkan utilisable")
    r = _run(_y4m([512]), produced, "--transfer", "pq", "--factor", "2", "--source-statistics", str(source))
    assert r.returncode == 0 and len([line for line in source.read_text().splitlines() if '"frame"' in line]) == 1


def test_unwritable_statistics_are_reported(tmp_path):
    r = _run(_y4m([512]), tmp_path / "absent" / "s.jsonl", "--transfer", "pq")
    assert r.returncode != 0 and b"statistiques des trames" in r.stderr
