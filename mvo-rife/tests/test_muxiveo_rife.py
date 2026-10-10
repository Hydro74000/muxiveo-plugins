"""Tests fonctionnels du binaire natif ``muxiveo-rife`` (mvo-rife).

Exécutés sur un vrai périphérique Vulkan (llvmpipe/lavapipe accepté en CI).
Binaire : ``MUXIVEO_RIFE_BIN`` ou ``muxiveo-rife`` dans le PATH ; ignorés sinon.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


def _rife_bin() -> str | None:
    candidate = os.environ.get("MUXIVEO_RIFE_BIN") or shutil.which("muxiveo-rife")
    return candidate if candidate and Path(candidate).is_file() else None


RIFE_BIN = _rife_bin() or ""


def _gpus() -> list[dict]:
    if not RIFE_BIN:
        return []
    out = subprocess.run([RIFE_BIN, "--list-gpus"], capture_output=True, text=True, timeout=120, check=False)
    try:
        return list(json.loads(out.stdout or "{}").get("gpus") or [])
    except ValueError:
        return []


GPUS = _gpus()

pytestmark = pytest.mark.skipif(
    not RIFE_BIN or not GPUS or shutil.which("ffmpeg") is None,
    reason="muxiveo-rife, un périphérique Vulkan et ffmpeg requis",
)


# ---------------------------------------------------------------------------
# y4m minimal
# ---------------------------------------------------------------------------

@dataclass
class Y4m:
    header: dict[str, str]
    frames: list[bytes]

    @property
    def fps(self) -> str:
        return self.header["F"]


def parse_y4m(data: bytes) -> Y4m:
    end = data.index(b"\n")
    tokens = data[:end].decode().split()
    assert tokens[0] == "YUV4MPEG2"
    header = {tok[0]: tok[1:] for tok in tokens[1:]}
    width, height = int(header["W"]), int(header["H"])
    cs = header.get("C", "420jpeg")
    depth = int(cs.split("p", 1)[1]) if "p" in cs[3:4] and cs[4:5].isdigit() else 8
    bps = 2 if depth > 8 else 1
    sub = {"420": (2, 2), "422": (2, 1), "444": (1, 1)}[cs[:3]]
    cw, ch = -(-width // sub[0]), -(-height // sub[1])
    size = (width * height + 2 * cw * ch) * bps
    frames: list[bytes] = []
    pos = end + 1
    while pos < len(data):
        line_end = data.index(b"\n", pos)
        assert data[pos:line_end].startswith(b"FRAME")
        pos = line_end + 1
        frames.append(data[pos:pos + size])
        pos += size
    return Y4m(header, frames)


def make_y4m(lavfi: str, *, frames: int, pix_fmt: str = "yuv420p", extra: list[str] | None = None) -> bytes:
    cmd = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", lavfi, "-frames:v", str(frames),
        *(extra or []), "-pix_fmt", pix_fmt, "-f", "yuv4mpegpipe", "-strict", "-1", "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def run_rife(data: bytes, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [RIFE_BIN, "--quiet", "-g", str(_test_gpu()), *args],
        input=data, capture_output=True, timeout=600, check=False,
    )


def _test_gpu() -> int:
    """GPU des tests : ``MUXIVEO_RIFE_GPU`` sinon le périphérique par défaut."""
    return int(os.environ.get("MUXIVEO_RIFE_GPU", "-1"))


def _psnr(a: bytes, b: bytes, depth: int) -> float:
    dtype = np.uint16 if depth > 8 else np.uint8
    x = np.frombuffer(a, dtype=dtype).astype(np.float64)
    y = np.frombuffer(b, dtype=dtype).astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    peak = float((1 << depth) - 1)
    return 99.0 if mse == 0 else 10 * np.log10(peak * peak / mse)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_version_and_gpu_listing() -> None:
    out = subprocess.run([RIFE_BIN, "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.startswith("muxiveo-rife ")
    assert all({"index", "name", "type"} <= set(gpu) for gpu in GPUS)


def test_overlong_y4m_header_is_rejected() -> None:
    """Un en-tête excessif est refusé avant la lecture des trames."""
    header = b"YUV4MPEG2 W160 H96 F25:1 Ip C420jpeg X" + b"a" * 4096 + b"\n"
    result = run_rife(header, "--factor", "2")
    assert result.returncode == 2
    assert b"en-t\xc3\xaate illisible" in result.stderr
    assert result.stdout == b""


@pytest.mark.parametrize("io_mode", ["files", "stdin", "stdout", "pipe"])
def test_distinct_files_and_pipes_preserve_direct_stream(tmp_path: Path, io_mode: str) -> None:
    """La protection des fichiers préserve les quatre modes E/S directs."""
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=2)
    source = tmp_path / "entrée.y4m"
    output = tmp_path / "sortie.y4m"
    source.write_bytes(data)
    stdin = io_mode in {"stdin", "pipe"}
    stdout = io_mode in {"stdout", "pipe"}
    result = subprocess.run(
        [RIFE_BIN, "-i", "-" if stdin else str(source),
         "-o", "-" if stdout else str(output), "--factor", "1"],
        input=data if stdin else None,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    actual = parse_y4m(result.stdout if stdout else output.read_bytes())
    expected = parse_y4m(data)
    assert actual.fps == expected.fps
    assert actual.frames == expected.frames
    assert source.read_bytes() == data


@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le"])
def test_factor_two_keeps_originals_bit_exact(pix_fmt: str) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=8, pix_fmt=pix_fmt)
    src = parse_y4m(data)
    proc = run_rife(data, "--factor", "2", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.fps == "50:1"
    assert len(out.frames) == 2 * len(src.frames)
    assert out.frames[0::2] == src.frames
    # dernière trame : duplication de la dernière source (pas de trame suivante)
    assert out.frames[-1] == src.frames[-1]
    # trames générées distinctes des sources (mouvement réel)
    assert out.frames[1] not in (src.frames[0], src.frames[1])
    assert b"done in=8 out=16" in proc.stderr


def test_factor_three_and_target_fps() -> None:
    data = make_y4m("testsrc2=size=128x64:rate=24000/1001", frames=5)
    proc = run_rife(data, "--factor", "3", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.fps == "72000:1001"
    assert len(out.frames) == 15

    proc = run_rife(data, "--fps", "60000/1001", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.fps == "60000:1001"
    assert len(out.frames) == 13  # ceil(5 × 2,5)


@pytest.mark.parametrize(
    ("pix_fmt", "min_psnr"),
    [("yuv420p", 60.0), ("yuv420p10le", 60.0), ("yuv444p16le", 70.0)],
)
def test_conversion_roundtrip_is_near_lossless(pix_fmt: str, min_psnr: float) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=2, pix_fmt=pix_fmt)
    src = parse_y4m(data)
    # testsrc2 est converti en YUV par swscale avec la matrice BT.601
    proc = run_rife(data, "--debug-roundtrip", "--matrix", "bt601")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    depth = 16 if "16" in pix_fmt else (10 if "10" in pix_fmt else 8)
    for a, b in zip(src.frames, out.frames):
        assert _psnr(a, b, depth) >= min_psnr


def test_scene_cut_duplicates_previous_frame() -> None:
    first = make_y4m("testsrc2=size=128x64:rate=25", frames=3)
    second = make_y4m("smptebars=size=128x64:rate=25", frames=3)
    data = first + second[second.index(b"\n") + 1:]
    src = parse_y4m(data)
    proc = run_rife(data, "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.frames[5] == src.frames[2]  # trame entre la coupe : duplication
    assert b"scenes=1" in proc.stderr


def test_static_pairs_are_duplicated() -> None:
    data = make_y4m("color=c=gray:size=96x64:rate=25", frames=4)
    src = parse_y4m(data)
    proc = run_rife(data, "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert all(frame == src.frames[0] for frame in out.frames)
    assert b"interpolated=0" in proc.stderr


def test_interlaced_input_is_rejected() -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    data = data.replace(b" Ip ", b" It ", 1)
    proc = run_rife(data, "--matrix", "bt709")
    assert proc.returncode == 2
    assert b"entrelac" in proc.stderr


def test_unknown_model_fails_with_gpu_exit_code() -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    proc = run_rife(data, "--model", "rife-inexistant", "--matrix", "bt709")
    assert proc.returncode == 3


@pytest.mark.parametrize("model", ["rife-v4.6", "rife-v4.15-lite"])
def test_uhd_mode_keeps_originals_and_changes_interpolation(model: str) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=4, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    normal = run_rife(data, "--model", model, "--factor", "2", "--matrix", "bt709")
    uhd = run_rife(data, "--model", model, "--uhd", "--factor", "2", "--matrix", "bt709")
    assert normal.returncode == 0, normal.stderr.decode()
    assert uhd.returncode == 0, uhd.stderr.decode()
    out = parse_y4m(uhd.stdout)
    assert len(out.frames) == 2 * len(src.frames)
    assert out.frames[0::2] == src.frames
    # graphe réécrit (échelles x2) : trames générées différentes du mode normal
    assert out.frames[1] != parse_y4m(normal.stdout).frames[1]


def test_uhd_mode_rejects_unrecognized_graph(tmp_path: Path) -> None:
    model = tmp_path / "rife-sans-interp"
    model.mkdir()
    (model / "flownet.param").write_text("7767517\n1 1\nInput in0 0 1 in0\n")
    (model / "flownet.bin").write_bytes(b"")
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    proc = run_rife(data, "--model", str(model), "--uhd", "--matrix", "bt709")
    assert proc.returncode == 3
    assert "mode UHD" in proc.stderr.decode()


@pytest.mark.parametrize("tta", ["2", "4", "8"])
def test_tta_keeps_originals_and_averages_variants(tta: str) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=4, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    plain = run_rife(data, "--factor", "2", "--matrix", "bt709")
    averaged = run_rife(data, "--tta", tta, "--factor", "2", "--matrix", "bt709")
    assert plain.returncode == 0, plain.stderr.decode()
    assert averaged.returncode == 0, averaged.stderr.decode()
    out = parse_y4m(averaged.stdout)
    assert len(out.frames) == 2 * len(src.frames)
    assert out.frames[0::2] == src.frames
    # moyenne de variantes : trame générée différente de l'inférence simple, mais proche
    plain_frame = parse_y4m(plain.stdout).frames[1]
    assert out.frames[1] != plain_frame
    assert _psnr(out.frames[1], plain_frame, 10) > 25.0


def test_tta_rejects_unsupported_count() -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    proc = run_rife(data, "--tta", "3", "--matrix", "bt709")
    assert proc.returncode == 1
    assert b"--tta" in proc.stderr


# ---------------------------------------------------------------------------
# Moteur hybride (RIFE + compensation de mouvement par blocs), 1.3.0+
# ---------------------------------------------------------------------------

def _rife_version() -> tuple[int, ...]:
    if not RIFE_BIN:
        return ()
    out = subprocess.run([RIFE_BIN, "--version"], capture_output=True, text=True, timeout=15, check=False).stdout
    parts = (out.split() + ["", ""])[1].split(".")
    return tuple(int(p) for p in parts if p.isdigit())


needs_hybrid = pytest.mark.skipif(_rife_version() < (1, 3, 0), reason="moteur hybride : muxiveo-rife ≥ 1.3.0")


@needs_hybrid
def test_gpu_listing_reports_nvof_availability() -> None:
    assert all(isinstance(gpu.get("nvof"), bool) and gpu.get("nvof_status") for gpu in GPUS)


@needs_hybrid
@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le"])
def test_hybrid_factor_two_keeps_originals_bit_exact(pix_fmt: str) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=8, pix_fmt=pix_fmt)
    src = parse_y4m(data)
    proc = run_rife(data, "--engine", "hybrid", "--factor", "2", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.fps == "50:1"
    assert out.frames[0::2] == src.frames
    assert out.frames[1] not in (src.frames[0], src.frames[1])
    assert b"done in=8 out=16" in proc.stderr
    depth = 10 if "10" in pix_fmt else 8
    plain = parse_y4m(run_rife(data, "--factor", "2", "--matrix", "bt709").stdout)
    # même scène que RIFE seul, sans être la même image
    assert _psnr(out.frames[1], plain.frames[1], depth) > 20.0


@needs_hybrid
def test_hybrid_target_fps_and_factor_four() -> None:
    data = make_y4m("testsrc2=size=128x64:rate=24000/1001", frames=5)
    proc = run_rife(data, "--engine", "hybrid", "--fps", "60000/1001", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.fps == "60000:1001"
    assert len(out.frames) == 13  # ceil(5 × 2,5)
    proc = run_rife(data, "--engine", "hybrid", "--factor", "4", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert len(out.frames) == 20
    assert out.frames[0::4] == parse_y4m(data).frames


@needs_hybrid
def test_hybrid_pairs_do_not_share_state() -> None:
    """Chaque paire est interpolée indépendamment, même après une paire figée sautée."""
    base = parse_y4m(make_y4m("testsrc2=size=160x96:rate=25", frames=5))
    header = make_y4m("testsrc2=size=160x96:rate=25", frames=1).split(b"\n", 1)[0] + b"\n"

    def stream(frames: list[bytes]) -> bytes:
        return header + b"".join(b"FRAME\n" + f for f in frames)

    frames = [base.frames[0], base.frames[1], base.frames[1], base.frames[2], base.frames[3], base.frames[4]]
    args = ("--engine", "hybrid", "--nvof", "off", "--factor", "2", "--matrix", "bt709")
    full = run_rife(stream(frames), *args)
    assert full.returncode == 0, full.stderr.decode()
    out = parse_y4m(full.stdout)
    assert b"static=1" in full.stderr
    for k in (0, 2, 3, 4):
        pair = run_rife(stream(frames[k:k + 2]), *args)
        assert pair.returncode == 0, pair.stderr.decode()
        assert out.frames[2 * k + 1] == parse_y4m(pair.stdout).frames[1], f"paire {k}"


@needs_hybrid
def test_hybrid_keeps_scene_cut_and_static_duplication() -> None:
    first = make_y4m("testsrc2=size=128x64:rate=25", frames=3)
    second = make_y4m("smptebars=size=128x64:rate=25", frames=3)
    data = first + second[second.index(b"\n") + 1:]
    src = parse_y4m(data)
    proc = run_rife(data, "--engine", "hybrid", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    assert parse_y4m(proc.stdout).frames[5] == src.frames[2]
    assert b"scenes=1" in proc.stderr

    still = make_y4m("color=c=gray:size=96x64:rate=25", frames=4)
    proc = run_rife(still, "--engine", "hybrid", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    assert b"interpolated=0" in proc.stderr


@needs_hybrid
@pytest.mark.parametrize("extra", [["--tta", "2"], ["--uhd"], ["--model", "rife-v4.15-lite"]])
def test_hybrid_combines_with_rife_options(extra: list[str]) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=4, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    proc = run_rife(data, "--engine", "hybrid", *extra, "--factor", "2", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.frames[0::2] == src.frames


@needs_hybrid
def test_hybrid_reports_optical_flow_status() -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=3)
    gpu = GPUS[_test_gpu()] if _test_gpu() >= 0 else next(
        (g for g in GPUS if g["index"] == 0), GPUS[0])
    for nvof, expected in (("off", "désactivé"), ("auto", "actif" if gpu.get("nvof") else "indisponible")):
        proc = subprocess.run(
            [RIFE_BIN, "-g", str(_test_gpu()), "--engine", "hybrid", "--nvof", nvof, "--matrix", "bt709"],
            input=data, capture_output=True, timeout=120, check=False,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        assert f"moteur hybrid | flux optique NVIDIA : {expected}" in proc.stderr.decode()


@needs_hybrid
@pytest.mark.parametrize("engine", ["rife", "hybrid", "mc"])
@pytest.mark.parametrize("size", ["1742x676", "1750x660"])
def test_frame_size_not_multiple_of_16_bytes(engine: str, size: str) -> None:
    # trame de taille non multiple de 16 octets : ncnn copie la taille alignée (débordement corrigé en 1.5.0)
    data = make_y4m(f"testsrc2=size={size}:rate=24", frames=4, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    proc = run_rife(data, "--engine", engine, "--factor", "2", "--matrix", "bt709")
    assert proc.returncode == 0, proc.stderr.decode()
    assert parse_y4m(proc.stdout).frames[0::2] == src.frames


@needs_hybrid
@pytest.mark.parametrize("args", [["--engine", "magic"], ["--nvof", "on"]])
def test_hybrid_rejects_unknown_option_values(args: list[str]) -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    proc = run_rife(data, *args, "--matrix", "bt709")
    assert proc.returncode == 1
    assert args[0].encode() in proc.stderr


# ---------------------------------------------------------------------------
# Grands mouvements du moteur hybride (RIFE à flux demi-résolution), 1.5.0+
# ---------------------------------------------------------------------------

needs_large_motion = pytest.mark.skipif(_rife_version() < (1, 5, 0), reason="grands mouvements : muxiveo-rife ≥ 1.5.0")


def _panning(step: int) -> bytes:
    """Panoramique horizontal de ``step`` px par image (mire recadrée)."""
    return make_y4m(f"testsrc2=size=640x192:rate=25,crop=320:192:n*{step}:0", frames=6)


def _run_verbose(data: bytes, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([RIFE_BIN, "-g", str(_test_gpu()), *args], input=data, capture_output=True,
                          timeout=600, check=False)


@needs_large_motion
def test_large_motion_pass_replaces_hybrid_on_fast_pan() -> None:
    data = _panning(48)
    src = parse_y4m(data)
    common = ("--engine", "hybrid", "--nvof", "off", "--scene-threshold", "0", "--factor", "2", "--matrix", "bt709")
    on = _run_verbose(data, *common)
    off = _run_verbose(data, *common, "--large-motion", "off")
    assert on.returncode == 0, on.stderr.decode()
    assert off.returncode == 0, off.stderr.decode()
    assert "grands mouvements : RIFE flux demi-résolution au-delà de 16 px" in on.stderr.decode()
    assert "grands mouvements : 5 paire(s) sur 5" in on.stderr.decode()
    assert "grands mouvements : désactivé" in off.stderr.decode()
    out_on, out_off = parse_y4m(on.stdout), parse_y4m(off.stdout)
    assert out_on.frames[0::2] == src.frames
    assert out_on.frames[1::2] != out_off.frames[1::2]


@needs_large_motion
def test_large_motion_pass_skipped_below_threshold() -> None:
    data = _panning(2)
    common = ("--engine", "hybrid", "--nvof", "off", "--scene-threshold", "0", "--factor", "2", "--matrix", "bt709")
    on = _run_verbose(data, *common, "--large-motion", "64")
    off = _run_verbose(data, *common, "--large-motion", "off")
    assert on.returncode == 0, on.stderr.decode()
    assert "grands mouvements : 0 paire(s) sur 5" in on.stderr.decode()
    # aucune paire au-delà du seuil : image identique à l'hybride seul (padding x2 sans effet sur 320x192)
    assert parse_y4m(on.stdout).frames == parse_y4m(off.stdout).frames


@needs_large_motion
def test_large_motion_option_values() -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    bad = run_rife(data, "--engine", "hybrid", "--large-motion", "fast", "--matrix", "bt709")
    assert bad.returncode == 1 and b"--large-motion" in bad.stderr
    custom = _run_verbose(data, "--engine", "hybrid", "--large-motion", "24", "--matrix", "bt709")
    assert custom.returncode == 0, custom.stderr.decode()
    assert "au-delà de 24 px" in custom.stderr.decode()
    uhd = _run_verbose(data, "--engine", "hybrid", "--uhd", "--matrix", "bt709")
    assert uhd.returncode == 0, uhd.stderr.decode()
    assert "grands mouvements : flux demi-résolution partout (--uhd)" in uhd.stderr.decode()


# ---------------------------------------------------------------------------
# Sélecteur appris par blocs et mode Ultra, 1.6.0+
# ---------------------------------------------------------------------------

needs_selector = pytest.mark.skipif(_rife_version() < (1, 6, 0), reason="sélecteur appris : muxiveo-rife ≥ 1.6.0")
_SEL = ("--engine", "hybrid", "--scene-threshold", "0", "--matrix", "bt709")


@needs_selector
@pytest.mark.parametrize("extra", [[], ["--ultra"], ["--nvof", "off"], ["--ultra", "--nvof", "off"]])
@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le"])
def test_selector_keeps_originals_bit_exact(extra: list[str], pix_fmt: str) -> None:
    data = _panning(6) if pix_fmt == "yuv420p" else make_y4m(
        "testsrc2=size=640x192:rate=25,crop=320:192:n*6:0", frames=6, pix_fmt=pix_fmt)
    src = parse_y4m(data)
    proc = run_rife(data, *_SEL, *extra, "--factor", "2")
    assert proc.returncode == 0, proc.stderr.decode()
    out = parse_y4m(proc.stdout)
    assert out.frames[0::2] == src.frames
    assert all(f not in src.frames for f in out.frames[1:-1:2])  # dernière image : duplication de fin de flux
    fps = run_rife(data, *_SEL, *extra, "--fps", "60/1")
    assert fps.returncode == 0, fps.stderr.decode()
    assert len(parse_y4m(fps.stdout).frames) == 15  # ceil(6 × 2,4)


@needs_selector
def test_selector_reports_weights_and_changes_interpolation() -> None:
    data = _panning(6)
    learned = _run_verbose(data, *_SEL, "--nvof", "off", "--model", "rife-v4.15")
    rule = _run_verbose(data, *_SEL, "--nvof", "off", "--model", "rife-v4.15", "--selector", "off")
    ultra = _run_verbose(data, *_SEL, "--nvof", "off", "--model", "rife-v4.15", "--ultra")
    for proc in (learned, rule, ultra):
        assert proc.returncode == 0, proc.stderr.decode()
    assert "sélecteur : appris (v4.15)" in learned.stderr.decode()
    assert "sélecteur : règle fixe" in rule.stderr.decode()
    assert "sélecteur : appris (v4.15, Ultra)" in ultra.stderr.decode()
    frames = [parse_y4m(p.stdout).frames[1::2] for p in (learned, rule, ultra)]
    assert frames[0] != frames[1] and frames[0] != frames[2]
    # même scène : écarts limités aux zones où les candidats divergent
    assert all(_psnr(a, b, 8) > 25.0 for a, b in zip(frames[0], frames[2]))
    v46 = _run_verbose(data, *_SEL, "--nvof", "off", "--model", "rife-v4.6")
    assert v46.returncode == 0, v46.stderr.decode()
    assert "sélecteur : appris (v4.6)" in v46.stderr.decode()


@needs_selector
def test_selector_uses_fine_tuned_model_weights() -> None:
    models = Path(RIFE_BIN).resolve().parent / "rife-models"
    if not (models / "rife-v4.15-mvo1" / "flownet.param").is_file():
        pytest.skip("modèle rife-v4.15-mvo1 absent")
    data = _panning(6)
    src = parse_y4m(data)
    proc = _run_verbose(data, *_SEL, "--nvof", "off", "--model", "rife-v4.15-mvo1", "--factor", "2")
    assert proc.returncode == 0, proc.stderr.decode()
    assert "sélecteur : appris (v4.15 MVO)" in proc.stderr.decode()
    assert parse_y4m(proc.stdout).frames[0::2] == src.frames


@needs_selector
@pytest.mark.parametrize("nvof", ["off", "auto"])
def test_quality_cached_sources_preserve_common_timesteps(nvof: str) -> None:
    """Les caches par paire ne modifient ni t=0,5, ni les sources après d'autres temps."""
    models = Path(RIFE_BIN).resolve().parent / "rife-models"
    if not (models / "rife-v4.15-mvo1" / "flownet.param").is_file():
        pytest.skip("modèle rife-v4.15-mvo1 absent")
    # Panoramique assez rapide pour exercer également le réseau à flux demi-résolution.
    data = _panning(48)
    args = (*_SEL, "--model", "rife-v4.15-mvo1", "--nvof", nvof)
    twice = run_rife(data, *args, "--factor", "2")
    four = run_rife(data, *args, "--factor", "4")
    assert twice.returncode == four.returncode == 0, (twice.stderr + four.stderr).decode()
    out2, out4 = parse_y4m(twice.stdout), parse_y4m(four.stdout)
    assert out4.frames[0::4] == parse_y4m(data).frames
    assert out4.frames[2::4] == out2.frames[1::2]


@needs_selector
def test_quality_multiple_timesteps_do_not_share_previous_pair() -> None:
    """Texture et écart A/B doivent être recalculés même sans NVOF à la paire suivante."""
    models = Path(RIFE_BIN).resolve().parent / "rife-models"
    if not (models / "rife-v4.15-mvo1" / "flownet.param").is_file():
        pytest.skip("modèle rife-v4.15-mvo1 absent")
    data = make_y4m("testsrc2=size=668x404:rate=24,crop=334:202:n*8:0", frames=5, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    header = data.split(b"\n", 1)[0] + b"\n"
    args = (*_SEL, "--model", "rife-v4.15-mvo1", "--nvof", "off", "--factor", "4")
    full = run_rife(data, *args)
    assert full.returncode == 0, full.stderr.decode()
    out = parse_y4m(full.stdout)
    for k in (0, 2, 3):
        pair = run_rife(header + b"".join(b"FRAME\n" + f for f in src.frames[k:k + 2]), *args)
        assert pair.returncode == 0, pair.stderr.decode()
        assert out.frames[4*k+1:4*k+4] == parse_y4m(pair.stdout).frames[1:4], f"paire {k}"


@needs_selector
@pytest.mark.parametrize(("source_rate", "target"), [("24000:1001", "60000/1001"), ("24:1", "60/1")])
@pytest.mark.parametrize("nvof", ["off", "auto"])
@pytest.mark.parametrize("backend", ["vulkan", "tensorrt"])
def test_quality_two_and_half_uses_correct_pair_and_timestep(source_rate: str, target: str,
                                                            nvof: str, backend: str) -> None:
    """À ×2,5, les temps 0,4/0,8 puis 0,2/0,6 restent propres à chaque paire."""
    models = Path(RIFE_BIN).resolve().parent / "rife-models"
    if not (models / "rife-v4.15-mvo1" / "flownet.param").is_file():
        pytest.skip("modèle rife-v4.15-mvo1 absent")
    gpu = _test_gpu()
    extra: tuple[str, ...] = ()
    if backend == "tensorrt":
        if not TRT_PLUGIN or not any(g.get("trt_compatible") for g in GPUS):
            pytest.skip("plugin TensorRT et GPU NVIDIA compatible requis")
        gpu = _trt_gpu()
        extra = ("--trt-plugin", TRT_PLUGIN)
    data = _panning(48).replace(b" F25:1 ", f" F{source_rate} ".encode(), 1)
    common = (RIFE_BIN, "--quiet", "-g", str(gpu), *_SEL, "--model", "rife-v4.15-mvo1",
              "--nvof", nvof, "--backend", backend, *extra)
    def invoke(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([*common, *args], input=data, capture_output=True, timeout=600, check=False)
    five, fractional = invoke("--factor", "5"), invoke("--fps", target)
    assert five.returncode == fractional.returncode == 0, (five.stderr + fractional.stderr).decode()
    out5, out25 = parse_y4m(five.stdout), parse_y4m(fractional.stdout)
    assert out25.frames == out5.frames[0::2]


@needs_selector
@pytest.mark.parametrize("extra", [[], ["--uhd"], ["--fp32"], ["--tta", "2"], ["--tta", "4"], ["--tta", "8"], ["--ultra"]])
def test_optional_head_cache_preserves_quality_at_two_and_half(extra: list[str]) -> None:
    """Le cache optionnel garde les mêmes sorties avec plusieurs t et plusieurs paires."""
    models = Path(RIFE_BIN).resolve().parent / "rife-models"
    if not (models / "rife-v4.15-mvo1" / "flownet.param").is_file():
        pytest.skip("modèle rife-v4.15-mvo1 absent")
    data = _panning(48).replace(b" F25:1 ", b" F24000:1001 ", 1)
    args = (*_SEL, "--model", "rife-v4.15-mvo1", "--backend", "vulkan", "--nvof", "off",
            "--fps", "60000/1001", *extra)
    plain, cached = run_rife(data, *args), _run_verbose(data, *args, "--feature-cache")
    assert plain.returncode == cached.returncode == 0, (plain.stderr + cached.stderr).decode()
    assert parse_y4m(plain.stdout).frames == parse_y4m(cached.stdout).frames
    assert "cache des têtes ncnn :" in cached.stderr.decode()


@needs_selector
def test_optional_head_cache_falls_back_when_reserve_cannot_be_kept() -> None:
    """Une réserve supérieure au budget interdit le cache, sans empêcher l'interpolation."""
    data = _panning(6)
    args = (*_SEL, "--model", "rife-v4.15", "--backend", "vulkan", "--nvof", "off", "--fps", "60/1")
    plain = run_rife(data, *args)
    cached = _run_verbose(data, *args, "--feature-cache", "--feature-cache-reserve", "2147483647")
    assert plain.returncode == cached.returncode == 0, (plain.stderr + cached.stderr).decode()
    assert parse_y4m(plain.stdout).frames == parse_y4m(cached.stdout).frames
    assert "cache des têtes ncnn : 0 réutilisation(s)" in cached.stderr.decode()
    bad = run_rife(data, *args, "--feature-cache-reserve", "0")
    assert bad.returncode == 1 and b"--feature-cache-reserve" in bad.stderr


@pytest.mark.parametrize("model", ["rife-v4.6", "rife-v4.15", "rife-v4.15-lite"])
@pytest.mark.parametrize("extra", [[], ["--uhd"], ["--tta", "8"]])
def test_optional_head_cache_preserves_other_rife_models(model: str, extra: list[str]) -> None:
    """RIFE seul : graphe compatible mis en cache, autres graphes recalculés sans changement."""
    models = Path(RIFE_BIN).resolve().parent / "rife-models"
    if not (models / model / "flownet.param").is_file():
        pytest.skip(f"modèle {model} absent")
    data = _panning(6).replace(b" F25:1 ", b" F24000:1001 ", 1)
    args = ("--model", model, "--backend", "vulkan", "--scene-threshold", "0", "--fps", "60000/1001", *extra)
    plain, cached = run_rife(data, *args), run_rife(data, *args, "--feature-cache")
    assert plain.returncode == cached.returncode == 0, (plain.stderr + cached.stderr).decode()
    assert parse_y4m(plain.stdout).frames == parse_y4m(cached.stdout).frames


@needs_selector
@pytest.mark.parametrize("size", ["96x64", "334x202", "720x480", "1742x676"])
def test_selector_any_size_without_seams(size: str) -> None:
    # bloc proportionnel à la largeur (8 px en SD … 64 px en 8K) : tailles non multiples du bloc acceptées
    data = make_y4m(f"testsrc2=size={size}:rate=24", frames=3, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    for extra in ([], ["--ultra"]):
        proc = run_rife(data, *_SEL, *extra, "--factor", "2")
        assert proc.returncode == 0, proc.stderr.decode()
        assert parse_y4m(proc.stdout).frames[0::2] == src.frames
    # aplat : tous les candidats concordent, la pondération par blocs ne crée aucune couture
    flat = make_y4m(f"color=c=0x406080:size={size}:rate=24,noise=alls=1:allf=t", frames=3)
    proc = run_rife(flat, *_SEL, "--ultra", "--factor", "2")
    assert proc.returncode == 0, proc.stderr.decode()
    mid = np.frombuffer(parse_y4m(proc.stdout).frames[1], dtype=np.uint8)
    w, h = (int(v) for v in size.split("x"))
    luma = mid[: w * h].reshape(h, w).astype(np.int16)
    assert int(np.abs(np.diff(luma, axis=1)).max()) <= 8 and int(np.abs(np.diff(luma, axis=0)).max()) <= 8


@needs_selector
def test_ultra_pairs_do_not_share_state() -> None:
    src = parse_y4m(_panning(6))
    header = _panning(6).split(b"\n", 1)[0] + b"\n"
    args = (*_SEL, "--ultra", "--nvof", "off", "--factor", "2")
    full = parse_y4m(run_rife(header + b"".join(b"FRAME\n" + f for f in src.frames), *args).stdout)
    for k in (0, 3):
        pair = run_rife(header + b"".join(b"FRAME\n" + f for f in src.frames[k:k + 2]), *args)
        assert pair.returncode == 0, pair.stderr.decode()
        assert full.frames[2 * k + 1] == parse_y4m(pair.stdout).frames[1], f"paire {k}"


@needs_selector
@pytest.mark.parametrize(
    "args", [["--ultra"], ["--ultra", "--engine", "hybrid", "--uhd"], ["--ultra", "--engine", "hybrid", "--tta", "2"],
             ["--ultra", "--engine", "hybrid", "--selector", "off"], ["--engine", "hybrid", "--selector", "on"]])
def test_ultra_and_selector_option_values(args: list[str]) -> None:
    proc = run_rife(make_y4m("testsrc2=size=96x64:rate=25", frames=2), *args, "--matrix", "bt709")
    assert proc.returncode == 1
    assert b"--ultra" in proc.stderr or b"--selector" in proc.stderr


@needs_selector
def test_selector_uses_nvidia_flow_candidate_when_available() -> None:
    gpu = GPUS[_test_gpu()] if _test_gpu() >= 0 else next((g for g in GPUS if g["index"] == 0), GPUS[0])
    if not gpu.get("nvof"):
        pytest.skip("flux optique NVIDIA indisponible sur ce GPU")
    data = _panning(6)
    src = parse_y4m(data)
    for extra in ([], ["--ultra"]):
        proc = _run_verbose(data, *_SEL, *extra, "--factor", "2")
        assert proc.returncode == 0, proc.stderr.decode()
        assert "sélecteur : 5 image(s) avec le candidat flux NVIDIA" in proc.stderr.decode()
        assert parse_y4m(proc.stdout).frames[0::2] == src.frames
    off = _run_verbose(data, *_SEL, "--nvof", "off", "--factor", "2")
    assert "sélecteur : 0 image(s) avec le candidat flux NVIDIA" in off.stderr.decode()


# ---------------------------------------------------------------------------
# Plugin TensorRT facultatif (mvo-rife-trt), 1.4.0+
# ---------------------------------------------------------------------------

needs_trt_support = pytest.mark.skipif(_rife_version() < (1, 4, 0), reason="plugin TensorRT : muxiveo-rife ≥ 1.4.0")
TRT_PLUGIN = os.environ.get("MUXIVEO_RIFE_TRT_PLUGIN", "")
needs_trt_plugin = pytest.mark.skipif(
    not TRT_PLUGIN or not any(g.get("trt_compatible") for g in GPUS),
    reason="plugin TensorRT (MUXIVEO_RIFE_TRT_PLUGIN) et GPU NVIDIA compatible requis",
)


@needs_trt_support
def test_gpu_listing_reports_tensorrt_compatibility() -> None:
    for gpu in GPUS:
        assert isinstance(gpu.get("trt_compatible"), bool) and gpu.get("trt_status")
        if "NVIDIA" not in gpu["name"]:
            assert gpu["trt_compatible"] is False


@needs_trt_support
def test_missing_trt_plugin_falls_back_to_vulkan(tmp_path: Path) -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=3)
    proc = subprocess.run(
        [RIFE_BIN, "-g", str(_test_gpu()), "--trt-plugin", str(tmp_path / "absent"), "--matrix", "bt709"],
        input=data, capture_output=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert "inférence RIFE : Vulkan (TensorRT indisponible" in proc.stderr.decode()
    assert len(parse_y4m(proc.stdout).frames) == 6


@needs_trt_support
def test_tensorrt_backend_without_plugin_is_refused() -> None:
    data = make_y4m("testsrc2=size=96x64:rate=25", frames=2)
    proc = run_rife(data, "--backend", "tensorrt", "--matrix", "bt709")
    assert proc.returncode == 3
    assert b"TensorRT" in proc.stderr


@needs_trt_support
@needs_trt_plugin
@pytest.mark.parametrize("extra", [[], ["--engine", "hybrid"], ["--uhd"], ["--tta", "2"],
                                  ["--tta", "4"], ["--tta", "8"], ["--engine", "hybrid", "--ultra"]])
def test_tensorrt_inference_keeps_originals_and_matches_vulkan(tmp_path: Path, extra: list[str]) -> None:
    data = make_y4m("testsrc2=size=160x96:rate=25", frames=6, pix_fmt="yuv420p10le")
    src = parse_y4m(data)
    common = ["--factor", "2", "--matrix", "bt709", "--trt-plugin", TRT_PLUGIN, "--trt-cache", str(tmp_path), *extra]
    trt = subprocess.run([RIFE_BIN, "-g", str(_trt_gpu()), "--backend", "tensorrt", *common],
                         input=data, capture_output=True, timeout=600, check=False)
    vk = subprocess.run([RIFE_BIN, "-g", str(_trt_gpu()), "--backend", "vulkan", *common],
                        input=data, capture_output=True, timeout=600, check=False)
    assert trt.returncode == 0, trt.stderr.decode()
    assert vk.returncode == 0, vk.stderr.decode()
    assert "inférence RIFE : TensorRT" in trt.stderr.decode()
    out = parse_y4m(trt.stdout)
    assert len(out.frames) == 2 * len(src.frames)
    assert out.frames[0::2] == src.frames
    # même modèle, deux implémentations (ncnn TNTwise / ONNX des poids officiels, fp16) : images proches.
    # Sur une mire 160x96 en fort mouvement, les bords (traitement légèrement différent) pèsent lourd :
    # 32 à 36 dB ici, écart < 0,1 dB sur les bancs 4K réels.
    assert _psnr(out.frames[1], parse_y4m(vk.stdout).frames[1], 10) > 28.0


def _trt_gpu() -> int:
    """Premier GPU compatible TensorRT."""
    return next(g["index"] for g in GPUS if g.get("trt_compatible"))


@needs_trt_support
@needs_trt_plugin
@pytest.mark.parametrize("extra", [[], ["--engine", "hybrid"], ["--uhd"], ["--tta", "2"]])
def test_tensorrt_cached_inputs_preserve_common_timesteps(extra: list[str]) -> None:
    """Changer t réutilise les entrées, puis une nouvelle paire ou un TTA les remplace."""
    data = _panning(48)
    common = ("--model", "rife-v4.15-mvo1", "--backend", "tensorrt", "--trt-plugin", TRT_PLUGIN,
              "--scene-threshold", "0", "--matrix", "bt709", "--nvof", "off", *extra)
    def run_factor(factor: str) -> subprocess.CompletedProcess:
        return subprocess.run([RIFE_BIN, "--quiet", "-g", str(_trt_gpu()), *common, "--factor", factor],
                              input=data, capture_output=True, timeout=600, check=False)
    twice, four = run_factor("2"), run_factor("4")
    assert twice.returncode == four.returncode == 0, (twice.stderr + four.stderr).decode()
    out2, out4 = parse_y4m(twice.stdout), parse_y4m(four.stdout)
    assert out4.frames[0::4] == parse_y4m(data).frames
    assert out4.frames[2::4] == out2.frames[1::2]
