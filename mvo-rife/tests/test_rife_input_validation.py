"""Entrées et options invalides refusées par le binaire, sans GPU (1.2.3+)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import pytest

RIFE_BIN = os.environ.get("MUXIVEO_RIFE_BIN") or shutil.which("muxiveo-rife") or ""
# Parsing strict des en-têtes et options introduit en 1.2.3.
_STRICT_MIN_VERSION = (1, 2, 3)


def _rife_version() -> tuple[int, ...] | None:
    if not RIFE_BIN:
        return None
    try:
        out = subprocess.run([RIFE_BIN, "--version"], capture_output=True, text=True, timeout=15, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out or "")
    return tuple(int(part) for part in match.groups()) if match else None


RIFE_VERSION = _rife_version()
pytestmark = pytest.mark.skipif(
    RIFE_VERSION is None or RIFE_VERSION < _STRICT_MIN_VERSION,
    reason="Binaire muxiveo-rife ≥ 1.2.3 requis (MUXIVEO_RIFE_BIN ou PATH)",
)


def _vulkan_loader_available() -> bool:
    """Vrai si le binaire voit au moins un périphérique Vulkan.

    `--list-gpus` réussit même sans chargeur (liste vide) : seul le contenu fait foi.
    """
    if not RIFE_BIN:
        return False
    probe = subprocess.run([RIFE_BIN, "--list-gpus"], capture_output=True, text=True, timeout=60, check=False)
    if probe.returncode != 0:
        return False
    try:
        return bool(json.loads(probe.stdout or "{}").get("gpus"))
    except ValueError:
        return False


# Le binaire vérifie le chargeur Vulkan avant de lire le flux : les refus
# d'en-tête et de cadence (avant tout calcul GPU) exigent un chargeur.
needs_vulkan_loader = pytest.mark.skipif(
    not _vulkan_loader_available(), reason="Chargeur Vulkan requis (llvmpipe/lavapipe accepté)",
)


def _run(args: list[str], stdin: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.run([RIFE_BIN, *args], input=stdin, capture_output=True, timeout=60, check=False)


@needs_vulkan_loader
@pytest.mark.parametrize("header", [
    b"YUV4MPEG2 W2147483647 H2 F25:1 Ip C420jpeg\n",
    b"YUV4MPEG2 W64 H32 F25:1x\n",
    b"YUV4MPEG2 W32768 H16384 F25:1\n",
])
def test_invalid_header_refused_before_gpu(header: bytes) -> None:
    result = _run(["--factor", "2"], stdin=header + b"FRAME\n")
    assert result.returncode == 2, result.stderr
    assert result.stdout == b""


@pytest.mark.parametrize("args", [
    ["--gpu", "1x"], ["--gpu", "99999999999"], ["--threads", "-1"], ["--padding", "abc"],
    ["--tta", "2x"], ["--tta", "3"], ["--scene-threshold", "nan"], ["--progress-interval", "inf"],
])
def test_invalid_numeric_options_refused(args: list[str]) -> None:
    result = _run(args)
    assert result.returncode == 1
    assert b"error:" in result.stderr


@needs_vulkan_loader
@pytest.mark.parametrize("args", [
    ["--factor", "2x"], ["--factor", "nan"], ["--factor", "0"], ["--fps", "60/0"],
    ["--fps", "9999999999999/1"],
])
def test_invalid_rate_refused_before_gpu(args: list[str]) -> None:
    result = _run(args, stdin=b"YUV4MPEG2 W64 H32 F25:1 Ip C420jpeg\nFRAME\n" + bytes(64 * 32 * 3 // 2))
    assert result.returncode == 1, result.stderr


@needs_vulkan_loader
@pytest.mark.parametrize("args", [
    ["--gpu", "256"], ["--threads", "1025"], ["--padding", "4097"],
    ["--scene-threshold", "1e10"], ["--progress-interval", "86401"],
])
def test_finite_numeric_options_have_no_arbitrary_caps(args: list[str]) -> None:
    frame = b"FRAME\n" + bytes(64 * 32 * 3 // 2)
    result = _run(["--factor", "1", *args], stdin=b"YUV4MPEG2 W64 H32 F25:1 Ip C420jpeg\n" + frame)
    assert result.returncode == 0, result.stderr
