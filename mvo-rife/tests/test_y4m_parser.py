"""Lecteur y4m natif compilé seul (sans ncnn ni Vulkan) sous UBSan/ASan."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CXX = shutil.which("g++") or shutil.which("clang++")
pytestmark = pytest.mark.skipif(CXX is None, reason="Compilateur C++ requis")

DRIVER = r"""
#include <cstdio>
#include <string>
#include "y4m.h"

int main()
{
    Y4mReader reader(stdin);
    std::string error;
    if (!reader.read_header(error))
    {
        printf("error: %s\n", error.c_str());
        return 2;
    }
    const FrameFormat& f = reader.format();
    printf("ok %d %d %lld %lld %d %d %d %lld\n", f.width, f.height, (long long)f.fps_num,
           (long long)f.fps_den, f.chroma_width(), f.chroma_height(), f.bit_depth,
           (long long)f.frame_bytes());
    return 0;
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory) -> Path:
    build = tmp_path_factory.mktemp("y4m-driver")
    source = build / "driver.cpp"
    source.write_text(DRIVER, encoding="utf-8")
    binary = build / "driver"
    flags = ["-fsanitize=undefined,address", "-fno-sanitize-recover=undefined"]
    command = [CXX, "-std=c++17", "-O1", "-g", *flags, "-I", str(SRC), str(SRC / "y4m.cpp"), str(source), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    if result.returncode != 0:
        # MinGW : bibliothèques libasan/libubsan absentes (échec au lien).
        if "sanitize" in result.stderr or "-lasan" in result.stderr or "-lubsan" in result.stderr:
            pytest.skip(f"Sanitizers indisponibles : {result.stderr[-300:]}")
        pytest.fail(result.stderr)
    return binary


def _parse(driver: Path, header: str) -> tuple[int, str]:
    result = subprocess.run(
        [str(driver)], input=(header + "\n").encode("ascii"), capture_output=True, timeout=30, check=False,
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    # Toute erreur d'UBSan/ASan fait échouer le cas, même sur un refus attendu.
    assert "runtime error" not in stderr and "AddressSanitizer" not in stderr, stderr
    return result.returncode, result.stdout.decode("utf-8", errors="replace").strip()


@pytest.mark.parametrize(("header", "expected"), [
    ("YUV4MPEG2 W1920 H1080 F24000:1001 Ip C420jpeg", "ok 1920 1080 24000 1001 960 540 8 3110400"),
    ("YUV4MPEG2 W3840 H2160 F25:1 Ip C420p10", "ok 3840 2160 25 1 1920 1080 10 24883200"),
    ("YUV4MPEG2 W1919 H1079 F30:1 C422p10", "ok 1919 1079 30 1 960 1079 10 8284562"),
    ("YUV4MPEG2 W7680 H4320 F60:1 C444p16", "ok 7680 4320 60 1 7680 4320 16 199065600"),
    ("YUV4MPEG2 W16384 H16384 F1:1", "ok 16384 16384 1 1 8192 8192 8 402653184"),
])
def test_valid_headers(driver: Path, header: str, expected: str) -> None:
    assert _parse(driver, header) == (0, expected)


@pytest.mark.parametrize(("header", "message"), [
    ("YUV4MPEG2 W2147483647 H2 F25:1 Ip C420jpeg", "largeur"),  # reproduction de l'audit (UBSan)
    ("YUV4MPEG2 W0 H2 F25:1", "largeur"),
    ("YUV4MPEG2 W-4 H2 F25:1", "largeur"),
    ("YUV4MPEG2 W4x H2 F25:1", "largeur"),
    ("YUV4MPEG2 W99999999999999999999 H2 F25:1", "largeur"),
    ("YUV4MPEG2 W32769 H2 F25:1", "largeur"),
    ("YUV4MPEG2 W64 H H 32", "hauteur"),
    ("YUV4MPEG2 W32768 H16384 F25:1", "trop grandes"),
    ("YUV4MPEG2 W4 H2 F25", "cadence"),
    ("YUV4MPEG2 W4 H2 F0:1", "cadence"),
    ("YUV4MPEG2 W4 H2 F25:0", "cadence"),
    ("YUV4MPEG2 W4 H2 F25:1x", "cadence"),
    ("YUV4MPEG2 W4 H2 F9223372036854775807:1", "cadence"),
    ("YUV4MPEG2 W4 H2 Fnan:1", "cadence"),
    ("YUV4MPEG2 W4 H2 F25:1 C420p99", "espace colorimétrique"),
    ("YUV4MPEG2 W4 H2 F25:1 C420p10x", "espace colorimétrique"),
    ("YUV4MPEG2 H2 F25:1", "dimensions"),
    ("YUV4MPEG2 W4 H2", "cadence"),
    ("MPEG2 W4 H2 F25:1", "signature"),
])
def test_invalid_headers_are_refused_without_undefined_behaviour(driver: Path, header: str, message: str) -> None:
    code, output = _parse(driver, header)
    assert code == 2
    assert output.startswith("error:") and message in output, output


def test_truncated_or_oversized_header_line(driver: Path) -> None:
    code, output = _parse(driver, "YUV4MPEG2 " + "X" * 5000)
    assert code == 2 and "illisible" in output
