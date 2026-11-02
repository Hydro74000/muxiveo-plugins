"""Capacités (``--capabilities``) et fichier de poids du sélecteur, sans GPU.

Binaire : ``MUXIVEO_RIFE_BIN`` ou ``muxiveo-rife`` dans le PATH ; ignorés sinon.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RIFE_BIN = os.environ.get("MUXIVEO_RIFE_BIN") or shutil.which("muxiveo-rife") or ""
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "selector" / "selector.txt"
pytestmark = pytest.mark.skipif(not RIFE_BIN or not Path(RIFE_BIN).is_file(), reason="muxiveo-rife requis")


def _capabilities(*args: str) -> dict:
    out = subprocess.run([RIFE_BIN, "--capabilities", *args], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def _version() -> str:
    out = subprocess.run([RIFE_BIN, "--version"], capture_output=True, text=True, timeout=60, check=True)
    return out.stdout.split()[1]


def test_capabilities_describe_contract_and_options():
    caps = _capabilities()
    assert caps["name"] == "muxiveo-rife" and caps["version"] == _version()
    assert caps["contract"] == 1 and caps["trt_abi"] == 1
    assert {"engine", "ultra", "tta", "uhd", "selector-weights", "large-motion", "trt-plugin",
            "feature-cache", "feature-cache-reserve"} <= set(caps["options"])
    assert caps["engines"] == ["rife", "hybrid", "mc"] and caps["tta"] == [1, 2, 4, 8]
    models_dir = Path(RIFE_BIN).resolve().parent / "rife-models"
    expected = sorted(p.name for p in models_dir.glob("*") if (p / "flownet.param").is_file() and (p / "flownet.bin").is_file())
    assert caps["models"] == expected


def test_selector_weights_file_is_read():
    caps = _capabilities("--selector-weights", str(WEIGHTS))
    assert caps["selector"]["status"] == "ok" and caps["selector"]["format"] == 1
    assert {"v415", "v415mvo", "v46"} <= set(caps["selector"]["families"])


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda text: "", "vide"),
    (lambda text: text.replace("muxiveo-rife-selector 1", "muxiveo-rife-selector 2", 1), "format 2"),
    (lambda text: text.replace("model v415mvo 5 0 0 7 2", "model v415mvo 5 0 0 8 2", 1), "nfeat"),
    (lambda text: text.replace("mu -5.8423319", "mu nan", 1), "mu"),
    (lambda text: text.rsplit("bias", 1)[0], "incomplet"),
])
def test_invalid_selector_weights_are_reported(tmp_path, mutate, message):
    bad = tmp_path / "selector.txt"
    bad.write_text(mutate(WEIGHTS.read_text(encoding="utf-8")), encoding="utf-8")
    selector = _capabilities("--selector-weights", str(bad))["selector"]
    assert message in selector["status"] and selector["families"] == []


def test_missing_selector_weights_are_reported(tmp_path):
    selector = _capabilities("--selector-weights", str(tmp_path / "absent.txt"))["selector"]
    assert selector["status"] == "fichier introuvable"


def _has_restore() -> bool:
    return "restore" in _capabilities().get("options", [])


@pytest.mark.skipif(not RIFE_BIN or not Path(RIFE_BIN).is_file() or not _has_restore(), reason="Restauration absente du binaire")
def test_capabilities_describe_restore():
    caps = _capabilities()
    assert {"restore", "restore-animation", "restore-fast", "restore-model", "restore-dither", "transfer",
            "restore-tile", "frame-statistics", "source-statistics"} <= set(caps["options"])
    restore = caps["restore"]
    assert restore["levels"] == ["auto", "leger", "moyen", "fort"] and restore["dither"] == ["auto", "off", "8"]
    models_dir = Path(RIFE_BIN).resolve().parent / "restore-models"
    files = ("r2_flow.ncnn.param", "r2_flow.ncnn.bin", "r2_body.ncnn.param", "r2_body.ncnn.bin")
    expected = sorted(p.name for p in models_dir.glob("*") if all((p / f).is_file() for f in files))
    assert restore["models"] == expected


@pytest.mark.skipif(not RIFE_BIN or not Path(RIFE_BIN).is_file() or not _has_restore(), reason="Restauration absente du binaire")
@pytest.mark.parametrize("args,message", [
    (["--restore", "maximal"], "--restore : auto, leger, moyen ou fort"),
    (["--restore", "auto", "--restore-dither", "4"], "--restore-dither : auto, off ou 8"),
    (["--restore", "auto", "--transfer", "linear"], "--transfer : sdr, pq ou hlg"),
    (["--restore", "auto", "--restore-tile", "16"], "--restore-tile"),
])
def test_restore_options_rejected(args, message):
    out = subprocess.run([RIFE_BIN, *args], capture_output=True, text=True, timeout=60, input="")
    assert out.returncode == 1 and message in out.stderr


@pytest.mark.skipif(not RIFE_BIN or not Path(RIFE_BIN).is_file() or not _has_restore(), reason="Restauration absente du binaire")
def test_restore_without_graphs_fails_cleanly(tmp_path):
    gpus = subprocess.run([RIFE_BIN, "--list-gpus"], capture_output=True, text=True, timeout=60)
    if gpus.returncode != 0 or not json.loads(gpus.stdout or "{}").get("gpus"):
        pytest.skip("aucun périphérique Vulkan")
    header = b"YUV4MPEG2 W64 H64 F24:1 Ip A1:1 C420p10\n"
    frame = b"FRAME\n" + bytes(64 * 64 * 3 // 2 * 2)
    out = subprocess.run([RIFE_BIN, "--factor", "1", "--matrix", "bt709", "--restore", "auto", "--restore-model",
                          str(tmp_path / "absent")], capture_output=True, input=header + frame, timeout=120)
    assert out.returncode == 3 and "graphe de restauration introuvable" in out.stderr.decode()
