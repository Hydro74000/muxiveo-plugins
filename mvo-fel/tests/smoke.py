"""Vérifie l'ABI et une reconstruction synthétique sur la bibliothèque livrée."""
from __future__ import annotations

import argparse
import ctypes as c
import json
from pathlib import Path

WRITE = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_size_t)
PROGRESS = c.CFUNCTYPE(None, c.c_void_p, c.c_int64)


def load(path: Path):
    lib = c.CDLL(str(path.resolve()))
    lib.mvo_fel_abi_version.restype = c.c_uint32
    assert lib.mvo_fel_abi_version() == 1
    lib.mvo_fel_capabilities.restype = c.c_char_p
    assert json.loads(lib.mvo_fel_capabilities())["pixel_format"] == "gbrp16le"
    lib.mvo_fel_create.argtypes = [c.c_char_p, c.c_int, c.c_int, c.c_int, c.c_int]
    lib.mvo_fel_create.restype = c.c_void_p
    lib.mvo_fel_run.argtypes = [c.c_void_p, WRITE, PROGRESS, c.c_void_p]
    lib.mvo_fel_run.restype = c.c_int
    lib.mvo_fel_error.argtypes = [c.c_void_p]
    lib.mvo_fel_error.restype = c.c_char_p
    for name in ("mvo_fel_cancel", "mvo_fel_destroy"):
        getattr(lib, name).argtypes = [c.c_void_p]
        getattr(lib, name).restype = None
    return lib


def smoke(library: Path, fixture: Path, output: Path | None = None, device: str = "auto") -> None:
    lib = load(library)
    for mode, expected in (("normal", 0), ("closed", 2), ("cancel", 3), ("bad_stream", 1), ("missing_source", 4)):
        path = fixture.with_name("missing-source.hevc") if mode == "missing_source" else fixture
        handle = lib.mvo_fel_create(str(path.resolve()).encode(), 999 if mode == "bad_stream" else 0,
                                    2, 24000, 1001)
        assert handle
        if device != "auto":
            lib.mvo_fel_set_device.argtypes = [c.c_void_p, c.c_char_p]
            lib.mvo_fel_set_device.restype = c.c_int
            assert lib.mvo_fel_set_device(handle, device.encode("ascii")) == 0
        received, frames = 0, 0
        header = bytearray()
        sink = output.open("wb") if output and mode == "normal" else None

        def write(_opaque, pointer, size):
            nonlocal received
            if mode == "closed":
                return 1
            data = c.string_at(pointer, size)
            if len(header) < 32:
                header.extend(data[:32-len(header)])
            received += size
            if sink:
                sink.write(data)
            return 0

        def progress(_opaque, count):
            nonlocal frames
            frames = count
            if mode == "cancel":
                lib.mvo_fel_cancel(handle)

        try:
            result = lib.mvo_fel_run(handle, WRITE(write), PROGRESS(progress), None)
            assert result == expected, (mode, result, lib.mvo_fel_error(handle))
            if mode == "normal":
                assert frames == 12, frames
                assert header.startswith(b"nut/multimedia container\x00"), header
                assert received > 12*256*144*6, received
        finally:
            lib.mvo_fel_destroy(handle)
            if sink:
                sink.close()
    print("ABI, NUT, 12 images/B-frames, fermeture aval, annulation, piste invalide et erreur de lecture : OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixtures/synthetic-fel.hevc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smoke(args.library, args.fixture, args.output, args.device)
