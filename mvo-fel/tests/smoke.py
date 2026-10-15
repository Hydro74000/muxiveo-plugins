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
    lib.mvo_fel_run_fd.argtypes = [c.c_void_p, c.c_ssize_t, PROGRESS, c.c_void_p]
    lib.mvo_fel_run_fd.restype = c.c_int
    lib.mvo_fel_set_output.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p, c.c_char_p]
    lib.mvo_fel_set_output.restype = c.c_int
    lib.mvo_fel_set_crop.argtypes = [c.c_void_p, *([c.c_int] * 4)]
    lib.mvo_fel_set_crop.restype = c.c_int
    lib.mvo_fel_set_timestamps.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_timestamps.restype = c.c_int
    lib.mvo_fel_set_start.argtypes = [c.c_void_p, c.c_int64]
    lib.mvo_fel_set_start.restype = c.c_int
    for name in ("mvo_fel_cancel", "mvo_fel_destroy"):
        getattr(lib, name).argtypes = [c.c_void_p]
        getattr(lib, name).restype = None
    return lib


def smoke(library: Path, fixture: Path, output: Path | None = None, device: str = "auto") -> None:
    lib = load(library)
    assert json.loads(lib.mvo_fel_capabilities()).get("seek") == 1
    for mode, expected in (("normal", 0), ("seek", 0), ("closed", 2), ("cancel", 3), ("bad_stream", 1),
                           ("missing_source", 4)):
        path = fixture.with_name("missing-source.hevc") if mode == "missing_source" else fixture
        handle = lib.mvo_fel_create(str(path.resolve()).encode(), 999 if mode == "bad_stream" else 0,
                                    2, 24000, 1001)
        assert handle
        if mode == "seek":
            # Cinq images à 24000/1001 : la sixième est la première produite.
            assert lib.mvo_fel_set_start(handle, 5*1001*1000000//24000) == 0
            assert lib.mvo_fel_set_start(handle, -1) == -1
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
            if mode == "seek":
                assert frames == 7, frames
                assert lib.mvo_fel_set_start(handle, 0) == -1
            if mode == "normal":
                assert frames == 12, frames
                assert header.startswith(b"nut/multimedia container\x00"), header
                assert received > 12*256*144*6, received
        finally:
            lib.mvo_fel_destroy(handle)
            if sink:
                sink.close()
    fd_output(lib, fixture)
    yuv_output(lib, fixture)
    print("ABI, NUT, 12 images/B-frames, seek, descripteur, Y'CbCr, fermeture aval, annulation, piste invalide et erreur de lecture : OK")


def yuv_output(lib, fixture: Path) -> None:
    """Sorties Y'CbCr 4:2:0 : taille des images, refus des valeurs inconnues ou tardives."""
    lib.mvo_fel_set_device.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_device.restype = c.c_int
    for (name, depth), device in ((formats, engine) for formats in (("yuv420p16le", 2), ("yuv420p10le", 2), ("yuv420p", 1))
                                  for engine in (b"cpu", b"auto")):
        handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
        assert lib.mvo_fel_set_device(handle, device) == 0
        assert lib.mvo_fel_set_output(handle, b"rgb48", b"topleft", b"blue") == -1
        assert lib.mvo_fel_set_output(handle, name.encode(), b"diagonal", b"blue") == -1
        assert lib.mvo_fel_set_output(handle, name.encode(), b"topleft", b"random") == -1
        assert lib.mvo_fel_set_output(handle, name.encode(), b"topleft", b"blue") == 0
        assert lib.mvo_fel_set_crop(handle, 1, 0, 0, 0) == -1
        assert lib.mvo_fel_set_crop(handle, 0, 8, 2, 8) == 0
        received = bytearray()
        result = lib.mvo_fel_run(handle, WRITE(lambda _o, p, n: received.extend(c.string_at(p, n)) or 0),
                                 PROGRESS(lambda *_: None), None)
        assert result == 0, (name, lib.mvo_fel_error(handle))
        assert lib.mvo_fel_set_output(handle, name.encode(), b"topleft", b"blue") == -1
        lib.mvo_fel_destroy(handle)
        frame = 254*128*3//2*depth
        assert 12*frame < len(received) < 12*frame + 4096, (name, len(received))


def fd_output(lib, fixture: Path) -> None:
    """Sortie directe sur pipe : même flux que le callback, puis aval fermé sans SIGPIPE."""
    import os
    import threading
    expected = bytearray()
    handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
    assert lib.mvo_fel_run(handle, WRITE(lambda _o, p, n: expected.extend(c.string_at(p, n)) or 0),
                           PROGRESS(lambda *_: None), None) == 0
    lib.mvo_fel_destroy(handle)
    for close_early in (False, True):
        read, write = os.pipe()
        received = bytearray()

        def consume():
            with os.fdopen(read, "rb") as stream:
                if close_early:
                    stream.read(4096)
                    return
                while chunk := stream.read(1 << 20):
                    received.extend(chunk)

        reader = threading.Thread(target=consume)
        reader.start()
        handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
        if os.name == "nt":
            import msvcrt
            target = msvcrt.get_osfhandle(write)
        else:
            target = write
        result = lib.mvo_fel_run_fd(handle, target, PROGRESS(lambda *_: None), None)
        os.close(write)
        reader.join()
        lib.mvo_fel_destroy(handle)
        if close_early:
            assert result == 2, result
        else:
            assert result == 0 and bytes(received) == bytes(expected), (result, len(received), len(expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixtures/synthetic-fel.hevc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smoke(args.library, args.fixture, args.output, args.device)
