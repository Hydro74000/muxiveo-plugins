"""Positionnement et flux coupé en milieu de GOP EL : images identiques au décodage complet de même rang.

Mire ``synthetic-fel-shifted-irap.mkv`` (``make_seek_fixture.py``) : IRAP BL et EL décalées, RASL dans les
deux couches. Chaque positionnement doit livrer, au rang demandé, exactement l'image du décodage complet
depuis le début (EL commencée à sa propre IRAP, au besoin depuis une image clé BL antérieure). L'extrait
``…-cut.mkv`` commence à une CRA BL au milieu d'un GOP EL : avant la première IRAP EL, l'image est celle de
la couche « mapped » (RPU appliqué, sans résiduel) et ces images sont comptées ; ensuite, l'image
reconstruite.
"""
from __future__ import annotations

import ctypes as c
import json
import sys
from pathlib import Path

from smoke import load

FIXTURES = Path(__file__).parent / "fixtures"
FRAME_US = 1001 * 1000000 / 24000


def session(lib, path: Path):
    handle = lib.mvo_fel_create(str(path.resolve()).encode(), 0, 4, 24000, 1001)
    assert handle
    assert lib.mvo_fel_set_output(handle, b"yuv420p10le", b"topleft", b"none") == 0
    assert lib.mvo_fel_set_device(handle, b"cpu") == 0
    assert lib.mvo_fel_set_precision(handle, b"exact") == 0
    size = lib.mvo_fel_capture_frame_size(handle)
    assert size, lib.mvo_fel_error(handle)
    return handle, size


def capture(lib, handle, size: int, start_us: int, count: int) -> list[tuple[bytes, bytes]]:
    """Images (reconstruite, mapped) de ``count`` rangs à partir de ``start_us``."""
    buffer = c.create_string_buffer(2 * count * size)
    assert lib.mvo_fel_capture_pair(handle, start_us, count, buffer, 2 * count * size) == 0, lib.mvo_fel_error(handle)
    raw = buffer.raw
    return [(raw[2 * k * size:(2 * k + 1) * size], raw[(2 * k + 1) * size:(2 * k + 2) * size]) for k in range(count)]


def main() -> None:
    lib = load(Path(sys.argv[1]))
    lib.mvo_fel_capture_frame_size.argtypes = [c.c_void_p]
    lib.mvo_fel_capture_frame_size.restype = c.c_size_t
    lib.mvo_fel_capture_pair.argtypes = [c.c_void_p, c.c_int64, c.c_int, c.c_void_p, c.c_size_t]
    lib.mvo_fel_capture_pair.restype = c.c_int
    lib.mvo_fel_statistics.argtypes = [c.c_void_p]
    lib.mvo_fel_statistics.restype = c.c_char_p

    handle, size = session(lib, FIXTURES / "synthetic-fel-shifted-irap.mkv")
    try:
        reference = capture(lib, handle, size, 0, 48)
        # Mire utile : une image par rang, et le résiduel FEL visible à chaque rang.
        assert len({frame for frame, _ in reference}) == 48
        assert all(frame != mapped for frame, mapped in reference)
        for rank in range(1, 47):
            # Instant entre les images rank - 1 et rank : première image gardée = rank.
            got = capture(lib, handle, size, round((rank - 0.5) * FRAME_US), 2)
            assert got == reference[rank:rank + 2], rank
        # Retour en arrière puis même position : mêmes images.
        assert capture(lib, handle, size, round(29.5 * FRAME_US), 2) == reference[30:32]
        assert capture(lib, handle, size, round(16.5 * FRAME_US), 3) == reference[17:20]
    finally:
        lib.mvo_fel_destroy(handle)

    handle, size = session(lib, FIXTURES / "synthetic-fel-shifted-irap-cut.mkv")
    try:
        frames = capture(lib, handle, size, 0, 32)  # CRA BL de l'image 16 : rangs 16 à 47
        unavailable = json.loads(lib.mvo_fel_statistics(handle))["el_unavailable_frames"]
    finally:
        lib.mvo_fel_destroy(handle)
    # Première IRAP EL deux unités d'accès après la CRA BL, trois RASL EL contre une RASL BL : 2 + 3 - 1.
    assert unavailable == 4, unavailable
    assert [frame for frame, _ in frames[:4]] == [mapped for _, mapped in reference[16:20]]
    assert [frame for frame, _ in frames[4:]] == [frame for frame, _ in reference[20:48]]
    print("Positionnements (46 rangs, retours en arrière) identiques au décodage complet ; extrait coupé en "
          "milieu de GOP EL : 4 images sans EL (mapped) puis images reconstruites : OK")


if __name__ == "__main__":
    main()
