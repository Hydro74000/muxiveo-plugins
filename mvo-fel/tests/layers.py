"""Le producteur pull conserve l'ordre B-frames, la cadence et les erreurs d'arrêt."""
import ctypes as c
import sys
from pathlib import Path
from smoke import load

CALLBACK = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p,
                      c.c_int64, c.c_int64, c.c_int, c.c_int)


def main() -> None:
    lib = load(Path(sys.argv[1]))
    lib.mvo_fel_next_layers.argtypes = [c.c_void_p, CALLBACK, c.c_void_p]
    lib.mvo_fel_next_layers.restype = c.c_int
    fixture = Path(__file__).parent/'fixtures/synthetic-fel.hevc'
    for mode in ('normal', 'closed', 'cancel', 'missing'):
        path = fixture if mode != 'missing' else fixture.with_name('absent.hevc')
        handle = lib.mvo_fel_create(str(path.resolve()).encode(), 0, 2, 24000, 1001)
        frames = []

        def receive(_opaque, bl, el, metadata, pts, duration, num, den):
            assert bl and el and metadata
            frames.append((pts, duration, num, den))
            if mode == 'cancel':
                lib.mvo_fel_cancel(handle)
            return int(mode == 'closed')

        callback = CALLBACK(receive)
        try:
            while True:
                before = len(frames)
                status = lib.mvo_fel_next_layers(handle, callback, None)
                if status:
                    break
                assert len(frames) == before+1
            assert status == {'normal': 5, 'closed': 2, 'cancel': 3, 'missing': 4}[mode]
            if mode == 'normal':
                assert frames == [(i, 1, 1001, 24000) for i in range(12)], frames
                assert lib.mvo_fel_next_layers(handle, callback, None) == 5
        finally:
            lib.mvo_fel_destroy(handle)
    print('Pull : douze images ordonnées, cadence rationnelle, EOF, fermeture, annulation et lecture : OK')


if __name__ == '__main__':
    main()
