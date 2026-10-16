"""Compare en pipe les pixels du NUT livré aux empreintes du renderer de référence."""
from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path


def validate_producer(lib: ctypes.CDLL, clip: Path, uuid: str, frame_bytes: int, expected: list[str],
                      precision: str = "fast", workers: int | None = None) -> tuple[dict, list[bytes]]:
    """Aucun intermédiaire vidéo ; FFmpeg sélectionne les images 0, 8 et 16 (rendues aussi brutes).
    Le résultat porte aussi les compteurs du rendu certifié (reprises, indécises)."""
    if not 0 < frame_bytes <= 8192*8192*6 or len(expected) != 3:
        raise ValueError("Dimensions ou nombre d'images de validation invalides")
    statistics: dict = {}
    backend, frames = producer_frames(lib, clip, uuid, frame_bytes, "gbrp16le",
                                      precision_setter(lib, precision, workers), statistics)
    hashes = [hashlib.sha256(frame).hexdigest() for frame in frames]
    if hashes != expected:
        raise RuntimeError(f"Pixels de la bibliothèque différents de la référence : moteur={backend!r}, "
                           f"empreintes={[h[:16] for h in hashes]}")
    return {"backend": backend, "frame_sha256": hashes, **certified_counters(statistics)}, frames


def certified_counters(statistics: dict) -> dict:
    """Composantes reprises et indécises du rendu certifié, lues dans les statistiques du producteur."""
    counters = {key: statistics.get(f"certified_{key}") for key in ("rescued", "undecided")}
    if not all(isinstance(value, int) and value >= 0 for value in counters.values()):
        raise RuntimeError(f"Compteurs du rendu certifié absents : {statistics}")
    return {f"certified_{key}": value for key, value in counters.items()}


def precision_setter(lib: ctypes.CDLL, precision: str, workers: int | None = None) -> Callable[[int], None]:
    """Réglage de précision (et du nombre d'images en vol) appliqué au contexte avant le rendu."""
    lib.mvo_fel_set_precision.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.mvo_fel_set_precision.restype = ctypes.c_int
    if workers is not None:
        lib.mvo_fel_set_render_workers.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.mvo_fel_set_render_workers.restype = ctypes.c_int

    def configure(context: int) -> None:
        if lib.mvo_fel_set_precision(context, precision.encode("ascii")):
            raise RuntimeError(f"Précision refusée : {precision}")
        if workers is not None and lib.mvo_fel_set_render_workers(context, workers):
            raise RuntimeError(f"Nombre d'images en vol refusé : {workers}")
    return configure


def producer_frames(lib: ctypes.CDLL, clip: Path, uuid: str, frame_bytes: int, pixel_format: str,
                    configure: Callable[[int], None] | None = None,
                    statistics: dict | None = None) -> tuple[str, list[bytes]]:
    """Images 0, 8 et 16 du NUT livré, brutes ; ``configure`` règle le contexte avant le rendu,
    ``statistics`` reçoit les statistiques du producteur après le rendu."""
    if not 0 < frame_bytes <= 8192*8192*6:
        raise ValueError("Dimensions de validation invalides")
    write_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    progress_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
    lib.mvo_fel_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mvo_fel_create.restype = ctypes.c_void_p
    lib.mvo_fel_set_device.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.mvo_fel_set_device.restype = ctypes.c_int
    lib.mvo_fel_run.argtypes = [ctypes.c_void_p, write_type, progress_type, ctypes.c_void_p]
    lib.mvo_fel_run.restype = ctypes.c_int
    lib.mvo_fel_cancel.argtypes = [ctypes.c_void_p]
    lib.mvo_fel_cancel.restype = None
    lib.mvo_fel_destroy.argtypes = [ctypes.c_void_p]
    lib.mvo_fel_destroy.restype = None
    lib.mvo_fel_backend.argtypes = [ctypes.c_void_p]
    lib.mvo_fel_backend.restype = ctypes.c_char_p
    context = lib.mvo_fel_create(str(clip.resolve()).encode("utf-8"), 0, 6, 0, 1)
    if not context:
        raise RuntimeError("Création du producteur de validation impossible")
    try:
        if lib.mvo_fel_set_device(context, uuid.encode("ascii")):
            raise RuntimeError("Moteur de validation refusé")
        if configure is not None:
            configure(context)
        frames: list[bytes] = []
        errors: list[str] = []
        expired = threading.Event()
        with tempfile.TemporaryFile() as diagnostics, subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-f", "nut", "-i", "pipe:0",
            "-map", "0:v:0", "-vf", r"select=eq(n\,0)+eq(n\,8)+eq(n\,16)",
            "-fps_mode", "passthrough", "-frames:v", "3", "-c:v", "rawvideo", "-threads", "1",
            "-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics) as process:
            assert process.stdin is not None and process.stdout is not None

            def read() -> None:
                try:
                    while True:
                        frame = bytearray()
                        while len(frame) < frame_bytes:
                            chunk = process.stdout.read(min(frame_bytes - len(frame), 1 << 20))
                            if not chunk:
                                if frame:
                                    errors.append("Image NUT tronquée")
                                return
                            frame += chunk
                        frames.append(bytes(frame))
                except Exception as exc:
                    errors.append(str(exc))

            def write(_opaque: object, data: int, size: int) -> int:
                try:
                    process.stdin.write(ctypes.string_at(data, size))
                    return 0
                except (OSError, ValueError):
                    return 1

            def timeout() -> None:
                expired.set()
                lib.mvo_fel_cancel(context)
                process.kill()

            reader = threading.Thread(target=read, daemon=True)
            timer = threading.Timer(120, timeout)
            reader.start()
            timer.start()
            try:
                status = lib.mvo_fel_run(context, write_type(write), progress_type(lambda *_: None), None)
                try:
                    process.stdin.close()
                except OSError:
                    # Lecteur déjà sorti : EPIPE sous POSIX, EINVAL sous Windows.
                    pass
                code = process.wait(timeout=15)
                reader.join(timeout=15)
                backend = (lib.mvo_fel_backend(context) or b"").decode("utf-8", errors="replace")
                if statistics is not None:
                    lib.mvo_fel_statistics.argtypes = [ctypes.c_void_p]
                    lib.mvo_fel_statistics.restype = ctypes.c_char_p
                    statistics.update(json.loads(lib.mvo_fel_statistics(context) or b"{}"))
            finally:
                timer.cancel()
                timer.join()
                if process.poll() is None:
                    process.kill()
                process.wait()
                reader.join()
            vulkan = uuid != "cpu"
            if (expired.is_set() or code or status not in {0, 2} or errors or len(frames) != 3
                    or backend.startswith("vulkan:") != vulkan):
                diagnostics.seek(0)
                detail = diagnostics.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Rendu de validation incomplet : statut={status}, ffmpeg={code}, "
                                   f"images={len(frames)}, moteur={backend!r}, erreurs={errors}, {detail}")
            return backend, frames
    finally:
        lib.mvo_fel_destroy(context)
