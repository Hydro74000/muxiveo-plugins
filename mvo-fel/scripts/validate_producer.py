"""Compare en pipe les pixels du NUT livré aux empreintes du renderer de référence."""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
import tempfile
import threading
from pathlib import Path


def validate_producer(lib: ctypes.CDLL, clip: Path, uuid: str, frame_bytes: int,
                      expected: list[str]) -> dict:
    """Aucun intermédiaire vidéo ; FFmpeg sélectionne les images 0, 8 et 16."""
    if not 0 < frame_bytes <= 8192*8192*6 or len(expected) != 3:
        raise ValueError("Dimensions ou nombre d'images de validation invalides")
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
            raise RuntimeError("GPU de validation refusé")
        hashes: list[str] = []
        errors: list[str] = []
        expired = threading.Event()
        with tempfile.TemporaryFile() as diagnostics, subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-f", "nut", "-i", "pipe:0",
            "-map", "0:v:0", "-vf", r"select=eq(n\,0)+eq(n\,8)+eq(n\,16)",
            "-fps_mode", "passthrough", "-frames:v", "3", "-c:v", "rawvideo", "-threads", "1",
            "-pix_fmt", "gbrp16le", "-f", "rawvideo", "pipe:1",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics) as process:
            assert process.stdin is not None and process.stdout is not None

            def read() -> None:
                try:
                    while True:
                        remaining = frame_bytes
                        sha = hashlib.sha256()
                        while remaining:
                            chunk = process.stdout.read(min(remaining, 1 << 20))
                            if not chunk:
                                if remaining != frame_bytes:
                                    errors.append("Image NUT tronquée")
                                return
                            sha.update(chunk)
                            remaining -= len(chunk)
                        hashes.append(sha.hexdigest())
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
                except BrokenPipeError:
                    pass
                code = process.wait(timeout=15)
                reader.join(timeout=15)
                backend = (lib.mvo_fel_backend(context) or b"").decode("utf-8", errors="replace")
            finally:
                timer.cancel()
                timer.join()
                if process.poll() is None:
                    process.kill()
                process.wait()
                reader.join()
            if expired.is_set() or code or status not in {0, 2} or errors or hashes != expected or not backend.startswith("vulkan:"):
                diagnostics.seek(0)
                detail = diagnostics.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Pixels de la bibliothèque différents de la référence : statut={status}, "
                                   f"ffmpeg={code}, images={len(hashes)}, erreurs={errors}, {detail}")
            return {"backend": backend, "frame_sha256": hashes}
    finally:
        lib.mvo_fel_destroy(context)
