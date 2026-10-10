"""Vérifie les images Vulkan du filtre privé, en lecture bornée sans vidéo sur disque."""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
import threading
from pathlib import Path


def escaped(value: str) -> str:
    value = ''.join('\\'+c if c in "\\':" else c for c in value)
    return ''.join('\\'+c if c in "\\'[],;" else c for c in value)


def validate_direct(binary: Path, library: Path, clip: Path, uuid: str,
                    frame_bytes: int, expected: list[str]) -> dict:
    if not 0 < frame_bytes <= 8192*8192*6 or len(expected) != 3:
        raise ValueError('Dimensions ou nombre d’images invalides')
    uuid = f'{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}'
    graph = (f'mvo_fel=source={escaped(str(clip.resolve()))}:library={escaped(str(library.resolve()))}:threads=6,'
             'hwdownload,format=gbrp16le[v]')
    hashes = []
    expired = threading.Event()
    with tempfile.TemporaryFile() as diagnostics, subprocess.Popen([
        str(binary.resolve()), '-v', 'error', '-nostdin', '-init_hw_device',
        f'vulkan=fel:{uuid},disable_multiplane=1', '-filter_hw_device', 'fel',
        '-filter_complex', graph, '-map', '[v]', '-frames:v', '17', '-fps_mode', 'passthrough',
        '-c:v', 'rawvideo', '-threads', '1', '-pix_fmt', 'gbrp16le', '-f', 'rawvideo', 'pipe:1',
    ], stdout=subprocess.PIPE, stderr=diagnostics) as process:
        assert process.stdout is not None

        def timeout() -> None:
            expired.set()
            process.kill()

        timer = threading.Timer(120, timeout)
        timer.start()
        try:
            for index in range(17):
                remaining = frame_bytes
                sha = hashlib.sha256()
                while remaining:
                    chunk = process.stdout.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise RuntimeError(f'Image directe {index} tronquée')
                    if index in (0, 8, 16):
                        sha.update(chunk)
                    remaining -= len(chunk)
                if index in (0, 8, 16):
                    hashes.append(sha.hexdigest())
            code = process.wait(timeout=15)
        finally:
            timer.cancel()
            timer.join()
            if process.poll() is None:
                process.kill()
            process.wait()
        if expired.is_set() or code or hashes != expected:
            diagnostics.seek(0)
            raise RuntimeError('Pixels du filtre direct différents de la référence : '
                               +diagnostics.read().decode('utf-8', errors='replace'))
    return {'frame_sha256': hashes, 'transport': 'vulkan', 'frames_read': 17}
