"""Sorties Y'CbCr 4:2:0 du NUT livré face à une conversion de référence en double (numpy).

Le RGB du producteur est attesté par ailleurs (empreintes du renderer de référence). Ici, ses
images Y'CbCr 10 bits (matrice BT.2020nc, plage limitée, position chroma, recadrage, tramage)
sont comparées à la conversion, recalculée indépendamment, de ce même RGB.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from validate_producer import producer_frames

KR, KB = 0.2627, 0.0593
KG = 1 - KR - KB
# Centre du filtre chroma (coordonnées luma, relatif à 2i / 2j), selon la position.
OFFSETS = {"left": (0.0, 0.5), "center": (0.5, 0.5), "topleft": (0.0, 0.0), "top": (0.5, 0.0),
           "bottomleft": (0.0, 1.0), "bottom": (0.5, 1.0)}
# Cas contrôlés : position UHD Blu-ray recadrée sans tramage (arrondi exact), puis tramage livré.
CASES = (("topleft", "none", (2, 4, 6, 8)), ("left", "blue", (0, 0, 0, 0)))


def dimensions(clip: Path) -> tuple[int, int]:
    output = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height", "-of", "csv=p=0", str(clip)],
                            check=True, capture_output=True, text=True).stdout
    width, height = (int(value) for value in output.strip().split(",")[:2])
    return width, height


def taps(offset: float) -> tuple[int, list[float]]:
    """Triangle de demi-largeur 2 échantillons luma, centré sur l'échantillon chroma."""
    first = 0 if offset >= 1 else -1
    return first, [max(0.0, 1 - abs(first + k - offset)/2)/2 for k in range(4)]


def reference(rgb: bytes, width: int, height: int, chroma: str,
              crop: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codes 10 bits non quantifiés (Y, Cb, Cr) ; les prises chroma hors image sont bornées à l'image."""
    planes = np.frombuffer(rgb, dtype="<u2").reshape(3, height, width).astype(np.float64) / 65535
    g, b, r = planes
    left, top, right, bottom = crop
    w, h = width - left - right, height - top - bottom
    luma = KR*r + KG*g + KB*b
    y = (219*luma[top:top+h, left:left+w] + 16)*4
    ox, oy = OFFSETS[chroma]
    fx, wx = taps(ox)
    fy, wy = taps(oy)
    sums = [np.zeros((h//2, w//2)) for _ in range(3)]
    for a in range(4):
        xs = np.clip(left + 2*np.arange(w//2) + fx + a, 0, width-1)
        for c in range(4):
            weight = wx[a]*wy[c]
            if weight == 0:
                continue
            ys = np.clip(top + 2*np.arange(h//2) + fy + c, 0, height-1)
            for k, plane in enumerate((r, g, b)):
                sums[k] += weight*plane[np.ix_(ys, xs)]
    r_s, g_s, b_s = sums
    l_s = KR*r_s + KG*g_s + KB*b_s
    cb = (224*(b_s - l_s)/(2*(1 - KB)) + 128)*4
    cr = (224*(r_s - l_s)/(2*(1 - KR)) + 128)*4
    return y, cb, cr


def p010_to_planar(frame: bytes, width: int, height: int) -> bytes:
    """P010 (codes en poids fort, chroma entrelacée) vers Y'CbCr 4:2:0 10 bits planaire."""
    data = np.frombuffer(frame, dtype="<u2") >> 6
    chroma = data[width*height:].reshape(-1, 2)
    return np.concatenate([data[:width*height], chroma[:, 0], chroma[:, 1]]).astype("<u2").tobytes()


def direct_options(case: dict) -> str:
    """Options du filtre direct produisant la même sortie finale (P010)."""
    left, top, right, bottom = case["crop"]
    return (f":format=p010:chroma={case['chroma']}:dither={case['dither']}:crop_left={left}:crop_top={top}"
            f":crop_right={right}:crop_bottom={bottom}")


def validate_yuv(lib: ctypes.CDLL, clip: Path, uuid: str, rgb_frames: list[bytes]) -> list[dict]:
    """Écarts au code 10 bits : ≤ 0,5 sans tramage (arrondi), < 1 avec (seuil dans [0, 1))."""
    width, height = dimensions(clip)
    if len(rgb_frames) != 3 or any(len(frame) != width*height*6 for frame in rgb_frames):
        raise RuntimeError("Images RGB de référence absentes ou de taille inattendue")
    lib.mvo_fel_set_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.mvo_fel_set_output.restype = ctypes.c_int
    lib.mvo_fel_set_crop.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mvo_fel_set_crop.restype = ctypes.c_int
    results = []
    for chroma, dither, crop in CASES:
        def configure(context: int) -> None:
            if (lib.mvo_fel_set_output(context, b"yuv420p10le", chroma.encode(), dither.encode())
                    or lib.mvo_fel_set_crop(context, *crop)):
                raise RuntimeError(f"Sortie Y'CbCr refusée : {chroma}, {dither}, {crop}")
        w, h = width - crop[0] - crop[2], height - crop[1] - crop[3]
        backend, frames = producer_frames(lib, clip, uuid, w*h*3, "yuv420p10le", configure)
        worst = 0.0
        for rgb, frame in zip(rgb_frames, frames):
            data = np.frombuffer(frame, dtype="<u2").astype(np.float64)
            planes = (data[:w*h].reshape(h, w), data[w*h:w*h*5//4].reshape(h//2, w//2),
                      data[w*h*5//4:].reshape(h//2, w//2))
            for produced, expected in zip(planes, reference(rgb, width, height, chroma, crop)):
                worst = max(worst, float(np.abs(produced - np.clip(expected, 0, 1023)).max()))
        # Marge de 0,02 code : rendu GPU en float32.
        limit = 0.52 if dither == "none" else 1.02
        if worst > limit:
            raise RuntimeError(f"Sortie Y'CbCr hors tolérance ({chroma}, {dither}, recadrage {crop}) : "
                               f"{worst:.4f} code 10 bits > {limit} ({backend})")
        results.append({"chroma": chroma, "dither": dither, "crop": list(crop), "max_code10": round(worst, 4),
                        "width": w, "height": h,
                        "frame_sha256": [hashlib.sha256(frame).hexdigest() for frame in frames]})
    return results
