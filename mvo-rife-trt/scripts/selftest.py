"""Auto-test local du plugin mvo-rife-trt (machine NVIDIA requise).

Pilote le plugin comme muxiveo-rife : contexte CUDA primaire, tampons fp16 NCHW, inférence synchrone.
Usage : python selftest.py <dossier du plugin> <modèle> <cache> [y4m 10 bits 4:2:0 BT.2020]
Sans y4m : entrées synthétiques. Affiche la construction du moteur, le temps d'inférence moyen et
vérifie que la sortie est finie et dans [0, 1].
"""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

KR, KB = 0.2627, 0.0593
LOG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)


class CreateParams(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("cuda_device", ctypes.c_int), ("model_path", ctypes.c_char_p),
                ("cache_dir", ctypes.c_char_p), ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("log", LOG), ("log_user", ctypes.c_void_p)]


class InferParams(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("in0", ctypes.c_uint64), ("in1", ctypes.c_uint64),
                ("out", ctypes.c_uint64), ("t", ctypes.c_float)]


ERR = ctypes.c_char_p
VERSION = ctypes.CFUNCTYPE(ctypes.c_char_p)
SUPPORTED = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)
CREATE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(CreateParams), ctypes.c_char_p, ctypes.c_size_t)
INFER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(InferParams), ctypes.c_char_p, ctypes.c_size_t)
DESTROY = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class Api(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("version", VERSION), ("device_supported", SUPPORTED),
                ("create", CREATE), ("infer", INFER), ("destroy", DESTROY)]


def frames_from_y4m(path: Path, w: int, h: int, count: int = 2):
    out = []
    with path.open("rb") as f:
        f.readline()
        size = w * h * 2 + 2 * (w // 2) * (h // 2) * 2
        for _ in range(count):
            f.readline()
            a = np.frombuffer(f.read(size), dtype="<u2").astype(np.float32)
            y = a[: w * h].reshape(h, w)
            u = a[w * h: w * h + w * h // 4].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
            v = a[w * h + w * h // 4:].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
            yn, cb, cr = (y - 64) / 876, (u - 512) / 896, (v - 512) / 896
            r = yn + 2 * (1 - KR) * cr
            b = yn + 2 * (1 - KB) * cb
            g = (yn - KR * r - KB * b) / (1 - KR - KB)
            out.append(np.clip(np.stack([r, g, b]), 0, 1).astype(np.float16)[None])
    return out


def main() -> None:
    plugin_dir, model, cache = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    y4m = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    cu = ctypes.CDLL("libcuda.so.1")
    assert cu.cuInit(0) == 0
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
    assert cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0
    assert cu.cuCtxPushCurrent_v2(ctx) == 0

    lib = ctypes.CDLL(str(plugin_dir / "libmvo_rife_trt.so"))
    lib.mvo_trt_get_api.restype = ctypes.POINTER(Api)
    api = lib.mvo_trt_get_api(1).contents
    err = ctypes.create_string_buffer(512)
    print("plugin", api.version().decode())
    if not api.device_supported(0, err, 512):
        sys.exit("non pris en charge : " + err.value.decode())

    if y4m:
        header = y4m.open("rb").readline().split()
        w = int(next(t for t in header if t.startswith(b"W"))[1:])
        h = int(next(t for t in header if t.startswith(b"H"))[1:])
        f0, f1 = frames_from_y4m(y4m, w, h)
    else:
        w, h = 1920, 1088
        rng = np.random.default_rng(0)
        f0 = rng.random((1, 3, h, w), dtype=np.float32).astype(np.float16)
        f1 = np.roll(f0, 8, axis=3)

    log = LOG(lambda _u, lvl, msg: print(("info" if lvl == 0 else "warning") + ":", msg.decode(), flush=True))
    params = CreateParams(ctypes.sizeof(CreateParams), 0, str(model).encode(), str(cache).encode(), w, h, log, None)
    t0 = time.perf_counter()
    session = api.create(ctypes.byref(params), err, 512)
    if not session:
        sys.exit("création impossible : " + err.value.decode())
    print(f"session prête en {time.perf_counter() - t0:.1f} s ({w}x{h})")

    nbytes = 3 * w * h * 2
    ptrs = []
    for arr in (f0, f1, None):
        p = ctypes.c_uint64()
        assert cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)) == 0
        if arr is not None:
            assert cu.cuMemcpyHtoD_v2(p, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(nbytes)) == 0
        ptrs.append(p.value)
    ip = InferParams(ctypes.sizeof(InferParams), ptrs[0], ptrs[1], ptrs[2], 0.5)
    t0 = time.perf_counter()
    if not api.infer(session, ctypes.byref(ip), err, 512):
        sys.exit("inférence impossible : " + err.value.decode())
    print(f"première inférence : {time.perf_counter() - t0:.2f} s")
    n = 20
    t0 = time.perf_counter()
    for i in range(n):
        ip.t = 0.25 + 0.5 * (i % 2)
        assert api.infer(session, ctypes.byref(ip), err, 512), err.value.decode()
    print(f"inférence : {1000 * (time.perf_counter() - t0) / n:.1f} ms")
    ip.t = 0.5
    assert api.infer(session, ctypes.byref(ip), err, 512)
    out = np.empty((1, 3, h, w), dtype=np.float16)
    assert cu.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(ptrs[2]), ctypes.c_size_t(nbytes)) == 0
    o = out.astype(np.float32)
    assert np.isfinite(o).all(), "sortie non finie"
    print(f"sortie : min {o.min():.3f} max {o.max():.3f} moyenne {o.mean():.3f} ; entrées {f0.astype(np.float32).mean():.3f} / {f1.astype(np.float32).mean():.3f}")
    api.destroy(session)
    print("ok")


if __name__ == "__main__":
    main()
