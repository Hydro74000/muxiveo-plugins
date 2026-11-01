"""Restauration (R2) : mesures de l'Auto et dosage (src/restore_auto.cpp) face à une référence NumPy."""

import ctypes as C
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
f32p, u16p, u8p, f64p, i32p = (np.ctypeslib.ndpointer(t, flags='C_CONTIGUOUS')
                               for t in (np.float32, np.uint16, np.uint8, np.float64, np.int32))


@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        pytest.skip('Compilateur C++ requis')
    suffix = {'win32': '.dll', 'darwin': '.dylib'}.get(sys.platform, '.so')
    out = tmp_path_factory.mktemp('restore_auto') / f'restore_auto{suffix}'
    flags = [] if sys.platform == 'win32' else ['-fPIC']
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', *flags, '-I' + str(ROOT / 'src'),
                    str(ROOT / 'src' / 'restore_auto.cpp'), str(ROOT / 'tests' / 'restore_auto_capi.cpp'), '-o', str(out)],
                   check=True, capture_output=True, timeout=120)
    lib = C.CDLL(str(out))
    lib.ra_grain_curve.argtypes = [u16p, C.c_int, C.c_int, f32p, f32p]
    lib.ra_median_curve.argtypes = [f32p, C.c_int, f32p]
    lib.ra_grain_map.argtypes = [u16p, C.c_int, C.c_int, f32p, f32p, u8p]
    lib.ra_residual_sigma.argtypes = [u16p, f32p, C.c_int, C.c_int]
    lib.ra_residual_sigma.restype = C.c_float
    lib.ra_cuts.argtypes = [u16p, C.c_int, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_int, i32p]
    lib.ra_cuts.restype = C.c_int
    lib.ra_sigma_bins.argtypes = [f32p, f32p, C.c_int, C.c_int, i32p, C.c_int, C.c_int, C.c_int, C.c_double, f64p]
    lib.ra_ratio.argtypes = [f64p]
    lib.ra_ratio.restype = C.c_double
    lib.ra_dose.argtypes = [u16p, f32p, C.c_size_t, C.c_double, C.c_uint16, C.c_uint16, u16p, C.c_longlong]
    lib.ra_percentile.argtypes = [f32p, C.c_size_t, C.c_double]
    lib.ra_percentile.restype = C.c_double
    lib.ra_gaussian_blur.argtypes = [f32p, C.c_int, C.c_int, C.c_double, f32p]
    lib.ra_area_resize.argtypes = [f32p, C.c_int, C.c_int, C.c_int, C.c_int, f32p]
    return lib


# --- références NumPy (mêmes conventions : OpenCV pour les flous, NumPy pour les centiles) ---------------------------
def ref_gauss(img, sigma):
    n = int(np.rint(sigma * 8 + 1)) | 1
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-0.5 / sigma ** 2 * x ** 2)
    k = (k / k.sum()).astype(np.float32)
    r = n // 2
    p = np.pad(img, ((0, 0), (r, r)), mode='reflect')  # miroir sans répétition du bord (BORDER_REFLECT_101)
    tmp = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(n))
    p = np.pad(tmp, ((r, r), (0, 0)), mode='reflect')
    return sum(k[i] * p[i:i + img.shape[0]] for i in range(n)).astype(np.float32)


def ref_box_edge(img, k):
    r = k // 2
    p = np.pad(img, r, mode='edge')
    h, w = img.shape
    return sum(p[i:i + h, j:j + w] for i in range(k) for j in range(k)) / (k * k)


def ref_sobel(img, dx):
    p = np.pad(img, 1, mode='reflect')
    h, w = img.shape
    s = lambda i, j: p[1 + i:1 + i + h, 1 + j:1 + j + w]  # noqa: E731
    if dx:
        return (s(-1, 1) - s(-1, -1)) + 2 * (s(0, 1) - s(0, -1)) + (s(1, 1) - s(1, -1))
    return (s(1, -1) - s(-1, -1)) + 2 * (s(1, 0) - s(-1, 0)) + (s(1, 1) - s(-1, 1))


EDGES = np.linspace(64, 940, 15)


def ref_grain_curve(y):
    t = y.astype(np.float32)
    hp = t - ref_box_edge(t, 3)
    sm = ref_box_edge(t, 5)
    g = np.abs(sm[1:-1, 2:] - sm[1:-1, :-2]) + np.abs(sm[2:, 1:-1] - sm[:-2, 1:-1])
    grad = np.pad(g, 1, mode='edge')
    flat = grad < np.quantile(grad.ravel()[::16], 0.3)
    sig = np.full(14, np.nan)
    for i in range(14):
        sel = flat & (sm >= np.float32(EDGES[i])) & (sm < np.float32(EDGES[i + 1]))
        if sel.sum() > 3000:
            v = hp[sel]
            v = v[::max(1, v.size // 200000)]
            lo, hi = np.quantile(v, [0.1587, 0.8413])
            sig[i] = (hi - lo) / 2
    ok = ~np.isnan(sig)
    idx = np.arange(14)
    sig = sig[[idx[ok][np.abs(idx[ok] - i).argmin()] for i in idx]] if ok.any() else np.full(14, 0.5)
    return (EDGES[:-1] + EDGES[1:]) / 2, sig


def synthetic(h=360, w=640, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ramp = np.linspace(80, 900, w)[None, :].repeat(h, 0)
    return np.clip(np.rint(ramp + rng.normal(0, sigma, (h, w))), 4, 1019).astype(np.uint16)


# --- tests ---------------------------------------------------------------------------------------------------------
def test_percentile_matches_numpy(lib):
    rng = np.random.default_rng(1)
    for n in (1, 2, 7, 1000, 12345):
        v = rng.normal(0, 3, n).astype(np.float32)
        for q in (0, 15.87, 30, 40, 50, 84.13, 100):
            assert lib.ra_percentile(v, n, q) == pytest.approx(np.percentile(v.astype(np.float64), q), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize('sigma', [1.2, 2.0, 3.0])
def test_gaussian_blur_matches_opencv_convention(lib, sigma):
    img = np.random.default_rng(2).normal(500, 40, (97, 131)).astype(np.float32)
    out = np.zeros_like(img)
    lib.ra_gaussian_blur(img, 131, 97, sigma, out)
    assert np.abs(out - ref_gauss(img, sigma)).max() < 2e-3


def test_area_resize(lib):
    img = np.random.default_rng(3).uniform(0, 1023, (64, 96)).astype(np.float32)
    out = np.zeros((16, 24), np.float32)
    lib.ra_area_resize(img, 96, 64, 24, 16, out)
    assert np.abs(out - img.reshape(16, 4, 24, 4).mean((1, 3))).max() < 1e-3
    flat = np.full((67, 50), 321.0, np.float32)
    out = np.zeros((16, 12), np.float32)
    lib.ra_area_resize(flat, 50, 67, 12, 16, out)
    assert np.abs(out - 321.0).max() < 1e-3


def test_grain_curve_matches_reference(lib):
    y = synthetic(sigma=2.5)
    c, s = np.zeros(14, np.float32), np.zeros(14, np.float32)
    lib.ra_grain_curve(np.ascontiguousarray(y), y.shape[1], y.shape[0], c, s)
    rc, rs = ref_grain_curve(y)
    assert np.allclose(c, rc, atol=1e-3)
    assert np.allclose(s, rs, rtol=1e-3, atol=1e-3)
    flat = np.full((200, 300), 16, np.uint16)  # aucune tranche mesurée
    lib.ra_grain_curve(flat, 300, 200, c, s)
    assert np.all(s == 0.5)


def test_median_curve(lib):
    sig = np.random.default_rng(4).uniform(0, 3, (4, 14)).astype(np.float32)
    out = np.zeros(14, np.float32)
    lib.ra_median_curve(np.ascontiguousarray(sig), 4, out)
    assert np.allclose(out, np.median(sig, 0), atol=1e-6)


def test_grain_map_matches_reference(lib):
    y = synthetic(sigma=1.5)
    c = ((EDGES[:-1] + EDGES[1:]) / 2).astype(np.float32)
    s = np.linspace(2.0, 0.2, 14).astype(np.float32)
    m = np.zeros(y.size, np.uint8)
    lib.ra_grain_map(np.ascontiguousarray(y), y.shape[1], y.shape[0], c, s, m)
    lum = ref_gauss(y.astype(np.float32), 2.0)
    ref = np.clip(np.rint(np.interp(lum, c, s) * 20), 0, 255).astype(np.uint8)
    assert np.mean(m == ref.ravel()) > 0.9999
    assert np.abs(m.astype(int) - ref.ravel().astype(int)).max() <= 1


def test_residual_sigma(lib):
    y = synthetic(sigma=3.0)
    out = ref_gauss(y.astype(np.float32), 1.5)
    got = lib.ra_residual_sigma(np.ascontiguousarray(y), np.ascontiguousarray(out), y.shape[1], y.shape[0])
    d = y.astype(np.float32) - out
    hp = d - ref_gauss(d, 1.2)
    assert got == pytest.approx(1.4826 * np.median(np.abs(hp - np.median(hp))), rel=1e-4)


def test_cut_detector_finds_scene_changes(lib):
    rng = np.random.default_rng(5)
    h, w = 72, 128
    frames = []
    for t in range(40):
        level = 200 if t < 17 else (700 if t < 30 else 400)
        y = np.clip(level + rng.normal(0, 2, (h, w)) + t * 0.3, 4, 1019)
        frames.append(np.concatenate([y.ravel(), np.full(h * w // 2, 512.0)]).astype(np.uint16))
    seq = np.ascontiguousarray(np.stack(frames))
    starts = np.zeros(40, np.int32)
    k = lib.ra_cuts(seq, 40, seq.shape[1], w, h, 48, 12, starts)
    assert starts[:k].tolist() == [0, 17, 30]


def test_sigma_bins_and_ratio_with_zero_flow(lib):
    h, w, sigma = 480, 560, 2.0  # ≥ 4000 pixels d'aplats par tranche
    lumas = np.ascontiguousarray(np.stack([synthetic(h, w, sigma, seed=10 + i).astype(np.float32) for i in range(5)]))
    flows = np.zeros((5, 4, h, w, 2), np.float32)
    got = np.zeros(14, np.float64)
    lib.ra_sigma_bins(lumas, flows, 5, 0, np.array([2], np.int32), 1, w, h, 1.0, got)
    # référence : médiane des 5 images (flux nuls), aplats sous le 40e centile du gradient lissé
    med = np.median(lumas, 0)
    grad = ref_gauss(np.hypot(ref_sobel(med, True), ref_sobel(med, False)).astype(np.float32), 2.0)
    flat = grad < np.percentile(grad, 40)
    res = lumas[2] - med
    ref = np.full(14, np.nan)
    for i in range(14):
        sel = flat & (med >= EDGES[i]) & (med < EDGES[i + 1])
        if sel.sum() > 4000:
            lo, hi = np.percentile(res[sel], [15.87, 84.13])
            ref[i] = (hi - lo) / 2 / np.sqrt(1 + np.pi / 10)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    ok = ~np.isnan(ref)
    assert ok.sum() >= 3 and np.allclose(got[ok], ref[ok], rtol=2e-3)  # seuil du 40e centile : quelques pixels basculent
    assert np.all(np.abs(got[ok] / sigma - 1) < 0.2)  # biais de la médiane corrigé
    target = [1.7984791736585508, 1.417293485886066, 1.4111985095540818, 1.1557043183924918, 0.8855069513296798,
              0.6741703446217169, 0.36706075453127496, 0.2554514033936689, 0.22548441513119627]
    common = [i for i in range(9) if ok[i]]
    expected = min(np.mean(got[common]) / np.mean([target[i] for i in common]), 10.0) if common else \
        min(np.mean(got[ok]) / np.median(target), 10.0)
    assert lib.ra_ratio(got) == pytest.approx(expected, rel=1e-9)
    assert np.isnan(lib.ra_ratio(np.full(14, np.nan)))


def test_dose_rounding_and_dither(lib):
    rng = np.random.default_rng(6)
    src = rng.integers(4, 1020, 50000).astype(np.uint16)
    out = (src + rng.normal(0, 4, src.size)).astype(np.float32)
    dst = np.zeros_like(src)
    lib.ra_dose(src, out, src.size, 0.37, 4, 1019, dst, -1)
    ref = np.clip(np.rint(out + np.float32(0.37) * (src.astype(np.float32) - out)), 4, 1019)
    assert np.array_equal(dst, ref.astype(np.uint16))
    a, b, c = (np.zeros_like(src) for _ in range(3))
    lib.ra_dose(src, out, src.size, 0.37, 4, 1019, a, 3)
    lib.ra_dose(src, out, src.size, 0.37, 4, 1019, b, 3)
    lib.ra_dose(src, out, src.size, 0.37, 4, 1019, c, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)  # déterministe, propre à chaque image
    err = a.astype(np.float64) - (out + np.float32(0.37) * (src.astype(np.float32) - out))
    inner = (a > 4) & (a < 1019)
    assert abs(err[inner].mean()) < 0.02 and 0.45 < err[inner].std() < 0.55  # triangulaire ±1 + arrondi
    assert np.abs(err[inner]).max() <= 1.5
