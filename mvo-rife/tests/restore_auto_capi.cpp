// Interface C de restore_auto pour les tests (chargée par ctypes) : tableaux contigus, valeurs absentes = NaN.
#include <cmath>
#include <map>

#include "restore_auto.h"

using namespace restore_auto;

extern "C" {

void ra_grain_curve(const uint16_t* y, int w, int h, float* centers, float* sigma)
{
    const GrainCurve c = grain_curve(y, w, h, w);
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        centers[b] = c.center[b];
        sigma[b] = c.sigma[b];
    }
}

void ra_median_curve(const float* sigmas, int count, float* sigma_out)
{
    std::vector<GrainCurve> curves(count);
    for (int i = 0; i < count; i++)
        for (int b = 0; b < GRAIN_BINS; b++)
            curves[i].sigma[b] = sigmas[i * GRAIN_BINS + b];
    const GrainCurve c = median_curve(curves);
    for (int b = 0; b < GRAIN_BINS; b++)
        sigma_out[b] = c.sigma[b];
}

void ra_grain_map(const uint16_t* y, int w, int h, const float* centers, const float* sigma, uint8_t* map)
{
    GrainCurve c;
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        c.center[b] = centers[b];
        c.sigma[b] = sigma[b];
    }
    grain_map(y, w, h, w, c, map);
}

float ra_residual_sigma(const uint16_t* src, const float* out, int w, int h) { return residual_sigma(src, w, out, w, w, h); }

// Images Y'CbCr consécutives (frame_size valeurs chacune, luma en tête) ; renvoie le nombre de débuts de plan écrits.
int ra_cuts(const uint16_t* frames, int n, std::size_t frame_size, int w, int h, int past, int look, int* starts)
{
    CutDetector det(past, look);
    for (int t = 0; t < n; t++)
        det.push(frames + (std::size_t)t * frame_size, w, h, w);
    det.finish();
    int k = 0;
    for (int t = 0; t < n; t++)
        if (det.is_cut(t))
            starts[k++] = t;
    return k;
}

// lumas : nframes × h × w ; flows : nframes × 4 × h × w × 2 ; image i = numéro first + i.
void ra_sigma_bins(const float* lumas, const float* flows, int nframes, int first, const int* centers, int ncenters, int w, int h,
                   double ar, double* sigma_out, double* pixels_out)
{
    std::map<int, AnalysisFrame> frames;
    const std::size_t n = (std::size_t)w * h;
    for (int i = 0; i < nframes; i++)
        frames[first + i] = AnalysisFrame{lumas + i * n, flows + i * n * 8};
    const auto s = sigma_bins(frames, std::vector<int>(centers, centers + ncenters), w, h, ar);
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        sigma_out[b] = s.sigma[b] ? *s.sigma[b] : NAN;
        pixels_out[b] = s.pixels[b];
    }
}

double ra_ratio(const double* sigma, const double* pixels)
{
    GrainBins s;
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        if (!std::isnan(sigma[b]))
            s.sigma[b] = sigma[b];
        s.pixels[b] = pixels[b];
    }
    const auto r = ratio_from_sigma(s);
    return r ? *r : NAN;
}

void ra_dose(const uint16_t* src, const float* out, std::size_t n, double kk, uint16_t lo, uint16_t hi, uint16_t* dst, long long dither_frame)
{
    dose(src, out, n, kk, lo, hi, dst, dither_frame, 0);
}

double ra_percentile(const float* v, std::size_t n, double q) { return detail::percentile(std::vector<float>(v, v + n), q); }

void ra_gaussian_blur(const float* src, int w, int h, double sigma, float* dst) { detail::gaussian_blur(src, w, h, sigma, dst); }

void ra_area_resize(const float* src, int w, int h, int dw, int dh, float* dst) { detail::area_resize(src, w, h, dw, dh, dst); }
}
