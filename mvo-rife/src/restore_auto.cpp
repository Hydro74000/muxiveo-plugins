// Restauration (R2) : mesures de l'Auto et dosage (voir restore_auto.h).
#include "restore_auto.h"

#include <algorithm>
#include <cmath>

namespace restore_auto
{

namespace
{

// Profil cible de grain (σ par tranche, Gladiator II, codes 10 bits) ; tranches hautes non mesurées.
constexpr double TARGET[GRAIN_BINS] = {1.7984791736585508, 1.417293485886066, 1.4111985095540818, 1.1557043183924918,
                                       0.8855069513296798, 0.6741703446217169, 0.36706075453127496, 0.2554514033936689,
                                       0.22548441513119627, -1.0, -1.0, -1.0, -1.0, -1.0};
constexpr double MEDIAN_CORR = 0.8724929518;  // 1 / √(1 + π / 10) : biais de la médiane de 5 images

int reflect101(int i, int n)
{
    if (n == 1)
        return 0;
    while (i < 0 || i >= n)
        i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}

int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// Moyenne k × k, bords répliqués (avg_pool2d après F.pad(..., mode="replicate")).
void box_replicate(const float* src, int w, int h, int k, float* dst)
{
    const int r = k / 2;
    std::vector<float> tmp((std::size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            double s = 0.0;
            for (int d = -r; d <= r; d++)
                s += src[(std::size_t)y * w + clampi(x + d, w)];
            tmp[(std::size_t)y * w + x] = (float)(s / k);
        }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            double s = 0.0;
            for (int d = -r; d <= r; d++)
                s += tmp[(std::size_t)clampi(y + d, h) * w + x];
            dst[(std::size_t)y * w + x] = (float)(s / k);
        }
}

// Sobel 3 × 3 (dérivée en x ou en y), bords en miroir sans répétition.
void sobel(const float* src, int w, int h, bool dx, float* dst)
{
    for (int y = 0; y < h; y++)
    {
        const float* rm = src + (std::size_t)reflect101(y - 1, h) * w;
        const float* r0 = src + (std::size_t)y * w;
        const float* rp = src + (std::size_t)reflect101(y + 1, h) * w;
        for (int x = 0; x < w; x++)
        {
            const int xm = reflect101(x - 1, w), xp = reflect101(x + 1, w);
            dst[(std::size_t)y * w + x] = dx ? (rm[xp] - rm[xm]) + 2.f * (r0[xp] - r0[xm]) + (rp[xp] - rp[xm])
                                             : (rp[xm] - rm[xm]) + 2.f * (rp[x] - rm[x]) + (rp[xp] - rm[xp]);
        }
    }
}

// Lecture bilinéaire comme cv2.remap(INTER_LINEAR, BORDER_REPLICATE) : coordonnée au 1/32 de pixel.
inline float bilinear(const float* img, int w, int h, int channels, int ch, float fx, float fy)
{
    const int X = (int)std::lrint(fx * 32.f), Y = (int)std::lrint(fy * 32.f);
    const int sx = X >> 5, sy = Y >> 5;
    const float ax = (X & 31) / 32.f, ay = (Y & 31) / 32.f;
    const int x0 = clampi(sx, w), x1 = clampi(sx + 1, w), y0 = clampi(sy, h), y1 = clampi(sy + 1, h);
    auto at = [&](int x, int y) { return img[((std::size_t)y * w + x) * channels + ch]; };
    return (1.f - ay) * ((1.f - ax) * at(x0, y0) + ax * at(x1, y0)) + ay * ((1.f - ax) * at(x0, y1) + ax * at(x1, y1));
}

double robust_sigma(std::vector<float> v)
{
    const double lo = detail::percentile(v, 15.87), hi = detail::percentile(std::move(v), 84.13);
    return (hi - lo) / 2.0;
}

double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// splitmix64 : bruit déterministe du tramage.
inline uint64_t mix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::vector<float> luma_float(const uint16_t* y, int w, int h, std::ptrdiff_t stride)
{
    std::vector<float> f((std::size_t)w * h);
    for (int r = 0; r < h; r++)
        for (int x = 0; x < w; x++)
            f[(std::size_t)r * w + x] = y[r * stride + x];
    return f;
}

}  // namespace

namespace detail
{

double percentile(std::vector<float> v, double q)
{
    if (v.empty())
        return NAN;
    const double pos = (v.size() - 1) * q / 100.0;
    const std::size_t lo = (std::size_t)std::floor(pos);
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    const double a = v[lo];
    if (lo + 1 >= v.size())
        return a;
    const double b = *std::min_element(v.begin() + lo + 1, v.end());
    return a + (pos - lo) * (b - a);
}

int gaussian_ksize(double sigma) { return ((int)std::lrint(sigma * 4 * 2 + 1)) | 1; }

void gaussian_blur(const float* src, int w, int h, double sigma, float* dst)
{
    const int n = gaussian_ksize(sigma), r = n / 2;
    std::vector<float> k(n);
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        const double x = i - (n - 1) * 0.5;
        const double v = std::exp(-0.5 / (sigma * sigma) * x * x);
        k[i] = (float)v;
        sum += v;
    }
    for (float& v : k)
        v = (float)(v / sum);
    std::vector<float> tmp((std::size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float s = 0.f;
            for (int i = -r; i <= r; i++)
                s += k[i + r] * src[(std::size_t)y * w + reflect101(x + i, w)];
            tmp[(std::size_t)y * w + x] = s;
        }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float s = 0.f;
            for (int i = -r; i <= r; i++)
                s += k[i + r] * tmp[(std::size_t)reflect101(y + i, h) * w + x];
            dst[(std::size_t)y * w + x] = s;
        }
}

void area_resize(const float* src, int w, int h, int dw, int dh, float* dst)
{
    // poids d'aire d'un axe (cv2.INTER_AREA) : chaque pixel de sortie couvre [i × s, (i + 1) × s) de l'entrée
    auto weights = [](int n, int dn, std::vector<std::vector<std::pair<int, double>>>& tab) {
        const double s = (double)n / dn;
        tab.assign(dn, {});
        for (int i = 0; i < dn; i++)
        {
            const double a = i * s, b = std::min((i + 1) * s, (double)n);
            for (int j = (int)std::floor(a); j < (int)std::ceil(b - 1e-9); j++)
            {
                const double ov = std::min(b, j + 1.0) - std::max(a, (double)j);
                if (ov > 1e-12)
                    tab[i].push_back({j, ov / (b - a)});
            }
        }
    };
    std::vector<std::vector<std::pair<int, double>>> wx, wy;
    weights(w, dw, wx);
    weights(h, dh, wy);
    std::vector<double> rows((std::size_t)h * dw);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < dw; x++)
        {
            double s = 0.0;
            for (auto [j, c] : wx[x])
                s += c * src[(std::size_t)y * w + j];
            rows[(std::size_t)y * dw + x] = s;
        }
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++)
        {
            double s = 0.0;
            for (auto [j, c] : wy[y])
                s += c * rows[(std::size_t)j * dw + x];
            dst[(std::size_t)y * dw + x] = (float)s;
        }
}

}  // namespace detail

double bin_edge(int i) { return i >= GRAIN_BINS ? 940.0 : 64.0 + i * (876.0 / GRAIN_BINS); }

GrainCurve grain_curve(const uint16_t* y, int w, int h, std::ptrdiff_t stride)
{
    const std::size_t n = (std::size_t)w * h;
    const std::vector<float> t = luma_float(y, w, h, stride);
    std::vector<float> avg3(n), sm(n), grad(n);
    box_replicate(t.data(), w, h, 3, avg3.data());
    box_replicate(t.data(), w, h, 5, sm.data());
    // gradient |∂x| + |∂y| de la luma moyennée, intérieur (h − 2) × (w − 2), bords répliqués
    for (int r = 0; r < h; r++)
        for (int x = 0; x < w; x++)
        {
            const int ri = std::clamp(r, 1, std::max(1, h - 2)), xi = std::clamp(x, 1, std::max(1, w - 2));
            const float gx = std::fabs(sm[(std::size_t)ri * w + xi + 1] - sm[(std::size_t)ri * w + xi - 1]);
            const float gy = std::fabs(sm[(std::size_t)(ri + 1) * w + xi] - sm[(std::size_t)(ri - 1) * w + xi]);
            grad[(std::size_t)r * w + x] = gx + gy;
        }
    std::vector<float> sub;
    sub.reserve(n / 16 + 1);
    for (std::size_t i = 0; i < n; i += 16)
        sub.push_back(grad[i]);
    const double thr = detail::percentile(std::move(sub), 30.0);
    std::array<std::vector<float>, GRAIN_BINS> vals;
    float edge[GRAIN_BINS + 1];
    for (int i = 0; i <= GRAIN_BINS; i++)
        edge[i] = (float)bin_edge(i);
    for (std::size_t i = 0; i < n; i++)
    {
        if (!(grad[i] < thr))
            continue;
        const float s = sm[i];
        if (s < edge[0] || s >= edge[GRAIN_BINS])
            continue;
        int b = std::min(GRAIN_BINS - 1, (int)((s - edge[0]) / (edge[1] - edge[0])));
        while (b > 0 && s < edge[b])
            b--;
        while (b < GRAIN_BINS - 1 && s >= edge[b + 1])
            b++;
        vals[b].push_back(t[i] - avg3[i]);
    }
    GrainCurve c;
    std::array<bool, GRAIN_BINS> ok{};
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        c.center[b] = (float)((bin_edge(b) + bin_edge(b + 1)) / 2.0);
        c.sigma[b] = NAN;
        if (vals[b].size() > 3000)
        {
            const std::size_t step = std::max<std::size_t>(1, vals[b].size() / 200000);
            std::vector<float> v;
            for (std::size_t i = 0; i < vals[b].size(); i += step)
                v.push_back(vals[b][i]);
            c.sigma[b] = (float)robust_sigma(std::move(v));
            ok[b] = true;
        }
    }
    if (std::none_of(ok.begin(), ok.end(), [](bool v) { return v; }))
    {
        c.sigma.fill(0.5f);
        return c;
    }
    const std::array<float, GRAIN_BINS> measured = c.sigma;
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        int best = -1;
        for (int j = 0; j < GRAIN_BINS; j++)
            if (ok[j] && (best < 0 || std::abs(j - b) < std::abs(best - b)))
                best = j;
        c.sigma[b] = measured[best];
    }
    return c;
}

GrainCurve median_curve(const std::vector<GrainCurve>& curves)
{
    GrainCurve c;
    if (curves.empty())
        return c;
    c.center = curves[0].center;
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        std::vector<double> v;
        for (const GrainCurve& g : curves)
            v.push_back(g.sigma[b]);
        c.sigma[b] = (float)median_of(std::move(v));
    }
    return c;
}

void grain_map(const uint16_t* y, int w, int h, std::ptrdiff_t stride, const GrainCurve& curve, uint8_t* map)
{
    const std::vector<float> t = luma_float(y, w, h, stride);
    std::vector<float> lum(t.size());
    detail::gaussian_blur(t.data(), w, h, 2.0, lum.data());
    for (std::size_t i = 0; i < lum.size(); i++)
    {
        const float v = lum[i];
        float s;
        if (v <= curve.center[0])
            s = std::max(curve.sigma[0], 0.f);
        else if (v >= curve.center[GRAIN_BINS - 1])
            s = std::max(curve.sigma[GRAIN_BINS - 1], 0.f);
        else
        {
            int b = 0;
            while (v > curve.center[b + 1])
                b++;
            const float a = (v - curve.center[b]) / (curve.center[b + 1] - curve.center[b]);
            s = std::max(curve.sigma[b], 0.f) + a * (std::max(curve.sigma[b + 1], 0.f) - std::max(curve.sigma[b], 0.f));
        }
        map[i] = (uint8_t)std::clamp(std::nearbyint(s * 20.f), 0.f, 255.f);
    }
}

float residual_sigma(const uint16_t* src_y, std::ptrdiff_t src_stride, const float* out_y, std::ptrdiff_t out_stride, int w, int h)
{
    const std::size_t n = (std::size_t)w * h;
    std::vector<float> d(n), blur(n);
    for (int r = 0; r < h; r++)
        for (int x = 0; x < w; x++)
            d[(std::size_t)r * w + x] = (float)src_y[r * src_stride + x] - out_y[r * out_stride + x];
    detail::gaussian_blur(d.data(), w, h, 1.2, blur.data());
    for (std::size_t i = 0; i < n; i++)
        d[i] -= blur[i];
    const double med = detail::percentile(d, 50.0);
    for (float& v : d)
        v = std::fabs(v - (float)med);
    return (float)std::max(1.4826 * detail::percentile(std::move(d), 50.0), 1e-3);
}

CutDetector::CutDetector(int past, int look, double factor, double floor) : past_(past), look_(look), factor_(factor), floor_(floor) {}

void CutDetector::push(const uint16_t* y, int w, int h, std::ptrdiff_t stride)
{
    const int dw = w / 4, dh = h / 4;
    const std::vector<float> t = luma_float(y, w, h, stride);
    std::vector<float> sm((std::size_t)dw * dh);
    detail::area_resize(t.data(), w, h, dw, dh, sm.data());
    double s = 0.0;
    if (frames_ > 0)
        for (std::size_t i = 0; i < sm.size(); i++)
            s += std::fabs(sm[i] - prev_[i]);
    diff_.push_back(frames_ > 0 ? (float)(s / sm.size()) : 0.f);
    cut_.push_back(frames_ == 0 ? 1 : 0);
    prev_ = std::move(sm);
    frames_++;
    decide();
}

void CutDetector::finish()
{
    finished_ = true;
    decide();
}

void CutDetector::decide()
{
    const int last = frames_ - 1;
    while (decided_ < last)
    {
        const int t = decided_ + 1;
        if (!finished_ && t + look_ > last)
            return;
        std::vector<float> ds(diff_.begin() + std::max(1, t - past_), diff_.begin() + std::min(last, t + look_) + 1);
        const double med = detail::percentile(std::move(ds), 50.0);
        if (diff_[t] > std::max(factor_ * med, floor_))
            cut_[t] = 1;
        decided_ = t;
    }
}

int CutDetector::shot_start(int t) const
{
    while (t > 0 && !cut_[t])
        t--;
    return std::max(t, 0);
}

std::array<std::optional<double>, GRAIN_BINS> sigma_bins(const std::map<int, AnalysisFrame>& frames, const std::vector<int>& centers,
                                                         int w, int h, double ar)
{
    static const int order[4] = {-2, -1, 1, 2};
    const std::size_t n = (std::size_t)w * h;
    std::array<std::vector<double>, GRAIN_BINS> acc;
    std::vector<float> med(n), res(n), gx(n), gy(n), mag(n), grad(n);
    std::vector<uint8_t> mask(n);
    for (int c : centers)
    {
        const AnalysisFrame& fc = frames.at(c);
        const float* nb[4];
        const float* fwd[4];
        const float* bwd[4];
        for (int j = 0; j < 4; j++)
        {
            const AnalysisFrame& fn = frames.at(c + order[j]);
            nb[j] = fn.luma;
            fwd[j] = fc.flows + (std::size_t)j * n * 2;
            bwd[j] = fn.flows + (std::size_t)(3 - j) * n * 2;  // flux de la voisine vers c (ordre −2, −1, +1, +2)
        }
        for (int r = 0; r < h; r++)
            for (int x = 0; x < w; x++)
            {
                const std::size_t i = (std::size_t)r * w + x;
                float v[5] = {fc.luma[i], 0.f, 0.f, 0.f, 0.f};
                bool keep = true;
                for (int j = 0; j < 4; j++)
                {
                    const float fx = fwd[j][2 * i], fy = fwd[j][2 * i + 1];
                    const float mx = x + fx, my = r + fy;
                    const float bx = bilinear(bwd[j], w, h, 2, 0, mx, my), by = bilinear(bwd[j], w, h, 2, 1, mx, my);
                    keep = keep && std::hypot(fx + bx, fy + by) < ar;
                    v[j + 1] = bilinear(nb[j], w, h, 1, 0, mx, my);
                }
                std::sort(v, v + 5);
                med[i] = v[2];
                mask[i] = keep;
                res[i] = fc.luma[i] - v[2];
            }
        sobel(med.data(), w, h, true, gx.data());
        sobel(med.data(), w, h, false, gy.data());
        for (std::size_t i = 0; i < n; i++)
            mag[i] = std::hypot(gx[i], gy[i]);
        detail::gaussian_blur(mag.data(), w, h, 2.0, grad.data());
        const double thr = detail::percentile(grad, 40.0);
        std::array<std::vector<float>, GRAIN_BINS> sel;
        for (std::size_t i = 0; i < n; i++)
        {
            if (!mask[i] || !(grad[i] < thr))
                continue;
            for (int b = 0; b < GRAIN_BINS; b++)
                if (med[i] >= bin_edge(b) && med[i] < bin_edge(b + 1))
                {
                    sel[b].push_back(res[i]);
                    break;
                }
        }
        for (int b = 0; b < GRAIN_BINS; b++)
            if (sel[b].size() > 4000)
                acc[b].push_back(robust_sigma(std::move(sel[b])) * MEDIAN_CORR);
    }
    std::array<std::optional<double>, GRAIN_BINS> out;
    for (int b = 0; b < GRAIN_BINS; b++)
        if (!acc[b].empty())
            out[b] = median_of(acc[b]);
    return out;
}

std::optional<double> ratio_from_sigma(const std::array<std::optional<double>, GRAIN_BINS>& sigma)
{
    double meas_sum = 0.0, common_s = 0.0, common_t = 0.0;
    int meas = 0, common = 0;
    std::vector<double> targets;
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        if (TARGET[b] > 0)
            targets.push_back(TARGET[b]);
        if (!sigma[b])
            continue;
        meas_sum += *sigma[b];
        meas++;
        if (TARGET[b] > 0)
        {
            common_s += *sigma[b];
            common_t += TARGET[b];
            common++;
        }
    }
    if (!meas)
        return std::nullopt;
    const double r = common ? (common_s / common) / (common_t / common) : (meas_sum / meas) / median_of(targets);
    return std::min(r, 10.0);
}

double auto_part(std::optional<double> ratio, double k) { return ratio ? std::min(1.0, k / std::max(*ratio, 1e-3)) : 0.25 * k; }

void dose(const uint16_t* src, const float* out, std::size_t n, double kk, uint16_t lo, uint16_t hi, uint16_t* dst,
          long long dither_frame, std::size_t dither_offset)
{
    const float k = (float)kk;
    for (std::size_t i = 0; i < n; i++)
    {
        float v = out[i] + k * ((float)src[i] - out[i]);
        if (dither_frame >= 0)
        {
            const uint64_t z = mix64(((uint64_t)dither_frame << 40) ^ (uint64_t)(dither_offset + i));
            v += (float)((z >> 40) * (1.0 / 16777216.0)) + (float)((z & 0xFFFFFFull) * (1.0 / 16777216.0)) - 1.f;
        }
        dst[i] = (uint16_t)std::clamp(std::nearbyint(v), (float)lo, (float)hi);
    }
}

}  // namespace restore_auto
