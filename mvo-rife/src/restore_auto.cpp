// Restauration (R2) : mesures de l'Auto et dosage (voir restore_auto.h).
#include "restore_auto.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace restore_auto
{

namespace
{

// Profil cible de grain (σ par tranche, Gladiator II, codes 10 bits) ; tranches hautes non mesurées.
constexpr double TARGET[GRAIN_BINS] = {1.7984791736585508, 1.417293485886066, 1.4111985095540818, 1.1557043183924918,
                                       0.8855069513296798, 0.6741703446217169, 0.36706075453127496, 0.2554514033936689,
                                       0.22548441513119627, -1.0, -1.0, -1.0, -1.0, -1.0};
constexpr double MEDIAN_CORR = 0.8724929518;  // 1 / √(1 + π / 10) : biais de la médiane de 5 images
constexpr double RATIO_GUARD = 2.0;           // borne de l'Auto v2 : 2 × médiane pondérée par tranche

int reflect101(int i, int n)
{
    if (n == 1)
        return 0;
    while (i < 0 || i >= n)
        i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}

int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

int g_threads = 0;  // 0 : selon le processeur

// Réserve de fils persistante : les boucles parallèles d'une image (une vingtaine) ne créent plus de fils à chaque appel.
// Plusieurs appelants simultanés (étage de Restauration, statistiques des trames) se partagent les fils ; le fil appelant
// traite lui aussi des bandes ; un appel depuis un fil de la réserve s'exécute sur place (pas d'attente circulaire).
thread_local bool tl_pool_worker = false;

class Pool
{
public:
    static Pool& instance()
    {
        static Pool pool;
        return pool;
    }

    // fn(i) pour i ∈ [0, n), réparti sur les fils de la réserve et le fil appelant ; rend la main quand tout est fait.
    void run(int n, const std::function<void(int)>& fn)
    {
        Job job{&fn, n};
        {
            std::lock_guard<std::mutex> lk(m);
            jobs.push_back(&job);
        }
        cv.notify_all();
        for (;;)
        {
            int i;
            {
                std::lock_guard<std::mutex> lk(m);
                if (job.next >= n)
                    break;
                i = job.next++;
                if (job.next >= n)
                    drop(&job);
            }
            fn(i);
            std::lock_guard<std::mutex> lk(m);
            if (++job.done == n)
                break;
        }
        std::unique_lock<std::mutex> lk(m);
        done_cv.wait(lk, [&] { return job.done == n; });
    }

private:
    struct Job
    {
        const std::function<void(int)>* fn;
        int n;
        int next = 0, done = 0;
    };

    Pool()
    {
        const int hw = std::clamp((int)std::thread::hardware_concurrency(), 1, 32);
        for (int i = 0; i < hw - 1; i++)
            threads.emplace_back([this] { work(); });
    }

    ~Pool()
    {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : threads)
            t.join();
    }

    void drop(Job* job)
    {
        const auto it = std::find(jobs.begin(), jobs.end(), job);
        if (it != jobs.end())
            jobs.erase(it);
    }

    void work()
    {
        tl_pool_worker = true;
        std::unique_lock<std::mutex> lk(m);
        for (;;)
        {
            cv.wait(lk, [&] { return stop || !jobs.empty(); });
            if (stop)
                return;
            Job* job = jobs.front();
            const int i = job->next++;
            if (job->next >= job->n)
                drop(job);
            lk.unlock();
            (*job->fn)(i);
            lk.lock();
            if (++job->done == job->n)
                done_cv.notify_all();
        }
    }

    std::mutex m;
    std::condition_variable cv, done_cv;
    std::deque<Job*> jobs;
    std::vector<std::thread> threads;
    bool stop = false;
};

// Exécute fn(i) pour chaque tâche i ∈ [0, t), en parallèle (sur place depuis un fil de la réserve).
void parallel_tasks(int t, const std::function<void(int)>& fn)
{
    if (t <= 1 || tl_pool_worker)
    {
        for (int i = 0; i < t; i++)
            fn(i);
        return;
    }
    Pool::instance().run(t, fn);
}

// Exécute fn(début, fin) sur des bandes de [0, n), en parallèle ; chaque élément ne dépend que de lui-même : résultats
// identiques quel que soit le découpage.
template <class F>
void parallel_for(int n, F&& fn)
{
    const int hw = g_threads > 0 ? g_threads : (int)std::thread::hardware_concurrency();
    const int t = std::min(std::clamp(hw, 1, 32), std::max(1, n / 16));
    if (t <= 1 || tl_pool_worker)
    {
        fn(0, n);
        return;
    }
    Pool::instance().run(t, [&](int i) {
        const int a = (int)((int64_t)n * i / t), b = (int)((int64_t)n * (i + 1) / t);
        fn(a, b);
    });
}

// Clé ordonnée d'un flottant (ordre des entiers non signés = ordre des flottants).
inline uint32_t float_key(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Éléments de rangs r0 ≤ r1 (0 = plus petit) en une passe : histogramme des 16 bits de tête (parallèle), puis sélection
// exacte dans la ou les cases retenues (collecte parallèle ; l'ordre de collecte ne change pas l'élément de rang donné).
std::pair<float, float> select_ranks(const std::vector<float>& v, size_t r0, size_t r1)
{
    const int n = (int)v.size();
    const int hw = g_threads > 0 ? g_threads : (int)std::thread::hardware_concurrency();
    const int t = std::min(std::clamp(hw, 1, 32), std::max(1, n / 65536));
    std::vector<uint32_t> hist((size_t)t * 65536, 0);
    parallel_tasks(t, [&](int i) {
        uint32_t* hi = hist.data() + (size_t)i * 65536;
        const int a = (int)((int64_t)n * i / t), b = (int)((int64_t)n * (i + 1) / t);
        for (int j = a; j < b; j++)
            hi[float_key(v[j]) >> 16]++;
    });
    // case de chaque rang et effectif cumulé avant elle
    size_t cum = 0, cum0 = 0, cum1 = 0;
    uint32_t bin0 = 65535, bin1 = 65535;
    bool found0 = false;
    for (uint32_t bin = 0; bin < 65536; bin++)
    {
        size_t c = 0;
        for (int i = 0; i < t; i++)
            c += hist[(size_t)i * 65536 + bin];
        if (!found0 && cum + c > r0)
        {
            bin0 = bin;
            cum0 = cum;
            found0 = true;
        }
        if (cum + c > r1)
        {
            bin1 = bin;
            cum1 = cum;
            break;
        }
        cum += c;
    }
    std::vector<std::vector<float>> part0(t), part1(t);
    parallel_tasks(t, [&](int i) {
        const int a = (int)((int64_t)n * i / t), b = (int)((int64_t)n * (i + 1) / t);
        for (int j = a; j < b; j++)
        {
            const uint32_t key = float_key(v[j]) >> 16;
            if (key == bin0)
                part0[i].push_back(v[j]);
            else if (key == bin1)
                part1[i].push_back(v[j]);
        }
    });
    std::vector<float> sub0, sub1;
    for (int i = 0; i < t; i++)
    {
        sub0.insert(sub0.end(), part0[i].begin(), part0[i].end());
        sub1.insert(sub1.end(), part1[i].begin(), part1[i].end());
    }
    std::nth_element(sub0.begin(), sub0.begin() + (r0 - cum0), sub0.end());
    const float a = sub0[r0 - cum0];
    if (bin1 == bin0)
    {
        std::nth_element(sub0.begin(), sub0.begin() + (r1 - cum0), sub0.end());
        return {a, sub0[r1 - cum0]};
    }
    std::nth_element(sub1.begin(), sub1.begin() + (r1 - cum1), sub1.end());
    return {a, sub1[r1 - cum1]};
}

// Médiane (centile 50 à interpolation linéaire, comme numpy.median).
double median_exact(const std::vector<float>& v)
{
    const size_t lo = (v.size() - 1) / 2;
    if ((v.size() - 1) % 2)
    {
        const auto [a, b] = select_ranks(v, lo, lo + 1);
        const double da = a;
        return 0.5 * (da + b);
    }
    return select_ranks(v, lo, lo).first;
}

// Moyenne k × k, bords répliqués (avg_pool2d après F.pad(..., mode="replicate")).
void box_replicate(const float* src, int w, int h, int k, float* dst)
{
    // même ordre de sommation par élément (d croissant, accumulateur double) ; intérieur sans calcul d'indice de bord
    const int r = k / 2;
    std::vector<float> tmp((std::size_t)w * h);
    parallel_for(h, [&](int y0, int y1) {
        std::vector<double> acc(w);
        for (int y = y0; y < y1; y++)
        {
            const float* row = src + (std::size_t)y * w;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int d = -r; d <= r; d++)
            {
                int x = 0;
                for (; x < std::min(w, r); x++)
                    acc[x] += row[clampi(x + d, w)];
                for (; x < w - r; x++)
                    acc[x] += row[x + d];
                for (; x < w; x++)
                    acc[x] += row[clampi(x + d, w)];
            }
            for (int x = 0; x < w; x++)
                tmp[(std::size_t)y * w + x] = (float)(acc[x] / k);
        }
    });
    parallel_for(h, [&](int y0, int y1) {
        std::vector<double> acc(w);
        for (int y = y0; y < y1; y++)
        {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int d = -r; d <= r; d++)
            {
                const float* row = tmp.data() + (std::size_t)clampi(y + d, h) * w;
                for (int x = 0; x < w; x++)
                    acc[x] += row[x];
            }
            for (int x = 0; x < w; x++)
                dst[(std::size_t)y * w + x] = (float)(acc[x] / k);
        }
    });
}

// Sobel 3 × 3 (dérivée en x ou en y), bords en miroir sans répétition.
void sobel(const float* src, int w, int h, bool dx, float* dst)
{
    parallel_for(h, [&](int ya, int yb) {
    for (int y = ya; y < yb; y++)
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
    });
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
    parallel_for(h, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++)
            for (int x = 0; x < w; x++)
                f[(std::size_t)r * w + x] = y[r * stride + x];
    });
    return f;
}

}  // namespace

namespace detail
{

void for_rows(int n, const std::function<void(int, int)>& fn) { parallel_for(n, fn); }

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
    // chaque élément : s = 0, puis s += k[i + r] × voisin, i croissant (même ordre qu'une boucle par pixel) ; intérieur
    // sans calcul d'indice de bord, boucles permutées (vectorisables)
    std::vector<float> tmp((std::size_t)w * h);
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
        {
            const float* row = src + (std::size_t)y * w;
            float* out = tmp.data() + (std::size_t)y * w;
            std::fill(out, out + w, 0.f);
            for (int i = -r; i <= r; i++)
            {
                const float kk = k[i + r];
                int x = 0;
                for (; x < std::min(w, r); x++)
                    out[x] += kk * row[reflect101(x + i, w)];
                for (; x < w - r; x++)
                    out[x] += kk * row[x + i];
                for (; x < w; x++)
                    out[x] += kk * row[reflect101(x + i, w)];
            }
        }
    });
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
        {
            float* out = dst + (std::size_t)y * w;
            std::fill(out, out + w, 0.f);
            for (int i = -r; i <= r; i++)
            {
                const float kk = k[i + r];
                const float* row = tmp.data() + (std::size_t)reflect101(y + i, h) * w;
                for (int x = 0; x < w; x++)
                    out[x] += kk * row[x];
            }
        }
    });
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
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
            for (int x = 0; x < dw; x++)
            {
                double s = 0.0;
                for (auto [j, c] : wx[x])
                    s += c * src[(std::size_t)y * w + j];
                rows[(std::size_t)y * dw + x] = s;
            }
    });
    parallel_for(dh, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
            for (int x = 0; x < dw; x++)
            {
                double s = 0.0;
                for (auto [j, c] : wy[y])
                    s += c * rows[(std::size_t)j * dw + x];
                dst[(std::size_t)y * dw + x] = (float)s;
            }
    });
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
    parallel_for(h, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++)
            for (int x = 0; x < w; x++)
            {
                const int ri = std::clamp(r, 1, std::max(1, h - 2)), xi = std::clamp(x, 1, std::max(1, w - 2));
                const float gx = std::fabs(sm[(std::size_t)ri * w + xi + 1] - sm[(std::size_t)ri * w + xi - 1]);
                const float gy = std::fabs(sm[(std::size_t)(ri + 1) * w + xi] - sm[(std::size_t)(ri - 1) * w + xi]);
                grad[(std::size_t)r * w + x] = gx + gy;
            }
    });
    std::vector<float> sub;
    sub.reserve(n / 16 + 1);
    for (std::size_t i = 0; i < n; i += 16)
        sub.push_back(grad[i]);
    const double thr = detail::percentile(std::move(sub), 30.0);
    float edge[GRAIN_BINS + 1];
    for (int i = 0; i <= GRAIN_BINS; i++)
        edge[i] = (float)bin_edge(i);
    // aplats répartis par tranche, par bandes de lignes en parallèle, puis réunis dans l'ordre des lignes (même ordre
    // que le parcours séquentiel : le sous-échantillonnage des tranches en dépend)
    const int bands = std::max(1, std::min(64, h / 16));
    std::vector<std::array<std::vector<float>, GRAIN_BINS>> part(bands);
    parallel_for(bands, [&](int b0, int b1) {
        for (int band = b0; band < b1; band++)
        {
            const std::size_t i0 = (std::size_t)((int64_t)h * band / bands) * w, i1 = (std::size_t)((int64_t)h * (band + 1) / bands) * w;
            for (std::size_t i = i0; i < i1; i++)
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
                part[band][b].push_back(t[i] - avg3[i]);
            }
        }
    });
    std::array<std::vector<float>, GRAIN_BINS> vals;
    for (int b = 0; b < GRAIN_BINS; b++)
        for (int band = 0; band < bands; band++)
            vals[b].insert(vals[b].end(), part[band][b].begin(), part[band][b].end());
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
    parallel_for(h, [&](int r0, int r1) {
    for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
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
    });
}

float residual_sigma(const uint16_t* src_y, std::ptrdiff_t src_stride, const float* out_y, std::ptrdiff_t out_stride, int w, int h)
{
    const std::size_t n = (std::size_t)w * h;
    std::vector<float> d(n), blur(n);
    parallel_for(h, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++)
            for (int x = 0; x < w; x++)
                d[(std::size_t)r * w + x] = (float)src_y[r * src_stride + x] - out_y[r * out_stride + x];
    });
    detail::gaussian_blur(d.data(), w, h, 1.2, blur.data());
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
            d[i] -= blur[i];
    });
    const float med = (float)median_exact(d);
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
            d[i] = std::fabs(d[i] - med);
    });
    return (float)std::max(1.4826 * median_exact(d), 1e-3);
}

namespace
{
// σ robuste du passe-haut d'un résidu déjà formé (d est modifié).
float sigma_of_residual(std::vector<float>& d, int w, int h)
{
    const std::size_t n = (std::size_t)w * h;
    std::vector<float> blur(n);
    detail::gaussian_blur(d.data(), w, h, 1.2, blur.data());
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
            d[i] -= blur[i];
    });
    const float med = (float)median_exact(d);
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
            d[i] = std::fabs(d[i] - med);
    });
    return (float)std::max(1.4826 * median_exact(d), 1e-3);
}

inline unsigned sample_at(const uint8_t* p, int bytes, std::size_t i)
{
    return bytes == 1 ? p[i] : (unsigned)(p[2 * i] | (p[2 * i + 1] << 8));
}
}  // namespace

float residual_sigma_fx(const uint8_t* src, int bytes_per_sample, const uint16_t* out_fx, float inv_fx, int w, int h)
{
    std::vector<float> d((std::size_t)w * h);
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
        {
            const float o = out_fx[i] * inv_fx;
            d[i] = (float)(uint16_t)sample_at(src, bytes_per_sample, i) - o;
        }
    });
    return sigma_of_residual(d, w, h);
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

GrainBins sigma_bins(const std::map<int, AnalysisFrame>& frames, const std::vector<int>& centers, int w, int h, double ar)
{
    static const int order[4] = {-2, -1, 1, 2};
    const std::size_t n = (std::size_t)w * h;
    std::array<std::vector<double>, GRAIN_BINS> acc;
    GrainBins out;
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
            {
                out.pixels[b] += (double)sel[b].size();
                acc[b].push_back(robust_sigma(std::move(sel[b])) * MEDIAN_CORR);
            }
    }
    for (int b = 0; b < GRAIN_BINS; b++)
        if (!acc[b].empty())
            out.sigma[b] = median_of(acc[b]);
    return out;
}

double flow_bias(int width) { return 0.91 - 0.15 * std::clamp((width - 1920) / 1920.0, 0.0, 1.0); }

std::optional<double> ratio_from_sigma(const GrainBins& bins, int width)
{
    double meas_sum = 0.0, common_s = 0.0, common_t = 0.0;
    int meas = 0, common = 0;
    std::vector<double> targets;
    std::vector<std::pair<double, double>> per_bin;  // (σ / profil, pixels)
    for (int b = 0; b < GRAIN_BINS; b++)
    {
        if (TARGET[b] > 0)
            targets.push_back(TARGET[b]);
        if (!bins.sigma[b])
            continue;
        meas_sum += *bins.sigma[b];
        meas++;
        if (TARGET[b] > 0)
        {
            common_s += *bins.sigma[b];
            common_t += TARGET[b];
            common++;
            per_bin.push_back({*bins.sigma[b] / TARGET[b], bins.pixels[b]});
        }
    }
    if (!meas)
        return std::nullopt;
    double r = common ? (common_s / common) / (common_t / common) : (meas_sum / meas) / median_of(targets);
    if (common)
    {
        std::stable_sort(per_bin.begin(), per_bin.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        double total = 0.0, cum = 0.0;
        for (const auto& p : per_bin)
            total += p.second;
        for (const auto& p : per_bin)
        {
            cum += p.second;
            if (cum >= total / 2)
            {
                r = std::min(r, RATIO_GUARD * p.first);
                break;
            }
        }
    }
    return std::min(r * flow_bias(width), 10.0);
}

double auto_part(std::optional<double> ratio, double k) { return ratio ? std::min(1.0, k / std::max(*ratio, 1e-3)) : 0.25 * k; }

void dose(const uint16_t* src, const float* out, std::size_t n, double kk, uint16_t lo, uint16_t hi, uint16_t* dst,
          long long dither_frame, std::size_t dither_offset, int dither_step)
{
    const float k = (float)kk;
    const float step = (float)std::max(1, dither_step);
    const int blocks = (int)((n + 65535) / 65536);
    parallel_for(blocks, [&](int b0, int b1) {
    for (std::size_t i = (std::size_t)b0 * 65536; i < std::min(n, (std::size_t)b1 * 65536); i++)
    {
        float v = out[i] + k * ((float)src[i] - out[i]);
        if (dither_frame >= 0)
        {
            const uint64_t z = mix64(((uint64_t)dither_frame << 40) ^ (uint64_t)(dither_offset + i));
            const float nz = (float)((z >> 40) * (1.0 / 16777216.0)) + (float)((z & 0xFFFFFFull) * (1.0 / 16777216.0)) - 1.f;
            if (step > 1.f)
            {
                // pas 8 bits : arrondi au multiple de `step` le plus proche, bornes ramenées au multiple intérieur
                const float q = std::clamp(std::nearbyint(v / step + nz), std::ceil(lo / step), std::floor(hi / step));
                dst[i] = (uint16_t)(q * step);
                continue;
            }
            v += nz;
        }
        dst[i] = (uint16_t)std::clamp(std::nearbyint(v), (float)lo, (float)hi);
    }
    });
}

namespace
{
// Flou binomial 3 × 3 ([1 2 1] séparable, / 16), bords répliqués.
void blur121(const float* src, int w, int h, float* dst)
{
    std::vector<float> tmp((std::size_t)w * h);
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
        {
            const float* r = src + (std::size_t)y * w;
            float* o = tmp.data() + (std::size_t)y * w;
            for (int x = 0; x < w; x++)
                o[x] = 0.25f * r[clampi(x - 1, w)] + 0.5f * r[x] + 0.25f * r[clampi(x + 1, w)];
        }
    });
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
        {
            const float* a = tmp.data() + (std::size_t)clampi(y - 1, h) * w;
            const float* c = tmp.data() + (std::size_t)y * w;
            const float* b2 = tmp.data() + (std::size_t)clampi(y + 1, h) * w;
            float* o = dst + (std::size_t)y * w;
            for (int x = 0; x < w; x++)
                o[x] = 0.25f * a[x] + 0.5f * c[x] + 0.25f * b2[x];
        }
    });
}

// Minimum et maximum 3 × 3 (séparables), bords répliqués.
void minmax3(const float* src, int w, int h, float* mn, float* mx)
{
    std::vector<float> tmn((std::size_t)w * h), tmx((std::size_t)w * h);
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
        {
            const float* r = src + (std::size_t)y * w;
            for (int x = 0; x < w; x++)
            {
                const float a = r[clampi(x - 1, w)], c = r[x], b2 = r[clampi(x + 1, w)];
                tmn[(std::size_t)y * w + x] = std::min(a, std::min(c, b2));
                tmx[(std::size_t)y * w + x] = std::max(a, std::max(c, b2));
            }
        }
    });
    parallel_for(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
        {
            const std::size_t ia = (std::size_t)clampi(y - 1, h) * w, ic = (std::size_t)y * w, ib = (std::size_t)clampi(y + 1, h) * w;
            for (int x = 0; x < w; x++)
            {
                mn[ic + x] = std::min(tmn[ia + x], std::min(tmn[ic + x], tmn[ib + x]));
                mx[ic + x] = std::max(tmx[ia + x], std::max(tmx[ic + x], tmx[ib + x]));
            }
        }
    });
}

// Contre-accentuage temporel d'un plan de luma dosé `d` (en place) : voir SharpenParams.
void contrasharpen(float* d, const uint8_t* src, int bytes_per_sample, const SharpenParams& sp)
{
    const int w = sp.width, h = sp.height;
    const std::size_t nl = (std::size_t)w * h;
    const float scale = std::ldexp(1.f, sp.bit_depth - 10);  // codes natifs par code 10 bits
    std::vector<float> ref(nl), s(nl), mn(nl), mx(nl), e(nl), mn2(nl), mx2(nl);
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
            ref[i] = sp.ref_fx[i] * sp.ref_inv;
    });
    // accentuation : écart au flou 3 × 3 ; bornes : écart à la référence sur 3 × 3
    blur121(d, w, h, s.data());
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
        {
            s[i] = d[i] - s[i];
            e[i] = ref[i] - d[i];
        }
    });
    minmax3(e.data(), w, h, mn.data(), mx.data());
    // seconde borne : ce que le débruitage a retiré (source − sortie dosée, 3 × 3, comme le ContraSharpening) ; sans
    // débruitage (part réinjectée 1), rien n'est rendu, même sur un contour où la référence s'écarte un peu de l'image
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
            e[i] = (float)sample_at(src, bytes_per_sample, i) - d[i];
    });
    minmax3(e.data(), w, h, mn2.data(), mx2.data());
    // texture de la référence : énergie locale (5 × 5) de son passe-haut
    blur121(ref.data(), w, h, e.data());
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
        {
            const float hf = ref[i] - e[i];
            e[i] = hf * hf;
        }
    });
    box_replicate(e.data(), w, h, 5, ref.data());  // ref : énergie moyenne 5 × 5 (référence plus utile ensuite)
    // seuil selon le grain attendu : courbe du plan à la luma de la source, σ × 1,25 / √5, au moins 0,25 code 10 bits
    const int levels = 1 << sp.bit_depth;
    std::vector<float> thr(levels);
    for (int v = 0; v < levels; v++)
    {
        const double y10 = v / (double)scale;
        double sg;
        const auto& c = sp.curve->center;
        const auto& g = sp.curve->sigma;
        if (y10 <= c[0])
            sg = std::max(g[0], 0.f);
        else if (y10 >= c[GRAIN_BINS - 1])
            sg = std::max(g[GRAIN_BINS - 1], 0.f);
        else
        {
            int b0 = 0;
            while (y10 > c[b0 + 1])
                b0++;
            const double t = (y10 - c[b0]) / (c[b0 + 1] - c[b0]);
            sg = std::max(g[b0], 0.f) + t * (std::max(g[b0 + 1], 0.f) - std::max(g[b0], 0.f));
        }
        thr[v] = (float)(std::max(sg * 1.25 / std::sqrt(5.0), 0.25) * scale);
    }
    const float k = (float)sp.strength;
    parallel_for(h, [&](int r0, int r1) {
        for (std::size_t i = (std::size_t)r0 * w; i < (std::size_t)r1 * w; i++)
        {
            // intersection des deux bornes (détail stable dans le temps et retiré par le débruitage) ; disjointes : rien
            const float lo = std::max(mn[i], mn2[i]), hi = std::min(mx[i], mx2[i]);
            const float lim = lo > hi ? 0.f : std::min(std::max(s[i], lo), hi);
            const float ss = std::fabs(lim) < std::fabs(s[i]) ? lim : s[i];
            const float t = thr[std::min<unsigned>(sample_at(src, bytes_per_sample, i), (unsigned)levels - 1)];
            const float mask = std::clamp((std::sqrt(ref[i]) - t) / t, 0.f, 1.f);
            d[i] += k * ss * mask;
        }
    });
}
}  // namespace

void dose_frame(const uint8_t* src, int bytes_per_sample, const uint16_t* out_fx, float inv_fx, std::size_t n, double kk,
                uint16_t lo, uint16_t hi, uint8_t* dst, long long dither_frame, int dither_step, const SharpenParams* sharpen)
{
    // mêmes opérations flottantes que dose() sur of = out_fx × inv_fx et la source en codes ; contre-accentuage de la
    // luma (facultatif) entre le dosage et l'arrondi
    const float k = (float)kk;
    const float step = (float)std::max(1, dither_step);
    const bool sharpening = sharpen && sharpen->strength > 0.0 && sharpen->ref_fx && sharpen->curve;
    const std::size_t nl = sharpening ? (std::size_t)sharpen->width * sharpen->height : 0;
    std::vector<float> luma(nl);
    if (sharpening)
    {
        parallel_for(sharpen->height, [&](int r0, int r1) {
            for (std::size_t i = (std::size_t)r0 * sharpen->width; i < (std::size_t)r1 * sharpen->width; i++)
            {
                const float o = out_fx[i] * inv_fx;
                luma[i] = o + k * ((float)(uint16_t)sample_at(src, bytes_per_sample, i) - o);
            }
        });
        contrasharpen(luma.data(), src, bytes_per_sample, *sharpen);
    }
    const int blocks = (int)((n + 65535) / 65536);
    parallel_for(blocks, [&](int b0, int b1) {
    for (std::size_t i = (std::size_t)b0 * 65536; i < std::min(n, (std::size_t)b1 * 65536); i++)
    {
        float v;
        if (i < nl)
            v = luma[i];
        else
        {
            const float o = out_fx[i] * inv_fx;
            v = o + k * ((float)(uint16_t)sample_at(src, bytes_per_sample, i) - o);
        }
        uint16_t q;
        if (dither_frame >= 0)
        {
            const uint64_t z = mix64(((uint64_t)dither_frame << 40) ^ (uint64_t)i);
            const float nz = (float)((z >> 40) * (1.0 / 16777216.0)) + (float)((z & 0xFFFFFFull) * (1.0 / 16777216.0)) - 1.f;
            if (step > 1.f)
            {
                const float qs = std::clamp(std::nearbyint(v / step + nz), std::ceil(lo / step), std::floor(hi / step));
                q = (uint16_t)(qs * step);
            }
            else
            {
                v += nz;
                q = (uint16_t)std::clamp(std::nearbyint(v), (float)lo, (float)hi);
            }
        }
        else
            q = (uint16_t)std::clamp(std::nearbyint(v), (float)lo, (float)hi);
        if (bytes_per_sample == 1)
            dst[i] = (uint8_t)q;
        else
        {
            dst[2 * i] = (uint8_t)(q & 0xff);
            dst[2 * i + 1] = (uint8_t)(q >> 8);
        }
    }
    });
}

void set_threads(int n) { g_threads = std::max(0, n); }

}  // namespace restore_auto
