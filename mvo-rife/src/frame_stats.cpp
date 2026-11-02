// Statistiques de chaque trame produite (voir frame_stats.h).
#include "frame_stats.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <mutex>

#include "restore_auto.h"

namespace
{
// EOTF PQ (SMPTE ST 2084) en double : signal normalisé → nits.
double pq(double e)
{
    const double m1 = 2610.0 / 16384, m2 = 2523.0 / 4096 * 128, c1 = 3424.0 / 4096, c2 = 2413.0 / 4096 * 32,
                 c3 = 2392.0 / 4096 * 32;
    const double p = std::pow(e, 1 / m2), n = p - c1 > 0 ? p - c1 : 0;
    return 10000 * std::pow(n / (c2 - c3 * p), 1 / m1);
}

// Échantillon i d'un plan (8 bits, ou 16 bits petit-boutiste).
inline uint32_t sample(const uint8_t* plane, size_t i, int bytes)
{
    return bytes == 1 ? plane[i] : (uint32_t)plane[2 * i] | ((uint32_t)plane[2 * i + 1] << 8);
}

// Interpolation d'un axe sous-échantillonné : indices voisins et poids du second, pour chaque position de luma.
struct Tap
{
    int a = 0, b = 0;
    float w = 0.f;
};

std::vector<Tap> taps(int luma, int chroma, int sub, bool cosited)
{
    std::vector<Tap> out((size_t)luma);
    for (int x = 0; x < luma; x++)
    {
        const double pos = sub == 1 ? x : (cosited ? x / 2.0 : (x - 0.5) / 2.0);
        Tap t;
        if (pos <= 0)
            t = Tap{0, 0, 0.f};
        else
        {
            t.a = std::min((int)pos, chroma - 1);
            t.b = std::min(t.a + 1, chroma - 1);
            t.w = (float)(pos - std::floor(pos));
        }
        out[(size_t)x] = t;
    }
    return out;
}

inline uint32_t to_code(float v)
{
    return (uint32_t)(std::clamp(v, 0.f, 1.f) * 65535.f + 0.5f);
}
}  // namespace

FrameStatsWriter::~FrameStatsWriter()
{
    if (file)
        fclose(file);
}

bool FrameStatsWriter::open(const std::string& path, const FrameFormat& f, const ColorParams& c, std::string& error)
{
    fmt = f;
    color = c;
    eotf.resize(65536);
    for (int k = 0; k < 65536; k++)
        eotf[(size_t)k] = (float)pq(k / 65535.0);
    edges.resize(BINS + 1);
    for (int i = 0; i <= BINS; i++)
        edges[(size_t)i] = i < BINS ? eotf[(size_t)64 * i] : 10000.f;
    file = fopen(path.c_str(), "wb");
    if (!file)
    {
        error = "statistiques des trames : ouverture de " + path + " impossible (" + strerror(errno) + ")";
        return false;
    }
    char head[160];
    snprintf(head, sizeof(head), "{\"format\":\"mvo-fel-frame-statistics\",\"version\":1,\"bins\":%d,\"width\":%d,\"height\":%d}\n",
             BINS, fmt.width, fmt.height);
    return write(head, error);
}

void FrameStatsWriter::measure(const uint8_t* frame, Frame& out) const
{
    const int w = fmt.width, h = fmt.height, cw = fmt.chroma_width(), ch = fmt.chroma_height();
    const int bytes = fmt.bytes_per_sample, depth = fmt.bit_depth;
    const uint8_t* py = frame;
    const uint8_t* pu = py + (size_t)fmt.luma_samples() * bytes;
    const uint8_t* pv = pu + (size_t)fmt.chroma_samples() * bytes;
    // Normalisation Y'CbCr → [0, 1] / [−0,5, 0,5]
    float y_off, y_scale, c_off, c_scale;
    if (color.full_range)
    {
        const float maxv = (float)((1 << depth) - 1);
        y_off = 0.f;
        y_scale = 1.f / maxv;
        c_off = (float)(1 << (depth - 1));
        c_scale = 1.f / maxv;
    }
    else
    {
        const float s = (float)(1 << (depth - 8));
        y_off = 16.f * s;
        y_scale = 1.f / (219.f * s);
        c_off = 128.f * s;
        c_scale = 1.f / (224.f * s);
    }
    const float kr = color.kr, kb = color.kb, kg = 1.f - kr - kb;
    const float cr_r = 2.f * (1.f - kr), cb_b = 2.f * (1.f - kb);
    const bool cosited_x = color.siting != ChromaSiting::Center;
    const bool cosited_y = color.siting == ChromaSiting::TopLeft;
    const std::vector<Tap> tx = taps(w, cw, fmt.sub_x, cosited_x);
    const std::vector<Tap> ty = taps(h, ch, fmt.sub_y, cosited_y);
    const float* table = eotf.data();
    const float* bounds = edges.data();
    std::vector<float> sums((size_t)h);  // somme de maxRGB de chaque ligne

    out = Frame();
    std::mutex merge;
    restore_auto::detail::for_rows(h, [&](int y0, int y1) {
        std::array<uint32_t, BINS> hm{}, hy{};
        std::array<uint32_t, 3> mx{};
        uint64_t le = 0;
        std::vector<float> urow((size_t)cw), vrow((size_t)cw);
        for (int y = y0; y < y1; y++)
        {
            const Tap& t = ty[(size_t)y];
            for (int j = 0; j < cw; j++)
            {
                const size_t a = (size_t)t.a * cw + j, b = (size_t)t.b * cw + j;
                const float ua = (float)sample(pu, a, bytes), ub = (float)sample(pu, b, bytes);
                const float va = (float)sample(pv, a, bytes), vb = (float)sample(pv, b, bytes);
                urow[(size_t)j] = ua + (ub - ua) * t.w;
                vrow[(size_t)j] = va + (vb - va) * t.w;
            }
            float sum = 0.f;
            for (int x = 0; x < w; x++)
            {
                const Tap& s = tx[(size_t)x];
                const float u = urow[(size_t)s.a] + (urow[(size_t)s.b] - urow[(size_t)s.a]) * s.w;
                const float v = vrow[(size_t)s.a] + (vrow[(size_t)s.b] - vrow[(size_t)s.a]) * s.w;
                const float yy = ((float)sample(py, (size_t)y * w + x, bytes) - y_off) * y_scale;
                const float cb = (u - c_off) * c_scale, cr = (v - c_off) * c_scale;
                const float r = yy + cr_r * cr, b = yy + cb_b * cb, g = (yy - kr * r - kb * b) / kg;
                const uint32_t qr = to_code(r), qg = to_code(g), qb = to_code(b);
                const uint32_t m = std::max(qr, std::max(qg, qb));
                mx[0] = std::max(mx[0], qr);
                mx[1] = std::max(mx[1], qg);
                mx[2] = std::max(mx[2], qb);
                hm[m >> 6]++;
                sum += table[m];
                const float lum = 0.2627f * table[qr] + 0.678f * table[qg] + 0.0593f * table[qb];
                le += lum <= 100.f;
                // plus grande borne ≤ Y
                uint32_t lo = 0, hi = BINS;
                while (hi - lo > 1)
                {
                    const uint32_t mid = (lo + hi) >> 1;
                    if (bounds[mid] <= lum)
                        lo = mid;
                    else
                        hi = mid;
                }
                hy[lo]++;
            }
            sums[(size_t)y] = sum;
        }
        std::lock_guard<std::mutex> lock(merge);
        for (int k = 0; k < BINS; k++)
        {
            out.hist_maxrgb[(size_t)k] += hm[(size_t)k];
            out.hist_y[(size_t)k] += hy[(size_t)k];
        }
        for (int c = 0; c < 3; c++)
            out.max_code[(size_t)c] = std::max(out.max_code[(size_t)c], mx[(size_t)c]);
        out.y_le_100 += le;
    });
    for (int y = 0; y < h; y++)  // lignes dans l'ordre : somme déterministe
        out.sum_maxrgb += sums[(size_t)y];
    out.pixels = (uint64_t)w * (uint64_t)h;
}

bool FrameStatsWriter::add(const uint8_t* frame, bool scene_start, std::string& error)
{
    Frame s;
    measure(frame, s);
    if ((scene_start || scene_first < 0) && !flush_scene(error))
        return false;
    if (scene_first < 0)
        scene_first = frame_index;
    scene_frames++;
    for (int k = 0; k < BINS; k++)
        scene_hist[(size_t)k] += s.hist_maxrgb[(size_t)k];
    // Y à 99,99 % : borne haute de la case atteinte
    const uint64_t target = s.pixels - s.pixels / 10000;
    uint64_t total = 0;
    int bin = BINS - 1;
    for (int k = 0; k < BINS; k++)
    {
        total += s.hist_y[(size_t)k];
        if (total >= target)
        {
            bin = k;
            break;
        }
    }
    // percentiles de maxRGB (1, 25, 50, 75, 90, 95, 99,98 %) : centre de la case atteinte
    static constexpr double quantiles[] = {0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.9998};
    std::string line;
    char buf[128];
    snprintf(buf, sizeof(buf), "{\"frame\":%lld,\"scene_start\":%s,\"max_nits\":[%.9g,%.9g,%.9g]", (long long)frame_index,
             scene_frames == 1 ? "true" : "false", eotf[s.max_code[0]], eotf[s.max_code[1]], eotf[s.max_code[2]]);
    line += buf;
    snprintf(buf, sizeof(buf), ",\"avg_nits\":%.9g,\"y9999_nits\":%.9g,\"y100\":%.9g,\"maxrgb_percentiles\":[",
             s.pixels ? s.sum_maxrgb / (double)s.pixels : 0.0, edges[(size_t)bin + 1],
             s.pixels ? (double)s.y_le_100 / (double)s.pixels : 0.0);
    line += buf;
    for (size_t q = 0; q < sizeof(quantiles) / sizeof(quantiles[0]); q++)
    {
        const double wanted = quantiles[q] * (double)s.pixels;
        uint64_t cumulative = 0;
        int index = BINS - 1;
        for (int k = 0; k < BINS; k++)
        {
            cumulative += s.hist_maxrgb[(size_t)k];
            if ((double)cumulative >= wanted)
            {
                index = k;
                break;
            }
        }
        snprintf(buf, sizeof(buf), "%s%.9g", q ? "," : "", eotf[(size_t)index * 64 + 32]);
        line += buf;
    }
    snprintf(buf, sizeof(buf), "],\"window\":[0,0,%d,%d]}\n", fmt.width, fmt.height);
    line += buf;
    frame_index++;
    return write(line, error);
}

bool FrameStatsWriter::flush_scene(std::string& error)
{
    if (scene_first < 0 || !scene_frames)
        return true;
    std::string line = "{\"scene_first\":" + std::to_string(scene_first) + ",\"frames\":" + std::to_string(scene_frames)
                       + ",\"maxrgb_hist\":[";
    bool first = true;
    for (int k = 0; k < BINS; k++)
        if (scene_hist[(size_t)k])
        {
            line += (first ? "[" : ",[") + std::to_string(k) + "," + std::to_string(scene_hist[(size_t)k]) + "]";
            first = false;
        }
    line += "]}\n";
    scene_hist.fill(0);
    scene_first = -1;
    scene_frames = 0;
    return write(line, error);
}

bool FrameStatsWriter::finish(std::string& error)
{
    if (!file)
        return true;
    bool ok = flush_scene(error);
    if (fclose(file) != 0 && ok)
    {
        error = "statistiques des trames : écriture impossible";
        ok = false;
    }
    file = nullptr;
    return ok;
}

bool FrameStatsWriter::write(const std::string& line, std::string& error)
{
    if (fwrite(line.data(), 1, line.size(), file) != line.size())
    {
        error = "statistiques des trames : écriture impossible";
        return false;
    }
    return true;
}
