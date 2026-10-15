// SPDX-License-Identifier: LGPL-2.1-or-later
// Adaptation CPU : libplacebo, colorspace.c, shaders/colorspace.c, filters.c
// et utils/libav_internal.h. Révision et attributions dans NOTICES.md.
#include "reconstruction.hpp"
#include "blue_noise.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MVO_FEL_AVX2 1
#endif
extern "C" {
#include <libavutil/pixdesc.h>
}

namespace muxiveo::fel {
namespace {
constexpr double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
constexpr double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
const Mat lms_to_rgb{{{3.06441879, -2.16597676, 0.10155818},
                      {-0.65612108, 1.78554118, -0.12943749},
                      {0.01736321, -0.04725154, 1.03004253}}};
Vec multiply(const Mat &m, const Vec &v) {
    Vec out{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i] += m[i][j] * v[j];
    return out;
}
double sign(double x) { return (x > 0) - (x < 0); }
double dot4_scalar(const double *a, const double *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}
#ifdef MVO_FEL_AVX2
__attribute__((target("avx2"))) double dot4_avx2(const double *a, const double *b) {
    const auto product = _mm256_mul_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
    double lanes[4];
    _mm256_storeu_pd(lanes, product);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif
using Dot4 = double (*)(const double *, const double *);
Dot4 select_dot4() {
#ifdef MVO_FEL_AVX2
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) return dot4_avx2;
#endif
    return dot4_scalar;
}
}

Mapping::Mapping(const AVDOVIMetadata &metadata) {
    const auto &h = *av_dovi_get_header(&metadata);
    const auto &m = *av_dovi_get_mapping(&metadata);
    const auto &c = *av_dovi_get_color(&metadata);
    if (h.bl_bit_depth < 8 || h.bl_bit_depth > 16 || h.el_bit_depth < 8 ||
        h.el_bit_depth > 16 || h.coef_log2_denom > 32)
        throw std::runtime_error("Précision RPU invalide");
    if (m.num_x_partitions > 1 || m.num_y_partitions > 1)
        throw std::runtime_error("Partitionnement spatial RPU non pris en charge");
    const double scale = std::ldexp(1.0, -h.coef_log2_denom);
    const double bl_max = (1u << h.bl_bit_depth) - 1;
    const double el_max = (1u << h.el_bit_depth) - 1;
    residual = !h.disable_residual_flag;
    if (residual && m.nlq_method_idc != AV_DOVI_NLQ_LINEAR_DZ)
        throw std::runtime_error("Méthode NLQ non prise en charge");
    for (int i = 0; i < 3; ++i) {
        nonlinear_offset[i] = av_q2d(c.ycc_to_rgb_offset[i]) * (65536.0 / 65535.0);
        for (int j = 0; j < 3; ++j) {
            nonlinear[i][j] = av_q2d(c.ycc_to_rgb_matrix[3*i+j]);
            linear[i][j] = av_q2d(c.rgb_to_lms_matrix[3*i+j]);
            if (!std::isfinite(nonlinear[i][j]) || !std::isfinite(linear[i][j]))
                throw std::runtime_error("Matrice RPU invalide");
        }
        offset[i] = m.nlq[i].nlq_offset / el_max;
        slope[i] = m.nlq[i].linear_deadzone_slope * el_max * scale;
        threshold[i] = (double(m.nlq[i].linear_deadzone_threshold) -
                        0.5 * m.nlq[i].linear_deadzone_slope) * scale;
        const auto &src = m.curves[i];
        auto &dst = curves[i];
        dst.count = src.num_pivots;
        if (dst.count < 2 || dst.count > 9)
            throw std::runtime_error("Nombre de pivots RPU invalide");
        for (unsigned p = 0; p < dst.count; ++p) {
            dst.pivots[p] = src.pivots[p] / bl_max;
            if (p && dst.pivots[p] <= dst.pivots[p-1])
                throw std::runtime_error("Pivots RPU non croissants");
        }
        for (unsigned p = 0; p + 1 < dst.count; ++p) {
            dst.method[p] = src.mapping_idc[p];
            if (src.mapping_idc[p] == AV_DOVI_MAPPING_POLYNOMIAL) {
                if (src.poly_order[p] > 2) throw std::runtime_error("Ordre polynomial RPU invalide");
                for (unsigned k = 0; k <= src.poly_order[p]; ++k)
                    dst.poly[p][k] = src.poly_coef[p][k] * scale;
            } else if (src.mapping_idc[p] == AV_DOVI_MAPPING_MMR) {
                dst.order[p] = src.mmr_order[p];
                if (dst.order[p] < 1 || dst.order[p] > 3)
                    throw std::runtime_error("Ordre MMR RPU invalide");
                dst.constant[p] = src.mmr_constant[p] * scale;
                for (unsigned k = 0; k < dst.order[p]; ++k)
                    for (int j = 0; j < 7; ++j) dst.mmr[p][k][j] = src.mmr_coef[p][k][j] * scale;
            } else throw std::runtime_error("Mapping RPU inconnu");
        }
    }
}

double spline16(double x) {
    x = std::abs(x);
    if (x >= 2) return 0;
    if (x < 1) return ((x - 9.0/5.0) * x - 1.0/5.0) * x + 1;
    x -= 1;
    return ((-x/3 + 4.0/5.0) * x - 7.0/15.0) * x;
}
double pq_eotf(double x) {
    const double p = std::pow(std::max(x, 0.0), 1/m2);
    const double denominator = c2 - c3*p;
    if (denominator <= 0) throw std::runtime_error("Signal PQ hors domaine");
    return std::pow(std::max(p-c1, 0.0)/denominator, 1/m1);
}
double pq_oetf(double x) {
    const double p = std::pow(std::max(x, 0.0), m1);
    return std::pow((c1+c2*p)/(1+c3*p), m2);
}
Vec reshape(const Mapping &mapping, const Vec &bl) {
    Vec sig{}, out{};
    for (int c = 0; c < 3; ++c) sig[c] = std::clamp(bl[c], 0.0, 1.0);
    const std::array<double, 7> terms{sig[0], sig[1], sig[2], sig[0]*sig[1],
        sig[0]*sig[2], sig[1]*sig[2], sig[0]*sig[1]*sig[2]};
    for (int c = 0; c < 3; ++c) {
        const auto &curve = mapping.curves[c];
        unsigned p = 0;
        while (p+2 < curve.count && sig[c] >= curve.pivots[p+1]) ++p;
        double s;
        if (curve.method[p] == AV_DOVI_MAPPING_POLYNOMIAL) {
            const auto &a = curve.poly[p];
            s = (a[2]*sig[c] + a[1])*sig[c] + a[0];
        } else {
            s = curve.constant[p];
            auto powers = terms;
            for (unsigned k = 0; k < curve.order[p]; ++k) {
                for (int j = 0; j < 7; ++j) {
                    s += curve.mmr[p][k][j]*powers[j];
                    powers[j] *= terms[j];
                }
            }
        }
        out[c] = std::clamp(s, curve.pivots[0], curve.pivots[curve.count-1]);
    }
    return out;
}
Vec compose(const Mapping &mapping, const Vec &bl, const Vec &el) {
    Vec out = reshape(mapping, bl);
    if (mapping.residual) for (int c = 0; c < 3; ++c) {
        const double d = el[c] - mapping.offset[c];
        // Spline16 peut déplacer un résiduel exactement neutre de quelques ulp.
        // Le saut NLQ sign(d)*threshold ne doit pas amplifier ce bruit d'arrondi.
        if (std::abs(d) <= 32*std::numeric_limits<double>::epsilon()) continue;
        out[c] += sign(d)*(std::abs(d)*mapping.slope[c] + mapping.threshold[c]);
    }
    return out;
}
Vec to_rgb(const Mapping &mapping, const Vec &signal) {
    Vec value = signal;
    for (int c = 0; c < 3; ++c) value[c] -= mapping.nonlinear_offset[c];
    value = multiply(mapping.nonlinear, value);
    for (auto &c : value) c = pq_eotf(c);
    value = multiply(lms_to_rgb, multiply(mapping.linear, value));
    for (auto &c : value) c = pq_oetf(c);
    return value;
}

double sample_plane(const AVFrame &f, int c, int x, int y, int bw, int bh, bool enhancement) {
    const auto *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(f.format));
    if (!desc || desc->nb_components != 3 || !(desc->flags & AV_PIX_FMT_FLAG_PLANAR) ||
        (desc->flags & (AV_PIX_FMT_FLAG_BE | AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_FLOAT)) ||
        desc->comp[c].depth > 16 || desc->comp[c].depth < 8)
        throw std::runtime_error("Format HEVC décodé non pris en charge");
    const int hs = c ? desc->log2_chroma_w : 0, vs = c ? desc->log2_chroma_h : 0;
    const int w = (f.width + (1 << hs)-1) >> hs, h = (f.height + (1 << vs)-1) >> vs;
    // EL : horizontal cosité, vertical centré ; chroma suivant la VUI.
    double layer_x = enhancement && f.width < bw ? -0.5 : 0;
    double layer_y = 0;
    double chroma_x = 0, chroma_y = 0;
    if (c) {
        switch (f.chroma_location) {
        case AVCHROMA_LOC_TOPLEFT: chroma_x = -0.5; chroma_y = -0.5; break;
        case AVCHROMA_LOC_LEFT: chroma_x = -0.5; break;
        case AVCHROMA_LOC_CENTER: break;
        case AVCHROMA_LOC_TOP: chroma_y = -0.5; break;
        case AVCHROMA_LOC_BOTTOMLEFT: chroma_x = -0.5; chroma_y = 0.5; break;
        case AVCHROMA_LOC_BOTTOM: chroma_y = 0.5; break;
        default: throw std::runtime_error("Position chromatique absente");
        }
    }
    const double rx = double(f.width)/bw, ry = double(f.height)/bh;
    const double px = ((x + 0.5 - layer_x)*rx - chroma_x)/(1 << hs) - 0.5;
    const double py = ((y + 0.5 - layer_y)*ry - chroma_y)/(1 << vs) - 0.5;
    const int ix = static_cast<int>(std::floor(px)), iy = static_cast<int>(std::floor(py));
    double result = 0, total = 0;
    const auto &comp = desc->comp[c];
    for (int j = iy-1; j <= iy+2; ++j) {
        const auto *row = f.data[comp.plane] + std::clamp(j, 0, h-1)*f.linesize[comp.plane];
        for (int i = ix-1; i <= ix+2; ++i) {
            const double weight = spline16(px-i)*spline16(py-j);
            const auto *p = row + std::clamp(i, 0, w-1)*comp.step + comp.offset;
            const unsigned v = comp.depth > 8 ? unsigned(p[0]) | (unsigned(p[1]) << 8) : p[0];
            result += weight*(v >> comp.shift);
            total += weight;
        }
    }
    return result/(total*((1u << comp.depth)-1));
}
FrameSampler::FrameSampler(const AVFrame &f, int bw, int bh, bool enhancement) {
    const auto *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(f.format));
    if (!desc || desc->nb_components != 3 || !(desc->flags & AV_PIX_FMT_FLAG_PLANAR) ||
        (desc->flags & (AV_PIX_FMT_FLAG_BE | AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_FLOAT)))
        throw std::runtime_error("Format HEVC décodé non pris en charge");
    for (int c = 0; c < 3; ++c) {
        const auto &comp = desc->comp[c];
        if (comp.depth > 16 || comp.depth < 8)
            throw std::runtime_error("Précision HEVC décodée non prise en charge");
        const int hs = c ? desc->log2_chroma_w : 0, vs = c ? desc->log2_chroma_h : 0;
        auto &plane = planes[c];
        plane.data = f.data[comp.plane]; plane.stride = f.linesize[comp.plane];
        plane.step = comp.step; plane.offset = comp.offset; plane.shift = comp.shift; plane.depth = comp.depth;
        double cx = 0, cy = 0;
        if (c) switch (f.chroma_location) {
            case AVCHROMA_LOC_TOPLEFT: cx = -.5; cy = -.5; break;
            case AVCHROMA_LOC_LEFT: cx = -.5; break;
            case AVCHROMA_LOC_CENTER: break;
            case AVCHROMA_LOC_TOP: cy = -.5; break;
            case AVCHROMA_LOC_BOTTOMLEFT: cx = -.5; cy = .5; break;
            case AVCHROMA_LOC_BOTTOM: cy = .5; break;
            default: throw std::runtime_error("Position chromatique absente");
        }
        auto axis = [](int count, int size, double ratio, int subsample, double layer, double chroma) {
            std::vector<Axis> result(count);
            for (int i = 0; i < count; ++i) {
                const double coordinate = ((i+.5-layer)*ratio-chroma)/(1 << subsample)-.5;
                const int first = static_cast<int>(std::floor(coordinate))-1;
                double sum = 0;
                for (int k = 0; k < 4; ++k) {
                    result[i].indices[k] = std::clamp(first+k, 0, size-1);
                    sum += result[i].weights[k] = spline16(coordinate-first-k);
                }
                for (auto &w : result[i].weights) w /= sum;
            }
            return result;
        };
        plane.horizontal = axis(bw, (f.width+(1 << hs)-1) >> hs, double(f.width)/bw,
                                hs, enhancement && f.width < bw ? -.5 : 0, cx);
        plane.vertical = axis(bh, (f.height+(1 << vs)-1) >> vs, double(f.height)/bh, vs, 0, cy);
    }
}
double FrameSampler::sample(int component, int x, int y) const {
    // Initialisation thread-safe, sans imposer AVX2 aux machines plus anciennes.
    static const Dot4 dot4 = select_dot4();
    const auto &p = planes[component];
    const auto &h = p.horizontal[x], &v = p.vertical[y];
    double rows[4];
    for (int j = 0; j < 4; ++j) {
        const auto *row = p.data + v.indices[j]*p.stride;
        double codes[4];
        for (int i = 0; i < 4; ++i) {
            const auto *pixel = row + h.indices[i]*p.step+p.offset;
            const unsigned code = p.depth > 8 ? pixel[0] + 256u*pixel[1] : pixel[0];
            codes[i] = code >> p.shift;
        }
        rows[j] = dot4(codes, h.weights.data());
    }
    return dot4(rows, v.weights.data())/((1u << p.depth)-1);
}
void reconstruct_rows(const FrameSampler &bl, const FrameSampler &el, const Mapping &mapping,
                      AVFrame &rgb, int first_row, int last_row) {
    for (int y = first_row; y < last_row; ++y) for (int x = 0; x < rgb.width; ++x) {
        Vec base{}, enhancement{};
        for (int c = 0; c < 3; ++c) {
            base[c] = bl.sample(c, x, y);
            enhancement[c] = el.sample(c, x, y);
        }
        const auto value = to_rgb(mapping, compose(mapping, base, enhancement));
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(value[c])) throw std::runtime_error("Pixel reconstruit non fini");
            const int plane = c == 0 ? 2 : c == 1 ? 0 : 1;
            auto *p = rgb.data[plane] + y*rgb.linesize[plane] + 2*x;
            const auto code = static_cast<unsigned>(std::lround(std::clamp(value[c], 0.0, 1.0)*65535));
            p[0] = code & 255; p[1] = code >> 8;
        }
    }
}
void reconstruct_rows(const AVFrame &bl, const AVFrame &el, const Mapping &mapping,
                      AVFrame &rgb, int first_row, int last_row) {
    reconstruct_rows(FrameSampler(bl, bl.width, bl.height, false),
                     FrameSampler(el, bl.width, bl.height, true), mapping, rgb, first_row, last_row);
}
namespace {
constexpr double kr = 0.2627, kb = 0.0593, kg = 1 - kr - kb;
double code(const AVFrame &rgb, int plane, int x, int y) {
    const auto *p = rgb.data[plane] + y*rgb.linesize[plane] + 2*x;
    return (p[0] | (p[1] << 8)) / 65535.0;
}
// gbrp16le : plan 0 = G, 1 = B, 2 = R.
Vec rgb_at(const AVFrame &rgb, int x, int y) {
    x = std::clamp(x, 0, rgb.width-1); y = std::clamp(y, 0, rgb.height-1);
    return {code(rgb, 2, x, y), code(rgb, 0, x, y), code(rgb, 1, x, y)};
}
void store(AVFrame &yuv, int plane, int x, int y, unsigned value, int depth) {
    auto *p = yuv.data[plane] + y*yuv.linesize[plane];
    if (depth == 8) { p[x] = static_cast<uint8_t>(value); return; }
    p[2*x] = value & 255; p[2*x+1] = static_cast<uint8_t>(value >> 8);
}
unsigned quantize(double value, double threshold, int depth) {
    const double maximum = (1u << depth) - 1;
    return static_cast<unsigned>(std::clamp(std::floor(value + threshold), 0.0, maximum));
}
// Poids bilinéaires (triangle étiré ×2) autour du centre 2i+offset.
void taps(double offset, std::array<double, 4> &weights, int &first) {
    first = offset >= 1 ? 0 : -1;
    for (int k = 0; k < 4; ++k) {
        const double d = std::abs(first + k - offset);
        weights[k] = std::max(0.0, 1 - d/2) / 2;
    }
}
}
}

namespace muxiveo::fel {
namespace {
int exact_weights(const double *w, std::array<int32_t, 4> &numerators) {
    static const int divisors[] = {1,2,4,5,8,10,16,20,32,40,64,80,128,160,256,320,512,640,1280,2560};
    for (int d : divisors) {
        bool exact = true;
        int sum = 0;
        for (int k = 0; k < 4 && exact; ++k) {
            const double scaled = w[k]*d;
            numerators[k] = static_cast<int32_t>(std::lround(scaled));
            exact = std::abs(scaled - numerators[k]) < 1e-9;
            sum += numerators[k];
        }
        if (exact && sum == d) return d;
    }
    return 0;
}
}

namespace {
// log2/exp2 double sans branche ni appel (erreur relative ≈ 1e-15) : boucles de lignes vectorisées
// par le compilateur (SSE2/NEON, AVX2+FMA choisi à l'exécution sur x86-64). Le float32 ne suffit pas :
// la matrice LMS→RGB peut compenser des termes de l'ordre de 0,4 jusqu'à 1e-5 (couleurs saturées
// lumineuses) et la PQ amplifie ensuite fortement toute erreur absolue près de zéro.
// Conversions entier ↔ double par constantes magiques : AVX2 n'a pas de conversion 64 bits.
[[gnu::always_inline]] inline double fast_log2(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, 8);
    // Exposant non biaisé : 2^52 + e lu comme double, moins 2^52 + 1023.
    const uint64_t exponent_bits = ((bits >> 52) & 0x7ff) | 0x4330000000000000u;
    double exponent;
    std::memcpy(&exponent, &exponent_bits, 8);
    exponent -= 4503599627371519.0;
    bits = (bits & 0x000fffffffffffffu) | 0x3ff0000000000000u;
    double m;
    std::memcpy(&m, &bits, 8);
    // m ∈ [√2/2, √2) : s = (m-1)/(m+1) ∈ ±0,1716, série atanh jusqu'à s^19 (reste < 1e-17).
    const bool high = m > 1.4142135623730951;
    m = high ? 0.5*m : m;
    exponent += high ? 1.0 : 0.0;
    const double s = (m - 1.0)/(m + 1.0), s2 = s*s;
    constexpr double k = 2.8853900817779268;  // 2/ln 2 : log2(m) = k·atanh(s)
    return exponent + s*k*(1 + s2*(1.0/3 + s2*(1.0/5 + s2*(1.0/7 + s2*(1.0/9 + s2*(1.0/11 + s2*(1.0/13 +
        s2*(1.0/15 + s2*(1.0/17 + s2*(1.0/19))))))))));
}
[[gnu::always_inline]] inline double fast_exp2(double x) {
    x = std::min(std::max(x, -1022.0), 1023.0);
    // Arrondi au plus proche par constante magique (1,5 × 2^52) : n figure dans les bits bas de t.
    const double t = x + 6755399441055744.0, n = t - 6755399441055744.0, f = (x - n)*0.69314718055994531;
    // e^f, |f| ≤ 0,347 : Taylor jusqu'à f^13 (reste < 1e-17).
    const double p = 1.0 + f*(1.0 + f*(1.0/2 + f*(1.0/6 + f*(1.0/24 + f*(1.0/120 + f*(1.0/720 + f*(1.0/5040 +
        f*(1.0/40320 + f*(1.0/362880 + f*(1.0/3628800 + f*(1.0/39916800 + f*(1.0/479001600 +
        f*(1.0/6227020800)))))))))))));
    uint64_t bits;
    std::memcpy(&bits, &t, 8);
    bits = (bits + 1023) << 52;
    double scale;
    std::memcpy(&scale, &bits, 8);
    return p*scale;
}
// x <= 0 donne 0 (sélection, sans branche).
[[gnu::always_inline]] inline double fast_pow(double x, double y) {
    const double value = fast_exp2(y*fast_log2(std::max(x, 1e-300)));
    return x > 0 ? value : 0.0;
}
// PQ en place sur un tableau ; NaN si le dénominateur s'annule (refusé ensuite comme non fini).
#define FEL_PQ_ROWS(SUFFIX, ATTRIBUTES) \
    ATTRIBUTES void pq_eotf_row##SUFFIX(double *v, int count) { \
        for (int i = 0; i < count; ++i) { \
            const double p = fast_pow(v[i], 1/m2); \
            const double d = c2 - c3*p; \
            const double value = fast_pow(std::max(p - c1, 0.0)/std::max(d, 1e-300), 1/m1); \
            v[i] = d > 0 ? value : NAN; \
        } \
    } \
    ATTRIBUTES void pq_oetf_row##SUFFIX(double *v, int count) { \
        for (int i = 0; i < count; ++i) { \
            const double p = fast_pow(v[i], m1); \
            v[i] = fast_pow((c1 + c2*p)/(1 + c3*p), m2); \
        } \
    }
// Sélections sans branche : fichier compilé avec -fno-trapping-math (CMakeLists), sinon GCC
// garde des branches autour d'opérations flottantes susceptibles de lever une exception.
FEL_PQ_ROWS(_default, )
#ifdef MVO_FEL_AVX2
FEL_PQ_ROWS(_avx2, __attribute__((target("avx2,fma"))))
// Détection au premier appel : __builtin_cpu_init explicite, l'ordre des constructeurs
// d'une bibliothèque chargée par dlopen n'étant pas garanti.
bool pq_avx2() {
    static const bool supported = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return supported;
}
#endif
void pq_eotf_row(double *v, int count) {
#ifdef MVO_FEL_AVX2
    if (pq_avx2()) return pq_eotf_row_avx2(v, count);
#endif
    pq_eotf_row_default(v, count);
}
void pq_oetf_row(double *v, int count) {
#ifdef MVO_FEL_AVX2
    if (pq_avx2()) return pq_oetf_row_avx2(v, count);
#endif
    pq_oetf_row_default(v, count);
}
}

bool IntegerAxis::build(int count, int size, double ratio, double layer, double chroma, int subsample) {
    index.resize(count); weight.resize(count); denominator.resize(count);
    weight_sum = 0;
    for (int i = 0; i < count; ++i) {
        // Même coordonnée que FrameSampler (centres des pixels, couche et position chroma).
        const double coordinate = ((i+.5-layer)*ratio-chroma)/(1 << subsample)-.5;
        const int first = static_cast<int>(std::floor(coordinate))-1;
        double w[4], sum = 0;
        for (int k = 0; k < 4; ++k) {
            index[i][k] = std::clamp(first+k, 0, size-1);
            sum += w[k] = spline16(coordinate-first-k);
        }
        for (double &value : w) value /= sum;
        denominator[i] = exact_weights(w, weight[i]);
        if (!denominator[i]) return false;
        int32_t magnitude = 0;
        for (int32_t value : weight[i]) magnitude += std::abs(value);
        weight_sum = std::max(weight_sum, magnitude);
    }
    return true;
}

std::array<int, 6> FastReconstruction::key(const AVFrame &bl, const AVFrame &el) {
    return {bl.width, bl.height, el.width, el.height, bl.chroma_location*16 + el.chroma_location, bl.format*1024 + el.format};
}

bool FastReconstruction::plane(Plane &plane, const AVFrame &frame, int c, int bw, int bh, bool el) {
    const auto *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
    if (!desc || desc->nb_components != 3 || !(desc->flags & AV_PIX_FMT_FLAG_PLANAR) ||
        (desc->flags & (AV_PIX_FMT_FLAG_BE | AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_FLOAT)) ||
        desc->comp[c].depth != 10 || desc->comp[c].step != 2 || desc->comp[c].shift || desc->comp[c].offset)
        return false;
    const int hs = c ? desc->log2_chroma_w : 0, vs = c ? desc->log2_chroma_h : 0;
    double cx = 0, cy = 0;
    if (c) switch (frame.chroma_location) {
        case AVCHROMA_LOC_TOPLEFT: cx = -.5; cy = -.5; break;
        case AVCHROMA_LOC_LEFT: cx = -.5; break;
        case AVCHROMA_LOC_CENTER: break;
        case AVCHROMA_LOC_TOP: cy = -.5; break;
        case AVCHROMA_LOC_BOTTOMLEFT: cx = -.5; cy = .5; break;
        case AVCHROMA_LOC_BOTTOM: cy = .5; break;
        default: return false;
    }
    plane.width = (frame.width + (1 << hs)-1) >> hs;
    plane.height = (frame.height + (1 << vs)-1) >> vs;
    return plane.horizontal.build(bw, plane.width, double(frame.width)/bw, el && frame.width < bw ? -.5 : 0, cx, hs) &&
           plane.vertical.build(bh, plane.height, double(frame.height)/bh, 0, cy, vs) &&
           // |code - offset| <= 1023 : sommes verticales puis horizontales en int32.
           1023.0*plane.horizontal.weight_sum*plane.vertical.weight_sum < 2147483647.0;
}

FastReconstruction::FastReconstruction(const AVFrame &bl, const AVFrame &el) {
    const auto *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(bl.format));
    width = bl.width; height = bl.height;
    geometry = key(bl, el);
    if (!desc || desc->comp[0].depth != 10 || desc->comp[0].step != 2 || desc->comp[0].shift) return;
    ok = plane(base_chroma[0], bl, 1, width, height, false) && plane(base_chroma[1], bl, 2, width, height, false);
    for (int c = 0; ok && c < 3; ++c) ok = plane(enhancement[c], el, c, width, height, true);
}

bool FastReconstruction::bind(const AVFrame &bl, const AVFrame &el, const AVDOVIMetadata &metadata) {
    if (!ok || key(bl, el) != geometry) return false;
    const auto *bd = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(bl.format));
    const auto *ed = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(el.format));
    const auto &mapping = *av_dovi_get_mapping(&metadata);
    luma = bl.data[bd->comp[0].plane]; luma_stride = bl.linesize[bd->comp[0].plane];
    for (int c = 0; c < 2; ++c) {
        base_chroma[c].data = bl.data[bd->comp[c+1].plane];
        base_chroma[c].stride = bl.linesize[bd->comp[c+1].plane];
    }
    for (int c = 0; c < 3; ++c) {
        enhancement[c].data = el.data[ed->comp[c].plane];
        enhancement[c].stride = el.linesize[ed->comp[c].plane];
        enhancement[c].offset = mapping.nlq[c].nlq_offset;
    }
    return true;
}

void FastReconstruction::rows(const Mapping &mapping, AVFrame &rgb, int first_row, int last_row) const {
    // Passe verticale par colonne source (int32), puis horizontale : 4 + 4 prises au lieu de 16.
    std::array<std::vector<int32_t>, 5> vertical;
    const Plane *planes[5] = {&base_chroma[0], &base_chroma[1], &enhancement[0], &enhancement[1], &enhancement[2]};
    for (int k = 0; k < 5; ++k) vertical[k].resize(planes[k]->width);
    std::vector<double> scale(5);
    std::vector<double> lms(3*width), linear(3*width);
    // LMS → RGB : produit lms_to_rgb · linear (matrice du RPU).
    double matrix[3][3];
    for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 3; ++k) {
            matrix[c][k] = 0;
            for (int j = 0; j < 3; ++j) matrix[c][k] += lms_to_rgb[c][j]*mapping.linear[j][k];
        }
    for (int y = first_row; y < last_row; ++y) {
        for (int k = 0; k < 5; ++k) {
            const Plane &p = *planes[k];
            const auto &rows = p.vertical.index[y];
            const auto &w = p.vertical.weight[y];
            const uint16_t *source[4];
            for (int b = 0; b < 4; ++b) source[b] = reinterpret_cast<const uint16_t *>(p.data + rows[b]*p.stride);
            int32_t *out = vertical[k].data();
            for (int x = 0; x < p.width; ++x)
                out[x] = w[0]*(source[0][x]-p.offset) + w[1]*(source[1][x]-p.offset) +
                         w[2]*(source[2][x]-p.offset) + w[3]*(source[3][x]-p.offset);
            scale[k] = 1.0/(double(p.vertical.denominator[y])*1023.0);
        }
        const auto *row = reinterpret_cast<const uint16_t *>(luma + y*luma_stride);
        // Signal non linéaire (mapping, NLQ, matrice Y'CbCr→LMS') par pixel, puis PQ par
        // tableaux contigus : boucles sans branche vectorisées par le compilateur.
        for (int x = 0; x < width; ++x) {
            double value[5];
            for (int k = 0; k < 5; ++k) {
                const Plane &p = *planes[k];
                const auto &i = p.horizontal.index[x];
                const auto &w = p.horizontal.weight[x];
                const int32_t *v = vertical[k].data();
                const int32_t sum = w[0]*v[i[0]] + w[1]*v[i[1]] + w[2]*v[i[2]] + w[3]*v[i[3]];
                value[k] = sum ? sum*scale[k]/p.horizontal.denominator[x] : 0.0;
            }
            Vec out = reshape(mapping, {row[x]/1023.0, value[0], value[1]});
            // Résiduel EL exact : un zéro entier n'applique ni seuil ni pente NLQ.
            if (mapping.residual) for (int c = 0; c < 3; ++c) {
                const double d = value[2+c];
                if (d != 0) out[c] += sign(d)*(std::abs(d)*mapping.slope[c] + mapping.threshold[c]);
            }
            for (int c = 0; c < 3; ++c) out[c] -= mapping.nonlinear_offset[c];
            for (int c = 0; c < 3; ++c)
                lms[c*width + x] = mapping.nonlinear[c][0]*out[0] + mapping.nonlinear[c][1]*out[1] +
                                   mapping.nonlinear[c][2]*out[2];
        }
        pq_eotf_row(lms.data(), 3*width);
        for (int x = 0; x < width; ++x) {
            const double l[3] = {lms[x], lms[width + x], lms[2*width + x]};
            for (int c = 0; c < 3; ++c)
                linear[c*width + x] = l[0]*matrix[c][0] + l[1]*matrix[c][1] + l[2]*matrix[c][2];
        }
        pq_oetf_row(linear.data(), 3*width);
        for (int c = 0; c < 3; ++c) {
            const int plane = c == 0 ? 2 : c == 1 ? 0 : 1;
            auto *p = reinterpret_cast<uint16_t *>(rgb.data[plane] + y*rgb.linesize[plane]);
            for (int x = 0; x < width; ++x) {
                const double v = linear[c*width + x];
                if (!std::isfinite(v)) throw std::runtime_error("Pixel reconstruit non fini");
                p[x] = static_cast<uint16_t>(std::lround(std::clamp(v, 0.0, 1.0)*65535.0));
            }
        }
    }
}
}

extern "C" double fel_dither_threshold(int dither, int x, int y) {
    static const uint8_t bayer[8][8] = {
        {0,32,8,40,2,34,10,42},{48,16,56,24,50,18,58,26},{12,44,4,36,14,46,6,38},{60,28,52,20,62,30,54,22},
        {3,35,11,43,1,33,9,41},{51,19,59,27,49,17,57,25},{15,47,7,39,13,45,5,37},{63,31,55,23,61,29,53,21}};
    if (dither == FEL_DITHER_ORDERED) return (bayer[y & 7][x & 7] + 0.5) / 64;
    if (dither == FEL_DITHER_BLUE) return (fel_blue_noise[(y & 63)*64 + (x & 63)] + 0.5) / 4096;
    return 0.5;
}

namespace muxiveo::fel {
void convert_rows(const AVFrame &rgb, AVFrame &yuv, const fel_output &output, int first, int last) {
    const int depth = fel_output_depth(output.format);
    const double scale = std::ldexp(1.0, depth - 8);
    double ox, oy;
    fel_chroma_offset(output.chroma, &ox, &oy);
    std::array<double, 4> wx{}, wy{};
    int fx, fy;
    taps(ox, wx, fx); taps(oy, wy, fy);
    for (int j = first; j < last; ++j) {
        for (int dy = 0; dy < 2; ++dy) {
            const int y = 2*j + dy;
            for (int x = 0; x < output.width; ++x) {
                const Vec c = rgb_at(rgb, output.crop_x + x, output.crop_y + y);
                const double luma = kr*c[0] + kg*c[1] + kb*c[2];
                store(yuv, 0, x, y, quantize((219*luma + 16)*scale,
                      fel_dither_threshold(output.dither, x, y), depth), depth);
            }
        }
        for (int i = 0; i < output.width/2; ++i) {
            Vec sum{};
            for (int b = 0; b < 4; ++b) for (int a = 0; a < 4; ++a) {
                const double w = wx[a]*wy[b];
                if (w == 0) continue;
                const Vec c = rgb_at(rgb, output.crop_x + 2*i + fx + a, output.crop_y + 2*j + fy + b);
                for (int k = 0; k < 3; ++k) sum[k] += w*c[k];
            }
            const double luma = kr*sum[0] + kg*sum[1] + kb*sum[2];
            const double cb = (sum[2] - luma) / (2*(1 - kb)), cr = (sum[0] - luma) / (2*(1 - kr));
            store(yuv, 1, i, j, quantize((224*cb + 128)*scale, fel_dither_threshold(output.dither, i+21, j+11), depth), depth);
            store(yuv, 2, i, j, quantize((224*cr + 128)*scale, fel_dither_threshold(output.dither, i+43, j+37), depth), depth);
        }
    }
}
}
