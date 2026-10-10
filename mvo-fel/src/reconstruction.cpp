// SPDX-License-Identifier: LGPL-2.1-or-later
// Adaptation CPU : libplacebo, colorspace.c, shaders/colorspace.c, filters.c
// et utils/libav_internal.h. Révision et attributions dans NOTICES.md.
#include "reconstruction.hpp"
#include <algorithm>
#include <cmath>
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
}
