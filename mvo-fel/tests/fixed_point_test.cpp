// SPDX-License-Identifier: LGPL-2.1-or-later
// Virgule fixe multi-mots (lot A2') : EOTF face au double-double (erreur ≤ moitié de la borne
// 2^-50·(1 + 1/ln(x/x0))), puis pipeline complet face au CPU certifié sur couches aléatoires :
// codes certains identiques, |R - R exact| ≤ moitié de la borne, part de composantes indécises.
#include "fixed_point.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace muxiveo::fel;
using fixed::u128;

static AVFrame *layer(int width, int height, std::mt19937 &random, int low, int high) {
    AVFrame *frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P10LE; frame->width = width; frame->height = height;
    frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
    if (av_frame_get_buffer(frame, 0) < 0) std::exit(1);
    std::uniform_int_distribution<int> code(low, high);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < (c ? height/2 : height); ++y) {
            auto *row = reinterpret_cast<uint16_t *>(frame->data[c] + y*frame->linesize[c]);
            for (int x = 0; x < (c ? width/2 : width); ++x) row[x] = static_cast<uint16_t>(code(random));
        }
    return frame;
}

static void metadata_for(AVDOVIMetadata &metadata, bool real, bool mmr) {
    auto &header = *av_dovi_get_header(&metadata);
    auto &mapping = *av_dovi_get_mapping(&metadata);
    auto &color = *av_dovi_get_color(&metadata);
    header.bl_bit_depth = header.el_bit_depth = 10;
    header.coef_log2_denom = 23;
    mapping.nlq_method_idc = AV_DOVI_NLQ_LINEAR_DZ;
    const int ycc[9] = {8192, 0, 12900, 8192, -1534, -3836, 8192, 15201, 0};
    const int lms[9] = {7222, 8771, 390, 2654, 12430, 1300, 0, 422, 15959};
    for (int c = 0; c < 3; ++c) {
        auto &curve = mapping.curves[c];
        curve = {};
        curve.num_pivots = 2; curve.pivots[1] = 1023;
        curve.poly_order[0] = 1; curve.poly_coef[0][1] = 1 << 23;
        if (mmr) {  // test GPU « mmr » à l'échelle 2^-23, ordre 3 (dernier ordre petit)
            curve.num_pivots = 3; curve.pivots[1] = 400; curve.pivots[2] = 1023;
            curve.mapping_idc[1] = AV_DOVI_MAPPING_MMR;
            curve.mmr_order[1] = 3; curve.mmr_constant[1] = 2000 << 7;
            for (int k = 0; k < 3; ++k) for (int j = 0; j < 7; ++j)
                curve.mmr_coef[1][k][j] = k == 2 ? 200*(j+1) : (3000 << 7)*(j+1)*(k ? -1 : 1);
        }
        mapping.nlq[c].nlq_offset = 512;
        mapping.nlq[c].linear_deadzone_slope = 64 << 7;
        mapping.nlq[c].linear_deadzone_threshold = 40 << 7;
        color.ycc_to_rgb_offset[c] = {real && c ? 1 << 27 : 0, 1 << 28};
        for (int j = 0; j < 3; ++j) {
            color.ycc_to_rgb_matrix[3*c+j] = real ? AVRational{ycc[3*c+j], 8192} : AVRational{c == j, 1};
            color.rgb_to_lms_matrix[3*c+j] = real ? AVRational{lms[3*c+j], 16384} : AVRational{c == j, 1};
        }
    }
}

int main() {
    // EOTF.
    std::mt19937_64 random(3);
    std::uniform_real_distribution<double> unit(0, 1);
    double worst = 0;
    for (int s = 0; s < 300000; ++s) {
        const double x = s % 2 ? std::exp(std::log(7.31e-7)*unit(random)) : unit(random);
        const u128 x64 = (u128)std::ldexp(x, 64);
        uint64_t mant; int n; double dl;
        if (!fixed::eotf(x64, 64, mant, n, dl) || !mant) continue;
        double kappa;
        const auto ref = certified::eotf({std::ldexp((double)x64, -64), 0}, kappa);
        const long double L = std::ldexp((long double)mant, n - 63);
        const double rel = std::abs((double)((L - (long double)ref.hi - ref.lo)/(long double)ref.hi));
        worst = std::max(worst, rel/(std::ldexp(1.0, -50)*(1 + 1/dl)));
    }
    std::printf("EOTF virgule fixe : erreur / borne ≤ %.4f\n", worst);
    if (!(worst <= 0.5)) return 1;
    // Pipeline complet.
    auto *metadata = av_dovi_metadata_alloc(nullptr);
    long long components = 0, uncertain = 0, mismatches = 0;
    double ratio = 0;
    long long per_mode[3][2] = {};
    for (int mode = 0; mode < 3; ++mode) {
        metadata_for(*metadata, mode == 1, mode == 2);
        const Mapping m(*metadata);
        fixed::Params params;
        if (!fixed::prepare(m, *metadata, params)) { std::puts("paramètres hors cadre"); return 1; }
        for (unsigned seed = 1; seed <= 40; ++seed) {
            std::mt19937 rnd(seed);
            const int low = mode == 2 ? 0 : mode == 1 ? 200 : 64, high = mode == 2 ? 1023 : mode == 1 ? 760 : 940;
            AVFrame *bl = layer(64, 32, rnd, low, high), *el = layer(32, 16, rnd, low, high);
            FastReconstruction fast(*bl, *el);
            if (!fast.valid() || !fast.bind(*bl, *el, *metadata)) return 1;
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 64; ++x) {
                    int64_t sums[5], den[5];
                    certified::ExactPixel px;
                    fast.exact_inputs(m, x, y, sums, den, px.luma, px.piece);
                    for (int c = 0; c < 2; ++c) { px.chroma[c] = sums[c]; px.chroma_den[c] = den[c]; }
                    for (int c = 0; c < 3; ++c) { px.residual[c] = sums[2+c]; px.residual_den[c] = den[2+c]; }
                    std::array<uint16_t, 3> codes{}, exact{};
                    fixed::i128 linear[3];
                    double bound[3];
                    const unsigned mask = fixed::pixel(params, px, codes, linear, bound);
                    std::array<certified::DD, 3> value{};
                    certified::classify(m, px, 7, exact, &value);
                    for (int c = 0; c < 3; ++c) {
                        ++components; ++per_mode[mode][0];
                        if (mask >> c & 1) { ++uncertain; ++per_mode[mode][1]; continue; }
                        if (codes[c] != exact[c]) ++mismatches;
                        const long double R = std::ldexp((long double)linear[c], -96);
                        ratio = std::max(ratio, (double)(std::abs(R - (long double)value[c].hi - value[c].lo)/bound[c]));
                    }
                }
            av_frame_free(&bl); av_frame_free(&el);
        }
    }
    av_free(metadata);
    for (int mode = 0; mode < 3; ++mode)
        std::printf("  %s : %lld indécises sur %lld\n", mode == 0 ? "identité" : mode == 1 ? "matrices réelles" : "MMR",
                    per_mode[mode][1], per_mode[mode][0]);
    std::printf("pipeline virgule fixe : %lld composantes, %lld indécises, %lld codes différents, écart / borne ≤ %.4f\n",
                components, uncertain, mismatches, ratio);
    // Indécises : signaux LMS' au-delà de 1,33 (hors cadre, vers le pôle), propres au stress
    // synthétique ; le contenu réel reste sous 1.
    return mismatches == 0 && ratio <= 0.5 && uncertain*100 < components ? 0 : 1;
}
