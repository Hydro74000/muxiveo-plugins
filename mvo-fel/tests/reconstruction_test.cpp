// SPDX-License-Identifier: LGPL-2.1-or-later
#include "reconstruction.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace muxiveo::fel;
void near(double actual, double expected, double tolerance = 1e-10) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
        std::cerr << actual << " != " << expected << '\n';
        std::exit(1);
    }
}
AVFrame *layer(int width, int height, std::mt19937 &random, int low = 0, int high = 1023) {
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

// Écart maximal (codes 16 bits) du chemin rapide (entier séparable + PQ vectorisée) face aux
// échantillonneurs et à la PQ de référence en double, sur des couches aléatoires.
CertifiedStats totals;
int fast_error(const AVDOVIMetadata &metadata, std::mt19937 &random, int low, int high) {
    AVFrame *bl = layer(64, 32, random, low, high), *el = layer(32, 16, random, low, high);
    AVFrame *a = av_frame_alloc(), *b = av_frame_alloc();
    for (AVFrame *rgb : {a, b}) {
        rgb->format = AV_PIX_FMT_GBRP16LE; rgb->width = 64; rgb->height = 32;
        if (av_frame_get_buffer(rgb, 0) < 0) std::exit(1);
    }
    const Mapping m(metadata);
    reconstruct_rows(*bl, *el, m, *a, 0, 32);
    FastReconstruction fast(*bl, *el);
    if (!fast.valid() || !fast.bind(*bl, *el, metadata)) std::exit(1);
    const auto stats = fast.rows(m, *b, 0, 32);
    // Chemin certifié = toutes les composantes en double-double, aucune indécise.
    AVFrame *d = av_frame_alloc();
    d->format = AV_PIX_FMT_GBRP16LE; d->width = 64; d->height = 32;
    if (av_frame_get_buffer(d, 0) < 0) std::exit(1);
    const auto extended = fast.rows(m, *d, 0, 32, true);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 32; ++y)
            if (std::memcmp(b->data[c] + y*b->linesize[c], d->data[c] + y*d->linesize[c], 128)) {
                std::cerr << "chemin certifié différent de la reprise double-double\n";
                std::exit(1);
            }
    if (stats.undecided || extended.undecided || extended.rescued != extended.components ||
        !(extended.worst_ratio <= 0.5)) {
        std::cerr << "borne du chemin double : " << extended.worst_ratio << '\n';
        std::cerr << "composantes indécises : " << stats.undecided << ' ' << extended.undecided << '\n';
        std::exit(1);
    }
    totals += stats;
    totals.worst_ratio = std::max(totals.worst_ratio, extended.worst_ratio);
    av_frame_free(&d);
    int worst = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 64; ++x)
                worst = std::max(worst, std::abs(reinterpret_cast<const uint16_t *>(a->data[c] + y*a->linesize[c])[x] -
                                                 reinterpret_cast<const uint16_t *>(b->data[c] + y*b->linesize[c])[x]));
    for (AVFrame *frame : {bl, el, a, b}) av_frame_free(&frame);
    return worst;
}

// ≤ 2 codes 16 bits (0,125 code PQ12). Matrices identité : la matrice LMS→RGB fixe compense
// des termes de l'ordre de 0,4 jusqu'à quelques 1e-5 (vert BT.2020 saturé vers 2000 cd/m²),
// et la PQ amplifie fortement l'erreur absolue près de zéro (audit du 15/10/2026, graine 66).
void fast_matches_reference(AVDOVIMetadata &metadata) {
    for (unsigned seed = 1; seed <= 200; ++seed) {
        std::mt19937 random(seed);
        if (const int worst = fast_error(metadata, random, 64, 940); worst > 2) {
            std::cerr << "chemin rapide, graine " << seed << " : " << worst << " codes 16 bits\n";
            std::exit(1);
        }
    }
    auto &mapping = *av_dovi_get_mapping(&metadata);
    for (int c = 0; c < 3; ++c) {
        auto &curve = mapping.curves[c];
        curve.num_pivots = 3; curve.pivots[1] = 400; curve.pivots[2] = 1023;
        curve.mapping_idc[1] = AV_DOVI_MAPPING_MMR;
        curve.mmr_order[1] = 2; curve.mmr_constant[1] = 2000;
        for (int k = 0; k < 2; ++k) for (int j = 0; j < 7; ++j) curve.mmr_coef[1][k][j] = 3000*(j+1)*(k ? -1 : 1);
    }
    // 200 graines : signaux exactement sur un pivot compris (segment choisi sur valeurs exactes).
    for (unsigned seed = 1; seed <= 200; ++seed) {
        std::mt19937 random(seed);
        if (const int worst = fast_error(metadata, random, 0, 1023); worst > 2) {
            std::cerr << "chemin rapide MMR, graine " << seed << " : " << worst << " codes 16 bits\n";
            std::exit(1);
        }
    }
}

// Table des seuils PQ : strictement croissante dans ]0, 1[, cohérente avec pq_eotf et pq_oetf.
void pq_thresholds_consistent() {
    double previous = 0;
    for (int k = 0; k < PQ_THRESHOLDS; ++k) {
        const double threshold = pq_threshold(k), signal = (k + 0.5)/65535;
        if (!(threshold > previous && threshold < 1)) {
            std::cerr << "seuil PQ " << k << " hors ordre : " << threshold << '\n';
            std::exit(1);
        }
        previous = threshold;
        near(threshold/pq_eotf(signal), 1, 1e-11);
        near(pq_oetf(threshold), signal, 1e-12);
    }
}
// Y'CbCr entier exact face à la formule en double, hors voisinage immédiat d'une frontière.
void yuv_matches_double() {
    std::mt19937 random(7);
    std::uniform_int_distribution<int> value(0, 65535);
    AVFrame *rgb = av_frame_alloc(), *yuv = av_frame_alloc();
    rgb->format = AV_PIX_FMT_GBRP16LE; rgb->width = 64; rgb->height = 32;
    yuv->format = AV_PIX_FMT_YUV420P10LE; yuv->width = 64; yuv->height = 32;
    if (av_frame_get_buffer(rgb, 0) < 0 || av_frame_get_buffer(yuv, 0) < 0) std::exit(1);
    auto at = [&](int plane, int x, int y) {
        x = std::clamp(x, 0, 63); y = std::clamp(y, 0, 31);
        return reinterpret_cast<uint16_t *>(rgb->data[plane] + y*rgb->linesize[plane])[x] / 65535.0;
    };
    const double kr = 0.2627, kb = 0.0593, kg = 1 - kr - kb;
    long long checked = 0;
    for (int dither : {FEL_DITHER_NONE, FEL_DITHER_ORDERED, FEL_DITHER_BLUE})
        for (int chroma : {1, 2, 3, 5}) {
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < 32; ++y)
                    for (int x = 0; x < 64; ++x)
                        reinterpret_cast<uint16_t *>(rgb->data[c] + y*rgb->linesize[c])[x] = uint16_t(value(random));
            fel_output output{};
            output.format = FEL_OUT_YUV420P10; output.width = 64; output.height = 32;
            output.dither = dither; output.chroma = chroma;
            convert_rows(*rgb, *yuv, output, 0, 16);
            auto check = [&](double v, double t, int plane, int x, int y) {
                const double q = v*4 + t;
                if (std::abs(q - std::round(q)) < 1e-9) return;  // frontière : arbitrage exact
                const int expected = std::clamp(int(std::floor(q)), 0, 1023);
                if (reinterpret_cast<uint16_t *>(yuv->data[plane] + y*yuv->linesize[plane])[x] != expected) {
                    std::cerr << "Y'CbCr entier différent du double : plan " << plane << '\n';
                    std::exit(1);
                }
                ++checked;
            };
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 64; ++x)
                    check(219*(kr*at(2, x, y) + kg*at(0, x, y) + kb*at(1, x, y)) + 16,
                          fel_dither_threshold(dither, x, y), 0, x, y);
            double ox, oy;
            fel_chroma_offset(chroma, &ox, &oy);
            for (int j = 0; j < 16; ++j)
                for (int i = 0; i < 32; ++i) {
                    Vec sum{};
                    double total = 0;
                    for (int b = -1; b <= 2; ++b)
                        for (int a = -1; a <= 2; ++a) {
                            const double w = std::max(0.0, 1 - std::abs(a - ox)/2)/2 *
                                             std::max(0.0, 1 - std::abs(b - oy)/2)/2;
                            total += w;
                            sum[0] += w*at(2, 2*i + a, 2*j + b);
                            sum[1] += w*at(0, 2*i + a, 2*j + b);
                            sum[2] += w*at(1, 2*i + a, 2*j + b);
                        }
                    if (std::abs(total - 1) > 1e-12) { std::cerr << "poids chroma\n"; std::exit(1); }
                    const double l = kr*sum[0] + kg*sum[1] + kb*sum[2];
                    check(224*(sum[2] - l)/(2*(1 - kb)) + 128, fel_dither_threshold(dither, i+21, j+11), 1, i, j);
                    check(224*(sum[0] - l)/(2*(1 - kr)) + 128, fel_dither_threshold(dither, i+43, j+37), 2, i, j);
                }
        }
    av_frame_free(&rgb); av_frame_free(&yuv);
    if (checked < 12*(64*32 + 2*32*16)*9/10) { std::cerr << "trop peu de valeurs vérifiées\n"; std::exit(1); }
}
int main() {
    pq_thresholds_consistent();
    yuv_matches_double();
    auto *metadata = av_dovi_metadata_alloc(nullptr);
    auto &header = *av_dovi_get_header(metadata);
    auto &mapping = *av_dovi_get_mapping(metadata);
    auto &color = *av_dovi_get_color(metadata);
    header.bl_bit_depth = header.el_bit_depth = 10;
    header.coef_log2_denom = 16;
    mapping.nlq_method_idc = AV_DOVI_NLQ_LINEAR_DZ;
    for (int c = 0; c < 3; ++c) {
        auto &curve = mapping.curves[c];
        curve.num_pivots = 2; curve.pivots[1] = 1023;
        curve.poly_order[0] = 1; curve.poly_coef[0][1] = 65536;
        mapping.nlq[c].nlq_offset = 512;
        mapping.nlq[c].linear_deadzone_slope = 64;
        mapping.nlq[c].linear_deadzone_threshold = 32;
        color.ycc_to_rgb_offset[c] = {0, 1};
        for (int j = 0; j < 3; ++j) {
            color.ycc_to_rgb_matrix[3*c+j] = {c == j ? 1 : 0, 1};
            color.rgb_to_lms_matrix[3*c+j] = {c == j ? 1 : 0, 1};
        }
    }
    Mapping m(*metadata);
    const Vec bl{0.3, 0.4, 0.5};
    for (int c = 0; c < 3; ++c) near(reshape(m, bl)[c], bl[c]);
    const auto composed = compose(m, bl, {513./1023, 512./1023, 511./1023});
    near(composed[0], .3 + 1./1024);
    near(composed[1], .4);
    near(composed[2], .5 - 1./1024);
    // Au voisinage du zéro NLQ, un ulp ne doit pas déclencher le seuil discontinu.
    Mapping neutral = m;
    neutral.threshold = {.001, .001, .001};
    const double offset = 512./1023;
    const Vec rounded{std::nextafter(offset, 1.), offset, std::nextafter(offset, 0.)};
    const auto no_residual = compose(neutral, bl, rounded);
    for (int c = 0; c < 3; ++c) near(no_residual[c], bl[c]);
    const auto small_residual = compose(neutral, bl, {offset+1e-9, offset, offset-1e-9});
    near(small_residual[0], bl[0]+neutral.threshold[0]+1e-9*neutral.slope[0]);
    near(small_residual[2], bl[2]-neutral.threshold[2]-1e-9*neutral.slope[2]);
    m.curves[1].method[0] = AV_DOVI_MAPPING_MMR;
    m.curves[1].order[0] = 3;
    m.curves[1].mmr[0][0][3] = .2;
    m.curves[1].mmr[0][1][4] = .3;
    m.curves[1].mmr[0][2][6] = .4;
    near(reshape(m, bl)[1], .2*.3*.4 + .3*std::pow(.3*.5, 2) + .4*std::pow(.3*.4*.5, 3));
    for (int i = 1; i <= 100; ++i) near(pq_eotf(pq_oetf(i/100.)), i/100., 1e-11);
    near(pq_oetf(.01), .508078421517399, 1e-12); // ST 2084 : 100 cd/m²
    near(spline16(0), 1); near(spline16(1), 0); near(spline16(2), 0);
    for (int i = 0; i < 100; ++i) {
        const double x = i/100.;
        near(spline16(x+1)+spline16(x)+spline16(x-1)+spline16(x-2), 1);
    }
    fast_matches_reference(*metadata);
    std::cout << "certifié : " << totals.components << " composantes, " << totals.rescued << " reprises, "
              << totals.changed << " différentes de l'arrondi double ; écart / borne ≤ " << totals.worst_ratio << '\n';
    mapping.curves[0].num_pivots = 10;
    try { Mapping bad(*metadata); return 1; } catch (const std::runtime_error &) {}
    av_free(metadata);
}
