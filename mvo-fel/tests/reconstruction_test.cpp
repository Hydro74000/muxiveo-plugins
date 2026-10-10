// SPDX-License-Identifier: LGPL-2.1-or-later
#include "reconstruction.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace muxiveo::fel;
void near(double actual, double expected, double tolerance = 1e-10) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
        std::cerr << actual << " != " << expected << '\n';
        std::exit(1);
    }
}
int main() {
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
    mapping.curves[0].num_pivots = 10;
    try { Mapping bad(*metadata); return 1; } catch (const std::runtime_error &) {}
    av_free(metadata);
}
