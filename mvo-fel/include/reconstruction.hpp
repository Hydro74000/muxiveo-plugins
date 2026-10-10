// SPDX-License-Identifier: LGPL-2.1-or-later
// Port CPU des opérations Dolby Vision de libplacebo (voir NOTICES.md).
#pragma once

#include <array>
#include <cstdint>
#include <vector>
extern "C" {
#include <libavutil/dovi_meta.h>
#include <libavutil/frame.h>
}

namespace muxiveo::fel {
using Vec = std::array<double, 3>;
using Mat = std::array<Vec, 3>;

struct Curve {
    unsigned count = 0;
    std::array<double, 9> pivots{};
    std::array<int, 8> method{};
    std::array<unsigned, 8> order{};
    std::array<Vec, 8> poly{};
    std::array<double, 8> constant{};
    std::array<std::array<std::array<double, 7>, 3>, 8> mmr{};
};

struct Mapping {
    std::array<Curve, 3> curves;
    Vec offset{}, slope{}, threshold{}, nonlinear_offset{};
    Mat nonlinear{}, linear{};
    bool residual = false;
    explicit Mapping(const AVDOVIMetadata &metadata);
};

double spline16(double x);
double pq_eotf(double x);
double pq_oetf(double x);
Vec reshape(const Mapping &mapping, const Vec &bl);
Vec compose(const Mapping &mapping, const Vec &bl, const Vec &el);
Vec to_rgb(const Mapping &mapping, const Vec &signal);

// Les coordonnées sont exprimées au centre des pixels BL (entiers).
double sample_plane(const AVFrame &frame, int component, int x, int y,
                    int bl_width, int bl_height, bool enhancement);
class FrameSampler {
    struct Axis {
        std::array<int, 4> indices{};
        std::array<double, 4> weights{};
    };
    struct Plane {
        const uint8_t *data;
        int stride, step, offset, shift, depth;
        std::vector<Axis> horizontal, vertical;
    };
    std::array<Plane, 3> planes;
public:
    FrameSampler(const AVFrame &frame, int bl_width, int bl_height, bool enhancement);
    double sample(int component, int x, int y) const;
};
void reconstruct_rows(const FrameSampler &bl, const FrameSampler &el, const Mapping &mapping,
                      AVFrame &rgb, int first_row, int last_row);
void reconstruct_rows(const AVFrame &bl, const AVFrame &el, const Mapping &mapping,
                      AVFrame &rgb, int first_row, int last_row);
}
