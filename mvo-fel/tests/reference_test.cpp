// SPDX-License-Identifier: LGPL-2.1-or-later
#include "reconstruction.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" {
void *fel_gpu_create(int, int, const char *);
void fel_gpu_destroy(void *);
int fel_gpu_render(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, uint16_t *);
int reference_init();
void reference_destroy();
int reference_pixels(const AVDOVIMetadata *, const float *, const float *, float *, int);
int reference_frame(const AVDOVIMetadata *, const AVFrame *, const AVFrame *, float *);
int reference_frame_kernel64(const AVDOVIMetadata *, const AVFrame *, const AVFrame *, float *);
}
using namespace muxiveo::fel;

void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
double compare(double actual, double expected) {
    check(std::isfinite(actual) && std::isfinite(expected), "Pixel non fini");
    return std::abs(std::clamp(actual, 0., 1.) - std::clamp(expected, 0., 1.))*4095;
}
AVFrame *frame(int w, int h, AVChromaLocation location) {
    auto *f = av_frame_alloc();
    f->width = w; f->height = h; f->format = AV_PIX_FMT_YUV420P10LE;
    f->chroma_location = location; f->color_range = AVCOL_RANGE_MPEG;
    f->color_primaries = AVCOL_PRI_BT2020; f->color_trc = AVCOL_TRC_SMPTE2084;
    f->colorspace = AVCOL_SPC_BT2020_NCL;
    check(av_frame_get_buffer(f, 32) >= 0, "Allocation image");
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < (c ? h/2 : h); ++y) for (int x = 0; x < (c ? w/2 : w); ++x) {
            // Variations douces + ruptures pour exposer une erreur de position chromatique.
            unsigned code = 512 + int(25*std::sin(x*.43+y*.29+c)) + (x % 7 == 0 ? 12 : 0);
            auto *p = f->data[c] + y*f->linesize[c] + 2*x;
            p[0] = code & 255; p[1] = code >> 8;
        }
    }
    return f;
}
int main() {
    try {
#ifdef FEL_TEST_GPU
        void *probe = fel_gpu_create(64,48,std::getenv("MVO_FEL_TEST_DEVICE"));
        if (!probe) return 77;
        fel_gpu_destroy(probe);
#endif
        check(reference_init(), "Initialisation Vulkan de référence");
        auto *md = av_dovi_metadata_alloc(nullptr);
        auto &h = *av_dovi_get_header(md);
        auto &m = *av_dovi_get_mapping(md);
        auto &c = *av_dovi_get_color(md);
        h.bl_bit_depth = h.el_bit_depth = 10; h.coef_log2_denom = 16;
        m.nlq_method_idc = AV_DOVI_NLQ_LINEAR_DZ;
        const int matrix[9] = {65536, 0, 91881, 65536, -22554, -46802, 65536, 116130, 0};
        const int linear[9] = {4408196, 5353728, 238131, 1619851, 7586529, 793630, 0, 257773, 9740729};
        for (int i = 0; i < 3; ++i) {
            auto &curve = m.curves[i];
            curve.num_pivots = 3; curve.pivots[1] = 512; curve.pivots[2] = 1023;
            for (int p = 0; p < 2; ++p) {
                curve.poly_order[p] = 2;
                curve.poly_coef[p][1] = 65536 - 128; curve.poly_coef[p][2] = 128;
            }
            m.nlq[i].nlq_offset = 512; m.nlq[i].linear_deadzone_slope = 32;
            m.nlq[i].linear_deadzone_threshold = 20;
            c.ycc_to_rgb_offset[i] = {i ? 32768 : 0, 65536};
            for (int j = 0; j < 3; ++j) {
                c.ycc_to_rgb_matrix[3*i+j] = {matrix[3*i+j], 65536};
                c.rgb_to_lms_matrix[3*i+j] = {linear[3*i+j], 10000000};
            }
        }
        constexpr int count = 257;
        std::vector<float> bl(4*count), el(4*count), out(4*count);
        for (int i = 0; i < count; ++i) for (int j = 0; j < 3; ++j) {
            bl[4*i+j] = j ? .5 + .03*std::sin(i*.3+j) : .2 + .6*i/(count-1);
            el[4*i+j] = (512 + (i % 31) - 15)/1023.f;
        }
        double maximum = 0;
        for (int order = 0; order <= 3; ++order) {
            if (order) for (int p = 0; p < 2; ++p) {
                m.curves[1].mapping_idc[p] = AV_DOVI_MAPPING_MMR;
                m.curves[1].mmr_order[p] = order;
                m.curves[1].mmr_coef[p][0][1] = 65536;
                for (int k = 0; k < order; ++k) m.curves[1].mmr_coef[p][k][3] = 128;
            }
            Mapping mapping(*md);
            check(reference_pixels(md, bl.data(), el.data(), out.data(), count), "Shader de référence");
            for (int i = 0; i < count; ++i) {
                Vec b{}, e{};
                for (int j = 0; j < 3; ++j) {
                    b[j] = bl[4*i+j];
                    e[j] = (512 + (i % 31) - 15)/1023.;
                }
                const auto rgb = to_rgb(mapping, compose(mapping, b, e));
                for (int j = 0; j < 3; ++j) {
                    const double error = compare(rgb[j], out[4*i+j]);
                    if (error > maximum && error > 2)
                        std::cerr << "order=" << order << " pixel=" << i << " channel=" << j
                                  << " CPU=" << rgb[j] << " reference=" << out[4*i+j] << '\n';
                    maximum = std::max(maximum, error);
                }
            }
        }
        std::cout << "Mapping/NLQ/matrices : erreur maximale PQ12 = " << maximum << '\n';
        check(maximum <= 2, "Tolérance mapping dépassée");
        bool frames_ok = true;
        for (int disabled : {1, 0}) {
        h.disable_residual_flag = disabled;
        for (auto location : {AVCHROMA_LOC_LEFT, AVCHROMA_LOC_TOPLEFT, AVCHROMA_LOC_CENTER, AVCHROMA_LOC_TOP, AVCHROMA_LOC_BOTTOMLEFT, AVCHROMA_LOC_BOTTOM}) {
            for (int reduction : {1, 2}) {
                auto *b = frame(64, 48, location), *e = frame(64/reduction, 48/reduction, location);
                auto *rgb = av_frame_alloc();
                rgb->format = AV_PIX_FMT_GBRP16LE; rgb->width = b->width; rgb->height = b->height;
                check(av_frame_get_buffer(rgb, 32) >= 0, "Allocation RGB");
                std::vector<float> reference(64*48*4);
                for (auto oracle : {reference_frame, reference_frame_kernel64}) {
                check(oracle(md, b, e, reference.data()), "Renderer de référence");
                reconstruct_rows(*b, *e, Mapping(*md), *rgb, 0, b->height);
                double error = 0;
                for (int y = 0; y < 48; ++y) for (int x = 0; x < 64; ++x) for (int j = 0; j < 3; ++j) {
                    int plane = j == 0 ? 2 : j == 1 ? 0 : 1;
                    auto *p = rgb->data[plane] + y*rgb->linesize[plane] + 2*x;
                    const double value = (p[0]+256*p[1])/65535.;
                    const double delta = compare(value, reference[4*(y*64+x)+j]);
                    if (delta > error && delta > 2)
                        std::cerr << "x=" << x << " y=" << y << " c=" << j << " CPU=" << value
                                  << " ref=" << reference[4*(y*64+x)+j] << '\n';
                    error = std::max(error, delta);
                }
                std::cout << "Spline16 reference=" << (oracle == reference_frame ? "gpu32" : "kernel64")
                          << " residual=" << !disabled << " chroma=" << location << " EL=1/" << reduction << " : PQ12 = " << error << '\n';
                frames_ok = frames_ok && error <= 2;
                }
#ifdef FEL_TEST_GPU
                void *gpu=fel_gpu_create(64,48,std::getenv("MVO_FEL_TEST_DEVICE"));
                check(gpu,"GPU de test");
                std::vector<uint16_t> actual(64*48*4);
                for(int threshold : {0,20}) {
                    for(int j=0;j<3;++j)m.nlq[j].linear_deadzone_threshold=threshold;
                    check(fel_gpu_render(gpu,md,b,e,actual.data()),"Reconstruction GPU");
                    check(reference_frame_kernel64(md,b,e,reference.data()),"Référence indépendante");
                    double delta=0;
                    for(size_t i=0;i<actual.size();++i)if(i%4!=3)
                        delta=std::max(delta,compare(actual[i]/65535.,reference[i]));
                    std::cout<<"GPU chroma="<<location<<" reduction="<<reduction<<" threshold="<<threshold<<" PQ12="<<delta<<'\n';
                    frames_ok=frames_ok && delta<=2;
                }
                for(int j=0;j<3;++j)m.nlq[j].linear_deadzone_threshold=20;
                fel_gpu_destroy(gpu);
#endif
                av_frame_free(&b); av_frame_free(&e); av_frame_free(&rgb);
            }
        }
        }
        check(frames_ok, "Tolérance image dépassée");
        av_free(md);
        reference_destroy();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
