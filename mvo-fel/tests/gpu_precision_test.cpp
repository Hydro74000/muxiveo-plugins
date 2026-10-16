// SPDX-License-Identifier: LGPL-2.1-or-later
// Précision du rendu GPU par mode face au CPU certifié, sur des couches aléatoires : mapping
// identité (annulations de la matrice LMS→RGB, audit du 15/10/2026), matrices Dolby réalistes, et
// MMR avec signaux exactement sur un pivot. Exact : identique (0 code) ; standard : ≤ 1 code
// 16 bits ; rapide (float32) : écart mesuré, sans borne générale.
#include "reconstruction.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>
extern "C" {
void *fel_gpu_create(int, int, const char *);
void fel_gpu_destroy(void *);
int fel_gpu_render(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, uint16_t *);
int fel_gpu_set_precision(void *, int);
}
using namespace muxiveo::fel;

static AVFrame *layer(int width, int height, std::mt19937 &random, int low, int high) {
    AVFrame *frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P10LE; frame->width = width; frame->height = height;
    frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
    if (av_frame_get_buffer(frame, 32) < 0) std::exit(1);
    std::uniform_int_distribution<int> code(low, high);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < (c ? height/2 : height); ++y) {
            auto *row = reinterpret_cast<uint16_t *>(frame->data[c] + y*frame->linesize[c]);
            for (int x = 0; x < (c ? width/2 : width); ++x) row[x] = static_cast<uint16_t>(code(random));
        }
    return frame;
}

static void metadata_for(AVDOVIMetadata &metadata, const std::string &mode) {
    auto &header = *av_dovi_get_header(&metadata);
    auto &mapping = *av_dovi_get_mapping(&metadata);
    auto &color = *av_dovi_get_color(&metadata);
    header.bl_bit_depth = header.el_bit_depth = 10;
    header.coef_log2_denom = 16;
    mapping.nlq_method_idc = AV_DOVI_NLQ_LINEAR_DZ;
    const int ycc[9] = {65536, 0, 91881, 65536, -22554, -46802, 65536, 116130, 0};
    const int lms[9] = {4408196, 5353728, 238131, 1619851, 7586529, 793630, 0, 257773, 9740729};
    const bool real = mode == "real";
    for (int c = 0; c < 3; ++c) {
        auto &curve = mapping.curves[c];
        curve = {};
        curve.num_pivots = 2; curve.pivots[1] = 1023;
        curve.poly_order[0] = 1; curve.poly_coef[0][1] = 65536;
        mapping.nlq[c].nlq_offset = 512;
        mapping.nlq[c].linear_deadzone_slope = 64;
        mapping.nlq[c].linear_deadzone_threshold = 32;
        color.ycc_to_rgb_offset[c] = {real && c ? 32768 : 0, real ? 65536 : 1};
        for (int j = 0; j < 3; ++j) {
            color.ycc_to_rgb_matrix[3*c+j] = real ? AVRational{ycc[3*c+j], 65536} : AVRational{c == j, 1};
            color.rgb_to_lms_matrix[3*c+j] = real ? AVRational{lms[3*c+j], 10000000} : AVRational{c == j, 1};
        }
        if (mode == "mmr") {
            curve.num_pivots = 3; curve.pivots[1] = 400; curve.pivots[2] = 1023;
            curve.mapping_idc[1] = AV_DOVI_MAPPING_MMR;
            curve.mmr_order[1] = 2; curve.mmr_constant[1] = 2000;
            for (int k = 0; k < 2; ++k) for (int j = 0; j < 7; ++j) curve.mmr_coef[1][k][j] = 3000*(j+1)*(k ? -1 : 1);
        }
    }
}

int main() {
    const char *device = std::getenv("MVO_FEL_TEST_DEVICE");
    void *gpu = fel_gpu_create(64, 32, device);
    if (!gpu) return 77;
    if (!fel_gpu_set_precision(gpu, FEL_PRECISION_EXACT)) {
        std::cout << "GPU sans calcul double : modes exact et standard non testés\n";
        fel_gpu_destroy(gpu);
        return 77;
    }
    bool ok = true;
    auto *metadata = av_dovi_metadata_alloc(nullptr);
    std::vector<uint16_t> rgba(64*32*4);
    for (const std::string mode : {"identity", "real", "mmr"}) {
        metadata_for(*metadata, mode);
        const Mapping m(*metadata);
        const int low = mode == "mmr" ? 0 : mode == "real" ? 200 : 64, high = mode == "mmr" ? 1023 : mode == "real" ? 760 : 940;
        for (const int precision : {FEL_PRECISION_EXACT, FEL_PRECISION_STANDARD, FEL_PRECISION_FAST}) {
            if (!fel_gpu_set_precision(gpu, precision)) return 1;
            int worst = 0;
            long differing = 0;
            for (unsigned seed = 1; seed <= 200; ++seed) {
                std::mt19937 random(seed);
                AVFrame *bl = layer(64, 32, random, low, high), *el = layer(32, 16, random, low, high);
                AVFrame *cpu = av_frame_alloc();
                cpu->format = AV_PIX_FMT_GBRP16LE; cpu->width = 64; cpu->height = 32;
                if (av_frame_get_buffer(cpu, 32) < 0) return 1;
                FastReconstruction fast(*bl, *el);
                if (!fast.valid() || !fast.bind(*bl, *el, *metadata)) return 1;
                fast.rows(m, *cpu, 0, 32);
                if (!fel_gpu_render(gpu, metadata, bl, el, rgba.data())) {
                    std::cerr << "Rendu GPU impossible\n";
                    return 1;
                }
                for (int y = 0; y < 32; ++y) for (int x = 0; x < 64; ++x) for (int c = 0; c < 3; ++c) {
                    const int plane = c == 0 ? 2 : c == 1 ? 0 : 1;
                    const int expected = reinterpret_cast<const uint16_t *>(cpu->data[plane] + y*cpu->linesize[plane])[x];
                    const int delta = std::abs(int(rgba[4*(y*64+x)+c]) - expected);
                    worst = std::max(worst, delta);
                    differing += delta != 0;
                }
                for (AVFrame *f : {bl, el, cpu}) av_frame_free(&f);
            }
            const char *name = precision == FEL_PRECISION_EXACT ? "exact" : precision == FEL_PRECISION_STANDARD ? "standard" : "rapide";
            std::cout << mode << " / " << name << " : écart max " << worst << " code(s) 16 bits, valeurs ≠ " << differing << '\n';
            if (worst > (precision == FEL_PRECISION_EXACT ? 0 : precision == FEL_PRECISION_STANDARD ? 1 : 65535)) ok = false;
        }
    }
    av_free(metadata);
    fel_gpu_destroy(gpu);
    if (!ok) std::cerr << "Précision GPU hors borne (exact : 0 code, standard : 1 code 16 bits)\n";
    return ok ? 0 : 1;
}
