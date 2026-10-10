// SPDX-License-Identifier: LGPL-2.1-or-later
// Oracle indépendant : exécute les shaders et le renderer du libplacebo épinglé.
#include <libplacebo/vulkan.h>
extern PFN_vkGetInstanceProcAddr fel_vk_get_proc_addr(void);
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/utils/libav.h>
#include <float.h>

static pl_log log_ctx;
static pl_vulkan vk;

int reference_init(void)
{
    log_ctx = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_simple, .log_level = PL_LOG_WARN));
    vk = pl_vulkan_create(log_ctx, pl_vulkan_params(.allow_software = true, .get_proc_addr = fel_vk_get_proc_addr()));
    return vk != NULL;
}

void reference_destroy(void)
{
    pl_vulkan_destroy(&vk);
    pl_log_destroy(&log_ctx);
}

static int reference_pixels_shape(const AVDOVIMetadata *metadata, const float *bl,
                                 const float *el, float *out, int width, int height,
                                 int centered_el)
{
    pl_gpu gpu = vk->gpu;
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                            PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE);
    if (!fmt) return 0;
    pl_tex b = pl_tex_create(gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt, .sampleable = true, .initial_data = bl));
    pl_tex e = pl_tex_create(gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt, .sampleable = true, .initial_data = el));
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt, .renderable = true, .host_readable = true));
    int ok = 0;
    pl_dispatch dp = pl_dispatch_create(log_ctx, gpu);
    pl_shader sh = NULL, eh = NULL;
    if (!b || !e || !dst) goto done;
    struct pl_dovi_metadata dovi = {0};
    pl_map_dovi_metadata(&dovi, metadata);
    // Variante haute précision : le résiduel est centré avant son transfert
    // float32, pour conserver son signe même sous un ulp de l'offset NLQ.
    if (centered_el) for (int c = 0; c < 3; ++c) dovi.nlq[c].offset = 0;
    // Signaux déjà normalisés, mais mêmes offsets qu'une texture YUV 16 bits.
    struct pl_color_repr repr = {
        .sys = PL_COLOR_SYSTEM_DOLBYVISION, .levels = PL_COLOR_LEVELS_FULL,
        .bits = { .sample_depth = 16, .color_depth = 16 }, .dovi = &dovi,
    };
    sh = pl_dispatch_begin(dp);
    eh = pl_shader_alloc(log_ctx, pl_shader_params(.gpu = gpu, .id = 1));
    pl_shader_sample_direct(sh, pl_sample_src(.tex = b));
    pl_shader_sample_direct(eh, pl_sample_src(.tex = e));
    pl_shader_decode_color_ex(sh, pl_color_decode_args(.repr = &repr, .enhancement_layer = eh));
    ok = pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = dst)) &&
         pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = out));
done:
    pl_dispatch_abort(dp, &sh);
    pl_shader_free(&eh);
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &b); pl_tex_destroy(gpu, &e); pl_tex_destroy(gpu, &dst);
    return ok;
}

int reference_pixels(const AVDOVIMetadata *metadata, const float *bl,
                     const float *el, float *out, int count)
{
    return reference_pixels_shape(metadata, bl, el, out, count, 1, 0);
}

static int reference_layer(const AVFrame *frame, int bw, int bh, int enhancement, float *out)
{
    pl_gpu gpu = vk->gpu;
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_LINEAR);
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    if (!fmt || !desc) return 0;
    float *data = calloc(4*bw*bh, sizeof(float));
    float *download = calloc(4*bw*bh, sizeof(float));
    if (!data || !download) { free(data); free(download); return 0; }
    int ok = 1;
    pl_dispatch dp = pl_dispatch_create(log_ctx, gpu);
    for (int c = 0; c < 3 && ok; ++c) {
        const int w = c ? AV_CEIL_RSHIFT(frame->width, desc->log2_chroma_w) : frame->width;
        const int h = c ? AV_CEIL_RSHIFT(frame->height, desc->log2_chroma_h) : frame->height;
        const AVComponentDescriptor *comp = &desc->comp[c];
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const uint8_t *p = frame->data[comp->plane] + y*frame->linesize[comp->plane] + x*comp->step;
            data[4*(y*w+x)] = (p[0]+256*p[1]) / (float)((1 << comp->depth)-1);
        }
        pl_tex src = pl_tex_create(gpu, pl_tex_params(.w = w, .h = h,
            .format = fmt, .sampleable = true, .initial_data = data));
        pl_tex mid = pl_tex_create(gpu, pl_tex_params(.w = w, .h = bh,
            .format = fmt, .sampleable = true, .renderable = true));
        pl_tex dst = pl_tex_create(gpu, pl_tex_params(.w = bw, .h = bh,
            .format = fmt, .renderable = true, .host_readable = true));
        if (!src || !mid || !dst) ok = 0;
        float cx = 0, cy = 0;
        if (c) pl_chroma_location_offset(pl_chroma_from_av(frame->chroma_location), &cx, &cy);
        const float lx = enhancement && frame->width < bw ? -.5f : 0;
        const float sx = lx + cx * bw/frame->width, sy = cy * bh/frame->height;
        const float rx = (float)w/bw, ry = (float)h/bh;
        pl_shader_obj lut = NULL;
        pl_shader sh = NULL;
        if (ok) {
            sh = pl_dispatch_begin(dp);
            ok = pl_shader_sample_ortho2(sh, pl_sample_src(.tex = src, .components = 1,
                .new_w = w, .new_h = bh, .rect = {0, -sy*ry, w, (bh-sy)*ry}),
                pl_sample_filter_params(.filter = pl_filter_spline16, .lut = &lut));
            ok = ok && pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = mid));
        }
        if (ok) {
            sh = pl_dispatch_begin(dp);
            ok = pl_shader_sample_ortho2(sh, pl_sample_src(.tex = mid, .components = 1,
                .new_w = bw, .new_h = bh, .rect = {-sx*rx, 0, (bw-sx)*rx, bh}),
                pl_sample_filter_params(.filter = pl_filter_spline16, .lut = &lut));
            ok = ok && pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = dst));
            ok = ok && pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = download));
        }
        if (ok) for (int i = 0; i < bw*bh; ++i) out[4*i+c] = download[4*i];
        pl_dispatch_abort(dp, &sh); pl_shader_obj_destroy(&lut);
        pl_tex_destroy(gpu, &src); pl_tex_destroy(gpu, &mid); pl_tex_destroy(gpu, &dst);
    }
    pl_dispatch_destroy(&dp);
    free(data); free(download);
    return ok;
}

int reference_frame(const AVDOVIMetadata *metadata, const AVFrame *bl,
                    const AVFrame *el, float *out)
{
    // Même ordre/positionnement que sample_el et pass_read_image, avec des FBO
    // 32 bits : les FBO 16 bits par défaut quantifient les plans avant le NLQ.
    // La référence teste la reconstruction haute précision sans ce plafonnement.
    const int count = bl->width*bl->height;
    float *b = calloc(4*count, sizeof(float)), *e = calloc(4*count, sizeof(float));
    int ok = b && e && reference_layer(bl, bl->width, bl->height, 0, b) &&
        reference_layer(el, bl->width, bl->height, 1, e) &&
        reference_pixels_shape(metadata, b, e, out, bl->width, bl->height, 0);
    free(b); free(e);
    return ok;
}

struct reference_axis { int pos[4]; double weight[4]; };

static void reference_axes(struct reference_axis *axis, int count, int size, double shift)
{
    for (int i = 0; i < count; ++i) {
        const double coordinate = (i+.5-shift)*size/count-.5;
        const int origin = floor(coordinate);
        double sum = 0;
        for (int k = 0; k < 4; ++k) {
            int tap = origin-1+k;
            axis[i].pos[k] = tap < 0 ? 0 : tap >= size ? size-1 : tap;
            // Évaluation indépendante par la bibliothèque de référence,
            // sans reprendre le noyau ni le sampler du moteur testé.
            sum += axis[i].weight[k] = pl_filter_sample(&pl_filter_spline16, coordinate-tap);
        }
        for (int k = 0; k < 4; ++k) axis[i].weight[k] /= sum;
    }
}

static int reference_layer_kernel64(const AVDOVIMetadata *metadata, const AVFrame *frame,
                                    int bw, int bh, int enhancement, float *out)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    if (!desc) return 0;
    struct reference_axis *horizontal = calloc(bw, sizeof(*horizontal));
    struct reference_axis *vertical = calloc(bh, sizeof(*vertical));
    if (!horizontal || !vertical) { free(horizontal); free(vertical); return 0; }
    for (int c = 0; c < 3; ++c) {
        const int w = c ? AV_CEIL_RSHIFT(frame->width, desc->log2_chroma_w) : frame->width;
        const int h = c ? AV_CEIL_RSHIFT(frame->height, desc->log2_chroma_h) : frame->height;
        const AVComponentDescriptor *comp = &desc->comp[c];
        float cx = 0, cy = 0;
        if (c) pl_chroma_location_offset(pl_chroma_from_av(frame->chroma_location), &cx, &cy);
        const double lx = enhancement && frame->width < bw ? -.5 : 0;
        reference_axes(horizontal, bw, w, lx+cx*(double)bw/frame->width);
        reference_axes(vertical, bh, h, cy*(double)bh/frame->height);
        const double maximum = (1 << comp->depth)-1;
        const AVDOVIRpuDataHeader *header = av_dovi_get_header(metadata);
        const AVDOVIDataMapping *mapping = av_dovi_get_mapping(metadata);
        const double offset = enhancement ? mapping->nlq[c].nlq_offset /
            (double)((1 << header->el_bit_depth)-1) : 0;
        for (int y = 0; y < bh; ++y) for (int x = 0; x < bw; ++x) {
            double value = 0;
            for (int dy = 0; dy < 4; ++dy) {
                double row = 0;
                const uint8_t *line = frame->data[comp->plane]+vertical[y].pos[dy]*frame->linesize[comp->plane];
                for (int dx = 0; dx < 4; ++dx) {
                    const uint8_t *p = line+horizontal[x].pos[dx]*comp->step;
                    row += ((p[0]+256*p[1])/maximum-offset)*horizontal[x].weight[dx];
                }
                value += row*vertical[y].weight[dy];
            }
            if (enhancement && fabs(value) <= 32*DBL_EPSILON) value = 0;
            out[4*((size_t)y*bw+x)+c] = value;
        }
    }
    free(horizontal); free(vertical);
    return 1;
}

int reference_frame_kernel64(const AVDOVIMetadata *metadata, const AVFrame *bl,
                             const AVFrame *el, float *out)
{
    // Oracle distinct : noyaux libplacebo évalués en double, shaders officiels
    // mapping/NLQ/couleurs, résiduel centré. Les écarts GPU bruts restent mesurés
    // séparément ; cette variante n'en masque pas les discontinuités float32.
    const size_t count = (size_t)bl->width*bl->height;
    float *b = calloc(4*count, sizeof(float)), *e = calloc(4*count, sizeof(float));
    int ok = b && e && reference_layer_kernel64(metadata, bl, bl->width, bl->height, 0, b) &&
        reference_layer_kernel64(metadata, el, bl->width, bl->height, 1, e) &&
        reference_pixels_shape(metadata, b, e, out, bl->width, bl->height, 1);
    free(b); free(e);
    return ok;
}
