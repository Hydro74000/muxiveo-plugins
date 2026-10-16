// SPDX-License-Identifier: LGPL-2.1-or-later
// Reconstruction Vulkan : aucun Display Management ni tone mapping.
#include <libplacebo/vulkan.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/utils/libav.h>
#include <libavutil/hwcontext_vulkan.h>
#include <stdio.h>
#include "output_format.h"
#include <math.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>

extern PFN_vkGetInstanceProcAddr fel_vk_get_proc_addr(void);

struct fel_gpu {
    pl_log log;
    pl_vulkan vk;
    pl_dispatch dispatch;
    pl_tex source[6], middle[6], scaled[6], target, planar[3];
    pl_shader_obj lut[12];
    pl_tex axes[3][6];
    pl_tex exact_axes[3][4];    /* numérateurs entiers, dénominateurs : x puis y */
    int exact[3];
    /* Composition fusionnée : axes entiers (chroma BL, luma EL, chroma EL ; x puis y). */
    pl_tex fused_axes[6];
    int fused, fused_key[6];
    int axis_w[3], axis_h[3], axis_loc[3];
    int width, height;
    pl_tex dither;
    int dither_mode;
    /* Temps GPU par étape (ns cumulés) : import, BL, résiduel EL, composition, sortie. */
    pl_timer timers[5];
    double gpu_ns[5];
    struct fel_output output;   /* sortie Vulkan directe (fel_gpu_configure) */
    /* Précision (fel_precision) ; double natif disponible sur le périphérique. */
    int precision, float64;
    pl_tex coef;                /* coefficients du RPU en double (render_precise) */
    double coef_values[640];
};

static void collect_timers(struct fel_gpu *p)
{
    for (int k = 0; k < 5; ++k) {
        uint64_t ns;
        while (p->timers[k] && (ns = pl_timer_query(p->vk->gpu, p->timers[k]))) p->gpu_ns[k] += (double)ns;
    }
}

void fel_gpu_timings(void *context, double *milliseconds)
{
    const struct fel_gpu *p = context;
    for (int k = 0; k < 5; ++k) milliseconds[k] = p ? p->gpu_ns[k] / 1e6 : 0;
}

int fel_gpu_device_name(void *context, char *buffer, size_t capacity)
{
    const struct fel_gpu *p = context;
    PFN_vkGetPhysicalDeviceProperties query = (PFN_vkGetPhysicalDeviceProperties)
        p->vk->get_proc_addr(p->vk->instance, "vkGetPhysicalDeviceProperties");
    if (!query) return 0;
    VkPhysicalDeviceProperties properties;
    query(p->vk->phys_device, &properties);
    snprintf(buffer, capacity, "%s", properties.deviceName);
    return 1;
}

void fel_gpu_destroy(void *context)
{
    struct fel_gpu *p = context;
    if (!p) return;
    if (p->vk) {
        for (int i = 0; i < 6; ++i) {
            pl_tex_destroy(p->vk->gpu, &p->source[i]);
            pl_tex_destroy(p->vk->gpu, &p->middle[i]);
            pl_tex_destroy(p->vk->gpu, &p->scaled[i]);
        }
        for (int c=0;c<3;++c) for(int a=0;a<6;++a) pl_tex_destroy(p->vk->gpu, &p->axes[c][a]);
        for (int c=0;c<3;++c) for(int a=0;a<4;++a) pl_tex_destroy(p->vk->gpu, &p->exact_axes[c][a]);
        for (int a=0;a<6;++a) pl_tex_destroy(p->vk->gpu, &p->fused_axes[a]);
        for (int i = 0; i < 12; ++i) pl_shader_obj_destroy(&p->lut[i]);
        pl_tex_destroy(p->vk->gpu, &p->target);
        for (int c=0;c<3;++c) pl_tex_destroy(p->vk->gpu, &p->planar[c]);
        pl_tex_destroy(p->vk->gpu, &p->dither);
        pl_tex_destroy(p->vk->gpu, &p->coef);
        for (int k = 0; k < 5; ++k) pl_timer_destroy(p->vk->gpu, &p->timers[k]);
    }
    pl_dispatch_destroy(&p->dispatch);
    pl_vulkan_destroy(&p->vk);
    pl_log_destroy(&p->log);
    free(p);
}

void *fel_gpu_create(int width, int height, const char *device)
{
    struct fel_gpu *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->width = width; p->height = height;
    p->log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_simple, .log_level = PL_LOG_WARN));
    /* Double natif demandé s'il existe (modes exact et standard). */
    static const VkPhysicalDeviceFeatures2 float64 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                                      .features = {.shaderFloat64 = VK_TRUE}};
    struct pl_vulkan_params params = {.allow_software = false, .get_proc_addr=fel_vk_get_proc_addr(), .features = &float64};
    if (!params.get_proc_addr) goto fail;
    if (device && strcmp(device,"auto")) {
        if (strlen(device)!=32) goto fail;
        for(int i=0;i<16;++i){unsigned byte;
            if(sscanf(device+2*i,"%2x",&byte)!=1)goto fail;
            params.device_uuid[i]=(uint8_t)byte;
        }
    }
    p->vk = pl_vulkan_create(p->log, &params);
    if (!p->vk) goto fail;
    p->float64 = p->vk->features && p->vk->features->features.shaderFloat64;
    p->dispatch = pl_dispatch_create(p->log, p->vk->gpu);
    for (int k = 0; k < 5; ++k) p->timers[k] = pl_timer_create(p->vk->gpu);
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_UNORM, 4, 16, 16, PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE);
    if (!fmt || !p->dispatch) goto fail;
    p->target = pl_tex_create(p->vk->gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt, .renderable = true, .sampleable = true, .host_readable = true));
    if (!p->target) goto fail;
    return p;
fail:
    fel_gpu_destroy(p);
    return NULL;
}

// Poids Spline16 calculés en double sur les axes uniquement, puis transmis en
// deux float32. Le GPU reste portable : aucune exigence shaderFloat64.
static double spline(double x)
{
    x=fabs(x);
    if(x>=2) return 0;
    if(x<1) return ((x-9.0/5.0)*x-1.0/5.0)*x+1;
    x-=1;
    return ((-x/3+4.0/5.0)*x-7.0/15.0)*x;
}

/* Dénominateur minimal rendant les quatre poids entiers (phases multiples de 1/8 :
 * diviseurs de 2560) ; 0 si la phase n'est pas rationnelle de ce type. */
static int exact_weights(const double *w, int *numerators)
{
    static const int divisors[] = {1,2,4,5,8,10,16,20,32,40,64,80,128,160,256,320,512,640,1280,2560};
    for (size_t d = 0; d < sizeof(divisors)/sizeof(*divisors); ++d) {
        int ok = 1, sum = 0;
        for (int k = 0; k < 4 && ok; ++k) {
            const double scaled = w[k] * divisors[d];
            numerators[k] = (int)lround(scaled);
            ok = fabs(scaled - numerators[k]) < 1e-9;
            sum += numerators[k];
        }
        if (ok && sum == divisors[d]) return divisors[d];
    }
    return 0;
}

/* Axes du résiduel EL : indices et poids hi/lo (float-float), et si toutes les phases le
 * permettent, numérateurs/dénominateurs entiers exacts. *weight_sum : max des Σ|num|. */
static int prepare_axis(struct fel_gpu *p, int c, int axis,
                        int count, int size, double ratio, double shift, int *exact, int *weight_sum)
{
    pl_fmt fmt=pl_find_fmt(p->vk->gpu, PL_FMT_FLOAT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    pl_fmt ifmt=pl_find_fmt(p->vk->gpu, PL_FMT_SINT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    float *data=calloc((size_t)count*12,sizeof(float));
    int32_t *integers=calloc((size_t)count*8,sizeof(int32_t));
    if(!fmt || !data || !integers) {free(data);free(integers);return 0;}
    *exact = ifmt != NULL;
    *weight_sum = 0;
    for(int x=0;x<count;++x){
        const double pos=(x+0.5-shift)*ratio-0.5;
        const int first=(int)floor(pos)-1;
        double weights[4],sum=0;
        int numerators[4], total=0;
        for(int k=0;k<4;++k) sum+=weights[k]=spline(pos-first-k);
        for(int k=0;k<4;++k){
            double w=weights[k]/sum;
            int index=first+k;
            weights[k]=w;
            if(index<0)index=0;
            if(index>=size)index=size-1;
            data[4*x+k]=(float)index;
            data[4*count+4*x+k]=(float)w;
            data[8*count+4*x+k]=(float)(w-(float)w);
        }
        const int denominator = exact_weights(weights, numerators);
        if (!denominator) *exact = 0;
        for(int k=0;k<4;++k){
            integers[4*x+k]=numerators[k];
            total+=abs(numerators[k]);
        }
        integers[4*count+4*x]=denominator;
        if (total > *weight_sum) *weight_sum = total;
    }
    int ok=1;
    for(int k=0;k<3;++k){
        pl_tex *tex=&p->axes[c][3*axis+k];
        ok=ok && pl_tex_recreate(p->vk->gpu, tex, pl_tex_params(.w=count,.h=1,
            .format=fmt,.sampleable=true,.host_writable=true));
        ok=ok && pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex=*tex,.ptr=data+4*count*k));
    }
    for(int k=0;k<2 && ok && *exact;++k){
        pl_tex *tex=&p->exact_axes[c][2*axis+k];
        ok=pl_tex_recreate(p->vk->gpu, tex, pl_tex_params(.w=count,.h=1,
            .format=ifmt,.sampleable=true,.host_writable=true)) &&
           pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex=*tex,.ptr=integers+4*count*k));
    }
    free(data);
    free(integers);
    return ok;
}

// Codes EL centrés avant interpolation ; sommes compensées et FMA conservent
// le signe des petits résiduels sans le saut erroné de sign(d)*threshold.
static int scale_residual(struct fel_gpu *p, int i, int w, int h,
                          double rx, double ry, double sx, double sy, int offset, int location)
{
    const int c=i-3;
    if(p->axis_w[c]!=w || p->axis_h[c]!=h || p->axis_loc[c]!=location){
        int exact_x, exact_y, sum_x, sum_y;
        if(!prepare_axis(p,c,0,p->width,w,rx,sx,&exact_x,&sum_x) ||
           !prepare_axis(p,c,1,p->height,h,ry,sy,&exact_y,&sum_y))return 0;
        /* Somme entière bornée : |code - offset| <= 1023, aucun débordement int32. */
        p->exact[c]=exact_x && exact_y && 1023.0*sum_x*sum_y < 2147483647.0;
        p->axis_w[c]=w;p->axis_h[c]=h;p->axis_loc[c]=location;
    }
    if(p->exact[c]){
        /* Phases rationnelles : somme entière exacte, zéro et signe du résiduel exacts. */
        char exact_body[1024];
        snprintf(exact_body,sizeof(exact_body),
            "ivec2 xy=ivec2(gl_FragCoord.xy);"
            "ivec4 ix=ivec4(texelFetch(ax0,ivec2(xy.x,0),0)),iy=ivec4(texelFetch(ax3,ivec2(xy.y,0),0));"
            "ivec4 wx=texelFetch(ex0,ivec2(xy.x,0),0),wy=texelFetch(ey0,ivec2(xy.y,0),0);"
            "int dx=texelFetch(ex1,ivec2(xy.x,0),0).x,dy=texelFetch(ey1,ivec2(xy.y,0),0).x;"
            "int total=0;"
            "for(int y=0;y<4;y++){int row=0;for(int x=0;x<4;x++){"
            "row+=wx[x]*(int(round(texelFetch(raw_el,ivec2(ix[x],iy[y]),0).r*65535.0))-%d);}"
            "total+=wy[y]*row;}"
            "color=vec4(float(total)/(float(dx)*float(dy)*1023.0),0.0,0.0,1.0);",offset);
        const struct pl_shader_desc exact_planes[7]={
            {.desc={.name="raw_el",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->source[i]}},
            {.desc={.name="ax0",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->axes[c][0]}},
            {.desc={.name="ax3",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->axes[c][3]}},
            {.desc={.name="ex0",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->exact_axes[c][0]}},
            {.desc={.name="ex1",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->exact_axes[c][1]}},
            {.desc={.name="ey0",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->exact_axes[c][2]}},
            {.desc={.name="ey1",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->exact_axes[c][3]}}};
        pl_shader sh=pl_dispatch_begin(p->dispatch);
        int ok=pl_shader_custom(sh,&(struct pl_custom_shader){.body=exact_body,.input=PL_SHADER_SIG_NONE,
            .output=PL_SHADER_SIG_COLOR,.descriptors=exact_planes,.num_descriptors=7,
            .output_w=p->width,.output_h=p->height});
        ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.timer=p->timers[2],.shader=&sh,.target=p->scaled[i]));
        pl_dispatch_abort(p->dispatch,&sh);
        return ok;
    }
    char body[2048];
    snprintf(body,sizeof(body),
        "ivec2 xy=ivec2(gl_FragCoord.xy);"
        "ivec4 ix=ivec4(texelFetch(ax0,ivec2(xy.x,0),0));"
        "vec4 hx=texelFetch(ax1,ivec2(xy.x,0),0),lx=texelFetch(ax2,ivec2(xy.x,0),0);"
        "ivec4 iy=ivec4(texelFetch(ax3,ivec2(xy.y,0),0));"
        "vec4 hy=texelFetch(ax4,ivec2(xy.y,0),0),ly=texelFetch(ax5,ivec2(xy.y,0),0);"
        "vec2 total=vec2(0.0);"
        "for(int y=0;y<4;y++){vec2 row=vec2(0.0);for(int x=0;x<4;x++){"
        "float code=round(texelFetch(raw_el,ivec2(ix[x],iy[y]),0).r*65535.0)-float(%d);"
        "row=fel_add(row,fel_mul(vec2(code,0.0),vec2(hx[x],lx[x])));}"
        "total=fel_add(total,fel_mul(row,vec2(hy[y],ly[y])));}"
        "float residual=(total.x+total.y)/1023.0;"
        // Borne d'arrondi double du CPU, pas un seuil perceptuel.
        "if(abs(residual)<=7.105427357601002e-15)residual=0.0;"
        "color=vec4(residual,0.0,0.0,1.0);",offset);
    struct pl_shader_desc planes[7]={
        {.desc={.name="raw_el",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->source[i]}}};
    const char *names[]={"ax0","ax1","ax2","ax3","ax4","ax5"};
    for(int a=0;a<6;++a)planes[a+1]=(struct pl_shader_desc){
        .desc={.name=names[a],.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->axes[c][a]}};
    pl_shader sh=pl_dispatch_begin(p->dispatch);
    int ok=pl_shader_custom(sh,&(struct pl_custom_shader){
        .header=
            "vec2 fel_add(vec2 a,vec2 b){precise float s=a.x+b.x;"
            "precise float v=s-a.x;precise float e=(a.x-(s-v))+(b.x-v)+a.y+b.y;"
            "precise float h=s+e;return vec2(h,e-(h-s));}"
            "vec2 fel_mul(vec2 a,vec2 b){precise float h=a.x*b.x;"
            "precise float l=fma(a.x,b.x,-h)+a.x*b.y+a.y*b.x;"
            "precise float s=h+l;return vec2(s,l-(s-h));}",
        .body=body,.input=PL_SHADER_SIG_NONE,.output=PL_SHADER_SIG_COLOR,
        .descriptors=planes,.num_descriptors=7,.output_w=p->width,.output_h=p->height});
    ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.timer=p->timers[2],.shader=&sh,.target=p->scaled[i]));
    pl_dispatch_abort(p->dispatch,&sh);
    return ok;
}

/* Couche décodée par Vulkan Video (P010 : codes 10 bits en poids fort, chroma entrelacée)
 * recopiée dans les textures R16 du téléversement (codes en poids faible) : le calcul aval
 * reste identique, bit à bit, au parcours du décodage logiciel. */
static int import_layer(struct fel_gpu *p, const AVFrame *frame, int enhancement)
{
    pl_gpu gpu = p->vk->gpu;
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 16, 16, PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE);
    struct pl_frame mapped = {0};
    if (!fmt || !pl_map_avframe_ex(gpu, &mapped, pl_avframe_params(.frame = frame))) return 0;
    int ok = mapped.num_planes == 2 && (!mapped.acquire || mapped.acquire(gpu, &mapped));
    for (int c = 0; c < 3 && ok; ++c) {
        const int i = enhancement*3 + c;
        const int w = c ? AV_CEIL_RSHIFT(frame->width, 1) : frame->width;
        const int h = c ? AV_CEIL_RSHIFT(frame->height, 1) : frame->height;
        const struct pl_plane *plane = &mapped.planes[c ? 1 : 0];
        const struct pl_shader_desc source = {.desc = {.name = "fel_layer", .type = PL_DESC_SAMPLED_TEX},
                                              .binding = {.object = plane->texture}};
        char body[256];
        snprintf(body, sizeof(body),
            "float code = round(texelFetch(fel_layer, ivec2(gl_FragCoord.xy), 0).%c*65535.0/64.0);"
            "color = vec4((code + 0.25)/65535.0, 0.0, 0.0, 1.0);", c == 2 ? 'g' : 'r');
        ok = pl_tex_recreate(gpu, &p->source[i], pl_tex_params(.w = w, .h = h, .format = fmt,
            .sampleable = true, .renderable = true));
        if (!ok) break;
        pl_shader sh = pl_dispatch_begin(p->dispatch);
        ok = pl_shader_custom(sh, &(struct pl_custom_shader){.body = body, .input = PL_SHADER_SIG_NONE,
            .output = PL_SHADER_SIG_COLOR, .descriptors = &source, .num_descriptors = 1,
            .output_w = w, .output_h = h});
        ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[0], .shader = &sh, .target = p->source[i]));
        pl_dispatch_abort(p->dispatch, &sh);
    }
    if (mapped.release) mapped.release(gpu, &mapped);
    pl_unmap_avframe(gpu, &mapped);
    return ok;
}

static int scale_layer(struct fel_gpu *p, const AVFrame *frame, int enhancement, const AVDOVIMetadata *metadata)
{
    pl_gpu gpu = p->vk->gpu;
    const int imported = frame->format == AV_PIX_FMT_VULKAN;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(imported ? AV_PIX_FMT_YUV420P10LE : frame->format);
    pl_fmt source_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    pl_fmt float_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32,
        PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR);
    if (!desc || !source_fmt || !float_fmt) return 0;
    if (imported && !import_layer(p, frame, enhancement)) return 0;
    for (int c = 0; c < 3; ++c) {
        const int i = enhancement*3+c;
        const int w = c ? AV_CEIL_RSHIFT(frame->width, desc->log2_chroma_w) : frame->width;
        const int h = c ? AV_CEIL_RSHIFT(frame->height, desc->log2_chroma_h) : frame->height;
        const AVComponentDescriptor *comp = &desc->comp[c];
        // Le prototype accepte les plans 16 bits contenant les codes HEVC 10 bits.
        if (!imported && (comp->depth != 10 || comp->step != 2 || comp->shift || comp->offset ||
            frame->linesize[comp->plane] <= 0)) return 0;
        if ((!imported && (!pl_tex_recreate(gpu, &p->source[i], pl_tex_params(.w = w, .h = h,
                .format = source_fmt, .sampleable = true, .host_writable = true)) ||
             !pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->source[i],
                .ptr = frame->data[comp->plane], .row_pitch = frame->linesize[comp->plane])))) ||
            !pl_tex_recreate(gpu, &p->middle[i], pl_tex_params(.w = w, .h = p->height,
                .format = float_fmt, .sampleable = true, .renderable = true)) ||
            !pl_tex_recreate(gpu, &p->scaled[i], pl_tex_params(.w = p->width, .h = p->height,
                .format = float_fmt, .sampleable = true, .renderable = true))) return 0;
        float cx = 0, cy = 0;
        if (c) pl_chroma_location_offset(pl_chroma_from_av(frame->chroma_location), &cx, &cy);
        const float lx = enhancement && frame->width < p->width ? -.5f : 0;
        const float sx = lx + cx*p->width/frame->width, sy = cy*p->height/frame->height;
        const float rx = (float)w/p->width, ry = (float)h/p->height;
        if (enhancement) {
            if (!scale_residual(p, i, w, h, rx, ry, sx, sy,
                    av_dovi_get_mapping(metadata)->nlq[c].nlq_offset, frame->chroma_location)) return 0;
            continue;
        }
        if (c == 0 && w == p->width && h == p->height) {
            /* Luma BL à la définition de sortie : Spline16 en phase nulle = identité exacte. */
            const struct pl_shader_desc source = {.desc = {.name = "fel_luma", .type = PL_DESC_SAMPLED_TEX},
                                                  .binding = {.object = p->source[i]}};
            pl_shader sh = pl_dispatch_begin(p->dispatch);
            int ok = pl_shader_custom(sh, &(struct pl_custom_shader){
                .body = "color = vec4(texelFetch(fel_luma, ivec2(gl_FragCoord.xy), 0).r*(65535.0/1023.0), 0.0, 0.0, 1.0);",
                .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR, .descriptors = &source,
                .num_descriptors = 1, .output_w = p->width, .output_h = p->height});
            ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[1], .shader = &sh, .target = p->scaled[i]));
            pl_dispatch_abort(p->dispatch, &sh);
            if (!ok) return 0;
            continue;
        }
        pl_shader sh = pl_dispatch_begin(p->dispatch);
        int ok = pl_shader_sample_ortho2(sh, pl_sample_src(.tex = p->source[i], .components = 1,
            .scale = 65535.f/1023.f, .new_w = w, .new_h = p->height,
            .rect = {0, -sy*ry, w, (p->height-sy)*ry}),
            pl_sample_filter_params(.filter = pl_filter_spline16, .lut = &p->lut[2*i]));
        ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[1], .shader = &sh, .target = p->middle[i]));
        if (ok) {
            sh = pl_dispatch_begin(p->dispatch);
            ok = pl_shader_sample_ortho2(sh, pl_sample_src(.tex = p->middle[i], .components = 1,
                .new_w = p->width, .new_h = p->height,
                .rect = {-sx*rx, 0, (p->width-sx)*rx, p->height}),
                pl_sample_filter_params(.filter = pl_filter_spline16, .lut = &p->lut[2*i+1]));
            ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[1], .shader = &sh, .target = p->scaled[i]));
        }
        pl_dispatch_abort(p->dispatch, &sh);
        if (!ok) return 0;
    }
    return 1;
}

/* Plans 10 bits en textures R16 à codes en poids faible : téléversement du décodage
 * logiciel ou import du décodage Vulkan Video. */
static int load_layer(struct fel_gpu *p, const AVFrame *frame, int enhancement)
{
    pl_gpu gpu = p->vk->gpu;
    if (frame->format == AV_PIX_FMT_VULKAN) return import_layer(p, frame, enhancement);
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    pl_fmt source_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    if (!desc || !source_fmt) return 0;
    for (int c = 0; c < 3; ++c) {
        const AVComponentDescriptor *comp = &desc->comp[c];
        const int w = c ? AV_CEIL_RSHIFT(frame->width, desc->log2_chroma_w) : frame->width;
        const int h = c ? AV_CEIL_RSHIFT(frame->height, desc->log2_chroma_h) : frame->height;
        if (comp->depth != 10 || comp->step != 2 || comp->shift || comp->offset || frame->linesize[comp->plane] <= 0 ||
            !pl_tex_recreate(gpu, &p->source[enhancement*3+c], pl_tex_params(.w = w, .h = h,
                .format = source_fmt, .sampleable = true, .host_writable = true)) ||
            !pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->source[enhancement*3+c],
                .ptr = frame->data[comp->plane], .row_pitch = frame->linesize[comp->plane]))) return 0;
    }
    return 1;
}

/* Axe entier compact (RGBA32I, 3 lignes : indices, numérateurs, dénominateur) ; 0 si une
 * phase n'est pas rationnelle. *sum : max des Σ|numérateurs|. */
static int fused_axis(struct fel_gpu *p, pl_tex *tex, int count, int size, double ratio, double shift, int *sum)
{
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_SINT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    int32_t *data = calloc((size_t)count*12, sizeof(int32_t));
    int ok = fmt && data;
    *sum = 0;
    for (int x = 0; ok && x < count; ++x) {
        const double pos = (x+0.5-shift)*ratio-0.5;
        const int first = (int)floor(pos)-1;
        double weights[4], total = 0;
        int numerators[4], magnitude = 0;
        for (int k = 0; k < 4; ++k) total += weights[k] = spline(pos-first-k);
        for (int k = 0; k < 4; ++k) {
            int index = first+k;
            weights[k] /= total;
            data[4*x+k] = index < 0 ? 0 : index >= size ? size-1 : index;
        }
        const int denominator = exact_weights(weights, numerators);
        ok = denominator > 0;
        for (int k = 0; k < 4; ++k) {
            data[4*count+4*x+k] = numerators[k];
            magnitude += abs(numerators[k]);
        }
        data[8*count+4*x] = denominator;
        if (magnitude > *sum) *sum = magnitude;
    }
    ok = ok && pl_tex_recreate(p->vk->gpu, tex, pl_tex_params(.w = count, .h = 3, .format = fmt,
        .sampleable = true, .host_writable = true)) &&
        pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex = *tex, .ptr = data));
    free(data);
    return ok;
}

/* Axes de la composition fusionnée pour ces géométries et positions chroma ; 1 si toutes les
 * phases sont rationnelles et les sommes entières bornées (int32). */
static int prepare_fused(struct fel_gpu *p, const AVFrame *bl, const AVFrame *el)
{
    const int key[6] = {bl->width, bl->height, el->width, el->height, bl->chroma_location, el->chroma_location};
    if (!memcmp(key, p->fused_key, sizeof(key))) return p->fused;
    memcpy(p->fused_key, key, sizeof(key));
    p->fused = 0;
    if (bl->width != p->width || bl->height != p->height) return 0;
    float bx, by, ex, ey;
    pl_chroma_location_offset(pl_chroma_from_av(bl->chroma_location), &bx, &by);
    pl_chroma_location_offset(pl_chroma_from_av(el->chroma_location), &ex, &ey);
    const int bw = AV_CEIL_RSHIFT(bl->width, 1), bh = AV_CEIL_RSHIFT(bl->height, 1);
    const int ew = el->width, eh = el->height, cw = AV_CEIL_RSHIFT(el->width, 1), ch = AV_CEIL_RSHIFT(el->height, 1);
    const double lx = el->width < p->width ? -.5 : 0;
    int sums[6];
    /* Mêmes formules que scale_layer : décalage = couche + chroma, rapport = plan / sortie. */
    int ok = fused_axis(p, &p->fused_axes[0], p->width, bw, (double)bw/p->width, (double)bx*p->width/bl->width, &sums[0]) &&
             fused_axis(p, &p->fused_axes[1], p->height, bh, (double)bh/p->height, (double)by*p->height/bl->height, &sums[1]) &&
             fused_axis(p, &p->fused_axes[2], p->width, ew, (double)ew/p->width, lx, &sums[2]) &&
             fused_axis(p, &p->fused_axes[3], p->height, eh, (double)eh/p->height, 0, &sums[3]) &&
             fused_axis(p, &p->fused_axes[4], p->width, cw, (double)cw/p->width, lx + (double)ex*p->width/el->width, &sums[4]) &&
             fused_axis(p, &p->fused_axes[5], p->height, ch, (double)ch/p->height, (double)ey*p->height/el->height, &sums[5]);
    /* |code| <= 1023 (BL) et |code - offset| <= 1023 (EL) : aucun débordement int32. */
    for (int a = 0; ok && a < 6; a += 2) ok = 1023.0*sums[a]*sums[a+1] < 2147483647.0;
    p->fused = ok;
    return ok;
}

/* Corps GLSL d'une somme Spline16 entière exacte : axes AX/AY, plans P et offsets O. */
#define FEL_TAPS(AX, AY) \
    "ivec4 ix=texelFetch(" AX ",ivec2(xy.x,0),0),nx=texelFetch(" AX ",ivec2(xy.x,1),0);" \
    "ivec4 iy=texelFetch(" AY ",ivec2(xy.y,0),0),ny=texelFetch(" AY ",ivec2(xy.y,1),0);" \
    "float norm=1.0/(float(texelFetch(" AX ",ivec2(xy.x,2),0).x)*float(texelFetch(" AY ",ivec2(xy.y,2),0).x)*1023.0);"

static int join_fused(struct fel_gpu *p, pl_shader sh, int enhancement, const AVDOVIMetadata *metadata)
{
    char body[2048];
    if (!enhancement) {
        const struct pl_shader_desc planes[5] = {
            {.desc = {.name = "fb0", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[0]}},
            {.desc = {.name = "fb1", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[1]}},
            {.desc = {.name = "fb2", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[2]}},
            {.desc = {.name = "fbx", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[0]}},
            {.desc = {.name = "fby", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[1]}},
        };
        snprintf(body, sizeof(body),
            "ivec2 xy=ivec2(gl_FragCoord.xy);"
            FEL_TAPS("fbx", "fby")
            "int tb=0,tr=0;"
            "for(int b=0;b<4;b++){int rb=0,rr=0;for(int a=0;a<4;a++){ivec2 q=ivec2(ix[a],iy[b]);"
            "rb+=nx[a]*int(round(texelFetch(fb1,q,0).r*65535.0));rr+=nx[a]*int(round(texelFetch(fb2,q,0).r*65535.0));}"
            "tb+=ny[b]*rb;tr+=ny[b]*rr;}"
            "color=vec4(round(texelFetch(fb0,xy,0).r*65535.0)/1023.0,float(tb)*norm,float(tr)*norm,1.0);");
        return pl_shader_custom(sh, &(struct pl_custom_shader){.body = body, .input = PL_SHADER_SIG_NONE,
            .output = PL_SHADER_SIG_COLOR, .descriptors = planes, .num_descriptors = 5,
            .output_w = p->width, .output_h = p->height});
    }
    const AVDOVIDataMapping *mapping = av_dovi_get_mapping(metadata);
    const struct pl_shader_desc planes[7] = {
        {.desc = {.name = "fe0", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[3]}},
        {.desc = {.name = "fe1", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[4]}},
        {.desc = {.name = "fe2", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[5]}},
        {.desc = {.name = "fex", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[2]}},
        {.desc = {.name = "fey", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[3]}},
        {.desc = {.name = "fcx", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[4]}},
        {.desc = {.name = "fcy", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[5]}},
    };
    /* Résiduels centrés avant interpolation : sommes entières, zéro et signe exacts. */
    snprintf(body, sizeof(body),
        "ivec2 xy=ivec2(gl_FragCoord.xy);vec3 residual;"
        "{" FEL_TAPS("fex", "fey")
        "int t=0;for(int b=0;b<4;b++){int r=0;for(int a=0;a<4;a++)"
        "r+=nx[a]*(int(round(texelFetch(fe0,ivec2(ix[a],iy[b]),0).r*65535.0))-%d);t+=ny[b]*r;}"
        "residual.x=float(t)*norm;}"
        "{" FEL_TAPS("fcx", "fcy")
        "int tb=0,tr=0;for(int b=0;b<4;b++){int rb=0,rr=0;for(int a=0;a<4;a++){ivec2 q=ivec2(ix[a],iy[b]);"
        "rb+=nx[a]*(int(round(texelFetch(fe1,q,0).r*65535.0))-%d);rr+=nx[a]*(int(round(texelFetch(fe2,q,0).r*65535.0))-%d);}"
        "tb+=ny[b]*rb;tr+=ny[b]*rr;}"
        "residual.y=float(tb)*norm;residual.z=float(tr)*norm;}"
        "color=vec4(residual,1.0);",
        mapping->nlq[0].nlq_offset, mapping->nlq[1].nlq_offset, mapping->nlq[2].nlq_offset);
    return pl_shader_custom(sh, &(struct pl_custom_shader){.body = body, .input = PL_SHADER_SIG_NONE,
        .output = PL_SHADER_SIG_COLOR, .descriptors = planes, .num_descriptors = 7,
        .output_w = p->width, .output_h = p->height});
}

static int join_layer(struct fel_gpu *p, pl_shader sh, int enhancement)
{
    const struct pl_shader_desc planes[3] = {
        {.desc = {.name = enhancement ? "ep0" : "bp0", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->scaled[3*enhancement]}},
        {.desc = {.name = enhancement ? "ep1" : "bp1", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->scaled[3*enhancement+1]}},
        {.desc = {.name = enhancement ? "ep2" : "bp2", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->scaled[3*enhancement+2]}},
    };
    return pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body = enhancement ?
            "color = vec4(texelFetch(ep0, ivec2(gl_FragCoord.xy), 0).r,"
            "texelFetch(ep1, ivec2(gl_FragCoord.xy), 0).r,"
            "texelFetch(ep2, ivec2(gl_FragCoord.xy), 0).r, 1.0);" :
            "color = vec4(texelFetch(bp0, ivec2(gl_FragCoord.xy), 0).r,"
            "texelFetch(bp1, ivec2(gl_FragCoord.xy), 0).r,"
            "texelFetch(bp2, ivec2(gl_FragCoord.xy), 0).r, 1.0);",
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR,
        .descriptors = planes, .num_descriptors = 3, .output_w = p->width, .output_h = p->height,
    });
}

/* ---- Précision du rendu couleur ----
 * Rapide : décodage Dolby Vision de libplacebo en float32 (dérive aux couleurs extrêmes).
 * Exact et standard : shader Muxiveo. Segment de mapping (pivots) et zéro du résiduel NLQ
 * décidés sur les sommes entières exactes ; mapping, NLQ et matrice Y'CbCr→LMS' en double.
 * Exact : EOTF, matrice LMS→RGB et OETF en double (identique au CPU). Standard : ces étapes
 * en float32 reformulé sans annulation, avec une borne d'erreur par pixel ; au-delà de
 * 0,4 code 16 bits, le pixel est recalculé en double (écart ≤ 1 code 16 bits).
 * Le double natif (shaderFloat64) est exigé : le float-float dépend de l'optimiseur du
 * pilote (termes correcteurs perdus, mesuré sur NVIDIA et AMD). */

#define FEL_CHANNEL 206                       /* coefficients par composante */
#define FEL_GLOBAL (3*FEL_CHANNEL)
#define FEL_COEFFICIENTS (FEL_GLOBAL + 22)    /* 640 doubles, texture 320 × 1 RGBA32UI */
_Static_assert(FEL_COEFFICIENTS == sizeof(((struct fel_gpu *)0)->coef_values) / sizeof(double), "coefficients");

static const double fel_lms_to_rgb[3][3] = {{3.06441879, -2.16597676, 0.10155818},
                                            {-0.65612108, 1.78554118, -0.12943749},
                                            {0.01736321, -0.04725154, 1.03004253}};

/* Mêmes formules que Mapping (CPU). Disposition par composante : nombre de pivots, pivots
 * (codes), bornes normalisées, pente et seuil NLQ, puis 8 segments de 24 valeurs (méthode,
 * ordre MMR, coefficients). Global : résiduel, offsets, matrices Y'CbCr→LMS' et LMS→RGB. */
static int fel_coefficients(const AVDOVIMetadata *metadata, double *v)
{
    const AVDOVIRpuDataHeader *h = av_dovi_get_header(metadata);
    const AVDOVIDataMapping *m = av_dovi_get_mapping(metadata);
    const AVDOVIColorMetadata *c = av_dovi_get_color(metadata);
    if (h->bl_bit_depth != 10 || h->el_bit_depth != 10 || h->coef_log2_denom > 32 ||
        m->num_x_partitions > 1 || m->num_y_partitions > 1 ||
        (!h->disable_residual_flag && m->nlq_method_idc != AV_DOVI_NLQ_LINEAR_DZ)) return 0;
    const double scale = ldexp(1.0, -(int)h->coef_log2_denom);
    for (int i = 0; i < 3; ++i) {
        const AVDOVIReshapingCurve *curve = &m->curves[i];
        const int base = i * FEL_CHANNEL;
        if (curve->num_pivots < 2 || curve->num_pivots > 9) return 0;
        v[base] = curve->num_pivots;
        for (int k = 0; k < curve->num_pivots; ++k) {
            if (k && curve->pivots[k] <= curve->pivots[k-1]) return 0;
            v[base + 1 + k] = curve->pivots[k];
        }
        v[base + 10] = curve->pivots[0] / 1023.0;
        v[base + 11] = curve->pivots[curve->num_pivots - 1] / 1023.0;
        v[base + 12] = m->nlq[i].linear_deadzone_slope * 1023.0 * scale;
        v[base + 13] = ((double)m->nlq[i].linear_deadzone_threshold - 0.5 * m->nlq[i].linear_deadzone_slope) * scale;
        for (int k = 0; k + 1 < curve->num_pivots; ++k) {
            double *piece = v + base + 14 + 24 * k;
            if (curve->mapping_idc[k] == AV_DOVI_MAPPING_POLYNOMIAL) {
                if (curve->poly_order[k] > 2) return 0;
                for (int j = 0; j <= curve->poly_order[k]; ++j) piece[2 + j] = curve->poly_coef[k][j] * scale;
            } else if (curve->mapping_idc[k] == AV_DOVI_MAPPING_MMR) {
                if (curve->mmr_order[k] < 1 || curve->mmr_order[k] > 3) return 0;
                piece[0] = 1;
                piece[1] = curve->mmr_order[k];
                piece[2] = curve->mmr_constant[k] * scale;
                for (int o = 0; o < curve->mmr_order[k]; ++o)
                    for (int j = 0; j < 7; ++j) piece[3 + 7*o + j] = curve->mmr_coef[k][o][j] * scale;
            } else return 0;
        }
    }
    v[FEL_GLOBAL] = !h->disable_residual_flag;
    for (int i = 0; i < 3; ++i) {
        v[FEL_GLOBAL + 1 + i] = av_q2d(c->ycc_to_rgb_offset[i]) * (65536.0 / 65535.0);
        for (int j = 0; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) sum += fel_lms_to_rgb[i][k] * av_q2d(c->rgb_to_lms_matrix[3*k + j]);
            v[FEL_GLOBAL + 4 + 3*i + j] = av_q2d(c->ycc_to_rgb_matrix[3*i + j]);
            v[FEL_GLOBAL + 13 + 3*i + j] = sum;
        }
    }
    for (int k = 0; k < FEL_COEFFICIENTS; ++k) if (!isfinite(v[k])) return 0;
    return 1;
}

/* Doubles exacts (motifs binaires) dans une texture d'entiers, téléversée s'ils changent. */
static int upload_coefficients(struct fel_gpu *p, const AVDOVIMetadata *metadata)
{
    double values[FEL_COEFFICIENTS] = {0};
    if (!fel_coefficients(metadata, values)) return 0;
    if (p->coef && !memcmp(values, p->coef_values, sizeof(values))) return 1;
    memcpy(p->coef_values, values, sizeof(values));
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_UINT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    return fmt && pl_tex_recreate(p->vk->gpu, &p->coef, pl_tex_params(.w = FEL_COEFFICIENTS / 2, .h = 1,
               .format = fmt, .sampleable = true, .host_writable = true)) &&
           pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex = p->coef, .ptr = p->coef_values));
}

/* Double natif : ln (série atanh jusqu'à s^19), e^x = 2^n·(1+u) (réduction par ln 2 puis /32,
 * Taylor jusqu'à r^7, cinq carrés sous forme expm1). Constantes en littéraux décimaux fixes. */
static const char fel_precise_header[] =
    "double fel_d(int i){uvec4 t=texelFetch(fel_coef,ivec2(i>>1,0),0);return packDouble2x32((i&1)==0?t.xy:t.zw);}\n"
    "double fel_pow2(int n){return packDouble2x32(uvec2(0u,uint(clamp(n,-1022,1023)+1023)<<20));}\n"
    "double fel_ln(double x){uvec2 w=unpackDouble2x32(x);int e=int((w.y>>20)&0x7ffu)-1023;"
    "double m=packDouble2x32(uvec2(w.x,(w.y&0x000fffffu)|0x3ff00000u));"
    "if(m>1.4142135623730951LF){m*=0.5LF;e+=1;}"
    "double s=(m-1.0LF)/(m+1.0LF),s2=s*s,acc=1.0LF/19.0LF;"
    "acc=acc*s2+1.0LF/17.0LF;acc=acc*s2+1.0LF/15.0LF;acc=acc*s2+1.0LF/13.0LF;acc=acc*s2+1.0LF/11.0LF;"
    "acc=acc*s2+1.0LF/9.0LF;acc=acc*s2+1.0LF/7.0LF;acc=acc*s2+1.0LF/5.0LF;acc=acc*s2+1.0LF/3.0LF;acc=acc*s2+1.0LF;"
    "return 2.0LF*s*acc+double(e)*0.69314718055994530942LF;}\n"
    "double fel_expm1n(double x,out int n){x=max(x,-700.0LF);double k=floor(x*1.4426950408889634LF+0.5LF);n=int(k);"
    "double r=(x-k*0.69314718055994530942LF)*0.03125LF,acc=1.0LF/5040.0LF;"
    "acc=acc*r+1.0LF/720.0LF;acc=acc*r+1.0LF/120.0LF;acc=acc*r+1.0LF/24.0LF;acc=acc*r+1.0LF/6.0LF;"
    "acc=acc*r+0.5LF;acc=acc*r+1.0LF;double u=acc*r;for(int i=0;i<5;++i)u=2.0LF*u+u*u;return u;}\n"
    "double fel_exp(double x){int n;double u=fel_expm1n(x,n);return (1.0LF+u)*fel_pow2(n);}\n"
    /* PQ EOTF sans annulation : p = 1+u, p-c1 = h+u, c2-c3·p = h-c3·u (h = 1-c1 = c2-c3). */
    "double fel_eotf(double x){if(x<=0.0LF)return 0.0LF;int n;double u=fel_expm1n(fel_ln(x)*(32.0LF/2523.0LF),n);"
    "if(n!=0)u=(1.0LF+u)*fel_pow2(n)-1.0LF;double num=0.1640625LF+u,den=0.1640625LF-18.6875LF*u;"
    "if(num<=0.0LF)return 0.0LF;if(den<=0.0LF)return 1.0e30LF;return fel_exp(fel_ln(num/den)*(8192.0LF/1305.0LF));}\n"
    "double fel_oetf(double y){double u=y>0.0LF?fel_exp(fel_ln(y)*0.1593017578125LF):0.0LF;"
    "return fel_exp(fel_ln((0.8359375LF+18.8515625LF*u)/(1.0LF+18.6875LF*u))*78.84375LF);}\n"
    /* Comparaisons entières exactes sur 64 bits (hi, lo). */
    "uvec2 fel_mul64(uint a,uint b){uint hi,lo;umulExtended(a,b,hi,lo);return uvec2(hi,lo);}\n"
    "bool fel_ge64(uvec2 a,uvec2 b){return a.x>b.x||(a.x==b.x&&a.y>=b.y);}\n"

/* Float32 du mode standard : ln/expm1/log1p précis (≈ 2 ulp ; ln/exp intégrés trop imprécis
 * près de 1). f_eotf rend l'erreur relative estimée de L ; f_oetf la pente dPQ/dy. */
    "float f_ln(float x){uint b=floatBitsToUint(x);int e=int(b>>23)-127;float m=uintBitsToFloat((b&0x007fffffu)|0x3f800000u);"
    "if(m>1.41421356){m*=0.5;e+=1;}float s=(m-1.0)/(m+1.0),s2=s*s;"
    "return 2.0*s*(1.0+s2*(0.33333333+s2*(0.2+s2*(0.14285714+s2*(0.11111111+s2*0.09090909)))))+float(e)*0.69314718;}\n"
    "float f_expm1n(float x,out int n){float k=floor(clamp(x,-87.0,88.0)*1.44269504+0.5);n=int(k);"
    "float r=(x-k*0.69314718)+k*1.9046542e-9;"
    "return r*(1.0+r*(0.5+r*(0.16666667+r*(0.041666667+r*(0.0083333333+r*(0.0013888889+r*(0.00019841270+r*0.000024801587)))))));}\n"
    "float f_exp(float x){int n;float u=f_expm1n(x,n);return ldexp(1.0+u,clamp(n,-126,127));}\n"
    "float f_log1p(float z){float s=z/(2.0+z),s2=s*s;return 2.0*s*(1.0+s2*(0.33333333+s2*(0.2+s2*(0.14285714+s2*0.11111111))));}\n"
    "float f_eotf(float x,float ex,out float rel){rel=0.0;if(x<=0.0){if(x+ex>5.0e-7)rel=1.0e30;return 0.0;}"
    "int n;float t=f_ln(x)*0.012683313515655966;float u=f_expm1n(t,n);if(n!=0)u=ldexp(1.0+u,n)-1.0;"
    "float num=0.1640625+u,den=0.1640625-18.6875*u;if(den<=0.0){rel=1.0e30;return 0.0;}if(num<=0.0)return 0.0;"
    "float Et=(ex/x+5.0e-7)*0.012683313515655966,Eu=Et*(1.0+abs(u))+5.0e-7*abs(u);"
    "float Eln=(Eu+5.0e-7*num)/num+(18.6875*Eu+5.0e-7*(0.1640625+18.6875*abs(u)))/den+5.0e-7;"
    "float lr=f_ln(num/den)*6.277394636015326;rel=Eln*6.277394636015326+5.0e-7*(1.0+abs(lr));return f_exp(lr);}\n"
    "float f_oetf(float y,out float slope){slope=0.0;if(y<=0.0)return 7.3242425e-7;"
    "int n;float t=f_ln(y)*0.1593017578125;float w=f_expm1n(t,n);float u=ldexp(1.0+w,clamp(n,-126,127));"
    "float omu=n==0?-w:1.0-u;float pq=f_exp(78.84375*f_log1p(-0.1640625*omu/(1.0+18.6875*u)));"
    "slope=pq*12.560119628906250*u/y*(18.8515625/(0.8359375+18.8515625*u)-18.6875/(1.0+18.6875*u));return pq;}\n";

/* Somme Spline16 entière : axes AX/AY, dénominateur entier D (Dx·Dy). */
#define FEL_EXACT_TAPS(AX, AY) \
    "ivec4 ix=texelFetch(" AX ",ivec2(xy.x,0),0),nx=texelFetch(" AX ",ivec2(xy.x,1),0);" \
    "ivec4 iy=texelFetch(" AY ",ivec2(xy.y,0),0),ny=texelFetch(" AY ",ivec2(xy.y,1),0);" \
    "int D=texelFetch(" AX ",ivec2(xy.x,2),0).x*texelFetch(" AY ",ivec2(xy.y,2),0).x;"

static int render_precise(struct fel_gpu *p, const AVDOVIMetadata *metadata)
{
    if (!upload_coefficients(p, metadata)) return 0;
    const AVDOVIDataMapping *mapping = av_dovi_get_mapping(metadata);
    const struct pl_shader_desc planes[13] = {
        {.desc = {.name = "fb0", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[0]}},
        {.desc = {.name = "fb1", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[1]}},
        {.desc = {.name = "fb2", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[2]}},
        {.desc = {.name = "fe0", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[3]}},
        {.desc = {.name = "fe1", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[4]}},
        {.desc = {.name = "fe2", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->source[5]}},
        {.desc = {.name = "fbx", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[0]}},
        {.desc = {.name = "fby", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[1]}},
        {.desc = {.name = "fex", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[2]}},
        {.desc = {.name = "fey", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[3]}},
        {.desc = {.name = "fcx", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[4]}},
        {.desc = {.name = "fcy", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->fused_axes[5]}},
        {.desc = {.name = "fel_coef", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->coef}},
    };
    const int H = FEL_CHANNEL, G = FEL_GLOBAL;
    char body[16384];
    int used = snprintf(body, sizeof(body),
        "ivec2 xy=ivec2(gl_FragCoord.xy);"
        "int code=clamp(int(round(texelFetch(fb0,xy,0).r*65535.0)),0,1023);"
        "int tb=0,tr=0,Db,te0=0,te1=0,te2=0,Dl,Dc;"
        "{" FEL_EXACT_TAPS("fbx", "fby")
        "for(int b=0;b<4;b++){int rb=0,rr=0;for(int a=0;a<4;a++){ivec2 q=ivec2(ix[a],iy[b]);"
        "rb+=nx[a]*int(round(texelFetch(fb1,q,0).r*65535.0));rr+=nx[a]*int(round(texelFetch(fb2,q,0).r*65535.0));}"
        "tb+=ny[b]*rb;tr+=ny[b]*rr;}Db=D;}"
        "{" FEL_EXACT_TAPS("fex", "fey")
        "for(int b=0;b<4;b++){int r=0;for(int a=0;a<4;a++)"
        "r+=nx[a]*(int(round(texelFetch(fe0,ivec2(ix[a],iy[b]),0).r*65535.0))-%d);te0+=ny[b]*r;}Dl=D;}"
        "{" FEL_EXACT_TAPS("fcx", "fcy")
        "for(int b=0;b<4;b++){int r1=0,r2=0;for(int a=0;a<4;a++){ivec2 q=ivec2(ix[a],iy[b]);"
        "r1+=nx[a]*(int(round(texelFetch(fe1,q,0).r*65535.0))-%d);r2+=nx[a]*(int(round(texelFetch(fe2,q,0).r*65535.0))-%d);}"
        "te1+=ny[b]*r1;te2+=ny[b]*r2;}Dc=D;}"
        /* Signaux BL bornés à [0, 1023·D] ; segment choisi par comparaison entière exacte. */
        "uvec2 full64=fel_mul64(1023u,uint(Db));uvec2 X64[3];X64[0]=uvec2(0u,uint(code));"
        "X64[1]=uvec2(0u,uint(max(tb,0)));X64[2]=uvec2(0u,uint(max(tr,0)));"
        "for(int c=1;c<3;++c)X64[c]=fel_ge64(X64[c],full64)?full64:X64[c];"
        "int piece[3];"
        "for(int c=0;c<3;++c){int B=c*%d,count=int(fel_d(B)),p=0;"
        "while(p+2<count){uint pc=uint(fel_d(B+2+p));if(!fel_ge64(X64[c],c==0?uvec2(0u,pc):fel_mul64(pc,uint(Db))))break;++p;}"
        "piece[c]=B+14+24*p;}"
        /* Mapping (polynôme, MMR en forme de Horner), NLQ et matrice Y'CbCr→LMS' en double. */
        "double full=1023.0LF*double(Db);"
        "double sig[3];sig[0]=double(code)/1023.0LF;sig[1]=double(X64[1].y)/full;sig[2]=double(X64[2].y)/full;"
        "if(X64[1]==full64)sig[1]=1.0LF;if(X64[2]==full64)sig[2]=1.0LF;"
        "double t7[7];t7[0]=sig[0];t7[1]=sig[1];t7[2]=sig[2];t7[3]=sig[0]*sig[1];t7[4]=sig[0]*sig[2];"
        "t7[5]=sig[1]*sig[2];t7[6]=t7[3]*sig[2];"
        "double res[3];"
        "for(int c=0;c<3;++c){int P=piece[c],B=c*%d;double s;"
        "if(fel_d(P)==0.0LF)s=(fel_d(P+4)*sig[c]+fel_d(P+3))*sig[c]+fel_d(P+2);"
        "else{s=fel_d(P+2);int order=int(fel_d(P+1));"
        "for(int j=0;j<7;++j){double acc=fel_d(P+3+7*(order-1)+j);"
        "for(int k=order-2;k>=0;--k)acc=acc*t7[j]+fel_d(P+3+7*k+j);s+=acc*t7[j];}}"
        "res[c]=clamp(s,fel_d(B+10),fel_d(B+11));}"
        /* Résiduel EL : zéro entier exact, sinon signe, pente et seuil NLQ. */
        "if(fel_d(%d)!=0.0LF){int te[3]=int[3](te0,te1,te2);int De[3]=int[3](Dl,Dc,Dc);"
        "for(int c=0;c<3;++c)if(te[c]!=0){double d=double(te[c])/(1023.0LF*double(De[c]));"
        "res[c]+=sign(d)*(abs(d)*fel_d(c*%d+12)+fel_d(c*%d+13));}}"
        "double v[3];for(int i=0;i<3;++i)v[i]=res[i]-fel_d(%d+1+i);"
        "double xd[3];for(int i=0;i<3;++i)xd[i]=fel_d(%d+4+3*i)*v[0]+fel_d(%d+5+3*i)*v[1]+fel_d(%d+6+3*i)*v[2];"
        "vec3 out3;bool refine=true;",
        mapping->nlq[0].nlq_offset, mapping->nlq[1].nlq_offset, mapping->nlq[2].nlq_offset,
        H, H, G, H, H, G, G, G, G);
    if (p->precision == FEL_PRECISION_STANDARD && used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            /* Float32 avec borne estimée (premier ordre, marges ×8 ; validée par stress, sans
             * preuve formelle) : erreur relative de L (EOTF), annulation de la matrice LMS→RGB,
             * puis PQ. La PQ est concave : sa pente sur [y-e, y+e] est maximale en y-e, d'où la
             * pente prise en y-e ; si y ≤ e, PQ(2e) majore l'écart. */
            "refine=false;float fL[3],fR[3];"
            "for(int i=0;i<3;++i){float x=float(xd[i]),rel;fL[i]=f_eotf(x,1.0e-7*abs(x),rel);fR[i]=rel;"
            "if(rel>1.0e20)refine=true;}"
            "for(int i=0;i<3&&!refine;++i){float y=0.0,ey=0.0;"
            "for(int k=0;k<3;++k){float m=float(fel_d(%d+13+3*i+k));y+=m*fL[k];ey+=abs(m)*fL[k]*(fR[k]+1.0e-6);}"
            "float slope;float pq=f_oetf(y,slope);float err;"
            "if(y>ey){f_oetf(y-ey,slope);err=slope*ey+2.0e-6*pq;}else err=f_oetf(2.0*ey,slope);"
            "if(err*65535.0>0.4)refine=true;out3[i]=(floor(clamp(pq,0.0,1.0)*65535.0+0.5)+0.25)/65535.0;}",
            G);
    if (used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            /* Double : EOTF, matrice LMS→RGB, OETF ; entier 16 bits arrondi au plus proche,
             * écrit avec +0,25 (exact quel que soit l'arrondi UNORM du pilote). */
            "if(refine){double lms[3];for(int i=0;i<3;++i)lms[i]=fel_eotf(xd[i]);"
            "for(int i=0;i<3;++i){double y=fel_d(%d+13+3*i)*lms[0]+fel_d(%d+14+3*i)*lms[1]+fel_d(%d+15+3*i)*lms[2];"
            "out3[i]=(float(floor(clamp(fel_oetf(y),0.0LF,1.0LF)*65535.0LF+0.5LF))+0.25)/65535.0;}}"
            "color=vec4(out3,1.0);",
            G, G, G);
    if (used <= 0 || used >= (int)sizeof(body)) return 0;
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = fel_precise_header, .body = body,
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR, .descriptors = planes, .num_descriptors = 13,
        .output_w = p->width, .output_h = p->height});
    ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[3], .shader = &sh, .target = p->target));
    pl_dispatch_abort(p->dispatch, &sh);
    return ok;
}

int fel_gpu_set_precision(void *context, int precision)
{
    struct fel_gpu *p = context;
    if (precision < FEL_PRECISION_FAST || precision > FEL_PRECISION_EXACT ||
        (precision != FEL_PRECISION_FAST && !p->float64)) return 0;
    p->precision = precision;
    return 1;
}

int fel_gpu_float64(const void *context)
{
    const struct fel_gpu *p = context;
    return p && p->float64;
}

static int render_target(struct fel_gpu *p, const AVDOVIMetadata *metadata,
                         const AVFrame *bl, const AVFrame *el)
{
    /* Phases rationnelles : une seule passe lit les plans (aucune texture 4K intermédiaire). */
    const int fused = (getenv("MVO_FEL_GPU_UNFUSED") == NULL || p->precision != FEL_PRECISION_FAST) &&
                      bl->format == el->format && prepare_fused(p, bl, el);
    /* Modes exact et standard : sommes entières exactes indispensables (phases rationnelles). */
    if (p->precision != FEL_PRECISION_FAST && !fused) return 0;
    if (fused ? !load_layer(p, bl, 0) || !load_layer(p, el, 1)
              : !scale_layer(p, bl, 0, metadata) || !scale_layer(p, el, 1, metadata)) return 0;
    if (p->precision != FEL_PRECISION_FAST) return render_precise(p, metadata);
    struct pl_dovi_metadata dovi = {0};
    pl_map_dovi_metadata(&dovi, metadata);
    for (int c=0;c<3;++c) dovi.nlq[c].offset=0;
    struct pl_color_repr repr = {.sys = PL_COLOR_SYSTEM_DOLBYVISION, .levels = PL_COLOR_LEVELS_FULL,
        .bits = {.sample_depth = 16, .color_depth = 16}, .dovi = &dovi};
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    pl_shader eh = pl_shader_alloc(p->log, pl_shader_params(.gpu = p->vk->gpu, .id = 1));
    int ok = eh && (fused ? join_fused(p, sh, 0, metadata) && join_fused(p, eh, 1, metadata)
                          : join_layer(p, sh, 0) && join_layer(p, eh, 1));
    if (ok) {
        pl_shader_decode_color_ex(sh, pl_color_decode_args(.repr = &repr, .enhancement_layer = eh));
        ok = pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[3], .shader = &sh, .target = p->target));
    }
    pl_dispatch_abort(p->dispatch, &sh);
    pl_shader_free(&eh);
    return ok;
}

int fel_gpu_render(void *context, const AVDOVIMetadata *metadata,
                   const AVFrame *bl, const AVFrame *el, uint16_t *rgba)
{
    struct fel_gpu *p = context;
    return render_target(p, metadata, bl, el) &&
           pl_tex_download(p->vk->gpu, pl_tex_transfer_params(.tex=p->target,.ptr=rgba));
}

static void downloaded(void *unused) { (void)unused; }

// Quantification identique au chemin RGBA16 ; séparation GBR sur GPU et
// rapatriement direct dans le paquet NUT, sans déinterleave ni copie CPU.
int fel_gpu_render_planar(void *context, const AVDOVIMetadata *metadata,
                         const AVFrame *bl, const AVFrame *el, AVFrame *output)
{
    struct fel_gpu *p = context;
    if (!render_target(p, metadata, bl, el)) return 0;
    pl_fmt fmt=pl_find_fmt(p->vk->gpu,PL_FMT_UNORM,1,16,16,PL_FMT_CAP_RENDERABLE);
    if(!fmt)return 0;
    int ok=1;
    for(int c=0;c<3 && ok;++c) {
        ok=pl_tex_recreate(p->vk->gpu,&p->planar[c],pl_tex_params(.w=p->width,.h=p->height,
            .format=fmt,.renderable=true,.host_readable=true));
        if(!ok)break;
        struct pl_shader_desc source={.desc={.name="rgb",.type=PL_DESC_SAMPLED_TEX},
            .binding={.object=p->target}};
        const char *body[]={"color=vec4(texelFetch(rgb,ivec2(gl_FragCoord.xy),0).g,0.0,0.0,1.0);",
                            "color=vec4(texelFetch(rgb,ivec2(gl_FragCoord.xy),0).b,0.0,0.0,1.0);",
                            "color=vec4(texelFetch(rgb,ivec2(gl_FragCoord.xy),0).r,0.0,0.0,1.0);"};
        pl_shader sh=pl_dispatch_begin(p->dispatch);
        ok=pl_shader_custom(sh,&(struct pl_custom_shader){.body=body[c],.input=PL_SHADER_SIG_NONE,
            .output=PL_SHADER_SIG_COLOR,.descriptors=&source,.num_descriptors=1,
            .output_w=p->width,.output_h=p->height});
        ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.timer=p->timers[4],.shader=&sh,.target=p->planar[c]));
        pl_dispatch_abort(p->dispatch,&sh);
        if(ok)ok=pl_tex_download(p->vk->gpu,pl_tex_transfer_params(.tex=p->planar[c],
            .ptr=output->data[c],.row_pitch=output->linesize[c],
            .callback=p->vk->gpu->limits.callbacks ? downloaded : NULL));
    }
    // Les tampons appartiennent au paquet ; même une erreur attend les DMA lancés.
    pl_gpu_finish(p->vk->gpu);
    collect_timers(p);
    return ok;
}

/* Seuils de tramage identiques au CPU (fel_dither_threshold), en texture 64×64. */
static int prepare_dither(struct fel_gpu *p, int mode)
{
    if (p->dither && p->dither_mode == mode) return 1;
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_FLOAT, 1, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    float data[64*64];
    if (!fmt) return 0;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) data[y*64+x] = (float)fel_dither_threshold(mode, x, y);
    if (!pl_tex_recreate(p->vk->gpu, &p->dither, pl_tex_params(.w = 64, .h = 64, .format = fmt,
            .sampleable = true, .host_writable = true)) ||
        !pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex = p->dither, .ptr = data))) return 0;
    p->dither_mode = mode;
    return 1;
}

static int dispatch(struct fel_gpu *p, pl_tex target, const char *body, int w, int h)
{
    const struct pl_shader_desc descriptors[2] = {
        {.desc = {.name = "fel_rgb", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->target}},
        {.desc = {.name = "fel_threshold", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->dither}},
    };
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.body = body, .input = PL_SHADER_SIG_NONE,
        .output = PL_SHADER_SIG_COLOR, .descriptors = descriptors, .num_descriptors = 2,
        .output_w = w, .output_h = h});
    ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[4], .shader = &sh, .target = target));
    pl_dispatch_abort(p->dispatch, &sh);
    return ok;
}

/* Y'CbCr BT.2020nc limité, 4:2:0 bilinéaire au point chroma, quantification tramée :
 * mêmes formules que convert_rows (CPU). Entiers écrits avec +0,25 : exacts que le
 * pilote arrondisse au plus proche ou vers zéro. v = NULL : chroma entrelacée (P010). */
static int convert(struct fel_gpu *p, const struct fel_output *o, pl_tex y, pl_tex u, pl_tex v)
{
    /* Constantes en expressions entières : snprintf("%g") suit LC_NUMERIC (virgule
     * décimale sous une locale française) et casserait le GLSL. */
    const int depth = fel_output_depth(o->format);
    const int scale = 1 << (depth - 8), maximum = (1 << depth) - 1;
    const int numerator = o->format == FEL_OUT_P010 ? 64 : 1, denominator = depth == 8 ? 255 : 65535;
    double ox, oy;
    int wx[4], wy[4], fx, fy;
    char body[4096];
    if (!depth || !prepare_dither(p, o->dither)) return 0;
    fel_chroma_offset(o->chroma, &ox, &oy);
    fx = ox >= 1 ? 0 : -1; fy = oy >= 1 ? 0 : -1;
    /* Poids bilinéaires en seizièmes exacts (0, 2, 4, 6 ou 8). */
    for (int k = 0; k < 4; ++k) {
        wx[k] = (int)lround(16 * fmax(0.0, 1 - fabs(fx + k - ox) / 2) / 2);
        wy[k] = (int)lround(16 * fmax(0.0, 1 - fabs(fy + k - oy) / 2) / 2);
    }
    snprintf(body, sizeof(body),
        "const float U = %d.0/%d.0, B = 0.25/%d.0;"
        "ivec2 xy = ivec2(gl_FragCoord.xy);"
        "vec3 c = texelFetch(fel_rgb, xy + ivec2(%d, %d), 0).rgb;"
        "float l = dot(c, vec3(0.2627, 0.678, 0.0593));"
        "float k = clamp(floor((219.0*l + 16.0)*%d.0 + texelFetch(fel_threshold, xy & 63, 0).r), 0.0, %d.0);"
        "color = vec4(k*U + B, 0.0, 0.0, 1.0);",
        numerator, denominator, denominator, o->crop_x, o->crop_y, scale, maximum);
    /* Mode exact : mêmes formules, même ordre et double comme convert_rows (CPU), sans
     * contraction en FMA (precise) : codes identiques au CPU. */
    const int exact = p->precision == FEL_PRECISION_EXACT;
    if (exact)
        snprintf(body, sizeof(body),
            "const float U = %d.0/%d.0, B = 0.25/%d.0;"
            "ivec2 xy=ivec2(gl_FragCoord.xy);"
            "dvec3 c=dvec3(round(texelFetch(fel_rgb, xy + ivec2(%d, %d), 0).rgb*65535.0))/65535.0LF;"
            "precise double l=0.2627LF*c.r+(1.0LF-0.2627LF-0.0593LF)*c.g+0.0593LF*c.b;"
            "precise double q=(219.0LF*l+16.0LF)*%d.0LF+double(texelFetch(fel_threshold, xy & 63, 0).r);"
            "color = vec4(float(clamp(floor(q),0.0LF,%d.0LF))*U + B, 0.0, 0.0, 1.0);",
            numerator, denominator, denominator, o->crop_x, o->crop_y, scale, maximum);
    if (!dispatch(p, y, body, o->width, o->height)) return 0;
    for (int plane = 0; plane < (v ? 2 : 1); ++plane) {
        const char *store = !v ? "color = vec4(qb*U + B, qr*U + B, 0.0, 1.0);"
                          : plane == 0 ? "color = vec4(qb*U + B, 0.0, 0.0, 1.0);" : "color = vec4(qr*U + B, 0.0, 0.0, 1.0);";
        snprintf(body, sizeof(body),
            "const float U = %d.0/%d.0, B = 0.25/%d.0;"
            "const float WX[4] = float[4](%d.0/16.0, %d.0/16.0, %d.0/16.0, %d.0/16.0);"
            "const float WY[4] = float[4](%d.0/16.0, %d.0/16.0, %d.0/16.0, %d.0/16.0);"
            "ivec2 ij = ivec2(gl_FragCoord.xy);"
            "vec3 s = vec3(0.0);"
            "for (int b = 0; b < 4; b++) for (int a = 0; a < 4; a++) {"
            "  float w = WX[a]*WY[b]; if (w == 0.0) continue;"
            "  ivec2 q = clamp(ivec2(%d + 2*ij.x + %d + a, %d + 2*ij.y + %d + b), ivec2(0), ivec2(%d, %d));"
            "  s += w*texelFetch(fel_rgb, q, 0).rgb; }"
            "float l = dot(s, vec3(0.2627, 0.678, 0.0593));"
            "float qb = clamp(floor((224.0*((s.b - l)/1.8814) + 128.0)*%d.0 + texelFetch(fel_threshold, (ij + ivec2(21, 11)) & 63, 0).r), 0.0, %d.0);"
            "float qr = clamp(floor((224.0*((s.r - l)/1.4746) + 128.0)*%d.0 + texelFetch(fel_threshold, (ij + ivec2(43, 37)) & 63, 0).r), 0.0, %d.0);"
            "%s",
            numerator, denominator, denominator, wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3],
            o->crop_x, fx, o->crop_y, fy, p->width - 1, p->height - 1,
            scale, maximum, scale, maximum, store);
        if (exact)
            snprintf(body, sizeof(body),
                "const float U = %d.0/%d.0, B = 0.25/%d.0;"
                "const double WX[4] = double[4](%d.0LF/16.0LF, %d.0LF/16.0LF, %d.0LF/16.0LF, %d.0LF/16.0LF);"
                "const double WY[4] = double[4](%d.0LF/16.0LF, %d.0LF/16.0LF, %d.0LF/16.0LF, %d.0LF/16.0LF);"
                "ivec2 ij = ivec2(gl_FragCoord.xy);"
                "precise dvec3 s = dvec3(0.0LF);"
                "for (int b = 0; b < 4; b++) for (int a = 0; a < 4; a++) {"
                "  double w = WX[a]*WY[b]; if (w == 0.0LF) continue;"
                "  ivec2 q = clamp(ivec2(%d + 2*ij.x + %d + a, %d + 2*ij.y + %d + b), ivec2(0), ivec2(%d, %d));"
                "  s += w*(dvec3(round(texelFetch(fel_rgb, q, 0).rgb*65535.0))/65535.0LF); }"
                "precise double l = 0.2627LF*s.r+(1.0LF-0.2627LF-0.0593LF)*s.g+0.0593LF*s.b;"
                "precise double cb = (s.b - l)/(2.0LF*(1.0LF-0.0593LF)), cr = (s.r - l)/(2.0LF*(1.0LF-0.2627LF));"
                "precise double vb = (224.0LF*cb + 128.0LF)*%d.0LF + double(texelFetch(fel_threshold, (ij + ivec2(21, 11)) & 63, 0).r);"
                "precise double vr = (224.0LF*cr + 128.0LF)*%d.0LF + double(texelFetch(fel_threshold, (ij + ivec2(43, 37)) & 63, 0).r);"
                "float qb = float(clamp(floor(vb), 0.0LF, %d.0LF)), qr = float(clamp(floor(vr), 0.0LF, %d.0LF));"
                "%s",
                numerator, denominator, denominator, wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3],
                o->crop_x, fx, o->crop_y, fy, p->width - 1, p->height - 1,
                scale, scale, maximum, maximum, store);
        if (!dispatch(p, plane == 0 ? u : v, body, o->width / 2, o->height / 2)) return 0;
    }
    return 1;
}

/* Sortie NUT Y'CbCr planaire : reconstruction, conversion et rapatriement des plans. */
int fel_gpu_render_yuv(void *context, const AVDOVIMetadata *metadata, const AVFrame *bl,
                       const AVFrame *el, const struct fel_output *o, AVFrame *output)
{
    struct fel_gpu *p = context;
    const int depth = fel_output_depth(o->format);
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_UNORM, 1, depth == 8 ? 8 : 16, depth == 8 ? 8 : 16,
                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    int ok = fmt && depth && o->format != FEL_OUT_P010 && render_target(p, metadata, bl, el);
    for (int c = 0; c < 3 && ok; ++c)
        ok = pl_tex_recreate(p->vk->gpu, &p->planar[c], pl_tex_params(.w = c ? o->width/2 : o->width,
            .h = c ? o->height/2 : o->height, .format = fmt, .renderable = true, .host_readable = true));
    ok = ok && convert(p, o, p->planar[0], p->planar[1], p->planar[2]);
    for (int c = 0; c < 3 && ok; ++c)
        ok = pl_tex_download(p->vk->gpu, pl_tex_transfer_params(.tex = p->planar[c],
            .ptr = output->data[c], .row_pitch = output->linesize[c],
            .callback = p->vk->gpu->limits.callbacks ? downloaded : NULL));
    /* Les tampons appartiennent au paquet ; même une erreur attend les DMA lancés. */
    pl_gpu_finish(p->vk->gpu);
    collect_timers(p);
    return ok;
}

void fel_gpu_configure(void *context, const struct fel_output *o)
{
    struct fel_gpu *p = context;
    p->output = *o;
}

/* features : ajoute float64 (liste étendue) ; la liste historique garde ses clés, car des
 * versions publiées de Muxiveo refusent une clé inconnue. */
size_t fel_gpu_devices(char *buffer, size_t capacity, int features)
{
    PFN_vkGetInstanceProcAddr proc=fel_vk_get_proc_addr();
    pl_vk_inst inst=proc ? pl_vk_inst_create(NULL,pl_vk_inst_params(.get_proc_addr=proc)) : NULL;
    if(!inst){if(buffer && capacity>=3)memcpy(buffer,"[]",3);return 3;}
    PFN_vkEnumeratePhysicalDevices enumerate=(PFN_vkEnumeratePhysicalDevices)
        inst->get_proc_addr(inst->instance,"vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties2 props=(PFN_vkGetPhysicalDeviceProperties2)
        inst->get_proc_addr(inst->instance,"vkGetPhysicalDeviceProperties2");
    PFN_vkGetPhysicalDeviceMemoryProperties memory=(PFN_vkGetPhysicalDeviceMemoryProperties)
        inst->get_proc_addr(inst->instance,"vkGetPhysicalDeviceMemoryProperties");
    PFN_vkGetPhysicalDeviceFeatures query_features=(PFN_vkGetPhysicalDeviceFeatures)
        inst->get_proc_addr(inst->instance,"vkGetPhysicalDeviceFeatures");
    uint32_t count=0;
    if(!enumerate || !props || !memory || !query_features || enumerate(inst->instance,&count,NULL)!=VK_SUCCESS || count>64){
        pl_vk_inst_destroy(&inst);return 0;}
    VkPhysicalDevice devices[64];
    if(enumerate(inst->instance,&count,devices)!=VK_SUCCESS){pl_vk_inst_destroy(&inst);return 0;}
    char json[65536];size_t used=0;json[used++]='[';
    int included=0;
    for(uint32_t i=0;i<count;++i){
        VkPhysicalDeviceIDProperties id={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 prop={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&id};
        props(devices[i],&prop);
        const VkPhysicalDeviceProperties *v=&prop.properties;
        if(v->deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU || v->apiVersion<PL_VK_MIN_VERSION)continue;
        char name[2*VK_MAX_PHYSICAL_DEVICE_NAME_SIZE+1],uuid[33];int n=0;
        for(const unsigned char *c=(const unsigned char*)v->deviceName;*c;++c){
            if(*c<32)continue;
            if(*c=='"' || *c=='\\')name[n++]='\\';
            name[n++]=*c;
        }name[n]=0;
        for(int k=0;k<16;++k)snprintf(uuid+2*k,3,"%02x",id.deviceUUID[k]);
        VkPhysicalDeviceMemoryProperties mem;memory(devices[i],&mem);
        VkPhysicalDeviceFeatures supported;query_features(devices[i],&supported);
        uint64_t bytes=0;
        for(uint32_t k=0;k<mem.memoryHeapCount;++k)
            if(mem.memoryHeaps[k].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)bytes+=mem.memoryHeaps[k].size;
        used+=(size_t)snprintf(json+used,sizeof(json)-used,
            "%s{\"uuid\":\"%s\",\"name\":\"%s\",\"vendor\":%u,\"index\":%u,"
            "\"kind\":\"%s\",\"memory_bytes\":%" PRIu64 "%s}",
            included++?",":"",uuid,name,v->vendorID,i,
            v->deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU?"integrated":"discrete",bytes,
            !features ? "" : supported.shaderFloat64 ? ",\"float64\":true" : ",\"float64\":false");
    }
    json[used++]=']';json[used++]=0;
    if(buffer && capacity>=used)memcpy(buffer,json,used);
    pl_vk_inst_destroy(&inst);
    return used;
}

// Contexte emprunté à FFmpeg ; aucune création d'un second périphérique GPU.
static void lock_queue(void *context,uint32_t family,uint32_t index)
{
#if FF_API_VULKAN_SYNC_QUEUES
    AVHWDeviceContext *device=context;
    AVVulkanDeviceContext *vk=device->hwctx;
    if(vk->lock_queue)vk->lock_queue(device,family,index);
#endif
}
static void unlock_queue(void *context,uint32_t family,uint32_t index)
{
#if FF_API_VULKAN_SYNC_QUEUES
    AVHWDeviceContext *device=context;
    AVVulkanDeviceContext *vk=device->hwctx;
    if(vk->unlock_queue)vk->unlock_queue(device,family,index);
#endif
}

void *fel_gpu_import(const void *context,int width,int height)
{
    const AVHWDeviceContext *device=context;
    if(device->type!=AV_HWDEVICE_TYPE_VULKAN)return NULL;
    const AVVulkanDeviceContext *vk=device->hwctx;
    struct fel_gpu *p=calloc(1,sizeof(*p));
    if(!p)return NULL;
    p->width=width;p->height=height;
    p->log=pl_log_create(PL_API_VER,pl_log_params(.log_cb=pl_log_simple,.log_level=PL_LOG_WARN));
    struct pl_vulkan_import_params params={.instance=vk->inst,.get_proc_addr=vk->get_proc_addr,
        .phys_device=vk->phys_dev,.device=vk->act_dev,.extensions=vk->enabled_dev_extensions,
        .num_extensions=vk->nb_enabled_dev_extensions,.features=&vk->device_features,
        .lock_queue=lock_queue,.unlock_queue=unlock_queue,.queue_ctx=(void*)device,
        .queue_graphics={.index=VK_QUEUE_FAMILY_IGNORED},
        .queue_compute={.index=VK_QUEUE_FAMILY_IGNORED},
        .queue_transfer={.index=VK_QUEUE_FAMILY_IGNORED},.max_api_version=VK_API_VERSION_1_3};
    for(int i=0;i<vk->nb_qf;++i) {
        const AVVulkanDeviceQueueFamily *q=&vk->qf[i];
        struct pl_vulkan_queue queue={.index=q->idx,.count=q->num,.flags=vk->queue_flags};
        if(q->flags&VK_QUEUE_GRAPHICS_BIT)params.queue_graphics=queue;
        if(q->flags&VK_QUEUE_COMPUTE_BIT)params.queue_compute=queue;
        if(q->flags&VK_QUEUE_TRANSFER_BIT)params.queue_transfer=queue;
    }
    p->vk=pl_vulkan_import(p->log,&params);
    if(!p->vk)goto fail;
    /* Fonctionnalités activées par FFmpeg (shaderFloat64 s'il existe). */
    p->float64=p->vk->features && p->vk->features->features.shaderFloat64;
    p->dispatch=pl_dispatch_create(p->log,p->vk->gpu);
    for (int k = 0; k < 5; ++k) p->timers[k] = pl_timer_create(p->vk->gpu);
    pl_fmt fmt=pl_find_fmt(p->vk->gpu,PL_FMT_UNORM,4,16,16,PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE);
    if(!fmt || !p->dispatch)goto fail;
    p->target=pl_tex_create(p->vk->gpu,pl_tex_params(.w=width,.h=height,.format=fmt,
        .renderable=true,.sampleable=true));
    if(!p->target)goto fail;
    return p;
fail:
    fel_gpu_destroy(p);
    return NULL;
}

int fel_gpu_render_vulkan(void *context,const AVDOVIMetadata *metadata,
                          const AVFrame *bl,const AVFrame *el,AVFrame *output)
{
    struct fel_gpu *p=context;
    if(output->format!=AV_PIX_FMT_VULKAN || bl->width!=p->width || bl->height!=p->height || !output->hw_frames_ctx)return 0;
    const AVHWFramesContext *frames=(const AVHWFramesContext*)output->hw_frames_ctx->data;
    const int p010=frames->sw_format==AV_PIX_FMT_P010;
    if(frames->sw_format!=AV_PIX_FMT_GBRP16LE && !p010)return 0;
    if(!render_target(p,metadata,bl,el))return 0;
    struct pl_frame target={0};
    if(!pl_map_avframe_ex(p->vk->gpu,&target,&(struct pl_avframe_params){.frame=output}))return 0;
    if(p010) {
        /* Image finale tramée par le plugin : ni filtre libplacebo ni second tramage. */
        int ok=target.num_planes==2 && p->output.format==FEL_OUT_P010 &&
               (!target.acquire || target.acquire(p->vk->gpu,&target));
        ok=ok && convert(p,&p->output,target.planes[0].texture,target.planes[1].texture,NULL);
        if(target.release)target.release(p->vk->gpu,&target);
        pl_gpu_flush(p->vk->gpu);
        pl_unmap_avframe(p->vk->gpu,&target);
        return ok;
    }
    int ok=target.num_planes==3 && (!target.acquire || target.acquire(p->vk->gpu,&target));
    struct pl_shader_desc source={.desc={.name="rgb",.type=PL_DESC_SAMPLED_TEX},.binding={.object=p->target}};
    const char *body[]={"color=vec4(texelFetch(rgb,ivec2(gl_FragCoord.xy),0).g,0.0,0.0,1.0);",
                        "color=vec4(texelFetch(rgb,ivec2(gl_FragCoord.xy),0).b,0.0,0.0,1.0);",
                        "color=vec4(texelFetch(rgb,ivec2(gl_FragCoord.xy),0).r,0.0,0.0,1.0);"};
    for(int c=0;c<3 && ok;++c) {
        pl_shader sh=pl_dispatch_begin(p->dispatch);
        ok=pl_shader_custom(sh,&(struct pl_custom_shader){.body=body[c],.input=PL_SHADER_SIG_NONE,
            .output=PL_SHADER_SIG_COLOR,.descriptors=&source,.num_descriptors=1,
            .output_w=p->width,.output_h=p->height});
        ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.timer=p->timers[4],.shader=&sh,.target=target.planes[c].texture));
        pl_dispatch_abort(p->dispatch,&sh);
    }
    if(target.release)target.release(p->vk->gpu,&target);
    pl_gpu_flush(p->vk->gpu);
    pl_unmap_avframe(p->vk->gpu,&target);
    return ok;
}
