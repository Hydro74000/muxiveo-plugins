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
#include "certified_abi.h"
#include "fixed_abi.h"
#include <math.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>

extern PFN_vkGetInstanceProcAddr fel_vk_get_proc_addr(void);
/* Table des seuils PQ (data/pq_thresholds.bin) : 65 535 parties hautes puis basses. */
extern const unsigned char fel_pq_threshold_bytes[];
/* Shaders GLSL de la virgule fixe (src/shaders, intégrés par CMake). */
extern const char fel_glsl_fixed_point[], fel_glsl_fixed_render[];
/* Reprises par image au plus (0,8 % d'une image UHD) ; au-delà, échec explicite du rendu. */
#define FEL_RESCUE_CAPACITY 65536

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
    double coef_values[674];
    /* Mode exact certifié : seuils PQ (texture 256 × 256 de doubles), liste compacte des
     * composantes indécises, correctifs du CPU certifié, diagnostic (lignes y % diag_stride). */
    int storable;
    pl_tex thresholds;
    pl_buf rescue, patch, diag, left;
    int diag_stride, self_test;
    int force_rescue;           /* tests : toutes les composantes listées ; 1 CPU certifié, 2 second niveau GPU (virgule fixe) */
    /* Variante du mode exact (lot A2') : 0 double, 1 virgule fixe multi-mots ; tables GLSL, seuils
     * en Q0.96, paramètres de l'image (fixed_abi.h). */
    int fixed_mode, fixed_ready;
    uint32_t fixed_tables[FEL_FIXED_TABLE_WORDS];
    pl_tex thr96;
    uint32_t fixed_words[FEL_FIXED_WORDS];
    long long rescued, undecided;
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

static int render_target(struct fel_gpu *p, const AVDOVIMetadata *metadata, const AVFrame *bl, const AVFrame *el);
/* Tests : rendu exact en virgule fixe avec diagnostic de toutes les lignes (FEL_DIAG_WORDS_FIXED
 * mots par pixel dans diag) ; 0 en cas d'échec. */
int fel_gpu_debug_fixed(void *context, const AVDOVIMetadata *metadata, const AVFrame *bl, const AVFrame *el,
                        uint32_t *diag)
{
    struct fel_gpu *p = context;
    const size_t words = (size_t)FEL_DIAG_WORDS_FIXED * p->height * p->width;
    p->fixed_mode = 1; p->diag_stride = 1;
    p->diag = pl_buf_create(p->vk->gpu, pl_buf_params(.size = 4 * words, .storable = true, .host_readable = true));
    const int ok = p->diag && fel_fixed_params(metadata, p->fixed_words, FEL_FIXED_WORDS) &&
                   render_target(p, metadata, bl, el) && pl_buf_read(p->vk->gpu, p->diag, 0, diag, 4 * words);
    pl_buf_destroy(p->vk->gpu, &p->diag);
    p->diag_stride = 0;
    return ok;
}

/* Tests : toutes les composantes du mode exact reprises par le CPU certifié (chaîne de reprise). */
void fel_gpu_force_rescue(void *context, int enabled)
{
    ((struct fel_gpu *)context)->force_rescue = enabled == 2 ? 2 : enabled != 0;
}

/* Contexte de rendu supplémentaire (lot P) : reprend la précision, la variante et le résultat de
 * l'auto-test d'un contexte du même périphérique, sans relancer l'auto-test. 0 si incompatible. */
int fel_gpu_inherit(void *context, const void *source)
{
    struct fel_gpu *p = context;
    const struct fel_gpu *s = source;
    if (!p || !s || (s->precision != FEL_PRECISION_FAST && !p->float64) ||
        (s->precision == FEL_PRECISION_EXACT && !p->storable)) return 0;
    p->precision = s->precision;
    p->self_test = s->self_test;
    p->fixed_mode = s->fixed_mode;
    p->force_rescue = s->force_rescue;
    return 1;
}

/* Variante du mode exact retenue par l'auto-test : 1 virgule fixe, 0 double. */
int fel_gpu_fixed_variant(const void *context)
{
    const struct fel_gpu *p = context;
    return p && p->fixed_mode == 1;
}

/* Mode exact : pixels repris par le CPU certifié et composantes indécises, cumulés. */
void fel_gpu_certified(void *context, long long *rescued, long long *undecided)
{
    const struct fel_gpu *p = context;
    *rescued = p ? p->rescued : 0;
    *undecided = p ? p->undecided : 0;
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
        pl_tex_destroy(p->vk->gpu, &p->thresholds);
        pl_tex_destroy(p->vk->gpu, &p->thr96);
        pl_buf_destroy(p->vk->gpu, &p->rescue);
        pl_buf_destroy(p->vk->gpu, &p->patch);
        pl_buf_destroy(p->vk->gpu, &p->left);
        pl_buf_destroy(p->vk->gpu, &p->diag);
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
    /* Image stockable : rendu exact en shader de calcul et correctifs du CPU certifié. */
    p->storable = (fmt->caps & PL_FMT_CAP_STORABLE) && p->vk->gpu->glsl.compute;
    p->target = pl_tex_create(p->vk->gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt, .renderable = true, .sampleable = true, .host_readable = true,
        .storable = p->storable));
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
#define FEL_EXTRA (FEL_GLOBAL + 22)           /* poids d'erreur des segments (3 × 8), puis ΔM (9) */
#define FEL_COEFFICIENTS (FEL_EXTRA + 34)     /* 674 doubles, texture 337 × 1 RGBA32UI */
_Static_assert(FEL_COEFFICIENTS == sizeof(((struct fel_gpu *)0)->coef_values) / sizeof(double), "coefficients");

static const double fel_lms_to_rgb[3][3] = {{3.06441879, -2.16597676, 0.10155818},
                                            {-0.65612108, 1.78554118, -0.12943749},
                                            {0.01736321, -0.04725154, 1.03004253}};

/* Mêmes formules que Mapping (CPU). Disposition par composante : nombre de pivots, pivots
 * (codes), bornes normalisées, pente et seuil NLQ, puis 8 segments de 24 valeurs (méthode,
 * ordre MMR, coefficients). Global : résiduel, offsets, matrices Y'CbCr→LMS' et LMS→RGB.
 * Extra (mode exact) : poids d'erreur des segments comme Mapping::weight (polynôme
 * 15·Σ|a|, MMR 80·(|c| + Σ|m|)) et erreur ΔM = 3,01·2^-53·Σ|A·B| du produit LMS→RGB. */
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
                v[FEL_EXTRA + 8*i + k] = 15 * (fabs(piece[2]) + fabs(piece[3]) + fabs(piece[4]));
            } else if (curve->mapping_idc[k] == AV_DOVI_MAPPING_MMR) {
                if (curve->mmr_order[k] < 1 || curve->mmr_order[k] > 3) return 0;
                piece[0] = 1;
                piece[1] = curve->mmr_order[k];
                piece[2] = curve->mmr_constant[k] * scale;
                double magnitude = fabs(piece[2]);
                for (int o = 0; o < curve->mmr_order[k]; ++o)
                    for (int j = 0; j < 7; ++j) magnitude += fabs(piece[3 + 7*o + j] = curve->mmr_coef[k][o][j] * scale);
                v[FEL_EXTRA + 8*i + k] = 80 * magnitude;
            } else return 0;
        }
    }
    v[FEL_GLOBAL] = !h->disable_residual_flag;
    for (int i = 0; i < 3; ++i) {
        v[FEL_GLOBAL + 1 + i] = av_q2d(c->ycc_to_rgb_offset[i]) * (65536.0 / 65535.0);
        for (int j = 0; j < 3; ++j) {
            double sum = 0, magnitude = 0;
            for (int k = 0; k < 3; ++k) {
                sum += fel_lms_to_rgb[i][k] * av_q2d(c->rgb_to_lms_matrix[3*k + j]);
                magnitude += fabs(fel_lms_to_rgb[i][k] * av_q2d(c->rgb_to_lms_matrix[3*k + j]));
            }
            v[FEL_GLOBAL + 4 + 3*i + j] = av_q2d(c->ycc_to_rgb_matrix[3*i + j]);
            v[FEL_GLOBAL + 13 + 3*i + j] = sum;
            v[FEL_EXTRA + 24 + 3*i + j] = 3.01 * ldexp(1.0, -53) * magnitude;
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
static const char fel_double_header[] =
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
    /* Mode exact certifié : EOTF, conditionnement κ = d ln L / d ln x et majorant |log2 L| + 1 ;
     * seuils PQ arrondis vers le bas (texture fel_thr). */
    "double fel_eotf_k(double x,out double kappa,out double mag,out bool pole){kappa=0.0LF;mag=1.0LF;pole=false;"
    "if(x<=0.0LF)return 0.0LF;int n;double u=fel_expm1n(fel_ln(x)*(32.0LF/2523.0LF),n);if(n!=0)u=(1.0LF+u)*fel_pow2(n)-1.0LF;"
    "double num=0.1640625LF+u,den=0.1640625LF-18.6875LF*u;if(den<=0.0LF){pole=true;return 0.0LF;}"
    "if(num<=0.0LF){kappa=1.0e300LF;return 0.0LF;}kappa=(1.0LF+u)/12.56011962890625LF*(1.0LF/num+18.6875LF/den);"
    "double t=fel_ln(num/den)*(8192.0LF/1305.0LF);mag=abs(t)*1.4426950408889634LF+1.0LF;return fel_exp(t);}\n"
    "double fel_T(int k){return packDouble2x32(texelFetch(fel_thr,ivec2(k&255,k>>8),0).xy);}\n"
    ;

/* Partie commune sans double (tous les GPU, mode rapide compris). */
static const char fel_common_header[] =
    /* Mode rapide : EOTF et OETF float32 sans annulation, à coût réduit. EOTF : p - 1 = expm1(ln(x)/m2)
     * par polynôme (|t| ≤ 0,18 utile, reste < 3e-11), p - c1 = h + u, c2 - c3·p = h - c3·u ; OETF :
     * (c1 + c2·v)/(1 + c3·v) = 1 - h(1 - v)/(1 + c3·v), puis exp(m2·log1p(…)). log2/exp2/pow natifs
     * là où l'erreur relative suffit. */
    "float g_eotf(float x){if(x<=0.0)return 0.0;float t=log2(x)*0.0087914030;if(t<-0.18)return 0.0;"
    "float u=t*(1.0+t*(0.5+t*(0.16666667+t*(0.041666668+t*(0.0083333338+t*(0.0013888889+t*0.00019841270))))));"
    "float num=0.1640625+u,den=0.1640625-18.6875*u;if(den<=0.0)return 1.0e30;if(num<=0.0)return 0.0;"
    "return exp2(log2(num/den)*6.277394636015326);}\n"
    "float g_log1p(float z){float s=z/(2.0+z),s2=s*s;return 2.0*s*(1.0+s2*(0.33333333+s2*(0.2+s2*(0.14285714+s2*0.11111111))));}\n"
    "float g_oetf(float y){if(y<=0.0)return 7.3095590e-7;float v=pow(y,0.1593017578125);"
    "return exp2(113.74749*g_log1p(-0.1640625*(1.0-v)/(1.0+18.6875*v)));}\n"
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

/* Seuils PQ en texture RG32UI 256 × 256 (dernier texel : +inf), liste de reprises et correctifs. */
static int prepare_exact(struct fel_gpu *p)
{
    if (p->thresholds && p->rescue && p->patch && p->left) return 1;
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_UINT, 2, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    uint32_t *data = malloc(65536 * 2 * sizeof(uint32_t));
    if (!fmt || !data) { free(data); return 0; }
    memcpy(data, fel_pq_threshold_bytes, 65535 * sizeof(double));
    data[2*65535] = 0; data[2*65535 + 1] = 0x7ff00000u;
    int ok = pl_tex_recreate(p->vk->gpu, &p->thresholds, pl_tex_params(.w = 256, .h = 256, .format = fmt,
                 .sampleable = true, .host_writable = true)) &&
             pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex = p->thresholds, .ptr = data));
    free(data);
    if (!ok) return 0;
    p->rescue = pl_buf_create(p->vk->gpu, pl_buf_params(.size = 4 + 4 * FEL_RESCUE_WORDS * (size_t)FEL_RESCUE_CAPACITY,
                                                        .storable = true, .host_readable = true, .host_writable = true));
    p->patch = pl_buf_create(p->vk->gpu, pl_buf_params(.size = 12 * (size_t)FEL_RESCUE_CAPACITY,
                                                       .storable = true, .host_readable = true, .host_writable = true));
    p->left = pl_buf_create(p->vk->gpu, pl_buf_params(.size = 4, .storable = true, .host_readable = true,
                                                      .host_writable = true));
    return p->rescue && p->patch && p->left;
}

static struct pl_buffer_var uint_array(const char *name, int count, size_t offset)
{
    struct pl_buffer_var var = {.var = {.name = name, .type = PL_VAR_UINT, .dim_v = 1, .dim_m = 1, .dim_a = count}};
    var.layout = pl_std430_layout(offset, &var.var);
    return var;
}

static int precise_descriptors(struct fel_gpu *p, struct pl_shader_desc *d, struct pl_buffer_var *rescue_vars,
                               enum pl_desc_access rescue_access);

/* Correctifs du CPU certifié écrits dans l'image cible (shader de calcul, imageStore) : premier
 * pixel de chaque plage listée, puis les pixels de droite aux entrées identiques, tant que l'écart
 * avec le dernier trouvé ne dépasse pas 4 (règle de liste du rendu exact). */
static int apply_patch(struct fel_gpu *p, const char *header, const uint32_t *patch, uint32_t count)
{
    if (patch) pl_buf_write(p->vk->gpu, p->patch, 0, patch, 12 * (size_t)count);   /* sinon : déjà sur le GPU */
    struct pl_buffer_var rescue_vars[2], var = uint_array("fel_patch", 3 * FEL_RESCUE_CAPACITY, 0);
    struct pl_shader_desc desc[17];
    precise_descriptors(p, desc, rescue_vars, PL_DESC_ACCESS_READONLY);
    desc[15] = (struct pl_shader_desc){.desc = {.name = "fel_target", .type = PL_DESC_STORAGE_IMG,
        .access = PL_DESC_ACCESS_WRITEONLY}, .binding = {.object = p->target}};
    desc[16] = (struct pl_shader_desc){.desc = {.name = "fel_patch_buf", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_READONLY}, .binding = {.object = p->patch}, .buffer_vars = &var, .num_buffer_vars = 1};
    const struct pl_shader_var number = {.var = {.name = "fel_patch_count", .type = PL_VAR_UINT, .dim_v = 1,
                                                 .dim_m = 1, .dim_a = 1}, .data = &count, .dynamic = true};
    char body[2048];
    snprintf(body, sizeof(body),
        "uint i=gl_GlobalInvocationID.x;if(i<fel_patch_count){uint b=%du*i,w=fel_patch[3u*i+1u];"
        "ivec2 xy=ivec2(int(fel_rescue[b]&0xffffu),int(fel_rescue[b]>>16));"
        "vec4 v=vec4((vec3(float(w&0xffffu),float(w>>16),float(fel_patch[3u*i+2u]))+0.25)/65535.0,1.0);"
        "imageStore(fel_target,xy,v);"
        "int last=xy.x;for(int x=xy.x+1;x<%d&&x-last<=4;++x){int c1,b1,r1,d1,f0,f1,f2,l1,k1;"
        "fel_inputs(ivec2(x,xy.y),c1,b1,r1,d1,f0,f1,f2,l1,k1);"
        "if(uint(c1)==(fel_rescue[b+1u]&0xffffu)&&uint(b1)==fel_rescue[b+2u]&&uint(r1)==fel_rescue[b+3u]&&"
        "uint(d1)==fel_rescue[b+4u]&&uint(f0)==fel_rescue[b+5u]&&uint(f1)==fel_rescue[b+6u]&&uint(f2)==fel_rescue[b+7u]&&"
        "uint(l1)==fel_rescue[b+8u]&&uint(k1)==fel_rescue[b+9u]){imageStore(fel_target,ivec2(x,xy.y),v);last=x;}}}",
        FEL_RESCUE_WORDS, p->width);
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = header, .body = body,
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_NONE, .descriptors = desc, .num_descriptors = 17,
        .variables = &number, .num_variables = 1, .compute = true, .compute_group_size = {64, 1}});
    ok = ok && pl_dispatch_compute(p->dispatch, pl_dispatch_compute_params(.shader = &sh,
                                   .dispatch_size = {(int)((count + 63) / 64), 1, 1}));
    pl_dispatch_abort(p->dispatch, &sh);
    return ok;
}

/* Composantes indécises du rendu exact : liste rapatriée, codes du CPU certifié, correctifs. */
static int resolve_rescues(struct fel_gpu *p, const AVDOVIMetadata *metadata, const char *header)
{
    uint32_t count = 0;
    if (!pl_buf_read(p->vk->gpu, p->rescue, 0, &count, sizeof(count))) return 0;
    if (count > FEL_RESCUE_CAPACITY) {
        fprintf(stderr, "mvo-fel : %u composantes à reprendre, capacité %d dépassée\n", count, FEL_RESCUE_CAPACITY);
        return 0;
    }
    if (!count) return 1;
    uint32_t *entries = malloc(4 * FEL_RESCUE_WORDS * (size_t)count), *patch = malloc(12 * (size_t)count);
    uint16_t *codes = malloc(6 * (size_t)count);
    int ok = entries && patch && codes &&
             pl_buf_read(p->vk->gpu, p->rescue, 4, entries, 4 * FEL_RESCUE_WORDS * (size_t)count);
    const int undecided = ok ? fel_certified_codes(metadata, entries, (int)count, codes) : -1;
    ok = ok && undecided >= 0;
    for (uint32_t i = 0; ok && i < count; ++i) {
        patch[3*i] = entries[FEL_RESCUE_WORDS * i];
        patch[3*i + 1] = codes[3*i] | (uint32_t)codes[3*i + 1] << 16;
        patch[3*i + 2] = codes[3*i + 2];
    }
    ok = ok && apply_patch(p, header, patch, count);
    if (ok) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t mask = entries[FEL_RESCUE_WORDS * i + 1] >> 28;
            p->rescued += (mask & 1) + (mask >> 1 & 1) + (mask >> 2 & 1);
        }
        p->undecided += undecided;
    }
    free(entries); free(patch); free(codes);
    return ok;
}

/* En-tête du rendu précis : fonctions communes et entrées exactes d'un pixel (code BL Y, sommes
 * Spline16 entières et dénominateurs), offsets NLQ de l'image. */
#define FEL_HEADER_SIZE (sizeof(fel_double_header) + sizeof(fel_common_header) + 4096)
/* prefix : fonctions propres au mode (double, float32 du mode rapide, virgule fixe) ; suffix : après
 * fel_inputs (rendu en virgule fixe). */
static int precise_header(char *header, size_t size, const AVDOVIMetadata *metadata, const char *prefix,
                          const char *suffix)
{
    const AVDOVIDataMapping *mapping = av_dovi_get_mapping(metadata);
    const int used = snprintf(header, size, "%s%s"
        "void fel_inputs(ivec2 xy,out int code,out int tb,out int tr,out int Db,out int te0,out int te1,out int te2,"
        "out int Dl,out int Dc){code=clamp(int(round(texelFetch(fb0,xy,0).r*65535.0)),0,1023);"
        "tb=0;tr=0;te0=0;te1=0;te2=0;"
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
        "te1+=ny[b]*r1;te2+=ny[b]*r2;}Dc=D;}}\n%s",
        prefix, fel_common_header,
        mapping->nlq[0].nlq_offset, mapping->nlq[1].nlq_offset, mapping->nlq[2].nlq_offset, suffix);
    return used > 0 && used < (int)size;
}

/* Descripteurs du rendu précis (sources, axes entiers, coefficients, seuils), puis liste de reprises. */
static int precise_descriptors(struct fel_gpu *p, struct pl_shader_desc *d, struct pl_buffer_var *rescue_vars,
                               enum pl_desc_access rescue_access)
{
    static const char *names[12] = {"fb0", "fb1", "fb2", "fe0", "fe1", "fe2", "fbx", "fby", "fex", "fey", "fcx", "fcy"};
    for (int i = 0; i < 12; ++i)
        d[i] = (struct pl_shader_desc){.desc = {.name = names[i], .type = PL_DESC_SAMPLED_TEX},
                                       .binding = {.object = i < 6 ? p->source[i] : p->fused_axes[i - 6]}};
    d[12] = (struct pl_shader_desc){.desc = {.name = "fel_coef", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->coef}};
    d[13] = (struct pl_shader_desc){.desc = {.name = "fel_thr", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->thresholds}};
    rescue_vars[0] = uint_array("fel_rescue_count", 1, 0);
    rescue_vars[1] = uint_array("fel_rescue", FEL_RESCUE_WORDS * FEL_RESCUE_CAPACITY,
                                rescue_vars[0].layout.offset + rescue_vars[0].layout.size);
    d[14] = (struct pl_shader_desc){.desc = {.name = "fel_rescue_buf", .type = PL_DESC_BUF_STORAGE, .access = rescue_access},
                                    .binding = {.object = p->rescue}, .buffer_vars = rescue_vars, .num_buffer_vars = 2};
    return 15;
}

/* Fin du corps du rendu exact (variables code, tb, …, seg, mask, out3, inside, xy) : reprise forcée
 * (tests), liste compacte des composantes indécises, sortie. */
static int rescue_epilogue(const struct fel_gpu *p, char *body, size_t size, int used)
{
    if (used <= 0 || used >= (int)size) return -1;
    return used + snprintf(body + used, size - used,
        "if(%d!=0)mask=7u;"
        /* Zone uniforme : un pixel dont les entrées brutes égalent celles d'un voisin de gauche à
         * distance ≤ 4 (phases Spline16 de période ≤ 4 : chroma EL suréchantillonnée ×4) a le même
         * calcul, donc le même masque : il n'est pas listé, la passe de correctifs propage le code
         * du premier pixel de la plage vers la droite. */
        "bool head=true;for(int s=1;s<=4&&head&&mask!=0u;++s)if(xy.x>=s){int c1,b1,r1,d1,f0,f1,f2,l1,k1;"
        "fel_inputs(xy-ivec2(s,0),c1,b1,r1,d1,f0,f1,f2,l1,k1);"
        "head=!(c1==code&&b1==tb&&r1==tr&&d1==Db&&f0==te0&&f1==te1&&f2==te2&&l1==Dl&&k1==Dc);}"
        "if(mask!=0u&&inside&&head){uint i=atomicAdd(fel_rescue_count,1u);if(i<%du){uint b=%du*i;"
        "fel_rescue[b]=uint(xy.x)|(uint(xy.y)<<16);"
        "fel_rescue[b+1u]=uint(code)|(uint(seg[0])<<16)|(uint(seg[1])<<20)|(uint(seg[2])<<24)|(mask<<28);"
        "fel_rescue[b+2u]=uint(tb);fel_rescue[b+3u]=uint(tr);fel_rescue[b+4u]=uint(Db);fel_rescue[b+5u]=uint(te0);"
        "fel_rescue[b+6u]=uint(te1);fel_rescue[b+7u]=uint(te2);fel_rescue[b+8u]=uint(Dl);fel_rescue[b+9u]=uint(Dc);}}"
        "color=vec4(out3,1.0);",
        p->force_rescue, FEL_RESCUE_CAPACITY, FEL_RESCUE_WORDS);
}

static int render_precise(struct fel_gpu *p, const AVDOVIMetadata *metadata)
{
    const int exact = p->precision == FEL_PRECISION_EXACT;
    if (!upload_coefficients(p, metadata) || !prepare_exact(p)) return 0;
    static const uint32_t zero = 0;
    if (exact) pl_buf_write(p->vk->gpu, p->rescue, 0, &zero, sizeof(zero));
    char header[FEL_HEADER_SIZE];
    if (!precise_header(header, sizeof(header), metadata, fel_double_header, "")) return 0;
    struct pl_buffer_var rescue_vars[2];
    const int diag_rows = p->diag_stride > 0 ? (p->height + p->diag_stride - 1) / p->diag_stride : 0;
    struct pl_buffer_var diag_var = uint_array("fel_diag", diag_rows ? FEL_DIAG_WORDS * diag_rows * p->width : 1, 0);
    struct pl_shader_desc planes[16];
    precise_descriptors(p, planes, rescue_vars, PL_DESC_ACCESS_READWRITE);
    planes[15] = (struct pl_shader_desc){.desc = {.name = "fel_diag_buf", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_WRITEONLY}, .binding = {.object = p->diag}, .buffer_vars = &diag_var, .num_buffer_vars = 1};
    const int H = FEL_CHANNEL, G = FEL_GLOBAL, E = FEL_EXTRA;
    char body[24576];
    int used = snprintf(body, sizeof(body),
        "ivec2 xy0=ivec2(gl_FragCoord.xy);bool inside=xy0.x<%d&&xy0.y<%d;ivec2 xy=min(xy0,ivec2(%d,%d));"
        "int code,tb,tr,Db,te0,te1,te2,Dl,Dc;fel_inputs(xy,code,tb,tr,Db,te0,te1,te2,Dl,Dc);"
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
        /* Résiduel EL : zéro entier exact, sinon signe, pente et seuil NLQ (en : poids d'erreur). */
        "double en[3]=double[3](0.0LF,0.0LF,0.0LF);"
        "if(fel_d(%d)!=0.0LF){int te[3]=int[3](te0,te1,te2);int De[3]=int[3](Dl,Dc,Dc);"
        "for(int c=0;c<3;++c)if(te[c]!=0){double d=double(te[c])/(1023.0LF*double(De[c]));"
        "res[c]+=sign(d)*(abs(d)*fel_d(c*%d+12)+fel_d(c*%d+13));en[c]=abs(d)*abs(fel_d(c*%d+12))+abs(fel_d(c*%d+13));}}"
        "double v[3];for(int i=0;i<3;++i)v[i]=res[i]-fel_d(%d+1+i);"
        "double xd[3];for(int i=0;i<3;++i)xd[i]=fel_d(%d+4+3*i)*v[0]+fel_d(%d+5+3*i)*v[1]+fel_d(%d+6+3*i)*v[2];"
        "vec3 out3;",
        p->width, p->height, p->width - 1, p->height - 1,
        H, H, G, H, H, H, H, G, G, G, G);
    if (!exact && used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            /* Float32 avec borne estimée (premier ordre, marges ×8 ; validée par stress, sans
             * preuve formelle) : erreur relative de L (EOTF), annulation de la matrice LMS→RGB,
             * puis PQ. La PQ est concave : sa pente sur [y-e, y+e] est maximale en y-e, d'où la
             * pente prise en y-e ; si y ≤ e, PQ(2e) majore l'écart. */
            "bool refine=false;float fL[3],fR[3];"
            "for(int i=0;i<3;++i){float x=float(xd[i]),rel;fL[i]=f_eotf(x,1.0e-7*abs(x),rel);fR[i]=rel;"
            "if(rel>1.0e20)refine=true;}"
            "for(int i=0;i<3&&!refine;++i){float y=0.0,ey=0.0;"
            "for(int k=0;k<3;++k){float m=float(fel_d(%d+13+3*i+k));y+=m*fL[k];ey+=abs(m)*fL[k]*(fR[k]+1.0e-6);}"
            "float slope;float pq=f_oetf(y,slope);float err;"
            "if(y>ey){f_oetf(y-ey,slope);err=slope*ey+2.0e-6*pq;}else err=f_oetf(2.0*ey,slope);"
            "if(err*65535.0>0.4)refine=true;out3[i]=(floor(clamp(pq,0.0,1.0)*65535.0+0.5)+0.25)/65535.0;}"
            /* Double : EOTF, matrice LMS→RGB, OETF ; entier 16 bits arrondi au plus proche,
             * écrit avec +0,25 (exact quel que soit l'arrondi UNORM du pilote). */
            "if(refine){double lms[3];for(int i=0;i<3;++i)lms[i]=fel_eotf(xd[i]);"
            "for(int i=0;i<3;++i){double y=fel_d(%d+13+3*i)*lms[0]+fel_d(%d+14+3*i)*lms[1]+fel_d(%d+15+3*i)*lms[2];"
            "out3[i]=(float(floor(clamp(fel_oetf(y),0.0LF,1.0LF)*65535.0LF+0.5LF))+0.25)/65535.0;}}"
            "color=vec4(out3,1.0);",
            G, G, G, G);
    if (exact && used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            /* Exact certifié : bornes du CPU certifié (docs/2026-10-16-cpu-certifie.md) avec
             * u = 2^-52 (une erreur d'un ulp entier par opération double du GPU), classement entre
             * seuils PQ ; composante indécise : code candidat, puis correctif du CPU certifié. */
            "const double U=1.0LF/4503599627370496.0LF;"
            "int seg[3];double ev[3];"
            "for(int c=0;c<3;++c){seg[c]=(piece[c]-c*%d-14)/24;"
            "ev[c]=(fel_d(%d+8*c+seg[c])+1.0LF)*U+5.3LF*U*en[c]+2.02LF*U*abs(fel_d(%d+1+c))+5.03LF*U*abs(v[c]);}"
            "double L[3],eL[3];bool pole=false;"
            "for(int i=0;i<3;++i){double e=1.01LF*(abs(fel_d(%d+4+3*i))*ev[0]+abs(fel_d(%d+5+3*i))*ev[1]+abs(fel_d(%d+6+3*i))*ev[2]);"
            "double s=xd[i],k,mag;bool pl;double l=fel_eotf_k(s,k,mag,pl);pole=pole||pl;"
            "if(s+e<=7.3095590257839e-7LF){L[i]=0.0LF;eL[i]=0.0LF;}"
            "else{L[i]=l;eL[i]=s-e>7.3095590257840e-7LF&&k*e<=1.0e-6LF*s&&e<=1.0e-6LF*s"
            "?l*((800.0LF*k+60.0LF+4.2LF*mag)*U+1.01LF*k*e/s)+1.0e-300LF:1.0e300LF;}}"
            "uint mask=0u;"
            "for(int c=0;c<3;++c){double R=0.0LF,e=0.0LF;"
            "for(int k=0;k<3;++k){double M=fel_d(%d+13+3*c+k),dM=fel_d(%d+24+3*c+k);"
            "R+=M*L[k];e+=(abs(M)*(1.0LF+4.0LF*U)+dM)*eL[k]+(dM+3.01LF*U*abs(M))*L[k];}"
            "e=e*(1.0LF+1.0e-12LF)+4.0LF*U*abs(R)+1.0e-300LF;"
            "int q=0;if(R>0.0LF){float y=float(R),sl;q=y>=1.0?65535:clamp(int(floor(f_oetf(y,sl)*65535.0+0.5)),0,65535);}"
            "for(int i=0;i<8&&q<65535&&R>fel_T(q);++i)++q;"
            "for(int i=0;i<8&&q>0&&R<=fel_T(q-1);++i)--q;"
            "bool certain=!pole&&(q==0||R-e>fel_T(q-1))&&(q==65535||R+e<=fel_T(q));"
            "if(!certain)mask|=1u<<c;"
            "out3[c]=(float(q)+0.25)/65535.0;",
            H, E, G, G, G, G, G, E);
    if (exact && diag_rows && used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            "if(inside&&xy.y%%%d==0){uint b=%du*(uint(xy.y/%d)*%du+uint(xy.x))+5u*uint(c);"
            "uvec2 r=unpackDouble2x32(R),w=unpackDouble2x32(e);fel_diag[b]=r.x;fel_diag[b+1u]=r.y;"
            "fel_diag[b+2u]=w.x;fel_diag[b+3u]=w.y;fel_diag[b+4u]=uint(q)|(certain?65536u:0u);}",
            p->diag_stride, FEL_DIAG_WORDS, p->diag_stride, p->width);
    if (exact && used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            "}");
    if (exact) used = rescue_epilogue(p, body, sizeof(body), used);
    if (used <= 0 || used >= (int)sizeof(body)) return 0;
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = header, .body = body,
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR, .descriptors = planes,
        .num_descriptors = exact ? (diag_rows ? 16 : 15) : 14,
        .output_w = p->width, .output_h = p->height, .compute = exact, .compute_group_size = {16, 16}});
    ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[3], .shader = &sh, .target = p->target));
    pl_dispatch_abort(p->dispatch, &sh);
    return ok && (!exact || resolve_rescues(p, metadata, header));
}

/* Virgule fixe (lot A2') : tables GLSL (une fois par contexte) et seuils PQ en Q0.96 (T·2^96 exact,
 * T ≥ 2^-43), texture RGBA32UI 256 × 256, dernier texel au-delà de 1. */
static int prepare_fixed(struct fel_gpu *p)
{
    if (p->fixed_ready && p->thr96) return 1;
    if (!fel_fixed_tables(p->fixed_tables, FEL_FIXED_TABLE_WORDS)) return 0;
    p->fixed_ready = 1;
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_UINT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    uint32_t *data = malloc(65536 * 4 * sizeof(uint32_t));
    if (!fmt || !data) { free(data); return 0; }
    for (int k = 0; k < 65536; ++k) {
        unsigned __int128 v = (unsigned __int128)2 << 96;                         /* au-delà de tout R ≤ 1 */
        if (k < 65535) {
            double t;
            memcpy(&t, fel_pq_threshold_bytes + 8 * (size_t)k, sizeof(t));
            int e;
            const double m = frexp(t, &e);                                       /* t = m·2^e, m ∈ [0,5 ; 1) */
            const uint64_t mant = (uint64_t)ldexp(m, 53);
            v = (unsigned __int128)mant << (e + 43);                             /* t·2^96 = mant·2^(e+43) */
        }
        for (int w = 0; w < 4; ++w) data[4 * k + w] = (uint32_t)(v >> (32 * w));
        /* Quatrième mot (nul pour t < 1) : ulp de t en Q0.96 = 2^(e+43) ; seuil exact < t + ulp. */
        if (k < 65535) {
            double t;
            memcpy(&t, fel_pq_threshold_bytes + 8 * (size_t)k, sizeof(t));
            int e;
            frexp(t, &e);
            data[4 * k + 3] = (uint32_t)(e + 43);
        }
    }
    const int ok = pl_tex_recreate(p->vk->gpu, &p->thr96, pl_tex_params(.w = 256, .h = 256, .format = fmt,
                       .sampleable = true, .host_writable = true)) &&
                   pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex = p->thr96, .ptr = data));
    free(data);
    return ok;
}

/* Mode exact en virgule fixe multi-mots (src/shaders/fixed_*.glsl, modèle src/fixed_point.cpp) ; mêmes
 * liste de reprises, correctifs et diagnostic (format virgule fixe) que le mode exact en double. */
/* Second niveau du mode exact en virgule fixe (lot L) : les entrées de la liste des indécises sont
 * reprises sur le GPU par l'EOTF multi-mots (une invocation par entrée, shader à part : la composition
 * ne paie pas ses registres), puis par le CPU certifié si elles restent indécises. Correctifs appliqués
 * par la passe commune ; sans reprise CPU, ils restent sur le GPU. d : descripteurs de la composition. */
static int resolve_fixed(struct fel_gpu *p, const AVDOVIMetadata *metadata, const char *header,
                         const struct pl_shader_desc *d, const struct pl_shader_var *parameters, const char *patch_header)
{
    uint32_t count = 0, left = 0;
    if (!pl_buf_read(p->vk->gpu, p->rescue, 0, &count, sizeof(count))) return 0;
    if (count > FEL_RESCUE_CAPACITY) {
        fprintf(stderr, "mvo-fel : %u composantes à reprendre, capacité %d dépassée\n", count, FEL_RESCUE_CAPACITY);
        return 0;
    }
    if (!count) return 1;
    pl_buf_write(p->vk->gpu, p->left, 0, &left, sizeof(left));
    struct pl_buffer_var patch_var = uint_array("fel_patch", 3 * FEL_RESCUE_CAPACITY, 0), left_var = uint_array("fel_left", 1, 0);
    struct pl_shader_desc desc[16];
    memcpy(desc, d, 14 * sizeof(*desc));                                       /* sources, seuils Q0.96, liste */
    desc[14] = (struct pl_shader_desc){.desc = {.name = "fel_patch_buf", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_WRITEONLY}, .binding = {.object = p->patch}, .buffer_vars = &patch_var, .num_buffer_vars = 1};
    desc[15] = (struct pl_shader_desc){.desc = {.name = "fel_left_buf", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_READWRITE}, .binding = {.object = p->left}, .buffer_vars = &left_var, .num_buffer_vars = 1};
    const struct pl_shader_var variables[3] = {parameters[0], parameters[1],
        {.var = {.name = "fel_level_n", .type = PL_VAR_UINT, .dim_v = 1, .dim_m = 1, .dim_a = 1}, .data = &count, .dynamic = true}};
    char body[2048];
    snprintf(body, sizeof(body),
        "uint i=gl_GlobalInvocationID.x;if(i<fel_level_n){uint b=%du*i,w1=fel_rescue[b+1u];"
        "int seg[3];vec3 out3;uvec4 Rd[3];float Ed[3];uint qd[3];"
        "uint mask=fx_render(int(w1&0xffffu),int(fel_rescue[b+2u]),int(fel_rescue[b+3u]),int(fel_rescue[b+4u]),"
        "int(fel_rescue[b+5u]),int(fel_rescue[b+6u]),int(fel_rescue[b+7u]),int(fel_rescue[b+8u]),int(fel_rescue[b+9u]),"
        "true,seg,out3,Rd,Ed,qd);if(%d==1)mask=7u;"
        "fel_patch[3u*i]=fel_rescue[b];fel_patch[3u*i+1u]=(qd[0]&0xffffu)|((qd[1]&0xffffu)<<16);"
        "fel_patch[3u*i+2u]=qd[2]&0xffffu;fel_rescue[b+1u]=(w1&0x0fffffffu)|(mask<<28);"
        "if(mask!=0u)atomicAdd(fel_left,1u);}",
        FEL_RESCUE_WORDS, p->force_rescue);
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = header, .body = body,
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_NONE, .descriptors = desc, .num_descriptors = 16,
        .variables = variables, .num_variables = 3, .compute = true, .compute_group_size = {64, 1}});
    ok = ok && pl_dispatch_compute(p->dispatch, pl_dispatch_compute_params(.shader = &sh,
                                   .dispatch_size = {(int)((count + 63) / 64), 1, 1}));
    pl_dispatch_abort(p->dispatch, &sh);
    ok = ok && pl_buf_read(p->vk->gpu, p->left, 0, &left, sizeof(left));
    if (!ok) return 0;
    if (!left) return apply_patch(p, patch_header, NULL, count);
    /* Indécises restantes : CPU certifié sur ces seules entrées, correctifs complétés sur l'hôte. */
    uint32_t *entries = malloc(4 * FEL_RESCUE_WORDS * (size_t)count), *patch = malloc(12 * (size_t)count),
             *subset = malloc(4 * FEL_RESCUE_WORDS * (size_t)left), *index = malloc(4 * (size_t)left);
    uint16_t *codes = malloc(6 * (size_t)left);
    ok = entries && patch && subset && index && codes &&
         pl_buf_read(p->vk->gpu, p->rescue, 4, entries, 4 * FEL_RESCUE_WORDS * (size_t)count) &&
         pl_buf_read(p->vk->gpu, p->patch, 0, patch, 12 * (size_t)count);
    uint32_t n = 0;
    for (uint32_t i = 0; ok && i < count; ++i)
        if (entries[FEL_RESCUE_WORDS * i + 1] >> 28 && n < left) {
            memcpy(subset + FEL_RESCUE_WORDS * n, entries + FEL_RESCUE_WORDS * i, 4 * FEL_RESCUE_WORDS);
            index[n++] = i;
        }
    const int undecided = ok && n ? fel_certified_codes(metadata, subset, (int)n, codes) : 0;
    ok = ok && undecided >= 0;
    for (uint32_t k = 0; ok && k < n; ++k) {
        patch[3*index[k] + 1] = codes[3*k] | (uint32_t)codes[3*k + 1] << 16;
        patch[3*index[k] + 2] = codes[3*k + 2];
        const uint32_t mask = subset[FEL_RESCUE_WORDS * k + 1] >> 28;
        p->rescued += (mask & 1) + (mask >> 1 & 1) + (mask >> 2 & 1);
    }
    if (ok) p->undecided += undecided;
    ok = ok && apply_patch(p, patch_header, patch, count);
    free(entries); free(patch); free(subset); free(index); free(codes);
    return ok;
}

static int render_fixed(struct fel_gpu *p, const AVDOVIMetadata *metadata)
{
    if (!upload_coefficients(p, metadata) || !prepare_exact(p) || !prepare_fixed(p)) return 0;
    static const uint32_t zero = 0;
    pl_buf_write(p->vk->gpu, p->rescue, 0, &zero, sizeof(zero));
    const size_t prefix_size = strlen(fel_glsl_fixed_point) + 1024;
    char *prefix = malloc(prefix_size), *header = malloc(prefix_size + FEL_HEADER_SIZE + strlen(fel_glsl_fixed_render)),
         *patch_header = malloc(FEL_HEADER_SIZE);
    int ok = prefix && header && patch_header;
    if (ok) snprintf(prefix, prefix_size, "const int FX_CURVE0=%d;const int FX_CURVE=%d;const int FX_GLOBAL=%d;"
                     "const int FXT_LNC=%d;const int FXT_INVC=%d;const int FXT_POW2=%d;const int FXT_EXPM1=%d;"
                     "const int FXT_LNK=%d;const int FXT_EXPK=%d;const int FXT_EM1K=%d;const int FXT_CONST=%d;\n%s",
                     FEL_FIXED_CURVE0, FEL_FIXED_CURVE, FEL_FIXED_GLOBAL, FXT_LNC, FXT_INVC, FXT_POW2, FXT_EXPM1,
                     FXT_LNK, FXT_EXPK, FXT_EM1K, FXT_CONST, fel_glsl_fixed_point);
    ok = ok && precise_header(header, prefix_size + FEL_HEADER_SIZE + strlen(fel_glsl_fixed_render), metadata,
                              prefix, fel_glsl_fixed_render) &&
         precise_header(patch_header, FEL_HEADER_SIZE, metadata, fel_double_header, "");
    struct pl_buffer_var rescue_vars[2];
    const int diag_rows = p->diag_stride > 0 ? (p->height + p->diag_stride - 1) / p->diag_stride : 0;
    struct pl_buffer_var diag_var = uint_array("fel_diag", diag_rows ? FEL_DIAG_WORDS_FIXED * diag_rows * p->width : 1, 0);
    struct pl_shader_desc d[16];
    precise_descriptors(p, d, rescue_vars, PL_DESC_ACCESS_READWRITE);
    d[12] = (struct pl_shader_desc){.desc = {.name = "fel_thr96", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->thr96}};
    d[13] = d[14];                                                                /* liste de reprises */
    d[14] = (struct pl_shader_desc){.desc = {.name = "fel_diag_buf", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_WRITEONLY}, .binding = {.object = p->diag}, .buffer_vars = &diag_var, .num_buffer_vars = 1};
    const struct pl_shader_var parameters[2] = {
        {.var = {.name = "fel_fp", .type = PL_VAR_UINT, .dim_v = 4, .dim_m = 1, .dim_a = FEL_FIXED_WORDS / 4},
         .data = p->fixed_words, .dynamic = true},
        {.var = {.name = "fx_t", .type = PL_VAR_UINT, .dim_v = 4, .dim_m = 1, .dim_a = FEL_FIXED_TABLE_WORDS / 4},
         .data = p->fixed_tables},
    };
    char body[8192];
    int used = snprintf(body, sizeof(body),
        "ivec2 xy0=ivec2(gl_FragCoord.xy);bool inside=xy0.x<%d&&xy0.y<%d;ivec2 xy=min(xy0,ivec2(%d,%d));"
        "int code,tb,tr,Db,te0,te1,te2,Dl,Dc;fel_inputs(xy,code,tb,tr,Db,te0,te1,te2,Dl,Dc);"
        "int seg[3];vec3 out3;uvec4 Rd[3];float Ed[3];uint qd[3];"
        "uint mask=fx_render(code,tb,tr,Db,te0,te1,te2,Dl,Dc,false,seg,out3,Rd,Ed,qd);",
        p->width, p->height, p->width - 1, p->height - 1);
    if (diag_rows && used > 0 && used < (int)sizeof(body))
        used += snprintf(body + used, sizeof(body) - used,
            "if(inside&&xy.y%%%d==0){uint b=%du*(uint(xy.y/%d)*%du+uint(xy.x));for(int c=0;c<3;++c){uint o=b+6u*uint(c);"
            "fel_diag[o]=Rd[c].x;fel_diag[o+1u]=Rd[c].y;fel_diag[o+2u]=Rd[c].z;fel_diag[o+3u]=Rd[c].w;"
            "fel_diag[o+4u]=floatBitsToUint(Ed[c]);fel_diag[o+5u]=qd[c];}}",
            p->diag_stride, FEL_DIAG_WORDS_FIXED, p->diag_stride, p->width);
    used = rescue_epilogue(p, body, sizeof(body), used);
    ok = ok && used > 0 && used < (int)sizeof(body);
    if (ok) {
        pl_shader sh = pl_dispatch_begin(p->dispatch);
        ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = header, .body = body,
            .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR, .descriptors = d,
            .num_descriptors = diag_rows ? 15 : 14, .variables = parameters, .num_variables = 2,
            .output_w = p->width, .output_h = p->height, .compute = true, .compute_group_size = {16, 16}});
        ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[3], .shader = &sh, .target = p->target));
        pl_dispatch_abort(p->dispatch, &sh);
        ok = ok && resolve_fixed(p, metadata, header, d, parameters, patch_header);
    }
    free(prefix); free(header); free(patch_header);
    return ok;
}

/* Couche 4:2:0 10 bits aléatoire (générateur congruentiel), position chroma en haut à gauche. */
static AVFrame *test_layer(int width, int height, uint32_t *state, int low, int high)
{
    AVFrame *frame = av_frame_alloc();
    if (!frame) return NULL;
    frame->format = AV_PIX_FMT_YUV420P10LE; frame->width = width; frame->height = height;
    frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
    if (av_frame_get_buffer(frame, 32) < 0) { av_frame_free(&frame); return NULL; }
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < (c ? (height + 1) / 2 : height); ++y) {
            uint16_t *row = (uint16_t *)(frame->data[c] + y * frame->linesize[c]);
            for (int x = 0; x < (c ? (width + 1) / 2 : width); ++x) {
                *state = *state * 1664525u + 1013904223u;
                row[x] = (uint16_t)(low + (*state >> 8) % (uint32_t)(high - low + 1));
            }
        }
    return frame;
}

/* RPU de test : matrices Dolby réalistes ; mmr : matrices identité et segment MMR d'ordre 2 à
 * fortes annulations (plages de codes du test fel_gpu_precision, signaux dans le domaine PQ). */
static void test_metadata(AVDOVIMetadata *metadata, int mmr)
{
    AVDOVIRpuDataHeader *header = av_dovi_get_header(metadata);
    AVDOVIDataMapping *mapping = av_dovi_get_mapping(metadata);
    AVDOVIColorMetadata *color = av_dovi_get_color(metadata);
    static const int ycc[9] = {65536, 0, 91881, 65536, -22554, -46802, 65536, 116130, 0};
    static const int lms[9] = {4408196, 5353728, 238131, 1619851, 7586529, 793630, 0, 257773, 9740729};
    header->bl_bit_depth = header->el_bit_depth = 10;
    header->coef_log2_denom = 16;
    header->disable_residual_flag = 0;
    mapping->nlq_method_idc = AV_DOVI_NLQ_LINEAR_DZ;
    for (int c = 0; c < 3; ++c) {
        AVDOVIReshapingCurve *curve = &mapping->curves[c];
        memset(curve, 0, sizeof(*curve));
        curve->num_pivots = 2; curve->pivots[1] = 1023;
        curve->poly_order[0] = 1; curve->poly_coef[0][1] = 65536;
        if (mmr) {
            curve->num_pivots = 3; curve->pivots[1] = 400; curve->pivots[2] = 1023;
            curve->mapping_idc[1] = AV_DOVI_MAPPING_MMR;
            curve->mmr_order[1] = 2; curve->mmr_constant[1] = 2000;
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < 7; ++j) curve->mmr_coef[1][k][j] = 3000 * (j + 1) * (k ? -1 : 1);
        }
        mapping->nlq[c].nlq_offset = 512;
        mapping->nlq[c].linear_deadzone_slope = 64;
        mapping->nlq[c].linear_deadzone_threshold = 32;
        color->ycc_to_rgb_offset[c] = (AVRational){!mmr && c ? 32768 : 0, 65536};
        for (int j = 0; j < 3; ++j) {
            color->ycc_to_rgb_matrix[3*c + j] = mmr ? (AVRational){c == j, 1} : (AVRational){ycc[3*c + j], 65536};
            color->rgb_to_lms_matrix[3*c + j] = mmr ? (AVRational){c == j, 1} : (AVRational){lms[3*c + j], 10000000};
        }
    }
}

/* Auto-test du rendu exact sur ce GPU, à la taille de l'image : couches aléatoires, deux RPU.
 * Sur deux lignes, chaque valeur linéaire R̂ du GPU doit rester à moins de la moitié de sa borne
 * de la valeur double-double, et chaque code certain égaler le code certifié du CPU. Échec :
 * mode exact refusé (rendu CPU). Le bilan est écrit sur stderr si MVO_FEL_GPU_SELF_TEST est défini. */
static int render_target(struct fel_gpu *p, const AVDOVIMetadata *metadata, const AVFrame *bl, const AVFrame *el);
/* Auto-test d'une variante du mode exact (0 double, 1 virgule fixe), à la taille de l'image, sur
 * deux RPU : sur deux lignes, |R̂ - R| ≤ moitié de la borne face au double-double et codes certains
 * identiques au CPU certifié. ms : temps GPU de composition des deux images. */
static int self_test_variant(struct fel_gpu *p, int fixed, AVDOVIMetadata *metadata, double *worst, int *mismatches,
                             double *ms)
{
    uint32_t state = 12345;
    p->fixed_mode = fixed;
    p->diag_stride = p->height > 1 ? p->height / 2 : 1;
    const int rows = (p->height + p->diag_stride - 1) / p->diag_stride;
    const size_t words = (size_t)(fixed ? FEL_DIAG_WORDS_FIXED : FEL_DIAG_WORDS) * rows * p->width;
    uint32_t *diag = malloc(4 * words);
    p->diag = diag ? pl_buf_create(p->vk->gpu, pl_buf_params(.size = 4 * words, .storable = true,
                                                             .host_readable = true)) : NULL;
    int ok = p->diag != NULL;
    *worst = 0; *mismatches = 0; *ms = 0;
    for (int mmr = 0; ok && mmr < 2; ++mmr) {
        test_metadata(metadata, mmr);
        AVFrame *bl = test_layer(p->width, p->height, &state, mmr ? 0 : 200, mmr ? 1023 : 760);
        AVFrame *el = test_layer((p->width + 1) / 2, (p->height + 1) / 2, &state, mmr ? 0 : 200, mmr ? 1023 : 760);
        double ratio = 0;
        pl_gpu_finish(p->vk->gpu);
        collect_timers(p);
        const double before = p->gpu_ns[3];
        ok = bl && el && (!fixed || fel_fixed_params(metadata, p->fixed_words, FEL_FIXED_WORDS)) &&
             render_target(p, metadata, bl, el) && pl_buf_read(p->vk->gpu, p->diag, 0, diag, 4 * words);
        pl_gpu_finish(p->vk->gpu);
        collect_timers(p);
        *ms += (p->gpu_ns[3] - before) / 1e6;
        const int count = ok ? fel_certified_check(metadata, bl, el, p->diag_stride, fixed, diag, &ratio) : -1;
        ok = ok && count >= 0;
        *mismatches += count > 0 ? count : 0;
        if (ratio > *worst || ratio != ratio) *worst = ratio;
        av_frame_free(&bl); av_frame_free(&el);
    }
    pl_buf_destroy(p->vk->gpu, &p->diag);
    p->diag_stride = 0;
    free(diag);
    return ok && *mismatches == 0 && *worst <= 0.5;
}

/* Auto-test du mode exact : variantes double et virgule fixe (lot A2'), la plus rapide des variantes
 * valides est retenue (MVO_FEL_GPU_FIXED=0 ou 1 : variante imposée). Échec des deux : mode exact
 * refusé (rendu CPU). Bilan sur stderr si MVO_FEL_GPU_SELF_TEST est défini. */
static int fel_gpu_self_test(struct fel_gpu *p)
{
    size_t size = 0;
    AVDOVIMetadata *metadata = av_dovi_metadata_alloc(&size);
    if (!metadata) return 0;
    const long long rescued = p->rescued, undecided = p->undecided;
    double gpu_ns[5];
    memcpy(gpu_ns, p->gpu_ns, sizeof(gpu_ns));
    const char *forced = getenv("MVO_FEL_GPU_FIXED");
    double worst[2] = {0, 0}, ms[2] = {0, 0};
    int mismatches[2] = {0, 0}, ok[2] = {0, 0}, tested[2] = {0, 0};
    if ((tested[0] = !forced || strcmp(forced, "1")))
        ok[0] = self_test_variant(p, 0, metadata, &worst[0], &mismatches[0], &ms[0]);
    if ((tested[1] = !forced || strcmp(forced, "0")))
        ok[1] = self_test_variant(p, 1, metadata, &worst[1], &mismatches[1], &ms[1]);
    p->fixed_mode = ok[1] && (!ok[0] || ms[1] < ms[0]) ? 1 : 0;
    if (getenv("MVO_FEL_GPU_SELF_TEST"))
        for (int v = 0; v < 2; ++v)
            if (!tested[v]) fprintf(stderr, "mvo-fel : auto-test exact %s non testé (MVO_FEL_GPU_FIXED)\n", v ? "virgule fixe" : "double");
            else fprintf(stderr, "mvo-fel : auto-test exact %s %s (écart / borne %.4g, codes différents %d, %.2f ms)%s\n",
                         v ? "virgule fixe" : "double", ok[v] ? "réussi" : "refusé", worst[v], mismatches[v], ms[v],
                         p->fixed_mode == v && ok[v] ? " : retenu" : "");
    p->rescued = rescued; p->undecided = undecided;
    memcpy(p->gpu_ns, gpu_ns, sizeof(gpu_ns));
    av_free(metadata);
    return ok[0] || ok[1];
}

int fel_gpu_set_precision(void *context, int precision)
{
    struct fel_gpu *p = context;
    if (precision < FEL_PRECISION_FAST || precision > FEL_PRECISION_EXACT ||
        (precision != FEL_PRECISION_FAST && !p->float64) || (precision == FEL_PRECISION_EXACT && !p->storable)) return 0;
    if (precision == FEL_PRECISION_EXACT && !p->self_test) {
        const int previous = p->precision;
        p->precision = precision;
        p->self_test = fel_gpu_self_test(p) ? 1 : -1;
        p->precision = previous;
    }
    if (precision == FEL_PRECISION_EXACT && p->self_test < 0) return 0;
    p->precision = precision;
    return 1;
}

int fel_gpu_float64(const void *context)
{
    const struct fel_gpu *p = context;
    return p && p->float64;
}

/* Mode rapide (lot R) : float32 seul, aucun double requis. Sommes Spline16 entières, segment et
 * zéro NLQ exacts comme en mode exact ; mapping, NLQ et matrices en float32 ; EOTF et OETF
 * reformulées sans annulation (g_eotf, g_oetf). */
static int render_fast(struct fel_gpu *p, const AVDOVIMetadata *metadata)
{
    char header[FEL_HEADER_SIZE];
    if (!upload_coefficients(p, metadata) ||
        !precise_header(header, sizeof(header), metadata, "float fel_f(int i){return fel_cf[i>>2][i&3];}\n", "")) return 0;
    struct pl_shader_desc d[16];
    struct pl_buffer_var unused[2];
    precise_descriptors(p, d, unused, PL_DESC_ACCESS_READONLY);   /* sources et axes : 12 premiers */
    /* Coefficients en tableau uniforme (cache de constantes), non en texture : ~100 lectures par pixel. */
    float floats[4 * ((FEL_COEFFICIENTS + 3) / 4)] = {0};
    for (int k = 0; k < FEL_COEFFICIENTS; ++k) floats[k] = (float)p->coef_values[k];
    const struct pl_shader_var coefficients = {.var = {.name = "fel_cf", .type = PL_VAR_FLOAT, .dim_v = 4, .dim_m = 1,
        .dim_a = (FEL_COEFFICIENTS + 3) / 4}, .data = floats, .dynamic = true};
    const int H = FEL_CHANNEL, G = FEL_GLOBAL;
    char body[8192];
    const int used = snprintf(body, sizeof(body),
        "ivec2 xy=ivec2(gl_FragCoord.xy);"
        "int code,tb,tr,Db,te0,te1,te2,Dl,Dc;fel_inputs(xy,code,tb,tr,Db,te0,te1,te2,Dl,Dc);"
        "uvec2 full64=fel_mul64(1023u,uint(Db));uvec2 X64[3];X64[0]=uvec2(0u,uint(code));"
        "X64[1]=uvec2(0u,uint(max(tb,0)));X64[2]=uvec2(0u,uint(max(tr,0)));"
        "for(int c=1;c<3;++c)X64[c]=fel_ge64(X64[c],full64)?full64:X64[c];"
        "int piece[3];"
        "for(int c=0;c<3;++c){int B=c*%d,count=int(fel_f(B)),p=0;"
        "while(p+2<count){uint pc=uint(fel_f(B+2+p));if(!fel_ge64(X64[c],c==0?uvec2(0u,pc):fel_mul64(pc,uint(Db))))break;++p;}"
        "piece[c]=B+14+24*p;}"
        "float full=1023.0*float(Db);"
        "float sig[3];sig[0]=float(code)/1023.0;sig[1]=X64[1]==full64?1.0:float(X64[1].y)/full;"
        "sig[2]=X64[2]==full64?1.0:float(X64[2].y)/full;"
        "float t7[7];t7[0]=sig[0];t7[1]=sig[1];t7[2]=sig[2];t7[3]=sig[0]*sig[1];t7[4]=sig[0]*sig[2];"
        "t7[5]=sig[1]*sig[2];t7[6]=t7[3]*sig[2];"
        "float res[3];"
        "for(int c=0;c<3;++c){int P=piece[c],B=c*%d;float s;"
        "if(fel_f(P)==0.0)s=(fel_f(P+4)*sig[c]+fel_f(P+3))*sig[c]+fel_f(P+2);"
        "else{s=fel_f(P+2);int order=int(fel_f(P+1));"
        "for(int j=0;j<7;++j){float acc=fel_f(P+3+7*(order-1)+j);"
        "for(int k=order-2;k>=0;--k)acc=acc*t7[j]+fel_f(P+3+7*k+j);s+=acc*t7[j];}}"
        "res[c]=clamp(s,fel_f(B+10),fel_f(B+11));}"
        "if(fel_f(%d)!=0.0){int te[3]=int[3](te0,te1,te2);int De[3]=int[3](Dl,Dc,Dc);"
        "for(int c=0;c<3;++c)if(te[c]!=0){float d=float(te[c])/(1023.0*float(De[c]));"
        "res[c]+=sign(d)*(abs(d)*fel_f(c*%d+12)+fel_f(c*%d+13));}}"
        "float L[3];for(int i=0;i<3;++i){float x=fel_f(%d+4+3*i)*(res[0]-fel_f(%d+1))+fel_f(%d+5+3*i)*(res[1]-fel_f(%d+2))"
        "+fel_f(%d+6+3*i)*(res[2]-fel_f(%d+3));L[i]=g_eotf(x);}"
        "vec3 out3;for(int i=0;i<3;++i){float y=fel_f(%d+13+3*i)*L[0]+fel_f(%d+14+3*i)*L[1]+fel_f(%d+15+3*i)*L[2];"
        "out3[i]=(floor(clamp(g_oetf(y),0.0,1.0)*65535.0+0.5)+0.25)/65535.0;}"
        "color=vec4(out3,1.0);",
        H, H, G, H, H, G, G, G, G, G, G, G, G, G);
    if (used <= 0 || used >= (int)sizeof(body)) return 0;
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = header, .body = body,
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR, .descriptors = d, .num_descriptors = 12,
        .variables = &coefficients, .num_variables = 1, .output_w = p->width, .output_h = p->height});
    ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[3], .shader = &sh, .target = p->target));
    pl_dispatch_abort(p->dispatch, &sh);
    return ok;
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
    /* Exact : virgule fixe si choisie (auto-test) et si le RPU entre dans son cadre, sinon double. */
    if (p->precision == FEL_PRECISION_EXACT && p->fixed_mode == 1 &&
        fel_fixed_params(metadata, p->fixed_words, FEL_FIXED_WORDS)) return render_fixed(p, metadata);
    if (p->precision != FEL_PRECISION_FAST) return render_precise(p, metadata);
    /* Rapide : shader float32 Muxiveo si le RPU entre dans son cadre (10 bits, sans partition, NLQ
     * linéaire) ; sinon, ou avec MVO_FEL_GPU_LIBPLACEBO (comparaisons), décodage libplacebo. */
    double supported[FEL_COEFFICIENTS];
    if (fused && !getenv("MVO_FEL_GPU_LIBPLACEBO") && fel_coefficients(metadata, supported))
        return render_fast(p, metadata);
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

/* floor(A/D + t/8192) en double, entiers exacts (|A|, D·8192 < 2^53) : quotient corrigé puis
 * comparaison du reste au seuil de tramage, sans int64 ni arrondi de division. */
static const char fel_quantize_header[] =
    "double fel_quant(double A,double D,double t){double q=floor(A/D),r=A-q*D;"
    "for(int i=0;i<2;++i){if(r<0.0LF){q-=1.0LF;r+=D;}if(r>=D){q+=1.0LF;r-=D;}}"
    "return q+(r*8192.0LF>=D*(8192.0LF-t)?1.0LF:0.0LF);}\n";

static int dispatch(struct fel_gpu *p, pl_tex target, const char *header, const char *body, int w, int h)
{
    const struct pl_shader_desc descriptors[2] = {
        {.desc = {.name = "fel_rgb", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->target}},
        {.desc = {.name = "fel_threshold", .type = PL_DESC_SAMPLED_TEX}, .binding = {.object = p->dither}},
    };
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    int ok = pl_shader_custom(sh, &(struct pl_custom_shader){.header = header, .body = body, .input = PL_SHADER_SIG_NONE,
        .output = PL_SHADER_SIG_COLOR, .descriptors = descriptors, .num_descriptors = 2,
        .output_w = w, .output_h = h});
    ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.timer = p->timers[4], .shader = &sh, .target = target));
    pl_dispatch_abort(p->dispatch, &sh);
    return ok;
}

/* Y'CbCr BT.2020nc limité, 4:2:0 bilinéaire au point chroma, quantification tramée :
 * mêmes formules que convert_rows (CPU). Entiers écrits avec +0,25 : exacts que le
 * pilote arrondisse au plus proche ou vers zéro. v = NULL : chroma entrelacée (P010).
 * Mode exact : arithmétique entière exacte de convert_rows, simulée en double (fel_quant). */
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
    /* Mode exact : luma floor(219·S·2^(d-8)/(10000·65535) + t) + 16·2^(d-8), S = 2627R + 6780G + 593B. */
    const int exact = p->precision == FEL_PRECISION_EXACT;
    if (exact)
        snprintf(body, sizeof(body),
            "const float U = %d.0/%d.0, B = 0.25/%d.0;"
            "ivec2 xy=ivec2(gl_FragCoord.xy);"
            "dvec3 c=dvec3(round(texelFetch(fel_rgb, xy + ivec2(%d, %d), 0).rgb*65535.0));"
            "double A=219.0LF*(2627.0LF*c.r+6780.0LF*c.g+593.0LF*c.b)*%d.0LF;"
            "double t=double(round(texelFetch(fel_threshold, xy & 63, 0).r*8192.0));"
            "double k=min(16.0LF*%d.0LF+fel_quant(A,655350000.0LF,t),%d.0LF);"
            "color = vec4(float(k)*U + B, 0.0, 0.0, 1.0);",
            numerator, denominator, denominator, o->crop_x, o->crop_y, scale, scale, maximum);
    if (!dispatch(p, y, exact ? fel_quantize_header : NULL, body, o->width, o->height)) return 0;
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
        /* Chroma exacte : poids en 8es par axe (sommes sur 64), 224·cb = 7X/(4·k·65535),
         * X = 9407B - 2627R - 6780G (k = 9407) ou 7373R - 6780G - 593B (k = 7373). */
        if (exact)
            snprintf(body, sizeof(body),
                "const float U = %d.0/%d.0, B = 0.25/%d.0;"
                "const double WX[4] = double[4](%d.0LF, %d.0LF, %d.0LF, %d.0LF);"
                "const double WY[4] = double[4](%d.0LF, %d.0LF, %d.0LF, %d.0LF);"
                "ivec2 ij = ivec2(gl_FragCoord.xy);"
                "dvec3 s = dvec3(0.0LF);"
                "for (int b = 0; b < 4; b++) for (int a = 0; a < 4; a++) {"
                "  double w = WX[a]*WY[b]; if (w == 0.0LF) continue;"
                "  ivec2 q = clamp(ivec2(%d + 2*ij.x + %d + a, %d + 2*ij.y + %d + b), ivec2(0), ivec2(%d, %d));"
                "  s += w*dvec3(round(texelFetch(fel_rgb, q, 0).rgb*65535.0)); }"
                "double tb = double(round(texelFetch(fel_threshold, (ij + ivec2(21, 11)) & 63, 0).r*8192.0));"
                "double tr = double(round(texelFetch(fel_threshold, (ij + ivec2(43, 37)) & 63, 0).r*8192.0));"
                "float qb = float(min(128.0LF*%d.0LF + fel_quant(7.0LF*(9407.0LF*s.b-2627.0LF*s.r-6780.0LF*s.g)*%d.0LF,"
                "  4.0LF*9407.0LF*65535.0LF, tb), %d.0LF));"
                "float qr = float(min(128.0LF*%d.0LF + fel_quant(7.0LF*(7373.0LF*s.r-6780.0LF*s.g-593.0LF*s.b)*%d.0LF,"
                "  4.0LF*7373.0LF*65535.0LF, tr), %d.0LF));"
                "%s",
                numerator, denominator, denominator, wx[0]/2, wx[1]/2, wx[2]/2, wx[3]/2, wy[0]/2, wy[1]/2, wy[2]/2, wy[3]/2,
                o->crop_x, fx, o->crop_y, fy, p->width - 1, p->height - 1,
                scale, scale, maximum, scale, scale, maximum, store);
        if (!dispatch(p, plane == 0 ? u : v, exact ? fel_quantize_header : NULL, body, o->width / 2, o->height / 2)) return 0;
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
    p->storable=(fmt->caps & PL_FMT_CAP_STORABLE) && p->vk->gpu->glsl.compute;
    p->target=pl_tex_create(p->vk->gpu,pl_tex_params(.w=width,.h=height,.format=fmt,
        .renderable=true,.sampleable=true,.storable=p->storable));
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
