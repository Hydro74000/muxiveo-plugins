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
    int axis_w[3], axis_h[3], axis_loc[3];
    int width, height;
    pl_tex dither;
    int dither_mode;
    struct fel_output output;   /* sortie Vulkan directe (fel_gpu_configure) */
};

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
        for (int i = 0; i < 12; ++i) pl_shader_obj_destroy(&p->lut[i]);
        pl_tex_destroy(p->vk->gpu, &p->target);
        for (int c=0;c<3;++c) pl_tex_destroy(p->vk->gpu, &p->planar[c]);
        pl_tex_destroy(p->vk->gpu, &p->dither);
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
    struct pl_vulkan_params params = {.allow_software = false, .get_proc_addr=fel_vk_get_proc_addr()};
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
    p->dispatch = pl_dispatch_create(p->log, p->vk->gpu);
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
        ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.shader=&sh,.target=p->scaled[i]));
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
    ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.shader=&sh,.target=p->scaled[i]));
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
        ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = p->source[i]));
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
            ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = p->scaled[i]));
            pl_dispatch_abort(p->dispatch, &sh);
            if (!ok) return 0;
            continue;
        }
        pl_shader sh = pl_dispatch_begin(p->dispatch);
        int ok = pl_shader_sample_ortho2(sh, pl_sample_src(.tex = p->source[i], .components = 1,
            .scale = 65535.f/1023.f, .new_w = w, .new_h = p->height,
            .rect = {0, -sy*ry, w, (p->height-sy)*ry}),
            pl_sample_filter_params(.filter = pl_filter_spline16, .lut = &p->lut[2*i]));
        ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = p->middle[i]));
        if (ok) {
            sh = pl_dispatch_begin(p->dispatch);
            ok = pl_shader_sample_ortho2(sh, pl_sample_src(.tex = p->middle[i], .components = 1,
                .new_w = p->width, .new_h = p->height,
                .rect = {-sx*rx, 0, (p->width-sx)*rx, p->height}),
                pl_sample_filter_params(.filter = pl_filter_spline16, .lut = &p->lut[2*i+1]));
            ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = p->scaled[i]));
        }
        pl_dispatch_abort(p->dispatch, &sh);
        if (!ok) return 0;
    }
    return 1;
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

static int render_target(struct fel_gpu *p, const AVDOVIMetadata *metadata,
                         const AVFrame *bl, const AVFrame *el)
{
    if (!scale_layer(p, bl, 0, metadata) || !scale_layer(p, el, 1, metadata)) return 0;
    struct pl_dovi_metadata dovi = {0};
    pl_map_dovi_metadata(&dovi, metadata);
    for (int c=0;c<3;++c) dovi.nlq[c].offset=0;
    struct pl_color_repr repr = {.sys = PL_COLOR_SYSTEM_DOLBYVISION, .levels = PL_COLOR_LEVELS_FULL,
        .bits = {.sample_depth = 16, .color_depth = 16}, .dovi = &dovi};
    pl_shader sh = pl_dispatch_begin(p->dispatch);
    pl_shader eh = pl_shader_alloc(p->log, pl_shader_params(.gpu = p->vk->gpu, .id = 1));
    int ok = eh && join_layer(p, sh, 0) && join_layer(p, eh, 1);
    if (ok) {
        pl_shader_decode_color_ex(sh, pl_color_decode_args(.repr = &repr, .enhancement_layer = eh));
        ok = pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = p->target));
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
        ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.shader=&sh,.target=p->planar[c]));
        pl_dispatch_abort(p->dispatch,&sh);
        if(ok)ok=pl_tex_download(p->vk->gpu,pl_tex_transfer_params(.tex=p->planar[c],
            .ptr=output->data[c],.row_pitch=output->linesize[c],
            .callback=p->vk->gpu->limits.callbacks ? downloaded : NULL));
    }
    // Les tampons appartiennent au paquet ; même une erreur attend les DMA lancés.
    pl_gpu_finish(p->vk->gpu);
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
    ok = ok && pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = target));
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
    return ok;
}

void fel_gpu_configure(void *context, const struct fel_output *o)
{
    struct fel_gpu *p = context;
    p->output = *o;
}

size_t fel_gpu_devices(char *buffer, size_t capacity)
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
    uint32_t count=0;
    if(!enumerate || !props || !memory || enumerate(inst->instance,&count,NULL)!=VK_SUCCESS || count>64){
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
        uint64_t bytes=0;
        for(uint32_t k=0;k<mem.memoryHeapCount;++k)
            if(mem.memoryHeaps[k].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)bytes+=mem.memoryHeaps[k].size;
        used+=(size_t)snprintf(json+used,sizeof(json)-used,
            "%s{\"uuid\":\"%s\",\"name\":\"%s\",\"vendor\":%u,\"index\":%u,"
            "\"kind\":\"%s\",\"memory_bytes\":%" PRIu64 "}",
            included++?",":"",uuid,name,v->vendorID,i,
            v->deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU?"integrated":"discrete",bytes);
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
    p->dispatch=pl_dispatch_create(p->log,p->vk->gpu);
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
        ok=ok && pl_dispatch_finish(p->dispatch,pl_dispatch_params(.shader=&sh,.target=target.planes[c].texture));
        pl_dispatch_abort(p->dispatch,&sh);
    }
    if(target.release)target.release(p->vk->gpu,&target);
    pl_gpu_flush(p->vk->gpu);
    pl_unmap_avframe(p->vk->gpu,&target);
    return ok;
}
