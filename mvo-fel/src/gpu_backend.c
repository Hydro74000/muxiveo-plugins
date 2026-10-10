// SPDX-License-Identifier: LGPL-2.1-or-later
// Reconstruction Vulkan : aucun Display Management ni tone mapping.
#include <libplacebo/vulkan.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/utils/libav.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>

extern PFN_vkGetInstanceProcAddr fel_vk_get_proc_addr(void);

struct fel_gpu {
    pl_log log;
    pl_vulkan vk;
    pl_dispatch dispatch;
    pl_tex source[6], middle[6], scaled[6], target;
    pl_shader_obj lut[12];
    pl_tex axes[3][6];
    int axis_w[3], axis_h[3], axis_loc[3];
    int width, height;
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
        for (int i = 0; i < 12; ++i) pl_shader_obj_destroy(&p->lut[i]);
        pl_tex_destroy(p->vk->gpu, &p->target);
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
    pl_fmt fmt = pl_find_fmt(p->vk->gpu, PL_FMT_UNORM, 4, 16, 16, PL_FMT_CAP_RENDERABLE);
    if (!fmt || !p->dispatch) goto fail;
    p->target = pl_tex_create(p->vk->gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt, .renderable = true, .host_readable = true));
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

static int prepare_axis(struct fel_gpu *p, int c, int axis,
                        int count, int size, double ratio, double shift)
{
    pl_fmt fmt=pl_find_fmt(p->vk->gpu, PL_FMT_FLOAT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    float *data=calloc((size_t)count*12,sizeof(float));
    if(!fmt || !data) {free(data);return 0;}
    for(int x=0;x<count;++x){
        const double pos=(x+0.5-shift)*ratio-0.5;
        const int first=(int)floor(pos)-1;
        double weights[4],sum=0;
        for(int k=0;k<4;++k) sum+=weights[k]=spline(pos-first-k);
        for(int k=0;k<4;++k){
            double w=weights[k]/sum;
            int index=first+k;
            if(index<0)index=0;
            if(index>=size)index=size-1;
            data[4*x+k]=(float)index;
            data[4*count+4*x+k]=(float)w;
            data[8*count+4*x+k]=(float)(w-(float)w);
        }
    }
    int ok=1;
    for(int k=0;k<3;++k){
        pl_tex *tex=&p->axes[c][3*axis+k];
        ok=ok && pl_tex_recreate(p->vk->gpu, tex, pl_tex_params(.w=count,.h=1,
            .format=fmt,.sampleable=true,.host_writable=true));
        ok=ok && pl_tex_upload(p->vk->gpu, pl_tex_transfer_params(.tex=*tex,.ptr=data+4*count*k));
    }
    free(data);
    return ok;
}

// Codes EL centrés avant interpolation ; sommes compensées et FMA conservent
// le signe des petits résiduels sans le saut erroné de sign(d)*threshold.
static int scale_residual(struct fel_gpu *p, int i, int w, int h,
                          double rx, double ry, double sx, double sy, int offset, int location)
{
    const int c=i-3;
    if(p->axis_w[c]!=w || p->axis_h[c]!=h || p->axis_loc[c]!=location){
        if(!prepare_axis(p,c,0,p->width,w,rx,sx) ||
           !prepare_axis(p,c,1,p->height,h,ry,sy))return 0;
        p->axis_w[c]=w;p->axis_h[c]=h;p->axis_loc[c]=location;
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

static int scale_layer(struct fel_gpu *p, const AVFrame *frame, int enhancement, const AVDOVIMetadata *metadata)
{
    pl_gpu gpu = p->vk->gpu;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    pl_fmt source_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    pl_fmt float_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32,
        PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR);
    if (!desc || !source_fmt || !float_fmt) return 0;
    for (int c = 0; c < 3; ++c) {
        const int i = enhancement*3+c;
        const int w = c ? AV_CEIL_RSHIFT(frame->width, desc->log2_chroma_w) : frame->width;
        const int h = c ? AV_CEIL_RSHIFT(frame->height, desc->log2_chroma_h) : frame->height;
        const AVComponentDescriptor *comp = &desc->comp[c];
        // Le prototype accepte les plans 16 bits contenant les codes HEVC 10 bits.
        if (comp->depth != 10 || comp->step != 2 || comp->shift || comp->offset ||
            frame->linesize[comp->plane] <= 0) return 0;
        if (!pl_tex_recreate(gpu, &p->source[i], pl_tex_params(.w = w, .h = h,
                .format = source_fmt, .sampleable = true, .host_writable = true)) ||
            !pl_tex_recreate(gpu, &p->middle[i], pl_tex_params(.w = w, .h = p->height,
                .format = float_fmt, .sampleable = true, .renderable = true)) ||
            !pl_tex_recreate(gpu, &p->scaled[i], pl_tex_params(.w = p->width, .h = p->height,
                .format = float_fmt, .sampleable = true, .renderable = true)) ||
            !pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->source[i],
                .ptr = frame->data[comp->plane], .row_pitch = frame->linesize[comp->plane]))) return 0;
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

int fel_gpu_render(void *context, const AVDOVIMetadata *metadata,
                           const AVFrame *bl, const AVFrame *el, uint16_t *rgba)
{
    struct fel_gpu *p = context;
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
        ok = pl_dispatch_finish(p->dispatch, pl_dispatch_params(.shader = &sh, .target = p->target)) &&
             pl_tex_download(p->vk->gpu, pl_tex_transfer_params(.tex = p->target, .ptr = rgba));
    }
    pl_dispatch_abort(p->dispatch, &sh);
    pl_shader_free(&eh);
    return ok;
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
