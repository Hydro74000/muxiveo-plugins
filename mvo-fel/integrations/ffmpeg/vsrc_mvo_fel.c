/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Source native FEL : couches CPU décodées, reconstruction et sortie Vulkan.
 * À compiler dans la révision FFmpeg épinglée, pas de greffe dans le FFmpeg système. */
#include "libavformat/avformat.h"
#include "libavutil/opt.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_vulkan.h"
#include "libavutil/pixdesc.h"
#include "compat/w32dlfcn.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"
#include "fel_direct_abi.h"

typedef struct FelSource {
    const AVClass *class;
    char *source, *library, *format, *chroma, *dither, *precision;
    int stream, threads, width, height, callback_error;
    int crop_left, crop_top, crop_right, crop_bottom, crop_w, crop_h, p010;
    AVRational time_base, frame_rate, sar;
    int64_t end_pts;
    void *module, *renderer;
    mvo_fel_context *context;
    AVBufferRef *frames;
    AVFrame *produced;
    mvo_fel_context *(*create)(const char *,int,int,int,int);
    void (*destroy)(mvo_fel_context *);
    const char *(*error)(const mvo_fel_context *);
    int (*next)(mvo_fel_context *,mvo_fel_layers,void *);
    void *(*gpu_create)(const void *,int,int);
    int (*render)(void *,const void *,const void *,const void *,void *);
    void (*gpu_destroy)(void *);
    int (*configure)(void *,const char *,const char *,int,int,int,int);
    int (*set_precision)(void *,const char *);
    int (*decoder_device)(mvo_fel_context *,const void *);
} FelSource;

#define OFFSET(x) offsetof(FelSource,x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
static const AVOption mvo_fel_options[]={
    {"source","Source FEL intacte",OFFSET(source),AV_OPT_TYPE_STRING,{.str=NULL},0,0,FLAGS},
    {"library","Bibliothèque mvo-fel exacte",OFFSET(library),AV_OPT_TYPE_STRING,{.str=NULL},0,0,FLAGS},
    {"stream","Index absolu de piste",OFFSET(stream),AV_OPT_TYPE_INT,{.i64=0},0,INT_MAX,FLAGS},
    {"threads","Budget de décodage",OFFSET(threads),AV_OPT_TYPE_INT,{.i64=12},1,16,FLAGS},
    {"format","Sortie : gbrp16 (RGB PQ) ou p010 (finale tramée)",OFFSET(format),AV_OPT_TYPE_STRING,{.str="gbrp16"},0,0,FLAGS},
    {"chroma","Position chroma de la sortie p010",OFFSET(chroma),AV_OPT_TYPE_STRING,{.str="left"},0,0,FLAGS},
    {"dither","Tramage de la sortie p010",OFFSET(dither),AV_OPT_TYPE_STRING,{.str="blue"},0,0,FLAGS},
    {"precision","Précision du rendu : fast, standard ou exact (double natif requis)",OFFSET(precision),AV_OPT_TYPE_STRING,{.str="fast"},0,0,FLAGS},
    {"crop_left","Recadrage p010 : marge gauche (paire)",OFFSET(crop_left),AV_OPT_TYPE_INT,{.i64=0},0,8192,FLAGS},
    {"crop_top","Recadrage p010 : marge haute (paire)",OFFSET(crop_top),AV_OPT_TYPE_INT,{.i64=0},0,8192,FLAGS},
    {"crop_right","Recadrage p010 : marge droite (paire)",OFFSET(crop_right),AV_OPT_TYPE_INT,{.i64=0},0,8192,FLAGS},
    {"crop_bottom","Recadrage p010 : marge basse (paire)",OFFSET(crop_bottom),AV_OPT_TYPE_INT,{.i64=0},0,8192,FLAGS},
    {NULL}
};
AVFILTER_DEFINE_CLASS(mvo_fel);

static av_cold int init(AVFilterContext *ctx)
{
    FelSource *s=ctx->priv;
    AVFormatContext *input=NULL;
    unsigned (*version)(void),(*abi)(void);
    int ret;
    if(!s->source || !s->library)return AVERROR(EINVAL);
    s->module=dlopen(s->library,RTLD_NOW|RTLD_LOCAL);
    if(!s->module) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : chargement FEL impossible : %s\n",dlerror());
        return AVERROR_EXTERNAL;
    }
#define LOAD(member,symbol) do { *(void **)(&s->member)=dlsym(s->module,symbol); \
    if(!s->member) { av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : symbole absent : %s\n",symbol); \
        return AVERROR_EXTERNAL; } } while(0)
    *(void **)(&version)=dlsym(s->module,"mvo_fel_libavutil_version");
    *(void **)(&abi)=dlsym(s->module,"mvo_fel_abi_version");
    if(!version || !abi || abi()!=1 || version()!=avutil_version()) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : ABI FFmpeg/FEL différente : version épinglée obligatoire.\n");
        return AVERROR_EXTERNAL;
    }
    LOAD(create,"mvo_fel_create");LOAD(destroy,"mvo_fel_destroy");LOAD(error,"mvo_fel_error");
    LOAD(next,"mvo_fel_next_layers");LOAD(gpu_create,"mvo_fel_vulkan_create");
    LOAD(render,"mvo_fel_vulkan_render");LOAD(gpu_destroy,"mvo_fel_vulkan_destroy");
#undef LOAD
    s->p010=s->format && !strcmp(s->format,"p010");
    if(s->format && !s->p010 && strcmp(s->format,"gbrp16"))return AVERROR(EINVAL);
    *(void **)(&s->configure)=dlsym(s->module,"mvo_fel_vulkan_configure");
    *(void **)(&s->decoder_device)=dlsym(s->module,"mvo_fel_set_decoder_device");
    *(void **)(&s->set_precision)=dlsym(s->module,"mvo_fel_vulkan_precision");
    if(s->p010 && !s->configure) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : sortie p010 absente de cette bibliothèque FEL.\n");
        return AVERROR_EXTERNAL;
    }
    ret=avformat_open_input(&input,s->source,NULL,NULL);
    if(ret<0)return ret;
    ret=avformat_find_stream_info(input,NULL);
    if(ret>=0) {
        if(s->stream>=input->nb_streams || input->streams[s->stream]->codecpar->codec_id!=AV_CODEC_ID_HEVC)
            ret=AVERROR(EINVAL);
        else {
            const AVStream *stream=input->streams[s->stream];
            s->width=stream->codecpar->width;s->height=stream->codecpar->height;
            s->time_base=stream->time_base;
            s->frame_rate=av_guess_frame_rate(input,input->streams[s->stream],NULL);
            s->sar=stream->sample_aspect_ratio.num ? stream->sample_aspect_ratio : (AVRational){1,1};
        }
    }
    avformat_close_input(&input);
    if(ret<0)return ret;
    s->context=s->create(s->source,s->stream,s->threads,s->frame_rate.num,s->frame_rate.den);
    return s->context ? 0 : AVERROR(ENOMEM);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    FelSource *s=ctx->priv;
    av_frame_free(&s->produced);
    if(s->gpu_destroy)s->gpu_destroy(s->renderer);
    if(s->destroy)s->destroy(s->context);
    av_buffer_unref(&s->frames);
    if(s->module)dlclose(s->module);
}

static int config_output(AVFilterLink *link)
{
    AVFilterContext *ctx=link->src;
    FelSource *s=ctx->priv;
    FilterLink *l=ff_filter_link(link);
    AVHWFramesContext *frames;
    AVVulkanFramesContext *vk;
    int ret;
    if(!ctx->hw_device_ctx || ((AVHWDeviceContext*)ctx->hw_device_ctx->data)->type!=AV_HWDEVICE_TYPE_VULKAN) {
        av_log(ctx,AV_LOG_ERROR,"FEL direct exige -init_hw_device vulkan et -filter_hw_device.\n");
        return AVERROR(EINVAL);
    }
    if(s->width<=0 || s->height<=0 || s->width>8192 || s->height>8192)return AVERROR(EINVAL);
    s->frames=av_hwframe_ctx_alloc(ctx->hw_device_ctx);
    if(!s->frames)return AVERROR(ENOMEM);
    s->crop_w=s->width-s->crop_left-s->crop_right;
    s->crop_h=s->height-s->crop_top-s->crop_bottom;
    if(s->p010) {
        /* Zone produite : marges paires incluses dans l'image source. */
        if((s->crop_left|s->crop_top|s->crop_right|s->crop_bottom)&1 || s->crop_w<=0 || s->crop_h<=0)
            return AVERROR(EINVAL);
    } else if(s->crop_left || s->crop_top || s->crop_right || s->crop_bottom)return AVERROR(EINVAL);
    frames=(AVHWFramesContext*)s->frames->data;
    frames->format=AV_PIX_FMT_VULKAN;frames->sw_format=s->p010 ? AV_PIX_FMT_P010 : AV_PIX_FMT_GBRP16LE;
    frames->width=s->p010 ? s->crop_w : s->width;frames->height=s->p010 ? s->crop_h : s->height;
    frames->initial_pool_size=3;
    vk=frames->hwctx;
    vk->usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|
              VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_STORAGE_BIT;
    vk->flags=AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE;
    ret=av_hwframe_ctx_init(s->frames);
    if(ret<0)return ret;
    s->renderer=s->gpu_create(ctx->hw_device_ctx->data,s->width,s->height);
    if(!s->renderer) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : initialisation du renderer FEL impossible.\n");
        return AVERROR_EXTERNAL;
    }
    /* Précision exacte ou standard : GPU calculant en double indispensable, sinon repli du transport. */
    if(s->precision && strcmp(s->precision,"fast") &&
       (!s->set_precision || s->set_precision(s->renderer,s->precision))) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : précision %s indisponible sur ce GPU.\n",s->precision);
        return AVERROR_EXTERNAL;
    }
    if(s->p010 && s->configure(s->renderer,s->chroma,s->dither,s->crop_left,s->crop_top,s->crop_w,s->crop_h)) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : sortie p010 refusée (position chroma, tramage ou zone).\n");
        return AVERROR_EXTERNAL;
    }
    /* Couches décodées sur le même GPU que le rendu, sans retour en RAM (contrôle au départ). */
    if(s->decoder_device && s->decoder_device(s->context,ctx->hw_device_ctx))
        av_log(ctx,AV_LOG_WARNING,"Décodage Vulkan des couches FEL indisponible : décodage CPU.\n");
    l->hw_frames_ctx=av_buffer_ref(s->frames);
    if(!l->hw_frames_ctx)return AVERROR(ENOMEM);
    l->frame_rate=s->frame_rate;
    link->w=frames->width;link->h=frames->height;link->time_base=s->time_base;link->sample_aspect_ratio=s->sar;
    return 0;
}

static int reconstructed(void *opaque,const void *base,const void *enhancement,const void *metadata,
                         int64_t pts,int64_t duration,int num,int den)
{
    FelSource *s=opaque;
    AVFrame *f=av_frame_alloc();
    int ret;
    if(!f) {s->callback_error=AVERROR(ENOMEM);return 1;}
    ret=av_hwframe_get_buffer(s->frames,f,0);
    if(ret>=0 && s->render(s->renderer,metadata,base,enhancement,f))ret=AVERROR_EXTERNAL;
    if(ret<0) {av_frame_free(&f);s->callback_error=ret;return 1;}
    f->pts=av_rescale_q(pts,(AVRational){num,den},s->time_base);
    f->duration=av_rescale_q(duration,(AVRational){num,den},s->time_base);
    if(f->duration<=0 && s->frame_rate.num>0)
        f->duration=av_rescale_q(1,av_inv_q(s->frame_rate),s->time_base);
    s->end_pts=f->pts+f->duration;
    f->sample_aspect_ratio=s->sar;
    f->color_primaries=AVCOL_PRI_BT2020;f->color_trc=AVCOL_TRC_SMPTE2084;
    if(s->p010) {
        f->color_range=AVCOL_RANGE_MPEG;f->colorspace=AVCOL_SPC_BT2020_NCL;
        f->chroma_location=av_chroma_location_from_name(s->chroma);
    } else {
        f->color_range=AVCOL_RANGE_JPEG;f->colorspace=AVCOL_SPC_RGB;
    }
    // Les instructions RPU appliquées ne sont pas recopiées sur l'image.
    s->produced=f;
    return 0;
}

static int request_frame(AVFilterLink *link)
{
    FelSource *s=link->src->priv;
    int ret=s->next(s->context,reconstructed,s);
    AVFrame *frame;
    if(ret==5) {
        ff_outlink_set_status(link,AVERROR_EOF,s->end_pts);
        return AVERROR_EOF;
    }
    if(ret || !s->produced) {
        av_log(link->src,AV_LOG_ERROR,"%s : %s\n",ret==MVO_FEL_INPUT_ERROR ? "MVO_FEL_SOURCE_ERROR" :
               "MVO_FEL_RECOVERABLE",s->error(s->context));
        return s->callback_error ? s->callback_error : AVERROR_EXTERNAL;
    }
    frame=s->produced;s->produced=NULL;
    return ff_filter_frame(link,frame);
}

static const AVFilterPad outputs[]={
    {.name="default",.type=AVMEDIA_TYPE_VIDEO,.config_props=config_output,.request_frame=request_frame},
};
const FFFilter ff_vsrc_mvo_fel={
    .p.name="mvo_fel",.p.description="Reconstruction FEL native vers des images Vulkan.",
    .p.priv_class=&mvo_fel_class,.p.flags=AVFILTER_FLAG_HWDEVICE,.p.inputs=NULL,.priv_size=sizeof(FelSource),
    .init=init,.uninit=uninit,FILTER_OUTPUTS(outputs),FILTER_SINGLE_PIXFMT(AV_PIX_FMT_VULKAN),
    .flags_internal=FF_FILTER_FLAG_HWFRAME_AWARE,
};
