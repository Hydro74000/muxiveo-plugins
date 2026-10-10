/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Raccordement Vulkan -> CUDA par transfert GPU FFmpeg. Aucun tampon RAM. */
#include "libavutil/hwcontext.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

typedef struct FelCuda {
    const AVClass *class;
    AVBufferRef *device,*frames;
} FelCuda;
static const AVOption mvo_fel_cuda_options[]={{NULL}};
AVFILTER_DEFINE_CLASS(mvo_fel_cuda);

static int query_formats(const AVFilterContext *ctx,AVFilterFormatsConfig **inputs,AVFilterFormatsConfig **outputs)
{
    static const enum AVPixelFormat in[]={AV_PIX_FMT_VULKAN,AV_PIX_FMT_NONE};
    static const enum AVPixelFormat out[]={AV_PIX_FMT_CUDA,AV_PIX_FMT_NONE};
    int ret=ff_formats_ref(ff_make_format_list(in),&inputs[0]->formats);
    if(ret<0)return ret;
    return ff_formats_ref(ff_make_format_list(out),&outputs[0]->formats);
}
static int config_output(AVFilterLink *link)
{
    AVFilterContext *ctx=link->src;
    FelCuda *s=ctx->priv;
    const FilterLink *in=ff_filter_link(ctx->inputs[0]);
    FilterLink *out=ff_filter_link(link);
    const AVHWFramesContext *source;
    AVHWFramesContext *frames;
    int ret;
    if(!in->hw_frames_ctx)return AVERROR(EINVAL);
    source=(const AVHWFramesContext*)in->hw_frames_ctx->data;
    if(source->format!=AV_PIX_FMT_VULKAN)return AVERROR(EINVAL);
    ret=av_hwdevice_ctx_create_derived(&s->device,AV_HWDEVICE_TYPE_CUDA,source->device_ref,0);
    if(ret<0) {
        av_log(ctx,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : périphérique CUDA dérivé impossible (%d).\n",ret);
        return ret;
    }
    s->frames=av_hwframe_ctx_alloc(s->device);
    if(!s->frames)return AVERROR(ENOMEM);
    frames=(AVHWFramesContext*)s->frames->data;
    frames->format=AV_PIX_FMT_CUDA;frames->sw_format=source->sw_format;
    frames->width=source->width;frames->height=source->height;frames->initial_pool_size=3;
    ret=av_hwframe_ctx_init(s->frames);
    if(ret<0)return ret;
    out->hw_frames_ctx=av_buffer_ref(s->frames);
    if(!out->hw_frames_ctx)return AVERROR(ENOMEM);
    link->w=ctx->inputs[0]->w;link->h=ctx->inputs[0]->h;
    link->time_base=ctx->inputs[0]->time_base;
    link->sample_aspect_ratio=ctx->inputs[0]->sample_aspect_ratio;
    out->frame_rate=in->frame_rate;
    return 0;
}
static int filter_frame(AVFilterLink *link,AVFrame *input)
{
    FelCuda *s=link->dst->priv;
    AVFrame *output=av_frame_alloc();
    int ret=output ? av_hwframe_get_buffer(s->frames,output,0) : AVERROR(ENOMEM);
    if(ret>=0)ret=av_hwframe_transfer_data(output,input,0);
    if(ret>=0)ret=av_frame_copy_props(output,input);
    av_frame_free(&input);
    if(ret<0) {
        av_log(link->dst,AV_LOG_ERROR,"MVO_FEL_RECOVERABLE : transfert Vulkan/CUDA impossible (%d).\n",ret);
        av_frame_free(&output);return ret;
    }
    return ff_filter_frame(link->dst->outputs[0],output);
}
static av_cold void uninit(AVFilterContext *ctx)
{
    FelCuda *s=ctx->priv;
    av_buffer_unref(&s->frames);av_buffer_unref(&s->device);
}
static const AVFilterPad inputs[]={ {.name="default",.type=AVMEDIA_TYPE_VIDEO,.filter_frame=filter_frame} };
static const AVFilterPad outputs[]={ {.name="default",.type=AVMEDIA_TYPE_VIDEO,.config_props=config_output} };
const FFFilter ff_vf_mvo_fel_cuda={
    .p.name="mvo_fel_cuda",.p.description="Transfert GPU Vulkan vers CUDA pour FEL.",
    .p.priv_class=&mvo_fel_cuda_class,.priv_size=sizeof(FelCuda),.uninit=uninit,
    FILTER_INPUTS(inputs),FILTER_OUTPUTS(outputs),FILTER_QUERY_FUNC2(query_formats),
    .flags_internal=FF_FILTER_FLAG_HWFRAME_AWARE,
};
