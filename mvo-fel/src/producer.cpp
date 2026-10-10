// SPDX-License-Identifier: LGPL-2.1-or-later
#include "fel_abi.h"
#include "fel_direct_abi.h"
#include "reconstruction.hpp"
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <iomanip>
#include <locale>
#include <sstream>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
// Préfixe stable de RpuDataHeader de la révision libdovi épinglée.
struct DoviHeader { uint8_t profile; const char *el_type; };
void *dovi_parse_unspec62_nalu(const uint8_t *, size_t);
const char *dovi_rpu_get_error(const void *);
const DoviHeader *dovi_rpu_get_header(const void *);
void dovi_rpu_free_header(const DoviHeader *);
void dovi_rpu_free(void *);
#ifdef MVO_FEL_VULKAN_PROTOTYPE
void *fel_gpu_create(int, int, const char *);
void fel_gpu_destroy(void *);
size_t fel_gpu_devices(char *, size_t);
int fel_gpu_device_name(void *, char *, size_t);
int fel_gpu_render(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, uint16_t *);
int fel_gpu_render_planar(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, AVFrame *);
void *fel_gpu_import(const void *, int, int);
int fel_gpu_render_vulkan(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, AVFrame *);
#endif
}

namespace {
using Clock = std::chrono::steady_clock;
struct Timer {
    double &elapsed;
    Clock::time_point start=Clock::now();
    explicit Timer(double &counter) : elapsed(counter) {}
    ~Timer() { elapsed+=std::chrono::duration<double>(Clock::now()-start).count(); }
};
struct Failure : std::runtime_error {
    int status;
    explicit Failure(const std::string &message, int code=MVO_FEL_ERROR)
        : std::runtime_error(message), status(code) {}
};
void check(int code, const char *operation, int status=MVO_FEL_ERROR) {
    if (code >= 0) return;
    char error[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(code, error, sizeof(error));
    throw Failure(std::string(operation) + " : " + error, status);
}
struct FrameFree { void operator()(AVFrame *p) const { av_frame_free(&p); } };
struct PacketFree { void operator()(AVPacket *p) const { av_packet_free(&p); } };
using Frame = std::unique_ptr<AVFrame, FrameFree>;
using Packet = std::unique_ptr<AVPacket, PacketFree>;
Frame frame() { Frame f(av_frame_alloc()); if (!f) throw std::bad_alloc(); return f; }
Packet packet() { Packet p(av_packet_alloc()); if (!p) throw std::bad_alloc(); return p; }
struct Layer {
    AVBSFContext *split = nullptr;
    AVCodecContext *decoder = nullptr;
    std::deque<Frame> ready;
    ~Layer() { av_bsf_free(&split); avcodec_free_context(&decoder); }
    void init(AVStream *stream, const char *mode, int threads) {
        const auto *bsf = av_bsf_get_by_name("dovi_split");
        if (!bsf) throw Failure("FFmpeg privé sans dovi_split");
        check(av_bsf_alloc(bsf, &split), "Allocation séparateur");
        check(avcodec_parameters_copy(split->par_in, stream->codecpar), "Paramètres source");
        split->time_base_in = stream->time_base;
        check(av_opt_set(split->priv_data, "mode", mode, 0), "Mode séparateur");
        check(av_bsf_init(split), "Initialisation séparateur");
        const auto *codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        decoder = avcodec_alloc_context3(codec);
        if (!decoder) throw std::bad_alloc();
        check(avcodec_parameters_to_context(decoder, split->par_out), "Paramètres décodeur");
        decoder->pkt_timebase = stream->time_base;
        decoder->thread_count = threads;
        decoder->err_recognition = AV_EF_EXPLODE;
        decoder->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
        check(avcodec_open2(decoder, codec, nullptr), "Ouverture décodeur HEVC");
    }
    void receive() {
        for (;;) {
            auto f = frame();
            const int ret = avcodec_receive_frame(decoder, f.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
            check(ret, "Décodage HEVC");
            if (f->decode_error_flags || (f->flags & AV_FRAME_FLAG_CORRUPT))
                throw Failure("Image HEVC corrompue");
            ready.push_back(std::move(f));
            if (ready.size() > 32) throw Failure("Désalignement BL/EL : file de décodage saturée");
        }
    }
    void send(AVPacket *input) {
        auto copy = packet();
        if (input) check(av_packet_ref(copy.get(), input), "Référence paquet");
        check(av_bsf_send_packet(split, input ? copy.get() : nullptr), "Séparation BL/EL");
        for (;;) {
            const int ret = av_bsf_receive_packet(split, copy.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            check(ret, "Lecture couche");
            int sent = avcodec_send_packet(decoder, copy.get());
            if (sent == AVERROR(EAGAIN)) { receive(); sent = avcodec_send_packet(decoder, copy.get()); }
            check(sent, "Envoi couche au décodeur");
            receive();
            av_packet_unref(copy.get());
        }
        if (!input) { check(avcodec_send_packet(decoder, nullptr), "Vidange HEVC"); receive(); }
    }
};
}

struct mvo_fel_context {
    std::string source, error;
    int stream_index, threads;
    AVRational fps;
    std::atomic<bool> cancelled{false};
    mvo_fel_write write = nullptr;
    mvo_fel_progress progress = nullptr;
    void *opaque = nullptr;
    AVFormatContext *input = nullptr, *output = nullptr;
    AVIOContext *io = nullptr;
    AVStream *stream = nullptr, *out_stream = nullptr;
    Layer bl, el;
    Frame rgb;
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    void *gpu_prototype = nullptr;
    bool gpu_attempted = false;
#endif
    int64_t frames = 0, last_pts = AV_NOPTS_VALUE;
    bool output_closed = false, raw = false, started = false;
    std::array<double,7> timing{}; // ouverture, lecture, BL, EL, pixels, NUT, pipe.
    std::string statistics;
    mvo_fel_layers layers = nullptr;
    bool pulling = false, drained = false;

    mvo_fel_context(const char *path, int index, int count, AVRational rate)
        : source(path), stream_index(index), threads(std::clamp(count, 1, 16)), fps(rate) {}
    ~mvo_fel_context() {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
        fel_gpu_destroy(gpu_prototype);
#endif
        avformat_close_input(&input);
        if (output) { output->pb = nullptr; avformat_free_context(output); }
        if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    }
    void check_cancelled() {
        if (cancelled.load()) throw Failure("Reconstruction annulée", MVO_FEL_CANCELLED);
    }
    static int interrupt(void *p) { return static_cast<mvo_fel_context *>(p)->cancelled.load(); }
    static int write_packet(void *p, const uint8_t *data, int size) {
        auto &self = *static_cast<mvo_fel_context *>(p);
        if (self.cancelled.load()) return AVERROR_EXIT;
        Timer timer(self.timing[6]);
        if (self.write(self.opaque, data, size)) {
            self.output_closed = true;
            return AVERROR(EPIPE);
        }
        return size;
    }
    void open() {
        input = avformat_alloc_context();
        if (!input) throw std::bad_alloc();
        input->interrupt_callback = {interrupt, this};
        check(avformat_open_input(&input, source.c_str(), nullptr, nullptr), "Ouverture source", MVO_FEL_INPUT_ERROR);
        check(avformat_find_stream_info(input, nullptr), "Inspection source", MVO_FEL_INPUT_ERROR);
        if (stream_index < 0 || static_cast<unsigned>(stream_index) >= input->nb_streams)
            throw Failure("Piste source absente");
        stream = input->streams[stream_index];
        if (stream->codecpar->codec_id != AV_CODEC_ID_HEVC) throw Failure("La source FEL doit être HEVC");
        raw = std::string(input->iformat->name) == "hevc";
        if (fps.num <= 0 || fps.den <= 0) fps = av_guess_frame_rate(input, stream, nullptr);
        if (raw && (fps.num <= 0 || fps.den <= 0)) throw Failure("Cadence du flux brut inconnue");
        // Décodeurs et reconstruction partagent le budget, jamais chacun le total.
        bl.init(stream, "bl_rpu", std::max(1, threads/3));
        el.init(stream, "el", std::max(1, threads/3));
    }
    void open_output(const AVFrame &base) {
        if (base.width <= 0 || base.height <= 0 || base.width > 8192 || base.height > 8192)
            throw Failure("Dimensions source invalides");
        check(avformat_alloc_output_context2(&output, nullptr, "nut", nullptr), "Création NUT");
        auto *buffer = static_cast<uint8_t *>(av_malloc(65536));
        if (!buffer) throw std::bad_alloc();
        io = avio_alloc_context(buffer, 65536, 1, this, nullptr, write_packet, nullptr);
        if (!io) { av_free(buffer); throw std::bad_alloc(); }
        io->seekable = 0;
        output->pb = io;
        output->flags |= AVFMT_FLAG_CUSTOM_IO;
        out_stream = avformat_new_stream(output, nullptr);
        if (!out_stream) throw std::bad_alloc();
        out_stream->time_base = raw ? av_inv_q(fps) : stream->time_base;
        out_stream->avg_frame_rate = fps;
        auto *p = out_stream->codecpar;
        p->codec_type = AVMEDIA_TYPE_VIDEO;
        p->codec_id = AV_CODEC_ID_RAWVIDEO;
        p->format = AV_PIX_FMT_GBRP16LE;
        p->codec_tag = avcodec_pix_fmt_to_codec_tag(AV_PIX_FMT_GBRP16LE);
        p->width = base.width; p->height = base.height;
        p->color_range = AVCOL_RANGE_JPEG;
        p->color_primaries = AVCOL_PRI_BT2020;
        p->color_trc = AVCOL_TRC_SMPTE2084;
        p->color_space = AVCOL_SPC_RGB;
        // NUT conserve ces tags ; le consommateur pose aussi setparams sur les images.
        av_dict_set(&out_stream->metadata, "color_primaries", "bt2020", 0);
        av_dict_set(&out_stream->metadata, "color_transfer", "smpte2084", 0);
        av_dict_set(&out_stream->metadata, "color_space", "gbr", 0);
        av_dict_set(&out_stream->metadata, "color_range", "pc", 0);
        check(avformat_write_header(output, nullptr), "En-tête NUT");
        rgb = frame();
        rgb->format = AV_PIX_FMT_GBRP16LE; rgb->width = base.width; rgb->height = base.height;
    }
    std::string device = "auto", backend = "cpu";
    void pair() {
        while (!bl.ready.empty() && !el.ready.empty()) {
            check_cancelled();
            auto base = std::move(bl.ready.front()); bl.ready.pop_front();
            auto enhancement = std::move(el.ready.front()); el.ready.pop_front();
            if (!base->opaque_ref || !enhancement->opaque_ref ||
                base->opaque_ref->size != sizeof(int64_t) || enhancement->opaque_ref->size != sizeof(int64_t) ||
                std::memcmp(base->opaque_ref->data, enhancement->opaque_ref->data, sizeof(int64_t)))
                throw Failure("Désalignement des access units BL/EL");
            if (!raw && (base->pts == AV_NOPTS_VALUE || base->pts != enhancement->pts))
                throw Failure("Horodatages BL/EL absents ou différents");
            const auto *sd = av_frame_get_side_data(base.get(), AV_FRAME_DATA_DOVI_METADATA);
            if (!sd) throw Failure("RPU absent de l'image BL");
            const auto *original = av_frame_get_side_data(base.get(), AV_FRAME_DATA_DOVI_RPU_BUFFER);
            if (!original) throw Failure("RPU original absent de l'image BL");
            const int kind = mvo_fel_classify(original->data, original->size);
            if (kind != 1 && kind != 3) throw Failure("RPU original invalide ou sans FEL pendant le décodage");
            const muxiveo::fel::Mapping mapping(*reinterpret_cast<const AVDOVIMetadata *>(sd->data));
            if (!mapping.residual) throw Failure("Résiduel désactivé pendant la reconstruction FEL");
            if (layers) {
                if(base->width<=0 || base->height<=0 || base->width>8192 || base->height>8192)
                    throw Failure("Dimensions source invalides");
                const auto timebase=raw ? av_inv_q(fps) : stream->time_base;
                const auto pts=raw ? frames : base->pts;
                if(last_pts!=AV_NOPTS_VALUE && pts<=last_pts)
                    throw Failure("Horodatages de présentation non croissants");
                last_pts=pts;
                if(layers(opaque,base.get(),enhancement.get(),sd->data,pts,
                          raw ? 1 : base->duration,timebase.num,timebase.den))
                    throw Failure("Consommateur de couches FEL arrêté",MVO_FEL_OUTPUT_CLOSED);
                ++frames;
                return;
            }
            if (!output) open_output(*base);
            if (base->width != rgb->width || base->height != rgb->height)
                throw Failure("Changement de dimensions pendant la reconstruction");
            auto p = packet();
            const int size = av_image_get_buffer_size(AV_PIX_FMT_GBRP16LE, rgb->width, rgb->height, 1);
            check(size, "Taille image"); check(av_new_packet(p.get(), size), "Allocation paquet RGB");
            check(av_image_fill_arrays(rgb->data, rgb->linesize, p->data,
                AV_PIX_FMT_GBRP16LE, rgb->width, rgb->height, 1), "Plans du paquet RGB");
            {
            Timer pixels(timing[4]);
#ifdef MVO_FEL_VULKAN_PROTOTYPE
            if (!gpu_attempted && device != "cpu") {
                gpu_attempted = true;
                gpu_prototype = fel_gpu_create(base->width, base->height, device.c_str());
                if (!gpu_prototype && device != "auto") throw Failure("Initialisation du GPU FEL sélectionné impossible");
                if (gpu_prototype) {
                    char name[256];
                    fel_gpu_device_name(gpu_prototype, name, sizeof(name));
                    backend = std::string("vulkan: ")+name;
                }
            }
            if (gpu_prototype) {
            if (!fel_gpu_render_planar(gpu_prototype,
                    reinterpret_cast<const AVDOVIMetadata *>(sd->data),
                    base.get(), enhancement.get(), rgb.get()))
                throw Failure("Échec de reconstruction du prototype Vulkan");
            } else
#endif
            {
            const int count = std::max(1, threads - 2*std::max(1, threads/3));
            const muxiveo::fel::FrameSampler bs(*base, base->width, base->height, false);
            const muxiveo::fel::FrameSampler es(*enhancement, base->width, base->height, true);
            std::vector<std::future<void>> workers;
            for (int i = 0; i < count; ++i) workers.push_back(std::async(std::launch::async, [&, i] {
                for (int y = i*base->height/count; y < (i+1)*base->height/count; ++y) {
                    check_cancelled();
                    muxiveo::fel::reconstruct_rows(bs, es, mapping, *rgb, y, y+1);
                }
            }));
            for (auto &worker : workers) worker.get();
            }
            }
            Timer nut(timing[5]);
            const AVRational timebase = raw ? av_inv_q(fps) : stream->time_base;
            p->pts = p->dts = av_rescale_q(raw ? frames : base->pts, timebase, out_stream->time_base);
            if (last_pts != AV_NOPTS_VALUE && p->pts <= last_pts)
                throw Failure("Horodatages de présentation non croissants");
            last_pts = p->pts;
            p->duration = av_rescale_q(raw ? 1 : base->duration, timebase, out_stream->time_base);
            p->stream_index = 0; p->flags |= AV_PKT_FLAG_KEY;
            check(av_interleaved_write_frame(output, p.get()), "Écriture NUT");
            ++frames;
            if (progress) progress(opaque, frames);
        }
    }
    void run() {
        if (started) throw Failure("Contexte déjà exécuté");
        started = true;
        { Timer timer(timing[0]); open(); }
        auto p = packet();
        int64_t identity = 0;
        for (;;) {
            check_cancelled();
            int ret;
            { Timer timer(timing[1]); ret=av_read_frame(input,p.get()); }
            if (ret == AVERROR_EOF) break;
            check(ret, "Lecture source", MVO_FEL_INPUT_ERROR);
            if (p->stream_index == stream_index) {
                av_buffer_unref(&p->opaque_ref);
                p->opaque_ref = av_buffer_alloc(sizeof(identity));
                if (!p->opaque_ref) throw std::bad_alloc();
                std::memcpy(p->opaque_ref->data, &identity, sizeof(identity)); ++identity;
                { Timer timer(timing[2]); bl.send(p.get()); }
                { Timer timer(timing[3]); el.send(p.get()); }
                pair();
            }
            av_packet_unref(p.get());
        }
        { Timer timer(timing[2]); bl.send(nullptr); }
        { Timer timer(timing[3]); el.send(nullptr); }
        pair();
        if (!bl.ready.empty() || !el.ready.empty()) throw Failure("Fin de flux BL/EL désalignée");
        if (!frames) throw Failure("Aucune image FEL reconstruite");
        check(av_write_trailer(output), "Fin NUT");
        avio_flush(io); check(io->error, "Vidange NUT");
    }
    int next() {
        if(!started) { started=true; pulling=true; Timer timer(timing[0]); open(); }
        if(!pulling)throw Failure("Contexte NUT déjà exécuté");
        const auto before=frames;
        pair();
        if(frames!=before)return MVO_FEL_OK;
        auto p=packet();
        while(!drained) {
            check_cancelled();
            int ret;
            {Timer timer(timing[1]);ret=av_read_frame(input,p.get());}
            if(ret==AVERROR_EOF) {
                {Timer timer(timing[2]);bl.send(nullptr);}
                {Timer timer(timing[3]);el.send(nullptr);}
                drained=true;
            } else {
                check(ret,"Lecture source",MVO_FEL_INPUT_ERROR);
                if(p->stream_index==stream_index) {
                    av_buffer_unref(&p->opaque_ref);
                    p->opaque_ref=av_buffer_alloc(sizeof(int64_t));
                    if(!p->opaque_ref)throw std::bad_alloc();
                    const auto identity=input_identity++;
                    std::memcpy(p->opaque_ref->data,&identity,sizeof(identity));
                    {Timer timer(timing[2]);bl.send(p.get());}
                    {Timer timer(timing[3]);el.send(p.get());}
                }
                av_packet_unref(p.get());
            }
            pair();
            if(frames!=before)return MVO_FEL_OK;
        }
        if(!bl.ready.empty() || !el.ready.empty())throw Failure("Fin de flux BL/EL désalignée");
        if(!frames)throw Failure("Aucune image FEL reconstruite");
        return 5;
    }
    int64_t input_identity=0;
};

uint32_t mvo_fel_abi_version() { return MVO_FEL_ABI; }
const char *mvo_fel_capabilities() {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    return "{\"abi\":1,\"version\":\"" MVO_FEL_VERSION "\",\"transport\":\"nut\",\"pixel_format\":\"gbrp16le\",\"cpu\":true,\"backend\":\"vulkan-prototype\",\"statistics\":1,\"device_selection\":1,\"experimental\":true}";
#else
    return "{\"abi\":1,\"version\":\"" MVO_FEL_VERSION "\",\"transport\":\"nut\",\"pixel_format\":\"gbrp16le\",\"cpu\":true,\"statistics\":1,\"device_selection\":1}";
#endif
}
int mvo_fel_classify(const uint8_t *nal, size_t size) {
    if (!nal || size < 3 || size > (1u << 20)) return -1;
    void *rpu = dovi_parse_unspec62_nalu(nal, size);
    if (!rpu) return -1;
    const auto *h = dovi_rpu_get_header(rpu);
    int result = -1;
    if (h && !dovi_rpu_get_error(rpu)) {
        result = h->profile == 7 ? 3 : 0;
        if (h->el_type) result = std::strcmp(h->el_type, "FEL") == 0 ? 1 :
            std::strcmp(h->el_type, "MEL") == 0 ? 2 : -1;
    }
    if (h) dovi_rpu_free_header(h);
    dovi_rpu_free(rpu);
    return result;
}
mvo_fel_context *mvo_fel_create(const char *source, int stream, int threads, int num, int den) {
    if (!source || !*source || stream < 0) return nullptr;
    try { return new mvo_fel_context(source, stream, threads, {num, den}); } catch (...) { return nullptr; }
}
size_t mvo_fel_devices(char *buffer, size_t capacity) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    return fel_gpu_devices(buffer, capacity);
#else
    if (buffer && capacity >= 3) std::memcpy(buffer, "[]", 3);
    return 3;
#endif
}
int mvo_fel_set_device(mvo_fel_context *p, const char *device) {
    if (!p || p->started || !device) return -1;
    const std::string value(device);
    if (value != "cpu" && value != "auto") {
        if (value.size() != 32 || value.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) return -1;
#ifndef MVO_FEL_VULKAN_PROTOTYPE
        return -1;
#endif
    }
    p->device = value;
    return 0;
}
const char *mvo_fel_backend(const mvo_fel_context *p) { return p ? p->backend.c_str() : ""; }
const char *mvo_fel_statistics(mvo_fel_context *p) {
    if (!p) return "{}";
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::fixed << std::setprecision(6) << "{\"frames\":" << p->frames;
    const char *names[]={"open_s","read_s","decode_bl_s","decode_el_s","pixels_s","nut_s","pipe_s"};
    for(size_t i=0;i<p->timing.size();++i)text << ",\"" << names[i] << "\":" << p->timing[i];
    text << '}';
    p->statistics=text.str();
    return p->statistics.c_str();
}
int mvo_fel_run(mvo_fel_context *p, mvo_fel_write write, mvo_fel_progress progress, void *opaque) {
    if (!p || !write) return MVO_FEL_ERROR;
    p->write = write; p->progress = progress; p->opaque = opaque;
    int result = MVO_FEL_OK;
    try { p->run(); }
    catch (const Failure &exc) { p->error = exc.what(); result = exc.status; }
    catch (const std::exception &exc) { p->error = exc.what(); result = MVO_FEL_ERROR; }
    catch (...) { p->error = "Erreur native inconnue"; result = MVO_FEL_ERROR; }
    if (p->cancelled.load()) return MVO_FEL_CANCELLED;
    return p->output_closed ? MVO_FEL_OUTPUT_CLOSED : result;
}
void mvo_fel_cancel(mvo_fel_context *p) { if (p) p->cancelled.store(true); }
const char *mvo_fel_error(const mvo_fel_context *p) { return p ? p->error.c_str() : "Contexte absent"; }
void mvo_fel_destroy(mvo_fel_context *p) { delete p; }
unsigned mvo_fel_libavutil_version() { return avutil_version(); }
int mvo_fel_next_layers(mvo_fel_context *p,mvo_fel_layers callback,void *opaque) {
    if(!p || !callback)return MVO_FEL_ERROR;
    p->layers=callback;p->opaque=opaque;
    try {p->check_cancelled();return p->next();}
    catch(const Failure &e){p->error=e.what();return p->cancelled.load() ? MVO_FEL_CANCELLED : e.status;}
    catch(const std::exception &e){p->error=e.what();return MVO_FEL_ERROR;}
    catch(...){p->error="Erreur native inconnue";return MVO_FEL_ERROR;}
}
void *mvo_fel_vulkan_create(const void *device,int w,int h) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    if(!device || w<=0 || h<=0 || w>8192 || h>8192)return nullptr;
    return fel_gpu_import(device,w,h);
#else
    return nullptr;
#endif
}
int mvo_fel_vulkan_render(void *gpu,const void *md,const void *bl,const void *el,void *output) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    if(!gpu || !md || !bl || !el || !output)return -1;
    return fel_gpu_render_vulkan(gpu,static_cast<const AVDOVIMetadata*>(md),
        static_cast<const AVFrame*>(bl),static_cast<const AVFrame*>(el),static_cast<AVFrame*>(output)) ? 0 : -1;
#else
    return -1;
#endif
}
void mvo_fel_vulkan_destroy(void *gpu) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    fel_gpu_destroy(gpu);
#endif
}
