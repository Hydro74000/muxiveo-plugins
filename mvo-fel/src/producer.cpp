// SPDX-License-Identifier: LGPL-2.1-or-later
#include "fel_abi.h"
#include "fel_direct_abi.h"
#include "reconstruction.hpp"
#include "output_format.h"
#include "frame_stats.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <map>
#include <locale>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <pthread.h>
#include <unistd.h>
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
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
size_t fel_gpu_devices(char *, size_t, int);
int fel_gpu_device_name(void *, char *, size_t);
int fel_gpu_render(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, uint16_t *);
int fel_gpu_render_planar(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, AVFrame *);
void *fel_gpu_import(const void *, int, int);
int fel_gpu_render_vulkan(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, AVFrame *);
int fel_gpu_render_yuv(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, const fel_output *, AVFrame *);
void fel_gpu_configure(void *, const fel_output *);
void fel_gpu_timings(void *, double *);
int fel_gpu_set_precision(void *, int);
int fel_gpu_fixed_variant(const void *);
int fel_gpu_inherit(void *, const void *);
void fel_gpu_certified(void *, long long *, long long *);
int fel_gpu_take_overflow(void *);
int fel_gpu_frame_statistics(void *, int, int, int, int, fel_frame_stats *);
void fel_gpu_set_cpu_threads(void *, int);
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
AVPixelFormat vulkan_format(AVCodecContext *, const AVPixelFormat *formats) {
    for (; *formats != AV_PIX_FMT_NONE; ++formats)
        if (*formats == AV_PIX_FMT_VULKAN) return *formats;
    return AV_PIX_FMT_NONE;
}
struct Layer {
    AVBSFContext *split = nullptr;
    AVCodecContext *decoder = nullptr;
    std::deque<Frame> ready;
    Layer() = default;
    Layer(const Layer &) = delete;
    Layer &operator=(const Layer &) = delete;
    ~Layer() { reset(); }
    // Libère décodeur et images (références au périphérique Vulkan comprises).
    void reset() { ready.clear(); av_bsf_free(&split); avcodec_free_context(&decoder); }
    // hardware : périphérique Vulkan FFmpeg ; images décodées sur ce GPU (Vulkan Video).
    void init(AVStream *stream, const char *mode, int threads, AVBufferRef *hardware = nullptr) {
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
        if (hardware) {
            decoder->hw_device_ctx = av_buffer_ref(hardware);
            if (!decoder->hw_device_ctx) throw std::bad_alloc();
            decoder->get_format = vulkan_format;
            // Images en vol : files d'appariement, de rendu et anticipation.
            decoder->extra_hw_frames = 16;
            decoder->thread_count = 1;
        }
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
// Interruption d'un étage par l'aval (file abandonnée) : pas une erreur en soi.
struct Stopped {};
// File bornée, un producteur et un consommateur ; l'annulation interrompt les attentes.
template <typename T> class Channel {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<T> items;
    size_t capacity;
    bool closed = false, aborted = false;
    static void wait(std::unique_lock<std::mutex> &lock, std::condition_variable &cv, double &waited) {
        const auto start = Clock::now();
        cv.wait_for(lock, std::chrono::milliseconds(50));
        waited += std::chrono::duration<double>(Clock::now()-start).count();
    }
public:
    explicit Channel(size_t size) : capacity(size) {}
    bool push(T value, const std::atomic<bool> &cancelled, double &waited) {
        std::unique_lock lock(mutex);
        while (!aborted && items.size() >= capacity && !cancelled.load()) wait(lock, changed, waited);
        if (aborted || cancelled.load()) return false;
        items.push_back(std::move(value));
        changed.notify_all();
        return true;
    }
    // Fin normale : les éléments restants sont rendus avant false.
    bool pop(T &value, const std::atomic<bool> &cancelled, double &waited) {
        std::unique_lock lock(mutex);
        while (!aborted && items.empty() && !closed && !cancelled.load()) wait(lock, changed, waited);
        if (aborted || cancelled.load() || items.empty()) return false;
        value = std::move(items.front());
        items.pop_front();
        changed.notify_all();
        return true;
    }
    void close() { std::lock_guard lock(mutex); closed = true; changed.notify_all(); }
    void abort() { std::lock_guard lock(mutex); aborted = true; items.clear(); changed.notify_all(); }
};
// Threads persistants de la reconstruction CPU, par blocs de lignes.
class RowPool {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work, done;
    std::function<void(int)> task;
    int next = 0, rows = 0, active = 0;
    uint64_t generation = 0;
    bool stop = false;
    std::exception_ptr failure;
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex);
                work.wait(lock, [&] { return stop || generation != seen; });
                if (stop) return;
                seen = generation;
            }
            for (;;) {
                int row;
                {
                    std::lock_guard lock(mutex);
                    if (failure || next >= rows) break;
                    row = next++;
                }
                try { task(row); }
                catch (...) { std::lock_guard lock(mutex); if (!failure) failure = std::current_exception(); }
            }
            std::lock_guard lock(mutex);
            if (--active == 0) done.notify_all();
        }
    }
public:
    explicit RowPool(int count) {
        for (int i = 0; i < std::max(1, count); ++i) workers.emplace_back([this] { loop(); });
    }
    ~RowPool() {
        { std::lock_guard lock(mutex); stop = true; }
        work.notify_all();
        for (auto &worker : workers) worker.join();
    }
    void run(int count, std::function<void(int)> function) {
        std::unique_lock lock(mutex);
        task = std::move(function); rows = count; next = 0; failure = nullptr;
        active = static_cast<int>(workers.size()); ++generation;
        work.notify_all();
        done.wait(lock, [&] { return active == 0; });
        if (failure) std::rethrow_exception(failure);
    }
};
// Image appariée et validée, prête pour la reconstruction ; possède BL et EL.
struct Job {
    Frame base, enhancement;
    const AVDOVIMetadata *metadata;
    muxiveo::fel::Mapping mapping;
    int64_t pts, duration;
    AVRational timebase;
    bool scene_start = false;  // drapeau de changement de plan du RPU propre de l'image
    std::array<int, 4> active{};  // zone active L5 du RPU (marges gauche, haut, droite, bas)
};
struct Rendered {
    Packet packet;
    int width, height;
    AVRational timebase;
    std::unique_ptr<fel_frame_stats> stats;  // statistiques de l'image (mvo_fel_set_frame_statistics)
    bool scene_start = false;
    std::array<int, 4> window{};  // fenêtre mesurée (x, y, largeur, hauteur), coordonnées de la source
};
}

struct mvo_fel_context {
    std::string source, error;
    int stream_index, threads;
    AVRational fps;
    std::atomic<bool> cancelled{false};
    mvo_fel_write write = nullptr;
    intptr_t fd = -1;
    mvo_fel_progress progress = nullptr;
    void *opaque = nullptr;
    AVFormatContext *input = nullptr, *output = nullptr;
    AVIOContext *io = nullptr;
    AVStream *stream = nullptr, *out_stream = nullptr;
    Layer bl, el;
    Frame scratch;   // RGB intermédiaire du rendu CPU (un seul fil de rendu)
    AVBufferPool *pool = nullptr;
    // Format produit : RGB PQ 16 bits par défaut, Y'CbCr 4:2:0 final ou 16 bits sur demande.
    fel_output output_format{FEL_OUT_GBRP16, FEL_DITHER_BLUE, AVCHROMA_LOC_LEFT, 0, 0, 0, 0};
    // Marges retirées des images Y'CbCr (gauche, haut, droite, bas), paires.
    std::array<int, 4> crop{};
    int width = 0, height = 0, packet_size = 0;
    std::unique_ptr<RowPool> rows;
    std::unique_ptr<muxiveo::fel::FastReconstruction> fast;
    int render_threads = 1;
    // Images refaites sur le CPU certifié après saturation de la liste des indécises du GPU
    // (mêmes octets) ; le verrou sérialise ce chemin entre les fils de rendu GPU.
    std::mutex cpu_mutex;
    std::atomic<long long> cpu_rescue_frames{0};
    // Statistiques de l'image finale (JSON Lines, mvo_fel_set_frame_statistics) : fichier, image
    // suivante, histogramme de maxRGB du plan en cours (écrit à la fin du plan).
    std::string stats_path;
    AVIOContext *stats_io = nullptr;
    int64_t stats_frame = 0, scene_first = -1, scene_frames = 0;
    std::array<uint64_t, FEL_STATS_BINS> scene_hist{};
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    void *gpu_prototype = nullptr;
    bool gpu_attempted = false;
    // Contextes de rendu supplémentaires (lot P) : plusieurs images en vol, une par fil de rendu.
    std::vector<void *> gpu_extra;
    int render_workers = 2;      // images en vol (mvo_fel_set_render_workers, 1 à 4)
#endif
    // Fil de rendu : contexte GPU (ou nul : CPU), cadre des plans du paquet, mesures propres.
    struct Worker {
        void *gpu = nullptr;
        Frame frame;
        double pixels = 0, starved = 0, blocked = 0;
    };
    std::mutex render_mutex;
    // Périphérique Vulkan FFmpeg partagé par les décodeurs et le rendu (Vulkan Video).
    AVBufferRef *hw_device = nullptr;
    bool hw_decode = false;
    std::string decode_note;
    // emitted : images appariées (décodage) ; frames : images écrites ou rendues à l'appelant.
    int64_t frames = 0, emitted = 0, last_pts = AV_NOPTS_VALUE;
    bool output_closed = false, raw = false, started = false;
    // Couche produite (mvo_fel_set_layer) : LAYER_RECONSTRUCTED, image reconstruite par le RPU et l'EL
    // (défaut) ; LAYER_BASE, couche de base lue comme un signal HDR10 (RPU identité synthétisé) ;
    // LAYER_MAPPED, couche de base passée par les courbes du RPU sans résiduel (lecture Dolby Vision
    // sans EL : P8.1, MEL). measure_only (mvo_fel_set_measure) : statistiques seules, aucune image.
    enum { LAYER_RECONSTRUCTED, LAYER_BASE, LAYER_MAPPED };
    int layer = LAYER_RECONSTRUCTED;
    bool measure_only = false;
    AVBufferRef *identity = nullptr;  // RPU identité partagé par les images (couche de base)
    std::array<double,7> timing{}; // ouverture, lecture, BL, EL, pixels, NUT, pipe.
    // Attentes des files : décodage bloqué, rendu affamé, rendu bloqué, écriture affamée.
    std::array<double,4> waits{};
    // Rendu certifié CPU : composantes reprises en double-double et indécises (aucune attendue).
    std::atomic<long long> certified_rescued{0}, certified_undecided{0};
    Clock::time_point began{}, ended{};
    std::string statistics;
    mvo_fel_layers layers = nullptr;
    bool pulling = false;
    // Seek : cible en base de temps de la piste, ou nombre d'images ignorées (HEVC brut).
    int64_t start_us = 0, target = AV_NOPTS_VALUE, skip = 0, skipped = 0;
    // Horodatages relatifs à la première image produite (comme FFmpeg en entrée d'un NUT).
    bool rebase_first = false;
    int64_t first_pts = AV_NOPTS_VALUE;
    // Mode pull : décodage anticipé sur un thread, une image rendue par appel.
    std::unique_ptr<Channel<std::unique_ptr<Job>>> prefetched;
    std::thread prefetcher;
    std::mutex failure_mutex;
    std::exception_ptr prefetch_failure;

    mvo_fel_context(const char *path, int index, int count, AVRational rate)
        : source(path), stream_index(index), threads(std::clamp(count, 1, 16)), fps(rate) {}
    ~mvo_fel_context() {
        stop_prefetch();
        rows.reset();
#ifdef MVO_FEL_VULKAN_PROTOTYPE
        for (void *gpu : gpu_extra) fel_gpu_destroy(gpu);
        fel_gpu_destroy(gpu_prototype);
#endif
        bl.reset(); el.reset();
        av_buffer_unref(&hw_device);
        avformat_close_input(&input);
        if (output) { output->pb = nullptr; avformat_free_context(output); }
        if (io) { av_freep(&io->buffer); avio_context_free(&io); }
        avio_closep(&stats_io);
        av_buffer_pool_uninit(&pool);
        av_buffer_unref(&identity);
    }
    void check_cancelled() {
        if (cancelled.load()) throw Failure("Reconstruction annulée", MVO_FEL_CANCELLED);
    }
    static int interrupt(void *p) { return static_cast<mvo_fel_context *>(p)->cancelled.load(); }
    // Écriture complète sur le pipe ; false si l'aval est fermé ou en erreur.
    bool write_fd(const uint8_t *data, size_t size) {
        while (size) {
#ifdef _WIN32
            DWORD written = 0;
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size, 1u << 30));
            if (!WriteFile(reinterpret_cast<HANDLE>(fd), data, chunk, &written, nullptr) || !written) return false;
#else
            const ssize_t written = ::write(static_cast<int>(fd), data, size);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) return false;
#endif
            data += written; size -= static_cast<size_t>(written);
        }
        return true;
    }
    static int write_packet(void *p, const uint8_t *data, int size) {
        auto &self = *static_cast<mvo_fel_context *>(p);
        if (self.cancelled.load()) return AVERROR_EXIT;
        Timer timer(self.timing[6]);
        if (self.fd >= 0 ? !self.write_fd(data, static_cast<size_t>(size)) : self.write(self.opaque, data, size) != 0) {
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
        if (stream_index < 0 || static_cast<unsigned>(stream_index) >= input->nb_streams)
            throw Failure("Piste source absente");
        // Seule la piste reconstruite est analysée puis lue : ni audio ni sous-titres décodés.
        for (unsigned i = 0; i < input->nb_streams; ++i)
            if (static_cast<int>(i) != stream_index) input->streams[i]->discard = AVDISCARD_ALL;
        check(avformat_find_stream_info(input, nullptr), "Inspection source", MVO_FEL_INPUT_ERROR);
        stream = input->streams[stream_index];
        if (stream->codecpar->codec_id != AV_CODEC_ID_HEVC) throw Failure("La source FEL doit être HEVC");
        raw = std::string(input->iformat->name) == "hevc";
        if (fps.num <= 0 || fps.den <= 0) fps = av_guess_frame_rate(input, stream, nullptr);
        if (raw && (fps.num <= 0 || fps.den <= 0)) throw Failure("Cadence du flux brut inconnue");
        if (start_us > 0) seek();
#ifdef MVO_FEL_VULKAN_PROTOTYPE
        if (!pulling && device != "cpu") prepare_gpu();
#endif
        if (hw_device && !hw_decode) {
            // Même GPU, même source, même piste : résultat réutilisé (passes, captures d'aperçu).
            static std::mutex cache_mutex;
            static std::map<std::string, std::pair<bool, std::string>> cache;
            // GPU désigné par UUID uniquement : « auto » peut changer de carte d'un appel à l'autre.
            const std::string key = device + "|" + source + "|" + std::to_string(stream_index) + "|"
                                  + std::to_string(layer != LAYER_RECONSTRUCTED);
            const bool cacheable = device.size() == 32;
            if (cacheable) {
                std::lock_guard lock(cache_mutex);
                const auto found = cache.find(key);
                if (found != cache.end()) std::tie(hw_decode, decode_note) = found->second;
            }
            if (decode_note.empty()) {
                hw_decode = verify_hw_decode();
                if (cacheable) {
                    std::lock_guard lock(cache_mutex);
                    cache[key] = {hw_decode, decode_note};
                }
            }
            backend += (backend.empty() ? "" : ", ") + decode_note;
        }
        // Décodeurs et reconstruction partagent le budget, jamais chacun le total. Rendu CPU
        // (~2–6 i/s UHD) : décodage au plus juste, le reste des threads à la reconstruction.
        bool cpu_render = true;
#ifdef MVO_FEL_VULKAN_PROTOTYPE
        cpu_render = gpu_prototype == nullptr && !pulling;
#endif
        const bool base_only = layer != LAYER_RECONSTRUCTED;
        const int bl_threads = base_only ? std::max(1, threads/2)
                             : cpu_render ? std::max(1, threads/4) : std::max(1, threads/3);
        const int el_threads = base_only ? 0 : cpu_render ? 1 : std::max(1, threads/3);
        render_threads = std::max(1, threads - bl_threads - el_threads);
#ifdef MVO_FEL_VULKAN_PROTOTYPE
        // Indécises reprises par le CPU certifié : fils de rendu CPU (inactifs en mode GPU) répartis
        // entre les contextes, jamais au-delà du budget donné par l'appelant.
        if (gpu_prototype) {
            const int share = std::max(1, render_threads / int(1 + gpu_extra.size()));
            fel_gpu_set_cpu_threads(gpu_prototype, share);
            for (void *extra : gpu_extra) fel_gpu_set_cpu_threads(extra, share);
        }
#endif
        bl.init(stream, "bl_rpu", bl_threads, hw_decode ? hw_device : nullptr);
        if (layer == LAYER_BASE) identity = identity_metadata();
        if (!base_only) el.init(stream, "el", el_threads, hw_decode ? hw_device : nullptr);
    }
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    // Choix automatique selon la précision. Rapide : premier GPU dédié, sinon le premier.
    // Standard : GPU dédié calculant en double, sinon intégré en double. Exact : GPU dédié
    // en double uniquement (sur iGPU, le double natif partout est plus lent que le CPU).
    // Chaîne vide : rendu CPU.
    std::string automatic_uuid() const {
        std::vector<char> json(mvo_fel_devices_features(nullptr, 0) + 1);
        if (json.size() < 3 || !mvo_fel_devices_features(json.data(), json.size())) return "";
        const std::string text(json.data());
        std::string candidates[2];   // dédié, intégré
        for (size_t at = text.find("{\"uuid\":\""); at != std::string::npos; at = text.find("{\"uuid\":\"", at + 1)) {
            const std::string uuid = text.substr(at + 9, 32), entry = text.substr(at, text.find('}', at) - at);
            const bool discrete = entry.find("\"kind\":\"discrete\"") != std::string::npos;
            const bool float64 = entry.find("\"float64\":true") != std::string::npos;
            if (precision != FEL_PRECISION_FAST && !float64) continue;
            std::string &slot = candidates[discrete ? 0 : 1];
            if (slot.empty()) slot = uuid;
        }
        if (!candidates[0].empty()) return candidates[0];
        return precision == FEL_PRECISION_EXACT ? "" : candidates[1];
    }
    // Rendu sur le périphérique Vulkan de FFmpeg (décodage Vulkan Video possible) ;
    // à défaut, renderer libplacebo autonome et décodage CPU, comme en 0.1.0.
    void prepare_gpu() {
        gpu_attempted = true;
        const int w = stream->codecpar->width, h = stream->codecpar->height;
        const std::string uuid = device == "auto" ? automatic_uuid() : device;
        // Aucun GPU éligible pour cette précision : rendu CPU (toujours en double).
        if (device == "auto" && precision != FEL_PRECISION_FAST && uuid.empty()) return;
        if (uuid.size() == 32) {
            const std::string dashed = uuid.substr(0, 8) + "-" + uuid.substr(8, 4) + "-" + uuid.substr(12, 4) + "-" +
                                       uuid.substr(16, 4) + "-" + uuid.substr(20);
            if (av_hwdevice_ctx_create(&hw_device, AV_HWDEVICE_TYPE_VULKAN, dashed.c_str(), nullptr, 0) >= 0) {
                gpu_prototype = fel_gpu_import(hw_device->data, w, h);
                if (!gpu_prototype) av_buffer_unref(&hw_device);
            }
        }
        if (!gpu_prototype) gpu_prototype = fel_gpu_create(w, h, uuid.size() == 32 ? uuid.c_str() : device.c_str());
        if (!gpu_prototype && device != "auto") throw Failure("Initialisation du GPU FEL sélectionné impossible");
        if (gpu_prototype) {
            char name[256];
            fel_gpu_device_name(gpu_prototype, name, sizeof(name));
            if (!fel_gpu_set_precision(gpu_prototype, precision)) {
                // Précision demandée impossible (double natif absent, image non stockable ou
                // auto-test exact refusé) : rendu CPU certifié.
                fel_gpu_destroy(gpu_prototype);
                gpu_prototype = nullptr;
                av_buffer_unref(&hw_device);
                backend = std::string("cpu (") + name + " : précision " + precision_label(precision) + " indisponible)";
                return;
            }
            backend = std::string("vulkan: ") + name + ", " + precision_label(precision);
            // Mode exact : variante retenue par l'auto-test (lot A2').
            if (precision == FEL_PRECISION_EXACT)
                backend += fel_gpu_fixed_variant(gpu_prototype) ? ", virgule fixe" : ", double";
            // Lot P : contextes supplémentaires sur le même périphérique (render_workers, 2 par défaut),
            // précision et auto-test hérités ; un échec réduit simplement le parallélisme.
            for (int i = 1; i < render_workers; ++i) {
                void *extra = hw_device ? fel_gpu_import(hw_device->data, w, h)
                                        : fel_gpu_create(w, h, uuid.size() == 32 ? uuid.c_str() : device.c_str());
                if (!extra || !fel_gpu_inherit(extra, gpu_prototype)) { fel_gpu_destroy(extra); break; }
                gpu_extra.push_back(extra);
            }
            if (!gpu_extra.empty()) backend += ", " + std::to_string(gpu_extra.size() + 1) + " images en vol";
        }
    }
#endif
    // Contrôle du décodage Vulkan Video sur les premières images (même départ) : deux décodages
    // Vulkan indépendants comparés au logiciel, codes BL/EL et RPU. Un décodeur instable (ex.
    // Pascal sous Windows) diverge d'un passage à l'autre : décodage CPU pour tout le job.
    bool verify_hw_decode() {
        constexpr int needed = 48;  // environ deux GOP UHD
        AVFormatContext *probe = avformat_alloc_context();
        if (!probe) throw std::bad_alloc();
        probe->interrupt_callback = {interrupt, this};
        struct Close { AVFormatContext *&p; ~Close() { avformat_close_input(&p); } } close{probe};
        if (avformat_open_input(&probe, source.c_str(), nullptr, nullptr) < 0) return false;
        for (unsigned i = 0; i < probe->nb_streams; ++i)
            if (static_cast<int>(i) != stream_index) probe->streams[i]->discard = AVDISCARD_ALL;
        if (avformat_find_stream_info(probe, nullptr) < 0) return false;
        AVStream *track = probe->streams[stream_index];
        if (target != AV_NOPTS_VALUE) av_seek_frame(probe, stream_index, target, AVSEEK_FLAG_BACKWARD);
        // Par couche : logiciel, Vulkan, Vulkan (second décodage indépendant). Mesure de la BL :
        // couche de base seule (flux sans EL : P8, HDR10).
        const int layer_count = layer != LAYER_RECONSTRUCTED ? 1 : 2;
        std::array<Layer, 6> layers;
        int compared[2] = {0, 0};
        bool same = true, available = true;
        try {
            for (int layer = 0; layer < layer_count; ++layer) {
                const char *mode = layer ? "el" : "bl_rpu";
                layers[3*layer].init(track, mode, std::max(1, threads/3));
                layers[3*layer + 1].init(track, mode, 1, hw_device);
                layers[3*layer + 2].init(track, mode, 1, hw_device);
            }
            auto p = packet();
            auto drain = [&] {
                for (int layer = 0; layer < layer_count && same; ++layer) {
                    auto &sw = layers[3*layer], &first = layers[3*layer + 1], &second = layers[3*layer + 2];
                    while (same && !sw.ready.empty() && !first.ready.empty() && !second.ready.empty()) {
                        same = same_frame(*sw.ready.front(), *first.ready.front(), layer == 0) &&
                               same_frame(*sw.ready.front(), *second.ready.front(), layer == 0);
                        sw.ready.pop_front(); first.ready.pop_front(); second.ready.pop_front();
                        ++compared[layer];
                    }
                }
            };
            bool eof = false;
            while (same && (compared[0] < needed || (layer_count > 1 && compared[1] < needed)) && !eof) {
                check_cancelled();
                const int ret = av_read_frame(probe, p.get());
                if (ret == AVERROR_EOF) {
                    for (int i = 0; i < 3*layer_count; ++i) layers[i].send(nullptr);
                    eof = true;
                } else if (ret < 0) {
                    return false;
                } else if (p->stream_index == stream_index) {
                    for (int i = 0; i < 3*layer_count; ++i) layers[i].send(p.get());
                }
                av_packet_unref(p.get());
                drain();
            }
        } catch (const Failure &failure) {
            if (failure.status == MVO_FEL_CANCELLED) throw;
            // Profil ou extension non pris en charge par le pilote : aucune image comparée.
            same = false;
            available = compared[0] > 0;
        }
        const bool ok = same && compared[0] > 0 && (layer_count == 1 || compared[0] == compared[1]);
        decode_note = ok ? "décodage Vulkan Video vérifié (2 × " + std::to_string(compared[0]) + " images)"
                    : available && compared[0] > 0 ? "décodage CPU (Vulkan Video différent du logiciel)"
                    : "décodage CPU (Vulkan Video indisponible)";
        return ok;
    }
    // Codes 10 bits identiques (P010 en poids fort face à yuv420p10le) et, pour le BL, même RPU.
    static bool same_frame(const AVFrame &sw, const AVFrame &hw, bool base) {
        if (sw.format != AV_PIX_FMT_YUV420P10LE || hw.format != AV_PIX_FMT_VULKAN ||
            sw.width != hw.width || sw.height != hw.height || sw.pts != hw.pts) return false;
        if (base) {
            const auto *a = av_frame_get_side_data(&sw, AV_FRAME_DATA_DOVI_RPU_BUFFER);
            const auto *b = av_frame_get_side_data(&hw, AV_FRAME_DATA_DOVI_RPU_BUFFER);
            if (!a || !b || a->size != b->size || std::memcmp(a->data, b->data, a->size) ||
                !av_frame_get_side_data(&hw, AV_FRAME_DATA_DOVI_METADATA)) return false;
        }
        Frame host = frame();
        if (av_hwframe_transfer_data(host.get(), &hw, 0) < 0 || host->format != AV_PIX_FMT_P010LE) return false;
        for (int y = 0; y < sw.height; ++y) {
            const auto *a = reinterpret_cast<const uint16_t *>(sw.data[0] + y*sw.linesize[0]);
            const auto *b = reinterpret_cast<const uint16_t *>(host->data[0] + y*host->linesize[0]);
            for (int x = 0; x < sw.width; ++x) if (b[x] != a[x] << 6) return false;
        }
        for (int y = 0; y < (sw.height + 1)/2; ++y) {
            const auto *u = reinterpret_cast<const uint16_t *>(sw.data[1] + y*sw.linesize[1]);
            const auto *v = reinterpret_cast<const uint16_t *>(sw.data[2] + y*sw.linesize[2]);
            const auto *b = reinterpret_cast<const uint16_t *>(host->data[1] + y*host->linesize[1]);
            for (int x = 0; x < (sw.width + 1)/2; ++x)
                if (b[2*x] != u[x] << 6 || b[2*x + 1] != v[x] << 6) return false;
        }
        return true;
    }
    void seek() {
        if (raw) {
            // Flux brut sans index : décodage depuis le début, images antérieures ignorées.
            skip = av_rescale_q_rnd(start_us, AV_TIME_BASE_Q, av_inv_q(fps),
                static_cast<AVRounding>(AV_ROUND_UP | AV_ROUND_PASS_MINMAX));
            return;
        }
        const int64_t origin = input->start_time != AV_NOPTS_VALUE ? input->start_time : 0;
        target = av_rescale_q(origin + start_us, AV_TIME_BASE_Q, stream->time_base);
        // Image clé antérieure ; un échec garde la lecture depuis le début (même résultat).
        av_seek_frame(input, stream_index, target, AVSEEK_FLAG_BACKWARD);
    }
    // Images antérieures à la cible, dont les RASL/RADL d'un GOP ouvert : leurs PTS précèdent
    // l'image clé atteinte. Écartées sur la BL, puis autant d'images sur l'EL (même rang de
    // présentation : les PTS de l'EL ne sont pas fiables si sa structure de GOP diffère).
    int64_t leading = 0, el_leading = 0;
    bool leading_done = false;
    void drop_leading() {
        while (!leading_done && !bl.ready.empty()) {
            const auto pts = bl.ready.front()->pts;
            if (target == AV_NOPTS_VALUE || (pts != AV_NOPTS_VALUE && pts >= target)) { leading_done = true; break; }
            bl.ready.pop_front();
            ++leading;
        }
        while (el_leading < leading && !el.ready.empty()) { el.ready.pop_front(); ++el_leading; }
    }
    int64_t output_pts(const AVFrame &base) const {
        if (raw) return emitted;
        return target == AV_NOPTS_VALUE ? base.pts : base.pts - target;
    }
    AVPixelFormat pixel_format() const {
        switch (output_format.format) {
        case FEL_OUT_YUV420P16: return AV_PIX_FMT_YUV420P16LE;
        case FEL_OUT_YUV420P10: return AV_PIX_FMT_YUV420P10LE;
        case FEL_OUT_YUV420P8: return AV_PIX_FMT_YUV420P;
        default: return AV_PIX_FMT_GBRP16LE;
        }
    }
    void open_output(int w, int h) {
        check(avformat_alloc_output_context2(&output, nullptr, "nut", nullptr), "Création NUT");
        auto *buffer = static_cast<uint8_t *>(av_malloc(65536));
        if (!buffer) throw std::bad_alloc();
        io = avio_alloc_context(buffer, 65536, 1, this, nullptr, write_packet, nullptr);
        if (!io) { av_free(buffer); throw std::bad_alloc(); }
        io->seekable = 0;
        // Charge utile d'une image écrite d'un bloc : une écriture par image, pas 760.
        io->direct = 1;
        output->pb = io;
        output->flags |= AVFMT_FLAG_CUSTOM_IO;
        out_stream = avformat_new_stream(output, nullptr);
        if (!out_stream) throw std::bad_alloc();
        out_stream->time_base = raw ? av_inv_q(fps) : stream->time_base;
        out_stream->avg_frame_rate = fps;
        auto *p = out_stream->codecpar;
        const bool yuv = output_format.format != FEL_OUT_GBRP16;
        p->codec_type = AVMEDIA_TYPE_VIDEO;
        p->codec_id = AV_CODEC_ID_RAWVIDEO;
        p->format = pixel_format();
        p->codec_tag = avcodec_pix_fmt_to_codec_tag(pixel_format());
        p->width = w; p->height = h;
        p->color_range = yuv ? AVCOL_RANGE_MPEG : AVCOL_RANGE_JPEG;
        p->color_primaries = AVCOL_PRI_BT2020;
        p->color_trc = AVCOL_TRC_SMPTE2084;
        p->color_space = yuv ? AVCOL_SPC_BT2020_NCL : AVCOL_SPC_RGB;
        if (yuv) p->chroma_location = static_cast<AVChromaLocation>(output_format.chroma);
        // NUT conserve ces tags ; le consommateur pose aussi setparams sur les images.
        av_dict_set(&out_stream->metadata, "color_primaries", "bt2020", 0);
        av_dict_set(&out_stream->metadata, "color_transfer", "smpte2084", 0);
        av_dict_set(&out_stream->metadata, "color_space", yuv ? "bt2020nc" : "gbr", 0);
        av_dict_set(&out_stream->metadata, "color_range", yuv ? "tv" : "pc", 0);
        check(avformat_write_header(output, nullptr), "En-tête NUT");
    }
    std::string device = "auto", backend = "cpu";
    // Précision du rendu GPU (fel_precision) ; défaut rapide pour les appelants antérieurs.
    int precision = FEL_PRECISION_FAST;
    static const char *precision_label(int value) {
        return value == FEL_PRECISION_EXACT ? "précision exacte (certifiée)"
             : value == FEL_PRECISION_STANDARD ? "précision standard (hybride borné)" : "précision rapide (float32)";
    }
    using Emit = std::function<bool(std::unique_ptr<Job>)>;
    // Image BL sans RPU propre (AU sans RPU ni EL, images finales de certains disques) : courbe
    // persistante de FFmpeg (dernier RPU reçu), résiduel désactivé, EL précédent comme plan
    // neutre (non lu). Aucune EL de ce rang n'est consommée.
    Frame last_el;
    std::atomic<long long> bl_only_frames{0};
    static bool own_rpu(const AVFrame &f) { return av_frame_get_side_data(&f, AV_FRAME_DATA_DOVI_RPU_BUFFER); }
    static void disable_residual(AVFrame &base) {
        const auto *sd = av_frame_get_side_data(&base, AV_FRAME_DATA_DOVI_METADATA);
        if (!sd) throw Failure("RPU absent de l'image BL");
        AVBufferRef *copy = av_buffer_alloc(sd->size);
        if (!copy) throw std::bad_alloc();
        std::memcpy(copy->data, sd->data, sd->size);
        av_dovi_get_header(reinterpret_cast<AVDOVIMetadata *>(copy->data))->disable_residual_flag = 1;
        av_frame_remove_side_data(&base, AV_FRAME_DATA_DOVI_METADATA);
        if (!av_frame_new_side_data_from_buf(&base, AV_FRAME_DATA_DOVI_METADATA, copy)) {
            av_buffer_unref(&copy);
            throw std::bad_alloc();
        }
    }
    // Apparie BL/EL/RPU par rang de présentation (n-ième image BL avec la n-ième image EL), quelle
    // que soit la structure de GOP de chaque couche ; false si emit demande l'arrêt. Une EL manquante
    // ou en trop décale les rangs suivants : couche orpheline en fin de flux, refusée.
    bool pair(const Emit &emit) {
        if (layer != LAYER_RECONSTRUCTED) return pair_base(emit);
        drop_leading();
        if (!leading_done || el_leading < leading) return true;
        while (!bl.ready.empty()) {
            check_cancelled();
            const bool base_only = !own_rpu(*bl.ready.front());
            if (!base_only && el.ready.empty()) break;  // EL de même rang encore en décodage
            auto base = std::move(bl.ready.front()); bl.ready.pop_front();
            Frame enhancement;
            if (base_only) {
                if (!last_el) throw Failure("Image BL sans RPU ni EL avant la première image FEL");
                enhancement.reset(av_frame_clone(last_el.get()));
                if (!enhancement) throw std::bad_alloc();
                disable_residual(*base);
            } else {
                enhancement = std::move(el.ready.front()); el.ready.pop_front();
                last_el.reset(av_frame_clone(enhancement.get()));
                if (!last_el) throw std::bad_alloc();
            }
            if (!raw && base->pts == AV_NOPTS_VALUE) throw Failure("Horodatage BL absent");
            const auto *sd = av_frame_get_side_data(base.get(), AV_FRAME_DATA_DOVI_METADATA);
            if (!sd) throw Failure("RPU absent de l'image BL");
            if (!base_only) {
                const auto *original = av_frame_get_side_data(base.get(), AV_FRAME_DATA_DOVI_RPU_BUFFER);
                if (!original) throw Failure("RPU original absent de l'image BL");
                const int kind = mvo_fel_classify(original->data, original->size);
                if (kind != 1 && kind != 3) throw Failure("RPU original invalide ou sans FEL pendant le décodage");
            }
            const auto *metadata = reinterpret_cast<const AVDOVIMetadata *>(sd->data);
            muxiveo::fel::Mapping mapping(*metadata);
            if (!base_only && !mapping.residual) throw Failure("Résiduel désactivé pendant la reconstruction FEL");
            if (skipped < skip) { ++skipped; continue; }
            if (base_only) ++bl_only_frames;
            // Changement de plan : RPU propre seulement (une courbe héritée ne commence pas de plan).
            const bool scene_start = !base_only && av_dovi_get_color(metadata)->scene_refresh_flag;
            if (!emit_job(std::move(base), std::move(enhancement), metadata, std::move(mapping), scene_start,
                          active_area(metadata), emit))
                return false;
        }
        return true;
    }
    // Zone active (L5) : bandes noires exclues des statistiques de l'image.
    static std::array<int, 4> active_area(const AVDOVIMetadata *metadata) {
        if (const AVDOVIDmData *l5 = metadata ? av_dovi_find_level(metadata, 5) : nullptr)
            return {l5->l5.left_offset, l5->l5.top_offset, l5->l5.right_offset, l5->l5.bottom_offset};
        return {};
    }
    // Horodatage, contrôles et transmission d'une image appariée ; false si emit demande l'arrêt.
    bool emit_job(Frame base, Frame enhancement, const AVDOVIMetadata *metadata, muxiveo::fel::Mapping mapping,
                  bool scene_start, std::array<int, 4> active, const Emit &emit) {
        if (base->width <= 0 || base->height <= 0 || base->width > 8192 || base->height > 8192)
            throw Failure("Dimensions source invalides");
        const auto timebase = raw ? av_inv_q(fps) : stream->time_base;
        auto pts = output_pts(*base);
        if (rebase_first) {
            if (first_pts == AV_NOPTS_VALUE) first_pts = pts;
            pts -= first_pts;
        }
        if (last_pts != AV_NOPTS_VALUE && pts <= last_pts)
            throw Failure("Horodatages de présentation non croissants");
        last_pts = pts;
        const int64_t duration = raw ? 1 : base->duration;
        ++emitted;
        return emit(std::unique_ptr<Job>(new Job{std::move(base), std::move(enhancement), metadata,
                                                 std::move(mapping), pts, duration, timebase, scene_start, active}));
    }
    // RPU identité : courbes polynomiales d'ordre 1 (pente 1), sans résiduel, matrices Y'CbCr → RGB
    // BT.2020 plage limitée et RGB → LMS des RPU de profil 8.1 : conversion HDR10 standard de la BL,
    // calculée par les mêmes chemins que la reconstruction (CPU certifié, GPU, statistiques).
    AVBufferRef *identity_metadata() const {
        const auto *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(stream->codecpar->format));
        const int depth = desc ? desc->comp[0].depth : 10;
        if (depth < 8 || depth > 16) throw Failure("Profondeur de la couche de base non prise en charge");
        size_t size = 0;
        AVDOVIMetadata *metadata = av_dovi_metadata_alloc(&size);
        if (!metadata) throw std::bad_alloc();
        AVBufferRef *buffer = av_buffer_create(reinterpret_cast<uint8_t *>(metadata), size, av_buffer_default_free,
                                               nullptr, 0);
        if (!buffer) { av_free(metadata); throw std::bad_alloc(); }
        auto *h = av_dovi_get_header(metadata);
        auto *m = av_dovi_get_mapping(metadata);
        auto *c = av_dovi_get_color(metadata);
        h->rpu_type = 2; h->rpu_format = 18; h->vdr_rpu_profile = 1; h->vdr_rpu_normalized_idc = 1;
        h->coef_data_type = 0; h->coef_log2_denom = 23;
        h->bl_bit_depth = h->el_bit_depth = depth; h->vdr_bit_depth = 12;
        h->disable_residual_flag = 1;
        m->nlq_method_idc = AV_DOVI_NLQ_NONE;
        m->num_x_partitions = m->num_y_partitions = 1;
        for (int i = 0; i < 3; ++i) {
            auto &curve = m->curves[i];
            curve.num_pivots = 2; curve.pivots[0] = 0; curve.pivots[1] = (1 << depth) - 1;
            curve.mapping_idc[0] = AV_DOVI_MAPPING_POLYNOMIAL;
            curve.poly_order[0] = 1; curve.poly_coef[0][1] = int64_t(1) << 23;
        }
        static constexpr int ycc[9] = {9574, 0, 13802, 9574, -1540, -5348, 9574, 17610, 0};
        static constexpr int lms[9] = {7222, 8771, 390, 2654, 12430, 1300, 0, 422, 15962};
        static constexpr int offsets[3] = {16777216, 134217728, 134217728};
        for (int i = 0; i < 9; ++i) {
            c->ycc_to_rgb_matrix[i] = av_make_q(ycc[i], 1 << 13);
            c->rgb_to_lms_matrix[i] = av_make_q(lms[i], 1 << 14);
        }
        for (int i = 0; i < 3; ++i) c->ycc_to_rgb_offset[i] = av_make_q(offsets[i], 1 << 28);
        c->signal_eotf = 65535; c->signal_bit_depth = 12; c->signal_full_range_flag = 1;
        c->source_min_pq = 62; c->source_max_pq = 3696; c->source_diagonal = 42;
        return buffer;
    }
    // Couches sans EL : base (RPU identité ; le RPU éventuel ne sert qu'à la zone active L5 et aux
    // changements de plan) ou mapped (courbes du RPU de l'image, résiduel désactivé). L'EL n'est ni
    // décodée ni lue (plan neutre : l'image de base elle-même).
    bool pair_base(const Emit &emit) {
        drop_leading();
        if (!leading_done) return true;
        while (!bl.ready.empty()) {
            check_cancelled();
            auto base = std::move(bl.ready.front()); bl.ready.pop_front();
            if (!raw && base->pts == AV_NOPTS_VALUE) throw Failure("Horodatage absent");
            bool scene_start = false;
            std::array<int, 4> active{};
            if (const auto *sd = av_frame_get_side_data(base.get(), AV_FRAME_DATA_DOVI_METADATA)) {
                const auto *source = reinterpret_cast<const AVDOVIMetadata *>(sd->data);
                scene_start = own_rpu(*base) && av_dovi_get_color(source)->scene_refresh_flag;
                active = active_area(source);
            } else if (layer == LAYER_MAPPED) {
                throw Failure("RPU absent : couche « mapped » réservée aux sources Dolby Vision");
            }
            if (layer == LAYER_MAPPED) {
                disable_residual(*base);  // copie du RPU de l'image, résiduel désactivé
            } else {
                av_frame_remove_side_data(base.get(), AV_FRAME_DATA_DOVI_METADATA);
                AVBufferRef *shared = av_buffer_ref(identity);
                if (!shared) throw std::bad_alloc();
                if (!av_frame_new_side_data_from_buf(base.get(), AV_FRAME_DATA_DOVI_METADATA, shared)) {
                    av_buffer_unref(&shared);
                    throw std::bad_alloc();
                }
            }
            const auto *sd = av_frame_get_side_data(base.get(), AV_FRAME_DATA_DOVI_METADATA);
            const auto *metadata = reinterpret_cast<const AVDOVIMetadata *>(sd->data);
            if (skipped < skip) { ++skipped; continue; }
            Frame enhancement(av_frame_clone(base.get()));
            if (!enhancement) throw std::bad_alloc();
            muxiveo::fel::Mapping mapping(*metadata);
            if (!emit_job(std::move(base), std::move(enhancement), metadata, std::move(mapping), scene_start, active,
                          emit))
                return false;
        }
        return true;
    }
    // Lecture, décodage BL/EL et appariement jusqu'à la fin du flux.
    void decode_all(const Emit &emit) {
        auto p = packet();
        for (;;) {
            check_cancelled();
            int ret;
            { Timer timer(timing[1]); ret = av_read_frame(input, p.get()); }
            if (ret == AVERROR_EOF) break;
            check(ret, "Lecture source", MVO_FEL_INPUT_ERROR);
            if (p->stream_index == stream_index) {
                { Timer timer(timing[2]); bl.send(p.get()); }
                if (layer == LAYER_RECONSTRUCTED) { Timer timer(timing[3]); el.send(p.get()); }
                if (!pair(emit)) return;
            }
            av_packet_unref(p.get());
        }
        { Timer timer(timing[2]); bl.send(nullptr); }
        if (layer == LAYER_RECONSTRUCTED) { Timer timer(timing[3]); el.send(nullptr); }
        if (!pair(emit)) return;
        if (!leading_done) el.ready.clear();  // cible au-delà de la dernière image
        if (!bl.ready.empty() || !el.ready.empty())
            throw Failure("Fin de flux BL/EL désalignée : nombre d'images EL différent des images BL avec RPU");
    }
    // Fenêtre mesurée (coordonnées de la source) : fenêtre encodée (recadrage des sorties Y'CbCr)
    // limitée à la zone active L5 ; fenêtre encodée entière si l'intersection est vide.
    std::array<int, 4> statistics_window(const Job &job, bool yuv) const {
        const int x0 = yuv ? output_format.crop_x : 0, y0 = yuv ? output_format.crop_y : 0;
        const int x1 = x0 + output_format.width, y1 = y0 + output_format.height;
        const int ax0 = std::max(x0, job.active[0]), ay0 = std::max(y0, job.active[1]);
        const int ax1 = std::min(x1, width - job.active[2]), ay1 = std::min(y1, height - job.active[3]);
        if (ax1 <= ax0 || ay1 <= ay0) return {x0, y0, x1 - x0, y1 - y0};
        return {ax0, ay0, ax1 - ax0, ay1 - ay0};
    }
    // Statistiques CPU (frame_stats.h) sur la fenêtre d'une image GBRP16 pleine taille (plans G, B, R) :
    // lignes par blocs parallèles, histogrammes fusionnés sous verrou ; même résultat que le GPU.
    void cpu_statistics(const AVFrame &image, const std::array<int, 4> &window, fel_frame_stats &stats) {
        const int x0 = window[0], y0 = window[1], w = window[2], h = window[3];
        constexpr int block = 8;
        std::vector<uint32_t> row_words(size_t(FEL_STATS_ROW_WORDS)*h);
        fel_stats_reset(&stats);
        std::mutex merge;
        rows->run((h + block - 1) / block, [&](int index) {
            check_cancelled();
            std::array<uint32_t, FEL_STATS_BINS> hm{}, hy{};
            for (int j = index*block; j < std::min(h, (index + 1)*block); ++j) {
                const auto line = [&](int plane) {
                    return reinterpret_cast<const uint16_t *>(image.data[plane] + (y0 + j)*image.linesize[plane]) + x0;
                };
                fel_stats_row(line(2), line(0), line(1), w, hm.data(), hy.data(), row_words.data() + size_t(FEL_STATS_ROW_WORDS)*j);
            }
            std::lock_guard lock(merge);
            for (int k = 0; k < FEL_STATS_BINS; ++k) { stats.hist_maxrgb[k] += hm[k]; stats.hist_y[k] += hy[k]; }
        });
        fel_stats_add_rows(&stats, row_words.data(), h, w);
    }
    // Ligne JSON des statistiques d'une image, puis histogramme du plan à sa fin (écriture ordonnée).
    void write_statistics(const Rendered &item) {
        if (!stats_io) {
            check(avio_open(&stats_io, stats_path.c_str(), AVIO_FLAG_WRITE), "Fichier de statistiques");
            std::ostringstream head;
            head.imbue(std::locale::classic());
            head << "{\"format\":\"mvo-fel-frame-statistics\",\"version\":1,\"bins\":" << FEL_STATS_BINS
                 << ",\"width\":" << item.width << ",\"height\":" << item.height << "}\n";
            avio_write(stats_io, reinterpret_cast<const unsigned char *>(head.str().data()), int(head.str().size()));
        }
        const fel_frame_stats &s = *item.stats;
        if (item.scene_start || scene_first < 0) flush_scene();
        if (scene_first < 0) scene_first = stats_frame;
        ++scene_frames;
        for (int k = 0; k < FEL_STATS_BINS; ++k) scene_hist[k] += s.hist_maxrgb[k];
        const float *eotf = fel_stats_eotf(), *edges = fel_stats_edges();
        // Percentile 99,99 % de Y : borne haute de la case atteinte (valeur sous laquelle tombe 99,99 %).
        const uint64_t target = s.pixels - s.pixels/10000;
        uint64_t total = 0;
        int bin = FEL_STATS_BINS - 1;
        for (int k = 0; k < FEL_STATS_BINS; ++k) {
            total += s.hist_y[k];
            if (total >= target) { bin = k; break; }
        }
        // Percentiles de maxRGB de l'image (1, 25, 50, 75, 90, 95, 99,98 %) : centre de la case atteinte.
        // Agrégés par plan dans Muxiveo (moyenne ; maximum pour 99,98 %), comme l'analyseur de référence.
        static constexpr double quantiles[] = {0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.9998};
        std::array<float, 7> percentiles{};
        for (size_t q = 0; q < percentiles.size(); ++q) {
            const double wanted = quantiles[q]*double(s.pixels);
            uint64_t cumulative = 0;
            int index = FEL_STATS_BINS - 1;
            for (int k = 0; k < FEL_STATS_BINS; ++k) {
                cumulative += s.hist_maxrgb[k];
                if (double(cumulative) >= wanted) { index = k; break; }
            }
            percentiles[q] = eotf[index*64 + 32];
        }
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(9) << "{\"frame\":" << stats_frame << ",\"scene_start\":" << (item.scene_start ? "true" : "false")
             << ",\"max_nits\":[" << eotf[s.max_code[0]] << ',' << eotf[s.max_code[1]] << ',' << eotf[s.max_code[2]] << ']'
             << ",\"avg_nits\":" << (s.pixels ? s.sum_maxrgb/double(s.pixels) : 0.0)
             << ",\"y9999_nits\":" << edges[bin + 1]
             << ",\"y100\":" << (s.pixels ? double(s.y_le_100)/double(s.pixels) : 0.0)
             << ",\"maxrgb_percentiles\":[";
        for (size_t q = 0; q < percentiles.size(); ++q) line << (q ? "," : "") << percentiles[q];
        line << "],\"window\":[" << item.window[0] << ',' << item.window[1] << ',' << item.window[2] << ','
             << item.window[3] << "]}\n";
        avio_write(stats_io, reinterpret_cast<const unsigned char *>(line.str().data()), int(line.str().size()));
        ++stats_frame;
    }
    // Histogramme de maxRGB du plan terminé (cases non vides : [case, nombre]).
    void flush_scene() {
        if (scene_first < 0 || !scene_frames) return;
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << "{\"scene_first\":" << scene_first << ",\"frames\":" << scene_frames << ",\"maxrgb_hist\":[";
        bool first = true;
        for (int k = 0; k < FEL_STATS_BINS; ++k)
            if (scene_hist[k]) { line << (first ? "" : ",") << '[' << k << ',' << scene_hist[k] << ']'; first = false; }
        line << "]}\n";
        avio_write(stats_io, reinterpret_cast<const unsigned char *>(line.str().data()), int(line.str().size()));
        scene_hist.fill(0);
        scene_first = -1;
        scene_frames = 0;
    }
    // Copie logicielle yuv420p10le d'une image décodée par Vulkan Video (P010), pour le CPU certifié.
    static Frame software_copy(const AVFrame &hw) {
        Frame host = frame(), out = frame();
        check(av_hwframe_transfer_data(host.get(), &hw, 0), "Copie de l'image GPU");
        if (host->format != AV_PIX_FMT_P010LE) throw Failure("Image GPU non P010");
        out->format = AV_PIX_FMT_YUV420P10LE; out->width = hw.width; out->height = hw.height;
        check(av_frame_get_buffer(out.get(), 0), "Image logicielle");
        check(av_frame_copy_props(out.get(), &hw), "Propriétés de l'image");
        for (int y = 0; y < hw.height; ++y) {
            const auto *a = reinterpret_cast<const uint16_t *>(host->data[0] + y*host->linesize[0]);
            auto *b = reinterpret_cast<uint16_t *>(out->data[0] + y*out->linesize[0]);
            for (int x = 0; x < hw.width; ++x) b[x] = a[x] >> 6;
        }
        for (int y = 0; y < (hw.height + 1)/2; ++y) {
            const auto *a = reinterpret_cast<const uint16_t *>(host->data[1] + y*host->linesize[1]);
            auto *u = reinterpret_cast<uint16_t *>(out->data[1] + y*out->linesize[1]);
            auto *v = reinterpret_cast<uint16_t *>(out->data[2] + y*out->linesize[2]);
            for (int x = 0; x < (hw.width + 1)/2; ++x) { u[x] = a[2*x] >> 6; v[x] = a[2*x + 1] >> 6; }
        }
        return out;
    }
    // Reconstruction d'une image dans un paquet du pool (thread de rendu uniquement).
    // Plusieurs fils de rendu en mode GPU (lot P) : initialisation sous verrou, cadre et contexte par fil.
    std::unique_ptr<Rendered> render(Job &job, Worker &w) {
        check_cancelled();
        const AVFrame &base = *job.base;
        const bool yuv = output_format.format != FEL_OUT_GBRP16;
        if (!yuv && (crop[0] | crop[1] | crop[2] | crop[3])) throw Failure("Recadrage réservé aux sorties Y'CbCr");
        std::unique_lock init(render_mutex);
        if (!width) {
            width = base.width; height = base.height;
            if (yuv && (width % 2 || height % 2)) throw Failure("Dimensions impaires incompatibles avec le 4:2:0");
            output_format.crop_x = crop[0]; output_format.crop_y = crop[1];
            output_format.width = width - crop[0] - crop[2]; output_format.height = height - crop[1] - crop[3];
            if (output_format.width <= 0 || output_format.height <= 0) throw Failure("Recadrage plus grand que l'image");
            packet_size = av_image_get_buffer_size(pixel_format(), output_format.width, output_format.height, 1);
            check(packet_size, "Taille image");
            // Tampons réutilisés : ni allocation ni défauts de page de 49,8 Mo par image UHD.
            pool = av_buffer_pool_init(static_cast<size_t>(packet_size), nullptr);
            if (!pool) throw std::bad_alloc();
        }
        init.unlock();
        if (base.width != width || base.height != height)
            throw Failure("Changement de dimensions pendant la reconstruction");
        if (!w.frame) {
            w.frame = frame();
            w.frame->format = pixel_format(); w.frame->width = output_format.width; w.frame->height = output_format.height;
        }
        AVFrame *rgb = w.frame.get();
        auto out = std::unique_ptr<Rendered>(new Rendered{packet(), output_format.width, output_format.height,
                                                          job.timebase, nullptr, job.scene_start});
        if (!stats_path.empty()) out->stats = std::make_unique<fel_frame_stats>();
        out->packet->buf = av_buffer_pool_get(pool);
        if (!out->packet->buf) throw std::bad_alloc();
        out->packet->data = out->packet->buf->data;
        out->packet->size = packet_size;
        check(av_image_fill_arrays(rgb->data, rgb->linesize, out->packet->data,
            pixel_format(), output_format.width, output_format.height, 1), "Plans du paquet");
        {
            Timer pixels(w.pixels);
            bool on_cpu = true;
            std::unique_lock<std::mutex> cpu_lock;
            Frame held_base, held_el;  // images GPU d'origine : job.metadata pointe dans leurs données
#ifdef MVO_FEL_VULKAN_PROTOTYPE
            if (w.gpu) {
                const bool ok = yuv
                    ? fel_gpu_render_yuv(w.gpu, job.metadata, job.base.get(), job.enhancement.get(),
                                         &output_format, rgb)
                    : fel_gpu_render_planar(w.gpu, job.metadata, job.base.get(), job.enhancement.get(), rgb);
                if (!ok && !fel_gpu_take_overflow(w.gpu)) throw Failure("Échec de reconstruction du prototype Vulkan");
                on_cpu = !ok;
                // Statistiques sur la cible RGB du GPU (fenêtre encodée ∩ zone active L5).
                if (ok && out->stats) {
                    const auto window = out->window = statistics_window(job, yuv);
                    if (!fel_gpu_frame_statistics(w.gpu, window[0], window[1], window[2], window[3], out->stats.get()))
                        throw Failure("Statistiques de l'image impossibles sur le GPU");
                }
                if (on_cpu) {
                    cpu_lock = std::unique_lock(cpu_mutex);
                    ++cpu_rescue_frames;
                    if (job.base->format == AV_PIX_FMT_VULKAN) {
                        held_base = std::move(job.base); held_el = std::move(job.enhancement);
                        job.base = software_copy(*held_base);
                        job.enhancement = software_copy(*held_el);
                    }
                }
            }
#endif
            if (on_cpu) {
                const AVFrame &base = *job.base;  // copie logicielle après un repli GPU
                if (base.format == AV_PIX_FMT_VULKAN) throw Failure("Image Vulkan sans rendu GPU");
                if (!rows) rows = std::make_unique<RowPool>(render_threads);
                // Y'CbCr : reconstruction RGB 16 bits intermédiaire, puis conversion par lignes chroma.
                AVFrame *target = rgb;
                if (yuv) {
                    if (!scratch) {
                        scratch = frame();
                        scratch->format = AV_PIX_FMT_GBRP16LE; scratch->width = width; scratch->height = height;
                        check(av_frame_get_buffer(scratch.get(), 0), "Image RGB intermédiaire");
                    }
                    target = scratch.get();
                }
                constexpr int block = 8;
                // Entiers exacts séparables si la géométrie s'y prête, sinon échantillonneurs en double.
                // MVO_FEL_CPU_REFERENCE : échantillonneurs en double (comparaisons de validation).
                static const bool reference = getenv("MVO_FEL_CPU_REFERENCE") != nullptr;
                if (reference) fast.reset();
                else if (!fast || !fast->bind(base, *job.enhancement, *job.metadata)) {
                    fast = std::make_unique<muxiveo::fel::FastReconstruction>(base, *job.enhancement);
                    if (fast->valid()) fast->bind(base, *job.enhancement, *job.metadata);
                }
                if (fast && fast->valid()) {
                    rows->run((height + block - 1) / block, [&](int index) {
                        check_cancelled();
                        const auto stats = fast->rows(job.mapping, *target, index*block, std::min(height, (index+1)*block));
                        certified_rescued += stats.rescued;
                        certified_undecided += stats.undecided;
                    });
                } else {
                    const muxiveo::fel::FrameSampler bs(base, width, height, false);
                    const muxiveo::fel::FrameSampler es(*job.enhancement, width, height, true);
                    rows->run((height + block - 1) / block, [&](int index) {
                        check_cancelled();
                        muxiveo::fel::reconstruct_rows(bs, es, job.mapping, *target, index*block,
                                                       std::min(height, (index+1)*block));
                    });
                }
                const int chroma_rows = output_format.height/2;
                if (yuv) rows->run((chroma_rows + block - 1) / block, [&](int index) {
                    check_cancelled();
                    muxiveo::fel::convert_rows(*scratch, *rgb, output_format, index*block,
                                               std::min(chroma_rows, (index+1)*block));
                });
                if (out->stats) cpu_statistics(yuv ? *scratch : *rgb, out->window = statistics_window(job, yuv), *out->stats);
            }
        }
        auto &p = *out->packet;
        p.pts = p.dts = job.pts;
        p.duration = job.duration;
        p.stream_index = 0; p.flags |= AV_PKT_FLAG_KEY;
        return out;
    }
    void write_rendered(Rendered &item) {
        if (!measure_only) {
            if (!output) open_output(item.width, item.height);
            Timer nut(timing[5]);
            av_packet_rescale_ts(item.packet.get(), item.timebase, out_stream->time_base);
            check(av_interleaved_write_frame(output, item.packet.get()), "Écriture NUT");
        }
        if (item.stats) write_statistics(item);
        ++frames;
        if (progress) progress(opaque, frames);
    }
    // Trois étages bornés : décodage/appariement, rendu, écriture sur le thread appelant
    // (contrat ABI des callbacks). Une erreur ou l'annulation abandonne les files.
    void run() {
        if (started) throw Failure("Contexte déjà exécuté");
        if (measure_only && stats_path.empty()) throw Failure("Mesure sans fichier de statistiques");
        started = true;
        began = Clock::now();
        { Timer timer(timing[0]); open(); }
        Channel<std::unique_ptr<Job>> jobs(3);
        Channel<std::unique_ptr<Rendered>> rendered(2);
        std::exception_ptr failure;
        auto fail = [&] {
            { std::lock_guard lock(failure_mutex); if (!failure) failure = std::current_exception(); }
            jobs.abort(); rendered.abort();
        };
        std::thread decoder([&] {
            try {
                decode_all([&](std::unique_ptr<Job> job) {
                    if (!jobs.push(std::move(job), cancelled, waits[0])) throw Stopped{};
                    return true;
                });
                jobs.close();
            } catch (const Stopped &) {
            } catch (...) { fail(); }
        });
        // Fils de rendu : un par contexte GPU (lot P), un seul en CPU (lignes déjà parallèles). Les
        // tickets, attribués à la prise du job sous verrou, rétablissent l'ordre avant l'écriture.
        std::vector<Worker> workers(1);
#ifdef MVO_FEL_VULKAN_PROTOTYPE
        workers[0].gpu = gpu_prototype;
        if (gpu_prototype) for (void *extra : gpu_extra) { workers.emplace_back(); workers.back().gpu = extra; }
#endif
        std::mutex pop_mutex, order_mutex;
        std::condition_variable order;
        int64_t next_ticket = 0, next_write = 0;
        std::atomic<int> active{static_cast<int>(workers.size())};
        std::atomic<bool> stopping{false};
        auto render_loop = [&](Worker &w) {
            try {
                for (;;) {
                    std::unique_ptr<Job> job;
                    int64_t ticket;
                    {
                        std::lock_guard lock(pop_mutex);
                        if (!jobs.pop(job, cancelled, w.starved)) break;
                        ticket = next_ticket++;
                    }
                    auto out = render(*job, w);
                    job.reset();
                    {
                        std::unique_lock lock(order_mutex);
                        order.wait(lock, [&] { return next_write == ticket || stopping.load() || cancelled.load(); });
                        if (stopping.load() || cancelled.load()) break;
                    }
                    const bool pushed = rendered.push(std::move(out), cancelled, w.blocked);
                    { std::lock_guard lock(order_mutex); ++next_write; }
                    order.notify_all();
                    if (!pushed) break;
                }
            } catch (...) { stopping = true; order.notify_all(); fail(); }
            if (--active == 0) rendered.close();
        };
        std::vector<std::thread> renderers;
        for (auto &w : workers) renderers.emplace_back(render_loop, std::ref(w));
        try {
            std::unique_ptr<Rendered> item;
            while (rendered.pop(item, cancelled, waits[3])) write_rendered(*item);
        } catch (...) { fail(); }
        stopping = true;
        order.notify_all();
        decoder.join();
        for (auto &t : renderers) t.join();
        for (const auto &w : workers) { timing[4] += w.pixels; waits[1] += w.starved; waits[2] += w.blocked; }
        rows.reset();
        if (failure) std::rethrow_exception(failure);
        check_cancelled();
        if (!frames) throw Failure(layer == LAYER_RECONSTRUCTED ? "Aucune image FEL reconstruite" : "Aucune image produite");
        if (!measure_only) {
            check(av_write_trailer(output), "Fin NUT");
            avio_flush(io); check(io->error, "Vidange NUT");
        }
        if (stats_io) {
            flush_scene();
            avio_flush(stats_io); check(stats_io->error, "Écriture des statistiques");
            avio_closep(&stats_io);
        }
    }
    void stop_prefetch() {
        if (!prefetcher.joinable()) return;
        prefetched->abort();
        prefetcher.join();
    }
    int next() {
        if (!started) {
            if (!stats_path.empty()) throw Failure("Statistiques de l'image indisponibles en transport direct");
            if (measure_only || layer != LAYER_RECONSTRUCTED)
                throw Failure("Mesure et couches base/mapped indisponibles en transport direct");
            started = true; pulling = true; began = Clock::now();
            { Timer timer(timing[0]); open(); }
            // Décodage de l'image suivante pendant le rendu de la précédente par l'appelant.
            prefetched = std::make_unique<Channel<std::unique_ptr<Job>>>(3);
            prefetcher = std::thread([this] {
                try {
                    decode_all([this](std::unique_ptr<Job> job) {
                        if (!prefetched->push(std::move(job), cancelled, waits[0])) throw Stopped{};
                        return true;
                    });
                    prefetched->close();
                } catch (const Stopped &) {
                } catch (...) {
                    { std::lock_guard lock(failure_mutex); prefetch_failure = std::current_exception(); }
                    prefetched->abort();
                }
            });
        }
        if (!pulling) throw Failure("Contexte NUT déjà exécuté");
        std::unique_ptr<Job> job;
        if (prefetched->pop(job, cancelled, waits[3])) {
            if (layers(opaque, job->base.get(), job->enhancement.get(), job->metadata, job->pts,
                       job->duration, job->timebase.num, job->timebase.den))
                throw Failure("Consommateur de couches FEL arrêté", MVO_FEL_OUTPUT_CLOSED);
            ++frames;
            return MVO_FEL_OK;
        }
        if (prefetcher.joinable()) prefetcher.join();
        { std::lock_guard lock(failure_mutex); if (prefetch_failure) std::rethrow_exception(prefetch_failure); }
        check_cancelled();
        if (!frames) throw Failure("Aucune image FEL reconstruite");
        return 5;
    }
};

uint32_t mvo_fel_abi_version() { return MVO_FEL_ABI; }
const char *mvo_fel_capabilities() {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    return "{\"abi\":1,\"version\":\"" MVO_FEL_VERSION "\",\"transport\":\"nut\",\"pixel_format\":\"gbrp16le\",\"cpu\":true,\"backend\":\"vulkan-prototype\",\"statistics\":1,\"device_selection\":1,\"seek\":1,\"fd_output\":1,\"timestamps\":[\"source\",\"zero\"],\"crop\":1,\"output_formats\":[\"gbrp16le\",\"yuv420p16le\",\"yuv420p10le\",\"yuv420p\"],\"dithers\":[\"none\",\"ordered\",\"blue\"],\"direct_formats\":[\"p010\"],\"precision\":[\"exact\",\"standard\",\"fast\"],\"device_features\":[\"float64\"],\"render_workers\":[1,4],\"frame_statistics\":1,\"measure\":[\"base\",\"mapped\",\"reconstructed\"],\"layers\":[\"base\",\"mapped\",\"reconstructed\"],\"experimental\":true}";
#else
    return "{\"abi\":1,\"version\":\"" MVO_FEL_VERSION "\",\"transport\":\"nut\",\"pixel_format\":\"gbrp16le\",\"cpu\":true,\"statistics\":1,\"device_selection\":1,\"seek\":1,\"fd_output\":1,\"timestamps\":[\"source\",\"zero\"],\"crop\":1,\"output_formats\":[\"gbrp16le\",\"yuv420p16le\",\"yuv420p10le\",\"yuv420p\"],\"dithers\":[\"none\",\"ordered\",\"blue\"],\"precision\":[\"exact\",\"standard\",\"fast\"],\"device_features\":[\"float64\"],\"frame_statistics\":1,\"measure\":[\"base\",\"mapped\",\"reconstructed\"],\"layers\":[\"base\",\"mapped\",\"reconstructed\"]}";
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
    // FFmpeg privé muet (pistes ignorées, zlib absent) : les erreurs passent par mvo_fel_error.
    static const bool quiet = [] { av_log_set_level(AV_LOG_FATAL); return true; }();
    (void)quiet;
    try { return new mvo_fel_context(source, stream, threads, {num, den}); } catch (...) { return nullptr; }
}
size_t mvo_fel_devices(char *buffer, size_t capacity) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    return fel_gpu_devices(buffer, capacity, 0);
#else
    if (buffer && capacity >= 3) std::memcpy(buffer, "[]", 3);
    return 3;
#endif
}
size_t mvo_fel_devices_features(char *buffer, size_t capacity) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    return fel_gpu_devices(buffer, capacity, 1);
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
namespace {
int chroma_from_name(const char *name) {
    const int value = name ? av_chroma_location_from_name(name) : -1;
    return value >= AVCHROMA_LOC_LEFT && value <= AVCHROMA_LOC_BOTTOM ? value : -1;
}
int dither_from_name(const char *name) {
    const std::string value(name ? name : "");
    return value == "none" ? FEL_DITHER_NONE : value == "ordered" ? FEL_DITHER_ORDERED
         : value == "blue" ? FEL_DITHER_BLUE : -1;
}
}
int mvo_fel_set_output(mvo_fel_context *p, const char *format, const char *chroma, const char *dither) {
    if (!p || p->started || !format) return -1;
    const std::string name(format);
    const int value = name == "gbrp16le" ? FEL_OUT_GBRP16 : name == "yuv420p16le" ? FEL_OUT_YUV420P16
                    : name == "yuv420p10le" ? FEL_OUT_YUV420P10 : name == "yuv420p" ? FEL_OUT_YUV420P8 : -1;
    const int location = chroma_from_name(chroma), mode = dither_from_name(dither);
    if (value < 0 || location < 0 || mode < 0) return -1;
    if (p->measure_only && value != FEL_OUT_GBRP16) return -1;
    p->output_format.format = value; p->output_format.chroma = location; p->output_format.dither = mode;
    return 0;
}
int mvo_fel_set_decoder_device(mvo_fel_context *p, const void *device) {
    const auto *reference = static_cast<const AVBufferRef *>(device);
    if (!p || p->started || !reference || !reference->data ||
        reinterpret_cast<const AVHWDeviceContext *>(reference->data)->type != AV_HWDEVICE_TYPE_VULKAN) return -1;
    av_buffer_unref(&p->hw_device);
    p->hw_device = av_buffer_ref(const_cast<AVBufferRef *>(reference));
    return p->hw_device ? 0 : -1;
}
int mvo_fel_set_crop(mvo_fel_context *p, int left, int top, int right, int bottom) {
    if (!p || p->started || left < 0 || top < 0 || right < 0 || bottom < 0 ||
        (left | top | right | bottom) & 1 || std::max({left, top, right, bottom}) > 8192) return -1;
    if (p->measure_only && (left | top | right | bottom)) return -1;
    p->crop = {left, top, right, bottom};
    return 0;
}
int mvo_fel_set_precision(mvo_fel_context *p, const char *mode) {
    if (!p || p->started || !mode) return -1;
    const std::string value(mode);
    if (value != "exact" && value != "standard" && value != "fast") return -1;
    p->precision = value == "exact" ? FEL_PRECISION_EXACT : value == "standard" ? FEL_PRECISION_STANDARD
                                                                                  : FEL_PRECISION_FAST;
    return 0;
}
int mvo_fel_set_frame_statistics(mvo_fel_context *p, const char *path) {
    if (!p || p->started || !path || !*path) return -1;
    fel_stats_eotf();  // tables communes initialisées avant tout fil de rendu
    p->stats_path = path;
    return 0;
}
namespace {
int layer_from_name(const char *name) {
    const std::string value(name ? name : "");
    return value == "base" ? mvo_fel_context::LAYER_BASE : value == "mapped" ? mvo_fel_context::LAYER_MAPPED
         : value == "reconstructed" ? mvo_fel_context::LAYER_RECONSTRUCTED : -1;
}
}
int mvo_fel_set_layer(mvo_fel_context *p, const char *layer) {
    const int value = layer_from_name(layer);
    if (!p || p->started || value < 0) return -1;
    p->layer = value;
    return 0;
}
int mvo_fel_set_measure(mvo_fel_context *p, const char *layer) {
    const int value = layer_from_name(layer);
    if (!p || p->started || value < 0) return -1;
    // Statistiques sur le RGB PQ pleine image : sortie Y'CbCr et recadrage sans objet.
    if (p->output_format.format != FEL_OUT_GBRP16 || (p->crop[0] | p->crop[1] | p->crop[2] | p->crop[3])) return -1;
    p->layer = value;
    p->measure_only = true;
    return 0;
}
int mvo_fel_set_render_workers(mvo_fel_context *p, int count) {
    if (!p || p->started || count < 1 || count > 4) return -1;
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    p->render_workers = count;
#endif
    return 0;
}
int mvo_fel_set_timestamps(mvo_fel_context *p, const char *mode) {
    if (!p || p->started || !mode) return -1;
    const std::string value(mode);
    if (value != "source" && value != "zero") return -1;
    p->rebase_first = value == "zero";
    return 0;
}
int mvo_fel_set_start(mvo_fel_context *p, int64_t start_us) {
    if (!p || p->started || start_us < 0) return -1;
    p->start_us = start_us;
    return 0;
}
const char *mvo_fel_statistics(mvo_fel_context *p) {
    if (!p) return "{}";
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::fixed << std::setprecision(6) << "{\"frames\":" << p->frames;
    const char *names[]={"open_s","read_s","decode_bl_s","decode_el_s","pixels_s","nut_s","pipe_s"};
    for(size_t i=0;i<p->timing.size();++i)text << ",\"" << names[i] << "\":" << p->timing[i];
    // Durée murale du thread producteur : base du débit réel, étages superposés compris.
    const auto end = p->ended.time_since_epoch().count() ? p->ended : Clock::now();
    const double wall = p->began.time_since_epoch().count() ? std::chrono::duration<double>(end-p->began).count() : 0;
    text << ",\"wall_s\":" << wall << ",\"fps\":" << (wall > 0 ? p->frames/wall : 0);
    // Reprises du rendu certifié (CPU, et GPU en mode exact) ; indécises : aucune attendue.
    long long rescued = p->certified_rescued, undecided = p->certified_undecided;
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    if (p->gpu_prototype) {
        // Tous les contextes de rendu (lot P).
        std::vector<void *> contexts{p->gpu_prototype};
        contexts.insert(contexts.end(), p->gpu_extra.begin(), p->gpu_extra.end());
        for (void *gpu : contexts) {
            long long gpu_rescued = 0, gpu_undecided = 0;
            fel_gpu_certified(gpu, &gpu_rescued, &gpu_undecided);
            rescued += gpu_rescued; undecided += gpu_undecided;
        }
    }
#endif
    text << ",\"certified_rescued\":" << rescued << ",\"certified_undecided\":" << undecided;
    // Images sans RPU ni EL recomposées sans résiduel (courbe du RPU précédent).
    text << ",\"bl_only_frames\":" << p->bl_only_frames.load();
    // Images refaites sur le CPU certifié (liste des indécises du GPU saturée).
    text << ",\"cpu_rescue_frames\":" << p->cpu_rescue_frames.load();
    // Attentes des files : l'étage le plus attendu par les autres est le goulot.
    const char *waits[]={"decode_blocked_s","render_starved_s","render_blocked_s","output_starved_s"};
    for(size_t i=0;i<p->waits.size();++i)text << ",\"" << waits[i] << "\":" << p->waits[i];
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    if (p->gpu_prototype) {
        // Temps GPU mesurés par étape (pl_timer), en millisecondes cumulées sur tous les contextes.
        double gpu[5] = {};
        std::vector<void *> contexts{p->gpu_prototype};
        contexts.insert(contexts.end(), p->gpu_extra.begin(), p->gpu_extra.end());
        for (void *context : contexts) {
            double part[5];
            fel_gpu_timings(context, part);
            for (int i = 0; i < 5; ++i) gpu[i] += part[i];
        }
        const char *stages[]={"gpu_import_ms","gpu_bl_ms","gpu_el_ms","gpu_compose_ms","gpu_output_ms"};
        for(int i=0;i<5;++i)text << ",\"" << stages[i] << "\":" << gpu[i];
    }
#endif
    text << '}';
    p->statistics=text.str();
    return p->statistics.c_str();
}
namespace {
#ifndef _WIN32
// SIGPIPE bloqué sur ce thread : l'écriture rend EPIPE, le signal éventuel est consommé.
struct SigpipeGuard {
    sigset_t set{}, previous{};
    bool blocked = false;
    SigpipeGuard() {
        sigemptyset(&set); sigaddset(&set, SIGPIPE);
        blocked = pthread_sigmask(SIG_BLOCK, &set, &previous) == 0;
    }
    ~SigpipeGuard() {
        if (!blocked) return;
        sigset_t pending;
        if (!sigismember(&previous, SIGPIPE) && sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE)) {
            int signal = 0;
            sigwait(&set, &signal);
        }
        pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    }
};
#endif
int run_context(mvo_fel_context *p) {
    int result = MVO_FEL_OK;
    try { p->run(); }
    catch (const Failure &exc) { p->error = exc.what(); result = exc.status; }
    catch (const std::exception &exc) { p->error = exc.what(); result = MVO_FEL_ERROR; }
    catch (...) { p->error = "Erreur native inconnue"; result = MVO_FEL_ERROR; }
    p->ended = Clock::now();
    if (p->cancelled.load()) return MVO_FEL_CANCELLED;
    return p->output_closed ? MVO_FEL_OUTPUT_CLOSED : result;
}
}
int mvo_fel_run_fd(mvo_fel_context *p, intptr_t fd, mvo_fel_progress progress, void *opaque) {
#ifdef _WIN32
    if (!p || fd == -1 || fd == 0) return MVO_FEL_ERROR;
#else
    if (!p || fd < 0) return MVO_FEL_ERROR;
    SigpipeGuard guard;
#endif
    p->fd = fd; p->progress = progress; p->opaque = opaque;
    return run_context(p);
}
int mvo_fel_run(mvo_fel_context *p, mvo_fel_write write, mvo_fel_progress progress, void *opaque) {
    if (!p || (!write && !p->measure_only)) return MVO_FEL_ERROR;
    p->write = write; p->progress = progress; p->opaque = opaque;
    return run_context(p);
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
int mvo_fel_vulkan_precision(void *gpu, const char *mode) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    const std::string value(mode ? mode : "");
    const int precision = value == "exact" ? FEL_PRECISION_EXACT : value == "standard" ? FEL_PRECISION_STANDARD
                        : value == "fast" ? FEL_PRECISION_FAST : -1;
    return gpu && precision >= 0 && fel_gpu_set_precision(gpu, precision) ? 0 : -1;
#else
    (void)gpu; (void)mode;
    return -1;
#endif
}
int mvo_fel_vulkan_configure(void *gpu, const char *chroma, const char *dither, int crop_x, int crop_y,
                             int width, int height) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    const int location = chroma_from_name(chroma), mode = dither_from_name(dither);
    if (!gpu || location < 0 || mode < 0 || crop_x < 0 || crop_y < 0 || width <= 0 || height <= 0 ||
        (crop_x | crop_y | width | height) & 1) return -1;
    const fel_output output{FEL_OUT_P010, mode, location, crop_x, crop_y, width, height};
    fel_gpu_configure(gpu, &output);
    return 0;
#else
    (void)gpu; (void)chroma; (void)dither; (void)crop_x; (void)crop_y; (void)width; (void)height;
    return -1;
#endif
}
void mvo_fel_vulkan_destroy(void *gpu) {
#ifdef MVO_FEL_VULKAN_PROTOTYPE
    fel_gpu_destroy(gpu);
#endif
}
