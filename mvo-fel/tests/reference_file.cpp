// SPDX-License-Identifier: LGPL-2.1-or-later
// Validation facultative sur média local : aucun extrait commercial dans le dépôt.
#include "reconstruction.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/sha.h>
int reference_init();
void reference_destroy();
int reference_frame(const AVDOVIMetadata *, const AVFrame *, const AVFrame *, float *);
int reference_pixels(const AVDOVIMetadata *, const float *, const float *, float *, int);
int reference_frame_kernel64(const AVDOVIMetadata *, const AVFrame *, const AVFrame *, float *);
void *fel_gpu_create(int, int, const char *);
void fel_gpu_destroy(void *);
int fel_gpu_device_name(void *, char *, size_t);
int fel_gpu_render(void *, const AVDOVIMetadata *, const AVFrame *, const AVFrame *, uint16_t *);
}
using namespace muxiveo::fel;
namespace {
void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
void checked(int code) {
    if (code >= 0) return;
    char message[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(code, message, sizeof(message));
    throw std::runtime_error(message);
}
struct FrameFree { void operator()(AVFrame *p) const { av_frame_free(&p); } };
struct PacketFree { void operator()(AVPacket *p) const { av_packet_free(&p); } };
using Frame = std::unique_ptr<AVFrame, FrameFree>;
using Packet = std::unique_ptr<AVPacket, PacketFree>;
struct Input {
    AVFormatContext *context = nullptr;
    explicit Input(const char *source) {
        checked(avformat_open_input(&context, source, nullptr, nullptr));
        checked(avformat_find_stream_info(context, nullptr));
    }
    ~Input() { avformat_close_input(&context); }
};
struct Layer {
    AVBSFContext *split = nullptr;
    AVCodecContext *decoder = nullptr;
    std::deque<Frame> ready;
    Layer(AVStream *stream, const char *mode) {
        checked(av_bsf_alloc(av_bsf_get_by_name("dovi_split"), &split));
        checked(avcodec_parameters_copy(split->par_in, stream->codecpar));
        split->time_base_in = stream->time_base;
        checked(av_opt_set(split->priv_data, "mode", mode, 0));
        checked(av_bsf_init(split));
        const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        decoder = avcodec_alloc_context3(codec);
        require(decoder, "Allocation décodeur");
        checked(avcodec_parameters_to_context(decoder, split->par_out));
        decoder->pkt_timebase = stream->time_base;
        decoder->thread_count = 2;
        decoder->err_recognition = AV_EF_EXPLODE;
        checked(avcodec_open2(decoder, codec, nullptr));
    }
    ~Layer() { av_bsf_free(&split); avcodec_free_context(&decoder); }
    void receive() {
        for (;;) {
            Frame f(av_frame_alloc());
            require(bool(f), "Allocation image");
            const int status = avcodec_receive_frame(decoder, f.get());
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            checked(status);
            require(!f->decode_error_flags && !(f->flags & AV_FRAME_FLAG_CORRUPT), "Image corrompue");
            ready.push_back(std::move(f));
            require(ready.size() <= 32, "Désalignement de la référence");
        }
    }
    void send(AVPacket *input) {
        Packet p(av_packet_alloc());
        require(bool(p), "Allocation paquet");
        if (input) checked(av_packet_ref(p.get(), input));
        checked(av_bsf_send_packet(split, input ? p.get() : nullptr));
        for (;;) {
            const int status = av_bsf_receive_packet(split, p.get());
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) break;
            checked(status);
            checked(avcodec_send_packet(decoder, p.get()));
            receive();
            av_packet_unref(p.get());
        }
        if (!input) { checked(avcodec_send_packet(decoder, nullptr)); receive(); }
    }
};
double compare(const AVFrame &bl, const AVFrame &el, int number, bool kernel64, bool performance = false, bool validate = false) {
    const auto *side = av_frame_get_side_data(&bl, AV_FRAME_DATA_DOVI_METADATA);
    require(side, "RPU absent de l'image décodée");
    const auto *metadata = reinterpret_cast<const AVDOVIMetadata *>(side->data);
    const Mapping mapping(*metadata);
    require(mapping.residual, "Image sans résiduel FEL");
    const FrameSampler bs(bl, bl.width, bl.height, false), es(el, bl.width, bl.height, true);
    Frame cpu(av_frame_alloc());
    cpu->width = bl.width; cpu->height = bl.height; cpu->format = AV_PIX_FMT_GBRP16LE;
    checked(av_frame_get_buffer(cpu.get(), 32));
    const auto start = std::chrono::steady_clock::now();
    reconstruct_rows(bs, es, mapping, *cpu, 0, bl.height);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (performance || validate) {
        if (!validate) {
        std::cout << "cpu_reconstruct_single_thread_s=" << seconds << std::endl;
        std::vector<Vec> signals(size_t(bl.width)*bl.height);
        auto t = std::chrono::steady_clock::now();
        for (int y = 0; y < bl.height; ++y) for (int x = 0; x < bl.width; ++x) {
            Vec b{}, e{};
            for (int c = 0; c < 3; ++c) { b[c] = bs.sample(c, x, y); e[c] = es.sample(c, x, y); }
            signals[size_t(y)*bl.width+x] = compose(mapping, b, e);
        }
        std::cout << "cpu_resample_and_compose_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count() << std::endl;
        t = std::chrono::steady_clock::now();
        double checksum = 0;
        for (const auto &signal : signals) { auto rgb = to_rgb(mapping, signal); checksum += rgb[0]+rgb[1]+rgb[2]; }
        std::cout << "cpu_colour_and_pq_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()
                  << " checksum=" << checksum << std::endl;
        }
        void *gpu = fel_gpu_create(bl.width, bl.height, std::getenv("MVO_FEL_TEST_DEVICE"));
        require(gpu, "Initialisation du prototype GPU");
        char device_name[256];
        require(fel_gpu_device_name(gpu, device_name, sizeof(device_name)), "Identification du GPU");
        std::cout << "gpu_device=" << device_name << std::endl;
        std::vector<uint16_t> output(4ull*bl.width*bl.height);
        std::vector<double> timings;
        for (int i = 0; i < (validate ? 2 : 17); ++i) {
            auto t = std::chrono::steady_clock::now();
            const int status = fel_gpu_render(gpu, metadata, &bl, &el, output.data());
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
            if (!status) { fel_gpu_destroy(gpu); require(false, "Échec du prototype GPU"); }
            if (i) timings.push_back(elapsed);
            else std::cout << "gpu_cold_frame_s=" << elapsed << std::endl;
        }
        fel_gpu_destroy(gpu);
        std::sort(timings.begin(), timings.end());
        const double median = .5*(timings[(timings.size()-1)/2]+timings[timings.size()/2]);
        double error = 0;
        size_t above = 0;
        for (int y = 0; y < bl.height; ++y) for (int x = 0; x < bl.width; ++x) for (int c = 0; c < 3; ++c) {
            const int plane = c == 0 ? 2 : c == 1 ? 0 : 1;
            const auto *p = cpu->data[plane]+y*cpu->linesize[plane]+2*x;
            const double delta = std::abs(int(output[4ull*(y*bl.width+x)+c])-int(p[0]+256*p[1]))*4095./65535;
            error = std::max(error, delta); above += delta > 2;
        }
        std::cout << "gpu_upload_resample_compose_colour_download_median_s=" << median
                  << " gpu_fps=" << 1/median << " gpu_min_s=" << timings.front() << " gpu_max_s=" << timings.back()
                  << " max_delta_cpu_pq12=" << error << " above_2=" << above << std::endl;
        std::vector<float> independent(4ull*bl.width*bl.height);
        require(reference_frame_kernel64(metadata, &bl, &el, independent.data()), "Référence indépendante GPU");
        double oracle_error = 0;
        size_t oracle_above = 0;
        for(size_t i=0;i<output.size();++i) if(i%4!=3) {
            const double delta=std::abs(output[i]/65535.-std::clamp(double(independent[i]),0.,1.))*4095.;
            require(std::isfinite(delta),"Référence GPU non finie");
            oracle_error=std::max(oracle_error,delta);oracle_above+=delta>2;
        }
        std::cout<<"gpu_frame="<<number<<" max_delta_kernel64_pq12="<<oracle_error<<" above_2="<<oracle_above<<std::endl;
        // Empreinte des pixels GBRP16LE pour vérifier aussi la bibliothèque livrée.
        AVSHA *sha = av_sha_alloc();
        require(sha, "Allocation SHA-256");
        av_sha_init(sha, 256);
        std::vector<uint8_t> row(2ull*bl.width);
        for (int c : {1, 2, 0}) for (int y=0; y<bl.height; ++y) {
            for (int x=0; x<bl.width; ++x) {
                const auto code=output[4ull*(y*bl.width+x)+c];
                row[2*x]=code & 255; row[2*x+1]=code >> 8;
            }
            av_sha_update(sha, row.data(), row.size());
        }
        uint8_t hash[32];
        av_sha_final(sha, hash);
        av_free(sha);
        std::cout<<"gpu_frame_bytes="<<6ull*bl.width*bl.height<<" gpu_frame_sha256=";
        for (auto byte : hash) std::cout<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(byte);
        std::cout<<std::dec<<std::setfill(' ')<<std::endl;
        return std::max(error,oracle_error);
    }
    std::vector<float> reference(4ull*bl.width*bl.height);
    require((kernel64 ? reference_frame_kernel64 : reference_frame)(metadata, &bl, &el, reference.data()), "Échec libplacebo");
    double maximum = 0, sum = 0;
    int worst_x = 0, worst_y = 0, worst_c = 0;
    double worst_actual = 0, worst_expected = 0;
    size_t above = 0;
    for (int y = 0; y < bl.height; ++y) for (int x = 0; x < bl.width; ++x) for (int c = 0; c < 3; ++c) {
        const int plane = c == 0 ? 2 : c == 1 ? 0 : 1;
        const auto *p = cpu->data[plane] + y*cpu->linesize[plane] + 2*x;
        const double actual = (p[0]+256*p[1])/65535.;
        const double expected = std::clamp(double(reference[4ull*(y*bl.width+x)+c]), 0., 1.);
        require(std::isfinite(expected), "Référence non finie");
        const double error = std::abs(actual-expected)*4095.;
        if (error > maximum) {
            maximum = error; worst_x = x; worst_y = y; worst_c = c;
            worst_actual = actual; worst_expected = expected;
        }
        sum += error; above += error > 2;
    }
    std::cout << "reference=" << (kernel64 ? "kernel64" : "gpu32") << " frame=" << number << " pts=" << bl.pts << " BL=" << bl.width << 'x' << bl.height
              << " EL=" << el.width << 'x' << el.height << " max_pq12=" << maximum
              << " mean_pq12=" << sum/(3ull*bl.width*bl.height) << " above_2=" << above
              << " cpu_single_thread_s=" << seconds << std::endl;
    // Comparaison de signal, sans score perceptuel ni décision de désactiver le FEL.
    Mapping no_el = mapping;
    no_el.residual = false;
    double effect = 0;
    for (int y = 0; y < bl.height; y += 32) for (int x = 0; x < bl.width; x += 32) {
        Vec b{}, e{};
        for (int c = 0; c < 3; ++c) { b[c] = bs.sample(c, x, y); e[c] = es.sample(c, x, y); }
        const auto full = to_rgb(mapping, compose(mapping, b, e));
        const auto base = to_rgb(no_el, compose(no_el, b, e));
        for (int c = 0; c < 3; ++c)
            effect = std::max(effect, std::abs(std::clamp(full[c], 0., 1.)-std::clamp(base[c], 0., 1.))*4095.);
    }
    std::cout << "frame=" << number << " max_el_effect_pq12_grid32=" << effect << std::endl;
    if (maximum > 2) {
        float b[4]{}, e[4]{}, result[4]{};
        for (int c = 0; c < 3; ++c) { b[c] = bs.sample(c, worst_x, worst_y); e[c] = es.sample(c, worst_x, worst_y); }
        require(reference_pixels(metadata, b, e, result, 1), "Diagnostic mapping");
        std::cout << std::setprecision(17) << "worst x=" << worst_x << " y=" << worst_y << " c=" << worst_c
                  << " cpu=" << worst_actual << " ref=" << worst_expected
                  << " ref_same_samples=" << result[worst_c]
                  << " bl=" << b[0] << ',' << b[1] << ',' << b[2]
                  << " el=" << e[0] << ',' << e[1] << ',' << e[2] << std::endl;
        for (int c = 0; c < 3; ++c)
            std::cout << "EL delta[" << c << "]=" << es.sample(c, worst_x, worst_y)-mapping.offset[c]
                      << " threshold=" << mapping.threshold[c] << std::endl;
    }
    return maximum;
}
}
int main(int argc, char **argv) {
    try {
        require(argc >= 2, "Usage : fel_reference_file source [stream=0] [frames=3] [stride=1] [gpu32|kernel64|perf|validate]");
        const int index = argc > 2 ? std::stoi(argv[2]) : 0;
        const int limit = argc > 3 ? std::stoi(argv[3]) : 3;
        const int stride = argc > 4 ? std::stoi(argv[4]) : 1;
        const bool kernel64 = argc > 5 && std::string(argv[5]) == "kernel64";
        const bool validate = argc > 5 && std::string(argv[5]) == "validate";
        const bool performance = argc > 5 && std::string(argv[5]) == "perf";
        require(argc <= 5 || kernel64 || performance || validate || std::string(argv[5]) == "gpu32", "Oracle inconnu");
        require(limit > 0 && stride > 0, "Paramètres invalides");
        Input input(argv[1]);
        require(index >= 0 && unsigned(index) < input.context->nb_streams, "Piste absente");
        auto *stream = input.context->streams[index];
        Layer bl(stream, "bl_rpu"), el(stream, "el");
        require(reference_init(), "Initialisation libplacebo");
        Packet p(av_packet_alloc());
        int decoded = 0, tested = 0;
        double maximum = 0;
        auto pairs = [&] {
            while (!bl.ready.empty() && !el.ready.empty()) {
                auto b = std::move(bl.ready.front()); bl.ready.pop_front();
                auto e = std::move(el.ready.front()); el.ready.pop_front();
                require(b->pts != AV_NOPTS_VALUE && b->pts == e->pts, "PTS BL/EL différents");
                if (decoded++ % stride == 0) { maximum = std::max(maximum, compare(*b, *e, decoded-1, kernel64, performance, validate)); ++tested; }
                if (tested == limit) return true;
            }
            return false;
        };
        bool done = false;
        while (!done) {
            int status = av_read_frame(input.context, p.get());
            if (status == AVERROR_EOF) break;
            checked(status);
            if (p->stream_index == index) { bl.send(p.get()); el.send(p.get()); done = pairs(); }
            av_packet_unref(p.get());
        }
        if (!done) { bl.send(nullptr); el.send(nullptr); pairs(); }
        reference_destroy();
        require(tested == limit, "Nombre d'images de référence insuffisant");
        require(maximum <= 2., "Tolérance maximale de deux codes PQ12 dépassée");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        reference_destroy();
        return 1;
    }
}
