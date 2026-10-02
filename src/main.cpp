// Muxiveo — muxiveo-rife : interpolation d'images RIFE v4 (Vulkan/ncnn) en flux y4m.
//
//   ffmpeg -i src.mkv -f yuv4mpegpipe -strict -1 - | muxiveo-rife --factor 2 | encodeur
//
// Les trames d'origine sont recopiées octet pour octet ; seules les trames
// intermédiaires sont générées. Les changements de scène (et les trames
// identiques) produisent une duplication au lieu d'une interpolation.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <shellapi.h>
#elif __APPLE__
#include <mach-o/dyld.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#include "engine.h"
#include "y4m.h"

#ifndef MUXIVEO_RIFE_VERSION
#define MUXIVEO_RIFE_VERSION "dev"
#endif
#ifndef MUXIVEO_RIFE_NCNN_VERSION
#define MUXIVEO_RIFE_NCNN_VERSION "?"
#endif

enum ExitCode
{
    EXIT_OK = 0,
    EXIT_USAGE = 1,
    EXIT_INPUT = 2,
    EXIT_GPU = 3,
    EXIT_IO = 4,
    EXIT_VRAM = 5,
};

static const char* DEFAULT_MODEL = "rife-v4.6";

// ---------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;

static fs::path path_from_utf8(const std::string& s)
{
#if _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, 0, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1)
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return fs::path(w);
#else
    return fs::path(s);
#endif
}

static std::string path_to_utf8(const fs::path& p)
{
#if _WIN32
    const std::wstring& w = p.native();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, 0, 0, 0, 0);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1)
        WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, 0, 0);
    return s;
#else
    return p.string();
#endif
}

static fs::path executable_dir(const char* argv0)
{
#if _WIN32
    wchar_t buf[32768];
    DWORD n = GetModuleFileNameW(0, buf, 32768);
    if (n > 0 && n < 32768)
        return fs::path(std::wstring(buf, n)).parent_path();
#elif __APPLE__
    char buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0)
        return fs::weakly_canonical(fs::path(buf)).parent_path();
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0)
        return fs::path(std::string(buf, (size_t)n)).parent_path();
#endif
    return fs::absolute(fs::path(argv0 ? argv0 : ".")).parent_path();
}

static FILE* open_file(const fs::path& path, bool write)
{
#if _WIN32
    return _wfopen(path.c_str(), write ? L"wb" : L"rb");
#else
    return fopen(path.c_str(), write ? "wb" : "rb");
#endif
}

static bool parse_ratio(const std::string& s, int64_t& num, int64_t& den)
{
    long long n = 0, d = 1;
    const char* sep = strpbrk(s.c_str(), "/:");
    if (sep)
    {
        if (sscanf(s.c_str(), "%lld", &n) != 1 || sscanf(sep + 1, "%lld", &d) != 1)
            return false;
    }
    else
    {
        // décimal toléré (ex. 59.94) : converti en fraction /1000
        double v = atof(s.c_str());
        if (v <= 0)
            return false;
        if (std::floor(v) == v)
        {
            n = (long long)v;
            d = 1;
        }
        else
        {
            n = (long long)std::llround(v * 1000.0);
            d = 1000;
        }
    }
    if (n <= 0 || d <= 0)
        return false;
    int64_t g = std::gcd((int64_t)n, (int64_t)d);
    num = n / g;
    den = d / g;
    return true;
}

static void reduce(int64_t& num, int64_t& den)
{
    int64_t g = std::gcd(num, den);
    if (g > 1)
    {
        num /= g;
        den /= g;
    }
}

// ---------------------------------------------------------------------------
// Tampons et files bornées
// ---------------------------------------------------------------------------

typedef std::vector<uint32_t> Words;
typedef std::shared_ptr<Words> FrameBuffer;

class BufferPool
{
public:
    explicit BufferPool(size_t _words) : words(_words) {}

    ~BufferPool()
    {
        for (Words* w : free_list)
            delete w;
    }

    FrameBuffer acquire()
    {
        Words* buf = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!free_list.empty())
            {
                buf = free_list.back();
                free_list.pop_back();
            }
        }
        if (!buf)
            buf = new Words(words, 0u);
        return FrameBuffer(buf, [this](Words* p) {
            std::lock_guard<std::mutex> lock(mutex);
            free_list.push_back(p);
        });
    }

private:
    size_t words;
    std::mutex mutex;
    std::vector<Words*> free_list;
};

template <typename T>
class BoundedQueue
{
public:
    explicit BoundedQueue(size_t _capacity) : capacity(_capacity), closed(false) {}

    bool push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_not_full.wait(lock, [this] { return closed || items.size() < capacity; });
        if (closed)
            return false;
        items.push_back(std::move(item));
        cond_not_empty.notify_one();
        return true;
    }

    // Retourne false quand la file est fermée et vide.
    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_not_empty.wait(lock, [this] { return closed || !items.empty(); });
        if (items.empty())
            return false;
        item = std::move(items.front());
        items.pop_front();
        cond_not_full.notify_one();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex);
        closed = true;
        cond_not_empty.notify_all();
        cond_not_full.notify_all();
    }

private:
    size_t capacity;
    bool closed;
    std::deque<T> items;
    std::mutex mutex;
    std::condition_variable cond_not_empty;
    std::condition_variable cond_not_full;
};

// ---------------------------------------------------------------------------
// Détection de changement de scène (score façon ffmpeg scdet, luma seule)
// ---------------------------------------------------------------------------

static inline uint32_t sample_at(const uint8_t* data, int64_t idx, int bps)
{
    if (bps == 2)
        return (uint32_t)data[idx * 2] | ((uint32_t)data[idx * 2 + 1] << 8);
    return data[idx];
}

// Différence absolue moyenne de luma, en pourcentage de la pleine échelle.
static double luma_mafd(const uint8_t* a, const uint8_t* b, const FrameFormat& fmt)
{
    const int step = std::max(1, std::min(fmt.width, fmt.height) / 540);
    const int bps = fmt.bytes_per_sample;
    uint64_t sad = 0;
    uint64_t count = 0;
    for (int y = 0; y < fmt.height; y += step)
    {
        const int64_t row = (int64_t)y * fmt.width;
        for (int x = 0; x < fmt.width; x += step)
        {
            int32_t d = (int32_t)sample_at(a, row + x, bps) - (int32_t)sample_at(b, row + x, bps);
            sad += (uint64_t)(d < 0 ? -d : d);
            count++;
        }
    }
    if (count == 0)
        return 0.0;
    return (double)sad * 100.0 / (double)count / (double)(1u << fmt.bit_depth);
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options
{
    std::string input = "-";
    std::string output = "-";
    std::string model = DEFAULT_MODEL;
    std::string factor;
    std::string fps;
    std::string matrix;
    std::string range;
    std::string chroma_loc;
    int gpu = -1;
    int threads = 2;
    int padding = 0;
    int tta = 1;
    double scene_threshold = 10.0;
    bool fp32 = false;
    bool uhd = false;
    bool list_gpus = false;
    bool version = false;
    bool quiet = false;
    bool verbose = false;
    bool allow_interlaced = false;
    bool roundtrip = false;
    double progress_interval = 1.0;
};

static void print_usage(FILE* fp)
{
    fprintf(fp,
            "Usage : muxiveo-rife [options]\n"
            "  -i <fichier|->          entrée y4m (défaut : stdin)\n"
            "  -o <fichier|->          sortie y4m (défaut : stdout)\n"
            "  --factor <n|p/q>        multiplicateur de cadence (défaut : 2)\n"
            "  --fps <num/den>         cadence de sortie cible (exclusif avec --factor)\n"
            "  -m, --model <nom|dir>   modèle RIFE v4 (défaut : %s, cherché dans <exe>/rife-models)\n"
            "  -g, --gpu <index>       GPU Vulkan (défaut : automatique)\n"
            "  --matrix <m>            bt709 | bt2020nc | bt601 | smpte240m | fcc\n"
            "  --range <r>             limited | full (défaut : en-tête y4m, sinon limited)\n"
            "  --chroma-loc <l>        left | center | topleft (défaut : en-tête y4m, sinon left)\n"
            "  --scene-threshold <s>   seuil de changement de scène 0-100 (défaut : 10 ; 0 = désactivé)\n"
            "  --uhd                   mode rapide : flux optique à demi-résolution (échelles x2)\n"
            "  --fp32                  calcul en float32 (plus lent, précision maximale)\n"
            "  --tta <n>               moyenne de n variantes : 2 = + sens temporel inverse, 4 = + miroir\n"
            "                          horizontal, 8 = + miroirs vertical et double (coût x n ; défaut : 1)\n"
            "  --padding <n>           padding du réseau (défaut : selon le modèle, x2 avec --uhd)\n"
            "  -j, --threads <n>       threads CPU ncnn (défaut : 2)\n"
            "  --allow-interlaced      accepter une entrée entrelacée\n"
            "  --progress-interval <s> intervalle des lignes de progression (défaut : 1)\n"
            "  --quiet                 pas de ligne de progression\n"
            "  --verbose               affiche les diagnostics Vulkan de ncnn\n"
            "  --list-gpus             liste les GPU Vulkan (JSON) et quitte\n"
            "  --version               affiche la version et quitte\n",
            DEFAULT_MODEL);
}

static bool parse_args(const std::vector<std::string>& args, Options& o, std::string& error)
{
    for (size_t i = 1; i < args.size(); i++)
    {
        const std::string& a = args[i];
        auto value = [&](std::string& dst) -> bool {
            if (i + 1 >= args.size())
            {
                error = "valeur manquante pour " + a;
                return false;
            }
            dst = args[++i];
            return true;
        };
        std::string v;

        if (a == "-i" || a == "--input") { if (!value(o.input)) return false; }
        else if (a == "-o" || a == "--output") { if (!value(o.output)) return false; }
        else if (a == "-m" || a == "--model") { if (!value(o.model)) return false; }
        else if (a == "--factor") { if (!value(o.factor)) return false; }
        else if (a == "--fps") { if (!value(o.fps)) return false; }
        else if (a == "--matrix") { if (!value(o.matrix)) return false; }
        else if (a == "--range") { if (!value(o.range)) return false; }
        else if (a == "--chroma-loc") { if (!value(o.chroma_loc)) return false; }
        else if (a == "-g" || a == "--gpu") { if (!value(v)) return false; o.gpu = atoi(v.c_str()); }
        else if (a == "-j" || a == "--threads") { if (!value(v)) return false; o.threads = atoi(v.c_str()); }
        else if (a == "--padding") { if (!value(v)) return false; o.padding = atoi(v.c_str()); }
        else if (a == "--tta")
        {
            if (!value(v)) return false;
            o.tta = atoi(v.c_str());
            if (o.tta != 1 && o.tta != 2 && o.tta != 4 && o.tta != 8)
            {
                error = "--tta : 1, 2, 4 ou 8 attendu (reçu : " + v + ")";
                return false;
            }
        }
        else if (a == "--scene-threshold") { if (!value(v)) return false; o.scene_threshold = atof(v.c_str()); }
        else if (a == "--progress-interval") { if (!value(v)) return false; o.progress_interval = atof(v.c_str()); }
        else if (a == "--fp32") o.fp32 = true;
        else if (a == "--uhd") o.uhd = true;
        else if (a == "--allow-interlaced") o.allow_interlaced = true;
        else if (a == "--debug-roundtrip") o.roundtrip = true;
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--list-gpus") o.list_gpus = true;
        else if (a == "--version") o.version = true;
        else if (a == "-h" || a == "--help")
        {
            print_usage(stdout);
            exit(EXIT_OK);
        }
        else
        {
            error = "option inconnue : " + a;
            return false;
        }
    }
    if (!o.factor.empty() && !o.fps.empty())
    {
        error = "--factor et --fps sont exclusifs";
        return false;
    }
    return true;
}

static bool resolve_color(const Options& o, const FrameFormat& fmt, ColorParams& color, std::string& error)
{
    std::string m = o.matrix;
    if (m.empty())
    {
        m = fmt.height > 576 ? "bt709" : "bt601";
        fprintf(stderr, "warning: matrice non précisée, %s supposée\n", m.c_str());
    }
    if (m == "bt709") { color.kr = 0.2126f; color.kb = 0.0722f; }
    else if (m == "bt2020nc" || m == "bt2020") { color.kr = 0.2627f; color.kb = 0.0593f; }
    else if (m == "bt601" || m == "smpte170m" || m == "bt470bg") { color.kr = 0.299f; color.kb = 0.114f; }
    else if (m == "smpte240m") { color.kr = 0.212f; color.kb = 0.087f; }
    else if (m == "fcc") { color.kr = 0.30f; color.kb = 0.11f; }
    else
    {
        error = "matrice non supportée : " + m;
        return false;
    }

    if (o.range == "full" || o.range == "pc" || o.range == "jpeg")
        color.full_range = true;
    else if (o.range == "limited" || o.range == "tv" || o.range == "mpeg")
        color.full_range = false;
    else if (o.range.empty())
        color.full_range = fmt.range == ColorRange::Full;
    else
    {
        error = "plage non supportée : " + o.range;
        return false;
    }

    if (o.chroma_loc == "left")
        color.siting = ChromaSiting::Left;
    else if (o.chroma_loc == "center")
        color.siting = ChromaSiting::Center;
    else if (o.chroma_loc == "topleft")
        color.siting = ChromaSiting::TopLeft;
    else if (o.chroma_loc.empty() || o.chroma_loc == "unspecified" || o.chroma_loc == "auto")
        color.siting = fmt.siting == ChromaSiting::Unknown ? ChromaSiting::Left : fmt.siting;
    else
    {
        error = "position chroma non supportée : " + o.chroma_loc;
        return false;
    }
    return true;
}

static fs::path resolve_model_dir(const std::string& model, const fs::path& exe_dir)
{
    fs::path p = path_from_utf8(model);
    std::error_code ec;
    if (fs::is_directory(p, ec))
        return p;
    return exe_dir / "rife-models" / p;
}

static int model_padding(const std::string& name)
{
    if (name.find("rife-v4.25-lite") != std::string::npos)
        return 128;
    if (name.find("rife-v4.25") != std::string::npos || name.find("rife-v4.26") != std::string::npos)
        return 64;
    return 32;
}

static int list_gpus()
{
    const int count = ncnn::get_gpu_count();
    const int def = count > 0 ? ncnn::get_default_gpu_index() : -1;
    printf("{\"version\": \"%s\", \"default\": %d, \"gpus\": [", MUXIVEO_RIFE_VERSION, def);
    for (int i = 0; i < count; i++)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        const char* type = "other";
        switch (info.type())
        {
        case 0: type = "discrete"; break;
        case 1: type = "integrated"; break;
        case 2: type = "virtual"; break;
        case 3: type = "cpu"; break;
        }
        std::string name = info.device_name();
        std::string escaped;
        for (char c : name)
        {
            if (c == '"' || c == '\\')
                escaped.push_back('\\');
            escaped.push_back(c);
        }
        printf("%s{\"index\": %d, \"name\": \"%s\", \"type\": \"%s\", \"fp16\": %s}", i ? ", " : "", i,
               escaped.c_str(), type, info.support_fp16_storage() ? "true" : "false");
    }
    printf("]}\n");
    return EXIT_OK;
}

// ---------------------------------------------------------------------------
// Traitement
// ---------------------------------------------------------------------------

struct SourceFrame
{
    FrameBuffer data;
    GpuFrame gpu;
    int64_t index = -1;
};

static int run(const Options& o, const fs::path& exe_dir)
{
    FILE* in = stdin;
    FILE* out = stdout;
    if (o.input != "-")
    {
        in = open_file(path_from_utf8(o.input), false);
        if (!in)
        {
            fprintf(stderr, "error: ouverture impossible : %s\n", o.input.c_str());
            return EXIT_IO;
        }
    }
    if (o.output != "-")
    {
        out = open_file(path_from_utf8(o.output), true);
        if (!out)
        {
            fprintf(stderr, "error: création impossible : %s\n", o.output.c_str());
            return EXIT_IO;
        }
    }
    setvbuf(in, 0, _IOFBF, 1 << 20);
    setvbuf(out, 0, _IOFBF, 1 << 20);

    std::string error;
    Y4mReader reader(in);
    if (!reader.read_header(error))
    {
        fprintf(stderr, "error: %s\n", error.c_str());
        return EXIT_INPUT;
    }
    const FrameFormat fmt = reader.format();

    if (!o.allow_interlaced && (fmt.interlace == 't' || fmt.interlace == 'b' || fmt.interlace == 'm'))
    {
        fprintf(stderr, "error: entrée entrelacée (I%c) : désentrelacer avant l'interpolation\n", fmt.interlace);
        return EXIT_INPUT;
    }

    // rapport de cadence A/B (sortie / entrée)
    int64_t ra = 2, rb = 1;
    if (!o.factor.empty() && !parse_ratio(o.factor, ra, rb))
    {
        fprintf(stderr, "error: --factor invalide : %s\n", o.factor.c_str());
        return EXIT_USAGE;
    }
    if (!o.fps.empty())
    {
        int64_t fn, fd;
        if (!parse_ratio(o.fps, fn, fd))
        {
            fprintf(stderr, "error: --fps invalide : %s\n", o.fps.c_str());
            return EXIT_USAGE;
        }
        ra = fn * fmt.fps_den;
        rb = fd * fmt.fps_num;
        reduce(ra, rb);
    }
    if (o.roundtrip)
    {
        ra = 1;
        rb = 1;
    }
    if (ra < rb)
    {
        fprintf(stderr, "error: la cadence de sortie doit être supérieure ou égale à la cadence d'entrée\n");
        return EXIT_USAGE;
    }

    ColorParams color;
    if (!resolve_color(o, fmt, color, error))
    {
        fprintf(stderr, "error: %s\n", error.c_str());
        return EXIT_USAGE;
    }

    FrameFormat out_fmt = fmt;
    out_fmt.fps_num = fmt.fps_num * ra;
    out_fmt.fps_den = fmt.fps_den * rb;
    reduce(out_fmt.fps_num, out_fmt.fps_den);

    // moteur GPU
    const fs::path model_dir = resolve_model_dir(o.model, exe_dir);
    const std::string model_name = path_to_utf8(model_dir.filename());
    RifeEngine engine;
    engine.set_tta(o.tta);
    const bool needs_gpu = ra != rb || o.roundtrip;
    if (needs_gpu)
    {
        if (!engine.init(o.gpu, o.fp32, o.threads, error)
                || !engine.load_model(model_dir, o.padding > 0 ? o.padding : model_padding(model_name) * (o.uhd ? 2 : 1),
                                      o.uhd, error)
                || !engine.configure(fmt, color, error))
        {
            fprintf(stderr, "error: %s\n", error.c_str());
            return EXIT_GPU;
        }
    }

    if (!o.quiet)
    {
        fprintf(stderr,
                "info: %dx%d %d bits sous-échantillonnage %d:%d | %lld/%lld -> %lld/%lld fps | modèle %s%s%s | GPU %s (%s)\n",
                fmt.width, fmt.height, fmt.bit_depth, fmt.sub_x, fmt.sub_y,
                (long long)fmt.fps_num, (long long)fmt.fps_den,
                (long long)out_fmt.fps_num, (long long)out_fmt.fps_den,
                model_name.c_str(), o.uhd ? " (uhd)" : "",
                o.tta > 1 ? (" (tta x" + std::to_string(o.tta) + ")").c_str() : "",
                needs_gpu ? engine.device_name().c_str() : "-",
                engine.uses_fp16() ? "fp16" : "fp32");
    }

    Y4mWriter writer(out);
    if (!writer.write_header(out_fmt, reader.passthrough_tokens()))
    {
        fprintf(stderr, "error: écriture de l'en-tête impossible\n");
        return EXIT_IO;
    }

    const size_t frame_bytes = fmt.frame_bytes();
    const size_t frame_words = fmt.padded_frame_bytes() / 4;
    BufferPool pool(frame_words);

    BoundedQueue<FrameBuffer> read_queue(3);
    BoundedQueue<FrameBuffer> write_queue(4);
    std::atomic<bool> read_failed(false);
    std::atomic<bool> write_failed(false);
    std::string read_error;

    std::thread reader_thread([&] {
        for (;;)
        {
            FrameBuffer buf = pool.acquire();
            std::string err;
            int r = reader.read_frame((uint8_t*)buf->data(), err);
            if (r <= 0)
            {
                if (r < 0)
                {
                    read_error = err;
                    read_failed = true;
                }
                break;
            }
            if (!read_queue.push(buf))
                break;
        }
        read_queue.close();
    });

    std::thread writer_thread([&] {
        FrameBuffer buf;
        while (write_queue.pop(buf))
        {
            if (!write_failed && !writer.write_frame((const uint8_t*)buf->data(), frame_bytes))
            {
                write_failed = true;
                write_queue.close();
            }
            buf.reset();
        }
        if (!write_failed && !writer.flush())
            write_failed = true;
    });

    int64_t frames_in = 0;
    int64_t frames_out = 0;
    int64_t interpolated = 0;
    int64_t scene_cuts = 0;
    int64_t static_pairs = 0;
    int exit_code = EXIT_OK;

    SourceFrame cur;
    SourceFrame nxt;
    bool has_next = false;
    bool pair_cut = false;
    bool pair_static = false;
    double prev_mafd = 0.0;

    auto fetch_next = [&]() -> bool {
        FrameBuffer buf;
        if (!read_queue.pop(buf))
            return false;
        nxt.data = buf;
        nxt.gpu.reset();
        nxt.index = frames_in++;
        return true;
    };

    auto analyse_pair = [&]() {
        const uint8_t* a = (const uint8_t*)cur.data->data();
        const uint8_t* b = (const uint8_t*)nxt.data->data();
        const double mafd = luma_mafd(a, b, fmt);
        const double diff = std::fabs(mafd - prev_mafd);
        const double score = std::min(100.0, std::min(mafd, diff));
        prev_mafd = mafd;
        pair_static = mafd == 0.0 && memcmp(a, b, frame_bytes) == 0;
        pair_cut = !pair_static && o.scene_threshold > 0.0 && score >= o.scene_threshold;
        if (pair_static)
            static_pairs++;
        if (pair_cut)
            scene_cuts++;
    };

    auto emit = [&](const FrameBuffer& buf) -> bool {
        if (write_failed || !write_queue.push(buf))
            return false;
        frames_out++;
        return true;
    };

    // échec GPU : mémoire vidéo insuffisante (code dédié) ou erreur d'exécution
    auto gpu_failure = [&](const std::string& err) {
        if (engine.out_of_memory())
        {
            fprintf(stderr,
                    "error: mémoire GPU insuffisante (VRAM) pour %dx%d avec le modèle %s : fermer les applications "
                    "qui utilisent le GPU%s\n",
                    fmt.width, fmt.height, model_name.c_str(),
                    o.uhd ? ", ou choisir un modèle plus léger" : ", activer le mode rapide (--uhd) ou choisir un modèle plus léger");
            exit_code = EXIT_VRAM;
        }
        else
        {
            fprintf(stderr, "error: %s\n", err.c_str());
            exit_code = EXIT_GPU;
        }
    };

    const auto t_start = std::chrono::steady_clock::now();
    auto t_report = t_start;

    if (fetch_next())
    {
        cur = std::move(nxt);
        nxt = SourceFrame();
        has_next = fetch_next();
        if (has_next)
            analyse_pair();

        for (int64_t k = 0;; k++)
        {
            const int64_t num = k * rb;
            const int64_t i = num / ra;
            const int64_t rem = num % ra;

            // avance la fenêtre [cur, nxt] jusqu'à la trame source i
            bool finished = false;
            while (cur.index < i)
            {
                if (!has_next)
                {
                    finished = true;
                    break;
                }
                cur = std::move(nxt);
                nxt = SourceFrame();
                has_next = fetch_next();
                if (has_next)
                    analyse_pair();
            }
            if (finished)
                break;

            bool ok;
            if (o.roundtrip)
            {
                std::string err;
                FrameBuffer outbuf = pool.acquire();
                if (!cur.gpu.ready && !engine.upload((const uint8_t*)cur.data->data(), cur.gpu, err))
                {
                    gpu_failure(err);
                    break;
                }
                if (!engine.roundtrip(cur.gpu, (uint8_t*)outbuf->data(), err))
                {
                    gpu_failure(err);
                    break;
                }
                ok = emit(outbuf);
            }
            else if (rem == 0 || !has_next || pair_cut || pair_static)
            {
                // trame d'origine, fin de flux, coupe ou trame figée : duplication exacte
                ok = emit(cur.data);
            }
            else
            {
                std::string err;
                if (!cur.gpu.ready && !engine.upload((const uint8_t*)cur.data->data(), cur.gpu, err))
                {
                    gpu_failure(err);
                    break;
                }
                if (!nxt.gpu.ready && !engine.upload((const uint8_t*)nxt.data->data(), nxt.gpu, err))
                {
                    gpu_failure(err);
                    break;
                }
                FrameBuffer outbuf = pool.acquire();
                const float t = (float)((double)rem / (double)ra);
                if (!engine.interpolate(cur.gpu, nxt.gpu, t, (uint8_t*)outbuf->data(), err))
                {
                    gpu_failure(err);
                    break;
                }
                interpolated++;
                ok = emit(outbuf);
            }
            if (!ok)
            {
                exit_code = EXIT_IO;
                break;
            }

            if (!o.quiet)
            {
                const auto now = std::chrono::steady_clock::now();
                const double since = std::chrono::duration<double>(now - t_report).count();
                if (since >= o.progress_interval)
                {
                    t_report = now;
                    const double elapsed = std::chrono::duration<double>(now - t_start).count();
                    fprintf(stderr, "progress in=%lld out=%lld interpolated=%lld scenes=%lld static=%lld fps=%.2f\n",
                            (long long)frames_in, (long long)frames_out, (long long)interpolated,
                            (long long)scene_cuts, (long long)static_pairs,
                            elapsed > 0 ? frames_out / elapsed : 0.0);
                    fflush(stderr);
                }
            }
        }
    }

    // libère les trames GPU avant le moteur
    cur.gpu.reset();
    nxt.gpu.reset();

    if (exit_code != EXIT_OK)
    {
        // le lecteur peut rester bloqué sur stdin : sortie immédiate
        fprintf(stderr, "done in=%lld out=%lld interpolated=%lld scenes=%lld static=%lld exit=%d\n",
                (long long)frames_in, (long long)frames_out, (long long)interpolated,
                (long long)scene_cuts, (long long)static_pairs, exit_code);
        fflush(stderr);
        std::_Exit(exit_code);
    }

    read_queue.close();
    write_queue.close();
    reader_thread.join();
    writer_thread.join();

    if (exit_code == EXIT_OK && read_failed)
    {
        fprintf(stderr, "error: %s\n", read_error.c_str());
        exit_code = EXIT_INPUT;
    }
    if (exit_code == EXIT_OK && write_failed)
    {
        fprintf(stderr, "error: écriture de la sortie interrompue (pipe fermé ?)\n");
        exit_code = EXIT_IO;
    }

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    fprintf(stderr, "done in=%lld out=%lld interpolated=%lld scenes=%lld static=%lld seconds=%.2f exit=%d\n",
            (long long)frames_in, (long long)frames_out, (long long)interpolated,
            (long long)scene_cuts, (long long)static_pairs, elapsed, exit_code);

    if (in != stdin)
        fclose(in);
    if (out != stdout)
        fclose(out);
    return exit_code;
}

// ncnn liste les GPU sur stderr à l'initialisation : masqué hors --verbose.
static int create_gpu_instance_quiet(bool verbose, const char* driver_path)
{
    if (verbose)
        return ncnn::create_gpu_instance(driver_path);

    fflush(stderr);
#if _WIN32
    int saved = _dup(_fileno(stderr));
    FILE* null_fp = _wfopen(L"NUL", L"w");
    if (null_fp)
        _dup2(_fileno(null_fp), _fileno(stderr));
#else
    int saved = dup(fileno(stderr));
    FILE* null_fp = fopen("/dev/null", "w");
    if (null_fp)
        dup2(fileno(null_fp), fileno(stderr));
#endif
    const int ret = ncnn::create_gpu_instance(driver_path);
    fflush(stderr);
#if _WIN32
    if (saved >= 0)
    {
        _dup2(saved, _fileno(stderr));
        _close(saved);
    }
#else
    if (saved >= 0)
    {
        dup2(saved, fileno(stderr));
        close(saved);
    }
#endif
    if (null_fp)
        fclose(null_fp);
    return ret;
}

static std::vector<std::string> utf8_args(int argc, char** argv)
{
    std::vector<std::string> args;
#if _WIN32
    (void)argc;
    (void)argv;
    int wargc = 0;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    for (int i = 0; i < wargc; i++)
    {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, 0, 0, 0, 0);
        std::string s(n > 0 ? n - 1 : 0, '\0');
        if (n > 1)
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, 0, 0);
        args.push_back(s);
    }
    LocalFree(wargv);
#else
    for (int i = 0; i < argc; i++)
        args.push_back(argv[i]);
#endif
    return args;
}

int main(int argc, char** argv)
{
#if _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#else
    signal(SIGPIPE, SIG_IGN);
#endif

    Options o;
    std::string error;
    if (!parse_args(utf8_args(argc, argv), o, error))
    {
        fprintf(stderr, "error: %s\n", error.c_str());
        print_usage(stderr);
        return EXIT_USAGE;
    }

    if (o.version)
    {
        printf("muxiveo-rife %s (ncnn %s)\n", MUXIVEO_RIFE_VERSION, MUXIVEO_RIFE_NCNN_VERSION);
        return EXIT_OK;
    }

    const fs::path exe_dir = executable_dir(argc > 0 ? argv[0] : 0);

    // macOS : MoltenVK livré à côté de l'exécutable (sinon chargeur Vulkan système).
    std::string driver_path;
#if __APPLE__
    {
        std::error_code ec;
        const fs::path moltenvk = exe_dir / "libMoltenVK.dylib";
        if (fs::is_regular_file(moltenvk, ec))
            driver_path = moltenvk.string();
    }
#endif

    if (create_gpu_instance_quiet(o.verbose, driver_path.empty() ? 0 : driver_path.c_str()) != 0 && !o.list_gpus)
    {
        fprintf(stderr, "error: Vulkan indisponible (pilote ou chargeur libvulkan absent)\n");
        return EXIT_GPU;
    }

    int ret;
    if (o.list_gpus)
        ret = list_gpus();
    else
        ret = run(o, exe_dir);

    ncnn::destroy_gpu_instance();
    return ret;
}
