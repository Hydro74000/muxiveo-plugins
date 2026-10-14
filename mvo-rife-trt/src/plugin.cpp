// mvo-rife-trt — inférence RIFE de muxiveo-rife avec NVIDIA TensorRT for RTX.
//
// Le plugin reçoit des tampons CUDA déjà remplis par muxiveo-rife (mémoire partagée avec Vulkan) et écrit
// l'image interpolée dans le tampon de sortie. Interface : mvo-rife/src/trt_plugin_abi.h.
// Le pilote CUDA est chargé à l'exécution ; TensorRT for RTX est livré à côté du plugin.

#include "trt_plugin_abi.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef MVO_RIFE_TRT_VERSION
#define MVO_RIFE_TRT_VERSION "0.0.0"
#endif

namespace fs = std::filesystem;

namespace {

// Dimensions extrêmes du profil (tenseurs paddés) : de la vignette au 8K, portrait compris.
constexpr int kMinSide = 32;
constexpr int kMaxSide = 8192;
// Architecture minimale de TensorRT for RTX : Turing (SM 7.5).
constexpr int kMinComputeCapability = 75;

// ---------------------------------------------------------------------------
// Pilote CUDA (chargé à l'exécution, cuda.h non requis)
// ---------------------------------------------------------------------------

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;
enum
{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_STREAM_NON_BLOCKING = 1
};
struct CUuuid
{
    unsigned char bytes[16];
};

struct Cuda
{
    void* lib = nullptr;
    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetUuid)(CUuuid*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxPopCurrent)(CUcontext*) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemsetD16Async)(CUdeviceptr, unsigned short, size_t, CUstream) = nullptr;
    std::string error;
    bool ok = false;
};

void* find_symbol(void* lib, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

Cuda& cuda()
{
    static Cuda c;
    static std::once_flag once;
    std::call_once(once, [] {
#if defined(_WIN32)
        c.lib = reinterpret_cast<void*>(LoadLibraryW(L"nvcuda.dll"));
#else
        c.lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!c.lib)
        {
            c.error = "pilote NVIDIA (CUDA) absent";
            return;
        }
        bool all = true;
        auto load = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(find_symbol(c.lib, name));
            all = all && fn != nullptr;
        };
        load(c.cuInit, "cuInit");
        load(c.cuDeviceGet, "cuDeviceGet");
        load(c.cuDeviceGetAttribute, "cuDeviceGetAttribute");
        load(c.cuDeviceGetUuid, "cuDeviceGetUuid");
        load(c.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        load(c.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
        load(c.cuCtxPushCurrent, "cuCtxPushCurrent_v2");
        load(c.cuCtxPopCurrent, "cuCtxPopCurrent_v2");
        load(c.cuStreamCreate, "cuStreamCreate");
        load(c.cuStreamDestroy, "cuStreamDestroy_v2");
        load(c.cuStreamSynchronize, "cuStreamSynchronize");
        load(c.cuMemAlloc, "cuMemAlloc_v2");
        load(c.cuMemFree, "cuMemFree_v2");
        load(c.cuMemsetD16Async, "cuMemsetD16Async");
        if (!all)
        {
            c.error = "symboles du pilote CUDA introuvables (pilote trop ancien ?)";
            return;
        }
        if (c.cuInit(0) != 0)
        {
            c.error = "initialisation CUDA impossible";
            return;
        }
        c.ok = true;
    });
    return c;
}

void set_error(char* error, size_t size, const std::string& message)
{
    if (!error || size == 0)
        return;
    const size_t n = std::min(size - 1, message.size());
    memcpy(error, message.data(), n);
    error[n] = '\0';
}

// Architecture du périphérique (ex. 89), 0 si illisible.
int compute_capability(const Cuda& c, CUdevice dev)
{
    int major = 0;
    int minor = 0;
    if (c.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != 0
        || c.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != 0)
        return 0;
    return major * 10 + minor;
}

std::string trt_version()
{
    const int v = getInferLibVersion();
    // Encodage TensorRT : major * 10000 + minor * 100 + patch.
    return std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." + std::to_string(v % 100);
}

// Contexte CUDA primaire rendu courant pour la durée d'un appel.
class ContextScope
{
public:
    ContextScope(const Cuda& c, CUcontext ctx) : cu(c), active(c.cuCtxPushCurrent(ctx) == 0) {}
    ~ContextScope()
    {
        if (active)
        {
            CUcontext popped = nullptr;
            cu.cuCtxPopCurrent(&popped);
        }
    }
    bool ok() const { return active; }

private:
    const Cuda& cu;
    bool active;
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kERROR)
        {
            std::lock_guard<std::mutex> lock(mutex);
            last_error = msg ? msg : "";
        }
    }
    std::string take_error()
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::string e;
        e.swap(last_error);
        return e;
    }

private:
    std::mutex mutex;
    std::string last_error;
};

bool read_file(const fs::path& path, std::vector<char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0)
        return false;
    data.resize(static_cast<size_t>(size));
    in.seekg(0);
    return static_cast<bool>(in.read(data.data(), size));
}

// Écriture atomique : fichier temporaire unique puis renommage (un autre processus peut lire en parallèle).
bool write_file_atomic(const fs::path& path, const void* data, size_t size)
{
    static std::atomic<unsigned> counter{0};
    const fs::path tmp = path.string() + ".tmp" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                         + "-" + std::to_string(counter++);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out || !out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)))
        {
            std::error_code ec;
            fs::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec)
    {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

// fp32 -> fp16 (arrondi au plus proche), pour la carte de temps.
unsigned short to_half(float value)
{
    uint32_t x;
    memcpy(&x, &value, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = x & 0x7fffffu;
    if (exponent <= 0)
        return static_cast<unsigned short>(sign);
    if (exponent >= 31)
        return static_cast<unsigned short>(sign | 0x7c00u);
    mantissa += 0x1000u;
    if (mantissa & 0x800000u)
    {
        mantissa = 0;
        exponent++;
    }
    return static_cast<unsigned short>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

std::string hex(const unsigned char* bytes, size_t n)
{
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++)
    {
        s.push_back(digits[bytes[i] >> 4]);
        s.push_back(digits[bytes[i] & 15]);
    }
    return s;
}

template <typename T>
struct TrtDelete
{
    void operator()(T* p) const { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

} // namespace

// Réseau TensorRT prêt à l'exécution : moteur, configuration, cache des noyaux, contexte et mémoire de travail.
struct TrtNet
{
    nvinfer1::IRuntime* runtime = nullptr;   // runtime propriétaire du moteur (doit lui survivre)
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IRuntimeConfig> runtime_config;
    TrtPtr<nvinfer1::IRuntimeCache> runtime_cache;
    TrtPtr<nvinfer1::IExecutionContext> context;
    CUdeviceptr workspace = 0;
    fs::path runtime_cache_path;
    bool first_inference = true;

    explicit operator bool() const { return static_cast<bool>(context); }

    void save_runtime_cache()
    {
        if (!runtime_cache || runtime_cache_path.empty())
            return;
        TrtPtr<nvinfer1::IHostMemory> blob(runtime_cache->serialize());
        if (blob && blob->size() > 0)
            write_file_atomic(runtime_cache_path, blob->data(), blob->size());
    }

    void release(Cuda& cu)
    {
        context.reset();
        runtime_config.reset();
        runtime_cache.reset();
        engine.reset();
        if (workspace)
            cu.cuMemFree(workspace);
        workspace = 0;
    }
};

// Caractéristiques des images (encodeur des familles v4.15), partagées par les sessions d'un même modèle et d'une
// même taille (inférence normale, demi-résolution, sens inverse) : l'encodeur n'est calculé qu'une fois par image
// tant que muxiveo-rife en transmet l'identité. Appels séquentiels (fil de muxiveo-rife), sans verrou interne.
struct FeatureStore
{
    struct Entry
    {
        uint64_t id = 0;
        CUdeviceptr buf = 0;
        uint64_t used = 0;
    };
    Cuda* cu = nullptr;
    CUcontext ctx = nullptr;
    Logger logger;
    TrtPtr<nvinfer1::IRuntime> runtime;   // propre : partagé au-delà de la session qui l'a créé
    TrtNet head;
    size_t bytes = 0;
    Entry cached[2];   // images de la paire courante (la suivante en réutilise une)
    Entry scratch[2];  // images sans identité : recalculées à chaque appel
    uint64_t tick = 0;

    ~FeatureStore()
    {
        if (!cu || !ctx || cu->cuCtxPushCurrent(ctx) != 0)
            return;
        head.release(*cu);
        runtime.reset();
        for (Entry* e : {&cached[0], &cached[1], &scratch[0], &scratch[1]})
            if (e->buf)
                cu->cuMemFree(e->buf);
        CUcontext popped = nullptr;
        cu->cuCtxPopCurrent(&popped);
    }
};

static std::mutex g_features_mutex;
static std::vector<std::pair<std::string, std::weak_ptr<FeatureStore>>> g_features;

struct MvoTrtSession
{
    Cuda* cu = nullptr;
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    Logger logger;
    mvo_trt_log_fn log = nullptr;
    void* log_user = nullptr;
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtNet full;                           // modèle complet (sans identités d'images, ou modèle sans encodeur)
    TrtNet body;                           // corps : caractéristiques fournies
    std::shared_ptr<FeatureStore> features;
    fs::path model_path;
    fs::path cache_dir;
    std::string device_suffix;             // -smXY-trtrtxV (moteurs) ; uuid ajouté pour les caches de noyaux
    std::string uuid_hex;
    CUdeviceptr timestep = 0;
    int width = 0;
    int height = 0;
    float last_t = -1.f;

    void info(const std::string& message) const
    {
        if (log)
            log(log_user, MVO_TRT_LOG_INFO, message.c_str());
    }

    ~MvoTrtSession()
    {
        if (!cu || !ctx)
            return;
        {
            ContextScope scope(*cu, ctx);
            full.release(*cu);
            body.release(*cu);
            features.reset();
            runtime.reset();
            if (timestep)
                cu->cuMemFree(timestep);
            if (stream)
                cu->cuStreamDestroy(stream);
        }
        cu->cuDevicePrimaryCtxRelease(dev);
    }
};

namespace {

const char* plugin_version()
{
    static const std::string v = std::string(MVO_RIFE_TRT_VERSION) + " (TensorRT-RTX " + trt_version() + ")";
    return v.c_str();
}

int device_supported(int cuda_device, char* error, size_t error_size)
{
    Cuda& c = cuda();
    if (!c.ok)
    {
        set_error(error, error_size, c.error);
        return 0;
    }
    CUdevice dev;
    if (c.cuDeviceGet(&dev, cuda_device) != 0)
    {
        set_error(error, error_size, "périphérique CUDA introuvable");
        return 0;
    }
    const int cc = compute_capability(c, dev);
    if (cc < kMinComputeCapability)
    {
        set_error(error, error_size,
                  "architecture " + std::to_string(cc / 10) + "." + std::to_string(cc % 10)
                      + " non prise en charge (Turing 7.5 ou plus récent requis)");
        return 0;
    }
    return 1;
}

// Moteur relu du cache, sinon construit depuis le modèle ONNX puis mis en cache.
bool load_or_build_engine(MvoTrtSession& s, TrtNet& n, const fs::path& model, const fs::path& engine_path,
                          std::string& error)
{
    std::vector<char> blob;
    if (read_file(engine_path, blob))
    {
        n.engine.reset(n.runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (n.engine)
            return true;
        s.info("moteur TensorRT en cache illisible : reconstruction");
    }

    std::vector<char> onnx;
    if (!read_file(model, onnx))
    {
        error = "modèle introuvable : " + model.u8string();
        return false;
    }
    s.info("construction du moteur TensorRT (première utilisation de ce modèle)");
    const auto t0 = std::chrono::steady_clock::now();
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(s.logger));
    if (!builder)
    {
        error = "constructeur TensorRT indisponible";
        return false;
    }
    // Réseau toujours typé strictement (TensorRT 11) : précision fp16 portée par le modèle ONNX.
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, s.logger));
    if (!network || !parser || !parser->parse(onnx.data(), onnx.size()))
    {
        error = "modèle ONNX refusé : " + s.logger.take_error();
        return false;
    }
    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // dimensions dynamiques (hauteur, largeur) pour chaque entrée ; canaux fixés par le modèle
    for (int i = 0; i < network->getNbInputs(); i++)
    {
        nvinfer1::ITensor* in = network->getInput(i);
        const int c = in->getDimensions().d[1];
        profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, c, kMinSide, kMinSide});
        profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, c, s.height, s.width});
        profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, c, kMaxSide, kMaxSide});
    }
    config->addOptimizationProfile(profile);
    config->setNbComputeCapabilities(1);
    config->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT, 0);
    TrtPtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized || serialized->size() == 0)
    {
        error = "construction du moteur TensorRT impossible : " + s.logger.take_error();
        return false;
    }
    if (!write_file_atomic(engine_path, serialized->data(), serialized->size()))
        s.info("moteur TensorRT non mis en cache (dossier en lecture seule ?)");
    n.engine.reset(n.runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!n.engine)
    {
        error = "moteur TensorRT illisible : " + s.logger.take_error();
        return false;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char line[96];
    std::snprintf(line, sizeof line, "moteur TensorRT construit en %.1f s (mis en cache)", seconds);
    s.info(line);
    return true;
}

// Réseau prêt : moteur (cache ou construction), configuration d'exécution, cache des noyaux, formes des entrées
// (1 × canaux × hauteur × largeur) et mémoire de travail.
bool prepare_net(MvoTrtSession& s, TrtNet& n, const fs::path& model, std::string& error)
{
    const std::string stem = model.stem().u8string() + s.device_suffix;
    n.runtime_cache_path = s.cache_dir / (stem + "-" + s.uuid_hex + ".rtcache");
    if (!load_or_build_engine(s, n, model, s.cache_dir / (stem + ".engine"), error))
        return false;
    n.runtime_config.reset(n.engine->createRuntimeConfig());
    if (!n.runtime_config)
    {
        error = "configuration d'exécution TensorRT impossible";
        return false;
    }
    n.runtime_config->setExecutionContextAllocationStrategy(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    n.runtime_config->setDynamicShapesKernelSpecializationStrategy(nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
    n.runtime_cache.reset(n.runtime_config->createRuntimeCache());
    if (n.runtime_cache)
    {
        std::vector<char> blob;
        // Cache validé par TensorRT (version, GPU, pilote) ; refusé = recompilation transparente.
        if (read_file(n.runtime_cache_path, blob))
            n.runtime_cache->deserialize(blob.data(), blob.size());
        n.runtime_config->setRuntimeCache(*n.runtime_cache);
    }
    n.context.reset(n.engine->createExecutionContext(n.runtime_config.get()));
    if (!n.context)
    {
        error = "contexte d'exécution TensorRT impossible : " + s.logger.take_error();
        return false;
    }
    for (int i = 0; i < n.engine->getNbIOTensors(); i++)
    {
        const char* name = n.engine->getIOTensorName(i);
        if (n.engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        const int c = n.engine->getTensorShape(name).d[1];
        if (!n.context->setInputShape(name, nvinfer1::Dims4{1, c, s.height, s.width}))
        {
            error = "dimensions refusées par le moteur : " + s.logger.take_error();
            return false;
        }
    }
    const size_t workspace_size = n.context->updateDeviceMemorySizeForShapes();
    if (workspace_size > 0)
    {
        if (s.cu->cuMemAlloc(&n.workspace, workspace_size) != 0)
        {
            n.workspace = 0;
            error = "mémoire GPU insuffisante (TensorRT)";
            return false;
        }
        n.context->setDeviceMemoryV2(reinterpret_cast<void*>(n.workspace), static_cast<int64_t>(workspace_size));
    }
    return true;
}

// Exécution synchrone d'un réseau sur le flux de la session ; première exécution : noyaux spécialisés mis en cache.
bool run_net(MvoTrtSession& s, TrtNet& n, std::string& error)
{
    const auto t0 = std::chrono::steady_clock::now();
    if (!n.context->enqueueV3(reinterpret_cast<cudaStream_t>(s.stream)) || s.cu->cuStreamSynchronize(s.stream) != 0)
    {
        error = "échec d'inférence TensorRT : " + s.logger.take_error();
        return false;
    }
    if (n.first_inference)
    {
        n.first_inference = false;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (seconds > 1.0)
        {
            char line[112];
            std::snprintf(line, sizeof line, "noyaux TensorRT spécialisés pour %dx%d en %.1f s (mis en cache)", s.width,
                          s.height, seconds);
            s.info(line);
        }
        n.save_runtime_cache();
    }
    return true;
}

// Caractéristiques partagées du modèle (encodeur « <base>-head.onnx » à côté du modèle, base sans « -uhd »), créées
// au besoin ; nullptr si le modèle n'a pas d'encodeur séparé.
std::shared_ptr<FeatureStore> shared_features(MvoTrtSession& s, std::string& error)
{
    std::string base = s.model_path.stem().u8string();
    if (base.size() > 4 && base.compare(base.size() - 4, 4, "-uhd") == 0)
        base.resize(base.size() - 4);
    const fs::path head = s.model_path.parent_path() / (base + "-head.onnx");
    std::error_code ec;
    if (!fs::is_regular_file(head, ec))
        return nullptr;
    const std::string key = head.u8string() + "|" + std::to_string(s.width) + "x" + std::to_string(s.height) + "|"
                            + std::to_string(static_cast<int>(s.dev));
    std::lock_guard<std::mutex> lock(g_features_mutex);
    for (auto it = g_features.begin(); it != g_features.end();)
    {
        if (auto f = it->second.lock())
        {
            if (it->first == key)
                return f;
            ++it;
        }
        else
            it = g_features.erase(it);
    }
    auto f = std::make_shared<FeatureStore>();
    f->cu = s.cu;
    f->ctx = s.ctx;
    f->runtime.reset(nvinfer1::createInferRuntime(f->logger));
    if (!f->runtime)
    {
        error = "runtime TensorRT indisponible (encodeur)";
        return nullptr;
    }
    f->head.runtime = f->runtime.get();
    if (!prepare_net(s, f->head, head, error))
        return nullptr;
    const nvinfer1::Dims out = f->head.context->getTensorShape("f0");
    if (out.nbDims != 4 || out.d[1] <= 0 || out.d[2] != s.height || out.d[3] != s.width)
    {
        error = "encodeur : sortie f0 inattendue";
        return nullptr;
    }
    f->bytes = static_cast<size_t>(out.d[1]) * out.d[2] * out.d[3] * 2;
    g_features.emplace_back(key, f);
    return f;
}

// Caractéristiques de l'image `in` d'identité `id` (0 : inconnue, emplacement temporaire `scratch`), calculées par
// l'encodeur sur le flux de la session si elles ne sont pas déjà connues. `keep` : entrée à ne pas évincer.
CUdeviceptr image_features(MvoTrtSession& s, uint64_t id, CUdeviceptr in, int scratch, const FeatureStore::Entry* keep,
                           FeatureStore::Entry*& entry, std::string& error)
{
    FeatureStore& f = *s.features;
    entry = nullptr;
    if (id)
    {
        for (FeatureStore::Entry& e : f.cached)
            if (e.id == id && e.buf)
            {
                e.used = ++f.tick;
                entry = &e;
                return e.buf;
            }
        for (FeatureStore::Entry& e : f.cached)
            if (&e != keep && (!entry || e.used < entry->used))
                entry = &e;
    }
    else
        entry = &f.scratch[scratch];
    if (!entry->buf && s.cu->cuMemAlloc(&entry->buf, f.bytes) != 0)
    {
        entry->buf = 0;
        error = "mémoire GPU insuffisante (caractéristiques TensorRT)";
        return 0;
    }
    entry->id = 0;   // invalide tant que l'encodeur n'a pas réussi
    if (!f.head.context->setTensorAddress("in0", reinterpret_cast<void*>(in))
        || !f.head.context->setTensorAddress("f0", reinterpret_cast<void*>(entry->buf)) || !run_net(s, f.head, error))
    {
        if (error.empty())
            error = "liaison des tenseurs impossible (encodeur) : " + s.logger.take_error();
        return 0;
    }
    entry->id = id;
    entry->used = ++f.tick;
    return entry->buf;
}

MvoTrtSession* create(const MvoTrtCreateParams* p, char* error, size_t error_size)
{
    if (!p || p->struct_size < sizeof(MvoTrtCreateParams) || !p->model_path || !p->cache_dir || p->width < kMinSide
        || p->height < kMinSide || p->width > kMaxSide || p->height > kMaxSide)
    {
        set_error(error, error_size, "paramètres de session invalides");
        return nullptr;
    }
    if (!device_supported(p->cuda_device, error, error_size))
        return nullptr;

    auto s = std::make_unique<MvoTrtSession>();
    s->cu = &cuda();
    s->log = p->log;
    s->log_user = p->log_user;
    s->width = p->width;
    s->height = p->height;
    Cuda& c = *s->cu;
    if (c.cuDeviceGet(&s->dev, p->cuda_device) != 0 || c.cuDevicePrimaryCtxRetain(&s->ctx, s->dev) != 0)
    {
        set_error(error, error_size, "contexte CUDA indisponible");
        s->ctx = nullptr;
        return nullptr;
    }
    ContextScope scope(c, s->ctx);
    if (!scope.ok())
    {
        set_error(error, error_size, "contexte CUDA indisponible");
        return nullptr;
    }

    try
    {
        s->model_path = fs::u8path(p->model_path);
        s->cache_dir = fs::u8path(p->cache_dir);
        std::error_code ec;
        fs::create_directories(s->cache_dir, ec);
        const int cc = compute_capability(c, s->dev);
        CUuuid uuid{};
        c.cuDeviceGetUuid(&uuid, s->dev);
        s->device_suffix = "-sm" + std::to_string(cc) + "-trtrtx" + trt_version();
        s->uuid_hex = hex(uuid.bytes, 16);

        s->runtime.reset(nvinfer1::createInferRuntime(s->logger));
        if (!s->runtime)
        {
            set_error(error, error_size, "runtime TensorRT indisponible");
            return nullptr;
        }
        s->full.runtime = s->body.runtime = s->runtime.get();
        std::string err;
        // Encodeur séparé (familles v4.15) : corps + caractéristiques partagées ; sinon modèle complet.
        const fs::path body = s->model_path.parent_path() / (s->model_path.stem().u8string() + "-body.onnx");
        if (fs::is_regular_file(body, ec))
        {
            s->features = shared_features(*s, err);
            if (s->features && !prepare_net(*s, s->body, body, err))
            {
                s->body.release(c);
                s->features.reset();
            }
            if (!s->features)
            {
                s->info("encodeur séparé indisponible (" + err + ") : modèle complet");
                err.clear();
            }
            else
                s->info("caractéristiques calculées une fois par image (encodeur séparé)");
        }
        if (!s->features && !prepare_net(*s, s->full, s->model_path, err))
        {
            set_error(error, error_size, err);
            return nullptr;
        }
        if (c.cuMemAlloc(&s->timestep, static_cast<size_t>(s->width) * s->height * 2) != 0)
        {
            s->timestep = 0;
            set_error(error, error_size, "mémoire GPU insuffisante (TensorRT)");
            return nullptr;
        }
        if (c.cuStreamCreate(&s->stream, CU_STREAM_NON_BLOCKING) != 0)
        {
            s->stream = nullptr;
            set_error(error, error_size, "flux CUDA impossible");
            return nullptr;
        }
    }
    catch (const std::exception& e)
    {
        set_error(error, error_size, std::string("erreur interne : ") + e.what());
        return nullptr;
    }
    return s.release();
}

int infer(MvoTrtSession* s, const MvoTrtInferParams* p, char* error, size_t error_size)
{
    if (!s || !p || p->struct_size < offsetof(MvoTrtInferParams, id0) || !p->in0 || !p->in1 || !p->out)
    {
        set_error(error, error_size, "paramètres d'inférence invalides");
        return 0;
    }
    Cuda& c = *s->cu;
    ContextScope scope(c, s->ctx);
    if (!scope.ok())
    {
        set_error(error, error_size, "contexte CUDA indisponible");
        return 0;
    }
    if (p->t != s->last_t)
    {
        if (c.cuMemsetD16Async(s->timestep, to_half(p->t), static_cast<size_t>(s->width) * s->height, s->stream) != 0)
        {
            set_error(error, error_size, "écriture de la carte de temps impossible");
            return 0;
        }
        s->last_t = p->t;
    }
    std::string err;
    // Identités des images transmises (muxiveo-rife ≥ 1.9.0) : encodeur une fois par image ; sinon modèle complet,
    // préparé à la première demande.
    const bool ids = p->struct_size >= sizeof(MvoTrtInferParams);
    if (s->features && ids)
    {
        FeatureStore::Entry* e0 = nullptr;
        FeatureStore::Entry* e1 = nullptr;
        const CUdeviceptr f0 = image_features(*s, p->id0, p->in0, 0, nullptr, e0, err);
        const CUdeviceptr f1 = f0 ? image_features(*s, p->id1, p->in1, 1, e0, e1, err) : 0;
        if (!f0 || !f1)
        {
            set_error(error, error_size, err);
            return 0;
        }
        if (!s->body.context->setTensorAddress("in0", reinterpret_cast<void*>(p->in0))
            || !s->body.context->setTensorAddress("in1", reinterpret_cast<void*>(p->in1))
            || !s->body.context->setTensorAddress("f0", reinterpret_cast<void*>(f0))
            || !s->body.context->setTensorAddress("f1", reinterpret_cast<void*>(f1))
            || !s->body.context->setTensorAddress("in2", reinterpret_cast<void*>(s->timestep))
            || !s->body.context->setTensorAddress("out0", reinterpret_cast<void*>(p->out)))
        {
            set_error(error, error_size, "liaison des tenseurs impossible : " + s->logger.take_error());
            return 0;
        }
        if (!run_net(*s, s->body, err))
        {
            set_error(error, error_size, err);
            return 0;
        }
        return 1;
    }
    if (!s->full)
    {
        try
        {
            if (!prepare_net(*s, s->full, s->model_path, err))
            {
                s->full.release(c);
                set_error(error, error_size, err);
                return 0;
            }
        }
        catch (const std::exception& e)
        {
            set_error(error, error_size, std::string("erreur interne : ") + e.what());
            return 0;
        }
    }
    if (!s->full.context->setTensorAddress("in0", reinterpret_cast<void*>(p->in0))
        || !s->full.context->setTensorAddress("in1", reinterpret_cast<void*>(p->in1))
        || !s->full.context->setTensorAddress("in2", reinterpret_cast<void*>(s->timestep))
        || !s->full.context->setTensorAddress("out0", reinterpret_cast<void*>(p->out)))
    {
        set_error(error, error_size, "liaison des tenseurs impossible : " + s->logger.take_error());
        return 0;
    }
    if (!run_net(*s, s->full, err))
    {
        set_error(error, error_size, err);
        return 0;
    }
    return 1;
}

void destroy(MvoTrtSession* s)
{
    delete s;
}

const MvoTrtApi kApi = {MVO_TRT_ABI_VERSION, plugin_version, device_supported, create, infer, destroy};

} // namespace

extern "C" MVO_TRT_EXPORT const MvoTrtApi* mvo_trt_get_api(uint32_t abi_version)
{
    return abi_version == MVO_TRT_ABI_VERSION ? &kApi : nullptr;
}
