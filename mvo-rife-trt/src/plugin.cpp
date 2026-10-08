// mvo-rife-trt — inférence RIFE de muxiveo-rife avec NVIDIA TensorRT for RTX.
//
// Le plugin reçoit des tampons CUDA déjà remplis par muxiveo-rife (mémoire partagée avec Vulkan) et écrit
// l'image interpolée dans le tampon de sortie. Interface : include/trt_plugin_abi.h.
// Le pilote CUDA est chargé à l'exécution ; TensorRT for RTX est livré à côté du plugin.

#include "trt_plugin_abi.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IRuntimeConfig> runtime_config;
    TrtPtr<nvinfer1::IRuntimeCache> runtime_cache;
    TrtPtr<nvinfer1::IExecutionContext> context;
    CUdeviceptr timestep = 0;
    CUdeviceptr workspace = 0;
    int width = 0;
    int height = 0;
    float last_t = -1.f;
    bool first_inference = true;
    fs::path runtime_cache_path;

    void info(const std::string& message) const
    {
        if (log)
            log(log_user, MVO_TRT_LOG_INFO, message.c_str());
    }

    void save_runtime_cache()
    {
        if (!runtime_cache || runtime_cache_path.empty())
            return;
        TrtPtr<nvinfer1::IHostMemory> blob(runtime_cache->serialize());
        if (blob && blob->size() > 0)
            write_file_atomic(runtime_cache_path, blob->data(), blob->size());
    }

    ~MvoTrtSession()
    {
        if (!cu || !ctx)
            return;
        {
            ContextScope scope(*cu, ctx);
            context.reset();
            runtime_config.reset();
            runtime_cache.reset();
            engine.reset();
            runtime.reset();
            if (workspace)
                cu->cuMemFree(workspace);
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
bool load_or_build_engine(MvoTrtSession& s, const fs::path& model, const fs::path& engine_path, std::string& error)
{
    std::vector<char> blob;
    if (read_file(engine_path, blob))
    {
        s.engine.reset(s.runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (s.engine)
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
    for (const char* name : {"in0", "in1", "in2"})
    {
        const int c = std::strcmp(name, "in2") == 0 ? 1 : 3;
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, c, kMinSide, kMinSide});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, c, s.height, s.width});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, c, kMaxSide, kMaxSide});
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
    s.engine.reset(s.runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!s.engine)
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
        const fs::path model = fs::u8path(p->model_path);
        const fs::path cache_dir = fs::u8path(p->cache_dir);
        std::error_code ec;
        fs::create_directories(cache_dir, ec);
        const int cc = compute_capability(c, s->dev);
        CUuuid uuid{};
        c.cuDeviceGetUuid(&uuid, s->dev);
        const std::string stem = model.stem().u8string() + "-sm" + std::to_string(cc) + "-trtrtx" + trt_version();
        s->runtime_cache_path = cache_dir / (stem + "-" + hex(uuid.bytes, 16) + ".rtcache");

        s->runtime.reset(nvinfer1::createInferRuntime(s->logger));
        if (!s->runtime)
        {
            set_error(error, error_size, "runtime TensorRT indisponible");
            return nullptr;
        }
        std::string err;
        if (!load_or_build_engine(*s, model, cache_dir / (stem + ".engine"), err))
        {
            set_error(error, error_size, err);
            return nullptr;
        }

        s->runtime_config.reset(s->engine->createRuntimeConfig());
        if (!s->runtime_config)
        {
            set_error(error, error_size, "configuration d'exécution TensorRT impossible");
            return nullptr;
        }
        s->runtime_config->setExecutionContextAllocationStrategy(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
        s->runtime_config->setDynamicShapesKernelSpecializationStrategy(nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
        s->runtime_cache.reset(s->runtime_config->createRuntimeCache());
        if (s->runtime_cache)
        {
            std::vector<char> blob;
            // Cache validé par TensorRT (version, GPU, pilote) ; refusé = recompilation transparente.
            if (read_file(s->runtime_cache_path, blob))
                s->runtime_cache->deserialize(blob.data(), blob.size());
            s->runtime_config->setRuntimeCache(*s->runtime_cache);
        }
        s->context.reset(s->engine->createExecutionContext(s->runtime_config.get()));
        if (!s->context)
        {
            set_error(error, error_size, "contexte d'exécution TensorRT impossible : " + s->logger.take_error());
            return nullptr;
        }
        const nvinfer1::Dims4 rgb{1, 3, s->height, s->width};
        const nvinfer1::Dims4 map{1, 1, s->height, s->width};
        if (!s->context->setInputShape("in0", rgb) || !s->context->setInputShape("in1", rgb)
            || !s->context->setInputShape("in2", map))
        {
            set_error(error, error_size, "dimensions refusées par le moteur : " + s->logger.take_error());
            return nullptr;
        }
        const size_t workspace_size = s->context->updateDeviceMemorySizeForShapes();
        if (workspace_size > 0)
        {
            if (c.cuMemAlloc(&s->workspace, workspace_size) != 0)
            {
                s->workspace = 0;
                set_error(error, error_size, "mémoire GPU insuffisante (TensorRT)");
                return nullptr;
            }
            s->context->setDeviceMemoryV2(reinterpret_cast<void*>(s->workspace), static_cast<int64_t>(workspace_size));
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
    if (!s || !p || p->struct_size < sizeof(MvoTrtInferParams) || !p->in0 || !p->in1 || !p->out)
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
    if (!s->context->setTensorAddress("in0", reinterpret_cast<void*>(p->in0))
        || !s->context->setTensorAddress("in1", reinterpret_cast<void*>(p->in1))
        || !s->context->setTensorAddress("in2", reinterpret_cast<void*>(s->timestep))
        || !s->context->setTensorAddress("out0", reinterpret_cast<void*>(p->out)))
    {
        set_error(error, error_size, "liaison des tenseurs impossible : " + s->logger.take_error());
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!s->context->enqueueV3(reinterpret_cast<cudaStream_t>(s->stream)) || c.cuStreamSynchronize(s->stream) != 0)
    {
        set_error(error, error_size, "échec d'inférence TensorRT : " + s->logger.take_error());
        return 0;
    }
    if (s->first_inference)
    {
        s->first_inference = false;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (seconds > 1.0)
        {
            char line[112];
            std::snprintf(line, sizeof line, "noyaux TensorRT spécialisés pour %dx%d en %.1f s (mis en cache)", s->width,
                          s->height, seconds);
            s->info(line);
        }
        s->save_runtime_cache();
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
