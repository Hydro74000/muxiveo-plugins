// Muxiveo — flux optique matériel NVIDIA (NVOFA), chargement dynamique. Voir nvof.h.
// En-têtes de structures : NVIDIA Optical Flow SDK (third_party/nvof, licence BSD 3 clauses).

#include "nvof.h"

#include <cstring>

#if _WIN32
#include <windows.h>
#define NVOFAPI_CALL __stdcall
#else
#include <dlfcn.h>
#define NVOFAPI_CALL
#endif

typedef struct NV_OF_ROI_RECT NV_OF_ROI_RECT;  // en-tête prévu pour C++ : déclaration explicite
#include "third_party/nvof/nvOpticalFlowCommon.h"

namespace {

// Sous-ensemble de l'API pilote CUDA (cuda.h non requis).
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef struct CUarray_st* CUarray;
typedef unsigned long long CUdeviceptr;
enum { CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2 };
struct CUDA_MEMCPY2D
{
    size_t srcXInBytes; size_t srcY; int srcMemoryType; const void* srcHost;
    CUdeviceptr srcDevice; CUarray srcArray; size_t srcPitch;
    size_t dstXInBytes; size_t dstY; int dstMemoryType; void* dstHost;
    CUdeviceptr dstDevice; CUarray dstArray; size_t dstPitch;
    size_t WidthInBytes; size_t Height;
};

// Table de fonctions NVOF (interface CUDA), même disposition que nvOpticalFlowCuda.h.
enum { NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2 };
struct NV_OF_BUFFER_STRIDE { uint32_t strideXInBytes; uint32_t strideYInBytes; };
struct NV_OF_CUDA_BUFFER_STRIDE_INFO { NV_OF_BUFFER_STRIDE strideInfo[3]; uint32_t numPlanes; };
struct NV_OF_CUDA_API_FUNCTION_LIST
{
    NV_OF_STATUS (NVOFAPI_CALL* nvCreateOpticalFlowCuda)(CUcontext, NvOFHandle*);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFInit)(NvOFHandle, const NV_OF_INIT_PARAMS*);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFCreateGPUBufferCuda)(NvOFHandle, const NV_OF_BUFFER_DESCRIPTOR*, int, NvOFGPUBufferHandle*);
    CUarray (NVOFAPI_CALL* nvOFGPUBufferGetCUarray)(NvOFGPUBufferHandle);
    CUdeviceptr (NVOFAPI_CALL* nvOFGPUBufferGetCUdeviceptr)(NvOFGPUBufferHandle);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFGPUBufferGetStrideInfo)(NvOFGPUBufferHandle, NV_OF_CUDA_BUFFER_STRIDE_INFO*);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFSetIOCudaStreams)(NvOFHandle, CUstream, CUstream);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFExecute)(NvOFHandle, const NV_OF_EXECUTE_INPUT_PARAMS*, NV_OF_EXECUTE_OUTPUT_PARAMS*);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFDestroyGPUBufferCuda)(NvOFGPUBufferHandle);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFDestroy)(NvOFHandle);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFGetLastError)(NvOFHandle, char*, uint32_t*);
    NV_OF_STATUS (NVOFAPI_CALL* nvOFGetCaps)(NvOFHandle, NV_OF_CAPS, uint32_t*, uint32_t*);
};

void* open_lib(const char* name)
{
#if _WIN32
    return (void*)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* find_sym(void* lib, const char* name)
{
#if _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

void close_lib(void* lib)
{
    if (!lib)
        return;
#if _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

} // namespace

struct NvofFlow::Impl
{
    void* cuda_lib = 0;
    void* nvof_lib = 0;
    CUresult (*cuInit)(unsigned) = 0;
    CUresult (*cuDeviceGetCount)(int*) = 0;
    CUresult (*cuDeviceGet)(CUdevice*, int) = 0;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = 0;
    CUresult (*cuCtxCreate)(CUcontext*, unsigned, CUdevice) = 0;
    CUresult (*cuCtxDestroy)(CUcontext) = 0;
    CUresult (*cuCtxSetCurrent)(CUcontext) = 0;
    CUresult (*cuCtxSynchronize)() = 0;
    CUresult (*cuMemcpy2D)(const CUDA_MEMCPY2D*) = 0;
    NV_OF_CUDA_API_FUNCTION_LIST f;
    CUcontext ctx = 0;
    struct Session
    {
        NvOFHandle h = 0;
        NvOFGPUBufferHandle in[2] = {0, 0};
        NvOFGPUBufferHandle out = 0;
    } fwd, bwd;

    bool copy(NvOFGPUBufferHandle b, void* host, size_t rowbytes, size_t rows, bool up)
    {
        NV_OF_CUDA_BUFFER_STRIDE_INFO si;
        if (f.nvOFGPUBufferGetStrideInfo(b, &si) != NV_OF_SUCCESS)
            return false;
        CUDA_MEMCPY2D c;
        memset(&c, 0, sizeof c);
        const CUdeviceptr dev = f.nvOFGPUBufferGetCUdeviceptr(b);
        if (up)
        {
            c.srcMemoryType = CU_MEMORYTYPE_HOST; c.srcHost = host; c.srcPitch = rowbytes;
            c.dstMemoryType = CU_MEMORYTYPE_DEVICE; c.dstDevice = dev; c.dstPitch = si.strideInfo[0].strideXInBytes;
        }
        else
        {
            c.srcMemoryType = CU_MEMORYTYPE_DEVICE; c.srcDevice = dev; c.srcPitch = si.strideInfo[0].strideXInBytes;
            c.dstMemoryType = CU_MEMORYTYPE_HOST; c.dstHost = host; c.dstPitch = rowbytes;
        }
        c.WidthInBytes = rowbytes;
        c.Height = rows;
        return cuMemcpy2D(&c) == 0;
    }

    bool open_session(Session& s, int w, int h, int grid)
    {
        if (f.nvCreateOpticalFlowCuda(ctx, &s.h) != NV_OF_SUCCESS)
            return false;
        NV_OF_INIT_PARAMS ip;
        memset(&ip, 0, sizeof ip);
        ip.width = (uint32_t)w;
        ip.height = (uint32_t)h;
        ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)grid;
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = NV_OF_PERF_LEVEL_SLOW;
        if (f.nvOFInit(s.h, &ip) != NV_OF_SUCCESS)
            return false;
        NV_OF_BUFFER_DESCRIPTOR in = {(uint32_t)w, (uint32_t)h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8};
        NV_OF_BUFFER_DESCRIPTOR out = {(uint32_t)((w + grid - 1) / grid), (uint32_t)((h + grid - 1) / grid), NV_OF_BUFFER_USAGE_OUTPUT,
                                       NV_OF_BUFFER_FORMAT_SHORT2};
        for (int i = 0; i < 2; i++)
            if (f.nvOFCreateGPUBufferCuda(s.h, &in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &s.in[i]) != NV_OF_SUCCESS)
                return false;
        return f.nvOFCreateGPUBufferCuda(s.h, &out, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &s.out) == NV_OF_SUCCESS;
    }

    void close_session(Session& s)
    {
        if (!s.h)
            return;
        for (NvOFGPUBufferHandle b : {s.in[0], s.in[1], s.out})
            if (b)
                f.nvOFDestroyGPUBufferCuda(b);
        f.nvOFDestroy(s.h);
        s = Session();
    }

    bool run(Session& s, const uint8_t* a, const uint8_t* b, int w, int h, int grid, int16_t* flow)
    {
        if (!copy(s.in[0], (void*)a, (size_t)w, (size_t)h, true) || !copy(s.in[1], (void*)b, (size_t)w, (size_t)h, true))
            return false;
        NV_OF_EXECUTE_INPUT_PARAMS ei;
        memset(&ei, 0, sizeof ei);
        ei.inputFrame = s.in[0];
        ei.referenceFrame = s.in[1];
        ei.disableTemporalHints = NV_OF_FALSE;  // flux successifs d'une même vidéo
        NV_OF_EXECUTE_OUTPUT_PARAMS eo;
        memset(&eo, 0, sizeof eo);
        eo.outputBuffer = s.out;
        if (f.nvOFExecute(s.h, &ei, &eo) != NV_OF_SUCCESS || cuCtxSynchronize() != 0)
            return false;
        return copy(s.out, flow, (size_t)((w + grid - 1) / grid) * 4, (size_t)((h + grid - 1) / grid), false);
    }
};

NvofFlow::NvofFlow() : d(0), ready(false), w(0), h(0)
{
}

NvofFlow::~NvofFlow()
{
    if (!d)
        return;
    if (d->ctx)
    {
        d->cuCtxSetCurrent(d->ctx);
        d->close_session(d->fwd);
        d->close_session(d->bwd);
        d->cuCtxDestroy(d->ctx);
    }
    close_lib(d->nvof_lib);
    close_lib(d->cuda_lib);
    delete d;
}

bool NvofFlow::init(int width, int height, const std::string& device_name, int rank, std::string& error)
{
    d = new Impl;
#if _WIN32
    d->cuda_lib = open_lib("nvcuda.dll");
    d->nvof_lib = open_lib("nvofapi64.dll");
#else
    d->cuda_lib = open_lib("libcuda.so.1");
    d->nvof_lib = open_lib("libnvidia-opticalflow.so.1");
#endif
    if (!d->cuda_lib || !d->nvof_lib)
    {
        error = "pilote NVIDIA (CUDA / Optical Flow) absent";
        return false;
    }
#define LOAD(field, name) d->field = (decltype(d->field))find_sym(d->cuda_lib, name)
    LOAD(cuInit, "cuInit");
    LOAD(cuDeviceGetCount, "cuDeviceGetCount");
    LOAD(cuDeviceGet, "cuDeviceGet");
    LOAD(cuDeviceGetName, "cuDeviceGetName");
    LOAD(cuCtxCreate, "cuCtxCreate_v2");
    LOAD(cuCtxDestroy, "cuCtxDestroy_v2");
    LOAD(cuCtxSetCurrent, "cuCtxSetCurrent");
    LOAD(cuCtxSynchronize, "cuCtxSynchronize");
    LOAD(cuMemcpy2D, "cuMemcpy2D_v2");
#undef LOAD
    typedef NV_OF_STATUS (NVOFAPI_CALL * CreateFn)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);
    CreateFn create = (CreateFn)find_sym(d->nvof_lib, "NvOFAPICreateInstanceCuda");
    if (!d->cuInit || !d->cuDeviceGetCount || !d->cuDeviceGet || !d->cuDeviceGetName || !d->cuCtxCreate
            || !d->cuCtxDestroy || !d->cuCtxSetCurrent || !d->cuCtxSynchronize || !d->cuMemcpy2D || !create)
    {
        error = "symboles CUDA / Optical Flow introuvables";
        return false;
    }
    memset(&d->f, 0, sizeof d->f);
    if (create(NV_OF_API_VERSION, &d->f) != NV_OF_SUCCESS || !d->f.nvCreateOpticalFlowCuda)
    {
        error = "API Optical Flow incompatible avec le pilote";
        return false;
    }
    int count = 0;
    if (d->cuInit(0) != 0 || d->cuDeviceGetCount(&count) != 0 || count <= 0)
    {
        error = "aucun GPU CUDA";
        return false;
    }
    // appariement avec le GPU Vulkan : même nom, puis rang parmi les homonymes
    CUdevice chosen = -1;
    int seen = 0;
    for (int i = 0; i < count && chosen < 0; i++)
    {
        CUdevice dev;
        char name[256] = {0};
        if (d->cuDeviceGet(&dev, i) != 0 || d->cuDeviceGetName(name, sizeof name, dev) != 0)
            continue;
        if (device_name == name && seen++ == rank)
        {
            chosen = dev;
            cuda_name = name;
        }
    }
    if (chosen < 0)
    {
        error = "GPU Vulkan « " + device_name + " » introuvable côté CUDA";
        return false;
    }
    if (d->cuCtxCreate(&d->ctx, 0, chosen) != 0)
    {
        error = "création du contexte CUDA impossible";
        return false;
    }
    // demi-résolution + grille 2 px, sinon pleine résolution + grille 4 px (Turing)
    static const int configs[2][2] = {{2, 2}, {1, 4}};
    bool opened = false;
    for (const auto& c : configs)
    {
        scale_ = c[0];
        grid_ = c[1];
        w = (width + scale_ - 1) / scale_;
        h = (height + scale_ - 1) / scale_;
        if (d->open_session(d->fwd, w, h, grid_) && d->open_session(d->bwd, w, h, grid_))
        {
            opened = true;
            break;
        }
        d->close_session(d->fwd);
        d->close_session(d->bwd);
    }
    if (!opened)
    {
        error = "session Optical Flow refusée (GPU sans NVOFA ou dimensions non prises en charge)";
        return false;
    }
    ready = true;
    return true;
}

bool NvofFlow::compute(const uint8_t* y0, const uint8_t* y1, std::vector<int16_t>& fwd, std::vector<int16_t>& bwd,
                       std::string& error)
{
    if (!ready)
        return false;
    d->cuCtxSetCurrent(d->ctx);
    fwd.resize((size_t)grid_w() * grid_h() * 2);
    bwd.resize(fwd.size());
    if (!d->run(d->fwd, y0, y1, w, h, grid(), fwd.data()) || !d->run(d->bwd, y1, y0, w, h, grid(), bwd.data()))
    {
        error = "échec d'exécution Optical Flow";
        return false;
    }
    return true;
}
