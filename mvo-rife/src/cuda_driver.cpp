// Muxiveo — pilote CUDA chargé à l'exécution. Voir cuda_driver.h.

#include "cuda_driver.h"

#include <cstring>

#if _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

enum
{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CUDA_EXTERNAL_MEMORY_DEDICATED = 1
};

// Disposition de cuda.h (CUDA_EXTERNAL_MEMORY_HANDLE_DESC / _BUFFER_DESC).
struct ExternalMemoryHandleDesc
{
    int type;
    union
    {
        int fd;
        struct
        {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

struct ExternalMemoryBufferDesc
{
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

void* find_sym(void* lib, const char* name)
{
#if _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

} // namespace

CudaDriver& CudaDriver::get()
{
    static CudaDriver driver;
    return driver;
}

CudaDriver::CudaDriver()
{
#if _WIN32
    lib = (void*)LoadLibraryA("nvcuda.dll");
#else
    lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!lib)
    {
        load_error = "pilote NVIDIA (CUDA) absent";
        return;
    }
    bool all = true;
#define LOAD(field, name) \
    field = (decltype(field))find_sym(lib, name); \
    all = all && field != nullptr
    LOAD(cuInit, "cuInit");
    LOAD(cuDriverGetVersion, "cuDriverGetVersion");
    LOAD(cuDeviceGetCount, "cuDeviceGetCount");
    LOAD(cuDeviceGet, "cuDeviceGet");
    LOAD(cuDeviceGetName, "cuDeviceGetName");
    LOAD(cuDeviceGetUuid, "cuDeviceGetUuid");
    LOAD(cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    LOAD(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    LOAD(cuCtxPushCurrent, "cuCtxPushCurrent_v2");
    LOAD(cuCtxPopCurrent, "cuCtxPopCurrent_v2");
    LOAD(cuImportExternalMemory, "cuImportExternalMemory");
    LOAD(cuExternalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer");
    LOAD(cuDestroyExternalMemory, "cuDestroyExternalMemory");
    LOAD(cuMemFree, "cuMemFree_v2");
#undef LOAD
    if (!all)
    {
        load_error = "pilote NVIDIA trop ancien (fonctions CUDA manquantes)";
        return;
    }
    if (cuInit(0) != 0)
    {
        load_error = "initialisation CUDA impossible";
        return;
    }
    cuDriverGetVersion(&version);
    loaded = true;
}

std::vector<CudaDeviceInfo> CudaDriver::devices() const
{
    std::vector<CudaDeviceInfo> list;
    int count = 0;
    if (!loaded || cuDeviceGetCount(&count) != 0)
        return list;
    for (int i = 0; i < count; i++)
    {
        CUdevice dev;
        if (cuDeviceGet(&dev, i) != 0)
            continue;
        CudaDeviceInfo info;
        info.ordinal = i;
        char name[256] = {0};
        cuDeviceGetName(name, sizeof name, dev);
        info.name = name;
        cuDeviceGetUuid(info.uuid, dev);
        int major = 0;
        int minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        info.compute_capability = major * 10 + minor;
        list.push_back(info);
    }
    return list;
}

bool CudaDriver::retain_primary(int ordinal, CUcontext& ctx) const
{
    CUdevice dev;
    return loaded && cuDeviceGet(&dev, ordinal) == 0 && cuDevicePrimaryCtxRetain(&ctx, dev) == 0;
}

void CudaDriver::release_primary(int ordinal) const
{
    CUdevice dev;
    if (loaded && cuDeviceGet(&dev, ordinal) == 0)
        cuDevicePrimaryCtxRelease(dev);
}

bool CudaDriver::push(CUcontext ctx) const
{
    return loaded && cuCtxPushCurrent(ctx) == 0;
}

void CudaDriver::pop() const
{
    CUcontext ctx = nullptr;
    if (loaded)
        cuCtxPopCurrent(&ctx);
}

bool CudaDriver::import_memory(intptr_t handle, size_t size, CUexternalMemory& mem, CUdeviceptr& ptr) const
{
    ExternalMemoryHandleDesc desc;
    memset(&desc, 0, sizeof desc);
#if _WIN32
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    desc.handle.win32.handle = (void*)handle;
#else
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = (int)handle;
#endif
    desc.size = size;
    desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    if (cuImportExternalMemory(&mem, &desc) != 0)
        return false;
    ExternalMemoryBufferDesc buf;
    memset(&buf, 0, sizeof buf);
    buf.size = size;
    if (cuExternalMemoryGetMappedBuffer(&ptr, mem, &buf) != 0)
    {
        cuDestroyExternalMemory(mem);
        mem = nullptr;
        return false;
    }
    return true;
}

void CudaDriver::release_memory(CUexternalMemory mem, CUdeviceptr ptr) const
{
    if (ptr)
        cuMemFree(ptr);
    if (mem)
        cuDestroyExternalMemory(mem);
}
