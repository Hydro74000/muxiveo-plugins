// Muxiveo — pilote CUDA chargé à l'exécution (libcuda.so.1 / nvcuda.dll), sans CUDA Toolkit.
// Sert à décrire les GPU NVIDIA (--list-gpus) et au partage de mémoire Vulkan / CUDA du plugin TensorRT.

#ifndef MUXIVEO_RIFE_CUDA_DRIVER_H
#define MUXIVEO_RIFE_CUDA_DRIVER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUextMemory_st* CUexternalMemory;
typedef unsigned long long CUdeviceptr;

// Description d'un GPU CUDA.
struct CudaDeviceInfo
{
    int ordinal = -1;
    std::string name;
    unsigned char uuid[16] = {0};
    int compute_capability = 0; // ex. 89
};

class CudaDriver
{
public:
    // Instance unique, chargée au premier appel ; ok() faux sans pilote NVIDIA.
    static CudaDriver& get();
    bool ok() const { return loaded; }
    const std::string& error() const { return load_error; }
    // Version de l'API pilote (ex. 12090 pour CUDA 12.9), 0 si inconnue.
    int driver_version() const { return version; }
    std::vector<CudaDeviceInfo> devices() const;

    // Contexte primaire du périphérique (partagé avec le plugin TensorRT).
    bool retain_primary(int ordinal, CUcontext& ctx) const;
    void release_primary(int ordinal) const;
    bool push(CUcontext ctx) const;
    void pop() const;

    // Import d'une allocation Vulkan exportée (descripteur de fichier sous Linux, HANDLE sous Windows ;
    // le descripteur appartient à CUDA après un import réussi, le HANDLE reste à fermer par l'appelant).
    bool import_memory(intptr_t handle, size_t size, CUexternalMemory& mem, CUdeviceptr& ptr) const;
    void release_memory(CUexternalMemory mem, CUdeviceptr ptr) const;

private:
    CudaDriver();
    bool loaded = false;
    int version = 0;
    std::string load_error;
    void* lib = nullptr;

    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuDriverGetVersion)(int*) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetUuid)(void*, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxPopCurrent)(CUcontext*) = nullptr;
    CUresult (*cuImportExternalMemory)(CUexternalMemory*, const void*) = nullptr;
    CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory, const void*) = nullptr;
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
};

#endif // MUXIVEO_RIFE_CUDA_DRIVER_H
