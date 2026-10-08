// Muxiveo — inférence RIFE par le plugin TensorRT facultatif. Voir trt_backend.h.

#include "trt_backend.h"

#include <cstdio>
#include <cstring>

#if _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

// Mémoire externe Vulkan (VK_KHR_external_memory_fd / _win32), absente de simplevk : valeurs de la spécification.
const VkStructureType kSTypePhysicalDeviceIdProperties = (VkStructureType)1000071004;
const VkStructureType kSTypeExternalMemoryBufferCreateInfo = (VkStructureType)1000072000;
const VkStructureType kSTypeExportMemoryAllocateInfo = (VkStructureType)1000072002;
const VkStructureType kSTypeMemoryGetWin32HandleInfo = (VkStructureType)1000073003;
const VkStructureType kSTypeMemoryGetFdInfo = (VkStructureType)1000074002;
const VkStructureType kSTypeMemoryDedicatedAllocateInfo = (VkStructureType)1000127001;
const VkStructureType kSTypePhysicalDeviceProperties2 = (VkStructureType)1000059001;
#if _WIN32
const uint32_t kHandleType = 0x2; // OPAQUE_WIN32
#else
const uint32_t kHandleType = 0x1; // OPAQUE_FD
#endif

struct PhysicalDeviceIdProperties
{
    VkStructureType sType;
    void* pNext;
    uint8_t deviceUUID[16];
    uint8_t driverUUID[16];
    uint8_t deviceLUID[8];
    uint32_t deviceNodeMask;
    VkBool32 deviceLUIDValid;
};
struct PhysicalDeviceProperties2
{
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceProperties properties;
};
struct ExternalMemoryBufferCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    uint32_t handleTypes;
};
struct ExportMemoryAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    uint32_t handleTypes;
};
struct MemoryDedicatedAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkImage image;
    VkBuffer buffer;
};
struct MemoryGetHandleInfo // VkMemoryGetFdInfoKHR / VkMemoryGetWin32HandleInfoKHR (même disposition)
{
    VkStructureType sType;
    const void* pNext;
    VkDeviceMemory memory;
    uint32_t handleType;
};
typedef VkResult(VKAPI_PTR* PFN_GetMemoryFd)(VkDevice, const MemoryGetHandleInfo*, int*);
typedef VkResult(VKAPI_PTR* PFN_GetMemoryWin32Handle)(VkDevice, const MemoryGetHandleInfo*, void**);
typedef void(VKAPI_PTR* PFN_GetPhysicalDeviceProperties2)(VkPhysicalDevice, PhysicalDeviceProperties2*);

const uint32_t kNvidiaVendorId = 0x10de;
// Architecture minimale de TensorRT for RTX (Turing) et API pilote du runtime livré (CUDA 12.9).
const int kMinComputeCapability = 75;
const int kMinDriverVersion = 12090;

#if _WIN32
const wchar_t* kPluginFile = L"mvo_rife_trt.dll";
#else
const char* kPluginFile = "libmvo_rife_trt.so";
#endif

void* open_plugin(const fs::path& dir)
{
    const fs::path path = dir / kPluginFile;
#if _WIN32
    // Dépendances (TensorRT for RTX) cherchées dans le dossier du plugin.
    return (void*)LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* plugin_symbol(void* lib, const char* name)
{
#if _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

std::string open_error()
{
#if _WIN32
    return "code " + std::to_string(GetLastError());
#else
    const char* e = dlerror();
    return e ? e : "?";
#endif
}

void log_to_stderr(void*, int level, const char* message)
{
    fprintf(stderr, "%s: TensorRT : %s\n", level == MVO_TRT_LOG_WARNING ? "warning" : "info", message ? message : "");
    fflush(stderr);
}

} // namespace

const MvoTrtApi* TrtBackend::load_api(const fs::path& plugin_dir, void*& handle, std::string& reason)
{
    handle = open_plugin(plugin_dir);
    if (!handle)
    {
        reason = "plugin illisible (" + open_error() + ")";
        return nullptr;
    }
    typedef const MvoTrtApi* (*GetApi)(uint32_t);
    GetApi get = (GetApi)plugin_symbol(handle, "mvo_trt_get_api");
    const MvoTrtApi* api = get ? get(MVO_TRT_ABI_VERSION) : nullptr;
    if (!api || api->abi_version != MVO_TRT_ABI_VERSION)
    {
        reason = "plugin incompatible avec cette version de muxiveo-rife (interface " + std::to_string(MVO_TRT_ABI_VERSION) + " attendue)";
        return nullptr;
    }
    return api;
}

int TrtBackend::cuda_ordinal(const ncnn::GpuInfo& info, std::string& reason)
{
    if (info.vendor_id() != kNvidiaVendorId)
    {
        reason = "GPU non NVIDIA";
        return -1;
    }
    const CudaDriver& cuda = CudaDriver::get();
    if (!cuda.ok())
    {
        reason = cuda.error();
        return -1;
    }
    // Appariement Vulkan / CUDA par UUID (repli : nom, puis rang parmi les homonymes).
    uint8_t uuid[16] = {0};
    bool have_uuid = false;
    PFN_GetPhysicalDeviceProperties2 props2 = (PFN_GetPhysicalDeviceProperties2)ncnn::vkGetPhysicalDeviceProperties2KHR;
    if (props2)
    {
        PhysicalDeviceIdProperties id;
        memset(&id, 0, sizeof id);
        id.sType = kSTypePhysicalDeviceIdProperties;
        PhysicalDeviceProperties2 p2;
        memset(&p2, 0, sizeof p2);
        p2.sType = kSTypePhysicalDeviceProperties2;
        p2.pNext = &id;
        props2(info.physicalDevice(), &p2);
        memcpy(uuid, id.deviceUUID, 16);
        for (uint8_t b : uuid)
            have_uuid = have_uuid || b != 0;
    }
    int same_name = 0;
    for (int i = 0; i < info.device_index(); i++)
        if (std::string(ncnn::get_gpu_info(i).device_name()) == info.device_name())
            same_name++;
    int ordinal = -1;
    int seen = 0;
    for (const CudaDeviceInfo& d : cuda.devices())
    {
        const bool match = have_uuid ? memcmp(d.uuid, uuid, 16) == 0 : (d.name == info.device_name() && seen++ == same_name);
        if (!match)
            continue;
        ordinal = d.ordinal;
        if (d.compute_capability < kMinComputeCapability)
        {
            reason = "architecture " + std::to_string(d.compute_capability / 10) + "." + std::to_string(d.compute_capability % 10)
                     + " non prise en charge (Turing 7.5 ou plus récent requis)";
            return -1;
        }
        break;
    }
    if (ordinal < 0)
    {
        reason = "GPU introuvable côté CUDA";
        return -1;
    }
    if (cuda.driver_version() < kMinDriverVersion)
    {
        reason = "pilote NVIDIA trop ancien (version 575 ou plus récente requise)";
        return -1;
    }
    return ordinal;
}

bool TrtBackend::load(const fs::path& dir, const ncnn::VulkanDevice* device, bool quiet, std::string& reason)
{
    vkdev = device;
    plugin_dir = dir;
    log_quiet = quiet;
    cuda_ordinal_ = cuda_ordinal(vkdev->info, reason);
    if (cuda_ordinal_ < 0)
        return false;
    // Export de la mémoire Vulkan : extension activée par le correctif ncnn de muxiveo-rife.
#if _WIN32
    if (!ncnn::vkGetDeviceProcAddr(vkdev->vkdevice(), "vkGetMemoryWin32HandleKHR"))
#else
    if (!ncnn::vkGetDeviceProcAddr(vkdev->vkdevice(), "vkGetMemoryFdKHR"))
#endif
    {
        reason = "export de mémoire Vulkan vers CUDA indisponible sur ce pilote";
        return false;
    }
    api = load_api(dir, library, reason);
    if (!api)
        return false;
    char err[512] = {0};
    if (!api->device_supported(cuda_ordinal_, err, sizeof err))
    {
        reason = err;
        return false;
    }
    if (!CudaDriver::get().retain_primary(cuda_ordinal_, cuda_ctx))
    {
        cuda_ctx = nullptr;
        reason = "contexte CUDA indisponible";
        return false;
    }
    summary = std::string("TensorRT, plugin ") + api->version();
    return true;
}

bool TrtBackend::create_slot(Slot& slot, size_t bytes, int width, int height, std::string& reason)
{
    VkDevice device = vkdev->vkdevice();
    ExternalMemoryBufferCreateInfo ext = {kSTypeExternalMemoryBufferCreateInfo, nullptr, kHandleType};
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof bi);
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.pNext = &ext;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (ncnn::vkCreateBuffer(device, &bi, nullptr, &slot.buffer) != VK_SUCCESS)
    {
        slot.buffer = 0;
        reason = "tampon Vulkan partagé impossible";
        return false;
    }
    VkMemoryRequirements req;
    ncnn::vkGetBufferMemoryRequirements(device, slot.buffer, &req);
    const uint32_t type = vkdev->find_memory_index(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    MemoryDedicatedAllocateInfo dedicated = {kSTypeMemoryDedicatedAllocateInfo, nullptr, 0, slot.buffer};
    ExportMemoryAllocateInfo exported = {kSTypeExportMemoryAllocateInfo, &dedicated, kHandleType};
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof ai);
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &exported;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (type == (uint32_t)-1 || ncnn::vkAllocateMemory(device, &ai, nullptr, &slot.device_memory) != VK_SUCCESS)
    {
        slot.device_memory = 0;
        reason = "mémoire GPU insuffisante (tampons TensorRT)";
        return false;
    }
    if (ncnn::vkBindBufferMemory(device, slot.buffer, slot.device_memory, 0) != VK_SUCCESS)
    {
        reason = "tampon Vulkan partagé impossible";
        return false;
    }

    MemoryGetHandleInfo get = {kSTypeMemoryGetFdInfo, nullptr, slot.device_memory, kHandleType};
    const CudaDriver& cuda = CudaDriver::get();
    bool imported = false;
    if (cuda.push(cuda_ctx))
    {
#if _WIN32
        get.sType = kSTypeMemoryGetWin32HandleInfo;
        PFN_GetMemoryWin32Handle fn = (PFN_GetMemoryWin32Handle)ncnn::vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
        void* handle = nullptr;
        if (fn && fn(device, &get, &handle) == VK_SUCCESS && handle)
        {
            imported = cuda.import_memory((intptr_t)handle, (size_t)req.size, slot.cuda_memory, slot.cuda_ptr);
            CloseHandle((HANDLE)handle);
        }
#else
        PFN_GetMemoryFd fn = (PFN_GetMemoryFd)ncnn::vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
        int fd = -1;
        if (fn && fn(device, &get, &fd) == VK_SUCCESS && fd >= 0)
        {
            imported = cuda.import_memory((intptr_t)fd, (size_t)req.size, slot.cuda_memory, slot.cuda_ptr);
            if (!imported)
                close(fd); // descripteur repris par CUDA seulement en cas de succès
        }
#endif
        cuda.pop();
    }
    if (!imported)
    {
        slot.cuda_memory = nullptr;
        slot.cuda_ptr = 0;
        reason = "partage de mémoire Vulkan / CUDA refusé par le pilote";
        return false;
    }

    slot.memory.buffer = slot.buffer;
    slot.memory.offset = 0;
    slot.memory.capacity = bytes;
    slot.memory.memory = slot.device_memory;
    slot.memory.mapped_ptr = nullptr;
    slot.memory.memory_type_index = type;
    slot.memory.access_flags = 0;
    slot.memory.stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    slot.memory.refcount = 0;
    // Mémoire externe (allocateur nul) : jamais libérée par ncnn ; NCHW contigu (cstep = largeur x hauteur).
    slot.mat = ncnn::VkMat(width, height, 3, &slot.memory, 2u, 1, nullptr);
    if ((int)slot.mat.cstep != width * height)
    {
        reason = "disposition des tampons incompatible avec TensorRT";
        return false;
    }
    return true;
}

void TrtBackend::destroy_slot(Slot& slot)
{
    if (slot.cuda_memory || slot.cuda_ptr)
    {
        const CudaDriver& cuda = CudaDriver::get();
        if (cuda.push(cuda_ctx))
        {
            cuda.release_memory(slot.cuda_memory, slot.cuda_ptr);
            cuda.pop();
        }
    }
    slot.mat.release();
    VkDevice device = vkdev ? vkdev->vkdevice() : 0;
    if (slot.buffer)
        ncnn::vkDestroyBuffer(device, slot.buffer, nullptr);
    if (slot.device_memory)
        ncnn::vkFreeMemory(device, slot.device_memory, nullptr);
    slot = Slot();
}

bool TrtBackend::open(const std::string& model, int width, int height, const fs::path& cache_dir, std::string& reason)
{
    if (!api || !cuda_ctx)
    {
        reason = "plugin non chargé";
        return false;
    }
    const fs::path model_path = plugin_dir / "models" / (model + ".onnx");
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec))
    {
        reason = "modèle " + model + " non fourni par le plugin";
        return false;
    }
    const size_t bytes = (size_t)width * height * 3 * 2;
    for (Slot& slot : slots)
        if (!create_slot(slot, bytes, width, height, reason))
            return false;

    const std::string model_utf8 = model_path.u8string();
    const std::string cache_utf8 = cache_dir.u8string();
    MvoTrtCreateParams params;
    memset(&params, 0, sizeof params);
    params.struct_size = sizeof params;
    params.cuda_device = cuda_ordinal_;
    params.model_path = model_utf8.c_str();
    params.cache_dir = cache_utf8.c_str();
    params.width = width;
    params.height = height;
    params.log = log_quiet ? nullptr : log_to_stderr;
    char err[512] = {0};
    const CudaDriver& cuda = CudaDriver::get();
    if (!cuda.push(cuda_ctx))
    {
        reason = "contexte CUDA indisponible";
        return false;
    }
    session = api->create(&params, err, sizeof err);
    cuda.pop();
    if (!session)
    {
        reason = err;
        return false;
    }
    return true;
}

bool TrtBackend::infer(float t, std::string& error)
{
    MvoTrtInferParams params;
    memset(&params, 0, sizeof params);
    params.struct_size = sizeof params;
    params.in0 = slots[0].cuda_ptr;
    params.in1 = slots[1].cuda_ptr;
    params.out = slots[2].cuda_ptr;
    params.t = t;
    char err[512] = {0};
    const CudaDriver& cuda = CudaDriver::get();
    if (!cuda.push(cuda_ctx))
    {
        error = "contexte CUDA indisponible";
        return false;
    }
    const bool ok = api->infer(session, &params, err, sizeof err) != 0;
    cuda.pop();
    if (!ok)
    {
        error = err;
        return false;
    }
    // Sortie écrite hors de Vulkan (inférence terminée) : prochaine lecture précédée d'une barrière complète.
    slots[2].memory.access_flags = 0;
    slots[2].memory.stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    return true;
}

TrtBackend::~TrtBackend()
{
    const CudaDriver& cuda = CudaDriver::get();
    if (session && cuda.push(cuda_ctx))
    {
        api->destroy(session);
        cuda.pop();
    }
    session = nullptr;
    for (Slot& slot : slots)
        destroy_slot(slot);
    if (cuda_ctx)
        cuda.release_primary(cuda_ordinal_);
    // Plugin laissé chargé : TensorRT n'est pas conçu pour être déchargé avant la fin du processus.
}
