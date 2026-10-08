// Muxiveo — inférence RIFE par le plugin TensorRT facultatif (mvo-rife-trt), GPU NVIDIA seulement.
// Le plugin est chargé à l'exécution ; les tenseurs résident dans des tampons Vulkan exportés vers CUDA
// (aucune copie par l'hôte). Sans plugin, ou au moindre échec, muxiveo-rife garde l'inférence Vulkan (ncnn).

#ifndef MUXIVEO_RIFE_TRT_BACKEND_H
#define MUXIVEO_RIFE_TRT_BACKEND_H

#include <filesystem>
#include <string>

#include "cuda_driver.h"
#include "trt_plugin_abi.h"

// ncnn
#include "gpu.h"
#include "mat.h"

class TrtBackend
{
public:
    TrtBackend() = default;
    ~TrtBackend();
    TrtBackend(const TrtBackend&) = delete;
    TrtBackend& operator=(const TrtBackend&) = delete;

    // Charge le plugin et apparie le GPU Vulkan à son périphérique CUDA. Faux = indisponible (reason).
    // quiet : pas de lignes « info: » du plugin (construction du moteur, spécialisation).
    bool load(const std::filesystem::path& plugin_dir, const ncnn::VulkanDevice* vkdev, bool quiet, std::string& reason);
    // Session d'un modèle (nom de dossier muxiveo-rife, suffixe « -uhd » éventuel), dimensions paddées.
    bool open(const std::string& model, int width, int height, const std::filesystem::path& cache_dir, std::string& reason);

    // Tampons partagés (fp16, planaires contigus, 3 canaux) : entrées à remplir, sortie lue après infer().
    const ncnn::VkMat& input(int index) const { return slots[index].mat; }
    const ncnn::VkMat& output() const { return slots[2].mat; }
    // Inférence synchrone (Vulkan doit avoir terminé d'écrire les entrées).
    bool infer(float t, std::string& error);

    std::string description() const { return summary; }

    // Charge la bibliothèque du plugin et vérifie son interface (NULL = raison).
    static const MvoTrtApi* load_api(const std::filesystem::path& plugin_dir, void*& handle, std::string& reason);
    // Ordinal CUDA du GPU Vulkan s'il peut exécuter TensorRT for RTX (Turing+, pilote assez récent), sinon -1.
    static int cuda_ordinal(const ncnn::GpuInfo& info, std::string& reason);

private:
    struct Slot
    {
        ncnn::VkBufferMemory memory{};
        ncnn::VkMat mat;
        VkBuffer buffer = 0;
        VkDeviceMemory device_memory = 0;
        CUexternalMemory cuda_memory = nullptr;
        CUdeviceptr cuda_ptr = 0;
    };
    bool create_slot(Slot& slot, size_t bytes, int width, int height, std::string& reason);
    void destroy_slot(Slot& slot);

    void* library = nullptr;
    const MvoTrtApi* api = nullptr;
    MvoTrtSession* session = nullptr;
    const ncnn::VulkanDevice* vkdev = nullptr;
    std::filesystem::path plugin_dir;
    bool log_quiet = false;
    int cuda_ordinal_ = -1;
    CUcontext cuda_ctx = nullptr;
    Slot slots[3];
    std::string summary;
};

#endif // MUXIVEO_RIFE_TRT_BACKEND_H
