// Muxiveo — moteur d'interpolation RIFE v4 sur Vulkan (ncnn).
// Entrée/sortie : trames YUV brutes au format y4m ; conversion YUV <-> RGB,
// normalisation et quantification faites sur le GPU.

#ifndef MUXIVEO_RIFE_ENGINE_H
#define MUXIVEO_RIFE_ENGINE_H

#include <filesystem>
#include <string>

#include "y4m.h"

// ncnn
#include "gpu.h"
#include "net.h"

// Allocateur de blobs Vulkan signalant les échecs d'allocation (VRAM saturée).
class TrackedBlobAllocator : public ncnn::VkBlobAllocator
{
public:
    explicit TrackedBlobAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkBlobAllocator(vkdev), failed(false)
    {
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        ncnn::VkBufferMemory* ptr = ncnn::VkBlobAllocator::fastMalloc(size);
        if (!ptr)
            failed = true;
        return ptr;
    }

    virtual ncnn::VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
    {
        ncnn::VkImageMemory* ptr = ncnn::VkBlobAllocator::fastMalloc(w, h, c, elemsize, elempack);
        if (!ptr)
            failed = true;
        return ptr;
    }

    bool failed;
};

struct ColorParams
{
    float kr = 0.2126f;  // BT.709 par défaut
    float kb = 0.0722f;
    bool full_range = false;
    ChromaSiting siting = ChromaSiting::Left;
};

// Trame déjà convertie en RGB paddé et résidente sur le GPU.
struct GpuFrame
{
    ncnn::VkMat rgb;
    bool ready = false;

    void reset()
    {
        rgb.release();
        ready = false;
    }
};

class RifeEngine
{
public:
    RifeEngine();
    ~RifeEngine();

    bool init(int gpu_index, bool fp32, int num_threads, std::string& error);
    // uhd : flux optique calculé à demi-résolution (voir uhd.h).
    bool load_model(const std::filesystem::path& dir, int padding, bool uhd, std::string& error);
    bool configure(const FrameFormat& fmt, const ColorParams& color, std::string& error);

    // TTA (test-time augmentation) : 1 = désactivé ; 2 = + sens temporel inverse ;
    // 4 = + retournement horizontal ; 8 = + retournements vertical et double.
    // Sorties moyennées (fp32) ; coût x n, VRAM d'une seule variante.
    void set_tta(int n) { tta = n; }

    // Envoie une trame brute (padded_frame_bytes() octets, alignée sur 4) et la convertit en RGB.
    bool upload(const uint8_t* frame, GpuFrame& out, std::string& error);

    // Génère la trame intermédiaire au temps t (0 < t < 1) et l'écrit, au format y4m, dans dst.
    bool interpolate(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error);

    // Diagnostic : conversion YUV -> RGB -> YUV sans réseau (mesure de la perte de conversion).
    bool roundtrip(const GpuFrame& a, uint8_t* dst, std::string& error);

    std::string device_name() const;
    bool uses_fp16() const { return opt.use_fp16_storage; }
    // Vrai si un échec GPU vient d'une allocation refusée (mémoire vidéo insuffisante).
    bool out_of_memory() const { return blob_vkallocator && blob_vkallocator->failed; }

private:
    bool convert_and_download(ncnn::VkCompute& cmd, const ncnn::VkMat& rgb, uint8_t* dst, std::string& error);
    bool record_timestep(ncnn::VkCompute& cmd, float t, ncnn::VkMat& timestep, std::string& error);
    bool record_network(ncnn::VkCompute& cmd, const ncnn::VkMat& in0, const ncnn::VkMat& in1, const ncnn::VkMat& timestep,
                        ncnn::VkMat& out, std::string& error);
    bool record_flip(ncnn::VkCompute& cmd, const ncnn::VkMat& src, int flip, ncnn::VkMat& dst, std::string& error);
    bool interpolate_tta(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error);
    int chroma_mode_x() const;
    int chroma_mode_y() const;
    ncnn::Pipeline* make_pipeline(const char* comp_data, int comp_size, int lx, int ly, int lz);

    ncnn::VulkanDevice* vkdev;
    TrackedBlobAllocator* blob_vkallocator;
    ncnn::VkAllocator* staging_vkallocator;
    ncnn::Net flownet;
    ncnn::Option opt;

    ncnn::Pipeline* pipeline_yuv_to_rgb;
    ncnn::Pipeline* pipeline_rgb_to_yuv;
    ncnn::Pipeline* pipeline_pack;
    ncnn::Pipeline* pipeline_timestep;
    ncnn::Pipeline* pipeline_tta_flip;
    ncnn::Pipeline* pipeline_tta_accumulate;
    ncnn::Pipeline* pipeline_tta_resolve;

    FrameFormat fmt;
    ColorParams color;
    int padding;
    int w_padded;
    int h_padded;
    int words;
    int pack_dispatch_w;
    int tta;
};

#endif // MUXIVEO_RIFE_ENGINE_H
