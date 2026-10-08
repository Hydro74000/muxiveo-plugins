// Muxiveo — moteur d'interpolation RIFE v4 sur Vulkan (ncnn), seul ou hybride avec une
// compensation de mouvement par blocs (MC).
// Entrée/sortie : trames YUV brutes au format y4m ; conversion YUV <-> RGB,
// normalisation et quantification faites sur le GPU.

#ifndef MUXIVEO_RIFE_ENGINE_H
#define MUXIVEO_RIFE_ENGINE_H

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "y4m.h"

class TrtBackend;

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

// Moteur d'interpolation : RIFE seul, compensation de mouvement par blocs (MC) seule
// (diagnostic), ou hybride (MC là où sa correspondance est fiable, RIFE ailleurs).
enum class InterpEngine
{
    Rife,
    Mc,
    Hybrid
};

// Trame déjà convertie en RGB paddé et résidente sur le GPU.
struct GpuFrame
{
    ncnn::VkMat rgb;
    ncnn::VkMat lum[4];   // pyramide de luminance 1, 1/2, 1/4, 1/8 (moteurs MC / hybride)
    uint64_t id = 0;      // numéro d'envoi : identifie la paire (cache du champ de vecteurs)
    bool ready = false;

    void reset()
    {
        rgb.release();
        for (ncnn::VkMat& l : lum)
            l.release();
        id = 0;
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
    // Moteur hybride : second réseau à flux demi-résolution, substitué à l'hybride là où le
    // déplacement entre les deux sources est grand (padding du modèle x2 requis).
    bool load_large_motion_model(const std::filesystem::path& dir, std::string& error);
    bool configure(const FrameFormat& fmt, const ColorParams& color, std::string& error);

    // TTA (test-time augmentation) : 1 = désactivé ; 2 = + sens temporel inverse ;
    // 4 = + retournement horizontal ; 8 = + retournements vertical et double.
    // Sorties moyennées (fp32) ; coût x n, VRAM d'une seule variante.
    void set_tta(int n) { tta = n; }
    void set_engine(InterpEngine e) { engine_mode = e; }
    // Inférence RIFE par le plugin TensorRT (nul = ncnn Vulkan). Le backend doit survivre aux interpolations.
    void set_trt(TrtBackend* backend) { trt = backend; }
    // Plugin TensorRT du réseau à flux demi-résolution (nul = ncnn Vulkan pour ce réseau).
    void set_trt_large(TrtBackend* backend) { trt_lm = backend; }
    // Seuil bas des grands mouvements (px entre les sources) ; remplacement complet à 3x.
    void set_large_motion_threshold(float px) { lm_lo = px; }
    bool large_motion() const { return lm_loaded; }
    // Paires (trames source consécutives) traitées / avec grands mouvements.
    int64_t large_motion_pairs_total() const { return lm_pairs_total; }
    int64_t large_motion_pairs_active() const { return lm_pairs_active; }
    bool uses_trt() const { return trt != nullptr; }
    const ncnn::VulkanDevice* device() const { return vkdev; }
    int padded_width() const { return w_padded; }
    int padded_height() const { return h_padded; }
    InterpEngine engine() const { return engine_mode; }

    // Candidats de flux matériel (NVOF) pour la paire courante : flux aller et retour sur une
    // grille de `grid` px (S10.5, entrelacés), réduits en médianes par bloc de 16.
    void set_flow_hints(const int16_t* fwd, const int16_t* bwd, int gw, int gh, int grid, float vscale = 1.f);
    void clear_flow_hints() { hints_host.clear(); hints_dirty = true; }
    int gpu_index() const { return gpu; }
    uint32_t vendor_id() const;

    // Envoie une trame brute (padded_frame_bytes() octets, alignée sur 4) et la convertit en RGB.
    bool upload(const uint8_t* frame, GpuFrame& out, std::string& error);

    // Génère la trame intermédiaire au temps t (0 < t < 1) et l'écrit, au format y4m, dans dst.
    // before_mc : appelé (moteurs MC / hybride) une fois RIFE soumis et terminé, juste avant la
    // recherche de mouvement — permet de recouvrir un calcul externe (flux NVOF) par RIFE.
    bool interpolate(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error,
                     const std::function<void()>& before_mc = {});

    // Diagnostic : conversion YUV -> RGB -> YUV sans réseau (mesure de la perte de conversion).
    bool roundtrip(const GpuFrame& a, uint8_t* dst, std::string& error);

    std::string device_name() const;
    bool uses_fp16() const { return opt.use_fp16_storage; }
    // Vrai si un échec GPU vient d'une allocation refusée (mémoire vidéo insuffisante).
    bool out_of_memory() const { return blob_vkallocator && blob_vkallocator->failed; }

private:
    bool convert_and_download(ncnn::VkCompute& cmd, const ncnn::VkMat& rgb, uint8_t* dst, std::string& error);
    bool record_timestep(ncnn::VkCompute& cmd, float t, ncnn::VkMat& timestep, std::string& error);
    bool record_network(ncnn::Net& net, ncnn::VkCompute& cmd, const ncnn::VkMat& in0, const ncnn::VkMat& in1,
                        const ncnn::VkMat& timestep, ncnn::VkMat& out, std::string& error);
    // Image RIFE au temps t : réseau ncnn enregistré dans cmd, ou plugin TensorRT (cmd soumis et réinitialisé,
    // sortie dans un tampon partagé). Échec du plugin : avertissement puis ncnn pour la suite.
    // large : réseau à flux demi-résolution (grands mouvements).
    bool infer_rife(ncnn::VkCompute& cmd, const ncnn::VkMat& in0, const ncnn::VkMat& in1, float t, ncnn::VkMat& out,
                    std::string& error, bool large = false);
    bool record_flip(ncnn::VkCompute& cmd, const ncnn::VkMat& src, int flip, ncnn::VkMat& dst, std::string& error);
    bool interpolate_tta(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error);
    bool rife_tta_rgb(const GpuFrame& a, const GpuFrame& b, float t, ncnn::VkMat& merged, std::string& error);
    bool record_mc(ncnn::VkCompute& cmd, const GpuFrame& a, const GpuFrame& b, float t, const ncnn::VkMat& rife,
                   ncnn::VkMat& out, std::string& error);
    bool record_large_motion(ncnn::VkCompute& cmd, const GpuFrame& a, const GpuFrame& b, float t, ncnn::VkMat& out,
                             std::string& error);
    void record_filter(ncnn::VkCompute& cmd, const ncnn::VkMat& src, ncnn::VkMat& dst, int r, int axis, int op,
                       int threshold, float thr);
    ncnn::VkMat float_mat(int n);
    int chroma_mode_x() const;
    int chroma_mode_y() const;
    ncnn::Pipeline* make_pipeline(const char* comp_data, int comp_size, int lx, int ly, int lz);

    ncnn::VulkanDevice* vkdev;
    TrackedBlobAllocator* blob_vkallocator;
    ncnn::VkAllocator* staging_vkallocator;
    ncnn::Net flownet;
    ncnn::Net flownet_lm;   // flux demi-résolution (grands mouvements, moteur hybride)
    ncnn::Option opt;

    ncnn::Pipeline* pipeline_yuv_to_rgb;
    ncnn::Pipeline* pipeline_rgb_to_yuv;
    ncnn::Pipeline* pipeline_pack;
    ncnn::Pipeline* pipeline_timestep;
    ncnn::Pipeline* pipeline_tta_flip;
    ncnn::Pipeline* pipeline_tta_accumulate;
    ncnn::Pipeline* pipeline_tta_resolve;
    ncnn::Pipeline* pipeline_mc_luma;
    ncnn::Pipeline* pipeline_mc_down;
    ncnn::Pipeline* pipeline_mc_search;
    ncnn::Pipeline* pipeline_mc_median;
    ncnn::Pipeline* pipeline_mc_recon;
    ncnn::Pipeline* pipeline_mc_filter;
    ncnn::Pipeline* pipeline_mc_region;
    ncnn::Pipeline* pipeline_mc_blend;
    ncnn::Pipeline* pipeline_mc_motion;
    ncnn::Pipeline* pipeline_trt_pack;
    TrtBackend* trt = nullptr;
    TrtBackend* trt_lm = nullptr;
    bool lm_loaded = false;
    float lm_lo = 16.f;
    float lm_floor = 0.f;   // poids minimal de la paire (mouvement d'ensemble)
    // décision « grands mouvements » prise une fois par paire (champ de vecteurs relu)
    bool lm_active = false;
    uint64_t lm_a = 0;
    uint64_t lm_b = 0;
    int64_t lm_pairs_total = 0;
    int64_t lm_pairs_active = 0;
    InterpEngine engine_mode;
    int lw[4];   // dimensions des niveaux de la pyramide
    int lh[4];
    ncnn::VkMat dummy;
    std::vector<float> hints_host;   // (bw × bh) × 2 candidats vec2, vide = aucun
    bool hints_dirty = false;
    ncnn::VkMat hints_gpu;
    // Champ de vecteurs calculé une fois par paire (à t = 0,5), réutilisé pour chaque position t.
    ncnn::VkMat cached_field;
    uint64_t cache_a = 0;
    uint64_t cache_b = 0;
    uint64_t upload_seq = 0;
    int gpu = 0;

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
