// Restauration (R2) : réseau de restauration sur GPU (ncnn Vulkan), une image à la fois.
//
// Portage de r1r_infer.run (banc, plan Restauration § 5 bis, jalon P2) : pour l'image t et ses 4 voisines du même plan
// (t − 2, t − 1, t + 1, t + 2, miroir aux bords et aux coupes : choix de l'appelant), R'G'B' sans écrêtage avec marge
// miroir de 32 px, flux SpyNet (graphe r2_flow) à `flow_scale` de la définition, agrandis, voisines recalées
// (rife.Warp), entrée 17 canaux (image, voisines, carte de grain, canal PQ), corps (graphe r2_body) par tuiles avec
// fenêtre de Hann, puis codes Y'CbCr flottants non arrondis (sous-échantillonnage chroma antirepliement du banc).
// Les images converties sont gardées en mémoire GPU (chacune sert à 5 images) jusqu'à drop_frame().
#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "engine.h"
#include "trt_backend.h"
#include "gpu.h"
#include "mat.h"
#include "net.h"
#include "pipeline.h"
#include "y4m.h"

class RestoreNet
{
public:
    RestoreNet() = default;
    ~RestoreNet();
    RestoreNet(const RestoreNet&) = delete;
    RestoreNet& operator=(const RestoreNet&) = delete;

    // Graphes <dir>/r2_flow.ncnn.{param,bin} et r2_body.ncnn.{param,bin} ; flow_scale : définition du flux (0,5 ; 0,25
    // pour les poids du mode rapide). opt : options du moteur (fp16 de stockage selon le GPU).
    bool init(ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, const std::filesystem::path& dir, double flow_scale,
              std::string& error);
    // pq : canal de transfert du réseau (1 pour SMPTE ST 2084, 0 sinon) ; tile : côté des tuiles du corps (multiple de 4).
    bool configure(const FrameFormat& fmt, const ColorParams& color, bool pq, int tile, std::string& error);

    // Image source (trame y4m brute, tampon lisible sur padded_frame_bytes()) : conversion et luma du flux, en cache.
    bool add_frame(int64_t index, const void* frame, std::string& error);
    void drop_frame(int64_t index);

    // Région de mesure du rapport de grain : flux pleine définition de l'image t recadrés (4 × ch × cw × 2, ordre
    // t − 2, t − 1, t + 1, t + 2, entrelacés x / y, en pixels).
    struct Crop
    {
        int x0 = 0, y0 = 0, w = 0, h = 0;
    };

    // Image t : voisines nb[4] (déjà ajoutées), carte de grain (w × h octets, σ × 20). Sortie : codes Y'CbCr flottants
    // [Y (w × h) | Cb (cw × ch) | Cr (cw × ch)] dans le tampon rapatrié lui-même (yuv.w valeurs, aucune copie) ; flux
    // recadrés si `crop` et `flows` non nuls.
    bool process(int64_t t, const int64_t nb[4], const uint8_t* grain_map, ncnn::Mat& yuv, const Crop* crop,
                 std::vector<float>* flows, std::string& error);

    // Inférence par le plugin TensorRT (interface 2) : graphes <onnx_dir>/r2_flow.onnx et r2_body.onnx, moteurs en cache
    // sous le nom « <tag>-r2_* » ; corps calculé sur l'image entière. Faux = Vulkan conservé (reason).
    bool enable_trt(const std::filesystem::path& plugin_dir, const std::filesystem::path& onnx_dir, const std::string& tag,
                    const std::filesystem::path& cache_dir, bool quiet, std::string& reason);
    bool uses_trt() const { return trt_on; }
    const std::string& trt_description() const { return trt_summary; }

    // Mémoire GPU épuisée lors du dernier appel (repli possible sur des tuiles plus petites).
    bool out_of_memory() const { return oom; }
    int tile_size() const { return tile; }
    void set_tile(int size) { tile = std::max(64, size / 4 * 4); }

    int padded_width() const { return W; }
    int padded_height() const { return H; }
    int flow_width() const { return ws; }
    int flow_height() const { return hs; }

    static constexpr int EDGE = 32;  // marge miroir (bande jamais apprise)
    static constexpr int PAD = 32;   // recouvrement des tuiles

private:
    struct Cached
    {
        ncnn::VkMat rgb;     // W × H × 3
        ncnn::VkMat flowin;  // wu × hu × 3 (luma du flux répliquée)
    };
    ncnn::Pipeline* make_pipeline(const char* data, int size, int lx, int ly, int lz);
    bool resize(ncnn::VkCompute& cmd, const ncnn::VkMat& src, int inc, int outw, int outh, int outc, float mul0, float mul1,
                ncnn::VkMat& dst);
    bool warp_from_flow(ncnn::VkCompute& cmd, const ncnn::VkMat& f, const ncnn::VkMat& rgb, ncnn::VkMat& full, ncnn::VkMat& warped,
                        std::string& error);
    bool assemble(ncnn::VkCompute& cmd, const ncnn::VkMat& rgb, const ncnn::VkMat warped[4], const uint8_t* grain_map, ncnn::VkMat& x);
    bool to_yuv(ncnn::VkCompute& cmd, const ncnn::VkMat& acc, const ncnn::VkMat& wsum, ncnn::VkMat& out);
    std::vector<ncnn::vk_constant_type> tile_constants(int mode, const ncnn::VkMat& x, const ncnn::VkMat& acc, int c, int x0, int y0, int tw,
                                                       int th, int tcstep, int wx0, int wy0, int tpw, int tph) const;
    void crop_flows(const ncnn::Mat host_flows[4], const Crop& crop, std::vector<float>& flows) const;
    void copy_dense(ncnn::VkCompute& cmd, const ncnn::VkMat& src, int c, const ncnn::VkMat& dst);
    bool process_trt(int64_t t, const int64_t nb[4], const uint8_t* grain_map, ncnn::Mat& yuv, const Crop* crop,
                     std::vector<float>* flows, std::string& error);
    void release();

    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::Option opt;
    ncnn::VkAllocator* blob_alloc = nullptr;
    ncnn::VkAllocator* staging_alloc = nullptr;
    ncnn::Net flow_net, body_net;
    ncnn::Layer* warp = nullptr;
    ncnn::Pipeline* p_yuv_to_rgb = nullptr;
    ncnn::Pipeline* p_luma_area = nullptr;
    ncnn::Pipeline* p_resize = nullptr;
    ncnn::Pipeline* p_assemble = nullptr;
    ncnn::Pipeline* p_tile = nullptr;
    ncnn::Pipeline* p_rgb_to_yuv = nullptr;
    ncnn::Pipeline* p_copy = nullptr;
    std::unique_ptr<TrtBackend> trt_flow, trt_body;
    bool trt_on = false;
    std::string trt_summary;
    double flow_scale = 0.5;
    FrameFormat fmt;
    ColorParams color;
    float pq = 0.f;
    int tile = 512;
    int W = 0, H = 0;    // image avec marge
    int ws = 0, hs = 0;  // définition du flux
    int wu = 0, hu = 0;  // entrée du graphe de flux (multiple de 32)
    size_t elemsize = 4;
    bool oom = false;
    std::map<int64_t, Cached> cache;
};
