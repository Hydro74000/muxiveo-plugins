// Muxiveo — moteur d'interpolation RIFE v4 sur Vulkan (ncnn).
// Dérivé du chemin `process_v4` de rife-ncnn-vulkan (nihui, TNTwise — MIT),
// réécrit pour des trames YUV haute profondeur (8 à 16 bits) sans passage par
// des pixels RGB 8 bits côté CPU.

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "rife_ops.h"
#include "uhd.h"

#include "pack_samples.comp.hex.h"
#include "rgb_to_yuv.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"
#include "tta_accumulate.comp.hex.h"
#include "tta_flip.comp.hex.h"
#include "tta_resolve.comp.hex.h"
#include "yuv_to_rgb.comp.hex.h"
#include "trt_pack.comp.hex.h"
#include "trt_backend.h"
#include "mc_blend.comp.hex.h"
#include "mc_down.comp.hex.h"
#include "mc_filter.comp.hex.h"
#include "mc_luma.comp.hex.h"
#include "mc_median.comp.hex.h"
#include "mc_recon.comp.hex.h"
#include "mc_region.comp.hex.h"
#include "mc_search.comp.hex.h"

// Luminance des noyaux MC en unités de code 10 bits (plage limitée) : seuils indépendants
// de la profondeur réelle du flux.
static const float MC_LUM_SCALE = 876.f;

// Réglages de la compensation de mouvement (valeurs validées sur les bancs de motifs répétitifs).
static const float MC_LAMBDA_COARSE = 0.05f;   // pénalité de longueur, niveau 1/8 exhaustif
static const float MC_LAMBDA = 0.5f;           // cohérence avec le prédicteur, niveaux 1/4 à 1/1
static const float MC_LAMBDA_PROP = 2.f;       // passes de propagation
static const int MC_PROP_PASSES = 2;           // propagation aux niveaux grossiers (repliement de période)
static const int MC_MARGIN_FINE = 8;           // fenêtre 32×32 aux niveaux 1/1 et 1/2
static const int MC_MARGIN_COARSE = 16;        // fenêtre 48×48 aux niveaux 1/4 et 1/8
static const int MC_COARSE_RADIUS = 8;         // recherche exhaustive ±8 px au niveau 1/8
static const float MC_Q_THRESHOLD = 20.f;      // erreur bilatérale d'un pixel mal expliqué
static const float MC_DENSITY = 0.25f;         // densité au-delà de laquelle RIFE prend la main
static const float MC_AGREE = 20.f;            // écart MC / RIFE sous lequel les deux sont moyennés

DEFINE_LAYER_CREATOR(Warp)

static FILE* open_binary(const std::filesystem::path& path)
{
#if _WIN32
    return _wfopen(path.c_str(), L"rb");
#else
    return fopen(path.c_str(), "rb");
#endif
}

RifeEngine::RifeEngine()
    : vkdev(0), blob_vkallocator(0), staging_vkallocator(0),
      pipeline_yuv_to_rgb(0), pipeline_rgb_to_yuv(0), pipeline_pack(0), pipeline_timestep(0),
      pipeline_tta_flip(0), pipeline_tta_accumulate(0), pipeline_tta_resolve(0),
      pipeline_mc_luma(0), pipeline_mc_down(0), pipeline_mc_search(0), pipeline_mc_median(0),
      pipeline_mc_recon(0), pipeline_mc_filter(0), pipeline_mc_region(0), pipeline_mc_blend(0), pipeline_trt_pack(0),
      engine_mode(InterpEngine::Rife), lw{0, 0, 0, 0}, lh{0, 0, 0, 0},
      padding(32), w_padded(0), h_padded(0), words(0), pack_dispatch_w(0), tta(1)
{
}

RifeEngine::~RifeEngine()
{
    delete pipeline_yuv_to_rgb;
    delete pipeline_rgb_to_yuv;
    delete pipeline_pack;
    delete pipeline_timestep;
    delete pipeline_tta_flip;
    delete pipeline_tta_accumulate;
    delete pipeline_tta_resolve;
    delete pipeline_mc_luma;
    delete pipeline_mc_down;
    delete pipeline_mc_search;
    delete pipeline_mc_median;
    delete pipeline_mc_recon;
    delete pipeline_mc_filter;
    delete pipeline_mc_region;
    delete pipeline_mc_blend;
    delete pipeline_trt_pack;
    dummy.release();
    hints_gpu.release();
    cached_field.release();

    flownet.clear();

    if (vkdev)
    {
        delete blob_vkallocator;
        if (staging_vkallocator)
            vkdev->reclaim_staging_allocator(staging_vkallocator);
    }
}

bool RifeEngine::init(int gpu_index, bool fp32, int num_threads, std::string& error)
{
    const int gpu_count = ncnn::get_gpu_count();
    if (gpu_count <= 0)
    {
        error = "aucun périphérique Vulkan disponible";
        return false;
    }
    if (gpu_index < 0)
        gpu_index = ncnn::get_default_gpu_index();
    if (gpu_index >= gpu_count)
    {
        error = "index GPU hors limites (" + std::to_string(gpu_index) + " / " + std::to_string(gpu_count) + ")";
        return false;
    }

    gpu = gpu_index;
    vkdev = ncnn::get_gpu_device(gpu_index);
    if (!vkdev)
    {
        error = "initialisation du périphérique Vulkan impossible";
        return false;
    }

    const bool fp16 = !fp32 && vkdev->info.support_fp16_storage();

    opt = ncnn::Option();
    opt.num_threads = std::max(1, num_threads);
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = fp16;
    opt.use_fp16_storage = fp16;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;

    blob_vkallocator = new TrackedBlobAllocator(vkdev);
    staging_vkallocator = vkdev->acquire_staging_allocator();
    return true;
}

bool RifeEngine::load_model(const std::filesystem::path& dir, int _padding, bool uhd, std::string& error)
{
    padding = std::max(1, _padding);

    flownet.opt = opt;
    flownet.set_vulkan_device(vkdev);
    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);

    const std::filesystem::path param_path = dir / "flownet.param";
    const std::filesystem::path model_path = dir / "flownet.bin";

    FILE* fp = open_binary(param_path);
    if (!fp)
    {
        error = "modèle introuvable : " + param_path.u8string();
        return false;
    }
    std::string param;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
        param.append(buf, n);
    fclose(fp);
    if (uhd)
    {
        std::string rewritten;
        if (!make_uhd_param(param, rewritten, error))
        {
            error = "mode UHD : " + error + " (" + param_path.u8string() + ")";
            return false;
        }
        param.swap(rewritten);
    }
    int ret = flownet.load_param_mem(param.c_str());
    if (ret != 0)
    {
        error = "flownet.param invalide : " + param_path.u8string();
        return false;
    }

    fp = open_binary(model_path);
    if (!fp)
    {
        error = "poids introuvables : " + model_path.u8string();
        return false;
    }
    ret = flownet.load_model(fp);
    fclose(fp);
    if (ret != 0)
    {
        error = "flownet.bin invalide : " + model_path.u8string();
        return false;
    }

    const std::vector<const char*>& inputs = flownet.input_names();
    const bool has_timestep = std::any_of(inputs.begin(), inputs.end(), [](const char* n) { return strcmp(n, "in2") == 0; });
    if (!has_timestep)
    {
        error = "modèle RIFE v4 requis (entrée in2 absente) : " + dir.u8string();
        return false;
    }
    return true;
}

ncnn::Pipeline* RifeEngine::make_pipeline(const char* comp_data, int comp_size, int lx, int ly, int lz)
{
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(comp_data, comp_size, opt, spirv) != 0)
        return 0;

    ncnn::Pipeline* pipeline = new ncnn::Pipeline(vkdev);
    pipeline->set_optimal_local_size_xyz(lx, ly, lz);
    if (pipeline->create(spirv.data(), spirv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        delete pipeline;
        return 0;
    }
    return pipeline;
}

bool RifeEngine::configure(const FrameFormat& _fmt, const ColorParams& _color, std::string& error)
{
    fmt = _fmt;
    color = _color;

    // Padding fourni par l'utilisateur : calcul en 64 bits avant conversion
    // vers les dimensions int de ncnn et les index des shaders.
    const int64_t padded_w = ((int64_t)fmt.width + padding - 1) / padding * padding;
    const int64_t padded_h = ((int64_t)fmt.height + padding - 1) / padding * padding;
    if (padded_w > INT32_MAX || padded_h > INT32_MAX
            || padded_w * padded_h > Y4M_MAX_LUMA_SAMPLES)
    {
        error = "dimensions après padding hors limites du moteur";
        return false;
    }
    w_padded = (int)padded_w;
    h_padded = (int)padded_h;
    words = (int)(fmt.padded_frame_bytes() / 4);
    pack_dispatch_w = std::min(words, 8192);

    pipeline_yuv_to_rgb = make_pipeline(yuv_to_rgb_comp_data, sizeof(yuv_to_rgb_comp_data), 8, 8, 1);
    pipeline_rgb_to_yuv = make_pipeline(rgb_to_yuv_comp_data, sizeof(rgb_to_yuv_comp_data), 8, 8, 1);
    pipeline_pack = make_pipeline(pack_samples_comp_data, sizeof(pack_samples_comp_data), 64, 4, 1);
    pipeline_timestep = make_pipeline(rife_v4_timestep_comp_data, sizeof(rife_v4_timestep_comp_data), 8, 8, 1);

    if (!pipeline_yuv_to_rgb || !pipeline_rgb_to_yuv || !pipeline_pack || !pipeline_timestep)
    {
        error = "compilation des shaders de conversion impossible";
        return false;
    }
    if (tta > 1)
    {
        pipeline_tta_flip = make_pipeline(tta_flip_comp_data, sizeof(tta_flip_comp_data), 8, 8, 1);
        pipeline_tta_accumulate = make_pipeline(tta_accumulate_comp_data, sizeof(tta_accumulate_comp_data), 8, 8, 1);
        pipeline_tta_resolve = make_pipeline(tta_resolve_comp_data, sizeof(tta_resolve_comp_data), 8, 8, 1);
    }
    if (tta > 1 && (!pipeline_tta_flip || !pipeline_tta_accumulate || !pipeline_tta_resolve))
    {
        error = "compilation des shaders TTA impossible";
        return false;
    }
    // copie vers les tampons partagés du plugin TensorRT (utilisée seulement s'il est actif)
    pipeline_trt_pack = make_pipeline(trt_pack_comp_data, sizeof(trt_pack_comp_data), 8, 8, 1);
    if (!pipeline_trt_pack)
    {
        error = "compilation du shader de copie TensorRT impossible";
        return false;
    }
    if (engine_mode != InterpEngine::Rife)
    {
        pipeline_mc_luma = make_pipeline(mc_luma_comp_data, sizeof(mc_luma_comp_data), 8, 8, 1);
        pipeline_mc_down = make_pipeline(mc_down_comp_data, sizeof(mc_down_comp_data), 8, 8, 1);
        pipeline_mc_search = make_pipeline(mc_search_comp_data, sizeof(mc_search_comp_data), 16, 16, 1);
        pipeline_mc_median = make_pipeline(mc_median_comp_data, sizeof(mc_median_comp_data), 8, 8, 1);
        pipeline_mc_recon = make_pipeline(mc_recon_comp_data, sizeof(mc_recon_comp_data), 8, 8, 1);
        pipeline_mc_filter = make_pipeline(mc_filter_comp_data, sizeof(mc_filter_comp_data), 8, 8, 1);
        pipeline_mc_region = make_pipeline(mc_region_comp_data, sizeof(mc_region_comp_data), 64, 1, 1);
        pipeline_mc_blend = make_pipeline(mc_blend_comp_data, sizeof(mc_blend_comp_data), 8, 8, 1);
        if (!pipeline_mc_luma || !pipeline_mc_down || !pipeline_mc_search || !pipeline_mc_median
                || !pipeline_mc_recon || !pipeline_mc_filter || !pipeline_mc_region || !pipeline_mc_blend)
        {
            error = "compilation des shaders de compensation de mouvement impossible";
            return false;
        }
        lw[0] = w_padded;
        lh[0] = h_padded;
        for (int l = 1; l < 4; l++)
        {
            lw[l] = (lw[l - 1] + 1) / 2;
            lh[l] = (lh[l - 1] + 1) / 2;
        }
        dummy.create(4, (size_t)4u, 1, blob_vkallocator);
        if (dummy.empty())
        {
            error = "allocation mémoire GPU impossible (MC)";
            return false;
        }
    }
    return true;
}

uint32_t RifeEngine::vendor_id() const
{
    return vkdev ? vkdev->info.vendor_id() : 0;
}

static float median_of(std::vector<float>& v)
{
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

void RifeEngine::set_flow_hints(const int16_t* fwd, const int16_t* bwd, int gw, int gh, int grid, float vscale)
{
    // grid : taille de cellule en pixels pleine résolution ; vscale : facteur des vecteurs
    const int bw = (lw[0] + 15) / 16;
    const int bh = (lh[0] + 15) / 16;
    const int cells = std::max(1, 16 / grid);
    hints_host.assign((size_t)bw * bh * 4, 0.f);
    std::vector<float> fx, fy, bx_, by_;
    for (int by = 0; by < bh; by++)
    {
        for (int bx = 0; bx < bw; bx++)
        {
            fx.clear(); fy.clear(); bx_.clear(); by_.clear();
            for (int j = 0; j < cells; j++)
            {
                const int gy = by * cells + j;
                if (gy >= gh)
                    break;
                for (int i = 0; i < cells; i++)
                {
                    const int gx = bx * cells + i;
                    if (gx >= gw)
                        break;
                    const size_t k = ((size_t)gy * gw + gx) * 2;
                    fx.push_back(vscale * fwd[k] / 32.f);
                    fy.push_back(vscale * fwd[k + 1] / 32.f);
                    bx_.push_back(-vscale * bwd[k] / 32.f);
                    by_.push_back(-vscale * bwd[k + 1] / 32.f);
                }
            }
            float* dst = &hints_host[((size_t)by * bw + bx) * 4];
            if (fx.empty())
                continue;   // bloc entièrement dans le padding : vecteur nul
            dst[0] = median_of(fx);
            dst[1] = median_of(fy);
            dst[2] = median_of(bx_);
            dst[3] = median_of(by_);
        }
    }
    hints_dirty = true;
}

ncnn::VkMat RifeEngine::float_mat(int n)
{
    ncnn::VkMat m;
    m.create(n, (size_t)4u, 1, blob_vkallocator);
    return m;
}

void RifeEngine::record_filter(ncnn::VkCompute& cmd, const ncnn::VkMat& src, ncnn::VkMat& dst, int r, int axis, int op,
                               int threshold, float thr)
{
    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = src;
    bindings[1] = dst;
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].i = w_padded;
    constants[1].i = h_padded;
    constants[2].i = r;
    constants[3].i = axis;
    constants[4].i = op;
    constants[5].i = threshold;
    constants[6].f = thr;
    ncnn::VkMat dispatcher;
    dispatcher.w = w_padded;
    dispatcher.h = h_padded;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_mc_filter, bindings, constants, dispatcher);
}

// Compensation de mouvement : pyramide bilatérale par blocs de 16 (1/8 → 1/1), reconstruction
// recouvrante, puis, en mode hybride, choix par région entre MC et RIFE (rife non vide).
bool RifeEngine::record_mc(ncnn::VkCompute& cmd, const GpuFrame& a, const GpuFrame& b, float t, const ncnn::VkMat& rife,
                           ncnn::VkMat& out, std::string& error)
{
    const int plane = w_padded * h_padded;
    // Champ unique par paire : recherche au milieu de la paire (t = 0,5), réutilisée pour toutes
    // les positions t (un tiers des recherches évité en 24 → 60, qualité égale aux bancs).
    const bool reuse = !cached_field.empty() && a.id == cache_a && b.id == cache_b;
    ncnn::VkMat field[4];
    if (reuse)
        field[0] = cached_field;
    for (int l = 3; l >= 0 && !reuse; l--)
    {
        const int bw = (lw[l] + 15) / 16;
        const int bh = (lh[l] + 15) / 16;
        const int iw = (fmt.width + (1 << l) - 1) >> l;
        const int ih = (fmt.height + (1 << l) - 1) >> l;
        ncnn::VkMat raw = float_mat(bw * bh * 2);
        if (raw.empty())
        {
            error = "allocation mémoire GPU impossible (vecteurs)";
            return false;
        }
        const bool coarse = l == 3;
        std::vector<ncnn::VkMat> bindings(5);
        bindings[0] = a.lum[l];
        bindings[1] = b.lum[l];
        bindings[2] = coarse ? dummy : field[l + 1];
        bindings[3] = raw;
        const bool hinted = l == 0 && !hints_host.empty() && (int)hints_host.size() == bw * bh * 4;
        if (hinted && hints_dirty)
        {
            ncnn::Option o = opt;
            o.blob_vkallocator = blob_vkallocator;
            o.workspace_vkallocator = blob_vkallocator;
            o.staging_vkallocator = staging_vkallocator;
            ncnn::Mat host((int)hints_host.size(), (void*)hints_host.data(), (size_t)4u, 1);
            cmd.record_clone(host, hints_gpu, o);
            hints_dirty = false;
        }
        bindings[4] = hinted ? hints_gpu : dummy;
        std::vector<ncnn::vk_constant_type> constants(15);
        constants[0].i = lw[l];
        constants[1].i = lh[l];
        constants[2].i = iw;
        constants[3].i = ih;
        constants[4].i = bw;
        constants[5].i = bh;
        constants[6].i = coarse ? 1 : (lw[l + 1] + 15) / 16;
        constants[7].i = coarse ? 1 : (lh[l + 1] + 15) / 16;
        constants[8].i = coarse ? 0 : 1;
        constants[9].i = MC_COARSE_RADIUS;
        constants[10].i = l == 0 ? 1 : 0;
        constants[11].i = hinted ? 2 : 0;
        constants[12].f = coarse ? MC_LAMBDA_COARSE : MC_LAMBDA;
        constants[13].f = 0.5f;
        constants[14].i = l <= 1 ? MC_MARGIN_FINE : MC_MARGIN_COARSE;
        ncnn::VkMat dispatcher;
        dispatcher.w = bw * 16;
        dispatcher.h = bh * 16;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_mc_search, bindings, constants, dispatcher);

        if (l == 0)
        {
            field[l] = raw;
            continue;
        }
        // niveaux grossiers : médiane 3×3, puis passes de propagation (voisins du même niveau,
        // prédicteur = médiane) qui corrigent les régions prises sur une mauvaise période d'un
        // motif répétitif
        field[l] = float_mat(bw * bh * 2);
        {
            std::vector<ncnn::VkMat> mb(2);
            mb[0] = raw;
            mb[1] = field[l];
            std::vector<ncnn::vk_constant_type> mc(2);
            mc[0].i = bw;
            mc[1].i = bh;
            ncnn::VkMat md;
            md.w = bw;
            md.h = bh;
            md.c = 1;
            cmd.record_pipeline(pipeline_mc_median, mb, mc, md);
        }
        for (int it = 0; it < MC_PROP_PASSES; it++)
        {
            ncnn::VkMat next = float_mat(bw * bh * 2);
            if (field[l].empty() || next.empty())
            {
                error = "allocation mémoire GPU impossible (vecteurs)";
                return false;
            }
            std::vector<ncnn::VkMat> pb(5);
            pb[0] = a.lum[l];
            pb[1] = b.lum[l];
            pb[2] = field[l];
            pb[3] = next;
            pb[4] = dummy;
            std::vector<ncnn::vk_constant_type> pc = constants;
            pc[6].i = bw;
            pc[7].i = bh;
            pc[8].i = 2;
            pc[11].i = 0;
            pc[12].f = MC_LAMBDA_PROP;
            cmd.record_pipeline(pipeline_mc_search, pb, pc, dispatcher);
            field[l] = next;
        }
    }

    if (!reuse)
    {
        cached_field = field[0];
        cache_a = a.id;
        cache_b = b.id;
    }
    ncnn::VkMat mc = float_mat(plane * 3);
    ncnn::VkMat q = float_mat(plane);
    ncnn::VkMat edge = float_mat(plane);
    out.create(w_padded, h_padded, 3, a.rgb.elemsize, 1, blob_vkallocator);
    if (mc.empty() || q.empty() || edge.empty() || out.empty() || out.cstep != a.rgb.cstep)
    {
        error = "allocation mémoire GPU impossible (reconstruction MC)";
        return false;
    }
    {
        std::vector<ncnn::VkMat> bindings(6);
        bindings[0] = a.rgb;
        bindings[1] = b.rgb;
        bindings[2] = field[0];
        bindings[3] = mc;
        bindings[4] = q;
        bindings[5] = edge;
        std::vector<ncnn::vk_constant_type> constants(11);
        constants[0].i = w_padded;
        constants[1].i = h_padded;
        constants[2].i = (int)a.rgb.cstep;
        constants[3].i = fmt.width;
        constants[4].i = fmt.height;
        constants[5].i = (lw[0] + 15) / 16;
        constants[6].i = (lh[0] + 15) / 16;
        constants[7].f = t;
        constants[8].f = color.kr;
        constants[9].f = color.kb;
        constants[10].f = MC_LUM_SCALE;
        ncnn::VkMat dispatcher;
        dispatcher.w = w_padded;
        dispatcher.h = h_padded;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_mc_recon, bindings, constants, dispatcher);
    }

    const bool hybrid = !rife.empty();
    ncnn::VkMat u = dummy;
    ncnn::VkMat diffs = dummy;
    if (hybrid)
    {
        if (rife.w != w_padded || rife.h != h_padded || rife.c != 3 || rife.cstep != out.cstep || rife.elemsize != out.elemsize)
        {
            error = "sortie RIFE incompatible avec le mode hybride";
            return false;
        }
        ncnn::VkMat t1 = float_mat(plane), qs = float_mat(plane), dens = float_mat(plane), edged = float_mat(plane);
        ncnn::VkMat u0 = float_mat(plane);
        u = float_mat(plane);
        ncnn::VkMat diff = float_mat(plane);
        diffs = float_mat(plane);
        if (t1.empty() || qs.empty() || dens.empty() || edged.empty() || u0.empty() || u.empty() || diff.empty() || diffs.empty())
        {
            error = "allocation mémoire GPU impossible (hybride)";
            return false;
        }
        record_filter(cmd, q, t1, 4, 0, 0, 0, 0.f);        // erreur lissée 9×9
        record_filter(cmd, t1, qs, 4, 1, 0, 0, 0.f);
        record_filter(cmd, qs, t1, 15, 0, 0, 1, MC_Q_THRESHOLD);  // densité de pixels non fiables, 31×31
        record_filter(cmd, t1, dens, 15, 1, 0, 0, 0.f);
        record_filter(cmd, edge, t1, 16, 0, 1, 0, 0.f);    // bord de cadre dilaté 33×33
        record_filter(cmd, t1, edged, 16, 1, 1, 0, 0.f);
        {
            std::vector<ncnn::VkMat> bindings(3);
            bindings[0] = dens;
            bindings[1] = edged;
            bindings[2] = u0;
            std::vector<ncnn::vk_constant_type> constants(2);
            constants[0].i = plane;
            constants[1].f = MC_DENSITY;
            ncnn::VkMat dispatcher;
            dispatcher.w = plane;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_mc_region, bindings, constants, dispatcher);
        }
        record_filter(cmd, u0, t1, 7, 0, 0, 0, 0.f);       // transition adoucie 15×15
        record_filter(cmd, t1, u, 7, 1, 0, 0, 0.f);
        {
            std::vector<ncnn::VkMat> bindings(5);
            bindings[0] = mc;
            bindings[1] = rife;
            bindings[2] = dummy;
            bindings[3] = diff;
            bindings[4] = out;
            std::vector<ncnn::vk_constant_type> constants(8);
            constants[0].i = w_padded;
            constants[1].i = h_padded;
            constants[2].i = (int)out.cstep;
            constants[3].i = 1;
            constants[4].f = color.kr;
            constants[5].f = color.kb;
            constants[6].f = MC_LUM_SCALE;
            constants[7].f = MC_AGREE;
            ncnn::VkMat dispatcher;
            dispatcher.w = w_padded;
            dispatcher.h = h_padded;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_mc_blend, bindings, constants, dispatcher);
        }
        record_filter(cmd, diff, t1, 2, 0, 0, 0, 0.f);     // écart MC / RIFE lissé 5×5
        record_filter(cmd, t1, diffs, 2, 1, 0, 0, 0.f);
    }
    {
        std::vector<ncnn::VkMat> bindings(5);
        bindings[0] = mc;
        bindings[1] = hybrid ? rife : out;
        bindings[2] = u;
        bindings[3] = diffs;
        bindings[4] = out;
        std::vector<ncnn::vk_constant_type> constants(8);
        constants[0].i = w_padded;
        constants[1].i = h_padded;
        constants[2].i = (int)out.cstep;
        constants[3].i = hybrid ? 2 : 0;
        constants[4].f = color.kr;
        constants[5].f = color.kb;
        constants[6].f = MC_LUM_SCALE;
        constants[7].f = MC_AGREE;
        ncnn::VkMat dispatcher;
        dispatcher.w = w_padded;
        dispatcher.h = h_padded;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_mc_blend, bindings, constants, dispatcher);
    }
    return true;
}

// Paramètres de normalisation des codes YUV selon la plage et la profondeur.
struct CodeScale
{
    float y_off;
    float y_range;
    float c_off;
    float c_range;
};

static CodeScale code_scale(const FrameFormat& fmt, bool full_range)
{
    const float shift = (float)(1 << (fmt.bit_depth - 8));
    CodeScale s;
    if (full_range)
    {
        const float maxcode = (float)((1 << fmt.bit_depth) - 1);
        s.y_off = 0.f;
        s.y_range = maxcode;
        s.c_off = (float)(1 << (fmt.bit_depth - 1));
        s.c_range = maxcode;
    }
    else
    {
        s.y_off = 16.f * shift;
        s.y_range = 219.f * shift;
        s.c_off = 128.f * shift;
        s.c_range = 224.f * shift;
    }
    return s;
}

bool RifeEngine::upload(const uint8_t* frame, GpuFrame& out, std::string& error)
{
    const size_t elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Option o = opt;
    o.blob_vkallocator = blob_vkallocator;
    o.workspace_vkallocator = blob_vkallocator;
    o.staging_vkallocator = staging_vkallocator;

    ncnn::VkCompute cmd(vkdev);

    ncnn::Mat raw(words, (void*)frame, (size_t)4u, 1);
    ncnn::VkMat raw_gpu;
    cmd.record_clone(raw, raw_gpu, o);

    out.rgb.create(w_padded, h_padded, 3, elemsize, 1, blob_vkallocator);
    if (raw_gpu.empty() || out.rgb.empty())
    {
        error = "allocation mémoire GPU impossible (upload)";
        return false;
    }

    const CodeScale s = code_scale(fmt, color.full_range);

    std::vector<ncnn::vk_constant_type> constants(18);
    constants[0].i = fmt.width;
    constants[1].i = fmt.height;
    constants[2].i = fmt.chroma_width();
    constants[3].i = fmt.chroma_height();
    constants[4].i = (int)fmt.luma_samples();
    constants[5].i = (int)(fmt.luma_samples() + fmt.chroma_samples());
    constants[6].i = fmt.bytes_per_sample;
    constants[7].i = out.rgb.w;
    constants[8].i = out.rgb.h;
    constants[9].i = (int)out.rgb.cstep;
    constants[10].i = chroma_mode_x();
    constants[11].i = chroma_mode_y();
    constants[12].f = s.y_off;
    constants[13].f = 1.f / s.y_range;
    constants[14].f = s.c_off;
    constants[15].f = 1.f / s.c_range;
    constants[16].f = color.kr;
    constants[17].f = color.kb;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = raw_gpu;
    bindings[1] = out.rgb;

    ncnn::VkMat dispatcher;
    dispatcher.w = out.rgb.w;
    dispatcher.h = out.rgb.h;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_yuv_to_rgb, bindings, constants, dispatcher);

    if (engine_mode != InterpEngine::Rife)
    {
        // pyramide de luminance, calculée une fois par trame source (réutilisée par deux paires)
        for (int l = 0; l < 4; l++)
        {
            out.lum[l] = float_mat(lw[l] * lh[l]);
            if (out.lum[l].empty())
            {
                error = "allocation mémoire GPU impossible (pyramide)";
                return false;
            }
        }
        {
            std::vector<ncnn::VkMat> lb(2);
            lb[0] = out.rgb;
            lb[1] = out.lum[0];
            std::vector<ncnn::vk_constant_type> lc(6);
            lc[0].i = w_padded;
            lc[1].i = h_padded;
            lc[2].i = (int)out.rgb.cstep;
            lc[3].f = color.kr;
            lc[4].f = color.kb;
            lc[5].f = MC_LUM_SCALE;
            ncnn::VkMat ld;
            ld.w = w_padded;
            ld.h = h_padded;
            ld.c = 1;
            cmd.record_pipeline(pipeline_mc_luma, lb, lc, ld);
        }
        for (int l = 1; l < 4; l++)
        {
            std::vector<ncnn::VkMat> db(2);
            db[0] = out.lum[l - 1];
            db[1] = out.lum[l];
            std::vector<ncnn::vk_constant_type> dc(4);
            dc[0].i = lw[l - 1];
            dc[1].i = lh[l - 1];
            dc[2].i = lw[l];
            dc[3].i = lh[l];
            ncnn::VkMat dd;
            dd.w = lw[l];
            dd.h = lh[l];
            dd.c = 1;
            cmd.record_pipeline(pipeline_mc_down, db, dc, dd);
        }
    }

    if (cmd.submit_and_wait() != 0)
    {
        error = "échec d'exécution GPU (conversion YUV -> RGB)";
        return false;
    }
    out.id = ++upload_seq;
    out.ready = true;
    return true;
}

bool RifeEngine::record_timestep(ncnn::VkCompute& cmd, float t, ncnn::VkMat& timestep, std::string& error)
{
    // carte de temps constante (entrée in2 du réseau v4)
    timestep.create(w_padded, h_padded, 1, opt.use_fp16_storage ? 2u : 4u, 1, blob_vkallocator);
    if (timestep.empty())
    {
        error = "allocation mémoire GPU impossible (timestep)";
        return false;
    }
    std::vector<ncnn::VkMat> bindings(1);
    bindings[0] = timestep;

    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].i = timestep.w;
    constants[1].i = timestep.h;
    constants[2].i = (int)timestep.cstep;
    constants[3].f = t;

    cmd.record_pipeline(pipeline_timestep, bindings, constants, timestep);
    return true;
}

bool RifeEngine::record_network(ncnn::VkCompute& cmd, const ncnn::VkMat& in0, const ncnn::VkMat& in1,
                                const ncnn::VkMat& timestep, ncnn::VkMat& out, std::string& error)
{
    ncnn::Extractor ex = flownet.create_extractor();
    ex.set_blob_vkallocator(blob_vkallocator);
    ex.set_workspace_vkallocator(blob_vkallocator);
    ex.set_staging_vkallocator(staging_vkallocator);

    ex.input("in0", in0);
    ex.input("in1", in1);
    ex.input("in2", timestep);
    if (ex.extract("out0", out, cmd) != 0 || out.empty())
    {
        error = "échec d'inférence RIFE";
        return false;
    }
    return true;
}

bool RifeEngine::infer_rife(ncnn::VkCompute& cmd, const ncnn::VkMat& in0, const ncnn::VkMat& in1, float t,
                            ncnn::VkMat& out, std::string& error)
{
    if (trt)
    {
        // Entrées copiées dans les tampons partagés (plans contigus), Vulkan terminé avant l'inférence CUDA.
        for (int i = 0; i < 2; i++)
        {
            const ncnn::VkMat& src = i == 0 ? in0 : in1;
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = src;
            bindings[1] = trt->input(i);
            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = w_padded;
            constants[1].i = h_padded;
            constants[2].i = (int)src.cstep;
            constants[3].i = (int)trt->input(i).cstep;
            ncnn::VkMat dispatcher;
            dispatcher.w = w_padded;
            dispatcher.h = h_padded;
            dispatcher.c = 3;
            cmd.record_pipeline(pipeline_trt_pack, bindings, constants, dispatcher);
        }
        if (cmd.submit_and_wait() != 0)
        {
            error = "échec d'exécution GPU (entrées TensorRT)";
            return false;
        }
        cmd.reset();
        std::string trt_error;
        if (trt->infer(t, trt_error))
        {
            out = trt->output();
            return true;
        }
        // Jamais d'échec d'encodage à cause du plugin : inférence ncnn pour la suite.
        fprintf(stderr, "warning: inférence TensorRT désactivée (%s) : inférence Vulkan pour la suite\n", trt_error.c_str());
        trt = nullptr;
    }
    ncnn::VkMat timestep;
    return record_timestep(cmd, t, timestep, error) && record_network(cmd, in0, in1, timestep, out, error);
}

bool RifeEngine::interpolate(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error,
                             const std::function<void()>& before_mc)
{
    if (engine_mode != InterpEngine::Rife)
    {
        ncnn::VkMat rife_rgb;
        if (engine_mode == InterpEngine::Hybrid && tta > 1 && !rife_tta_rgb(a, b, t, rife_rgb, error))
            return false;
        ncnn::VkCompute cmd(vkdev);
        if (engine_mode == InterpEngine::Hybrid && tta <= 1)
        {
            if (!infer_rife(cmd, a.rgb, b.rgb, t, rife_rgb, error))
                return false;
        }
        if (before_mc)
        {
            // RIFE exécuté pendant que l'appelant termine ses candidats (flux NVOF)
            if (cmd.submit_and_wait() != 0)
            {
                error = "échec d'exécution GPU (RIFE)";
                return false;
            }
            cmd.reset();
            before_mc();
        }
        ncnn::VkMat out_rgb;
        return record_mc(cmd, a, b, t, rife_rgb, out_rgb, error) && convert_and_download(cmd, out_rgb, dst, error);
    }
    if (tta > 1)
        return interpolate_tta(a, b, t, dst, error);

    ncnn::VkCompute cmd(vkdev);
    ncnn::VkMat out_rgb;
    if (!infer_rife(cmd, a.rgb, b.rgb, t, out_rgb, error))
        return false;
    return convert_and_download(cmd, out_rgb, dst, error);
}

bool RifeEngine::record_flip(ncnn::VkCompute& cmd, const ncnn::VkMat& src, int flip, ncnn::VkMat& dst, std::string& error)
{
    // nouveau tampon : create() réutiliserait celui d'une matrice de mêmes dimensions (écriture sur la source)
    dst.release();
    dst.create(src.w, src.h, src.c, src.elemsize, 1, blob_vkallocator);
    if (dst.empty())
    {
        error = "allocation mémoire GPU impossible (TTA)";
        return false;
    }
    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = src;
    bindings[1] = dst;

    std::vector<ncnn::vk_constant_type> constants(8);
    constants[0].i = src.w;
    constants[1].i = src.h;
    constants[2].i = src.c;
    constants[3].i = src.w;
    constants[4].i = (int)src.cstep;
    constants[5].i = dst.w;
    constants[6].i = (int)dst.cstep;
    constants[7].i = flip;

    cmd.record_pipeline(pipeline_tta_flip, bindings, constants, dst);
    return true;
}

// Moyenne des variantes (retournement x sens temporel). Une soumission par
// variante : la VRAM reste celle d'une seule inférence (+ accumulateur fp32).
bool RifeEngine::interpolate_tta(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error)
{
    ncnn::VkMat merged;
    if (!rife_tta_rgb(a, b, t, merged, error))
        return false;
    ncnn::VkCompute cmd(vkdev);
    return convert_and_download(cmd, merged, dst, error);
}

bool RifeEngine::rife_tta_rgb(const GpuFrame& a, const GpuFrame& b, float t, ncnn::VkMat& merged, std::string& error)
{
    const int nflips = tta >= 8 ? 4 : (tta >= 4 ? 2 : 1);
    const int variants = nflips * 2;

    ncnn::VkMat acc;
    acc.create(w_padded, h_padded, 3, 4u, 1, blob_vkallocator);
    if (acc.empty())
    {
        error = "allocation mémoire GPU impossible (TTA)";
        return false;
    }

    for (int i = 0; i < variants; i++)
    {
        const int flip = i / 2;        // 0 : aucun, 1 : horizontal, 2 : vertical, 3 : les deux
        const bool reverse = (i % 2) == 1;

        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat in0;
        ncnn::VkMat in1;
        if (!flip)
        {
            in0 = a.rgb;
            in1 = b.rgb;
        }
        else if (!record_flip(cmd, a.rgb, flip, in0, error) || !record_flip(cmd, b.rgb, flip, in1, error))
            return false;
        if (reverse)
            std::swap(in0, in1);

        ncnn::VkMat out;
        if (!infer_rife(cmd, in0, in1, reverse ? 1.f - t : t, out, error))
            return false;
        if (out.w != w_padded || out.h != h_padded || out.c != 3)
        {
            error = "sortie RIFE de dimensions inattendues (TTA)";
            return false;
        }

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = out;
        bindings[1] = acc;

        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].i = out.w;
        constants[1].i = out.h;
        constants[2].i = 3;
        constants[3].i = out.w;
        constants[4].i = (int)out.cstep;
        constants[5].i = acc.w;
        constants[6].i = (int)acc.cstep;
        constants[7].i = flip;
        constants[8].i = i == 0 ? 1 : 0;

        cmd.record_pipeline(pipeline_tta_accumulate, bindings, constants, acc);
        if (cmd.submit_and_wait() != 0)
        {
            error = "échec d'exécution GPU (TTA)";
            return false;
        }
    }

    ncnn::VkCompute cmd(vkdev);
    merged.create(w_padded, h_padded, 3, opt.use_fp16_storage ? 2u : 4u, 1, blob_vkallocator);
    if (merged.empty())
    {
        error = "allocation mémoire GPU impossible (TTA)";
        return false;
    }
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = acc;
        bindings[1] = merged;

        std::vector<ncnn::vk_constant_type> constants(8);
        constants[0].i = acc.w;
        constants[1].i = acc.h;
        constants[2].i = 3;
        constants[3].i = acc.w;
        constants[4].i = (int)acc.cstep;
        constants[5].i = merged.w;
        constants[6].i = (int)merged.cstep;
        constants[7].f = 1.f / (float)variants;

        cmd.record_pipeline(pipeline_tta_resolve, bindings, constants, merged);
    }
    if (cmd.submit_and_wait() != 0)
    {
        error = "échec d'exécution GPU (TTA)";
        return false;
    }
    return true;
}

bool RifeEngine::roundtrip(const GpuFrame& a, uint8_t* dst, std::string& error)
{
    ncnn::VkCompute cmd(vkdev);
    return convert_and_download(cmd, a.rgb, dst, error);
}

bool RifeEngine::convert_and_download(ncnn::VkCompute& cmd, const ncnn::VkMat& out_rgb, uint8_t* dst, std::string& error)
{
    ncnn::Option o = opt;
    o.blob_vkallocator = blob_vkallocator;
    o.workspace_vkallocator = blob_vkallocator;
    o.staging_vkallocator = staging_vkallocator;

    const CodeScale s = code_scale(fmt, color.full_range);

    ncnn::VkMat yuvf;
    yuvf.create((int)fmt.total_samples(), (size_t)4u, 1, blob_vkallocator);
    ncnn::VkMat packed;
    packed.create(words, (size_t)4u, 1, blob_vkallocator);
    if (yuvf.empty() || packed.empty())
    {
        error = "allocation mémoire GPU impossible (download)";
        return false;
    }
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = out_rgb;
        bindings[1] = yuvf;

        std::vector<ncnn::vk_constant_type> constants(16);
        constants[0].i = fmt.width;
        constants[1].i = fmt.height;
        constants[2].i = fmt.chroma_width();
        constants[3].i = fmt.chroma_height();
        constants[4].i = (int)fmt.luma_samples();
        constants[5].i = (int)(fmt.luma_samples() + fmt.chroma_samples());
        constants[6].i = out_rgb.w;
        constants[7].i = (int)out_rgb.cstep;
        constants[8].i = chroma_mode_x();
        constants[9].i = chroma_mode_y();
        constants[10].f = s.y_off;
        constants[11].f = s.y_range;
        constants[12].f = s.c_off;
        constants[13].f = s.c_range;
        constants[14].f = color.kr;
        constants[15].f = color.kb;

        ncnn::VkMat dispatcher;
        dispatcher.w = fmt.width;
        dispatcher.h = fmt.height;
        dispatcher.c = 2;
        cmd.record_pipeline(pipeline_rgb_to_yuv, bindings, constants, dispatcher);
    }

    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = yuvf;
        bindings[1] = packed;

        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].i = (int)fmt.total_samples();
        constants[1].i = words;
        constants[2].i = fmt.bytes_per_sample;
        constants[3].i = pack_dispatch_w;
        constants[4].f = (float)((1 << fmt.bit_depth) - 1);

        ncnn::VkMat dispatcher;
        dispatcher.w = pack_dispatch_w;
        dispatcher.h = (words + pack_dispatch_w - 1) / pack_dispatch_w;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_pack, bindings, constants, dispatcher);
    }

    ncnn::Mat out(words, (void*)dst, (size_t)4u, 1);
    cmd.record_clone(packed, out, o);

    if (cmd.submit_and_wait() != 0)
    {
        error = "échec d'exécution GPU (conversion RGB -> YUV)";
        return false;
    }
    if (out.data != (void*)dst)
    {
        // dst contient padded_frame_bytes() octets (contrat du pool de trames).
        const size_t frame_bytes = fmt.padded_frame_bytes();
        if (out.empty() || out.total() * out.elemsize < frame_bytes)
        {
            error = "buffer de sortie GPU incomplet";
            return false;
        }
        // ncnn a réalloué la destination : copie bornée à la taille de la trame.
        std::copy_n(static_cast<const uint8_t*>(out.data), frame_bytes, dst);
    }
    return true;
}

// Mode de rééchantillonnage chroma par axe (0 : aucun, 1 : co-situé, 2 : centré).
int RifeEngine::chroma_mode_x() const
{
    if (fmt.sub_x == 1)
        return 0;
    return color.siting == ChromaSiting::Center ? 2 : 1;
}

int RifeEngine::chroma_mode_y() const
{
    if (fmt.sub_y == 1)
        return 0;
    return color.siting == ChromaSiting::TopLeft ? 1 : 2;
}

std::string RifeEngine::device_name() const
{
    return vkdev ? std::string(vkdev->info.device_name()) : std::string();
}
