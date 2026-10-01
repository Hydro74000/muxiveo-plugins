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

#include "pack_samples.comp.hex.h"
#include "rgb_to_yuv.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"
#include "yuv_to_rgb.comp.hex.h"

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
      padding(32), w_padded(0), h_padded(0), words(0), pack_dispatch_w(0)
{
}

RifeEngine::~RifeEngine()
{
    delete pipeline_yuv_to_rgb;
    delete pipeline_rgb_to_yuv;
    delete pipeline_pack;
    delete pipeline_timestep;

    flownet.clear();

    if (vkdev)
    {
        if (blob_vkallocator)
            vkdev->reclaim_blob_allocator(blob_vkallocator);
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

    blob_vkallocator = vkdev->acquire_blob_allocator();
    staging_vkallocator = vkdev->acquire_staging_allocator();
    return true;
}

bool RifeEngine::load_model(const std::filesystem::path& dir, int _padding, std::string& error)
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
    int ret = flownet.load_param(fp);
    fclose(fp);
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

    w_padded = (fmt.width + padding - 1) / padding * padding;
    h_padded = (fmt.height + padding - 1) / padding * padding;
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

    if (cmd.submit_and_wait() != 0)
    {
        error = "échec d'exécution GPU (conversion YUV -> RGB)";
        return false;
    }
    out.ready = true;
    return true;
}

bool RifeEngine::interpolate(const GpuFrame& a, const GpuFrame& b, float t, uint8_t* dst, std::string& error)
{
    const size_t elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::VkCompute cmd(vkdev);

    // carte de temps constante (entrée in2 du réseau v4)
    ncnn::VkMat timestep;
    timestep.create(w_padded, h_padded, 1, elemsize, 1, blob_vkallocator);
    {
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = timestep;

        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].i = timestep.w;
        constants[1].i = timestep.h;
        constants[2].i = (int)timestep.cstep;
        constants[3].f = t;

        cmd.record_pipeline(pipeline_timestep, bindings, constants, timestep);
    }

    ncnn::VkMat out_rgb;
    {
        ncnn::Extractor ex = flownet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ex.input("in0", a.rgb);
        ex.input("in1", b.rgb);
        ex.input("in2", timestep);
        if (ex.extract("out0", out_rgb, cmd) != 0 || out_rgb.empty())
        {
            error = "échec d'inférence RIFE";
            return false;
        }
    }

    return convert_and_download(cmd, out_rgb, dst, error);
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

    ncnn::VkMat packed;
    packed.create(words, (size_t)4u, 1, blob_vkallocator);
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
        // ncnn a réalloué la destination : recopie explicite
        memcpy(dst, out.data, (size_t)words * 4u);
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
