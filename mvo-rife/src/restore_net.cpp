// Restauration (R2) : réseau de restauration sur GPU (voir restore_net.h).
#include "restore_net.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "command.h"
#include "layer.h"
#include "mat.h"

#include "restore_assemble.comp.hex.h"
#include "restore_luma_area.comp.hex.h"
#include "restore_resize.comp.hex.h"
#include "restore_rgb_to_yuv_aa.comp.hex.h"
#include "restore_tile.comp.hex.h"
#include "restore_yuv_to_rgb.comp.hex.h"

ncnn::Layer* Warp_layer_creator(void* userdata);  // engine.cpp (couche rife.Warp du moteur)

namespace
{

bool read_file(const std::filesystem::path& path, std::vector<unsigned char>& bytes)
{
#ifdef _WIN32
    FILE* fp = _wfopen(path.wstring().c_str(), L"rb");
#else
    FILE* fp = fopen(path.string().c_str(), "rb");
#endif
    if (!fp)
        return false;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
        bytes.insert(bytes.end(), buf, buf + n);
    fclose(fp);
    return true;
}

bool load_net(ncnn::Net& net, const ncnn::Option& opt, ncnn::VulkanDevice* vkdev, const std::filesystem::path& dir,
              const char* name, std::string& error)
{
    net.opt = opt;
    net.set_vulkan_device(vkdev);
    net.register_custom_layer("rife.Warp", Warp_layer_creator);
    std::vector<unsigned char> param, model;
    const std::filesystem::path pp = dir / (std::string(name) + ".ncnn.param");
    const std::filesystem::path mp = dir / (std::string(name) + ".ncnn.bin");
    if (!read_file(pp, param) || !read_file(mp, model))
    {
        error = "graphe de restauration introuvable : " + pp.u8string();
        return false;
    }
    param.push_back(0);
    if (net.load_param_mem((const char*)param.data()) != 0 || net.load_model(model.data()) <= 0)
    {
        error = "graphe de restauration illisible : " + pp.u8string();
        return false;
    }
    return true;
}

struct Scale
{
    float y_off, y_range, c_off, c_range;
};

Scale code_scale(const FrameFormat& fmt, bool full_range)
{
    const float shift = (float)(1 << (fmt.bit_depth - 8));
    if (full_range)
    {
        const float maxcode = (float)((1 << fmt.bit_depth) - 1);
        return {0.f, maxcode, (float)(1 << (fmt.bit_depth - 1)), maxcode};
    }
    return {16.f * shift, 219.f * shift, 128.f * shift, 224.f * shift};
}

ncnn::VkMat dispatch_of(int w, int h, int c)
{
    ncnn::VkMat d;
    d.w = w;
    d.h = h;
    d.c = c;
    return d;
}

}  // namespace

RestoreNet::~RestoreNet() { release(); }

void RestoreNet::release()
{
    cache.clear();
    for (ncnn::Pipeline** p : {&p_yuv_to_rgb, &p_luma_area, &p_resize, &p_assemble, &p_tile, &p_rgb_to_yuv})
    {
        delete *p;
        *p = nullptr;
    }
    if (warp)
    {
        warp->destroy_pipeline(opt);
        delete warp;
        warp = nullptr;
    }
    flow_net.clear();
    body_net.clear();
    if (vkdev && blob_alloc)
        vkdev->reclaim_blob_allocator(blob_alloc);
    if (vkdev && staging_alloc)
        vkdev->reclaim_staging_allocator(staging_alloc);
    blob_alloc = staging_alloc = nullptr;
}

ncnn::Pipeline* RestoreNet::make_pipeline(const char* data, int size, int lx, int ly, int lz)
{
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(data, size, opt, spirv) != 0)
        return nullptr;
    ncnn::Pipeline* p = new ncnn::Pipeline(vkdev);
    p->set_optimal_local_size_xyz(lx, ly, lz);
    if (p->create(spirv.data(), spirv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        delete p;
        return nullptr;
    }
    return p;
}

bool RestoreNet::init(ncnn::VulkanDevice* dev, const ncnn::Option& base, const std::filesystem::path& dir, double fs,
                      std::string& error)
{
    release();
    vkdev = dev;
    flow_scale = fs;
    opt = base;
    blob_alloc = vkdev->acquire_blob_allocator();
    staging_alloc = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_alloc;
    opt.workspace_vkallocator = blob_alloc;
    opt.staging_vkallocator = staging_alloc;
    elemsize = opt.use_fp16_storage ? 2u : 4u;
    if (!load_net(flow_net, opt, vkdev, dir, "r2_flow", error) || !load_net(body_net, opt, vkdev, dir, "r2_body", error))
        return false;
    warp = Warp_layer_creator(nullptr);
    warp->vkdev = vkdev;
    if (warp->create_pipeline(opt) != 0)
    {
        error = "couche de recalage indisponible";
        return false;
    }
    p_yuv_to_rgb = make_pipeline(restore_yuv_to_rgb_comp_data, sizeof(restore_yuv_to_rgb_comp_data), 8, 8, 1);
    p_luma_area = make_pipeline(restore_luma_area_comp_data, sizeof(restore_luma_area_comp_data), 8, 8, 1);
    p_resize = make_pipeline(restore_resize_comp_data, sizeof(restore_resize_comp_data), 8, 8, 1);
    p_assemble = make_pipeline(restore_assemble_comp_data, sizeof(restore_assemble_comp_data), 8, 8, 1);
    p_tile = make_pipeline(restore_tile_comp_data, sizeof(restore_tile_comp_data), 8, 8, 1);
    p_rgb_to_yuv = make_pipeline(restore_rgb_to_yuv_aa_comp_data, sizeof(restore_rgb_to_yuv_aa_comp_data), 8, 8, 1);
    if (!p_yuv_to_rgb || !p_luma_area || !p_resize || !p_assemble || !p_tile || !p_rgb_to_yuv)
    {
        error = "compilation des shaders de restauration impossible";
        return false;
    }
    return true;
}

bool RestoreNet::configure(const FrameFormat& f, const ColorParams& c, bool pq_transfer, int tile_size, std::string& error)
{
    fmt = f;
    color = c;
    pq = pq_transfer ? 1.f : 0.f;
    tile = std::max(64, tile_size / 4 * 4);
    cache.clear();
    W = fmt.width + 2 * EDGE;
    H = fmt.height + 2 * EDGE;
    // définition du flux (R2Net.flows : arrondi au pair, au plus proche) puis entrée du graphe (multiple de 32)
    ws = (int)std::nearbyint(W * flow_scale / 2) * 2;
    hs = (int)std::nearbyint(H * flow_scale / 2) * 2;
    wu = (ws + 31) / 32 * 32;
    hu = (hs + 31) / 32 * 32;
    if (ws < 32 || hs < 32)
    {
        error = "image trop petite pour la restauration";
        return false;
    }
    return true;
}

bool RestoreNet::resize(ncnn::VkCompute& cmd, const ncnn::VkMat& src, int inc, int outw, int outh, int outc, float mul0,
                        float mul1, ncnn::VkMat& dst)
{
    dst.create(outw, outh, outc, elemsize, 1, blob_alloc);
    if (dst.empty())
        return false;
    std::vector<ncnn::vk_constant_type> k(10);
    k[0].i = src.w;
    k[1].i = src.h;
    k[2].i = inc;
    k[3].i = (int)src.cstep;
    k[4].i = outw;
    k[5].i = outh;
    k[6].i = outc;
    k[7].i = (int)dst.cstep;
    k[8].f = mul0;
    k[9].f = mul1;
    cmd.record_pipeline(p_resize, {src, dst}, k, dispatch_of(outw, outh, outc));
    return true;
}

bool RestoreNet::add_frame(int64_t index, const void* frame, std::string& error)
{
    if (cache.count(index))
        return true;
    Cached& e = cache[index];
    ncnn::VkCompute cmd(vkdev);
    ncnn::VkMat raw;
    {
        const size_t words = fmt.padded_frame_bytes() / 4;
        ncnn::Mat host((int)words, (void*)frame, (size_t)4u, 1);
        cmd.record_clone(host, raw, opt);
    }
    e.rgb.create(W, H, 3, elemsize, 1, blob_alloc);
    ncnn::VkMat small;
    small.create(ws, hs, 1, elemsize, 1, blob_alloc);
    if (raw.empty() || e.rgb.empty() || small.empty())
    {
        cache.erase(index);
        error = "allocation mémoire GPU impossible (restauration)";
        return false;
    }
    const Scale s = code_scale(fmt, color.full_range);
    std::vector<ncnn::vk_constant_type> k(19);
    k[0].i = fmt.width;
    k[1].i = fmt.height;
    k[2].i = fmt.chroma_width();
    k[3].i = fmt.chroma_height();
    k[4].i = (int)fmt.luma_samples();
    k[5].i = (int)(fmt.luma_samples() + fmt.chroma_samples());
    k[6].i = fmt.bytes_per_sample;
    k[7].i = W;
    k[8].i = H;
    k[9].i = (int)e.rgb.cstep;
    k[10].i = fmt.sub_x == 1 ? 0 : (color.siting == ChromaSiting::Center ? 2 : 1);
    k[11].i = fmt.sub_y == 1 ? 0 : (color.siting == ChromaSiting::TopLeft ? 1 : 2);
    k[12].i = EDGE;
    k[13].f = s.y_off;
    k[14].f = 1.f / s.y_range;
    k[15].f = s.c_off;
    k[16].f = 1.f / s.c_range;
    k[17].f = color.kr;
    k[18].f = color.kb;
    cmd.record_pipeline(p_yuv_to_rgb, {raw, e.rgb}, k, dispatch_of(W, H, 1));
    std::vector<ncnn::vk_constant_type> a(5);
    a[0].i = W;
    a[1].i = H;
    a[2].i = (int)e.rgb.cstep;
    a[3].i = ws;
    a[4].i = hs;
    cmd.record_pipeline(p_luma_area, {e.rgb, small}, a, dispatch_of(ws, hs, 1));
    if (!resize(cmd, small, 1, wu, hu, 3, 1.f, 1.f, e.flowin))
    {
        cache.erase(index);
        error = "allocation mémoire GPU impossible (restauration)";
        return false;
    }
    if (cmd.submit_and_wait() != 0)
    {
        cache.erase(index);
        error = "conversion de l'image impossible (restauration)";
        return false;
    }
    return true;
}

void RestoreNet::drop_frame(int64_t index) { cache.erase(index); }

bool RestoreNet::process(int64_t t, const int64_t nb[4], const uint8_t* grain_map, std::vector<float>& yuv, const Crop* crop,
                         std::vector<float>* flows, std::string& error)
{
    auto fail = [&](const char* what) {
        error = std::string("restauration : ") + what;
        return false;
    };
    if (!cache.count(t))
        return fail("image absente du cache");
    for (int j = 0; j < 4; j++)
        if (!cache.count(nb[j]))
            return fail("voisine absente du cache");
    const Cached& ct = cache.at(t);
    ncnn::VkCompute cmd(vkdev);

    // flux et voisines recalées
    ncnn::VkMat warped[4], full[4];
    for (int j = 0; j < 4; j++)
    {
        const Cached& cn = cache.at(nb[j]);
        ncnn::VkMat f;
        {
            ncnn::Extractor ex = flow_net.create_extractor();
            ex.input("in0", ct.flowin);
            ex.input("in1", cn.flowin);
            if (ex.extract("out0", f, cmd) != 0)
                return fail("graphe de flux en échec");
        }
        ncnn::VkMat f_pack1;
        if (f.elempack != 1)
            vkdev->convert_packing(f, f_pack1, 1, cmd, opt);
        else
            f_pack1 = f;
        ncnn::VkMat fs;
        if (!resize(cmd, f_pack1, 2, ws, hs, 2, (float)ws / wu, (float)hs / hu, fs)
                || !resize(cmd, fs, 2, W, H, 2, (float)W / ws, (float)H / hs, full[j]))
            return fail("allocation mémoire GPU impossible");
        std::vector<ncnn::VkMat> bottoms = {cn.rgb, full[j]};
        std::vector<ncnn::VkMat> tops(1);
        if (warp->forward(bottoms, tops, cmd, opt) != 0)
            return fail("recalage en échec");
        if (tops[0].elempack != 1)
            vkdev->convert_packing(tops[0], warped[j], 1, cmd, opt);
        else
            warped[j] = tops[0];
    }

    // entrée du corps (17 canaux)
    ncnn::VkMat map_gpu, x;
    {
        const size_t n = (size_t)fmt.width * fmt.height;
        std::vector<uint32_t> words((n + 3) / 4 + 4, 0u);
        std::memcpy(words.data(), grain_map, n);
        ncnn::Mat host((int)words.size(), words.data(), (size_t)4u, 1);
        cmd.record_clone(host, map_gpu, opt);
    }
    x.create(W, H, 17, elemsize, 1, blob_alloc);
    if (map_gpu.empty() || x.empty())
        return fail("allocation mémoire GPU impossible");
    {
        std::vector<ncnn::vk_constant_type> k(8);
        k[0].i = W;
        k[1].i = H;
        k[2].i = (int)ct.rgb.cstep;
        k[3].i = fmt.width;
        k[4].i = fmt.height;
        k[5].i = EDGE;
        k[6].i = (int)x.cstep;
        k[7].f = pq;
        cmd.record_pipeline(p_assemble, {ct.rgb, warped[0], warped[1], warped[2], warped[3], map_gpu, x}, k, dispatch_of(W, H, 1));
    }

    // corps par tuiles, accumulation pondérée (fenêtre de Hann), comme r1r_infer.run
    ncnn::VkMat acc, wsum, dummy;
    acc.create(W, H, 3, 4u, 1, blob_alloc);
    wsum.create(W, H, 1, 4u, 1, blob_alloc);
    dummy.create(4, 1, 1, elemsize, 1, blob_alloc);
    if (acc.empty() || wsum.empty() || dummy.empty())
        return fail("allocation mémoire GPU impossible");
    auto tile_constants = [&](int mode, int c, int x0, int y0, int tw, int th, int tcstep, int wx0, int wy0, int tpw, int tph) {
        std::vector<ncnn::vk_constant_type> k(16);
        k[0].i = mode;
        k[1].i = W;
        k[2].i = H;
        k[3].i = (int)x.cstep;
        k[4].i = c;
        k[5].i = x0;
        k[6].i = y0;
        k[7].i = tw;
        k[8].i = th;
        k[9].i = tcstep;
        k[10].i = wx0;
        k[11].i = wy0;
        k[12].i = tile + 2 * PAD;
        k[13].i = (int)acc.cstep;
        k[14].i = tpw;
        k[15].i = tph;
        return k;
    };
    cmd.record_pipeline(p_tile, {x, dummy, acc, wsum}, tile_constants(2, 0, 0, 0, W, H, 0, 0, 0, W, H), dispatch_of(W, H, 1));
    for (int ty = 0; ty < H; ty += tile)
        for (int tx = 0; tx < W; tx += tile)
        {
            const int ya = std::max(0, ty - PAD), yb = std::min(H, ty + tile + PAD);
            const int xa = std::max(0, tx - PAD), xb = std::min(W, tx + tile + PAD);
            const int tw = xb - xa, th = yb - ya, tpw = (tw + 3) / 4 * 4, tph = (th + 3) / 4 * 4;
            ncnn::VkMat tin, tout, tout1;
            tin.create(tpw, tph, 17, elemsize, 1, blob_alloc);
            if (tin.empty())
                return fail("allocation mémoire GPU impossible");
            cmd.record_pipeline(p_tile, {x, tin, acc, wsum},
                                tile_constants(0, 17, xa, ya, tw, th, (int)tin.cstep, 0, 0, tpw, tph), dispatch_of(tpw, tph, 1));
            {
                ncnn::Extractor ex = body_net.create_extractor();
                ex.input("in0", tin);
                if (ex.extract("out0", tout, cmd) != 0)
                    return fail("corps du réseau en échec");
            }
            if (tout.elempack != 1)
                vkdev->convert_packing(tout, tout1, 1, cmd, opt);
            else
                tout1 = tout;
            cmd.record_pipeline(p_tile, {tout1, dummy, acc, wsum},
                                tile_constants(1, 3, xa, ya, tw, th, (int)tout1.cstep, xa - (tx - PAD), ya - (ty - PAD), tpw, tph),
                                dispatch_of(tw, th, 1));
        }

    // codes Y'CbCr flottants (sous-échantillonnage chroma antirepliement)
    const int cw = fmt.chroma_width(), ch = fmt.chroma_height();
    const int cmx = fmt.sub_x == 1 ? 0 : (color.siting == ChromaSiting::Center ? 2 : 1);
    const int cmy = fmt.sub_y == 1 ? 0 : (color.siting == ChromaSiting::TopLeft ? 1 : 2);
    const int cwx = cmx == 0 ? fmt.width : cw;
    ncnn::VkMat tmp, out;
    tmp.create(cwx * fmt.height * 2, (size_t)4u, 1, blob_alloc);
    out.create((int)(fmt.luma_samples() + 2 * fmt.chroma_samples()), (size_t)4u, 1, blob_alloc);
    if (tmp.empty() || out.empty())
        return fail("allocation mémoire GPU impossible");
    const Scale s = code_scale(fmt, color.full_range);
    for (int pass = 0; pass < 2; pass++)
    {
        std::vector<ncnn::vk_constant_type> k(16);
        k[0].i = pass;
        k[1].i = W;
        k[2].i = (int)acc.cstep;
        k[3].i = EDGE;
        k[4].i = fmt.width;
        k[5].i = fmt.height;
        k[6].i = cw;
        k[7].i = ch;
        k[8].i = cmx;
        k[9].i = cmy;
        k[10].f = s.y_off;
        k[11].f = s.y_range;
        k[12].f = s.c_off;
        k[13].f = s.c_range;
        k[14].f = color.kr;
        k[15].f = color.kb;
        cmd.record_pipeline(p_rgb_to_yuv, {acc, wsum, tmp, out}, k,
                            pass == 0 ? dispatch_of(std::max(fmt.width, cwx), fmt.height, 1) : dispatch_of(cw, ch, 1));
    }
    // copies brutes (record_clone : aucune conversion) ; flux fp16 convertis ensuite
    ncnn::Mat host_out;
    cmd.record_clone(out, host_out, opt);
    ncnn::Mat host_flows[4];
    const bool want_flows = crop && flows;
    if (want_flows)
        for (int j = 0; j < 4; j++)
            cmd.record_clone(full[j], host_flows[j], opt);
    if (cmd.submit_and_wait() != 0)
        return fail("exécution GPU en échec");
    yuv.assign((const float*)host_out.data, (const float*)host_out.data + host_out.w);
    if (want_flows)
    {
        flows->assign((size_t)4 * crop->w * crop->h * 2, 0.f);
        for (int j = 0; j < 4; j++)
        {
            ncnn::Mat m = host_flows[j];
            if (m.elemsize == 2)
            {
                ncnn::Mat f32;
                ncnn::cast_float16_to_float32(m, f32, opt);
                m = f32;
            }
            for (int c = 0; c < 2; c++)
            {
                const float* plane = m.channel(c);
                for (int y = 0; y < crop->h; y++)
                    for (int xx = 0; xx < crop->w; xx++)
                        (*flows)[(((size_t)j * crop->h + y) * crop->w + xx) * 2 + c] =
                            plane[(size_t)(EDGE + crop->y0 + y) * m.w + EDGE + crop->x0 + xx];
            }
        }
    }
    return true;
}
