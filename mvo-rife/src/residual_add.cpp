#include "residual_add.h"
#include "command.h"
#include "gpu.h"
#include "pipeline.h"
#include "residual_add.comp.hex.h"

namespace {
class AddRelu : public ncnn::Layer
{
public:
    AddRelu() { support_vulkan = true; support_vulkan_packing = true; }
    int load_param(const ncnn::ParamDict& pd) override { slope = pd.get(0, 0.f); return 0; }

    int create_pipeline(const ncnn::Option& opt) override
    {
        // Les formes non compatibles utilisent exactement les deux couches natives.
        // Le repli CPU conserve notamment le broadcasting de BinaryOp.
        ncnn::ParamDict add_params, relu_params;
        add_params.set(0, 0);
        relu_params.set(0, slope);
        for (int device = 0; device < (vkdev ? 2 : 1); device++)
        {
            add[device] = device ? ncnn::create_layer_vulkan("BinaryOp") : ncnn::create_layer_naive("BinaryOp");
            relu[device] = device ? ncnn::create_layer_vulkan("ReLU") : ncnn::create_layer_naive("ReLU");
            if (!add[device] || !relu[device]) return -1;
            if (device) { add[device]->vkdev = vkdev; relu[device]->vkdev = vkdev; }
            if (add[device]->load_param(add_params) || relu[device]->load_param(relu_params)
                || add[device]->create_pipeline(opt) || relu[device]->create_pipeline(opt)) return -1;
        }
        if (!vkdev) return 0;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(residual_add_comp_data, sizeof(residual_add_comp_data), opt, spirv)) return -1;
        pipeline = new ncnn::Pipeline(vkdev);
        pipeline->set_optimal_local_size_xyz(64, 1, 1);
        return pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), {});
    }

    int destroy_pipeline(const ncnn::Option& opt) override
    {
        delete pipeline;
        pipeline = nullptr;
        for (int device = 0; device < 2; device++)
        {
            if (add[device]) { add[device]->destroy_pipeline(opt); delete add[device]; add[device] = nullptr; }
            if (relu[device]) { relu[device]->destroy_pipeline(opt); delete relu[device]; relu[device] = nullptr; }
        }
        return 0;
    }

    int forward(const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out,
                const ncnn::Option& opt) const override
    {
        const int result = add[0]->forward(in, out, opt);
        return result ? result : relu[0]->forward_inplace(out[0], opt);
    }

    int forward(const std::vector<ncnn::VkMat>& in, std::vector<ncnn::VkMat>& out,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        if (in.size() != 2) return -1;
        const ncnn::VkMat& a = in[0];
        const ncnn::VkMat& b = in[1];
        const size_t count = a.total() * a.elempack;
        if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.d != b.d || a.c != b.c
            || a.elempack != b.elempack || a.elemsize != b.elemsize || a.cstep != b.cstep
            || count % 4 != 0 || (a.elempack != 1 && a.elempack != 4))
        {
            const int result = add[1]->forward(in, out, cmd, opt);
            return result ? result : relu[1]->forward_inplace(out[0], cmd, opt);
        }
        out[0].create_like(a, opt.blob_vkallocator);
        if (out[0].empty()) return -100;
        if (out[0].cstep != a.cstep) return -1;
        std::vector<ncnn::vk_constant_type> constants(2);
        constants[0].u32 = count / 4;
        constants[1].f = slope;
        ncnn::VkMat dispatch;
        dispatch.w = constants[0].u32;
        dispatch.h = dispatch.c = 1;
        cmd.record_pipeline(pipeline, {a, b, out[0]}, constants, dispatch);
        return 0;
    }

private:
    float slope = 0.f;
    ncnn::Pipeline* pipeline = nullptr;
    ncnn::Layer* add[2] = {nullptr, nullptr};
    ncnn::Layer* relu[2] = {nullptr, nullptr};
};
}

DEFINE_LAYER_CREATOR(AddRelu)
