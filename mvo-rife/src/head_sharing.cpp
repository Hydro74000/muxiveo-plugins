// Un graphe inconnu, des paramètres différents ou une lecture inattendue interdisent le partage.
#include "head_sharing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
struct Layer
{
    std::string kind;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::string> params;
};

int value(const Layer& layer, const char* key, int fallback)
{
    const auto it = layer.params.find(key);
    if (it == layer.params.end()) return fallback;
    size_t consumed = 0;
    const int result = std::stoi(it->second, &consumed);
    if (consumed != it->second.size()) throw std::invalid_argument("Paramètre entier requis");
    return result;
}

bool branch(const std::vector<Layer>& layers, const std::map<std::string, size_t>& producers,
            std::string blob, const char* source, std::vector<size_t>& path)
{
    std::set<size_t> visited;
    int conv = 0, deconv = 0;
    while (blob != source)
    {
        auto found = producers.find(blob);
        if (found == producers.end() || !visited.insert(found->second).second) return false;
        const Layer& layer = layers[found->second];
        if (layer.inputs.size() != 1) return false;
        if (layer.kind == "Split")
        {
            if (!layer.params.empty()) return false;
        }
        else
        {
            if (layer.outputs.size() != 1) return false;
            if (layer.kind == "Convolution") conv++;
            else if (layer.kind == "Deconvolution") deconv++;
            else if (layer.kind != "Crop") return false;
            path.push_back(found->second);
        }
        blob = layer.inputs[0];
    }
    std::reverse(path.begin(), path.end());
    return conv == 3 && deconv == 1;
}
}

HeadSharingProof::HeadSharingProof(const std::string& param)
{
    try
    {
        std::istringstream stream(param);
        int magic, layer_count, blob_count;
        if (!(stream >> magic >> layer_count >> blob_count) || magic != 7767517 || layer_count <= 0) return;
        std::string line;
        std::getline(stream, line);
        std::vector<Layer> layers;
        std::map<std::string, size_t> producers;
        while (std::getline(stream, line))
        {
            std::istringstream row(line);
            Layer layer;
            std::string name, token;
            int inputs, outputs;
            if (!(row >> layer.kind)) continue;
            if (!(row >> name >> inputs >> outputs) || inputs < 0 || outputs <= 0) return;
            for (int i = 0; i < inputs; i++)
            {
                if (!(row >> token)) return;
                layer.inputs.push_back(token);
            }
            for (int i = 0; i < outputs; i++)
            {
                if (!(row >> token) || !producers.emplace(token, layers.size()).second) return;
                layer.outputs.push_back(token);
            }
            while (row >> token)
            {
                const size_t eq = token.find('=');
                if (eq == std::string::npos || !layer.params.emplace(token.substr(0, eq), token.substr(eq + 1)).second) return;
            }
            layers.push_back(std::move(layer));
        }
        if (layers.size() != (size_t)layer_count || producers.size() != (size_t)blob_count) return;
        std::vector<size_t> paths[2];
        if (!branch(layers, producers, "154", "in0", paths[0]) ||
            !branch(layers, producers, "166", "in1", paths[1]) || paths[0].size() != paths[1].size()) return;
        std::map<size_t, int> sides;
        for (size_t i = 0; i < paths[0].size(); i++)
        {
            const Layer& a = layers[paths[0][i]];
            const Layer& b = layers[paths[1][i]];
            if (a.kind != b.kind || a.params != b.params) return;
            sides[paths[0][i]] = 0;
            sides[paths[1][i]] = 1;
        }
        const std::set<std::string> no_weights = {"Input", "Split", "Crop", "Interp", "Concat", "PixelShuffle",
                                                "BinaryOp", "Eltwise", "Sigmoid", "ReLU", "Clip", "rife.Warp", "rife.AddRelu"};
        for (size_t i = 0; i < layers.size(); i++)
        {
            const Layer& layer = layers[i];
            const auto side_it = sides.find(i);
            const int side = side_it == sides.end() ? -1 : side_it->second;
            if (layer.kind == "Convolution" || layer.kind == "Deconvolution")
            {
                if (value(layer, layer.kind == "Convolution" ? "19" : "28", 0) != 0 ||
                    (layer.kind == "Convolution" && value(layer, "8", 0) != 0)) return;
                const int count = value(layer, "6", 0), channels = value(layer, "0", 0), bias = value(layer, "5", 0);
                if (count <= 0 || channels <= 0 || (bias != 0 && bias != 1)) return;
                loads.push_back({count, 0, side});
                if (bias) loads.push_back({channels, 1, side});
            }
            else if (layer.kind == "MemoryData")
            {
                long long count = value(layer, "0", 0);
                for (const char* key : {"1", "11", "2"})
                {
                    const int extent = value(layer, key, 0);
                    if (extent < 0 || count > std::numeric_limits<int>::max() / std::max(1, extent)) return;
                    count *= std::max(1, extent);
                }
                if (count < 0) return;
                if (count) loads.push_back({(int)count, value(layer, "21", 1), -1});
            }
            else if (!no_weights.count(layer.kind)) return;
        }
        valid = true;
    }
    catch (const std::exception&)
    {
        // La lecture native reste inchangée ; seule la mutualisation est refusée.
        valid = false;
    }
}

bool HeadSharingProof::next_load(int& count, int& type) const
{
    if (!valid || cursor >= loads.size()) return false;
    count = loads[cursor].count;
    type = loads[cursor].type;
    return true;
}

void HeadSharingProof::observe(int count, int type, const void* data, size_t bytes)
{
    if (!valid) return;
    if (cursor >= loads.size()) { valid = false; return; }
    const Load& expected = loads[cursor++];
    if (expected.count != count || expected.type != type || !data) { valid = false; return; }
    if (expected.side < 0) return;
    if (bytes != (size_t)count * sizeof(float)) { valid = false; return; }
    const auto* begin = static_cast<const unsigned char*>(data);
    for (int i = 0; i < count; i++)
    {
        float scalar;
        std::memcpy(&scalar, begin + i * sizeof(float), sizeof(float));
        if (!std::isfinite(scalar)) { valid = false; return; }
    }
    heads[expected.side].emplace_back(begin, begin + bytes);
}

bool HeadSharingProof::equivalent() const
{
    return valid && cursor == loads.size() && !heads[0].empty() && heads[0] == heads[1];
}
