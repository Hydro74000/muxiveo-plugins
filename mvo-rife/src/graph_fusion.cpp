#include "graph_fusion.h"

#include <cmath>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace {
struct Layer
{
    std::string kind, name;
    std::vector<std::string> in, out;
    std::map<std::string, std::string> params;
};

bool value_is(const Layer& layer, const char* key, const char* expected)
{
    const auto found = layer.params.find(key);
    return found == layer.params.end() || found->second == expected;
}

bool valid_slope(const Layer& layer)
{
    if (layer.params.empty()) return true;
    if (layer.params.size() != 1 || !layer.params.count("0")) return false;
    std::istringstream stream(layer.params.at("0"));
    stream.imbue(std::locale::classic());
    float slope;
    if (!(stream >> slope) || !std::isfinite(slope)) return false;
    stream >> std::ws;
    return stream.eof();
}
}

int fuse_residual_activations(const std::string& param, std::string& result)
{
    result = param;
    std::istringstream stream(param);
    int magic, layer_count, blob_count;
    if (!(stream >> magic >> layer_count >> blob_count) || magic != 7767517 || layer_count <= 0) return 0;
    std::string line, token;
    std::getline(stream, line);
    std::vector<Layer> layers;
    std::map<std::string, int> consumers;
    std::set<std::string> names, blobs;
    while (std::getline(stream, line))
    {
        std::istringstream row(line);
        Layer layer;
        int ni, no;
        if (!(row >> layer.kind)) continue;
        if (!(row >> layer.name >> ni >> no) || ni < 0 || no <= 0 || !names.insert(layer.name).second) return 0;
        for (int i = 0; i < ni; i++)
        {
            if (!(row >> token) || !blobs.count(token)) return 0;
            layer.in.push_back(token);
            consumers[token]++;
        }
        for (int i = 0; i < no; i++)
        {
            if (!(row >> token) || !blobs.insert(token).second) return 0;
            layer.out.push_back(token);
        }
        while (row >> token)
        {
            const auto eq = token.find('=');
            if (eq == std::string::npos || !layer.params.emplace(token.substr(0, eq), token.substr(eq + 1)).second) return 0;
        }
        layers.push_back(std::move(layer));
    }
    if (layers.size() != (size_t)layer_count || blobs.size() != (size_t)blob_count) return 0;
    int fused = 0;
    std::vector<Layer> output;
    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer& a = layers[i];
        if (i + 1 < layers.size())
        {
            const Layer& b = layers[i + 1];
            bool known = true;
            for (const auto& item : a.params)
                known = known && (item.first == "0" || item.first == "1" || item.first == "2");
            if (a.kind == "BinaryOp" && known && value_is(a, "0", "0") && value_is(a, "1", "0")
                && a.in.size() == 2 && a.out.size() == 1 && a.out[0] != "out0"
                && consumers[a.out[0]] == 1 && b.kind == "ReLU" && b.in == a.out
                && b.out.size() == 1 && valid_slope(b))
            {
                // Le nom de l'addition reste unique ; aucune couche pondérée ne change.
                a.kind = "rife.AddRelu";
                a.out = b.out;
                a.params = b.params;
                i++;
                fused++;
            }
        }
        output.push_back(std::move(a));
    }
    if (!fused) return 0;
    std::ostringstream rewritten;
    rewritten.imbue(std::locale::classic());
    rewritten << "7767517\n" << layer_count - fused << " " << blob_count - fused << "\n";
    for (const auto& layer : output)
    {
        rewritten << layer.kind << " " << layer.name << " " << layer.in.size() << " " << layer.out.size();
        for (const auto& value : layer.in) rewritten << " " << value;
        for (const auto& value : layer.out) rewritten << " " << value;
        for (const auto& item : layer.params) rewritten << " " << item.first << "=" << item.second;
        rewritten << "\n";
    }
    result = rewritten.str();
    return fused;
}
