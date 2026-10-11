#include "flow_pruning.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
struct Layer
{
    std::string kind, name, line;
    std::vector<std::string> inputs, outputs;
    std::map<std::string, std::string> params;
};

bool integer(const std::string& value, size_t& result)
{
    if (value.empty()) return false;
    result = 0;
    for (const char c : value)
    {
        if (c < '0' || c > '9' || result > (std::numeric_limits<size_t>::max() - (c - '0')) / 10)
            return false;
        result = result * 10 + (c - '0');
    }
    return true;
}

std::string get(const Layer& layer, const std::string& key, const char* fallback = "")
{
    const auto found = layer.params.find(key);
    return found == layer.params.end() ? fallback : found->second;
}

bool parse(const std::string& param, std::vector<Layer>& layers)
{
    std::istringstream source(param);
    std::string line, magic, extra, count_a, count_b;
    if (!std::getline(source, line)) return false;
    std::istringstream header(line);
    if (!(header >> magic) || magic != "7767517" || header >> extra) return false;
    if (!std::getline(source, line)) return false;
    std::istringstream counts(line);
    size_t expected_layers, expected_blobs;
    if (!(counts >> count_a >> count_b) || counts >> extra || !integer(count_a, expected_layers) ||
        !integer(count_b, expected_blobs) || expected_layers > param.size() || expected_blobs > param.size())
        return false;
    std::set<std::string> names, produced;
    while (std::getline(source, line))
    {
        std::istringstream tokens(line);
        Layer layer;
        layer.line = line;
        if (!(tokens >> layer.kind)) continue;
        std::string a, b, token;
        size_t ni, no;
        if (!(tokens >> layer.name >> a >> b) || !integer(a, ni) || !integer(b, no) ||
            ni > line.size() || no > line.size() || !names.insert(layer.name).second)
            return false;
        for (size_t i = 0; i < ni; i++)
        {
            if (!(tokens >> token) || !produced.count(token)) return false;
            layer.inputs.push_back(token);
        }
        for (size_t i = 0; i < no; i++)
        {
            if (!(tokens >> token) || !produced.insert(token).second) return false;
            layer.outputs.push_back(token);
        }
        while (tokens >> token)
        {
            const size_t equal = token.find('=');
            if (equal == std::string::npos || equal == 0 || equal + 1 == token.size() ||
                !layer.params.emplace(token.substr(0, equal), token.substr(equal + 1)).second)
                return false;
        }
        layers.push_back(std::move(layer));
    }
    return layers.size() == expected_layers && produced.size() == expected_blobs;
}

bool bounded_crop(const Layer& layer)
{
    if (layer.inputs.size() != 1 || layer.outputs.size() != 1 || layer.params.size() != 3 ||
        get(layer, "-23311") != "1,0") return false;
    const std::string start = get(layer, "-23309"), end = get(layer, "-23310");
    size_t a, b;
    return start.compare(0, 2, "1,") == 0 && end.compare(0, 2, "1,") == 0 &&
           integer(start.substr(2), a) && integer(end.substr(2), b) && a < b && b <= 5;
}

std::set<size_t> targets(const std::vector<Layer>& layers)
{
    std::map<std::string, std::vector<size_t>> consumers;
    for (size_t i = 0; i < layers.size(); i++)
        for (const auto& input : layers[i].inputs) consumers[input].push_back(i);
    std::set<size_t> result;
    for (size_t i = 0; i < layers.size(); i++)
    {
        const auto& layer = layers[i];
        if (layer.kind != "Deconvolution" || get(layer, "0") != "24" ||
            layer.inputs.size() != 1 || layer.outputs.size() != 1 || get(layer, "7", "1") != "1")
            continue;
        const auto& uses = consumers[layer.outputs[0]];
        if (uses.size() != 1) continue;
        const auto& shuffle = layers[uses[0]];
        if (shuffle.kind != "PixelShuffle" || shuffle.inputs.size() != 1 || shuffle.outputs.size() != 1 ||
            shuffle.params.size() != 1 || get(shuffle, "0") != "2") continue;
        std::vector<std::string> pending = shuffle.outputs;
        std::set<std::string> visited;
        bool valid = true;
        size_t crops = 0;
        while (!pending.empty() && valid)
        {
            const std::string blob = pending.back();
            pending.pop_back();
            if (!visited.insert(blob).second) continue;
            if (blob == "out0" || consumers[blob].empty()) { valid = false; break; }
            for (const size_t index : consumers[blob])
            {
                const auto& next = layers[index];
                if (next.kind == "Crop")
                {
                    valid = valid && bounded_crop(next);
                    crops++;
                }
                else if (next.kind == "Split" && next.inputs.size() == 1 && !next.outputs.empty() && next.params.empty())
                    pending.insert(pending.end(), next.outputs.begin(), next.outputs.end());
                else if (next.kind == "Interp" && next.inputs.size() == 1 && next.outputs.size() == 1 && get(next, "0") == "2")
                {
                    for (const auto& entry : next.params)
                        if (entry.first != "0" && entry.first != "1" && entry.first != "2") valid = false;
                    pending.insert(pending.end(), next.outputs.begin(), next.outputs.end());
                }
                else valid = false;
            }
        }
        if (valid && crops >= 2) result.insert(i);
    }
    return result;
}

uint32_t read_tag(const std::vector<unsigned char>& data, size_t offset)
{
    return uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8) |
           (uint32_t(data[offset + 2]) << 16) | (uint32_t(data[offset + 3]) << 24);
}

// Les limites s'appliquent avant toute multiplication ou lecture du binaire.
bool product(size_t a, size_t b, size_t limit, size_t& result)
{
    if (b && a > limit / b) return false;
    result = a * b;
    return result <= limit;
}

bool rewrite(std::vector<Layer>& layers, const std::set<size_t>& selected,
             const std::vector<unsigned char>& data, std::vector<unsigned char>& output)
{
    const std::set<std::string> weightless{"Input", "Split", "Concat", "Interp", "PixelShuffle", "Crop",
        "ReLU", "BinaryOp", "rife.Warp", "Sigmoid", "Eltwise"};
    size_t position = 0;
    for (size_t i = 0; i < layers.size(); i++)
    {
        auto& layer = layers[i];
        if (layer.kind == "Convolution" || layer.kind == "Deconvolution")
        {
            size_t count = 0, channels = 0, payload = 0, bias = 0;
            if (get(layer, "5", "0") != "1" || get(layer, "7", "1") != "1" ||
                get(layer, "28", "0") != "0" ||
                (layer.kind == "Convolution" && get(layer, "19", "0") != "0") || get(layer, "8", "0") != "0" ||
                !integer(get(layer, "6"), count) || !count || !integer(get(layer, "0"), channels) || !channels ||
                data.size() - position < 4) return false;
            const uint32_t tag = read_tag(data, position);
            if (tag != 0 && tag != 0x01306b47) return false;
            const size_t width = tag ? 2 : 4;
            if (!product(count, width, data.size() - position - 4, payload) ||
                payload > std::numeric_limits<size_t>::max() - 3) return false;
            const size_t padded = (payload + 3) / 4 * 4;
            if (padded > data.size() - position - 4 ||
                !product(channels, 4, data.size() - position - 4 - padded, bias)) return false;
            const size_t end = position + 4 + padded + bias;
            if (selected.count(i))
            {
                if (channels != 24 || count % 24) return false;
                const size_t kept = count / 24 * 20, bytes = kept * width;
                output.insert(output.end(), data.begin() + position, data.begin() + position + 4 + bytes);
                output.insert(output.end(), (4 - bytes % 4) % 4, 0);
                output.insert(output.end(), data.begin() + position + 4 + padded, data.begin() + position + 4 + padded + 80);
                layer.params["0"] = "20";
                layer.params["6"] = std::to_string(kept);
            }
            else output.insert(output.end(), data.begin() + position, data.begin() + end);
            position = end;
        }
        else if (layer.kind == "MemoryData")
        {
            size_t width = 0, height = 0, channels = 0, count = 0;
            if (get(layer, "11", "0") != "0" || get(layer, "21", "1") != "1" ||
                !integer(get(layer, "0", "0"), width) || !integer(get(layer, "1", "0"), height) ||
                !integer(get(layer, "2", "1"), channels) || !width ||
                !product(width, std::max(size_t(1), height), data.size() - position, count) ||
                !product(count, std::max(size_t(1), channels), data.size() - position, count) ||
                !product(count, 4, data.size() - position, count)) return false;
            output.insert(output.end(), data.begin() + position, data.begin() + position + count);
            position += count;
        }
        else if (!weightless.count(layer.kind)) return false;
    }
    return position == data.size();
}
} // namespace

int prune_unused_flow(const std::string& param, const std::vector<unsigned char>& weights,
                      std::string& output_param, std::vector<unsigned char>& output_weights)
{
    output_param = param;
    output_weights.clear();
    std::vector<Layer> layers;
    if (!parse(param, layers)) return 0;
    const auto selected = targets(layers);
    if (selected.empty()) return 0;
    std::vector<unsigned char> rewritten;
    if (!rewrite(layers, selected, weights, rewritten)) return 0;
    std::ostringstream output;
    output << "7767517\n" << layers.size() << ' ';
    size_t blobs = 0;
    for (const auto& layer : layers) blobs += layer.outputs.size();
    output << blobs << '\n';
    for (size_t i = 0; i < layers.size(); i++)
    {
        const auto& layer = layers[i];
        if (!selected.count(i)) { output << layer.line << '\n'; continue; }
        output << layer.kind << ' ' << layer.name << ' ' << layer.inputs.size() << ' ' << layer.outputs.size();
        for (const auto& input : layer.inputs) output << ' ' << input;
        for (const auto& blob : layer.outputs) output << ' ' << blob;
        for (const auto& entry : layer.params) output << ' ' << entry.first << '=' << entry.second;
        output << '\n';
    }
    output_param = output.str();
    output_weights.swap(rewritten);
    return static_cast<int>(selected.size());
}
