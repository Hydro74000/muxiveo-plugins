// Muxiveo — mode UHD de RIFE v4 (voir uhd.h).
//
// Un bloc IFBlock d'échelle s (Practical-RIFE, IFNet v4) :
//   x    = interp(x, 1/s)                  Interp (entrée : Concat images/features)
//   flow = interp(flow, 1/s) / s           Interp + BinaryOp div (absent si s = 1)
//   tmp  = interp(lastconv(...), s)        Interp (entrée : PixelShuffle/Deconvolution)
//   flow = tmp[:, 0:4] * s                 Crop 0-4 + BinaryOp mul, ou coefficient
//                                          d'un Eltwise somme (absent si s = 1)
// Le mode UHD remplace s par 2s dans chaque bloc ; les opérations absentes du
// dernier bloc (s = 1) sont insérées.

#include "uhd.h"

#include <algorithm>
#include <cstddef>
#include <locale>
#include <sstream>
#include <vector>

namespace {

struct Layer
{
    std::string type;
    std::string name;
    std::vector<std::string> in;
    std::vector<std::string> out;
    std::vector<std::string> params;
};

// Insertion différée d'une couche après `after` ; `blob` est remplacé par la sortie de la couche.
struct Insertion
{
    size_t after;
    Layer layer;
    std::string blob;
};

double to_double(const std::string& s, double fallback)
{
    std::istringstream is(s);
    is.imbue(std::locale::classic());
    double v = fallback;
    if (!(is >> v))
        return fallback;
    return v;
}

std::string to_string_e(double v)
{
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::scientific << v;
    return os.str();
}

bool param_find(const Layer& l, const std::string& key, std::string& value)
{
    for (const std::string& p : l.params)
    {
        const size_t eq = p.find('=');
        if (eq != std::string::npos && p.compare(0, eq, key) == 0 && eq == key.size())
        {
            value = p.substr(eq + 1);
            return true;
        }
    }
    return false;
}

std::string param_get(const Layer& l, const std::string& key, const std::string& fallback)
{
    std::string v;
    return param_find(l, key, v) ? v : fallback;
}

void param_set(Layer& l, const std::string& key, const std::string& value)
{
    const std::string entry = key + "=" + value;
    for (std::string& p : l.params)
    {
        const size_t eq = p.find('=');
        if (eq != std::string::npos && p.compare(0, eq, key) == 0 && eq == key.size())
        {
            p = entry;
            return;
        }
    }
    l.params.push_back(entry);
}

// BinaryOp à opérande scalaire (0=op 1=1 2=valeur).
bool is_scalar_binary(const Layer& l, const char* op)
{
    return l.type == "BinaryOp" && param_get(l, "0", "0") == op && param_get(l, "1", "0") == "1";
}

void scale_scalar(Layer& l, double k)
{
    param_set(l, "2", to_string_e(to_double(param_get(l, "2", "0"), 0.0) * k));
}

Layer scalar_binary(const std::string& name, const std::string& input, const char* op, double value)
{
    Layer l;
    l.type = "BinaryOp";
    l.name = name;
    l.in.push_back(input);
    l.out.push_back(input + "_uhd");
    l.params.push_back(std::string("0=") + op);
    l.params.push_back("1=1");
    l.params.push_back("2=" + to_string_e(value));
    return l;
}

class Graph
{
public:
    std::vector<Layer> layers;

    int producer(const std::string& blob) const
    {
        for (size_t i = 0; i < layers.size(); i++)
            if (std::find(layers[i].out.begin(), layers[i].out.end(), blob) != layers[i].out.end())
                return (int)i;
        return -1;
    }

    // Couche productrice en remontant les Split.
    int source(const std::string& blob) const
    {
        int p = producer(blob);
        while (p >= 0 && layers[p].type == "Split" && !layers[p].in.empty())
            p = producer(layers[p].in[0]);
        return p;
    }

    std::vector<int> consumers(const std::string& blob) const
    {
        std::vector<int> res;
        for (size_t i = 0; i < layers.size(); i++)
            if (std::find(layers[i].in.begin(), layers[i].in.end(), blob) != layers[i].in.end())
                res.push_back((int)i);
        return res;
    }

    // Consommateurs finaux en traversant les Split.
    std::vector<int> consumers_through_split(const std::string& blob) const
    {
        std::vector<int> res;
        for (int c : consumers(blob))
        {
            if (layers[c].type == "Split")
            {
                for (const std::string& o : layers[c].out)
                {
                    const std::vector<int> sub = consumers_through_split(o);
                    res.insert(res.end(), sub.begin(), sub.end());
                }
            }
            else
                res.push_back(c);
        }
        return res;
    }
};

// Multiplie par k le coefficient appliqué à `blob` dans un Eltwise somme (-23301=n,c0,c1…).
bool scale_eltwise_coeff(Layer& l, const std::string& blob, double k)
{
    if (l.type != "Eltwise" || param_get(l, "0", "0") != "1")
        return false;
    std::string coeffs;
    if (!param_find(l, "-23301", coeffs))
        return false;
    std::vector<std::string> items;
    std::stringstream ss(coeffs);
    std::string item;
    while (std::getline(ss, item, ','))
        items.push_back(item);
    const auto it = std::find(l.in.begin(), l.in.end(), blob);
    const size_t pos = (size_t)(it - l.in.begin());
    if (it == l.in.end() || items.size() != l.in.size() + 1 || pos + 1 >= items.size())
        return false;
    items[pos + 1] = to_string_e(to_double(items[pos + 1], 1.0) * k);
    std::string joined = items[0];
    for (size_t i = 1; i < items.size(); i++)
        joined += "," + items[i];
    param_set(l, "-23301", joined);
    return true;
}

} // namespace

bool make_uhd_param(const std::string& param, std::string& result, std::string& error)
{
    std::istringstream is(param);
    std::string magic;
    std::string counts;
    if (!std::getline(is, magic) || !std::getline(is, counts))
    {
        error = "flownet.param tronqué";
        return false;
    }
    int layer_count = 0;
    int blob_count = 0;
    {
        std::istringstream cs(counts);
        if (!(cs >> layer_count >> blob_count))
        {
            error = "en-tête flownet.param invalide";
            return false;
        }
    }

    Graph g;
    std::string line;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        Layer l;
        int ni = 0;
        int no = 0;
        if (!(ls >> l.type >> l.name >> ni >> no))
            continue;
        l.in.resize(ni);
        l.out.resize(no);
        for (std::string& b : l.in)
            ls >> b;
        for (std::string& b : l.out)
            ls >> b;
        std::string p;
        while (ls >> p)
            l.params.push_back(p);
        g.layers.push_back(l);
    }

    std::vector<Insertion> inserts;
    int interps = 0;
    for (size_t i = 0; i < g.layers.size(); i++)
    {
        if (g.layers[i].type != "Interp" || g.layers[i].in.size() != 1 || g.layers[i].out.size() != 1)
            continue;
        const int src = g.source(g.layers[i].in[0]);
        const std::string src_type = src >= 0 ? g.layers[src].type : std::string();
        const bool up = src_type == "PixelShuffle" || src_type == "Deconvolution";
        const double f = to_double(param_get(g.layers[i], "1", "1"), 1.0);
        const std::string nf = to_string_e(up ? f * 2.0 : f / 2.0);
        param_set(g.layers[i], "1", nf);
        param_set(g.layers[i], "2", nf);
        interps++;

        const std::string out = g.layers[i].out[0];
        if (!up)
        {
            if (src_type == "Concat")
                continue; // images + features : redimensionnement seul
            // flux réduit : divisé par l'échelle
            const std::vector<int> cons = g.consumers(out);
            if (cons.size() == 1 && is_scalar_binary(g.layers[cons[0]], "3"))
                scale_scalar(g.layers[cons[0]], 2.0);
            else
                inserts.push_back({i, scalar_binary(g.layers[i].name + "_uhd_div", out, "3", 2.0), out});
            continue;
        }

        // sortie du bloc : flux (canaux 0-3) multiplié par l'échelle
        int crop = -1;
        int crops = 0;
        for (int c : g.consumers_through_split(out))
        {
            const Layer& l = g.layers[c];
            if (l.type == "Crop" && param_get(l, "-23309", "") == "1,0" && param_get(l, "-23310", "") == "1,4")
            {
                crop = c;
                crops++;
            }
        }
        if (crops != 1 || g.layers[crop].out.size() != 1)
        {
            error = "structure RIFE v4 non reconnue (flux du bloc " + g.layers[i].name + ")";
            return false;
        }
        const std::string flow = g.layers[crop].out[0];
        const std::vector<int> cons = g.consumers(flow);
        if (cons.size() == 1 && is_scalar_binary(g.layers[cons[0]], "2"))
            scale_scalar(g.layers[cons[0]], 2.0);
        else if (!(cons.size() == 1 && scale_eltwise_coeff(g.layers[cons[0]], flow, 2.0)))
            inserts.push_back({(size_t)crop, scalar_binary(g.layers[crop].name + "_uhd_mul", flow, "2", 2.0), flow});
    }
    if (interps == 0)
    {
        error = "modèle sans couche Interp : mode UHD non applicable";
        return false;
    }

    // rebranche les consommateurs puis insère (de la fin vers le début)
    for (const Insertion& ins : inserts)
        for (Layer& l : g.layers)
            std::replace(l.in.begin(), l.in.end(), ins.blob, ins.layer.out[0]);
    std::stable_sort(inserts.begin(), inserts.end(), [](const Insertion& a, const Insertion& b) { return a.after > b.after; });
    for (const Insertion& ins : inserts)
        g.layers.insert(g.layers.begin() + (ptrdiff_t)ins.after + 1, ins.layer);

    std::ostringstream os;
    os << magic << "\n" << layer_count + (int)inserts.size() << " " << blob_count + (int)inserts.size() << "\n";
    for (const Layer& l : g.layers)
    {
        os << l.type << " " << l.name << " " << l.in.size() << " " << l.out.size();
        for (const std::string& b : l.in)
            os << " " << b;
        for (const std::string& b : l.out)
            os << " " << b;
        for (const std::string& p : l.params)
            os << " " << p;
        os << "\n";
    }
    result = os.str();
    return true;
}
