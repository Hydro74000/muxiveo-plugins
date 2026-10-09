// Lecture des poids du sélecteur (format décrit dans selector_weights.h).
#include "selector_weights.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "numparse.h"

namespace {

// Nombre d'indices attendu pour un jeu de candidats (voir RifeEngine::set_dense_flow).
int expected_features(uint32_t candidates)
{
    const int nd = 1 + ((candidates & 2u) ? 1 : 0) + ((candidates & 8u) ? 1 : 0) + ((candidates & 16u) ? 1 : 0);
    return 2 * nd + 3 + ((candidates & 16u) ? 2 : 0) + 2;
}

int count_bits(uint32_t v)
{
    int n = 0;
    for (; v; v &= v - 1)
        n++;
    return n;
}

bool read_floats(std::istringstream& in, float* dst, int count)
{
    std::string token;
    for (int i = 0; i < count; i++)
    {
        double v = 0.0;
        if (!(in >> token) || !parse_double_strict(token, -1e9, 1e9, v))
            return false;
        dst[i] = std::strtof(token.c_str(), nullptr);   // arrondi direct en float (comme un littéral compilé)
    }
    return !(in >> token);
}

}   // namespace

bool load_selector_weights(const std::filesystem::path& path, std::vector<SelectorModel>& models, std::string& error)
{
    models.clear();
    std::ifstream file(path);
    if (!file)
    {
        error = "fichier introuvable";
        return false;
    }
    std::string line;
    int line_no = 0;
    bool header = false;
    SelectorModel* current = nullptr;
    int weights_read = 0;
    bool mu = false, sd = false, bias = false;
    auto fail = [&](const std::string& what) {
        error = "ligne " + std::to_string(line_no) + " : " + what;
        models.clear();
        return false;
    };
    auto complete = [&]() { return !current || (mu && sd && bias && weights_read == current->ncand); };
    while (std::getline(file, line))
    {
        line_no++;
        if (const size_t hash = line.find('#'); hash != std::string::npos)
            line.erase(hash);
        std::istringstream in(line);
        std::string key;
        if (!(in >> key))
            continue;
        if (!header)
        {
            int64_t version = 0;
            std::string v;
            if (key != "muxiveo-rife-selector" || !(in >> v) || !parse_int64_strict(v, 0, 1000, version))
                return fail("en-tête « muxiveo-rife-selector <version> » attendu");
            if (version != SELECTOR_WEIGHTS_FORMAT)
                return fail("format " + v + " non pris en charge (" + std::to_string(SELECTOR_WEIGHTS_FORMAT) + " attendu)");
            header = true;
            continue;
        }
        if (key == "model")
        {
            if (!complete())
                return fail("modèle précédent incomplet");
            SelectorModel m{};
            std::string tok[6];
            int64_t val[5];
            for (auto& t : tok)
                if (!(in >> t))
                    return fail("model <famille> <candidats> <bmin> <bmax> <nfeat> <ncand> attendu");
            const int64_t bounds[5][2] = {{1, 31}, {0, 4096}, {0, 4096}, {1, 16}, {1, 5}};
            for (int i = 0; i < 5; i++)
                if (!parse_int64_strict(tok[i + 1], bounds[i][0], bounds[i][1], val[i]))
                    return fail("valeur invalide : " + tok[i + 1]);
            m.family = tok[0];
            m.candidates = (uint32_t)val[0];
            m.bmin = (int)val[1];
            m.bmax = (int)val[2];
            m.nfeat = (int)val[3];
            m.ncand = (int)val[4];
            if (!(m.candidates & 1u) || !(m.candidates & 4u))
                return fail("candidats R1 et MC obligatoires");
            if (m.ncand != count_bits(m.candidates))
                return fail("ncand différent du nombre de candidats");
            if (m.nfeat != expected_features(m.candidates))
                return fail("nfeat incohérent avec les candidats");
            if (m.bmin > m.bmax || (m.bmin == 0) != (m.bmax == 0))
                return fail("bande de blocs invalide");
            std::fill(std::begin(m.sd), std::end(m.sd), 1.f);
            models.push_back(m);
            current = &models.back();
            weights_read = 0;
            mu = sd = bias = false;
            continue;
        }
        if (!current)
            return fail("« model » attendu avant « " + key + " »");
        bool ok = true;
        if (key == "mu" && !mu)
            ok = mu = read_floats(in, current->mu, current->nfeat);
        else if (key == "sd" && !sd)
            ok = sd = read_floats(in, current->sd, current->nfeat);
        else if (key == "weight" && weights_read < current->ncand)
        {
            ok = read_floats(in, current->weight + weights_read * current->nfeat, current->nfeat);
            weights_read += ok ? 1 : 0;
        }
        else if (key == "bias" && !bias)
            ok = bias = read_floats(in, current->bias, current->ncand);
        else
            return fail("entrée inattendue : " + key);
        if (!ok)
            return fail(key + " : nombre de réels incorrect");
    }
    if (!header)
        return fail("fichier vide");
    if (!complete())
        return fail("dernier modèle incomplet");
    if (models.empty())
        return fail("aucun modèle");
    for (const SelectorModel& m : models)
        for (int i = 0; i < m.nfeat; i++)
            if (m.sd[i] <= 0.f)
            {
                error = "sd nul ou négatif (famille " + m.family + ")";
                models.clear();
                return false;
            }
    return true;
}

std::vector<std::string> selector_families(const std::vector<SelectorModel>& models)
{
    std::vector<std::string> families;
    for (const SelectorModel& m : models)
        if (std::find(families.begin(), families.end(), m.family) == families.end())
            families.push_back(m.family);
    return families;
}
