// Poids du sélecteur appris du moteur hybride, lus au lancement dans <exe>/rife-models/selector.txt
// (ou --selector-weights) : améliorer le sélecteur revient à remplacer ce fichier.
//
// Sélecteur linéaire par blocs : indices = désaccords |Ck - R1| (luma 10 bits limitée / 1023), texture, écart
// temporel, luminance, désaccords dilatés 3x3, [incohérence et amplitude NVOF / 64], bord (bloc à moins de B/2 px
// d'un bord), échelle log2(bloc / 16) ; x = (log(f + 1e-3) - mu) / sd (bord, échelle : (v - mu) / sd),
// poids = softmax(W x + b) sur les candidats présents. Modèles sans NV appris sur MC calculé sans flux NVIDIA.
//
// Format texte (jetons séparés par des espaces, « # » jusqu'à la fin de ligne = commentaire) :
//   muxiveo-rife-selector 1
//   model <famille> <candidats> <bmin> <bmax> <nfeat> <ncand>
//   mu <nfeat réels>
//   sd <nfeat réels>
//   weight <nfeat réels>          (ncand lignes, candidat par candidat)
//   bias <ncand réels>
#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct SelectorModel
{
    std::string family;   // modèle RIFE de R1 / R05 / Rbwd (v415, v415mvo, v46…)
    uint32_t candidates;  // bits : 1 R1, 2 R05, 4 MC, 8 Rbwd, 16 NV
    int bmin;             // bande de taille de bloc (px) ; 0 = modèle global
    int bmax;
    int nfeat;
    int ncand;
    float mu[16];
    float sd[16];
    float weight[5 * 16];  // weight[candidat * nfeat + indice]
    float bias[5];
};

// Version du format lue ; un fichier d'une autre version est refusé.
constexpr int SELECTOR_WEIGHTS_FORMAT = 1;

// Lit et valide le fichier de poids (dimensions cohérentes avec les candidats, réels finis).
bool load_selector_weights(const std::filesystem::path& path, std::vector<SelectorModel>& models, std::string& error);

// Familles distinctes du fichier, dans l'ordre de première apparition.
std::vector<std::string> selector_families(const std::vector<SelectorModel>& models);
