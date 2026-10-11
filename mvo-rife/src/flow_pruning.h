// Élimination en mémoire des seules sorties de flux dont l'absence d'usage est prouvée.
#ifndef MUXIVEO_FLOW_PRUNING_H
#define MUXIVEO_FLOW_PRUNING_H

#include <string>
#include <vector>

// Retourne le nombre de branches réduites. Un format inconnu conserve le graphe
// original et laisse les poids de sortie vides ; aucun fichier n'est modifié.
int prune_unused_flow(const std::string& param, const std::vector<unsigned char>& weights,
                      std::string& output_param, std::vector<unsigned char>& output_weights);

#endif
