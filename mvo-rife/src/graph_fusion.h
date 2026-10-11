// Fusions sans poids supplémentaires ; un graphe non reconnu reste inchangé.
#ifndef MUXIVEO_GRAPH_FUSION_H
#define MUXIVEO_GRAPH_FUSION_H
#include <string>

int fuse_residual_activations(const std::string& param, std::string& result);
#endif
