// Muxiveo — mode UHD de RIFE v4 : flux optique calculé à demi-résolution.
//
// Équivalent du paramètre `scale=0.5` de Practical-RIFE (recommandé pour la
// 4K) : l'échelle de chaque bloc IFBlock est doublée (16/8/4/2/1 -> 32/16/8/4/2)
// par réécriture du graphe ncnn au chargement, sans fichier de modèle dédié.

#ifndef MUXIVEO_RIFE_UHD_H
#define MUXIVEO_RIFE_UHD_H

#include <string>

// Réécrit le texte d'un flownet.param RIFE v4 pour le mode UHD.
// Retourne false (et renseigne error) si la structure du graphe n'est pas reconnue.
bool make_uhd_param(const std::string& param, std::string& result, std::string& error);

#endif // MUXIVEO_RIFE_UHD_H
