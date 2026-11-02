// Statistiques de chaque trame produite (--frame-statistics), au format « mvo-fel-frame-statistics » version 1 de
// l'extension mvo-fel (mvo-fel/docs/statistiques-image.md) : Muxiveo en tire les MaxCLL / MaxFALL exacts de la sortie,
// comme pour une intégration FEL.
//
// Mesure : trame Y'CbCr (matrice et plage de --matrix / --range) remontée en 4:4:4 par interpolation bilinéaire selon
// la position de la chroma, R'G'B' quantifié en codes PQ 16 bits, lumière linéaire par une table EOTF PQ de 65 536
// valeurs (calculée en double) ; maxRGB = max(R, G, B), luminance Y = 0,2627 R + 0,6780 G + 0,0593 B ; histogrammes de
// 1 024 cases (maxRGB : code PQ décalé de 6 bits ; Y : bornes EOTF(64 · i)) ; somme de maxRGB en float par ligne puis
// en double entre lignes. Fenêtre mesurée : la trame entière. Changements de plan : fournis par l'appelant.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "engine.h"
#include "y4m.h"

class FrameStatsWriter
{
public:
    static constexpr int BINS = 1024;
    ~FrameStatsWriter();
    bool open(const std::string& path, const FrameFormat& fmt, const ColorParams& color, std::string& error);
    // Trame au format d'entrée (frame_bytes octets) ; scene_start : première trame d'un plan (la première l'est toujours).
    bool add(const uint8_t* frame, bool scene_start, std::string& error);
    // Histogramme du dernier plan, fermeture du fichier.
    bool finish(std::string& error);

    // Mesure d'une trame (exposée pour les tests) : codes PQ 16 bits maximaux par canal, somme de maxRGB (nits), pixels
    // de Y ≤ 100 nits, histogrammes.
    struct Frame
    {
        std::array<uint32_t, 3> max_code{};
        double sum_maxrgb = 0.0;
        uint64_t y_le_100 = 0, pixels = 0;
        std::array<uint32_t, BINS> hist_maxrgb{}, hist_y{};
    };
    void measure(const uint8_t* frame, Frame& out) const;

private:
    bool write(const std::string& line, std::string& error);
    bool flush_scene(std::string& error);

    FrameFormat fmt;
    ColorParams color;
    FILE* file = nullptr;
    std::vector<float> eotf, edges;
    int64_t frame_index = 0, scene_first = -1, scene_frames = 0;
    std::array<uint64_t, BINS> scene_hist{};
};
