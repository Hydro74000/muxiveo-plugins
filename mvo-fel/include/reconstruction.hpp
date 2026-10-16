// SPDX-License-Identifier: LGPL-2.1-or-later
// Port CPU des opérations Dolby Vision de libplacebo (voir NOTICES.md).
#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "output_format.h"
extern "C" {
#include <libavutil/dovi_meta.h>
#include <libavutil/frame.h>
}

namespace muxiveo::fel {
using Vec = std::array<double, 3>;
using Mat = std::array<Vec, 3>;

struct Curve {
    unsigned count = 0;
    std::array<double, 9> pivots{};
    std::array<int, 9> codes{};   // pivots en codes BL : choix de segment exact sur sommes entières
    std::array<int, 8> method{};
    std::array<unsigned, 8> order{};
    std::array<Vec, 8> poly{};
    std::array<double, 8> constant{};
    std::array<std::array<std::array<double, 7>, 3>, 8> mmr{};
    // Poids d'erreur du segment : |ŝ - s| ≤ (weight + 1)·u pour un calcul en unité d'arrondi u
    // (polynôme : 15·Σ|a| ; MMR : 80·(|c| + Σ|m|), termes dans [0, 1] ; +u : pivot de bornage).
    std::array<double, 8> weight{};
};

struct Mapping {
    std::array<Curve, 3> curves;
    Vec offset{}, slope{}, threshold{}, nonlinear_offset{};
    Mat nonlinear{}, linear{};
    double bl_max = 1023;  // pivots exacts : codes / bl_max
    bool residual = false;
    explicit Mapping(const AVDOVIMetadata &metadata);
};

double spline16(double x);
// Seuil d'arrondi PQ k+½ (k = 0..65534) dans le domaine linéaire, arrondi vers le bas en double :
// pq_threshold(k) < EOTF((k+½)/65535) < suivant(pq_threshold(k)). Partie basse (double-double) :
// |EOTF((k+½)/65535) - pq_threshold(k) - pq_threshold_low(k)| ≤ 2^-105·seuil. Table data/pq_thresholds.bin.
constexpr int PQ_THRESHOLDS = 65535;
double pq_threshold(int k);
double pq_threshold_low(int k);
// Matrice LMS → RGB BT.2020 (constantes en double : valeurs exactes de la référence).
const Mat &lms_to_rgb_matrix();
double pq_eotf(double x);
// EOTF du chemin certifié (vectorisée, erreur bornée par certified::eotf_relative_error) et
// conditionnement relatif κ = d ln L / d ln x, en place sur un tableau.
void pq_eotf_kappa(double *values, double *kappa, int count);
double pq_oetf(double x);
Vec reshape(const Mapping &mapping, const Vec &bl);
// Segments déjà choisis (index de pivot par composante), sans comparaison flottante.
Vec reshape(const Mapping &mapping, const Vec &bl, const std::array<unsigned, 3> &piece);
Vec compose(const Mapping &mapping, const Vec &bl, const Vec &el);
Vec to_rgb(const Mapping &mapping, const Vec &signal);

// Les coordonnées sont exprimées au centre des pixels BL (entiers).
double sample_plane(const AVFrame &frame, int component, int x, int y,
                    int bl_width, int bl_height, bool enhancement);
class FrameSampler {
    struct Axis {
        std::array<int, 4> indices{};
        std::array<double, 4> weights{};
    };
    struct Plane {
        const uint8_t *data;
        int stride, step, offset, shift, depth;
        std::vector<Axis> horizontal, vertical;
    };
    std::array<Plane, 3> planes;
public:
    FrameSampler(const AVFrame &frame, int bl_width, int bl_height, bool enhancement);
    double sample(int component, int x, int y) const;
};
void reconstruct_rows(const FrameSampler &bl, const FrameSampler &el, const Mapping &mapping,
                      AVFrame &rgb, int first_row, int last_row);

// Axe Spline16 à poids entiers exacts (phases multiples de 1/8 : dénominateurs ≤ 2560).
struct IntegerAxis {
    std::vector<std::array<int32_t, 4>> index, weight;
    std::vector<int32_t> denominator;
    int32_t weight_sum = 0;  // max des Σ|poids| : borne des sommes entières
    // false si une phase n'est pas rationnelle de ce type.
    bool build(int count, int size, double ratio, double layer, double chroma, int subsample);
};
// Compteurs du rendu certifié : composantes, reprises en double-double, indécises (décision par
// la valeur centrale, aucune attendue), différentes de l'arrondi double historique.
// Mode étendu : écart maximal |R̂ - R| rapporté à la borne du chemin double (≤ 1 attendu).
struct CertifiedStats {
    long long components = 0, rescued = 0, undecided = 0, changed = 0;
    double worst_ratio = 0;
    CertifiedStats &operator+=(const CertifiedStats &o) {
        components += o.components; rescued += o.rescued; undecided += o.undecided; changed += o.changed;
        worst_ratio = worst_ratio > o.worst_ratio ? worst_ratio : o.worst_ratio;
        return *this;
    }
};
// Reconstruction CPU séparable en entiers exacts (zéro et signe du résiduel exacts) ;
// valid() faux si une géométrie sort de ce cadre : FrameSampler en double en repli.
class FastReconstruction {
    struct Plane {
        const uint8_t *data = nullptr;
        int stride = 0, width = 0, height = 0, offset = 0;
        IntegerAxis horizontal, vertical;
    };
    std::array<Plane, 2> base_chroma;
    std::array<Plane, 3> enhancement;
    const uint8_t *luma = nullptr;
    int luma_stride = 0, width = 0, height = 0;
    std::array<int, 6> geometry{};
    bool ok = false;
    static bool plane(Plane &plane, const AVFrame &frame, int component, int bw, int bh, bool el);
    static std::array<int, 6> key(const AVFrame &bl, const AVFrame &el);
public:
    // Axes calculés une fois pour la géométrie (dimensions, positions chroma) des couches.
    FastReconstruction(const AVFrame &bl, const AVFrame &el);
    bool valid() const { return ok; }
    // Plans et offsets NLQ de l'image courante ; false si la géométrie a changé.
    bool bind(const AVFrame &bl, const AVFrame &el, const AVDOVIMetadata &metadata);
    // Codes RGB16 certifiés : arrondi exact de la formule de référence (voir certified.hpp).
    // extended : toutes les composantes calculées en double-double (tests).
    CertifiedStats rows(const Mapping &mapping, AVFrame &rgb, int first_row, int last_row,
                        bool extended = false) const;
};
void reconstruct_rows(const AVFrame &bl, const AVFrame &el, const Mapping &mapping,
                      AVFrame &rgb, int first_row, int last_row);
// RGB PQ 16 bits (gbrp16le) vers Y'CbCr 4:2:0 limité : lignes chroma [first, last) et
// les deux lignes luma de chacune. Arithmétique entière exacte (constantes BT.2020 rationnelles,
// poids chroma en 64es, seuils de tramage en 8192es).
void convert_rows(const AVFrame &rgb, AVFrame &yuv, const fel_output &output, int first, int last);
}
