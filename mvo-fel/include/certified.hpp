// SPDX-License-Identifier: LGPL-2.1-or-later
// Référence certifiée du rendu FEL.
//
// Le code RGB16 de référence est l'arrondi exact (au plus proche, seuil exact → code supérieur)
// de la formule évaluée en réels sur : les sommes Spline16 entières (rationnels exacts), les
// paramètres du RPU tels que stockés dans Mapping (doubles), les pivots codes/bl_max exacts, les
// constantes PQ exactes (m1, m2, c1, c2, c3) et la matrice LMS→RGB en double.
//
// Le chemin double (FastReconstruction::rows) encadre chaque valeur linéaire par une borne
// d'erreur démontrée (docs/2026-10-16-cpu-certifie.md) et la classe entre les seuils PQ précalculés.
// Les rares composantes non décidées sont reprises ici en double-double (≈ 106 bits), avec la
// même borne pour une unité d'arrondi 2^-100. Unité compilée sans contraction (CMakeLists) :
// two_sum et le produit exact de Dekker ne sont valides que sans FMA implicite.
#pragma once

#include "reconstruction.hpp"

namespace muxiveo::fel::certified {
// Zéro de l'EOTF x0 = c1^m2 = 7,3095590257839663e-7 : bornes strictes (marge relative 1e-14).
constexpr double PQ_ZERO_LOW = 7.3095590257839e-7, PQ_ZERO_HIGH = 7.3095590257840e-7;
struct DD { double hi = 0, lo = 0; };
DD log(const DD &x);
DD exp(const DD &x);
// EOTF PQ en double-double ; kappa = conditionnement relatif d ln L / d ln x (0 si L = 0,
// infini près du zéro de l'EOTF). Signal au-delà du pôle : exception.
DD eotf(const DD &x, double &kappa);
// Erreur relative démontrée de l'EOTF calculée (chemin double : u = 2^-53, double-double : 2^-100) ;
// validée face à decimal (tests/certified_eotf.py) avec une marge d'au moins 2.
double eotf_relative_error(double value, double kappa, double u);

// Entrées exactes d'un pixel du chemin entier : code BL Y (sur 1023), sommes Spline16 entières
// (numérateur, dénominateur) des chroma BL et des résiduels EL (offset NLQ retiré), segments.
struct ExactPixel {
    int luma = 0;
    std::array<int64_t, 2> chroma{}, chroma_den{};
    std::array<int64_t, 3> residual{}, residual_den{};
    std::array<unsigned, 3> piece{};
};
// Codes RGB16 (ordre R, G, B) des composantes du masque (bit c) recalculés en double-double.
// Renvoie le masque des composantes indécises : encadrement double-double encore à cheval sur
// un seuil, code choisi par la valeur centrale.
// linear (facultatif) : valeurs linéaires R, G, B en double-double.
unsigned classify(const Mapping &mapping, const ExactPixel &pixel, unsigned mask,
                  std::array<uint16_t, 3> &codes, std::array<DD, 3> *linear = nullptr);
}
