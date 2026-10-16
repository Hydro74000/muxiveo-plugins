// SPDX-License-Identifier: LGPL-2.1-or-later
// Rendu certifié en virgule fixe multi-mots (lot A2') : modèle exact du shader GPU de même nom.
//
// Formats : Qa.b non signé, Sa.b signé (valeur = entier / 2^b).
// - Signaux σ = n/(1023·D) en Q0.64 (division entière exacte, arrondi vers le bas) ; monômes par
//   produits Q0.64 tronqués (erreur < 2^-64 chacun).
// - Mapping, NLQ, décalage : accumulation exacte sur 128 bits à l'échelle 2^-(64+d), d le
//   dénominateur binaire des coefficients du RPU ; matrice Y'CbCr→LMS' exacte (entrées n/2^q).
// - EOTF : ln par table de 64 entrées et série de degré 8 (S5.58), num = c1·expm1(ln(x/x0)/m2)
//   sans annulation, g = num/den par corrections de Newton vérifiées, L = exp(ln(g)/m1) par table
//   2^(j/64) et Taylor ; erreur relative ≤ 2^-50·(1 + 1/ln(x/x0)) (docs/2026-10-16-a2-virgule-fixe.md).
// - Matrice LMS→RGB en S2.61, valeurs linéaires et seuils PQ en Q0.96 ; classement comme le CPU
//   certifié, composantes indécises reprises par certified::classify.
#pragma once

#include "certified.hpp"

namespace muxiveo::fel::fixed {
__extension__ typedef __int128 i128;            // GCC et Clang (MinGW, macOS) ; MSVC hors cadre
__extension__ typedef unsigned __int128 u128;
// Tables et constantes partagées avec le shader (calculées en double-double).
struct Tables {
    uint64_t inv_c[64], pow2[64], expm1[16];      // 1/c_j Q0.64, 2^(j/64) Q1.63, e^(j/64)-1 Q0.64
    int64_t ln_c[64];                              // ln c_j, c_j = 1 + (2j+1)/128, S63
    int64_t ln_coef[9], exp_coef[9];               // (-1)^(k+1)/k et 1/k! en S2.62 (k ≤ 8)
    uint64_t expm1_coef[9];                        // 1/k! en Q0.64 (k ≥ 2)
    uint64_t ln2_q64, c32_2523_q64, c8192_1305_q61, c64_ln2_q56;
    int64_t lnx0_s58;                              // ln x0 = m2·ln c1
    uint64_t ln2_64_q90[2];                        // ln2/64 en Q0.90 (bas, haut)
};
const Tables &tables();

// Paramètres d'une image (RPU) ; ok faux si un coefficient sort du cadre (coefficient hors de
// 32 bits, matrice Y'CbCr→LMS' non dyadique, d > 32) : rendu en double.
struct Params {
    bool ok = false;
    int d = 0, q = 0;                              // échelles 2^-d (coefficients), 2^-q (matrice N)
    bool residual = false;
    struct Curve {
        unsigned count = 0;
        int codes[9] = {};
        int method[8] = {}, order[8] = {};
        int32_t coef[8][24] = {};                  // polynôme a0..a2, ou MMR c puis m[k][j]
        i128 low = 0, high = 0;                // bornes code/1023 à l'échelle 2^-(64+d), tronquées
        double error[8] = {};                      // erreur absolue du segment (valeur)
    } curves[3];
    int64_t slope[3] = {}, threshold2[3] = {};     // pente entière, 2·seuil - pente (entiers du RPU)
    i128 offset[3] = {};                       // décalage Y'CbCr à l'échelle 2^-(64+d), tronqué
    int32_t n[9] = {};                             // matrice Y'CbCr→LMS' · 2^q
    int64_t m[9] = {};                             // matrice LMS→RGB (A·linear) en S2.61
    double m_error[9] = {};                        // |M_S61 - M exacte| majoré
};
bool prepare(const Mapping &mapping, const AVDOVIMetadata &metadata, Params &params);

// EOTF en virgule fixe : x > 0 à l'échelle 2^-fraction ; L = mant·2^(n-63). dl = ln(x/x0) (valeur
// approchée, pour la borne). false hors cadre (x ≳ 1,33, vers le pôle). L = 0 (mant nul) si x ≤ x0.
bool eotf(u128 x, int fraction, uint64_t &mant, int &n, double &dl);

// Valeurs linéaires R, G, B en Q0.96 (signées) et bornes d'erreur (valeur) d'un pixel ; masque
// des composantes non certaines (domaine ou borne). Codes candidats classés sur la table Q0.96.
unsigned pixel(const Params &params, const certified::ExactPixel &input, std::array<uint16_t, 3> &codes,
               i128 (&linear)[3], double (&bound)[3]);
}
