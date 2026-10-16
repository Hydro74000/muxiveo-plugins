/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Interface C de la virgule fixe multi-mots (lot A2') pour le backend GPU. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <libavutil/dovi_meta.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Paramètres d'une image en mots 32 bits (tableau uniforme du shader) :
 * 0 : d (dénominateur binaire des coefficients), 1 : q (matrice N·2^q), 2 : résiduel ;
 * courbe c à FEL_FIXED_CURVE0 + c·FEL_FIXED_CURVE : +0 pivots, +1..9 codes, +10..17 méthodes,
 * +18..25 ordres MMR, +26 + 24k + i coefficients entiers, +218 et +222 bornes code/1023 (128 bits),
 * +226 + k erreur du segment (float) ; global à FEL_FIXED_GLOBAL : +2c pente, +6+2c 2·seuil - pente
 * (64 bits), +12+4c décalage (128 bits), +24+i matrice N (int32), +33+2i matrice LMS→RGB S2.61,
 * +51+i son erreur (float). Entiers petit-boutistes, complément à deux. */
#define FEL_FIXED_CURVE0 3
#define FEL_FIXED_CURVE 236
#define FEL_FIXED_GLOBAL (FEL_FIXED_CURVE0 + 3*FEL_FIXED_CURVE)
#define FEL_FIXED_WORDS (FEL_FIXED_GLOBAL + 61)
/* Paramètres du RPU ; 0 si hors cadre (rendu en double pour cette image). */
int fel_fixed_params(const AVDOVIMetadata *metadata, uint32_t *words, int capacity);
/* Tables et constantes (valeurs 64 bits, deux mots chacune) du tableau uniforme fx_t du shader :
 * ln c_j, 1/c_j, 2^(j/64) (64 chacune), e^(j/64) - 1 (16), coefficients ln, exp, expm1 (9 chacune),
 * constantes (7 : ln 2, 32/2523, 8192/1305, 64/ln 2, ln x0, ln2/64 bas et haut). Renvoie le nombre de
 * mots, 0 si la capacité est insuffisante. */
#define FXT_LNC 0
#define FXT_INVC 64
#define FXT_POW2 128
#define FXT_EXPM1 192
#define FXT_LNK 208
#define FXT_EXPK 217
#define FXT_EM1K 226
#define FXT_CONST 235
#define FEL_FIXED_TABLE_WORDS 484   /* 242 valeurs 64 bits, 121 uvec4 */
int fel_fixed_tables(uint32_t *words, int capacity);
#ifdef __cplusplus
}
#endif
