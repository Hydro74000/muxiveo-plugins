/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Interface C du CPU certifié pour le backend GPU (gpu_backend.c). */
#pragma once
#include <stdint.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/frame.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Reprise GPU : 10 mots par entrée (FEL_RESCUE_WORDS) :
 * 0 : x | y << 16 ; 1 : code BL Y | segments << 16 (4 bits par composante) | masque des
 * composantes indécises << 28 ;
 * 2, 3 : sommes chroma BL Cb, Cr ; 4 : leur dénominateur ; 5–7 : sommes résiduelles EL Y, Cb, Cr
 * (offset NLQ retiré) ; 8, 9 : dénominateurs EL luma et chroma. */
#define FEL_RESCUE_WORDS 10
/* Codes RGB16 certifiés (R, G, B par entrée). Renvoie le nombre de composantes indécises,
 * -1 si le RPU ou un signal est refusé. */
int fel_certified_codes(const AVDOVIMetadata *metadata, const uint32_t *entries, int count, uint16_t *codes);
/* Diagnostic GPU (FEL_DIAG_WORDS mots par pixel, lignes y % stride == 0) : par composante R̂ et
 * sa borne (doubles en deux mots, poids faible d'abord), puis code | certain << 16. Compare au
 * double-double : *worst = max |R̂ - R| / borne ; renvoie le nombre de codes certains différents
 * du code certifié, -1 si la géométrie ou le RPU sort du chemin entier. */
#define FEL_DIAG_WORDS 15
/* Format virgule fixe (lot A2') : 6 mots par composante, R en Q0.96 sur 128 bits (complément à deux),
 * borne en float32, puis code | certain << 16. */
#define FEL_DIAG_WORDS_FIXED 18
int fel_certified_check(const AVDOVIMetadata *metadata, const AVFrame *bl, const AVFrame *el,
                        int stride, int fixed_format, const uint32_t *diag, double *worst);
#ifdef __cplusplus
}
#endif
