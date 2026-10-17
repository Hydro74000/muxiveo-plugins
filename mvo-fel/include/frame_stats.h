/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Statistiques de l'image finale (HDR10+ recalculé, MaxCLL/MaxFALL) : même calcul sur CPU et GPU.
 *
 * Entrée : codes RGB PQ 16 bits (BT.2020) de l'image reconstruite, sur la fenêtre encodée.
 * Lumière linéaire par une table EOTF float commune (fel_stats_eotf, nits) ; luminance
 * Y = 0,2627 R + 0,6780 G + 0,0593 B en float, sans contraction (identique au GPU, « precise »).
 * maxRGB est classé par code PQ (code >> 6 : 1024 cases), Y par les bornes fel_stats_edges
 * (EOTF des codes 64·i, nits). Sommes de maxRGB linéaire en float par ligne, dans l'ordre des
 * colonnes, puis en double entre lignes : résultats identiques au bit près entre CPU et GPU. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define FEL_STATS_BINS 1024
/* Mots par ligne du résultat GPU : max R, G, B (codes), somme maxRGB (bits float), pixels Y ≤ 100 nits. */
#define FEL_STATS_ROW_WORDS 5
struct fel_frame_stats {
    uint32_t hist_maxrgb[FEL_STATS_BINS];  /* maxRGB, case = code PQ >> 6 */
    uint32_t hist_y[FEL_STATS_BINS];       /* luminance Y, case i : edges[i] ≤ Y < edges[i+1] */
    uint32_t max_code[3];                  /* codes PQ maximaux de R, G, B */
    double sum_maxrgb;                     /* somme de maxRGB linéaire (nits) */
    uint64_t pixels, y_le_100;             /* pixels mesurés ; luminance ≤ 100 nits */
};
/* Table EOTF PQ des 65 536 codes (float, nits) et bornes des cases de Y (1 025 valeurs, nits). */
const float *fel_stats_eotf(void);
const float *fel_stats_edges(void);
/* Une ligne de count pixels (plans R, G, B) : histogrammes cumulés, résultats de la ligne dans row
 * (FEL_STATS_ROW_WORDS mots, même format que le GPU). */
void fel_stats_row(const uint16_t *r, const uint16_t *g, const uint16_t *b, int count,
                   uint32_t *hist_maxrgb, uint32_t *hist_y, uint32_t *row);
/* Remise à zéro, puis assemblage des lignes (ordre des lignes) : maxima, somme double, compteurs. */
void fel_stats_reset(struct fel_frame_stats *stats);
void fel_stats_add_rows(struct fel_frame_stats *stats, const uint32_t *rows, int count, int width);
#ifdef __cplusplus
}
#endif
