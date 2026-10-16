/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Format des images produites : RGB PQ 16 bits, ou Y'CbCr BT.2020nc limité 4:2:0
 * (sous-échantillonnage bilinéaire au point chroma, quantification tramée). */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
enum fel_output_format {
    FEL_OUT_GBRP16 = 0, FEL_OUT_YUV420P16, FEL_OUT_YUV420P10, FEL_OUT_YUV420P8, FEL_OUT_P010
};
enum fel_dither { FEL_DITHER_NONE = 0, FEL_DITHER_ORDERED, FEL_DITHER_BLUE };
/* Précision du rendu couleur GPU (le CPU produit toujours la référence certifiée). */
enum fel_precision { FEL_PRECISION_FAST = 0, FEL_PRECISION_STANDARD, FEL_PRECISION_EXACT };
struct fel_output {
    int format, dither;
    int chroma;            /* AVChromaLocation */
    int crop_x, crop_y;    /* origine de la zone produite dans l'image reconstruite */
    int width, height;     /* dimensions produites (paires en 4:2:0) */
};
/* Profondeur des échantillons quantifiés ; 0 pour le RGB. */
static inline int fel_output_depth(int format)
{
    return format == FEL_OUT_YUV420P16 ? 16 : format == FEL_OUT_YUV420P10 || format == FEL_OUT_P010 ? 10
         : format == FEL_OUT_YUV420P8 ? 8 : 0;
}
/* Centre du filtre chroma en coordonnées luma, relatif à 2i / 2j. */
static inline void fel_chroma_offset(int chroma, double *x, double *y)
{
    /* AVChromaLocation : 1 left, 2 center, 3 topleft, 4 top, 5 bottomleft, 6 bottom. */
    *x = chroma == 2 || chroma == 4 || chroma == 6 ? 0.5 : 0.0;
    *y = chroma == 3 || chroma == 4 ? 0.0 : chroma == 5 || chroma == 6 ? 1.0 : 0.5;
}
/* Seuil de quantification dans [0, 1) : 0,5 sans tramage. */
double fel_dither_threshold(int dither, int x, int y);
#ifdef __cplusplus
}
#endif
