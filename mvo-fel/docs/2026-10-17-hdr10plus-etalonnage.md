# Étalonnage des statistiques HDR10+ (lot B, 17/10/2026)

Les statistiques de l'image (`docs/statistiques-image.md`) servent à Muxiveo pour recalculer le
HDR10+ d'une sortie dont le FEL est intégré. Deux références sur un disque UHD réel (Gladiator,
version longue, P7 FEL + HDR10+ profil B, extraits de 5 min et 1 min à 39 min).

## Face au L1 Dolby Vision (image reconstruite)

Le L1 du RPU est calculé par l'analyseur Dolby sur le master, c'est-à-dire l'image que le FEL
reconstruit. Statistiques du plugin (Lent, fenêtre recadrée sur la zone active), par plan, en
codes PQ 12 bits :

| | 99 plans (5 min) | 10 plans (1 min) |
|---|---|---|
| moyenne de maxRGB (PQ) | +10 (p90 46) | +57 (p90 83) |
| maximum | +156 | +192 |
| minimum | −44 | −27 |

La moyenne concorde. Le L1 retient des extrêmes robustes (pixels isolés écartés) : maximum plus
bas, minimum plus haut que les nôtres. Le HDR10+ demande le vrai maximum (MaxScl) : pas de
correction.

## Face au HDR10+ de la source (BL)

Le HDR10+ du disque décrit la BL. Mêmes définitions appliquées à la BL décodée (numpy, zone
active), toutes les images d'une minute, rapport médian nous / source par plan :

| MaxScl | AverageRGB | Y99 (position 5) | Y100nit (position 10) |
|---:|---:|---:|---:|
| 0,998 | 0,964 | 1,01 | 0,999 |

Règle d'agrégation des percentiles de maxRGB par plan (rapport médian, écart p90) :

| Percentile | histogramme cumulé du plan | maximum des images | moyenne des images |
|---|---|---|---|
| 50 % | 0,988 (0,23) | 1,54 | **1,008 (0,08)** |
| 75 % | 0,935 (0,22) | 1,52 | **0,967 (0,11)** |
| 90 % | 0,967 (0,19) | 1,37 | **0,967 (0,04)** |
| 95 % | 0,963 (0,25) | 1,25 | **0,972 (0,07)** |
| 99,98 % | 0,875 (0,38) | **0,995 (0,03)** | 0,754 |

Retenu : percentiles 1 à 95 % = moyenne des percentiles de chaque image ; 99,98 % = leur
maximum (comme Y99 et MaxScl). Le plugin émet donc les percentiles de chaque image
(`maxrgb_percentiles`) ; l'histogramme cumulé du plan reste un diagnostic.

Échantillonner une image sur huit sous-estime les maxima (MaxScl 0,83) : la mesure couvre
toutes les images.

## Fenêtre

Les valeurs de la source concordent sur la zone active, sans les bandes noires. Le plugin limite
la mesure à la zone active L5 de chaque image : un encodage pleine image (x265, 3840 × 2160) et
un encodage recadré (NVEncC, 3840 × 1600) du même extrait donnent le même HDR10+.
