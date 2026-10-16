# Rapide amélioré (lot R)

16 octobre 2026. Lot R du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md).

## Principe

Le mode `fast` (Rapide) n'utilise plus le décodage Dolby Vision de libplacebo, mais un shader float32 Muxiveo. Aucun double n'est requis : il fonctionne sur tous les GPU.

- **Comme le mode exact :** sommes Spline16 entières, choix du segment par comparaison entière exacte, zéro NLQ exact.
- **En float32 :** mapping (polynôme ou MMR), NLQ, matrices.
- **EOTF sans annulation :** p − 1 = expm1(ln(x)/m2), par un polynôme de degré 7 (|t| ≤ 0,18 utile, reste < 3·10⁻¹¹) ; ensuite p − c₁ = h + u et c₂ − c₃·p = h − c₃·u, avec h = 0,1640625.
- **OETF sans annulation :** (c₁ + c₂·v)/(1 + c₃·v) = 1 − h(1 − v)/(1 + c₃·v), puis exp(m2·log1p(…)) ; log1p par la série atanh.
- **Fonctions natives :** `log2`, `exp2` et `pow` sont utilisées là où leur erreur relative suffit. Les séries complètes de toutes les fonctions donnaient une précision un peu meilleure (MMR : 10 codes au lieu de 24), mais coûtaient 25 % de débit de plus sur iGPU.
- **Coefficients du RPU :** passés en tableau uniforme.
- **Repli :** `MVO_FEL_GPU_LIBPLACEBO=1` rétablit le décodage libplacebo, pour comparaison ; une géométrie non rationnelle y passe aussi.

## Validation

Écart au CPU certifié, `fel_gpu_precision_test`, 200 graines par stress. Unité : codes 16 bits.

| Stress | libplacebo : écart max | libplacebo : valeurs différentes | Shader R : écart max | Shader R : valeurs différentes |
|---|---:|---:|---:|---:|
| Identité (annulations LMS→RGB) | 1 026 | 131 011 | 112 | 5 132 |
| Matrices Dolby réalistes | 9 151 | 159 298 | 1 | 5 220 |
| MMR avec signaux sur un pivot | 664 | 86 903 | 24 | 4 299 |

L'écart résiduel du stress identité vient de la matrice LMS→RGB en float32. Elle compense des termes de l'ordre de 0,4 jusqu'à 10⁻⁵ (couleurs saturées lumineuses), et le float32 ne peut pas corriger cette annulation sans précision étendue, que ce mode exclut.

Débit, 300 images UHD, sortie 10 bits :

| GPU | libplacebo | Shader R |
|---|---:|---:|
| RTX 4070 Ti SUPER | 76,6 i/s | 74–78 i/s |
| iGPU Raphael | 22,6 i/s | 19,8 i/s (−12 %) |

Scènes réelles, attestation de schéma 4 (Ryan, Toy Story 5, Moana ; RTX et Raphael) : réussie.

| Mesure | libplacebo | Shader R |
|---|---|---|
| Écart max au CPU | 1–2 codes 16 bits (0,125 PQ12 sur Toy Story 5) | **1 code 16 bits** sur toutes les scènes (0,0625 PQ12) |
| Écart à l'oracle `kernel64` | 0,077–0,087 PQ12 | 0,066–0,098 PQ12 |

Sur contenu réel, le décodage libplacebo restait déjà proche du CPU. Le lot R corrige surtout les cas extrêmes (couleurs saturées lumineuses, mappings à fortes annulations), où l'ancien mode Rapide dérivait de plusieurs centaines de codes.

Les textes de Muxiveo (infobulle « dérive colorimétrique forte » du mode Rapide) seront revus au lot V, au vu de ces mesures.
