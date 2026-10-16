# Lent en virgule fixe multi-mots (lot A2')

16 octobre 2026. Lot A2' du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md). Il remplace l'étage float32 d'A2, abandonné : en float32, l'erreur relative de L plafonne vers 10⁻⁶.

## Principe

Le mode `exact` (Lent) dispose d'une seconde variante, entièrement en arithmétique entière sur mots de 32 bits. Elle donne les **mêmes codes** que le CPU certifié et la variante double d'[A1](2026-10-16-gpu-exact-certifie.md) : même classement entre seuils, mêmes reprises par le CPU, même auto-test.

Le modèle de référence est `src/fixed_point.cpp`, traduit opération par opération dans `src/shaders/fixed_point.glsl` et `fixed_render.glsl`. Le GPU reproduit la référence au bit près : vérifié sur 4 configurations × 6 144 composantes.

| Étape | Format | Calcul |
|---|---|---|
| Signaux BL σ = n/(1023·D) | Q0.64 | division entière exacte (estimation flottante, corrections par le reste, vérification 0 ≤ reste < D) |
| Monômes et puissances (MMR d'ordre ≤ 3) | Q0.64 | produits tronqués, erreur < 17·2⁻⁶⁴ |
| Mapping, NLQ, décalage | 128 bits, échelle 2⁻⁽⁶⁴⁺ᵈ⁾ | accumulation exacte (coefficients entiers du RPU, d ≤ 28) ; NLQ = ⌊\|te\|·pente·2⁶⁴/De⌋ + (2·seuil − pente)·2⁶³ |
| Matrice Y'CbCr→LMS' | 128 bits, échelle 2⁻⁽⁶⁴⁺ᵈ⁺ᑫ⁾ | exacte (entrées n/2^q, \|n\| < 2²⁰) |
| ln x | S5.58 | table de 64 entrées (c_j = 1 + (2j+1)/128) et série de degré 8 ; e·ln 2 sur 128 bits |
| num = p − c₁ | Q0.64 | c₁·expm1(ln(x/x₀)/m2), sans annulation (table e^(j/64) et Taylor) |
| g = num/den | Q1.63 | division vérifiée ; cadre num < 43/256 (x ≲ 1,33), au-delà reprise CPU |
| L = g^(1/m1) | mantisse Q1.63 et exposant | ln g, produit par 8192/1305, exponentielle par table 2^(j/64) et Taylor d'ordre 8 |
| Matrice LMS→RGB, R, seuils | Q0.96 | M en S2.61 (produit double, erreur 3,01·2⁻⁵³·Σ\|A·B\| + 2⁻⁶²) ; seuils T·2⁹⁶ exacts |

**Borne de L.** L'erreur relative est d'au plus 2⁻⁵⁰·(1 + 1/ln(x/x₀)). La résolution absolue de ln x (environ 3·2⁻⁵⁸) domine près du zéro de l'EOTF. Ailleurs, l'erreur vaut environ 10⁻¹⁶.

La propagation de l'erreur du mapping suit le conditionnement κ, comme pour A1. Ici κ est calculé à partir de num et den exacts, donc sans annulation.

La borne est calculée en float32, avec une marge de 10⁻³. Elle est contrôlée face au double-double : l'erreur observée vaut au plus 0,37 fois la borne pour l'EOTF et 0,23 fois pour le pipeline (`tests/fixed_point_test.cpp`).

## Choix de la variante

Au premier passage en mode exact, l'auto-test rend les deux images synthétiques avec chaque variante. Chacune doit respecter ces critères :

- borne tenue face au double-double ;
- codes certains identiques au CPU certifié (diagnostic dans un format propre à la virgule fixe).

La plus rapide des variantes valides est retenue. Une image dont le RPU sort du cadre de la virgule fixe est rendue en double. C'est le cas d'un coefficient hors de 32 bits, d'un d > 28, d'une matrice non dyadique ou d'une pente hors de 32 bits. `MVO_FEL_GPU_FIXED=0` ou `1` impose une variante (tests).

## Validation et mesures

| Contrôle | Résultat |
|---|---|
| GPU face à la référence C++ (4 configurations, 6 144 composantes chacune) | identique au bit près |
| `fel_gpu_precision_test`, virgule fixe imposée (3 × 200 graines, reprise forcée, zones uniformes) | **0 code différent** du CPU certifié |
| Attestation de schéma 4 (3 scènes réelles, RTX et Raphael, transport direct) | réussie : `exact` identique au CPU certifié ; acceptée par `package.py` |
| Reprises sur scènes réelles (RTX, virgule fixe) | **0** composante reprise sur les 4 extraits (A1 en double : jusqu'à 24 par image) ; 0 indécise |
| Auto-test RTX 4070 Ti SUPER | double 44,3 ms, virgule fixe 21,9 ms : virgule fixe retenue |
| Auto-test iGPU Raphael | double 0,74 s, virgule fixe 1,73 s : double retenu (multiplication entière au quart de débit, voir [B0](2026-10-16-b0-entiers-multimots.md)) |

Composition d'une image UHD sur la RTX :

| Variante | Composition |
|---|---:|
| A1 double | 18,7 ms |
| Virgule fixe, premier jet | 44,7 ms |
| Virgule fixe optimisée | **7,8 ms** |

Le premier jet perdait son temps en mémoire locale : tableaux constants indexés dynamiquement, composantes de `uvec4` indexées par une variable, puissances des monômes indexées par l'ordre MMR. Une fois ces trois défauts corrigés, la composition tombe à 7,8 ms.

Débit du plugin, 300 images UHD, sortie 10 bits, sur la RTX :

| Version | Débit |
|---|---:|
| Avant le chantier | 27,6 i/s |
| A1 (double) | 34,4 i/s |
| **A2'** (virgule fixe) | **43,6 i/s** |

Sur l'extrait de référence, avec le décodage Vulkan Video, le débit atteint 51,4 i/s. Le libellé du moteur indique la variante retenue (« précision exacte (certifiée), virgule fixe »).

Le débit n'est plus limité par la composition. L'étage de rendu enchaîne en série téléversement, composition, rapatriement de la liste et téléchargement.

## Limites

- Sur iGPU AMD (RDNA2), la virgule fixe reste plus lente que le double. Le mode exact y reste lui-même plus lent que le CPU certifié.
- Les signaux LMS' au-delà de 1,33, vers le pôle, sont repris par le CPU. Le contenu réel du corpus reste sous 1.
- L'auto-test dure environ 70 ms sur la RTX et 2,5 s sur Raphael, à chaque ouverture en mode exact.
