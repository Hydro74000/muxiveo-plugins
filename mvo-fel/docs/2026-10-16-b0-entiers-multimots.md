# Banc B0 : mapping en entiers multi-mots

16 octobre 2026. Lot B0 du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md).

## Banc

L'outil local `b0_bench.cpp` (non livré) mesure le même bloc en deux variantes. Le bloc comprend :

- les signaux BL : trois divisions ;
- les 7 monômes et leurs puissances ;
- un MMR d'ordre 3 sur les trois composantes ;
- le bornage ;
- le NLQ : trois divisions ;
- le décalage ;
- la matrice Y'CbCr→LMS'.

Les deux variantes :

- **double** : même code que le shader exact ;
- **virgule fixe sur mots de 32 bits** : signaux et monômes en Q0.64 (produits `umulExtended` tronqués) ; MMR accumulé exactement sur 128 bits (coefficient entier × Q0.64) ; divisions par estimation flottante et deux corrections par le reste exact ; NLQ et matrice exacts sur 128 bits.

Les entrées sont générées par hachage, identiques pour les deux variantes : image UHD, dénominateurs Spline16 1, 80 et 6 400. On retient le meilleur temps GPU (`pl_timer`), la première exécution (compilation) étant exclue.

## Résultats

| GPU | Double | Virgule fixe | Rapport double / fixe | Écart relatif max |
|---|---:|---:|---:|---:|
| iGPU Raphael (RDNA2) | 79,6 ms | 146,9 ms | **0,54** | 9,2·10⁻¹⁵ |
| RTX 4070 Ti SUPER | 4,72 ms | 1,79 ms | **2,64** | 9,2·10⁻¹⁵ |

## Décision

**Critère du plan :** au moins 2 fois plus rapide que le double sur Raphael. Il n'est pas atteint : sur RDNA2, la multiplication entière 32 bits tourne au quart du débit, contre 1/16 pour le double. **B1 est arrêté**, et l'exactitude sans `shaderFloat64` (Arc, Mac) n'est pas poursuivie par cette voie.

**Observation pour A2 et T0 :** sur la RTX, où le double tourne au 1/64, la virgule fixe multi-mots calcule le mapping 2,64 fois plus vite, avec une précision des termes de l'ordre de 10⁻¹⁴. Après A2, l'EOTF passera dans l'étage float32 et le mapping deviendra le poste dominant du rendu exact sur GPU dédié. Cette variante est alors candidate pour les GPU NVIDIA, sans Tensor Cores. Elle devient aussi la référence à battre pour T0.
