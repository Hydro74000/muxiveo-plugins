# Proposition : rendu FEL GPU exact

16 octobre 2026. Prototype de recherche : branche locale `proto/gpu-exact` (commit `ed295b2`), non fusionnée.

## Le problème

Le GPU calcule aujourd'hui la partie couleur en float32, soit environ 7 chiffres significatifs. Ce calcul est fait par le décodage Dolby Vision de libplacebo : mapping polynomial/MMR, NLQ, matrices et PQ. Deux mécanismes le mettent en défaut.

1. **Annulation.** Pour une couleur très saturée et très lumineuse, une composante est la petite différence de deux grands nombres. Par exemple, le rouge d'un vert BT.2020 pur à 2 000 cd/m² vaut 0,43 − 0,43 ≈ 2·10⁻⁵. Avec 7 chiffres, cette différence est mal connue. La PQ amplifie ensuite fortement toute erreur près de zéro.
2. **Pivots.** Le mapping Dolby est découpé en segments séparés par des pivots. Quand un signal tombe exactement sur un pivot, l'arrondi peut choisir le mauvais segment.

Le stress synthétique (graines 1 à 200) compare le GPU à la référence CPU en double. L'écart maximal est en codes 16 bits ; 16 codes 16 bits = 1 code PQ12.

| Cas | Écart maximal du GPU actuel | Pixels > 2 PQ12 |
|---|---:|---:|
| Mapping identité (audit) | 1 478 | 0,006 % |
| Matrices Dolby réalistes | 4 778 (299 PQ12) | 0,1 % |
| MMR avec pivots atteints | 33 096 (2 068 PQ12) | 0,002 % |

Sur films réels, le GPU actuel s'écarte du CPU de 1 code 16 bits sur 6 à 8 % des valeurs, et de 2 codes au maximum.

## Approches mesurées

RTX 4070 Ti SUPER et iGPU AMD Raphael. Débit du plugin seul, extrait de 300 images UHD, sortie 10 bits finale.

| Approche | Stress (face au CPU exact) | Films réels | RTX | iGPU AMD |
|---|---|---|---:|---:|
| Actuel (libplacebo float32) | jusqu'à 4 778 codes | ≤ 2 codes, 6–8 % des valeurs ≠ | 76 i/s | 21 i/s |
| Float-float (deux float32) | ≤ 14 codes | — | — | — |
| Double natif partout | identique au bit près | identique au bit près | 33 i/s | 2,3 i/s |
| **Hybride borné (proposé)** | **≤ 1 code** | **≤ 1 code, 0,25 % des valeurs ≠** | **73–74,5 i/s** | **8,5 i/s** |

### Float-float : écarté

Les transformations exactes (`two_sum`, `two_prod`) sont justes isolément sur les deux GPU. Composées, elles perdent des termes correcteurs, différemment selon le compilateur du pilote :

- sur NVIDIA, la partie basse de ln 2 disparaît dans la réduction de `exp` ;
- sur AMD, `expm1` tombe à 2·10⁻⁷ d'erreur relative.

Une garantie qui dépend de l'optimiseur de chaque pilote n'en est pas une.

### Double natif partout

Le résultat est identique au CPU au bit près, sur le stress comme sur les films. C'est cependant beaucoup trop lent : les GPU grand public calculent le double 16 à 64 fois plus lentement que le float32.

## Proposition : rendu hybride borné

Un shader Muxiveo remplace le décodage Dolby Vision de libplacebo. Les poids Spline16 entiers exacts actuels sont conservés.

1. **Branches exactes en entiers.** Choix du segment de mapping (pivots) et zéro du résiduel NLQ, sur les sommes entières exactes. Le défaut des pivots disparaît sans coût.
2. **Mapping, NLQ et matrice Y'CbCr→LMS' en double.** Ces étapes demandent peu d'opérations, et les annulations du MMR y sont comprises.
3. **EOTF, matrice LMS→RGB et OETF en float32 reformulé, sans annulation.**
   - EOTF : p − c1 = h + u et c2 − c3·p = h − c3·u, avec u = expm1(ln x / m2) et h = 1 − c1 = c2 − c3.
   - OETF : ln(base) = log1p(−h·(1 − u)/(1 + c3·u)).
   - ln, expm1 et log1p sont écrits à la main ; les versions intégrées sont trop imprécises près de 1.
4. **Borne d'erreur calculée pour chaque pixel.** Elle propage l'erreur de l'EOTF, l'annulation de la matrice et la pente de la PQ. Si elle dépasse 0,4 code 16 bits, le pixel est recalculé entièrement en double.

Garantie : au plus 1 code 16 bits d'écart avec le CPU exact, partout, soit 1/64 de code 10 bits, sous la quantification finale. Les pixels recalculés sont identiques au CPU au bit près.

Part de pixels recalculés en double sur films réels :

| Scène | Part |
|---|---:|
| Il faut sauver le soldat Ryan ×2 | 0 % |
| Moana ×3 | 0 % |
| Toy Story 5 | 1,1 % (aplats noirs) |

### Matériel

- Le rendu exige `shaderFloat64` : NVIDIA (toutes générations), AMD, Intel jusqu'à Gen11.
- Le transport direct en profite aussi : le périphérique Vulkan créé par FFmpeg active déjà `shaderFloat64`.
- Sans double : Intel Arc sous Windows, macOS (MoltenVK).
- iGPU AMD Raphael : 29 ms par image aujourd'hui, 42 ms avec le float32 précis et sa borne, puis +56 ms pour le mapping en double. Le double y est 16 fois plus lent ; sur un GPU dédié, le même surcoût reste de l'ordre de la milliseconde.

## Points à décider

1. **Cible de précision.** Recommandé : au plus 1 code 16 bits partout, avec l'hybride. L'alternative, « identique au CPU au bit près partout », impose le double partout (RTX : 33 i/s).
2. **iGPU.** Optimiser (borne plus fine permettant de garder le mapping en float32 quand c'est sûr), et faire choisir par Auto le moteur exact le plus rapide. Sur Raphael, l'hybride (8,5 i/s) égale aujourd'hui le CPU (8,1 i/s).
3. **GPU sans double.** Options :
   - Auto sur CPU, exact ;
   - GPU float32 en choix manuel avec mention de sa précision ;
   - émulation entière à développer.
4. **Référence.** L'ancienne référence CPU (poids Spline16 en double) tranche elle-même mal les pivots exacts. Le CPU rapide (échantillonnage entier exact) deviendrait la référence de l'attestation.

## Lots de mise en œuvre

| Lot | Contenu |
|---|---|
| G1 | Shader hybride : branches entières, mapping en double, EOTF/OETF reformulées, borne, raffinement. |
| G2 | Coefficients du RPU en texture d'entiers ou en tampon uniforme ; détection de `shaderFloat64` (autonome et import FFmpeg) ; repli selon la décision 3. |
| G3 | Tests : stress GPU 3 × 200 graines (≤ 1 code face au CPU exact) dans `fel_gpu_reference_test`. Attestation de schéma 4 : GPU face au CPU exact, tolérance 1 code 16 bits. |
| G4 | iGPU : borne plus fine, mesure, choix Auto du moteur exact le plus rapide. |
| G5 | Attestation finale Linux et Windows (GTX 1070), documentation, réponse. |
