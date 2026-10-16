# CPU certifié (lot C1)

16 octobre 2026. Lot C1 du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md).

## Référence

Le code RGB16 de référence est l'**arrondi exact** de la formule évaluée en réels. Les entrées de cette formule sont :

- les sommes Spline16 entières, prises comme rationnels exacts (numérateur sur Dx·Dy·1023) ;
- les paramètres du RPU tels que `Mapping` les stocke, en doubles (`av_q2d` des rationnels du RPU) ;
- les bornes du mapping, prises comme les pivots exacts `code/bl_max` ;
- les constantes PQ exactes (m1, m2, c1, c2, c3) ;
- la matrice LMS→RGB BT.2020 en doubles.

Conventions : code k si T(k − ½) ≤ R < T(k + ½), avec T(j) = EOTF(j/65535). Une valeur exactement égale à un seuil prend donc le code supérieur. R ≤ T(½) donne 0, R ≥ T(65534,5) donne 65535.

Le chemin entier (`FastReconstruction`) produit désormais ce code certifié. Le repli `FrameSampler` (géométrie non rationnelle, ou `MVO_FEL_CPU_REFERENCE`) reste un calcul double non certifié.

## Chemin double : borne démontrée

Notations : u = 2⁻⁵³, κ = d ln L / d ln x (conditionnement relatif de l'EOTF). Chaque borne compte une erreur d'au plus u par opération. Elle reste donc valable si le compilateur contracte une multiplication et une addition en FMA, qui n'arrondit qu'une fois.

| Étape | Calcul | Borne |
|---|---|---|
| Signaux BL | σ₀ = b/1023 (1 arrondi) ; σ₁,₂ = s·(1/(Dy·1023))/Dx (3 arrondis) ; bornage [0, 1], 1-lipschitzien | erreur relative ≤ 3,01u |
| Polynôme | Horner (a₂σ + a₁)σ + a₀ | 15u·Σ\|a\| |
| MMR | termes : produits de ≤ 3 signaux (≤ 11,03u) ; puissances ≤ 3 (≤ 35,1u) ; produit par le coefficient (+u) ; somme récursive de 22 termes (γ₂₁) : 57,1u par terme | 80u·(\|c\| + Σ\|m\|), termes dans [0, 1] |
| Bornage aux pivots | pivots double = code/bl_max à u près | + u |
| NLQ | \|d\|·pente + seuil, d à 3,01u près | 5,3u·(\|d\|·\|pente\| + \|seuil\|) |
| Décalage, matrice Y'CbCr→LMS' | v = s − offset (2 arrondis), x = Σ N·v (γ₃) | e_x = 1,01·Σⱼ \|Nᵢⱼ\|·e_v(j), avec e_v ⊇ 2,02u·\|offset\| + 5,03u·\|v\| |
| EOTF calculée | `fast_log2` / `fast_exp2` (reste < 10⁻¹⁷) | erreur relative ≤ (800κ + 60 + 4,2·(\|log₂ L\| + 1))·u |
| Erreur du signal | voir ci-dessous | 1,01·L̂·κ̂·e_x/x̂ |
| Matrice LMS→RGB | M̂ = fl(A·B), \|M̂ − M\| ≤ ΔM = 3,01u·Σ\|A·B\| ; R̂ = Σ M̂·L̂ | e_R = Σₖ (\|M̂\|(1 + 4u) + ΔM)·e_L + (ΔM + 3,01u·\|M̂\|)·L̂ |
| Arrondi des bornes | R̂ ± e_R calculés en double | e_R·(1 + 10⁻¹²) + 4u·\|R̂\| + 10⁻³⁰⁰ |

**EOTF calculée.** Au premier ordre, l'erreur relative vaut (394κ + 30 + 2,1·|log₂ L|)·u. La borne retenue la double. Le détail :

1. p = x^(1/m2) : erreur relative ≤ 4u.
2. p − c₁ : exacte (lemme de Sterbenz, p ∈ [c₁/2, 2c₁]).
3. g = (p − c₁)/(c₂ − c₃p) : erreur relative ≤ 62,8κu + 2u.
4. L = g^(1/m1) : la puissance multiplie l'erreur de g par 6,28 et ajoute la sienne, 17,5u + 2,1·|log₂ L|·u.

Dans le chemin vectorisé, |log₂ L| est majoré par |t| + 1, où t est le log₂ calculé.

**Erreur du signal.** Si κ̂·e_x ≤ 10⁻⁶·x̂, e_x ≤ 10⁻⁶·x̂ et x̂ − e_x > x₀ (zéro de l'EOTF, x₀ = c₁^m2 ≈ 7,31·10⁻⁷) :

- la variation relative de κ/x sur [x̂ − e_x, x̂ + e_x] est inférieure à (1 + 2κ)·e_x/(x̂ − e_x), soit moins de 3·10⁻⁶ ;
- κ̂ est fiable à 2·10⁻⁵ près, car p̂ − c₁ ≥ 2·10⁻¹¹.

Le facteur 1,01 couvre ces écarts. Si x̂ + e_x ≤ x₀, L = 0 exactement. Dans tous les autres cas, la composante est reprise.

## Classement

Le code k est certain si T_bas(k − ½) < R̂ − e_R et R̂ + e_R ≤ T_bas(k + ½). Les seuils T_bas sont ceux de la [table du lot C0](../scripts/generate_pq_thresholds.py), arrondis vers le bas : T_bas < T exact < suivant(T_bas).

Le candidat k est l'arrondi de l'OETF rapide, c'est-à-dire l'ancien code. S'il est faux, ce qui ne peut arriver qu'au voisinage d'un seuil, il échoue au test et part en reprise.

## Reprise double-double

La reprise concerne les pixels dont une composante n'a pas été décidée, rassemblés en liste par ligne. Elle repart des entrées exactes : codes, sommes entières, dénominateurs et segments.

- **Calcul.** Mêmes formules, en double-double. Chaque opération a une erreur relative ≤ 2⁻¹⁰⁴ ; les mêmes bornes sont donc valables avec u = 2⁻¹⁰⁰.
- **`log`.** Deux pas de Newton sur `exp`, depuis `std::log`.
- **`exp`.** Réduction par n·ln 2 (ln 2 en double-double), r/32, Taylor jusqu'à l'ordre 14, puis cinq élévations au carré sous la forme m·(2 + m).
- **Matrice LMS→RGB.** Produits exacts de Dekker.
- **Seuils.** En double-double, grâce à la partie basse de la table : |T − (T_haut + T_bas)| ≤ 2⁻¹⁰⁵·T.
- **Décision.** Le code est certain si l'écart au seuil dépasse e_R + 2⁻⁹⁸·(|R| + T).
- **Composante indécise.** Il faudrait R à environ 10⁻²⁷ près d'un seuil. Le code est alors choisi par la valeur centrale et compté ; aucun cas n'a été observé.
- **Compilation.** L'unité `src/certified.cpp` est compilée avec `-ffp-contract=off`, et `FLT_EVAL_METHOD == 0` est vérifié à la compilation.

## Y'CbCr en entiers exacts

kr = 2627/10000, kb = 593/10000 et kg = 6780/10000 ; les seuils de tramage sont exprimés en 8192es.

- **Luma.** floor((219·S·2^(d−8)·8192 + 16·N·2^(d−8)·8192 + t·N) / (N·8192)), avec S = 2627R + 6780G + 593B et N = 10000·65535.
- **Chroma.** Sommes pondérées sur 64 (poids bilinéaires en huitièmes par axe) ; 224·cb = 7X/(4·9407·65535), avec X = 9407B − 2627R − 6780G. Même forme pour Cr, avec 7373.
- **Bornes.** Tous les numérateurs restent sous 4·10¹⁷ (int64).

Le calcul en double coïncide hors frontière. Sur une frontière exacte, possible seulement sans tramage, l'entier tranche toujours vers le haut.

## Validation

| Contrôle | Résultat |
|---|---|
| EOTF face à `decimal` (4 515 signaux, du zéro au pôle) | erreur ≤ 0,098 × borne (double), 0,0022 × borne (double-double) ; critère 0,5 |
| Stress natif : 400 couches aléatoires, polynôme et MMR | chemin certifié = tout en double-double ; 0 indécise ; \|R̂ − R\| ≤ 0,097 × e_R |
| Scènes réelles : Ryan ×2, Toy Story 5, Moana ×2 ; 3 images UHD chacune (75 M composantes par scène) | reprises : 4 à 39 par scène (5·10⁻⁸ à 5·10⁻⁷) ; 0 indécise ; **0 code différent de l'arrondi double historique** |
| Contrôle étendu (1 ligne sur 64 en double-double) | identique ; \|R̂ − R\| ≤ 0,12 × e_R |
| Y'CbCr entier face à la formule double (12 configurations) | identique hors voisinage 10⁻⁹ d'une frontière |
| Attestation de schéma 4 (3 scènes, RTX 4070 Ti SUPER et iGPU Raphael, transport direct) | réussie : `exact` identique au CPU certifié octet pour octet, `standard` ≤ 1 code 16 bits |
| Débit CPU (7800X3D, 300 images, sortie 10 bits) | 8,04–8,11 → 7,89 i/s (≈ −2,5 %) ; rendu 124 → 126 ms/image |

Outils de mesure locaux : `certified_real.cpp` et `convert_bench.cpp`. Ils ne sont pas livrés.

## Suite

- Le mode GPU `exact` calcule encore le Y'CbCr en double. Il peut donc différer du CPU sur une frontière exacte, uniquement sans tramage. Le lot A1 passe le GPU en entiers et en classement par seuils.
- Les compteurs (`CertifiedStats`) seront exposés dans l'attestation de schéma 5 (lot V).
