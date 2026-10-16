# EOTF à deux niveaux (lot L, premier levier)

16 octobre 2026. Lot L du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md), premier levier du [profil de la composition](2026-10-16-profil-lent.md).

L'EOTF en virgule fixe multi-mots d'[A2'](2026-10-16-a2-virgule-fixe.md) coûtait 4 ms sur 8 à 9 ms de composition UHD. Sa précision (2⁻⁵⁰) n'est nécessaire qu'au voisinage d'un seuil PQ : une cellule de code 16 bits mesure environ 1,5·10⁻⁴ en relatif sur L.

## Principe

- **Premier niveau (`fixed::eotf_fast`, `fx_eotf_fast`).** Il suit le même découpage qu'A2' : ln par table de 64 entrées, `num = c₁·expm1(ln(x/x₀)/m2)`, puis exp par table `2^(j/64)`. Trois différences :
  - les produits se font sur 32 × 32 bits, les additions sur 64 bits ;
  - les séries sont plus courtes ;
  - il n'y a pas de division : `ln g = ln num − ln den`.

  Une borne démontrée accompagne chaque L. La composante est classée avec cette borne, exactement comme au second niveau.
- **Second niveau.** C'est l'EOTF multi-mots d'A2', inchangée. Elle traite les seuls pixels dont une composante reste indécise : environ 10⁻⁴ des pixels sur contenu réel.
- **Repli CPU.** Le CPU certifié ne reprend que ce que le second niveau ne décide pas. C'est rare : 0 sur les extraits réels mesurés.

Les codes sont identiques au CPU certifié dans tous les cas. Un code n'est retenu que s'il est certain, quel que soit le niveau qui l'a décidé.

## Borne du premier niveau

| Étape | Calcul | Erreur |
|---|---|---|
| ln | r = d/c_j par un produit 32 × 32 (troncatures de d, de 1/c_j et du produit : ≤ 2,5·2⁻³⁸) ; série de degré 4 en S1.30 (calcul ≤ 2⁻³⁷, troncature r⁵/5 ≤ 2⁻³⁷·³) ; e·ln 2 et ln c_j sur 64 bits | absolue < 6,2·2⁻³⁸, retenue **E_LN = 2⁻³⁵** |
| expm1 | expm1(r) = r + r²·b. r²·b est calculé sur des mantisses normalisées de 32 bits (erreur relative ≤ 2⁻²⁸ sur r²·b ≤ r/128), ce qui garde la précision relative même pour r petit ; b par une série de degré 5 (troncature ≤ 2⁻³⁹·⁵) | relative ≤ 2⁻³⁵, retenue **E_EM = 2⁻³⁴** |
| exp | K estimé en flottant puis corrigé exactement (0 ≤ r < ln 2/64) ; série de degré 4 en Q1.31 (≤ 2⁻³⁶·²) ; produit par 2^(j/64) sur 64 × 32 bits | relative ≤ 2⁻³⁶, retenue **E_EXP = 2⁻³⁵** |

Ces erreurs se composent en quatre temps :

1. **Erreur relative de num.** `r_num ≤ E_LN·(1/dl + 0,01) + E_EM + 2⁻⁵⁵/dl`, avec dl = ln(x/x₀).
2. **Dénominateur.** Il en hérite à travers f = 1 + c₃·num/den.
3. **ln L.** Avec 1/m1 = 8192/1305 : `|Δ ln L| ≤ B = (1/m1)·(1,01·f·r_num + 2·E_LN)`.
4. **Borne finale.** L'erreur relative de L est au plus `1,25·(B·(1 + B) + 1,01·E_EXP)`. Le facteur (1 + B) majore e^B − 1, et 1,25 est la marge.

**Domaine.** La linéarisation suppose r_num ≤ 2⁻¹², soit dl ≳ 10⁻⁷. Plus près du zéro de l'EOTF, le premier niveau renonce et le second décide. Sans cette garde, le test a mesuré une erreur relative de plusieurs centaines pour L ≈ 10⁻⁸⁰.

**Encadrement.** Celui de R en Q0.96 passe en unités 2⁻⁶⁴ quand il ne tient plus sur 64 bits en unités 2⁻⁹⁶, soit au-delà de 2⁻³² en valeur. La borne du premier niveau peut atteindre environ 10⁻⁸ près du pôle. L'ancienne garde (`e96 < 1,8·10¹⁹`) renvoyait la moitié des composantes au second niveau.

## Deux passes

Avec les deux EOTF dans le même shader, la composition coûtait 8,4 ms au lieu de 6,5 ms, même quand le second niveau n'était jamais exécuté. Le shader réserve les registres de son chemin le plus lourd, et l'occupation du GPU baisse pour tous les pixels.

Le rendu se fait donc en deux passes :

1. **Composition.** Seule l'EOTF rapide y figure. Les composantes indécises sont inscrites dans la liste de reprises habituelle, avec leurs entrées exactes et la déduplication des zones uniformes.
2. **Second niveau** (`resolve_fixed`). Une invocation par entrée de la liste exécute `fx_render` avec la seule EOTF multi-mots. Elle écrit les codes dans le tampon de correctifs et compte les entrées encore indécises.
   - S'il n'en reste aucune, la passe de correctifs applique ce tampon directement, sans aller-retour avec l'hôte.
   - Sinon, seules les entrées restantes passent par le CPU certifié.

`certified_rescued` compte désormais les composantes reprises par le CPU, et plus celles du second niveau GPU.

## Validation

| Contrôle | Résultat |
|---|---|
| EOTF rapide face au double-double (`fel_fixed_test`) : 1,08 M points, échelles d'entrée 2⁻⁶⁴ à 2⁻¹¹², voisinage de x₀, pôle | erreur ≤ **0,26 × borne** |
| Pipeline C++ à deux niveaux face au CPU certifié (stress : identité, matrices réelles, MMR) | 0 code différent ; écart ≤ 0,13 × borne ; second niveau sur 0,2 % des pixels (cas hors cadre du stress) |
| GPU face au modèle C++ (`fixed_debug`, 4 configurations × 6 144 composantes) | identique au bit près (premier niveau) |
| `fel_gpu_precision_fixed` (virgule fixe imposée) : identité, réel, MMR, reprise CPU forcée, zones uniformes | 0 code différent |
| Idem, mode `level1` : toutes les composantes listées puis reprises par le second niveau GPU (1 228 800) | 0 code différent ; 4 332 reprises CPU (0,35 %, hors cadre) |
| Auto-test RTX | virgule fixe 15,4 ms (21,9 avant), double 44,3 ms : virgule fixe retenue ; écart ≤ 0,07 × borne |

Le test `fel_gpu_precision` échoue désormais si la variante imposée par `MVO_FEL_GPU_FIXED` est refusée par l'auto-test. Auparavant, le rendu se repliait en silence sur le double. Ce test a révélé un mot réservé du GLSL (`precise`) qui empêchait la compilation de la variante.

## Mesures

FEL seul, sortie Y'CbCr 10 bits vers `/dev/null`, environ 480 images UHD, RTX 4070 Ti SUPER.

| Extrait | Composition avant | Composition après | Lent, 2 images en vol, avant | Lent, 2 images en vol, après |
|---|---:|---:|---:|---:|
| Ryan (MMR) | 9,3 ms | **6,6 ms** | 81 i/s | **106 i/s** (+30 %) |
| Toy Story 5 (polynômes) | 7,9 ms | **5,2 ms** | 95 i/s | **127 i/s** (+34 %) |

**Profil après le lot.** L'EOTF rapide coûte 1,5 ms, contre 4,1 ms auparavant. Le second niveau prend un temps négligeable : environ 880 composantes par image sur Ryan, 380 sur Toy Story 5.

**Raphael.** La virgule fixe y reste plus lente que le double (1,9 contre 2,3 i/s). L'auto-test garde donc le double.

**Objectif de L.** Toy Story 5 dépasse les 116 i/s de NVEncC par défaut ; Ryan s'en approche. Le levier suivant est la sortie fusionnée (1,9 ms).
