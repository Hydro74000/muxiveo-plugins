# Profil de la composition du Lent (lot L)

16 octobre 2026. Lot L du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md). Avec un encodeur rapide (NVEncC par défaut, ≈ 116 i/s), le Lent RTX limite le débit : 81 à 95 i/s ([mesures de bout en bout](2026-10-16-mesures-bout-en-bout.md)).

## Méthode

On utilise une copie instrumentée du shader en virgule fixe, non livrée. Une constante de compilation arrête `fx_render` après une section donnée. Les valeurs déjà calculées sont consommées dans la sortie, pour que le compilateur ne les élimine pas. L'auto-test est contourné, car la sortie tronquée est fausse.

La mesure est le temps GPU de composition (`gpu_compose_ms`), avec une image en vol. Les extraits sont ceux de M : Ryan (MMR) et Toy Story 5 (polynômes), 480 images UHD, sortie 10 bits.

## Résultats

Coût de chaque section, en ms par image. C'est la différence entre deux coupes successives.

| Section | Ryan (MMR) | Toy Story 5 (polynômes) |
|---|---:|---:|
| Lecture des entrées (sommes Spline16 BL et EL) | 0,8 | 0,8 |
| Signaux et monômes (3 divisions) | ≈ 0 | ≈ 0 |
| Mapping (polynômes ou MMR) | 1,5 | 0,4 |
| NLQ (2 divisions par composante) | 1,2 | 1,2 |
| Matrice Y'CbCr → LMS' | 0,3 | 0,3 |
| **EOTF × 3 (ln, expm1, division, ln, exp en multi-mots)** | **4,1** | **4,0** |
| Matrice LMS → RGB, bornes et classement | 1,6 | 1,4 |
| **Total de la composition** | **9,3** | **7,9** |
| Passe de sortie Y'CbCr en entiers exacts (séparée) | 1,9 | 1,8 |

La sortie de la NLQ comprend aussi la lecture de la couche EL : la coupe précédente ne la consommant pas, le compilateur la supprime.

## Leviers

1. **EOTF à deux niveaux (≈ 4 ms).** Le classement n'a besoin de la précision 2⁻⁵⁰ qu'au voisinage d'un seuil. Une cellule de code 16 bits mesure environ 1,5·10⁻⁴ en relatif sur L.
   - Une EOTF à environ 2⁻³² d'erreur relative, sur mots de 32 bits avec des séries courtes, laisserait moins d'une composante sur 10⁵ indécise.
   - Les composantes indécises reprendraient l'EOTF multi-mots actuelle, dans le même shader.
   - La borne du premier niveau se démontre comme celle d'A2'.
   - Gain attendu : de l'ordre de 3 ms par image.
2. **Sortie fusionnée (≈ 1 à 1,5 ms).** La conversion Y'CbCr exacte écrit puis relit une cible RGBA16 UHD (66 Mo). Elle pourrait être faite par la composition elle-même, ou partager sa lecture.
3. **Tensor Cores (T, ≤ 1,3 ms et seulement sur MMR).** Nul sur un contenu à polynômes. Il ne vient qu'après les deux leviers précédents, si le MMR redevient le poste dominant.

Objectif de L : Lent RTX ≥ 116 i/s, pour que NVEncC par défaut n'attende plus le FEL. Il faut environ 30 % de temps GPU en moins par image, ce que le levier 1 apporterait à lui seul.
