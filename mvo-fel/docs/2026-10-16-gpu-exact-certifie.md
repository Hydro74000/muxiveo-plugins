# Lent GPU certifié (lot A1)

16 octobre 2026. Lot A1 du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md). Le lot suit le [CPU certifié](2026-10-16-cpu-certifie.md).

## Principe

Le mode `exact` (Lent) du GPU produit désormais **la référence certifiée** : les mêmes codes RGB16 que le CPU certifié, au bit près.

1. **Calcul GPU en double.** Le mapping, le NLQ, la matrice Y'CbCr→LMS', l'EOTF et la matrice LMS→RGB sont calculés en double. L'OETF en double disparaît.
2. **Borne d'erreur.** Chaque valeur linéaire R̂ est encadrée par la borne du CPU certifié, avec u = 2⁻⁵² : chaque opération double du GPU peut se tromper d'un ulp entier, ce qui couvre une division moins précise que l'IEEE.
3. **Classement.** Le code candidat vient d'une OETF float32, ajustée par au plus 8 pas. Il est classé entre les seuils PQ arrondis vers le bas, lus dans une texture 256 × 256 de doubles. Le test de certitude est celui du CPU.
4. **Liste compacte.** Une composante non décidée garde son code candidat. Son pixel est ajouté, par compteur atomique, à une liste de 10 mots : position, code BL Y, segments, masque, sommes Spline16 et dénominateurs. La liste est rapatriée après la passe ; le CPU certifié calcule les codes exacts à partir de ces entrées, sans relire les images. Une passe de calcul les écrit ensuite dans l'image cible (`imageStore`). Capacité : 65 536 pixels par image ; au-delà, le rendu échoue explicitement.
5. **Y'CbCr exact.** La conversion reprend l'arithmétique entière du CPU, simulée en double : tous les entiers restent sous 2⁵³, et `fel_quant` corrige le quotient puis compare le reste au seuil de tramage. Aucun `shaderInt64` n'est requis.

Conditions sur le périphérique : double natif, image cible stockable (RGBA16 UNORM) et shaders de calcul. Sinon, le mode exact est refusé et le rendu passe sur le CPU certifié.

## Auto-test au démarrage

Au premier passage en mode exact, le GPU rend deux images synthétiques à la taille de la vidéo :

- couches aléatoires avec matrices Dolby réalistes ;
- MMR à fortes annulations.

Sur deux lignes, il écrit R̂, sa borne et son code dans un tampon de diagnostic, qui est comparé au double-double du CPU (`fel_certified_check`). Pour l'accepter, il faut :

- un écart |R̂ − R| d'au plus la moitié de la borne ;
- des codes certains identiques au CPU certifié.

Sinon, le mode exact est refusé et le rendu passe sur le CPU. Un pilote dont l'arithmétique double sortirait des hypothèses de la borne ne peut donc pas produire de code faux en silence. `MVO_FEL_GPU_SELF_TEST=1` affiche le bilan.

## Validation

| Contrôle | Résultat |
|---|---|
| Auto-test (RTX 4070 Ti SUPER, iGPU Raphael) | réussi : écart ≤ 0,082 × borne, 0 code différent |
| `fel_gpu_precision_test`, 3 × 200 graines (identité, matrices réelles, MMR) | exact : **0 code différent** du CPU certifié sur les deux GPU ; standard ≤ 1 |
| Attestation de schéma 4 (3 scènes réelles, RTX et Raphael, transport direct P010) | réussie : `exact` identique au CPU certifié octet pour octet, RGB et Y'CbCr ; acceptée par `package.py` |
| Reprises sur scènes réelles (RTX, compteur `certified_rescued`) | extrait de référence : 7 284 composantes sur 300 images (24 par image) ; Ryan : 3 par image ; Toy Story 5 : 23 ; Moana : 11 ; **0 indécise** |
| Débit, 300 images UHD, sortie 10 bits | RTX : 27,6 → 33–34,5 i/s (composition 18,8 ms) ; Raphael : 2,1 → 2,5 i/s |

## Limites

- Le gain de débit (+20 à 25 %) reste en deçà du facteur 2 espéré : l'EOTF en double (deux logarithmes, deux exponentielles et plusieurs divisions par composante) domine. L'étage float32 du lot A2 vise ce coût.
- Sur iGPU, le mode exact reste plus lent que le CPU certifié (7,9 i/s). Le choix automatique ne le retient pas.
- Une synchronisation GPU → CPU par image sert à rapatrier la liste. Elle est de petite taille et ne coûte presque rien.
