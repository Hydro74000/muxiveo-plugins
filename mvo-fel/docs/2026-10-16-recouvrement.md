# Recouvrement du pipeline de rendu (lot P)

16 octobre 2026. Lot P du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md), décidé à la place des Tensor Cores (T0), reportés.

## Constat

Après [A2'](2026-10-16-a2-virgule-fixe.md), le mode Lent compose une image UHD en 7,8 ms sur la RTX, mais plafonnait à 43–51 i/s.

L'étage de rendu traitait chaque image de bout en bout et en série, sur un seul fil : téléversement des couches, composition, rapatriement de la liste des reprises, conversion, téléchargement, puis attente de la fin du GPU. Le GPU restait inactif pendant les phases CPU de chaque image.

## Principe

Plusieurs fils de rendu tournent en parallèle, chacun avec son propre contexte GPU sur le même périphérique : contexte importé de FFmpeg en Vulkan Video, sinon contexte autonome. Deux images sont ainsi en vol.

- **Contextes supplémentaires.** Ils héritent de la précision, de la variante du mode exact et du résultat de l'auto-test du contexte principal (`fel_gpu_inherit`), sans relancer l'auto-test. Si leur création échoue, le parallélisme est simplement réduit.
- **Ordre des images.** Chaque fil prend les jobs dans l'ordre, sous verrou, avec un ticket. Le résultat n'est écrit que lorsque son ticket est le suivant : l'ordre des images est conservé.
- **Erreur et annulation.** Elles réveillent les fils en attente d'ordre.
- **Rendu CPU.** Il reste sur un seul fil, ses lignes étant déjà parallèles.
- **Réglage.** L'option d'ABI `mvo_fel_set_render_workers` (capacité `render_workers`, lot Q) fixe le nombre d'images en vol, de 1 à 4 ; la valeur par défaut est 2. Le libellé du moteur indique « 2 images en vol ».
- **Statistiques.** Les temps GPU et les compteurs de reprises sont additionnés sur tous les contextes.

## Mesures

Débit du plugin seul, 300 images UHD, sortie 10 bits, décodage Vulkan Video.

| GPU | Mode | 1 fil | 2 fils | 3 fils |
|---|---|---:|---:|---:|
| RTX 4070 Ti SUPER | Lent (virgule fixe) | 50,5 i/s | **57,3 i/s** | 57,5 i/s |
| RTX 4070 Ti SUPER | Standard | 77,7 i/s | 77,8 i/s | |
| RTX 4070 Ti SUPER | Rapide | 78,1 i/s | 76,6 i/s | |
| iGPU Raphael | Rapide | 17,8 i/s | 20,2 i/s | |
| iGPU Raphael | Standard | 7,4 i/s | 7,4 i/s | |

Standard et Rapide plafonnent vers 78 i/s, la limite du reste de la chaîne (décodage et sortie NUT). Le Lent reste limité par le rendu. Un troisième fil n'apporte rien : les contextes se partagent le même GPU.

Chaque contexte supplémentaire alloue ses propres textures (couches, cible RGBA16, sorties, tampons de reprises), soit de l'ordre de 150 Mo en UHD.

## Validation

Attestation de schéma 4 avec deux images en vol (3 scènes, RTX et Raphael, transport direct) : réussie. Les empreintes des images sont identiques, ce qui confirme l'ordre et les codes. Les écarts au CPU certifié sont inchangés : `exact` 0 code, `standard` et `fast` 1 code 16 bits.

Ces mesures portent sur le plugin seul. Le workflow complet est mesuré au [lot M](2026-10-16-mesures-bout-en-bout.md) : P est neutre face à x265, NVEncC et RIFE sur le même GPU, et donne de la marge en régime établi.
