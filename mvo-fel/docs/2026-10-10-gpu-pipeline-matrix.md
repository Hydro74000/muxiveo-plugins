# Placement du FELMerge : x265, NVEncC et RIFE

## Question et périmètre

Choisir où exécuter **le FELMerge uniquement** : CPU, iGPU ou GPU dédié. RIFE,
ses réglages, les filtres et les encodeurs conservent leur fonctionnement.
Il ne s'agit pas d'optimiser ou de déplacer ces autres étapes.

Le benchmark compare séparément x265 medium CRF 18 et NVEncC default
CQP 18:20:22, HEVC 10 bits avec RPU P8.1 conservé. Les débits et la qualité des
deux encodeurs ne sont pas supposés équivalents : chaque surcoût FEL se calcule
sur **le même encodeur avec le même réglage RIFE**, en référence au BL seul.

## Protocole local

- Linux x86_64, Ryzen 7 7800X3D (8 cœurs/16 threads), budget Muxiveo 12 threads.
- iGPU AMD Raphael/Mendocino, RTX 4070 Ti SUPER 16 Go.
- FFmpeg 8.1.2, NVEncC 9.36 r4153, MVO-RIFE 1.7.0/ncnn 20260526,
  pilote NVIDIA 615.71.09 ; dépendances privées du FEL épinglées séparément.
- Extrait FEL UHD réel : 73 images, 3840×2160, 24000/1001 i/s, environ 3,04 s.
- Sortie x265 : 3840×2160. Sortie NVEncC Dolby Vision : 3840×1600, recadrage
  L5 automatique déjà imposé par le workflow NVEncC existant. Cette géométrie
  est identique entre BL/FEL/placements pour cet encodeur. Les charges RIFE
  x265/NVEncC ne doivent donc pas être comparées comme si seul l'encodeur changeait.
- RIFE désactivé ou ×2 Balanced, mode Normal, TTA 1, sur sa RTX habituelle.
  Backend Vulkan/hybride de l'extension installée, sans extension TensorRT.
- Placements FEL explicites pour isoler leur effet ; BL et GPU répétés trois
  fois, ordre inversé un passage sur deux. CPU : contrôle du coût de secours.
- Durée du workflow complet : préparation RPU, reconstruction/encodage,
  injection et assemblage. Découpe et vérification externe hors chronométrage.
- Sorties vérifiées : compte exact (73 ou 146), géométrie identique dans chaque
  groupe, PQ 10 bits, P8.1/compatibilité HDR10 1, BL/RPU présents et EL absente,
  reconstruction réussie sans repli BL pour les cas FEL.
- Charge système échantillonnée toutes les 0,5 s : GPU calcul, NVENC, NVDEC,
  mémoire, puissance et température si disponibles. Ces relevés portent sur
  toute la machine, incluent les phases de préparation/muxage et l'affichage.
  Ils ne prouvent pas l'attribution exacte de chaque cycle à un processus.

Le premier extrait avait hérité des tags de statistiques du film entier,
causant une fausse détection VFR par MediaInfo. La découpe de benchmark retire
ces tags sans modifier les paquets vidéo/RPU. La garde VFR du workflow reste
inchangée. Les premiers essais rejetés sont exclus des comparaisons.

Bibliothèque GPU statique validée numériquement :
`80a437dac548f6b2ee58131d142a2c3448b10282cbe4acf8ac4f119f1bb448f6`.
Résultats bruts, journaux et télémétrie :
`/var/mnt/Drive2/muxiveo-fel-dev/real-moana/fel-matrix-tj5d6cw8/results.json`.
Les médias restent privés et hors dépôt.

## Résultats

Médianes de trois passages ; durée du workflow complet en secondes :

| Encodeur | RIFE habituel | BL seul | FEL RTX | FEL iGPU | Hausse FEL RTX sur BL |
|---|---|---:|---:|---:|---:|
| x265 medium | Non | 13,82 | 17,66 | 19,03 | +27,8 % |
| x265 medium | RTX, ×2 Balanced | 21,03 | 22,98 | 22,88 | +9,3 % |
| NVEncC default | Non | 5,84 | 12,82 | 16,19 | +119,5 % |
| NVEncC default | RTX, ×2 Balanced | 14,53 | 15,63 | 16,98 | +7,6 % |

Le dédié gagne dans trois configurations. Avec x265/RIFE, l'iGPU est **0,11 s
(0,5 %) plus rapide** sur cette courte série : quasi-égalité pratique, pas une
preuve que le dédié gagne absolument partout. La référence BL de ce groupe
varie de 20,82 à 21,61 s. Aucun déport ne démontre un gain suffisant pour
compliquer Auto sur cette machine ; le choix manuel iGPU permet de le répéter
sur un extrait plus long ou de libérer de la mémoire sur la RTX.

Le CPU reste un contrôle du coût de secours, un passage par cas : x265 83,66 s
sans RIFE et 85,27 s avec RIFE ; NVEncC 80,16 s sans RIFE et 80,32 s avec RIFE.
Ces passages ne sont pas mélangés aux médianes GPU. Les sorties ont aussi été
auditées en ordre de présentation : pire écart PTS **0,5 ms**, y compris après
interpolation. Les quatre vérifications supplémentaires Auto choisissent bien
la RTX et réussissent sans repli. Au total : 45 sorties contrôlées, comprenant
les 36 passages BL/GPU, quatre contrôles CPU, quatre contrôles Auto et le
diagnostic RIFE/iGPU isolé.

L'essai diagnostic RIFE sur iGPU a pris 347,83 s sans FEL, contre environ
20,8 s sur RTX avec x265. Le développement des placements RIFE a été arrêté
pour conserver le périmètre demandé : le benchmark principal ne déplace que
le FEL. Cet essai isolé est exclu des quatre comparaisons ci-dessus.

### Occupation de la RTX

Activité GPU moyenne sur le workflow (médiane des moyennes des trois passages),
mémoire maximale parmi les passages. Ces pourcentages décrivent l'activité
rapportée par le pilote, pas une fraction mesurée des FLOPS ou de la bande
passante disponibles.

| Encodeur / RIFE | FEL | Activité GPU | NVENC | NVDEC | Pic mémoire RTX |
|---|---|---:|---:|---:|---:|
| x265 / non | BL seul | 0,0 % | 0 % | 0 % | 68 Mio |
| x265 / non | RTX | 9,2 % | 0 % | 0 % | 604 Mio |
| x265 / oui | BL seul | 36,5 % | 0 % | 0 % | 2366 Mio |
| x265 / oui | RTX | 36,6 % | 0 % | 0 % | 2893 Mio |
| x265 / oui | iGPU | 33,9 % | 0 % | 0 % | 2366 Mio |
| NVEncC / non | BL seul | 0,6 % | 3,5 % | 7,5 % | 1783 Mio |
| NVEncC / non | RTX | 2,8 % | 0,7 % | 0 % | 1665 Mio |
| NVEncC / oui | BL seul | 43,1 % | 3,0 % | 0 % | 2878 Mio |
| NVEncC / oui | RTX | 40,1 % | 3,5 % | 0 % | 3449 Mio |
| NVEncC / oui | iGPU | 37,2 % | 3,1 % | 0 % | 2866 Mio |

La mémoire supplémentaire du FEL RTX avec RIFE est d'environ 0,5–0,6 Gio ;
le total reste sous 3,4 Gio sur cette carte de 16 Gio. Les pics d'activité GPU
avec NVEncC/RIFE passent de 92 % (BL) à 89 % (FEL RTX). Des contentions brèves
restent possibles, mais **le déport iGPU n'améliore pas ce workflow**.

NVEncC sans FEL ni RIFE utilise ici NVDEC. Le producteur FEL décode BL et EL
sur CPU ; NVDEC est donc nul dans ses cas. Avec RIFE, la préparation existante
passe aussi par les images CPU. Le gros pourcentage +119,5 % sans RIFE n'est
donc pas une preuve de saturation GPU : il inclut double décodage de source,
reconstruction, transferts et transport RGB. Le FEL GPU ne réalise pas un
deuxième encodage compressé. Aucun changement des décodeurs ou encodeurs
existants n'a été apporté pour améliorer ces chiffres.

## Décision Auto et choix manuels

Auto applique **GPU dédié disponible → iGPU disponible → CPU**. La mémoire
annoncée et la mémoire libre connue doivent couvrir le budget prudent FEL.
Les allocations natives vérifient ensuite le besoin réel.

Entre plusieurs cartes dédiées éligibles, Auto tient compte de la charge de
calcul connue, des réservations FEL de l'application et des périphériques
aval identifiés. Une carte peu chargée et sans autre étape en cours est
préférée à une carte de même classe déjà occupée. Un moteur NVENC chargé
n'est pas confondu avec le calcul Vulkan. Sans télémétrie, la charge reste
inconnue ; le choix conserve la priorité de classe et les identités connues.

Les GPU sont identifiés par UUID. L'index 0 de NVEncC/CUDA n'est pas supposé
être le GPU 0 de Vulkan ou de RIFE. Les homonymes ou identités non résolues
restent conservateurs. Le choix est figé entre les deux passes ; les
réservations sont libérées à la fin du producteur ou en cas d'erreur.

Les choix manuels **Auto / chaque GPU détecté / CPU** restent disponibles par
piste et profil. Aucun réglage de RIFE ou d'encodeur n'est modifié par ce choix.
La case FEL reste décochée par défaut, P5/P7 compris.

## Autres machines et OS

Cette priorité constitue une politique initiale, pas une preuve que chaque
GPU dédié existant bat chaque iGPU dans tous les logiciels et toutes les
charges. Il faut répéter la matrice sur le matériel cible pour déroger à Auto.

| Configuration | Politique initiale | Validation physique restante |
|---|---|---|
| CPU seul / pilote Vulkan absent | CPU | Autres familles CPU, coût et stabilité |
| iGPU seul Intel/AMD | iGPU si compatible et mémoire suffisante | Intel et autres générations AMD |
| iGPU + GPU dédié | Dédié ; déport FEL manuel pour comparer | Autres couples, GPU mobiles, limites de puissance |
| Plusieurs GPU dédiés | Carte dédiée la moins contrainte identifiée | Cartes identiques, asymétriques, plusieurs jobs |
| GPU à faible mémoire disponible | Autre GPU éligible puis CPU | Pression mémoire, dimensions et autres modèles RIFE |
| Windows x64 | Même règle ; nvidia-smi facultatif | Build GPU, pilotes NVIDIA/AMD/Intel, annulation et pipes |
| macOS arm64 / mémoire unifiée | GPU compatible puis CPU | MoltenVK, budgets partagés, VideoToolbox, paquet GPU |

Un iGPU partage généralement la mémoire et le budget énergétique du CPU : une
carte libre ne garantit pas un meilleur workflow x265. Sur portable, la chauffe
et les limites de puissance exigent des extraits plus longs. Les moyennes
système, les charges inconnues et les tests de politique simulés ne remplacent
pas ces mesures physiques.

Le prototype GPU empaqueté est **Linux x86_64 uniquement** pour l'instant.
Le chemin CPU et les scripts gardent un fonctionnement sans NVIDIA, sans
sysfs et sans outil de télémétrie obligatoire. Le chargement du module CPU
n'exige pas le chargeur Vulkan.

## Reproduire

Depuis Muxiveo, énumérer séparément les GPU FEL et ceux de RIFE, puis garder
le périphérique RIFE habituel. Exemple sur la machine locale (RIFE index 1) :

```sh
QT_QPA_PLATFORM=offscreen python3 -m scripts.benchmark_fel_matrix /chemin/film.mkv \
  --library /chemin/libmvo_fel.so --work-dir /chemin/benchmarks \
  --duration 3 --runs 3 --threads 12 \
  --fel-devices off cpu all --rife-gpus off 1 \
  --codecs libx265 nvencc_hevc
```

`--rife-gpus off auto` conserve le choix automatique RIFE. `--codecs libx265`
permet la mesure sans NVIDIA. `--fel-devices off auto` vérifie le choix Auto,
et les UUID permettent de forcer chaque placement. `--resume results.json
--runs 3` complète les passages manquants avec le moteur et l'extrait exacts.

Pour une caractérisation stable, répéter ensuite sur 30–60 s et plusieurs
scènes, après chauffe, avec même qualité RIFE et même parallélisme. Les
résultats de 3 s incluent beaucoup de coûts fixes ; ils ne donnent pas un
pourcentage universel ni une durée prédite pour un film complet.
