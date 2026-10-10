# Reconstruction FEL GPU : précision et prototype local

Investigation du 10 octobre 2026 sur Ryzen 7 7800X3D, son iGPU AMD et une RTX
4070 Ti SUPER, sous Linux x86_64. **La divergence PQ du backend GPU est corrigée
sur le corpus local** sans relever le seuil de deux codes PQ12. Un paquet de
prototype autonome a été construit et installé dans un dossier de test. Cela
ne constitue pas encore une release multiplateforme ni une validation TV DV.

## Cause et correction PQ

Le premier prototype interpolait l'EL en float32 autour de l'offset NLQ. Une
erreur minuscule près du zéro pouvait changer le signe du résiduel. Le terme
discontinu `sign(d)*threshold` amplifiait cette erreur : jusqu'à **2,749 codes
PQ12**, donc échec du critère strict. Il ne s'agissait pas d'un manque de bits
dans l'encodage final ni d'un traitement Display Management.

Le backend `src/gpu_backend.c` procède désormais ainsi :

1. Retrouver les codes entiers EL depuis les textures UNORM et soustraire
   l'offset **avant** interpolation.
2. Calculer en double les coefficients et indices Spline16 des axes sur CPU ;
   les transmettre sous forme de couples float32 haut/bas au GPU.
3. Effectuer l'interpolation avec FMA, additions compensées et règles `precise`.
   Pas de dépendance à `shaderFloat64`, trop restrictive pour certains GPU.
4. Appliquer les shaders officiels libplacebo NLQ/mapping/couleurs sur le
   résiduel centré, avec offset NLQ nul équivalent. La garde du zéro est celle
   du CPU, environ 7,1×10⁻¹⁵ normalisé, et non un epsilon float32 élargi.

Les textures, tables et programmes sont réutilisés. Il n'y a ni suppression
perceptuelle de résiduel, ni détection de faux FEL, ni tone mapping, ni trim
d'écran, ni Display Management dans la reconstruction.

## Validation numérique réelle

Trois scènes du média FEL fourni, images 0, 8 et 16 de chaque scène, tous pixels
et canaux RGB. Comparaisons après normalisation en PQ12, limite maximale **2**.

| GPU | Maximum face au CPU | Maximum face à l'oracle indépendant `kernel64` |
|---|---:|---:|
| AMD intégré RADV Raphael/Mendocino | 0,312428 | 0,309435 |
| NVIDIA RTX 4070 Ti SUPER | 1,374690 | 1,359730 |

Les 18 comparaisons d'images passent. `kernel64` utilise les noyaux Spline16
officiels libplacebo évalués en double et ses shaders NLQ/mapping/couleurs,
indépendamment du sampler de production. Les tests synthétiques GPU couvrent
polynômes/MMR, seuils NLQ positifs/négatifs, EL pleine/demi-résolution et les
six positions chromatiques. Le prototype accepte les plans HEVC 10 bits dans
des conteneurs 16 bits ; les formats hors de ce contrat sont refusés.

L'ancien oracle brut `gpu32` reste distinct et conserve son résultat historique
de 2,279 codes face au CPU près du zéro NLQ. La correction porte sur le backend
de production, pas sur une modification de cet oracle ni de la tolérance.
Voir le [rapport réel](2026-10-10-real-fel-validation.md).

Attestation initiale : `real-moana/gpu-deliverable-validation.json`, sous
`/var/mnt/Drive2/muxiveo-fel-dev/`. Bibliothèque validée :

```text
80a437dac548f6b2ee58131d142a2c3448b10282cbe4acf8ac4f119f1bb448f6
```

La revue ajoute une attestation de schéma 2 :
`real-moana/gpu-reviewed-validation.json`. En plus du renderer de référence,
elle exécute cette bibliothèque et compare les empreintes des pixels de son
NUT aux images du renderer, sur les deux cartes et les trois scènes. Cette
vérification couvre aussi l'ordre GBR, la sérialisation et l'ordre des images
échantillonnées. Les anciennes attestations seules ne permettent plus de
construire un nouveau paquet GPU.

## Coût mesuré et circulation des images

La reconstruction CPU initiale coûtait 3,709 s sur une image UHD avec un thread
dans le microbenchmark : 2,290 s de sampling/mapping/NLQ et 1,292 s de couleurs
et transferts PQ. Jusqu'à douze `std::pow` par pixel et des taps Spline16
scalaires expliquent ce coût. Un meilleur pool CPU aiderait, mais ne suffit pas
à rendre ce chemin compétitif sur cette machine.

Après correction PQ, le rendu GPU répété atteint environ **77–82 i/s sur la
RTX**, contre **14–15 i/s sur l'iGPU AMD**. Téléversements et rapatriements
compris ; décodage, pipes et encodage exclus. Ce microbenchmark de la même image
ne décrit pas le débit d'un film.

Le producteur décode BL et EL sur CPU, reconstruit sur le GPU choisi et produit
un **NUT horodaté RGB PQ GBRP16 en pipe**. FFmpeg réalise les filtres,
conversions et le dithering ; RIFE éventuel intervient ensuite, puis l'encodeur
final. Le FFmpeg privé du plugin n'encode pas une vidéo compressée intermédiaire.
Il n'y a qu'un encodage compressé final par passe.

Sur le même extrait de 73 images UHD, DV conservé :

| Parcours | BL seul, médiane | FEL, médiane | Hausse |
|---|---:|---:|---:|
| x265 medium CRF 18, FEL RTX, 3 passages | 13,80 s | 18,03 s | +30,64 % |
| NVEncC default CQP 18:20:22, FEL RTX, 3 passages | 5,92 s | 12,79 s | +116,19 % |
| NVEncC default, FEL iGPU, 3 passages | 5,86 s | 16,31 s | +178,58 % |

Les séries ont chacune leur référence BL remesurée. La première et la troisième
utilisent le build GPU de développement à libplacebo partagé ; la deuxième
utilise la bibliothèque statique exacte du paquet. Tous passent les contrôles
de sortie et n'ont pas déclenché de repli BL. Le CPU initial donnait 83,50 s
(x265) et 80,19 s (NVEncC), dans une série antérieure.

**Une carte moins occupée peut être trop lente** : déporter le FEL vers cet
iGPU dégrade NVEncC seul. Les placements avec RIFE doivent être mesurés
séparément ; la [matrice complète](2026-10-10-gpu-pipeline-matrix.md) compare les
deux encodeurs et les charges des moteurs GPU. Ces extraits courts incluent
préparation, démarrage, injection et muxage ; pas d'extrapolation au film entier.

## Choix matériel et portabilité

L'interface propose **Auto / chaque GPU énuméré / CPU**. Un UUID stable est
conservé par piste et profil. Une carte disparue reste visible comme indisponible
dans le profil et provoque un choix Auto averti pour l'exécution. Le mode Copy
désactive le contrôle en conservant son état ; la case FEL reste décochée par
défaut sur tous les profils.

Auto tient compte des périphériques aval identifiés, des réservations FEL en
cours et de la télémétrie facultative. La priorité est **GPU dédié disponible,
puis iGPU, puis CPU** ; la charge départage les GPU d'une même classe, sans
déplacer RIFE ni modifier les encodeurs. CUDA/NVENC et Vulkan/RIFE peuvent avoir
des indices différents : sur cette machine, la RTX est CUDA 0 et Vulkan 1.
Une identité ambiguë reste conservatrice ; aucune charge inconnue n'est
présentée comme une carte libre. Les règles et leurs limites sont documentées
avec la matrice. Les choix explicites restent possibles pour comparer ou
contourner une décision Auto inadaptée.

Vulkan évite une obligation NVIDIA. AMD fonctionne ici ; Intel, plusieurs
cartes dédiées, Windows et macOS/MoltenVK restent à tester physiquement.
L'absence de `nvidia-smi` ou de sysfs ne bloque pas le moteur. Le chemin CPU est
portable et le chargement du plugin ne dépend pas du chargeur Vulkan.

## Paquet local et isolation

Le build GPU épingle libplacebo et glslang 16.0.0, liés statiquement avec FFmpeg
minimal et libdovi. Les symboles tiers sont masqués ; seuls les symboles de
l'ABI `mvo_fel_*` sont exportés. Aucun FFmpeg ajouté au PATH ni remplacement du
binaire configuré dans Muxiveo. Le chargeur graphique du système est ouvert
uniquement au moment de l'utilisation GPU.

`scripts/build.py --gpu` prépare les dépendances privées. Le packaging exige
une attestation `scripts/validate_gpu.py` correspondant aux SHA-256 exacts du
moteur, des sources et des dépendances, trois scènes et tous les GPU énumérés
sous le seuil. Le prototype n'est empaquetable que pour **Linux x86_64** pour
l'instant. L'archive contient notices et sources nécessaires au reliage.

Une archive de 71 Mo a été construite, puis installée dans un dossier de test
par le gestionnaire Extensions de Muxiveo. Chargement ABI, manifestes, sommes
et liste dynamique des deux GPU vérifiés. Aucune publication ni modification
de l'installation utilisateur n'a été effectuée.

Vérifications locales : quatre tests CTest natifs réussis ; six workflows réels
x265 DV/HDR10/SDR, FFmpeg/NVENC, NVEncC et BL seul réussis sur le build statique ;
307 tests Python ciblés, Ruff et Mypy sur l'application et les scripts FEL au vert.
La matrice RIFE ajoute des mesures séparées, sans remplacer la validation PQ.

## Goulot identifié, puis deux transports optimisés

Les deux voies décrites ci-dessous sont maintenant implémentées :
[pipe NUT allégé et raccordement GPU direct](2026-10-10-fel-transport-optimizations.md).
Les coûts historiques suivants motivent ces travaux ; le rapport lié donne
les mesures actualisées et les limites du prototype direct.

Le surcoût n'impose aucun facteur ×2. Dans la matrice complète, NVEncC seul
passe de 5,84 à 12,82 s (+119,5 %), tandis que NVEncC avec RIFE passe de 14,53 à
15,63 s (+7,6 %). Il s'agit de temps de workflow sur 73 images source, incluant
les coûts fixes. Le décodage supplémentaire et la reconstruction sont
nécessaires, mais leurs temps ne s'additionnent pas nécessairement au temps
d'encodage lorsque les étapes se chevauchent. Un profilage interne sur un
extrait plus long est nécessaire pour attribuer le surcoût.

Avant l'optimisation, chaque image UHD rapatriait 66,4 Mo RGBA, puis transportait 49,8 Mo
GBRP16 vers FFmpeg avant conversion YUV et nouveau téléversement à NVEncC.
La réduction de précision doit rester après les traitements qui en ont besoin.
Les prochaines optimisations utiles sont une conversion/dithering GPU validée
vers P010 lorsque possible et quelques images en vol avec transferts
asynchrones bornés. Le NUT standard ne transporte pas un objet GPU partagé
entre processus : ce prototype ne revendique pas un parcours sans copie RAM.

Un raccordement plus direct ne demande pas de réécrire la compression :
[l'API NVENC accepte des ressources GPU externes](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html#input-buffers-allocated-externally)
et [l'API x265 reçoit des images brutes](https://x265.readthedocs.io/en/master/api.html#pictures).
Un filtre FEL peut donc préparer leurs entrées. Il faut cependant développer
le transport et la synchronisation : les
[shaders personnalisés NVEncC](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md)
ne constituent pas à eux seuls un chargeur EL/RPU ni un plugin FEL complet.
Ces pistes concernent le FEL et son raccordement ; les réglages et traitements
RIFE/encodage existants restent la référence pour les comparaisons.

La lecture P8.1 sur matériel Dolby Vision, un corpus plus large et les builds
GPU des autres OS restent des critères de release générale.
