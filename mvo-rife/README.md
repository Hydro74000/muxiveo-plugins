# mvo-rife — interpolation d'images (moteur muxiveo-rife)

Interpolation d'images **RIFE v4** sur GPU (Vulkan via [ncnn](https://github.com/Tencent/ncnn)),
en flux **y4m** : `stdin → stdout`. Utilisé par le workflow d'encodage de Muxiveo pour
multiplier la cadence (ex. 29,97 → 59,94 fps).

Extension de Muxiveo : installée, mise à jour et supprimée par l'application (page Extensions), dans le dossier
utilisateur. Le moteur, ses modèles, les poids de son sélecteur et ses préréglages évoluent sans release de
Muxiveo : l'application retient la version la plus récente dont le contrat (`--capabilities`) lui est connu.

## Nouveautés 1.7.1

- Calculs MC inutilisés supprimés et invariants réutilisés dans les préréglages hybrides, Ultra compris.
- Recherche MC : prédicteur partagé et vecteurs strictement identiques dédupliqués, sans changer les candidats utiles.
- Entrées TensorRT réutilisées entre sorties et entre paires ; Ultra et TTA temporel inversent leurs adresses sans recopie.
- Cache optionnel des caractéristiques Vulkan pour v4.15 et Light, y compris Ultra, UHD et TTA 2/4/8,
  avec réserve VRAM et repli automatique. Aucun changement de modèle ni de poids.
- Priorité aux conversions 23,976 → 59,94 et 24 → 60. Voir le [rapport des mesures](docs/2026-10-10-performance-qualite.md).

```
ffmpeg -i src.mkv -map 0:v:0 -pix_fmt yuv420p10le -f yuv4mpegpipe -strict -1 - \
  | muxiveo-rife --factor 2 --matrix bt2020nc --chroma-loc topleft \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libx265 out.hevc
```

## Garanties

- Trames d'origine recopiées **octet pour octet** ; seules les trames intermédiaires sont générées.
- Entrées 4:2:0 / 4:2:2 / 4:4:4, **8 à 16 bits** ; conversion YUV ↔ RGB sur le GPU
  (matrice, plage et position chroma respectées), calcul fp16 (ou `--fp32`).
- Coupes de scène (score façon `scdet`) et trames figées : duplication au lieu d'un fondu.
- Nombre de trames de sortie exact : `N × facteur` pour un facteur entier
  (la dernière trame est dupliquée), ce qui garde l'alignement des métadonnées par trame.
- Fonctionne sur tout GPU Vulkan (NVIDIA, AMD, Intel, Apple via MoltenVK) ; llvmpipe en secours (lent).

## Mode rapide (`--uhd`)

Flux optique calculé à demi-résolution, équivalent du `scale=0.5` de Practical-RIFE : l'échelle
de chaque bloc du réseau est doublée (16/8/4/2/1 → 32/16/8/4/2) par réécriture du graphe ncnn au
chargement (`src/uhd.cpp`), sans modèle dédié ; padding doublé. Mesuré en 4K (RTX 4070 Ti SUPER,
×2) : +16 % (`rife-v4.26`) à +36 % (`rife-v4.25-heavy`) de débit pour −0,1 à −0,6 dB de PSNR ;
en 1080p, environ −1 dB.

## Moteur hybride (`--engine hybrid`, 1.3.0)

RIFE retient parfois la mauvaise période sur les motifs fins et répétitifs (barreaux devant un bardage
rayé, grilles en panoramique) : les barreaux ondulent ou se dédoublent. Le moteur hybride ajoute, dans le
même lot de commandes Vulkan que RIFE, une **compensation de mouvement par blocs** :

1. pyramide de luminance 1/8 → 1/1, recherche bilatérale par blocs de 16 centrée sur l'image à créer
   (recherche exhaustive au niveau 1/8, puis raffinement avec pénalité de cohérence), deux passes de
   propagation aux niveaux grossiers (repliement de période) ; un seul champ de vecteurs par paire,
   réutilisé pour toutes les positions intermédiaires ;
2. reconstruction recouvrante (OBMC) des blocs voisins, en écartant le côté qui échantillonne hors image ;
3. décision par région : RIFE là où la densité de pixels mal expliqués dépasse 25 % (fenêtre de 31 px)
   et au bord du cadre, compensation ailleurs, moyenne des deux là où ils coïncident ; transition adoucie.

**Flux optique NVIDIA (`--nvof auto|off`)** : sur une carte NVIDIA compatible (Turing ou plus récente,
pilote avec CUDA et Optical Flow), les médianes par bloc du flux matériel aller / retour servent de
candidats supplémentaires au niveau final. Le pilote est chargé à l'exécution
(`libnvidia-opticalflow.so.1` + `libcuda.so.1`, `nvofapi64.dll` + `nvcuda.dll`) : aucune dépendance
au lancement. Le flux est calculé sur un thread, pendant l'inférence RIFE. Sans carte NVIDIA, ou si le
GPU n'a pas d'accélérateur de flux optique (Pascal et antérieurs), le moteur reste entièrement en Vulkan,
avec une qualité très proche. `--list-gpus` indique la disponibilité par GPU (`nvof`, `nvof_status`).

Mesures (4K, RTX 4070 Ti SUPER, PSNR-Y en dB image entière / zone des barreaux) : barrières ×2
35,30 / 32,54 (RIFE v4.6) → 36,48 / 34,67 (hybride v4.6) ; ville avec hélice ×2 identique à RIFE.
Vrai 24 → 59,94 en 4K : hybride ≈ 1,35 × le temps de RIFE avec `rife-v4.6`, ≈ 2 × avec `rife-v4.15`.
`--engine mc` (compensation seule) sert au diagnostic.

**Grands mouvements (`--large-motion auto|off|<px>`, 1.5.0)** : dans le flou de bougé et les occultations
d'un mouvement rapide (panoramique derrière un poteau, objet flou qui traverse le cadre), la compensation
par blocs trouve des vecteurs faux à faible coût et déforme l'image (« gouttes », contours ondulés), et
RIFE pleine résolution suit mal les grands déplacements. Le moteur hybride charge donc un second réseau
RIFE à flux demi-résolution (même graphe que `--uhd`, padding du modèle ×2) et l'utilise là où le
déplacement entre les deux sources est grand :

1. décision par paire : champ de vecteurs relu une fois ; passe lancée si au moins 1 % des blocs
   dépassent 1,5 × le seuil (aucun coût sur les plans calmes) ;
2. poids par pixel : amplitude du champ dilatée (maximum sur 9 × 9 blocs), rampe du seuil (16 px par
   défaut) à 3 × le seuil ; plancher pour toute l'image quand le mouvement médian de la paire dépasse
   1,5 × le seuil (complet à 2,5 ×), la compensation sous-estimant les vecteurs d'un panoramique flou.

Avec le plugin TensorRT, ce réseau utilise le modèle `<modèle>-uhd` du plugin (sinon Vulkan). Mesures
(4K, images paires interpolées ×2 comparées aux impaires, PSNR-Y en dB image entière / 1 % des blocs
64 × 64 les pires) : passage derrière un poteau 27,33 / 16,09 → 29,09 / 16,93, descente en rappel
34,16 / 19,63 → 34,70 / 20,44 ; barreaux et hélice inchangés (36,75 / 34,82 et 35,39 / 32,86, image
entière / zone). Vrai 24 → 59,94 en 4K (Vulkan) : +20 % de temps sur une séquence de barreaux où la passe
s'active pour 43 % des paires, rien sur un plan calme.

## Sélecteur appris (`--selector auto|off`, `--ultra`, 1.6.0)

Le moteur hybride ne combine plus RIFE et la compensation par une règle fixe : un **sélecteur appris** choisit,
zone par zone, le mélange des candidats calculés pour chaque image :

| Candidat | Origine | Présent |
|---|---|---|
| R1 | RIFE (modèle choisi) | toujours |
| R05 | RIFE à flux demi-résolution | paires à grand mouvement (`--large-motion`) |
| MC | compensation de mouvement par blocs | toujours |
| Rbwd | RIFE dans le sens inverse (sources échangées, `1 - t`) | `--ultra` |
| NV | déformation dense par le flux optique NVIDIA | carte NVIDIA avec NVOFA |

- Blocs proportionnels à la largeur de l'image (`largeur / 120`, de 8 px en SD à 64 px en 8K) : même
  comportement de la SD à la 8K. Poids des candidats interpolés entre centres de blocs (aucune couture).
- Indices par bloc : désaccords de chaque candidat avec R1 (et maximum sur 3 × 3 blocs), texture, écart
  temporel, luminance, incohérence et amplitude du flux NVIDIA, proximité du bord, taille de bloc ; modèle
  linéaire suivi d'un softmax (`src/shaders/sel_*.comp`, coût négligeable).
- Poids appris par famille de modèle RIFE (`rife-v4.6`, `rife-v4.15`, `rife-v4.15-mvo1`), avec et sans flux
  NVIDIA, et par bande de taille de bloc quand c'est utile : fichier de données `selector/selector.txt`, livré
  dans `rife-models/selector.txt` et lu au lancement (format : `src/selector_weights.h` ; autre fichier :
  `--selector-weights`). Améliorer le sélecteur revient à remplacer ce fichier. Fichier absent ou invalide :
  règle fixe, raison affichée sur la ligne `info: moteur` ; `--ultra` est alors refusé (code 3).
- `--ultra` ajoute le candidat Rbwd (une inférence RIFE de plus, environ 1,5 × le temps du préréglage
  Qualité) ; exige le moteur hybride et le sélecteur, incompatible avec `--uhd` et `--tta`.
- `--selector off` revient à la règle fixe de la 1.5 (diagnostic).

**Modèle `rife-v4.15-mvo1`** : RIFE v4.15 affiné par Muxiveo, graphe ncnn identique au modèle d'origine
(seuls les poids changent : même coût, `--uhd` et TensorRT inchangés) ; publié par l'extension
`mvo-rife-models` du dépôt muxiveo-plugins (release `mvo-rife-models-v1.0.0`) et épinglé par sha256 dans
`models.json`.

Mesures (images paires interpolées ×2 comparées aux impaires, ΔPSNR-Y moyen par rapport à RIFE v4.15 seul ;
scènes jamais vues à l'apprentissage : 30 scènes de 10 films en 4K, réduites en 1080p, 720p et SD, et 30
scènes de 5 titres SD / 720p natifs) :

| Configuration | 4K | 1080p | SD / 720p natifs | SD / 720p réduits |
|---|---|---|---|---|
| Hybride 1.5 (règle fixe) | +0,31 | +0,01 | −0,05 | −0,28 |
| Hybride v4.15-mvo1 (Qualité) | +0,81 | +0,52 | +0,34 | +0,32 |
| Hybride v4.15-mvo1 + `--ultra` (Ultra) | +0,86 | +0,60 | +0,43 | +0,41 |
| Qualité sans flux NVIDIA | +0,77 | +0,51 | +0,35 | +0,32 |
| Hybride v4.6 (Équilibré), par rapport à RIFE v4.6 | +0,73 | +0,60 | +0,43 | +0,55 |

## Accélération NVIDIA (TensorRT, 1.4.0)

Avec le plugin facultatif `mvo-rife-trt` ([dépôt muxiveo-plugins](https://github.com/Hydro74000/muxiveo-plugins)),
l'inférence RIFE passe par NVIDIA TensorRT for RTX sur les GPU Turing ou plus récents (Linux et Windows x86-64) ;
la compensation de mouvement, la conversion des couleurs et le reste du traitement restent en Vulkan.

```
muxiveo-rife --trt-plugin <dossier du plugin> [--trt-cache <dossier>] [--backend auto|vulkan|tensorrt] ...
```

- Les images passent de Vulkan à CUDA par des tampons partagés (`VK_KHR_external_memory_fd` / `_win32`,
  activées par un correctif de ncnn : voir UPSTREAM.md), sans copie par le processeur.
- Le moteur TensorRT d'un modèle est construit au premier usage puis mis en cache (`--trt-cache`, défaut :
  `~/.cache/muxiveo/trt-engines` ou `%LOCALAPPDATA%\Muxiveo\cache\trt-engines`), de même que les noyaux
  spécialisés pour chaque résolution. Le cache est validé par TensorRT (version, GPU, pilote).
- `auto` (défaut) : TensorRT si le plugin est présent et le GPU compatible, sinon Vulkan ; la ligne
  `info: inférence RIFE : …` donne le moteur retenu ou la raison du repli. Une erreur du plugin en cours de
  traitement bascule sur Vulkan pour la suite (une ligne `warning`) : le traitement n'échoue jamais à cause de lui.
  `tensorrt` : erreur (code 3) si TensorRT est indisponible.
- Modèles fournis par le plugin : `rife-v4.6`, `rife-v4.15`, `rife-v4.15-lite` et leurs variantes `--uhd` ;
  calcul fp16 (avec `--fp32`, inférence Vulkan).
- `--list-gpus` indique `trt_compatible` (Turing ou plus récent, pilote NVIDIA 575 ou plus récent), et avec
  `--trt-plugin`, `trt` et `trt_status` (plugin réellement utilisable).

## Cache optionnel des têtes ncnn (`--feature-cache`)

Désactivé par défaut. Sur les graphes compatibles v4.15 et Light, cette option conserve les caractéristiques des
deux images sources pour les sorties successives d'une même paire, en particulier à 23,976 → 59,94 et
24 → 60. Le graphe et la forme des tenseurs sont vérifiés ; les réseaux normal et à flux demi-résolution
ont des caches séparés. Chaque entrée est identifiée par sa source, son côté du graphe et son miroir :
Ultra et TTA 2/4/8 conservent leurs propres caractéristiques, sans supposer que les deux branches partagent
les mêmes poids. Le calcul et les poids restent identiques. v4.6 n'a pas ces têtes indépendantes et suit
son calcul courant.

Une première inférence amorce l'allocateur. La réutilisation demande ensuite une mesure de mémoire fiable
et une réserve d'au moins 1 Gio et 20 % du budget GPU, en plus du coût estimé des têtes. Si la réserve
manque ou ne peut pas être mesurée, le moteur recalcule normalement. L'inférence TensorRT suit son
graphe courant. La ligne `info: cache des têtes ncnn` compte les
réutilisations effectives. `--feature-cache-reserve <MiB>` permet d'augmenter la réserve minimale
(défaut : 1024), par exemple pour garder davantage de place à un encodeur GPU. Voir les
[mesures ciblées et décisions](docs/2026-10-10-performance-qualite.md).

## TTA (`--tta 2|4|8`)

Moyenne de plusieurs inférences de la même trame intermédiaire : `2` ajoute le sens temporel inverse
(trames échangées, temps `1 - t`), `4` le miroir horizontal, `8` les miroirs vertical et double. Les sorties
sont remises à l'endroit puis moyennées en fp32 ; coût ×n, VRAM d'une seule inférence (une soumission par
variante). Lisse les petites erreurs d'estimation (léger flou là où les variantes divergent) ; sans effet sur
les erreurs d'appariement de motifs répétitifs (barreaux en panoramique), communes à toutes les variantes.

## Sortie stderr (lue par Muxiveo)

```
info: inférence RIFE : TensorRT, plugin 1.0.0 (TensorRT-RTX 1.6.1)
info: moteur hybrid | flux optique NVIDIA : actif | grands mouvements : RIFE flux demi-résolution au-delà de 16 px | sélecteur : appris (v4.15 MVO, Ultra)
info: 3832x1592 10 bits … | 24/1 -> 48/1 fps | modèle rife-v4.15-mvo1 | GPU … (fp16)
progress in=120 out=240 interpolated=119 scenes=1 static=0 fps=28.4
info: grands mouvements : 57 paire(s) sur 238
info: sélecteur : 238 image(s) avec le candidat flux NVIDIA
done in=240 out=480 interpolated=238 scenes=1 static=0 seconds=16.89 exit=0
```

Codes de sortie : `0` OK, `1` usage, `2` entrée invalide, `3` GPU/modèle, `4` E/S (pipe fermé),
`5` mémoire GPU (VRAM) insuffisante.
`--list-gpus` affiche les GPU en JSON (avec la disponibilité du flux optique NVIDIA).

En-têtes y4m et options numériques sont lus strictement (1.2.3) : jetons
entièrement numériques, dimensions de 1 à 32768 et au plus 2²⁸ échantillons de
luma (16384 × 16384), termes de cadence et rapport sortie/entrée sur 31 bits,
réels finis ; une entrée hors limites est refusée (code `2` pour le flux, `1`
pour une option) avant tout calcul.

Avec `-i` et `-o`, l'entrée et la sortie doivent désigner des fichiers distincts,
y compris à travers un lien symbolique ou un hardlink. Sinon, la commande est
refusée avec le code `1` avant toute écriture. Le pipeline Muxiveo utilise
`stdin → stdout` et conserve ce fonctionnement.

## Contrat avec Muxiveo (`--capabilities`, 1.7.0)

`muxiveo-rife --capabilities` écrit sur stdout, sans initialiser le GPU, un JSON : `contract` (flux y4m, règle du
nombre de trames, lignes `info` / `progress` / `done` sur stderr, codes de sortie ; incrémenté seulement à une
rupture), `version`, `options` reconnues, `engines`, `tta`, `backends`, `trt_abi`, `models` présents dans
`rife-models/`, et `selector` (`format`, `status` = `ok` ou raison, `families`). Muxiveo valide les réglages
d'après ces capacités.

Les préréglages proposés par Muxiveo (Rapide, Équilibré, Qualité, Ultra, Light : identifiants fixes, libellés
et règles d'interface dans l'application) prennent leur moteur, leur modèle et d'éventuels arguments
supplémentaires dans `presets.json` : un nouveau modèle ou réglage ne demande pas de release de Muxiveo.

## Build

Prérequis : CMake ≥ 3.20, compilateur C++17, Git (ncnn et glslang sont téléchargés au tag épinglé).
Aucun SDK Vulkan requis : ncnn charge le pilote dynamiquement (`NCNN_SIMPLEVK`).

```
cmake -S mvo-rife -B build/mvo-rife -G Ninja
cmake --build build/mvo-rife                     # copie aussi rife-models/selector.txt
python3 mvo-rife/scripts/fetch_models.py build/mvo-rife/rife-models
MUXIVEO_RIFE_BIN=build/mvo-rife/muxiveo-rife pytest mvo-rife/tests     # GPU Vulkan (llvmpipe accepté)
python3 mvo-rife/scripts/package.py linux-x86_64 build/mvo-rife/muxiveo-rife build/dist
```

Option : `-DMUXIVEO_RIFE_NCNN_SOURCE_DIR=<checkout ncnn>` pour compiler hors ligne.

Releases : Linux compilé dans `manylinux_2_28` (glibc ≥ 2.28, runtime C++ statique), macOS 12+ (arm64,
MoltenVK livré à côté du binaire), Windows x64 (runtime MSVC statique).

Publication : incrémenter `project()` du CMakeLists, pousser, puis le tag `mvo-rife-vX.Y.Z`. La CI
(`.github/workflows/mvo-rife.yml`) publie les archives (`manifest.json` : SHA-256 de chaque fichier, contrat,
modèles, préréglages) et ajoute la version au flux `mvo-rife.json` de la release `extensions-feed`. Une rupture
du contrat (incrément de `MUXIVEO_RIFE_CONTRACT`) n'est proposée qu'aux versions de Muxiveo qui la connaissent.

Les modèles (`models.json`, sha256 épinglés) sont cherchés dans `<dossier de l'exécutable>/rife-models/<nom>`
ou passés avec `-m <dossier>`. Préréglages Muxiveo (`presets.json` ; Ultra et Light ajoutent `--ultra` / `--uhd`) :

| Préréglage | Moteur | Modèle |
|---|---|---|
| Rapide | `rife` | `rife-v4.6` |
| Équilibré (défaut) | `hybrid` | `rife-v4.6` |
| Qualité | `hybrid` | `rife-v4.15-mvo1` |
| Ultra | `hybrid` + `--ultra` | `rife-v4.15-mvo1` |
| Light (petites cartes graphiques) | `rife` + `--uhd` | `rife-v4.15-lite` |

Choix issus d'un banc de 17 modèles sur 4 contenus (v4.6 : meilleur VMAF moyen, débit le plus élevé, VRAM la
plus basse ; v4.15-lite + `--uhd` : configuration la plus sobre, 1,7 Go en 4K), puis du banc des motifs
répétitifs (hybride + v4.15 : meilleur résultat ; v4.25 / v4.26 sans gain sur ces contenus).

Provenance du code et procédure de mise à jour : [UPSTREAM.md](UPSTREAM.md).
