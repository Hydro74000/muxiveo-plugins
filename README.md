# muxiveo-rife

Interpolation d'images **RIFE v4** sur GPU (Vulkan via [ncnn](https://github.com/Tencent/ncnn)),
en flux **y4m** : `stdin → stdout`. Utilisé par le workflow d'encodage de Muxiveo pour
multiplier la cadence (ex. 29,97 → 59,94 fps).

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

## TTA (`--tta 2|4|8`)

Moyenne de plusieurs inférences de la même trame intermédiaire : `2` ajoute le sens temporel inverse
(trames échangées, temps `1 - t`), `4` le miroir horizontal, `8` les miroirs vertical et double. Les sorties
sont remises à l'endroit puis moyennées en fp32 ; coût ×n, VRAM d'une seule inférence (une soumission par
variante). Lisse les petites erreurs d'estimation (léger flou là où les variantes divergent) ; sans effet sur
les erreurs d'appariement de motifs répétitifs (barreaux en panoramique), communes à toutes les variantes.

## Sortie stderr (lue par Muxiveo)

```
info: moteur hybrid | flux optique NVIDIA : actif
info: 3832x1592 10 bits … | 24/1 -> 48/1 fps | modèle rife-v4.26 (uhd) | GPU … (fp16)
progress in=120 out=240 interpolated=119 scenes=1 static=0 fps=28.4
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

## Build

Prérequis : CMake ≥ 3.20, compilateur C++17, Git (ncnn et glslang sont téléchargés au tag épinglé).
Aucun SDK Vulkan requis : ncnn charge le pilote dynamiquement (`NCNN_SIMPLEVK`).

```
cmake -S native/muxiveo-rife -B build/muxiveo-rife -G Ninja
cmake --build build/muxiveo-rife
python3 native/muxiveo-rife/scripts/fetch_models.py build/muxiveo-rife/rife-models
```

Option : `-DMUXIVEO_RIFE_NCNN_SOURCE_DIR=<checkout ncnn>` pour compiler hors ligne.

Releases : Linux compilé dans `manylinux_2_28` (glibc ≥ 2.28, runtime C++ statique), macOS 12+ (arm64,
MoltenVK livré à côté du binaire), Windows x64 (runtime MSVC statique).

Publication automatique : incrémenter la version (`project()` du CMakeLists **et** `MUXIVEO_RIFE_VERSION`
de `core/version.py`) puis pousser. Le workflow de release de Muxiveo (`release.yml`) compile et publie
`muxiveo-rife-vX.Y.Z` si cette release n'existe pas, avant d'empaqueter l'application. Un code natif
modifié sans changement de version fait échouer la release. Le tag manuel `muxiveo-rife-vX.Y.Z` reste possible.

Les modèles (`models.json`, sha256 épinglés) sont cherchés dans `<dossier de l'exécutable>/rife-models/<nom>`
ou passés avec `-m <dossier>`. Préréglages Muxiveo :

| Préréglage | Moteur | Modèle |
|---|---|---|
| Rapide | `rife` | `rife-v4.6` |
| Normal (défaut) | `hybrid` | `rife-v4.6` |
| Qualité | `hybrid` | `rife-v4.15` |
| Light (petites cartes graphiques) | `rife` + `--uhd` | `rife-v4.15-lite` |

Choix issus d'un banc de 17 modèles sur 4 contenus (v4.6 : meilleur VMAF moyen, débit le plus élevé, VRAM la
plus basse ; v4.15-lite + `--uhd` : configuration la plus sobre, 1,7 Go en 4K), puis du banc des motifs
répétitifs (hybride + v4.15 : meilleur résultat ; v4.25 / v4.26 sans gain sur ces contenus).

Provenance du code et procédure de mise à jour : [UPSTREAM.md](UPSTREAM.md).
