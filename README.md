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

## Sortie stderr (lue par Muxiveo)

```
info: 3832x1592 10 bits … | 24/1 -> 48/1 fps | modèle rife-v4.26 (uhd) | GPU … (fp16)
progress in=120 out=240 interpolated=119 scenes=1 static=0 fps=28.4
done in=240 out=480 interpolated=238 scenes=1 static=0 seconds=16.89 exit=0
```

Codes de sortie : `0` OK, `1` usage, `2` entrée invalide, `3` GPU/modèle, `4` E/S (pipe fermé),
`5` mémoire GPU (VRAM) insuffisante.
`--list-gpus` affiche les GPU en JSON.

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

Les modèles (`models.json`, sha256 épinglés) sont cherchés dans `<dossier de l'exécutable>/rife-models/<nom>`
ou passés avec `-m <dossier>`. Préréglages Muxiveo : `rife-v4.6` (Rapide et Équilibré, défaut) et
`rife-v4.15-lite` (Light, petites cartes graphiques, toujours avec `--uhd`). Choix issu d'un banc de 17 modèles
sur 4 contenus : v4.6 obtient le meilleur VMAF moyen, le débit le plus élevé et la VRAM la plus basse ; v4.15-lite
+ `--uhd` est la configuration la plus sobre (1,7 Go en 4K).

Provenance du code et procédure de mise à jour : [UPSTREAM.md](UPSTREAM.md).
