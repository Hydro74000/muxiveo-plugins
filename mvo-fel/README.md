# mvo-fel — Reconstruction Dolby Vision FEL

Extension facultative de Muxiveo. Le moteur reconstruit l'apport de la couche FEL
dans les pixels avant les filtres, le tone mapping, l'interpolation et l'encodage,
indépendamment de la conservation du RPU en sortie.

**Préversion publiée : `mvo-fel-v0.1.0`.** La version 0.2.0 est en cours de
finalisation dans le dépôt : les lots 1 à 9 du
[plan de continuation](docs/2026-10-15-plan-continuation.md) sont réalisés et les
défauts de l'[audit du 15/10/2026](docs/2026-10-15-audit-continuation.md) corrigés
([réponse](docs/2026-10-15-reponse-audit.md)). Restent l'attestation de la
bibliothèque définitive (Linux et Windows), les mesures de bout en bout, la
construction portable et le paquet ; puis la publication et l'épinglage dans
Muxiveo, qui attendent l'accord du mainteneur. La lecture sur matériel Dolby
Vision reste à valider avant une release générale.

Dans Encodage → HDR → Dolby Vision, la case reste **décochée par défaut pour tous
les profils, P5/P7 compris**. Seule une activation explicite demande la
reconstruction, puis libdovi confirme la présence d'un FEL au lancement. Un MEL
ou une source sans EL donnent une opération sans effet annoncée au journal.

## Plateformes et validation (15/10/2026)

| Plateforme | Rendu | Décodage BL/EL | Validation |
|---|---|---|---|
| Linux x86_64 | Vulkan (AMD, NVIDIA), CPU | Vulkan Video contrôlé, sinon CPU | 6 scènes FEL réelles × 2 GPU |
| Windows x86_64 | Vulkan (NVIDIA, Intel), CPU | Vulkan Video contrôlé, sinon CPU | 6 scènes × GTX 1070 + UHD 630 ([lot 9](docs/2026-10-15-lot9-plateformes.md)) |
| macOS arm64 | CPU ; GPU MoltenVK en choix manuel seulement | CPU | CI uniquement : sonde non bloquante |

Écart maximal sur toutes les attestations : 0,125 code PQ12 face au CPU et 0,107
face à l'oracle `kernel64`, pour une limite de 2. Cette borne GPU est mesurée sur
contenu réel : les shaders calculent en float32, comme libplacebo, et sur le
domaine synthétique une composante proche de zéro dans une couleur saturée très
lumineuse peut la dépasser. Le CPU produit la référence certifiée : l'arrondi exact de la
formule, voir [CPU certifié](docs/2026-10-16-cpu-certifie.md).

Débit du plugin seul, 300 images UHD, sortie 10 bits finale (Ryzen 7800X3D) :

| Moteur | 0.1.0 | 0.2.0 |
|---|---:|---:|
| RTX 4070 Ti SUPER | 12,3 i/s (callback Python, RGB) | 85–88 i/s |
| iGPU AMD Raphael | ~14 i/s | 22,8 i/s |
| CPU, 12 threads | 0,7 i/s | 7,9 i/s (certifié) |

Les durées de workflow complètes (x265, NVEncC, FFmpeg/NVENC NUT et direct) sont
consignées avec le plan de continuation.

## Précision du rendu GPU

Muxiveo propose trois modes (`mvo_fel_set_precision`, capacité `precision`). Le CPU produit
toujours la référence certifiée, quel que soit le mode.

| Mode | Calcul GPU | Écart au CPU | RTX 4070 Ti SUPER | iGPU Raphael |
|---|---|---|---:|---:|
| `exact` (Lent, recommandé) | classement certifié entre seuils PQ, reprises par le CPU certifié, Y'CbCr en entiers exacts ; variante choisie par auto-test : virgule fixe multi-mots ([A2'](docs/2026-10-16-a2-virgule-fixe.md)) ou double ([A1](docs/2026-10-16-gpu-exact-certifie.md)) | identique au bit près (référence certifiée) | 43,6–51,4 i/s (virgule fixe) | 2,5 i/s (double) |
| `standard` | float32 reformulé, borne d'erreur par pixel, recalcul en double au-delà de 0,4 code | ≤ 1 code 16 bits | 66,7 i/s | 8,1 i/s |
| `fast` (défaut de l'ABI) | shader float32 Muxiveo : sommes, segments et zéro NLQ exacts, EOTF et OETF sans annulation ([détails](docs/2026-10-16-rapide-ameliore.md)) | 1 code 16 bits sur scènes réelles ; jusqu'à 112 codes 16 bits sur le stress synthétique le plus défavorable | 74–78 i/s | 19,8 i/s |

Débit du plugin seul, 300 images UHD, sortie 10 bits finale ; CPU (7800X3D) : 7,9 i/s.

Dans les modes `exact` et `standard` :

- le segment de mapping (pivots) et le zéro du résiduel NLQ sont décidés sur les sommes
  entières exactes ;
- le mapping, la NLQ et la matrice Y'CbCr→LMS' sont calculés en double ;
- un GPU calculant en double (`shaderFloat64`, clé `float64` de
  `mvo_fel_devices_features`) est requis ; sinon le rendu passe sur le CPU.

Le float-float (deux float32) a été écarté : ses termes correcteurs sont perdus selon
l'optimiseur du pilote, ce qui a été mesuré sur NVIDIA et sur AMD
([proposition](docs/2026-10-16-gpu-exact-proposition.md)).

Choix automatique du GPU :

| Mode | GPU retenu |
|---|---|
| `exact` | GPU dédié calculant en double (sur iGPU, plus lent que le CPU), sinon CPU |
| `standard` | GPU dédié, puis intégré, calculant en double, sinon CPU |
| `fast` | GPU dédié, sinon le premier GPU |

`fel_gpu_precision_test` vérifie chaque mode sur trois stress de 200 graines face au CPU.

## Architecture

- **Bibliothèque** `libmvo_fel.so`, `mvo_fel.dll` ou `libmvo_fel.dylib`. ABI C 1
  et extensions additives annoncées par `mvo_fel_capabilities()` : `seek`,
  `fd_output`, `output_formats` / `dithers`, `crop`, `timestamps`,
  `direct_formats`. L'adaptateur Python `ctypes` est dans Muxiveo
  (`core/fel/`).
- **Dépendances statiques et symboles privés** : FFmpeg minimal (démuxeurs,
  HEVC, `dovi_split`, NUT ; `hevc_vulkan` dans la version GPU), libdovi,
  libplacebo et glslang. Aucune modification du PATH ni du FFmpeg système. Les
  chargeurs Vulkan et pilotes sont ouverts à l'exécution.
- **Pipeline natif à trois étages bornés** :
  1. lecture, décodage BL/EL et appariement strict avec le RPU (ordre de
     présentation, identité des access units) ;
  2. rendu ;
  3. écriture.

  Les paquets viennent d'un pool réutilisé. Le seek natif positionne la source
  (aperçus) sans reconstruire ce qui précède.
- **Rendu GPU** (Vulkan) : une seule passe de composition lit les plans BL/EL.
  Le rééchantillonnage Spline16 se fait en entiers exacts : chroma BL et trois
  résiduels EL, avec zéro et signe du résiduel exacts. Viennent ensuite le shader
  couleur Muxiveo (modes `exact`/`standard`) ou les shaders libplacebo (mode `fast`),
  puis la conversion de sortie (en double en mode `exact`).
- **Rendu CPU certifié** : même arithmétique entière, passes séparables, PQ et matrice
  LMS→RGB en double vectorisées (AVX2+FMA choisi à l'exécution, NEON sur arm64) avec
  une borne d'erreur démontrée par composante, classement entre seuils PQ précalculés
  et reprise en double-double des rares cas indécidés (≈ 10⁻⁷) : le code est l'arrondi
  exact de la formule ([détails](docs/2026-10-16-cpu-certifie.md)). Conversion Y'CbCr en
  entiers exacts. Repli en double non certifié pour une géométrie non rationnelle.
- **Décodage Vulkan Video** : au départ du job, 48 images par couche sont
  décodées deux fois sur le GPU et comparées au décodage logiciel. Au moindre
  écart, ou sans l'extension, le décodage reste sur CPU.
- **Sorties** : RGB PQ 16 bits, ou Y'CbCr BT.2020nc limité 4:2:0 (16, 10 ou
  8 bits) sous-échantillonné à la position chroma du BL, avec un tramage
  déterministe (bruit bleu par défaut) identique sur CPU et GPU. Recadrage pair
  et PTS relatifs possibles.
- **Transports** :
  - NUT écrit directement sur le pipe du consommateur (descripteur ou HANDLE),
    sans callback ;
  - NVEncC lit le plugin sans étage FFmpeg quand l'aval ne touche pas aux
    pixels ;
  - direct expérimental Linux/NVIDIA : P010 final → CUDA → `hevc_nvenc` par
    `ffmpeg-fel`, avec couches décodées sur le même GPU.
- **Hors périmètre** : ni Display Management, ni trim d'écran, ni détection de
  « faux FEL ». Installation, mise à jour et suppression depuis Extensions,
  release autonome `mvo-fel-vX.Y.Z` et flux `extensions-feed/mvo-fel.json`.
  Aucun téléchargement pendant l'encodage.

Décisions documentées :

- [lot 7](docs/2026-10-15-lot7-coeurs-tensoriels.md) : cœurs tensoriels non
  retenus, aucun gain de bout en bout mesurable ;
- [lot 9](docs/2026-10-15-lot9-plateformes.md) : direct VAAPI/QSV non livré.

## Construction et paquet local

Prérequis : Python ≥ 3.10, CMake ≥ 3.22, Ninja, make, compilateur C++17,
pkg-config, NASM et Rust ≥ 1.88. Windows utilise MSYS2 UCRT64 et Rust GNU.
Les sources des dépendances sont téléchargées pendant la construction, avec
révisions et SHA-256 épinglés dans [dependencies.json](dependencies.json).

Depuis ce dossier :

```sh
python3 scripts/build.py build
python3 scripts/package.py linux-x86_64 build build/dist
```

Remplacer la plateforme par `windows-x86_64` ou `macos-arm64` selon le système.
La construction exécute les tests mathématiques et une reconstruction via l'ABI.
Le packaging refait ce test sur la bibliothèque copiée dans le paquet, puis
produit le manifeste et les sommes SHA-256. La CI dédiée construit les trois
plateformes indépendamment de RIFE.
Les builds CPU Linux, Windows et macOS, ainsi que la référence numérique Mesa,
ont passé la CI pour la première publication du code.

Pour les seuls tests mathématiques, des en-têtes FFmpeg système suffisent :

```sh
cmake -S . -B build/math -G Ninja -DMVO_FEL_MATH_ONLY=ON
cmake --build build/math
ctest --test-dir build/math --output-on-failure
```

La référence numérique nécessite Vulkan (Mesa llvmpipe convient), shaderc,
lcms2, libunwind, Meson et Jinja2. Elle construit libplacebo épinglé :

```sh
python3 scripts/reference.py build
```

Les tests comparent mapping polynomial/MMR, NLQ, matrices et images complètes
avec Spline16 et positions chromatiques identiques. Les intermédiaires de la
référence sont en float32 ; les FBO 16 bits du renderer par défaut introduisent
une quantification supplémentaire. Les maxima locaux observés sont 0,046 et
0,411 code PQ 12 bits, sous la limite de 2. Cela ne remplace pas une validation
sur de vrais films FEL. Les tests de workflows résident dans Muxiveo, notamment
`tests/integration/test_encode_fel.py`.

Pour un média FEL local (jamais ajouté au dépôt), le projet de référence produit
aussi `build/oracle/fel_reference_file` :

```sh
build/oracle/fel_reference_file /chemin/extrait.mkv 0 3 8 gpu32
build/oracle/fel_reference_file /chemin/extrait.mkv 0 3 8 kernel64
```

Les arguments indiquent la piste globale, le nombre d'images comparées et leur
espacement. `gpu32` conserve les shaders de rééchantillonnage ; `kernel64`
évalue les noyaux officiels libplacebo en double et centre le résiduel avant les
shaders NLQ/mapping/couleurs. Cette seconde référence évite la perte du signe
près du zéro NLQ en float32. Les deux résultats restent distincts : le média
réel testé atteint 0,066 code PQ12 en `kernel64`, mais 2,279 en `gpu32` avec
quelques pixels hors du seuil de 2. Cette divergence historique de l'oracle brut
reste documentée. Le nouveau backend de production centre l'EL avant
interpolation et calcule Spline16 avec compensation de précision. Sur trois
scènes, il reste sous deux codes PQ12 face au CPU **et** à l'oracle indépendant
`kernel64`, sur les deux GPU testés.

Le prototype GPU demande aussi Meson et les en-têtes Vulkan :

```sh
python3 scripts/build.py build --gpu
build/plugin/fel_reference_file extrait.mkv 0 3 8 validate
python3 scripts/validate_gpu.py build/plugin/libmvo_fel.so \
  build/plugin/fel_reference_file build/gpu-validation.json \
  extrait-1.mkv extrait-2.mkv extrait-3.mkv
python3 scripts/package.py linux-x86_64 build build/dist \
  --validation build/gpu-validation.json
```

`validate_gpu.py` teste tous les GPU énumérés sur trois scènes distinctes, dans
chaque précision : `fast` sur tout GPU, `standard` et `exact` sur les GPU
calculant en double. Il utilise le FFmpeg de validation pour lire en pipe le NUT
produit par la bibliothèque exacte : les images 0, 8 et 16 doivent avoir les
mêmes empreintes GBRP16LE que le renderer, comparé aux références CPU et
`kernel64`. Aucun intermédiaire vidéo n'est écrit.

Critères par précision :

- `fast` : ≤ 2 codes PQ12 face au CPU et à `kernel64` ;
- `standard` : en plus, ≤ 1 code 16 bits face au chemin CPU rapide (sommes entières exactes) ;
- `exact` : 0 code ; images RGB et Y'CbCr du producteur identiques octet pour octet à celles du CPU.

Les sorties finales Y'CbCr 4:2:0 10 bits du même NUT sont ensuite comparées à une
conversion de référence en double (numpy), recalculée depuis ce RGB attesté :

- position chroma `topleft` recadrée sans tramage : arrondi exact, ≤ 0,5 code ;
- position `left` avec le tramage livré : < 1 code.

L'attestation de schéma 4 contient, par précision, ces empreintes et ces écarts,
ainsi que les SHA-256 du moteur, des sources et des dépendances. Les schémas 1 à 3
ne sont plus acceptés.
Le packaging refuse un paquet GPU sans attestation valide. Il est autorisé sur
Linux et Windows ; macOS reste CPU, faute de contenu réel en CI. Le paquet est
autonome hors pilote graphique : `package.py` vérifie les dépendances ELF, ou les
imports de DLL Windows. La voie CPU reste utilisable sans chargeur Vulkan. La CI
prépare les candidats GPU Linux et Windows avec l'entrée `prepare_gpu`, et sonde
MoltenVK sur macOS. Leur publication exige ensuite l'attestation sur les GPU
physiques et les scènes réelles.

Le transport direct se construit et s'atteste séparément, sur Linux NVIDIA :

```sh
python3 scripts/build_direct.py build
python3 scripts/validate_gpu.py build/plugin/libmvo_fel.so \
  build/plugin/fel_reference_file build/gpu-validation.json \
  extrait-1.mkv extrait-2.mkv extrait-3.mkv --direct-ffmpeg build/ffmpeg-direct/ffmpeg
python3 scripts/package.py linux-x86_64 build build/dist \
  --validation build/gpu-validation.json --direct-ffmpeg build/ffmpeg-direct/ffmpeg \
  --max-glibc 2.28
```

L'attestation vérifie aussi les images Vulkan produites par ce binaire exact,
après lecture de 17 images, et refuse une différence d'empreinte avec le renderer
de référence aux images 0/8/16, dans chaque précision (option `precision` du
filtre). Les images P010 finales du filtre (position chroma, recadrage, tramage)
doivent être identiques, octet pour octet, aux sorties Y'CbCr attestées du
producteur. Le paquet contient le binaire `ffmpeg-fel`, ses
sources et configurations, les en-têtes NVIDIA MIT épinglés et leurs notices.
Muxiveo ne l'utilise que si le manifeste du plugin l'annonce et que le workflow
est compatible. Sinon, la reconstruction FEL continue par NUT.

Un paquet local peut être installé sans release depuis la racine Muxiveo :

```sh
python3 -m scripts.install_fel_prototype /chemin/mvo-fel-0.1.0-linux-x86_64-prototype.tar.gz
```

Cette commande utilise le gestionnaire Extensions et ses vérifications de
manifeste, sommes et ABI. `--root /chemin/test` isole l'installation du dossier
utilisateur. Après installation normale, rouvrir Muxiveo pour retrouver
l'extension et la liste dynamique des GPU. La case FEL reste décochée.

Les workflows réels s'exécutent dans le dépôt Muxiveo :

```sh
MUXIVEO_TEST_FEL_SOURCE=/chemin/film.mkv \
MUXIVEO_TEST_FEL_LIBRARY=/chemin/libmvo_fel.so \
QT_QPA_PLATFORM=offscreen pytest tests/integration/test_encode_fel_real.py
```

La fixture prélève environ une seconde à 480 s (`MUXIVEO_TEST_FEL_START` pour
changer ce point), au CRA, sans les RASL antérieures. Le fichier fourni reste
intact. Aucun média commercial n'est téléchargé ni redistribué.

Pour comparer le workflow Dolby Vision existant et le même workflow avec FEL,
depuis le dépôt Muxiveo :

```sh
QT_QPA_PLATFORM=offscreen python3 -m scripts.benchmark_fel_workflow /chemin/film.mkv \
  --library /chemin/libmvo_fel.so --work-dir /chemin/benchmarks \
  --duration 3 --runs 3 --threads 12
```

Le script compare x265 medium CRF 18 et NVEncC preset default CQP 18:20:22,
avec RPU conservé et sortie 10 bits. Il alterne l'ordre BL/FEL, mesure le
workflow complet (préparation, encodage, injection et assemblage), vérifie les
sorties hors chronométrage et conserve les temps bruts et médianes dans
`results.json`. `--codecs libx265` permet de mesurer sans NVIDIA. La découpe de
l'extrait est hors chronométrage ; des extraits courts ne prédisent pas le coût
d'un film complet.
Le [benchmark local appairé](docs/2026-10-10-fel-workflow-benchmark.md) contient
les mesures sur le média réel fourni.
Les [deux optimisations de transport](docs/2026-10-10-fel-transport-optimizations.md)
décrivent la nouvelle voie directe et le pipe allégé. Pour comparer FFmpeg/NVENC,
utiliser `--codecs hevc_nvenc`, puis ajouter
`--direct-ffmpeg /chemin/ffmpeg-fel` pour la série directe.
L'[investigation GPU](docs/2026-10-10-performance-gpu.md) détaille la correction
PQ, la distribution locale et les performances. La
[matrice de placements](docs/2026-10-10-gpu-pipeline-matrix.md) compare x265 et
NVEncC, avec/sans RIFE, et explicite les configurations encore à mesurer.

Pour reproduire cette matrice depuis Muxiveo :

```sh
QT_QPA_PLATFORM=offscreen python3 -m scripts.benchmark_fel_matrix /chemin/film.mkv \
  --library /chemin/libmvo_fel.so --work-dir /chemin/benchmarks \
  --duration 3 --runs 3 --fel-devices off cpu all --rife-gpus off auto
```

Les placements sont explicites pour comparer les mêmes étapes. Les charges
GPU calcul/encodage/décodage et la mémoire NVIDIA sont échantillonnées si
`nvidia-smi` existe ; sysfs est facultatif sur Linux AMD/Intel. L'absence de
télémétrie reste une valeur inconnue. `--codecs libx265` fonctionne sans NVIDIA ;
`--rife-gpus off` permet une machine sans RIFE. `--resume results.json --runs 3`
complète les passages manquants avec les mêmes paramètres.

## Sources et remplacement des composants

Chaque paquet contient `sources.tar.gz` : code du moteur, scripts de build,
sources FFmpeg/libdovi, archives vérifiées et dépendances Rust vendoriées.
Pour reconstruire ou remplacer un composant LGPL :

```sh
mkdir rebuild
tar -xf sources.tar.gz -C rebuild
cd rebuild/mvo-fel
mkdir -p build
mv ../dependencies build/sources
python3 scripts/build.py build
```

Les dépendances déjà présentes sont réutilisées et Cargo fonctionne hors ligne
avec les sources vendoriées. Modifier les sources extraites du composant voulu
avant la construction, puis remplacer la bibliothèque du plugin par celle de
`build/plugin/` (ABI inchangée). Aucun objet propriétaire de Muxiveo n'est
nécessaire au reliage. Pour changer la révision téléchargée, mettre aussi à jour
le verrou de dépendances. Le manifeste du paquet de remplacement doit être
régénéré par `scripts/package.py` avant installation.

## Licences

Code C++ adapté de libplacebo : **LGPL-2.1-or-later**, voir [NOTICES.md](NOTICES.md)
et [LICENSES](LICENSES/). FFmpeg est construit sans composants GPL/nonfree ;
libdovi est sous MIT. Aucun code GPL de FelBaker/DoViBaker n'est intégré.
La licence MIT du code Python de Muxiveo reste inchangée.
