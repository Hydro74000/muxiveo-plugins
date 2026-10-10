# mvo-fel — Reconstruction Dolby Vision FEL

Extension facultative de Muxiveo. Le moteur reconstruit l'apport de la couche FEL
dans les pixels avant les filtres, le tone mapping, l'interpolation et l'encodage,
indépendamment de la conservation du RPU en sortie.

**Première préversion : `mvo-fel-v0.1.0`, installable depuis Extensions.**
Le paquet Linux x86_64 propose le CPU, Vulkan et le transport direct expérimental
NVIDIA, sur une base glibc 2.28. Les paquets Windows x86_64 et macOS arm64
proposent le CPU ; le GPU sur ces deux OS reste à valider. Les tests couvrent AMD
et NVIDIA, des séquences synthétiques et trois scènes FEL réelles. La lecture sur
matériel Dolby Vision reste à valider avant une release générale. Le [plan](PLAN.md) et
le [rapport réel](docs/2026-10-10-real-fel-validation.md) détaillent ces limites.

Dans Encodage → HDR → Dolby Vision, la case reste **décochée par défaut pour tous
les profils, P5/P7 compris**. Seule une activation explicite demande la
reconstruction, puis libdovi confirme la présence d'un FEL au lancement. Un MEL
ou une source sans EL donnent une opération sans effet annoncée au journal.

## Architecture

- Bibliothèque `libmvo_fel.so`, `mvo_fel.dll` ou `libmvo_fel.dylib`, ABI C
  versionnée et adaptateur Python `ctypes` dans Muxiveo.
- FFmpeg minimal et libdovi **liés statiquement dans le plugin**, symboles privés.
  Aucune modification du PATH ou du FFmpeg système. Le FFmpeg configuré dans
  Muxiveo reste utilisé pour le parcours en pipe. Le plugin ne fournit que les composants nécessaires au HEVC
  BL/EL et au transport NUT ; les autres codecs restent ceux de Muxiveo.
- Transport direct expérimental Linux/NVIDIA : un **FFmpeg privé supplémentaire**
  avec source `mvo_fel` et raccordement `mvo_fel_cuda`, uniquement pour cette
  reconstruction et l'encodeur `hevc_nvenc`. Images Vulkan → conversion P010
  et dithering → transfert GPU CUDA → NVENC, sans NUT ni retour des images RGB
  en RAM. Les décodeurs BL/EL restent CPU. Les pilotes ≥ 570, la même carte
  physique et une géométrie sans redimensionnement sont requis. Autres codecs,
  RIFE, SDR, filtres non pris en charge, autre GPU ou pilote inconnu conservent
  le pipe. Aucun encodeur ou traitement RIFE n'est réécrit ou déplacé.
- Calcul CPU portable, AVX2 sélectionné à l'exécution si disponible, sans GPU
  obligatoire. NVIDIA n'est pas requis : le consommateur peut être un encodeur
  logiciel ou un chemin matériel existant de Muxiveo.
- Prototype Vulkan : libplacebo et glslang statiques privés, pilote Vulkan
  chargé seulement pour le GPU. Choix dynamique **Auto / GPU détectés / CPU**
  dans Muxiveo, enregistré par UUID dans la piste et les profils. La liste FEL
  n'assimile pas les indices Vulkan, CUDA/NVENC et RIFE.
- Décodage BL/EL → reconstruction RGB PQ `gbrp16le` → **NUT en pipe** → filtres →
  tone mapping éventuel → RIFE éventuel → **un encodage compressé final**. Aucun
  fichier vidéo décompressé complet. Le mode deux passes refait la reconstruction
  pour chaque passe ; une erreur FEL peut provoquer une reprise entière sur BL.
- Pas de Display Management, de trim d'écran ni d'analyse des « faux FEL ».
- Installation, mise à jour et suppression depuis Extensions, release autonome
  `mvo-fel-vX.Y.Z` et flux `extensions-feed/mvo-fel.json`. Aucun téléchargement
  pendant l'encodage. Une demande sans moteur disponible produit un avertissement
  et utilise le BL, sans effacer le choix de l'utilisateur.

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

`validate_gpu.py` teste tous les GPU énumérés sur trois scènes distinctes.
Il utilise aussi le FFmpeg de validation pour lire en pipe le NUT produit par
la bibliothèque exacte : les images 0, 8 et 16 doivent avoir les mêmes
empreintes GBRP16LE que le renderer comparé aux références CPU et `kernel64`.
Aucun intermédiaire vidéo n'est écrit. L'attestation de schéma 2 contient ces
empreintes et les SHA-256 du moteur, des sources et des dépendances ; le
schéma 1, qui testait seulement le renderer séparé, n'est plus accepté.
Le packaging refuse un prototype sans attestation valide, ou sur
une autre plateforme. Le paquet est autonome hors pilote graphique ; la voie
CPU reste utilisable sans chargeur Vulkan. La CI peut préparer le candidat GPU
Linux avec l'entrée `prepare_gpu` ; sa publication exige ensuite l'attestation
sur les GPU physiques et les scènes réelles. Les autres OS conservent le CPU.

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
de référence aux images 0/8/16. Le paquet contient le binaire `ffmpeg-fel`, ses
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
