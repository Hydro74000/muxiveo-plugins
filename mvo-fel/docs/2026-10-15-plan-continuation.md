# mvo-fel — plan de continuation (15 octobre 2026)

Suite du [PLAN.md](../PLAN.md) après la préversion `mvo-fel-v0.1.0`. Ce plan repose
sur une relecture complète du plugin (`src/`, `include/`, `integrations/ffmpeg/`,
`scripts/`, `tests/`, `docs/`) et de son intégration Muxiveo (`core/fel/`, runner,
workflows). Les chemins `mediarecode/…` désignent le dépôt Muxiveo.

## 1. Point de départ

Le socle est livré : ABI C, FFmpeg/libdovi/libplacebo privés, CPU + Vulkan, NUT,
direct Vulkan→CUDA→NVENC (Linux), repli BL, interface/profils, attestation PQ12,
flux Extensions. L'activation reste manuelle.

Mesures existantes (7800X3D + RTX 4070 Ti SUPER, 721 images UHD, 1 passage) :

| Entrée FFmpeg/NVENC | Durée | Écart BL |
|---|---:|---:|
| BL seul | 21,1 s | — |
| FEL NUT | 61,2 s | +190 % |
| FEL direct | 24,2 s | +14 % |

Le rendu GPU seul atteint 77–82 i/s, transferts compris. **Le shader n'est pas
le goulot.** Le coût vient de la sérialisation, du transport et du décodage CPU.

## 2. Constats de l'audit

### Défauts

| # | Constat | Emplacement |
|---|---|---|
| B1 | **Aperçu image FEL** : `-ss` reste devant `-i pipe:0`. Le plugin reconstruit donc toutes les images depuis 0 jusqu'à la scène tirée au hasard, potentiellement des dizaines de milliers d'images par capture. | `mediarecode/core/workflows/encode/workflow.py:2699`, `mediarecode/core/fel/pipeline.py:31-36` |
| B2 | **Source relue en entier deux fois** pour le RPU : extraction en mode 0 pour la classification, puis en mode 2 pour la copie P8.1. `FelExecution.original_rpu` n'est jamais réutilisé. | `mediarecode/core/fel/preparation.py:34`, `runtime/nvencc_execution.py:529`, `runtime/multi_video.py:201`, `runtime/metadata_inject.py:240`, `core/fel/context.py:13` |
| B3 | **Position chroma incohérente selon le transport** : le direct impose `chroma_location=topleft`, le NUT garde la valeur zscale par défaut (left). La valeur du BL source n'est consultée nulle part. | `mediarecode/core/fel/direct.py:63`, `core/fel/pipeline.py:10-13` |
| B4 | **Tramage différent selon le transport** : error diffusion avec zimg, bruit bleu avec libplacebo. La sortie dépend donc de l'éligibilité au direct. | idem |

### Goulots

| # | Constat | Emplacement |
|---|---|---|
| P1 | Le pipe NUT passe par un callback Python ctypes à chaque bloc de 64 Kio, soit environ 760 appels par image UHD. Chaque appel prend le GIL et fait trois copies : `string_at`, memoryview puis BufferedWriter de `Popen.stdin`. | `mediarecode/core/fel/engine.py:209-221`, `src/producer.cpp:192-194`, `mediarecode/core/runner.py:839,870` |
| P2 | Le pipeline natif est entièrement sérialisé : lecture → décodage BL → décodage EL → upload → rendu → download → `pl_gpu_finish` → NUT → pipe. Aucune image n'est en vol. | `src/producer.cpp:223-341`, `src/gpu_backend.c:317` |
| P3 | Le décodage HEVC BL+EL se fait sur CPU, avec threads/3 par décodeur, y compris en GPU et en direct. C'est le plafond probable du direct (~30 i/s). | `src/producer.cpp:185-186` |
| P4 | Passes GPU coûteuses : la luma BL passe deux fois en Spline16 alors que le ratio est 1 (identité). Douze intermédiaires float32 pleine résolution, trois passes de séparation GBR, environ 13 dispatchs par image. Les uploads partent de mémoire non épinglée. | `src/gpu_backend.c:198-233,297-315` |
| P5 | La conversion RGB→YUV se fait côté FFmpeg avec deux zscale CPU (RGB16 → 444p16, puis 420p10 en error diffusion). Le transport fait 49,8 Mo par image, contre 24,9 Mo en 4:2:0 sur 16 bits. | `mediarecode/core/fel/pipeline.py:10-13,48-53` |
| P6 | NVEncC : plugin → FFmpeg (zscale) → NUT → NVEncC. Un processus intermédiaire existe uniquement pour convertir. | `mediarecode/core/workflows/encode/runtime/nvencc.py:951` |
| P7 | Chemin CPU : tout en double, 12 `pow` par pixel, luma BL rééchantillonnée pour rien, threads recréés à chaque image (`std::async`). `dot4_avx2` réduit horizontalement par un store et ne vectorise rien d'utile. Résultat : ~1 i/s UHD, et Windows/macOS n'ont **que** ce chemin. | `src/reconstruction.cpp:35-40,254-289`, `src/producer.cpp:285-295` |

Mineur : `sources.tar.gz` embarque `build/sources` en entier (archives et arbres
extraits) ainsi que les `__pycache__` (`scripts/package.py:137-139`).

## 3. Règles du chantier

- On optimise **uniquement le FELMerge**. Réglages d'encodeur, RIFE et filtres
  restent inchangés.
- Tolérance inchangée : PQ12 ≤ 2 face au CPU et à `kernel64`. Tout nouveau noyau
  GPU repasse l'attestation de schéma 2, avec images exactes de la bibliothèque
  livrée.
- Un gain mesuré l'emporte sur la simplicité. Un gain non mesuré n'est pas livré.
- Bancs de 30 à 60 s, plusieurs scènes, au moins trois passages, après chauffe.
  Médias de test sur disque (Drive2), jamais en tmpfs.
- Aucune publication sans accord (ni plugin, ni Muxiveo). Ordre : release du
  plugin, puis épinglage `MVO_FEL_VERSION`.
- TensorRT n'est pas pertinent : ce n'est pas un réseau, et la licence (clause
  2.13) l'exclurait de toute façon. « Tensoriel » désigne ici les cœurs
  tensoriels via `VK_KHR_cooperative_matrix`.

## 4. Lots

Ordre recommandé : **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9**. Les lots 8
(CPU) et 9 (plateformes) peuvent avancer en parallèle après le lot 6.

### Lot 0 — Mesure de référence

- Chronomètres par étage en **temps occupé par thread**, pas en frontières
  d'appel. Débit i/s par étage. Profondeur des files.
- `benchmark_fel_matrix` sur 30–60 s : BL, NUT, direct ; x265 medium, NVEncC,
  FFmpeg/NVENC.
- Sortie : tableau de référence daté, conservé avec les résultats bruts.

Corpus vérifié le 15/10 avec `dovi_tool info --summary` sur les 240 premiers
RPU. Les médias restent privés et hors dépôt.

| Source | Type | Usage |
|---|---|---|
| Moana, extraits `real-moana/` (Drive2) | P7 FEL, une piste, animation | attestation PQ12 (3 scènes) |
| Toy Story 5 (`sabnzbd/complete/`) | P7 FEL, une piste, animation | bancs 30–60 s, nouvelle attestation |
| Il faut sauver le soldat Ryan (`sabnzbd/complete/`) | P7 FEL, une piste, **prise de vue réelle avec grain**, L1 jusqu'à 734 nits | bancs 30–60 s, attestation (résiduel bruité, NLQ autour de zéro) |
| Evil Dead Burn (`sabnzbd/complete/`) | P7 MEL, une piste | non-régression « sans effet (mel) » |
| Alita, BDMV `00688.m2ts` | P7 MEL **sur deux PID** (BL 0x1011 HDR10+, EL 0x1015) | non-régression BDMV |

Remarque sur les formats :

- Le plugin ne lit que des P7 sur une seule piste (`dovi_split`). Un BDMV sur
  deux PID n'est pas reconstructible, et le RPU se trouve dans le PID EL.
  Comportement attendu aujourd'hui : repli BL averti, à vérifier. Prendre en
  charge ce format reste hors périmètre sauf décision contraire.

### Lot 1 — Corrections B1–B4

1. **Seek natif (B1).** Option additive à l'ABI 1, capacité `seek: 1`.
   - Fonctionnement : `av_seek_frame` en arrière jusqu'à l'image clé, puis
     décodage et rejet jusqu'au PTS cible.
   - Après un seek, l'appariement se fait **par identité**. Avec un GOP ouvert,
     les RASL peuvent être rejetées différemment par BL et EL : une image sans
     contrepartie est écartée avant la première sortie, ensuite elle reste une
     erreur stricte.
   - Muxiveo transfère `-ss` au producteur.
   - Tests : GOP ouverts et fermés, B-frames, départ non nul.
2. **Réutiliser le RPU original (B2).** Produire la copie P8.1 par
   `dovi_tool editor -j {"mode":2}` sur `fel_original_N.bin`. Vérifier sur le
   corpus l'égalité octet à octet avec `-m 2 extract-rpu`. On supprime ainsi une
   lecture complète de la source.
3. **Une seule source pour la position chroma (B3).** Prendre `chroma_location`
   du BL décodé (VUI). L'appliquer à la conversion, quel que soit le transport,
   et vérifier la signalisation du flux final.
4. **Un seul tramage (B4).** Il sera fixé au lot 4. D'ici là, documenter l'écart.

### Lot 2 — Transport natif sans Python (P1)

- ABI additive `mvo_fel_run_fd(ctx, intptr_t fd_ou_handle, progress, opaque)`,
  capacité `fd_output: 1`.
  - POSIX : `write` avec EINTR, et EPIPE → `OUTPUT_CLOSED`.
  - Windows : `WriteFile` sur HANDLE ; `ERROR_NO_DATA` ou `ERROR_BROKEN_PIPE` →
    `OUTPUT_CLOSED`.
- AVIO en mode `direct` : un paquet image correspond à une écriture, sans
  découpage en 64 Kio.
- Muxiveo transmet `procs[0].stdin.fileno()`, ou `msvcrt.get_osfhandle` sous
  Windows, sans BufferedWriter. Le callback reste en secours si la capacité est
  absente.
- Critère de sortie : empreintes NUT identiques, gain mesuré sur 721 images.

### Lot 3 — Pipeline asynchrone borné (P2)

- Trois étages et des files de 2 à 3 images :
  1. lecture + décodage BL/EL ;
  2. upload + rendu GPU asynchrone (anneau de `pl_buf` mappés, plus de
     `pl_gpu_finish` à chaque image) ;
  3. download dans un anneau + écriture.
- Ordre et identité BL/EL inchangés. Annulation et fermeture aval préservées.
- Pool de threads persistant pour le chemin CPU.
- Budget mémoire recalculé : `runtime_helpers.py`, réserve FEL de 3 Gio.
- Critère de sortie : images bit à bit identiques au chemin synchrone.

### Lot 4 — Conversion de sortie sur GPU et format négocié (P5, B4)

- Un shader fait RGB PQ → Y'CbCr BT.2020nc limité, sous-échantillonnage 4:2:0
  (position chroma du lot 1) et tramage. Un seul algorithme, choisi après mesure.
- Format de sortie choisi par une règle unique Muxiveo `fel_output_format(video)`,
  partagée par l'aperçu, le workflow et le CLI :
  - **10 bits final tramé** (P010 ou yuv420p10) quand l'aval ne modifie pas les
    pixels (crop, setparams, format, hwupload). Le transport passe à 24,9 Mo et
    les deux zscale disparaissent.
  - **yuv420p16** quand des filtres, le tone mapping ou RIFE suivent. Même
    volume, et une précision supérieure à l'entrée du parcours BL actuel
    (4:2:0 10 bits).
  - `gbrp16le` reste le secours pour une exigence RGB.
- Validé par l'utilisateur le 15/10 : sortie finale tramée dans le plugin. Pas
  de contre-indication, à quatre conditions :
  - **Tramage déterministe** (motif fonction du seul index d'image). Les deux
    passes doivent voir des images identiques.
  - **Profondeur paramétrée** : 8 ou 10 bits selon `resolve_output_bit_depth`,
    H.264 8 bits compris.
  - **Règle complète** : tout étage qui modifie les pixels impose 16 bits, y
    compris les `--vpp-*` de NVEncC, le redimensionnement matériel et le
    `scale_cuda`/`scale_vaapi` de conversion.
  - **L'error diffusion se parallélise mal sur GPU.** Choisir entre bruit bleu
    et ordonné sur mesure : débit à CRF constant et métriques.
- Critère de sortie : qualité finale ≥ parcours actuel (PSNR, SSIM, ΔE ITP sur
  images décodées, mêmes réglages d'encodeur), sans régression du compte d'images
  ni des PTS.

### Lot 5 — NVEncC sans étage intermédiaire (P6)

Sans filtre, le plugin écrit le NUT ou le y4m P010 directement dans l'entrée de
NVEncC (`--avsw`). Avec des filtres, la chaîne actuelle est conservée. NVEncC
lui-même n'est pas modifié.

### Lot 6 — Tout sur GPU via Vulkan (P3, P4)

1. **Décodage Vulkan Video des couches BL et EL** dans le FFmpeg privé du plugin
   (`--enable-vulkan`, hwaccel HEVC).
   - Le périphérique FFmpeg est créé par le plugin puis importé dans libplacebo
     (`fel_gpu_import` existe). Les images décodées sont mappées sans upload
     (`pl_map_avframe_ex`).
   - Sonde par GPU : `VK_KHR_video_decode_h265`, Main10, dimensions EL.
   - Vérifications sur le corpus :
     - les empreintes BL/EL du décodage GPU égalent celles du décodage logiciel
       (le décodage HEVC est normatif) ; sinon le GPU est refusé pour le
       décodage ;
     - la side data DOVI est présente sur les images hwaccel.
   - Repli : décodage CPU par job.
   - macOS / MoltenVK : pas de Vulkan Video, donc décodage CPU.
2. **Noyaux compute fusionnés.**
   - Lecture directe des images décodées, sans intermédiaires float32.
   - Chemin identité pour la luma BL.
   - Une ou deux dispatchs au total : rééchantillonnage, NLQ/mapping/couleurs et
     conversion du lot 4.
3. **Rééchantillonnage EL exact en entiers.**
   - Dans le cas standard (EL 1920×1080 → 3840×2160), les phases Spline16 sont
     rationnelles, donc les poids sont des entiers exacts :
     - luma horizontale `[-3, 23, 23, -3] / 40` ;
     - luma verticale `[-23, 273, 83, -13] / 320` ;
     - chroma au plus `/ 2560` par axe.
   - Avec des codes centrés |c| ≤ 512, la somme 2D de la luma reste
     ≤ 10 436 608 < 2²⁴. Elle est donc **exacte**, même avec des accumulateurs
     FP32.
   - Le signe du résiduel devient exact. L'émulation float-float et la garde
     7,1×10⁻¹⁵ disparaissent.
   - La chroma demande deux passes séparables, avec intermédiaire int32 et
     accumulation 64 bits ou découpage.
   - Ratios non standard : le chemin float-float actuel est conservé.
4. Critère de sortie : nouvelle attestation sur AMD et NVIDIA, 3 scènes, sous
   2 PQ12. Comparaison ajoutée à un oracle **rationnel exact**.

### Lot 7 — Cœurs tensoriels (si disponibles)

- Variante du noyau du lot 6.3 via `VK_KHR_cooperative_matrix`. Les formes et
  types sont sondés à l'exécution : NVIDIA RTX, AMD RDNA3, Intel Arc.
- Luma EL : entrées FP16 exactes (codes ≤ 512, poids ≤ 273), produits et sommes
  < 2²⁴, donc **exact en FP16×FP16→FP32**. Les poids étant périodiques (deux
  phases par axe), chaque tuile est le produit par une matrice constante.
- Chroma et ratios atypiques : découpage int8/FP16 en limbs, ou noyau ALU.
- Candidat secondaire : évaluation MMR en GEMM (22 MAC par pixel et par canal).
- **Ce lot n'est conservé que s'il est bit à bit identique au noyau entier ALU
  et mesurablement plus rapide de bout en bout.** Le calcul (~0,4 GMAC/image)
  n'est pas le goulot actuel : le gain est à prouver une fois les lots 2 à 6
  livrés. Sinon, abandon documenté.

### Lot 8 — Chemin CPU de secours (P7)

- Mêmes poids entiers que le lot 6.3, identité pour la luma BL.
- SIMD **entre pixels** : AVX2 / AVX-512 choisis à l'exécution, NEON sur
  macOS arm64.
- PQ en float32 ou en table, avec borne d'erreur prouvée face à `kernel64`.
- Pool persistant.
- Objectif à mesurer : au moins ×5. Prioritaire tant que Windows/macOS n'ont pas
  de GPU.

### Lot 9 — Plateformes et raccordements matériels

- **Windows x64 GPU** : libplacebo et glslang statiques en MinGW,
  `vulkan-1.dll` chargé dynamiquement, Vulkan Video, raccordement CUDA/NVENC.
  - Validation sur le PC sandbox Windows : GTX 1070 et iGPU Intel. La prise en
    charge Vulkan Video de Pascal est à sonder.
- **Direct étendu** :
  - VAAPI (AMD/Intel Linux) par Vulkan → DRM PRIME → `hwmap` ;
  - QSV par dérivation VAAPI ;
  - AMF (D3D11) : recherche seulement.
  - L'encodeur est alors celui du FFmpeg privé, comme pour NVENC aujourd'hui.
    Validé par l'utilisateur le 15/10, aux conditions suivantes :
    - Détection des capacités refaite **sur le binaire privé** (presets, modes
      de débit, `-compression_level` VAAPI selon le pilote). Une option non
      prise en charge renvoie vers NUT.
    - **Aucune dépendance système évitable.** Les répartiteurs libva,
      libva-drm, libdrm et libvpl sont liés en statique dans `ffmpeg-fel`. C'est
      le modèle du FFmpeg BtbN déjà embarqué par Muxiveo : même recette, mêmes
      versions que BtbN pour mutualiser.
    - Le binaire BtbN lui-même ne peut pas être réutilisé, puisqu'il doit
      contenir les filtres `mvo_fel`.
    - Restent forcément sur le système, comme le pilote NVIDIA pour NVENC, les
      composants propres au matériel : pilote VA (`iHD` / `radeonsi_drv_video`),
      runtime oneVPL GPU (`libmfx-gen`) et pilote noyau.
    - Chemin de recherche des pilotes compilé sur les emplacements usuels des
      distributions, `LIBVA_DRIVERS_PATH` respecté.
    - `package.py` vérifie qu'aucune libva ni libvpl n'apparaît dans les
      `NEEDED`. La liste autorisée ne change pas.
    - Le journal indique quel FFmpeg encode : un FFmpeg configuré sur mesure
      n'encode pas en direct.
    - Sortie comparée au FFmpeg configuré, mêmes arguments.
- **Intel** : serveur jonsbo en production. Uniquement un paquet portable dans
  un dossier temporaire, nettoyé ensuite.
- **macOS arm64 : runners GitHub uniquement** (pas de Mac disponible).
  - Une tâche CI sonde MoltenVK et l'exposition d'un GPU Metal dans la VM du
    runner (non garantie). Les tests numériques synthétiques GPU tournent
    contre la référence libplacebo.
  - L'attestation sur contenu réel y est impossible, car les médias ne sont pas
    publiables. Ajouter un corpus synthétique de stress : RPU poly/MMR variés,
    NLQ et EL autour de zéro, positions chroma, géométries.
  - Paquet par défaut : CPU optimisé NEON (lot 8). Le GPU macOS est proposé
    seulement si la sonde et le corpus de stress passent, **en choix manuel**,
    avec Auto sur CPU tant qu'aucun contenu réel n'est validé. Cette borne a été
    validée par l'utilisateur le 15/10 et doit être documentée dans le README.
- **Windows** : attestation sur contenu réel possible sur le PC sandbox, avec
  copie privée des extraits, nettoyés ensuite.

### Lot 10 — Validation de release générale

- Corpus FEL élargi.
- Lecture P8.1 et repli HDR10 sur un téléviseur Dolby Vision (utilisateur).
- Bancs répétés.
- Mise à jour de la politique Auto avec la matrice.
- Docs sans aucun chiffre TensorRT.
- Release `mvo-fel-v0.2.0` puis épinglage dans Muxiveo, **après accord**.

## 5. Objectifs à mesurer

Ce sont des cibles de validation, pas des prédictions.

| Parcours | Aujourd'hui | Cible |
|---|---:|---:|
| FFmpeg/NVENC direct, Linux | +14 % | ≤ +5 % (lot 6) |
| NUT → NVEncC / FFmpeg | +190 % | ≤ +40 % (lots 2–5), puis ≤ +20 % (lot 6) |
| x265 medium | +15–28 % | coût FEL masqué par l'encodeur |
| CPU UHD | ~1 i/s | ≥ 5 i/s (lot 8) |

## 6. Décisions

Prises le 15/10 :

1. Corpus : Toy Story 5 et Il faut sauver le soldat Ryan (FEL), Evil Dead Burn
   et Alita (MEL). Voir le lot 0.
2. Sortie tramée par le plugin : oui, aux conditions du lot 4.
3. VAAPI/QSV par le FFmpeg privé : oui, tout ce qui peut l'être est embarqué en
   statique. Voir le lot 9.
4. macOS : runners GitHub uniquement. Le GPU est proposé en choix manuel, Auto
   reste sur CPU tant qu'aucun contenu réel n'est validé. Voir le lot 9.

## 7. Hors périmètre

- Détection des « faux FEL ».
- Display Management.
- Modifications de RIFE et des encodeurs.
- Mémoire partagée FEL→RIFE entre processus Vulkan (piste future, RIFE non modifié).
