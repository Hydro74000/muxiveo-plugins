# Provenance du code dérivé

`muxiveo-rife` est un pseudo-fork maintenu dans le dépôt Muxiveo. Seule une petite
partie du code amont est reprise ; le reste est propre à Muxiveo.

## Amont

| Composant | Dépôt | Révision | Licence |
|---|---|---|---|
| rife-ncnn-vulkan | https://github.com/nihui/rife-ncnn-vulkan | — | MIT (`LICENSES/MIT-rife-ncnn-vulkan.txt`) |
| fork modèles v4.7–v4.26 | https://github.com/TNTwise/rife-ncnn-vulkan | `13338e38debe2e400b3eeecf6792312d01a692f9` | MIT |
| ncnn | https://github.com/Tencent/ncnn | tag `20260526` | BSD-3-Clause (`LICENSES/BSD-3-Clause-ncnn.txt`) |
| modèles RIFE | https://github.com/hzwer/Practical-RIFE | via TNTwise (`models.json`) | MIT (`LICENSES/MIT-Practical-RIFE.txt`) |
| NVIDIA Optical Flow SDK (en-tête) | https://github.com/NVIDIA/NVIDIAOpticalFlowSDK | `edb50da3cf849840d680249aa6dbef248ebce2ca` (API 2.0) | BSD-3-Clause (`LICENSES/BSD-3-Clause-NVIDIA-Optical-Flow-SDK.txt`) |

## Fichiers repris

| Fichier | Origine | Modifications Muxiveo |
|---|---|---|
| `src/warp.cpp`, `src/rife_ops.h` | TNTwise `src/` | pipeline `warp_pack8` retiré (ncnn ≥ 2025 n'a plus de chemin Vulkan pack8 : `afpvec8` supprimé) |
| `src/shaders/warp.comp`, `warp_pack4.comp` | TNTwise `src/` | aucune |
| `src/shaders/rife_v4_timestep.comp` | TNTwise `src/` | aucune |
| `cmake/generate_shader_comp_header.cmake` | TNTwise `src/` | octets castés en `(char)` : shaders UTF-8 (commentaires accentués) acceptés |
| `src/third_party/nvof/nvOpticalFlowCommon.h` | NVIDIA Optical Flow SDK | aucune (structures et énumérations seulement ; la table de fonctions CUDA est redéclarée dans `src/nvof.cpp`) |
| ncnn `src/gpu.cpp` (au build) | ncnn, tag épinglé | `cmake/patch_ncnn.cmake` insère `cmake/ncnn_external_memory.inc` : active `VK_KHR_external_memory_fd` / `_win32` si le GPU les propose (tampons partagés avec CUDA pour le plugin TensorRT). Correctif idempotent, appliqué au tag téléchargé comme à `MUXIVEO_RIFE_NCNN_SOURCE_DIR` ; ancre introuvable = erreur de configuration |

## Code propre à Muxiveo

- `src/engine.{h,cpp}` : moteur RIFE v4 réécrit à partir de `RIFE::process_v4`
  (chemin non-TTA). Entrées/sorties YUV brutes 8–16 bits ; conversion YUV ↔ RGB,
  normalisation et quantification sur le GPU (`yuv_to_rgb.comp`, `rgb_to_yuv.comp`,
  `pack_samples.comp`). Plus de pixels RGB 8 bits côté CPU (perte de précision
  inacceptable en HDR 10 bits).
- `src/y4m.{h,cpp}`, `src/main.cpp` : streaming y4m stdin → stdout, cadence de
  sortie exacte (rationnelle), détection de coupes (score façon `scdet`) et des
  trames figées, recopie octet pour octet des trames d'origine.
- Plugin TensorRT (1.4.0) : `src/trt_plugin_abi.h` (interface C de référence, copiée dans le dépôt
  muxiveo-plugins), `src/trt_backend.{h,cpp}` (chargement du plugin, tampons Vulkan exportés vers CUDA,
  repli), `src/cuda_driver.{h,cpp}` (pilote CUDA chargé à l'exécution), `src/shaders/trt_pack.comp`.
- Moteur hybride (1.3.0) : `src/shaders/mc_*.comp` (pyramide de luminance, recherche
  bilatérale par blocs, propagation, OBMC, décision par région) et `src/nvof.{h,cpp}`
  (flux optique matériel NVIDIA, pilote chargé à l'exécution).

## Mettre à jour l'amont

1. ncnn : changer `MUXIVEO_RIFE_NCNN_TAG` dans `CMakeLists.txt` (vérifier que l'ancre du correctif
   `cmake/patch_ncnn.cmake` existe toujours), recompiler, relancer
   `tests/native/test_muxiveo_rife.py` (aller-retour de conversion + vérité terrain).
2. Modèles : changer `commit` dans `models.json`, recalculer les `sha256`
   (`sha256sum`), vérifier le `padding` attendu dans `model_padding()` (`src/main.cpp`).
3. Shaders `warp*` : comparer avec l'amont, reporter les correctifs.
4. En-tête NVOF : seulement si une nouvelle version de l'API est nécessaire (`NV_OF_API_VERSION`) ;
   vérifier la disposition de `NV_OF_CUDA_API_FUNCTION_LIST` dans `src/nvof.cpp`.
