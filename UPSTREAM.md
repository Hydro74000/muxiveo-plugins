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

## Fichiers repris

| Fichier | Origine | Modifications Muxiveo |
|---|---|---|
| `src/warp.cpp`, `src/rife_ops.h` | TNTwise `src/` | pipeline `warp_pack8` retiré (ncnn ≥ 2025 n'a plus de chemin Vulkan pack8 : `afpvec8` supprimé) |
| `src/shaders/warp.comp`, `warp_pack4.comp` | TNTwise `src/` | aucune |
| `src/shaders/rife_v4_timestep.comp` | TNTwise `src/` | aucune |
| `cmake/generate_shader_comp_header.cmake` | TNTwise `src/` | octets castés en `(char)` : shaders UTF-8 (commentaires accentués) acceptés |

## Code propre à Muxiveo

- `src/engine.{h,cpp}` : moteur RIFE v4 réécrit à partir de `RIFE::process_v4`
  (chemin non-TTA). Entrées/sorties YUV brutes 8–16 bits ; conversion YUV ↔ RGB,
  normalisation et quantification sur le GPU (`yuv_to_rgb.comp`, `rgb_to_yuv.comp`,
  `pack_samples.comp`). Plus de pixels RGB 8 bits côté CPU (perte de précision
  inacceptable en HDR 10 bits).
- `src/y4m.{h,cpp}`, `src/main.cpp` : streaming y4m stdin → stdout, cadence de
  sortie exacte (rationnelle), détection de coupes (score façon `scdet`) et des
  trames figées, recopie octet pour octet des trames d'origine.

## Mettre à jour l'amont

1. ncnn : changer `MUXIVEO_RIFE_NCNN_TAG` dans `CMakeLists.txt`, recompiler, relancer
   `tests/native/test_muxiveo_rife.py` (aller-retour de conversion + vérité terrain).
2. Modèles : changer `commit` dans `models.json`, recalculer les `sha256`
   (`sha256sum`), vérifier le `padding` attendu dans `model_padding()` (`src/main.cpp`).
3. Shaders `warp*` : comparer avec l'amont, reporter les correctifs.
