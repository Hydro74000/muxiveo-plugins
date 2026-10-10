# Plugins Muxiveo

Extensions facultatives de [Muxiveo](https://github.com/Hydro74000/muxiveo), téléchargées à la demande par
l'application (page Extensions) sur les seules machines compatibles. Muxiveo fonctionne entièrement sans
elles.

## Plugins

| Plugin | Rôle | Plates-formes | État |
|---|---|---|---|
| [`mvo-rife`](mvo-rife/README.md) — Interpolation d'images | Moteur MVO-RIFE (muxiveo-rife : RIFE v4 Vulkan, moteur hybride, sélecteur appris), modèles et préréglages | Linux x86_64, Windows x64, macOS arm64 ; GPU Vulkan | 1.7.0 |
| [`mvo-rife-models`](mvo-rife-models/README.md) — Modèles d'interpolation | Modèles RIFE affinés par Muxiveo (ncnn pour muxiveo-rife, PyTorch pour l'export ONNX) | toutes | 1.0.0 |
| [`mvo-rife-trt`](mvo-rife-trt/README.md) — Accélération NVIDIA (TensorRT) | Inférence RIFE de MVO-RIFE sur les Tensor Cores (TensorRT for RTX) ; repli automatique sur Vulkan | Linux x86_64, Windows x64 ; GPU NVIDIA Turing ou plus récent | 1.1.0 |
| [`mvo-fel`](mvo-fel/README.md) — Reconstruction Dolby Vision FEL | Reconstruction BL + EL + RPU avant encodage, bibliothèque CPU chargée par Muxiveo | Cibles : Linux x86_64, Windows x64, macOS arm64 | En développement, non publié |

## Organisation

- Un dossier par plugin (`<plugin>/`), avec ses sources, son build et sa documentation.
- Releases taguées `<plugin>-vX.Y.Z` (jamais « latest ») ; chaque archive contient un `manifest.json`
  (version, contrat ou interface, modèles, SHA-256 des fichiers).
- Flux des versions : release `extensions-feed`, un `<plugin>.json` par plugin mis à jour par sa CI
  (`scripts/update_feed.py`). Muxiveo y choisit la version la plus récente compatible avec lui (contrat de
  `mvo-rife`, interface de `mvo-rife-trt`), avec une version minimale épinglée en repli, puis vérifie les sommes
  SHA-256 avant installation.
- L'interface C entre muxiveo-rife et `mvo-rife-trt` est `mvo-rife/src/trt_plugin_abi.h`, incluse directement
  par le plugin.

## Licences

Le code de ce dépôt est sous licence MIT (`LICENSE`), sauf indication contraire : le code C++ de
[`mvo-fel`](mvo-fel/NOTICES.md), adapté de libplacebo, est sous LGPL-2.1-or-later.
Les plugins peuvent embarquer des composants tiers sous
leur propre licence (par exemple NVIDIA TensorRT for RTX) : leurs notices sont livrées dans chaque plugin et
présentées à l'installation.
