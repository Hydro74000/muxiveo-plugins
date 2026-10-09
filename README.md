# Plugins Muxiveo

Extensions facultatives de [Muxiveo](https://github.com/Hydro74000/muxiveo), téléchargées à la demande par
l'application (Paramètres > Extensions) sur les seules machines compatibles. Muxiveo fonctionne entièrement sans
elles.

## Plugins

| Plugin | Rôle | Plates-formes | État |
|---|---|---|---|
| [`mvo-rife-models`](mvo-rife-models/README.md) — Modèles d'interpolation | Modèles RIFE affinés par Muxiveo (ncnn pour muxiveo-rife, PyTorch pour l'export ONNX) | toutes | 1.0.0 |
| [`mvo-rife-trt`](mvo-rife-trt/README.md) — Accélération NVIDIA (TensorRT) | Inférence RIFE de MVO-RIFE sur les Tensor Cores (TensorRT for RTX) ; repli automatique sur Vulkan | Linux x86_64, Windows x64 ; GPU NVIDIA Turing ou plus récent | 1.0.0 |

## Organisation

- Un dossier par plugin (`<plugin>/`), avec ses sources, son build et sa documentation.
- Releases taguées `<plugin>-vX.Y.Z` (jamais « latest ») ; chaque release publie un `manifest.json`
  (version, ABI, version minimale de muxiveo-rife, SHA-256 des fichiers, licences).
- Muxiveo épingle la version de chaque plugin et vérifie les sommes SHA-256 avant installation.
- L'interface C entre muxiveo-rife et ses plugins est définie dans le dépôt Muxiveo
  (`native/muxiveo-rife`) ; ce dépôt en garde une copie, contrôlée par la CI.

## Licences

Le code de ce dépôt est sous licence MIT (`LICENSE`). Les plugins peuvent embarquer des composants tiers sous
leur propre licence (par exemple NVIDIA TensorRT for RTX) : leurs notices sont livrées dans chaque plugin et
présentées à l'installation.
