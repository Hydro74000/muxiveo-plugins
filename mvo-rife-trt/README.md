# mvo-rife-trt — Accélération NVIDIA (TensorRT) pour Muxiveo

Plugin facultatif de [Muxiveo](https://github.com/Hydro74000/muxiveo) : l'interpolation d'images MVO-RIFE
(outil natif `muxiveo-rife`) exécute son réseau RIFE avec **NVIDIA TensorRT for RTX** sur les Tensor Cores des
cartes NVIDIA, au lieu du calcul Vulkan générique. Les images interpolées sont les mêmes à l'œil ; le traitement
est nettement plus rapide.

Muxiveo propose ce plugin uniquement sur les machines compatibles et l'installe, le met à jour et le supprime
lui-même (Paramètres > Extensions). Sans lui, ou au moindre problème, l'interpolation continue en Vulkan.

## Compatibilité

- GPU NVIDIA **Turing (RTX 20xx, GTX 16xx) ou plus récent** ; pilote NVIDIA **575 ou plus récent**.
- Linux x86-64 (glibc 2.28 ou plus récente) et Windows 10/11 x64.
- muxiveo-rife 1.4.0 ou plus récent (interface 1).
- Préréglages couverts : Rapide, Équilibré, Qualité et Light (modèles RIFE v4.6, v4.15 et v4.15-lite, avec ou
  sans le mode Fast).

Au premier usage d'un modèle, le moteur TensorRT est préparé pour votre carte puis mis en cache (quelques
secondes) ; les utilisations suivantes démarrent immédiatement.

## Contenu d'une release

`mvo-rife-trt-<version>-<plate-forme>.tar.gz` (Linux) ou `.zip` (Windows) :

| Fichier | Rôle |
|---|---|
| `libmvo_rife_trt.so` / `mvo_rife_trt.dll` | plugin (code de ce dépôt, MIT) |
| `libtensorrt_rtx.so.1`, `libtensorrt_onnxparser_rtx.so.1` / `tensorrt_rtx_1_6.dll`, `tensorrt_onnxparser_rtx_1_6.dll` | NVIDIA TensorRT for RTX (licence NVIDIA) |
| `models/*.onnx` | modèles RIFE exportés des poids officiels (Practical-RIFE, MIT) |
| `LICENSES/` | licences et mentions des composants |
| `manifest.json` | version, interface, empreintes SHA-256 de chaque fichier |

## Construire

Prérequis : CMake ≥ 3.20, compilateur C++17, Python 3. Le SDK TensorRT for RTX est téléchargé à la version
épinglée (`sdk.json`, SHA-256 vérifié) ; aucun CUDA Toolkit n'est nécessaire.

```
SDK=$(python3 scripts/fetch_sdk.py linux-x86_64 build/sdk)
cmake -S . -B build/plugin -G Ninja -DTRTRTX_ROOT="$SDK" && cmake --build build/plugin
python3 models/export_onnx.py build/weights build/models          # torch + onnx requis
python3 scripts/package.py linux-x86_64 build/plugin/libmvo_rife_trt.so "$SDK" build/models build/dist
python3 scripts/selftest.py build/dist/mvo-rife-trt-*-linux-x86_64 build/models/rife-v4.6.onnx build/cache   # GPU NVIDIA
```

L'interface avec muxiveo-rife (`include/trt_plugin_abi.h`) est une copie de la référence du dépôt Muxiveo ;
`scripts/check_abi.py` (CI) vérifie qu'elles sont identiques.

## Publier

Incrémenter `project(mvo-rife-trt VERSION …)` dans `CMakeLists.txt`, puis pousser le tag
`mvo-rife-trt-vX.Y.Z` : la CI construit les deux plates-formes et publie la release. Muxiveo épingle ensuite cette
version (`core/version.py`).

## Licences

- Code de ce dépôt : MIT.
- NVIDIA TensorRT for RTX : licence NVIDIA (`LICENSES/NVIDIA-TensorRT-RTX-License.txt`), redistribué intégré à
  Muxiveo, pour un usage sur GPU NVIDIA ; mentions des composants tiers de TensorRT dans
  `LICENSES/NVIDIA-TensorRT-RTX-Acknowledgements.txt`.
- Architectures des modèles reprises de [vs-rife](https://github.com/HolyWu/vs-rife) (MIT), poids
  [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT).
