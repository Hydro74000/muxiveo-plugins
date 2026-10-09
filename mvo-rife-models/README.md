# mvo-rife-models — Modèles d'interpolation Muxiveo

Modèles RIFE produits par Muxiveo, indépendants de toute carte graphique : muxiveo-rife les utilise en Vulkan
sur toutes les machines, l'extension [`mvo-rife-trt`](../mvo-rife-trt/README.md) les convertit en ONNX pour
TensorRT.

| Modèle | Base | Utilisé par |
|---|---|---|
| `rife-v4.15-mvo1` | `rife-v4.15` (même architecture et même coût) | préréglages Qualité et Ultra de MVO-RIFE |

## Organisation

- `catalog.json` : source de vérité (version, modèles, fichiers ncnn et PyTorch, tailles, SHA-256, noms d'asset).
- Nom d'un modèle : `<modèle de base>-mvo<N>` ; une nouvelle version porte un nouveau nom (cache TensorRT et
  poids du sélecteur de muxiveo-rife propres).
- Modèles ncnn : graphe identique au modèle de base, seuls les poids changent (`--uhd` et TensorRT inchangés).
- Release `mvo-rife-models-vX.Y.Z` (jamais « latest »), cumulative : tous les modèles du catalogue,
  `manifest.json` (SHA-256 de chaque fichier) et licences.

## Publier une version

```bash
python mvo-rife-models/scripts/package.py <dossier des modèles> dist/mvo-rife-models
gh release create mvo-rife-models-vX.Y.Z dist/mvo-rife-models/* --latest=false --title "mvo-rife-models X.Y.Z"
```

Consommateurs, à mettre à jour avec la version (sommes identiques au catalogue) :

| Dépôt | Fichier | Usage |
|---|---|---|
| Muxiveo | `native/muxiveo-rife/models.json` | fichiers ncnn livrés avec muxiveo-rife |
| muxiveo-plugins | `mvo-rife-trt/models/models.json` | poids PyTorch exportés en ONNX |

## Licence

Poids dérivés de RIFE (Practical-RIFE, hzwer), licence MIT : `LICENSES/MIT-Practical-RIFE.txt`.
