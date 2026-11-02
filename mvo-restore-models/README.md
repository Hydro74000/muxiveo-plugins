# mvo-restore-models — Modèles du débruitage MVO-R2

Graphes du débruiteur **MVO-R2** de Muxiveo (filtre « Débruitage » du panneau Encodage), exécutés par le moteur
`muxiveo-rife` (option `--restore`, 1.12.0) en Vulkan sur toutes les machines, ou par l'extension
[`mvo-rife-trt`](../mvo-rife-trt/README.md) (TensorRT) sur les cartes NVIDIA. Ils sont livrés dans l'archive de
l'extension [`mvo-rife`](../mvo-rife/README.md) (`restore-models/<modèle>/`) : rien à installer à part.

| Modèle | Rôle |
|---|---|
| `r2` | débruitage temporel sur une fenêtre de 5 images recalées, flux optique à demi-définition (défaut) |
| `r2q` | mode rapide : mêmes réseaux, flux optique au quart de la définition (poids réappris) |

Chaque modèle contient deux graphes : `r2_flow` (flux optique entre l'image courante et ses voisines) et `r2_body`
(image débruitée à partir des voisines recalées), au format ncnn (`.ncnn.param`, `.ncnn.bin`) pour Vulkan et ONNX
(`.onnx`, hauteur et largeur dynamiques) pour TensorRT ; `model.json` indique la définition du flux quand elle
diffère du défaut.

## Organisation

- `catalog.json` : source de vérité (version, modèles, fichiers, tailles, SHA-256, noms d'asset).
- Une nouvelle version des poids porte un nouveau nom de modèle (caches TensorRT propres).
- Release `mvo-restore-models-vX.Y.Z` (jamais « latest »), cumulative : tous les modèles du catalogue,
  `manifest.json` (SHA-256 de chaque fichier) et licence.

## Publier une version

```bash
python mvo-restore-models/scripts/package.py <dossier des modèles> dist/mvo-restore-models
gh release create mvo-restore-models-vX.Y.Z dist/mvo-restore-models/* --latest=false --title "mvo-restore-models X.Y.Z"
```

Puis reporter les sommes dans `mvo-rife/restore_models.json` (fichiers livrés avec muxiveo-rife).

## Licence

Poids et graphes produits par Muxiveo, licence MIT : `LICENSES/MIT-mvo-restore-models.txt`.
