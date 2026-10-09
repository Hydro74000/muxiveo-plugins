# Modèles Muxiveo

Modèles d'interpolation produits par Muxiveo, indépendants de toute extension : muxiveo-rife les utilise en
Vulkan sur toutes les cartes, l'extension `mvo-rife-trt` les convertit en ONNX pour TensorRT.

- `catalog.json` : source de vérité (modèle de base, fichiers ncnn et PyTorch, tailles, SHA-256).
- Nom : `<modèle de base>-mvo<N>` (ex. `rife-v4.15-mvo1`) ; une nouvelle version porte un nouveau nom (cache
  TensorRT et poids du sélecteur propres).
- Publication : release `models-<nom>` de ce dépôt, jamais « latest » ; fichiers à la racine de la release :
  `https://github.com/Hydro74000/muxiveo-plugins/releases/download/models-<nom>/<fichier>`.
- Modèles ncnn : graphe identique au modèle de base, seuls les poids changent (`--uhd` et TensorRT inchangés).

Consommateurs (sommes à garder identiques au catalogue) :

| Dépôt | Fichier | Usage |
|---|---|---|
| Muxiveo | `native/muxiveo-rife/models.json` | fichiers ncnn embarqués avec muxiveo-rife |
| muxiveo-plugins | `mvo-rife-trt/models/models.json` | poids PyTorch exportés en ONNX |
