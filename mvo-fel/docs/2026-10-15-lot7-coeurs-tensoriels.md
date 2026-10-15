# Lot 7 — cœurs tensoriels : décision mesurée

Le plan de continuation (§4, lot 7) conditionne ce lot à deux critères :

1. un résultat **identique bit à bit** au noyau entier ALU ;
2. un gain **mesurable de bout en bout**.

Sinon, l'abandon est documenté. Ce document est cette documentation.

## Ce que les cœurs tensoriels pourraient accélérer

Seul le rééchantillonnage Spline16 relève d'un produit matriciel :

- chroma BL (2×) ;
- résiduels EL (luma 2×, chroma 4×).

Le mapping polynomial/MMR, la NLQ et les matrices de couleur sont des calculs par pixel trop petits. Les fonctions PQ sont transcendantes et ne s'y prêtent pas.

### Exactitude

L'exactitude est atteignable :

- Les poids de la luma EL sont entiers (/40 et /320). Les codes centrés (|c| ≤ 1023) et les poids (≤ 273) tiennent exactement en FP16. Le premier étage séparable s'accumule exactement en FP32.
- Le second étage fait monter les intermédiaires jusqu'à 53 000, au-delà de la plage entière exacte du FP16. Il faudrait découper en limbs int8 ou FP16, au prix de plusieurs produits par étage.

## Mesures (15/10/2026, plugin après le lot 6 fusionné)

Extrait de 300 images UHD de Toy Story 5, sortie 10 bits finale écrite par descripteur dans un puits.

| GPU | Coopérative matrix | Débit | GPU/image | Composition (rééchantillonnages inclus) | Goulot |
|---|---|---:|---:|---:|---|
| RTX 4070 Ti SUPER | `VK_KHR` + `VK_NV` | 70–88 i/s | 1,03 ms | 0,74 ms | écriture du pipe |
| iGPU AMD Raphael (RDNA2) | non | 22,8 i/s | 34,4 ms | 29 ms | GPU |

- **Sur la RTX**, toute la composition pèse 0,74 ms sur un budget de 11 à 14 ms par image. Le rendu attend l'écriture : `render_blocked_s` est supérieur au temps GPU cumulé. Un noyau tensoriel infiniment rapide libérerait au plus 0,74 ms de GPU par image, sans aucun gain de bout en bout.
- Avec RIFE sur la même carte, l'interpolation domine (plusieurs dizaines de ms par image). Le gain resterait inférieur à 2 %.
- **Sur l'iGPU AMD**, le seul cas limité par le GPU, l'extension est absente : RDNA2 n'a pas de WMMA. Le lot ne peut donc rien y apporter.

## Décision

Le lot n'est pas implémenté.

Le gain de bout en bout est nul sur le matériel disponible qui expose l'extension. Il est impossible là où le GPU est le goulot. Ajouter un second noyau pour gagner 0,7 ms de GPU sur une carte inactive ajouterait du code et un risque de divergence (compilation `GL_KHR_cooperative_matrix`, découpage en limbs) sans contrepartie mesurable.

## À rouvrir si

- un GPU **limité par le calcul** expose la coopérative matrix : iGPU RDNA3/RDNA4, Intel Arc/Xe2 (XMX) ;
- ou la composition devient le goulot (sources 8K, compositions multiples).

Le critère reste le même : comparaison au noyau entier, bit à bit, sur les six scènes de l'attestation.

## Optimisation retenue à la place

La **composition fusionnée** (lot 6, audit) répond au même besoin sans dépendre du matériel :

- un seul dispatch ;
- aucune texture 4K float32 intermédiaire ;
- poids entiers exacts.

Gains mesurés :

- iGPU AMD : 41,1 → 34,4 ms par image ;
- RTX : 1,66 → 1,03 ms par image.

Précision face au CPU : 0,0625 code PQ12 au maximum sur 11 des 12 comparaisons.
