# Mesures de bout en bout (lot M)

16 octobre 2026. Lot M du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md). Il s'agit de savoir si le recouvrement ([P](2026-10-16-recouvrement.md)) ou les Tensor Cores (T) servent dans un vrai workflow, sur un GPU partagé avec l'encodeur.

## Protocole

- **Workflow.** Muxiveo complet (`scripts/benchmark_fel_workflow.py`), extrait de 10 s de Toy Story 5 (P7 FEL, 241 images UHD), FEL en mode Lent sur la RTX 4070 Ti SUPER, décodage Vulkan Video.
- **Mesures.** Deux passages par cas, médianes. La référence BL est le même workflow sans reconstruction FEL.
- **Images en vol.** 1 ou 2 (P).

## Résultats

| Encodeur (même GPU que le FEL pour NVEncC) | BL | FEL, 1 image en vol | FEL, 2 images en vol | Goulot (mesures internes du plugin) |
|---|---:|---:|---:|---|
| NVEncC HEVC 10 bits, preset par défaut | 16,0–16,2 s | 21,55 s (+33 %) | 21,45 s (+34 %) | lecture du pipe NUT par NVEncC (≈ 65 i/s) |
| NVEncC « standard HQ » (VBR 34 Mbit/s, 2 passes au quart, UHQ, lookahead 24, AQ temporelle) | 19,0 s | 24,81 s (+30 %) | 24,59 s (+29 %) | lecture du pipe par NVEncC |
| x265 medium (CPU) | 34,5–34,7 s | 34,47 s (+0 %) | 34,81 s (+0 %) | x265 (≈ 7 i/s) |
| RIFE 24 → 59,94 i/s (Équilibré, Vulkan) + NVEncC HQ, 603 images en sortie | 54,3–54,8 s | 58,79 s (+8 %) | 59,25 s (+8 %) | RIFE (le plugin attend 31 à 37 s sur 37) |

**Composition FEL.** Elle reste à 7,8–8 ms par image avec NVEncC, même en UHQ : aucune concurrence mesurable. Avec RIFE sur le même GPU, elle passe à 15 ms sans devenir le goulot.

**Surcoût du FEL.** Sur ces extraits de 10 s, il tient surtout à des coûts fixes par job : ouverture de 1,1 à 2,2 s (vérification Vulkan Video sur 96 images, auto-test, compilation des shaders), plus la préparation FEL. Sur un film entier, ils deviennent négligeables.

**Régime établi.** Le FEL produit 61 à 65 i/s avec 1 image en vol, autant que ce que NVEncC HQ absorbe. Avec 2 images en vol, il monte vers 78 i/s et ne peut plus freiner l'encodeur.

## Gain maximal des Tensor Cores (T)

Les Tensor Cores ne pourraient reprendre que l'accumulation des coefficients du MMR. Les monômes, l'EOTF, les divisions et le classement restent scalaires.

Pour mesurer cette accumulation, elle est calculée une seconde fois, multipliée par un zéro que le compilateur ne peut pas deviner. Les codes restent identiques et aucune reprise n'apparaît. Résultat sur Ryan (contenu MMR) : composition 9,07 → 10,35–10,41 ms, soit **1,3 ms par image (14 %)**. Le gain est nul sur le contenu à polynômes (Toy Story 5, Moana).

Dans le pire cas mesuré (RIFE et NVEncC sur le même GPU), ce gain vaudrait au plus 241 × 1,3 ms ≈ 0,3 s sur 54 s, soit moins de 1 %.

## Décisions

- **T est classé.** Il est réouvrable si un workflow devient limité par le calcul FEL.
- **P est conservé, 2 images en vol par défaut.** Il donne une marge en régime établi face à NVEncC et accélère le mode Rapide sur iGPU.
- **Option d'ABI `mvo_fel_set_render_workers` (Q).** Muxiveo pourra passer à 1 image en vol quand l'interpolation RIFE tourne sur le même GPU : aucun gain dans ce cas, et environ 150 Mo de VRAM économisés.
