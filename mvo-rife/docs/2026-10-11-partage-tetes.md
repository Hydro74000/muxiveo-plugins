# Partage des têtes entre branches — prototype du 11 octobre 2026

Le cache de la 1.7.1 distingue les deux branches du graphe. Lorsque B devient A,
il recalcule donc les caractéristiques de cette même source. Ce prototype permet
leur réutilisation si les opérations, paramètres et poids décodés sont identiques.
Il conserve les clés de source, miroir et réseau normal/demi-résolution, ainsi que
la réserve de VRAM et le repli existants. L'option reste désactivée par défaut.

Un graphe inconnu ou des poids différents conserve les caches séparés. La preuve
porte sur l'instantané des poids effectivement chargé ; elle ne dépend pas du nom
du modèle. Aucun poids n'est modifié. Les graphes v4.15 compatibles, Light et
l'Ultra existant peuvent en bénéficier. v4.6 ne possède pas ces têtes indépendantes.

## Mesures ciblées

RTX 4070 Ti SUPER, Linux, Vulkan avec NVOF automatique, six répétitions alternées
après amorçage. Référence : moteur 1.7.1 avec son cache déjà activé. Temps de
traitement natif, hors initialisation et encodage final ; ce n'est pas le benchmark
exhaustif de fin de chantier.

| Qualité avec cache | Conversion | Référence médiane | Prototype médian | Gain de débit | Plages référence / prototype |
|---|---|---:|---:|---:|---|
| 1920×800 | ×2 | 1,005 s | 0,945 s | +6,35 % | 0,99–1,01 / 0,94–0,96 s |
| 1920×800 | ×2,5 | 1,560 s | 1,500 s | +4,00 % | 1,56–1,57 / 1,49–1,51 s |
| 3840×1600 | ×2 | 12,185 s | 11,315 s | +7,69 % | 12,06–12,29 / 11,28–11,35 s |
| 3840×1600 | ×2,5 | 18,905 s | 18,110 s | +4,39 % | 18,81–19,08 / 18,02–18,33 s |

Les témoins 3840×1600 en ×2,5 donnent −0,96 % pour Équilibré et −0,51 % pour
Qualité sans cache, avec des plages recouvrantes. Aucun gain n'est attribué à ces
chemins ; leurs petites variations restent à confirmer avant intégration.

## Validation et limites

- Suite native Linux NVIDIA : 172 tests réussis. Contrôle TensorRT complémentaire :
  17 tests réussis, couvrant les 15 tests ignorés dans la première passe.
- Preuve CPU : paramètres différents, poids différents, NaN, lecture incomplète,
  mauvais type de poids et graphes non reconnus refusent le partage.
- Contrôle natif : modifier seulement la seconde tête désactive effectivement le
  partage et conserve la sortie de l'inférence sans cache.
- 54 des 56 comparaisons initiales sont strictement identiques, avec ×2, ×2,5,
  23,976→59,94 et 24→60, UHD, TTA 2/4/8, Ultra, Light et réserve VRAM forcée.
- Deux cas 3840×1600 en ×2,5 ont des écarts d'au plus un code en 10 bits. Une
  variation existe aussi entre deux rendus de référence Équilibré : 190 valeurs
  sur 829 440 000. Cela demande une analyse distincte, sans déclarer ces cas
  identiques ou conclure automatiquement à une absence de régression.
- Sur une séquence native 3840×2160 à 24→60, Équilibré et Qualité donnent des
  images et métriques strictement identiques entre deux références et deux
  prototypes, y compris les pires blocs mesurés.

État : recherche, non livré. Aucun gain qualitatif supplémentaire n'est démontré.
Les contrôles Windows, macOS et autres GPU restent à réaliser pour ce changement.
La réserve finale des futurs apprentissages n'a pas été consommée.
