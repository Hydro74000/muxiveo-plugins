# Performances Qualité et portée aux autres modes — 10 octobre 2026

## Périmètre et décision

Optimiser le préréglage Qualité (`hybrid`, `rife-v4.15-mvo1`) avant l’Ultra adaptatif. Le ×2,5, soit 23,976 → 59,94 et 24 → 60, est prioritaire : l’utilisateur prévoit plus de 80 % des usages à cette cadence. Une qualité supérieure est souhaitable ; aucune baisse n’est acceptée. L’identité binaire sert ici de preuve de conservation pour les changements de calcul redondant, sans constituer une obligation pour les chantiers d’amélioration d’image.

Ordre retenu : implémenter 1/2 ; mesurer 5/6 et retenir celles qui apportent un gain même faible sans perte ; prévoir 3 en option avec contrôle de VRAM ; évaluer 4 ensuite. Le benchmark exhaustif reste réservé à la version finale.

## Calculs conservés et calculs supprimés

**1 — Candidat MC direct.** La recherche de mouvement, les vecteurs, les échantillonnages, les pondérations et le calcul RGB fp32 restent identiques. Le chemin MC seul écrit directement dans le format du réseau, avec le même arrondi. Le sélecteur n’utilise pas les cartes q/edge de l’ancienne règle hybride : ce chemin évite leur calcul, cinq plans intermédiaires fp32 et la passe de recopie RGB. Le chemin de l’ancienne règle conserve ses cartes et sa fusion.

**2 — Invariants par paire.** La cohérence aller/retour NVOF et les cartes incohérence/amplitude ne dépendent pas de t. Les statistiques de texture, d’écart A/B et de ces cartes sont également invariantes. Elles sont réutilisées ; la luminance R1 et les désaccords entre candidats sont recalculés à chaque temps. Les identités des sources et la présence de NV protègent le cache des statistiques ; tout nouveau flux invalide les cartes.

Les grandes cartes sont libérées avant le réseau de la paire suivante. Leur conservation entre deux sorties demande une réserve de VRAM : `heapBudget - heapUsage`, au moins 1 Gio et 20 % du budget. Si l’extension de mesure n’est pas disponible ou que la réserve manque, le moteur revient au recalcul des cartes. Le budget total n’est jamais assimilé à la mémoire libre. Ces caches représentent quatre plans fp32 pleine résolution ; leur coût entre deux inférences est réel.

**5 — Ordonnancement, prototype écarté.** Lancer NVOF avant les uploads et ne fournir le callback de jonction que lorsqu’un résultat est attendu réduit certaines attentes. La première mesure courte en 4K ×2,5 indiquait 9,895 → 9,835 s, mais le contrôle sur un autre extrait plus long donnait 20,79 → 20,86 s. Le gain ne se confirme donc pas. Le prototype est archivé dans le banc privé et l’ordonnancement courant est conservé.

**6 — Entrées partagées TensorRT.** Chaque backend mémorise séparément les identités des sources déjà copiées dans ses deux entrées. Seules les entrées modifiées sont recopiées. Une identité nulle impose la copie et invalide la réutilisation suivante. Le réseau normal et le réseau à flux demi-résolution ont des caches distincts. Les barrières nécessaires aux accès Vulkan/CUDA sont conservées. Le contrôle long en 4K ×2,5 confirme un gain faible, régulier sur six répétitions mesurées ; en 1080p le gain est trop faible pour conclure. Les chiffres de ce backend restent dans le banc privé.

Extension en 1.7.1 : les variantes TTA portent aussi leur identité de source et de miroir. L'affectation des deux tampons maximise le nombre d'images déjà présentes : l'image commune aux deux paires consécutives reste en place, même à ×2. Ultra et le sens temporel inverse du TTA relient les tampons dans l'ordre opposé ; aucune recopie ni allocation supplémentaire n'est nécessaire. Changer de miroir impose de remplir les entrées correspondantes. L'ABI du plugin 1.1.0 est conservée.

## Validation et méthode de mesure

Binaire de référence 1.7.0 archivé avant modification, mêmes modèles et poids, GPU RTX 4070 Ti SUPER. Les mesures ciblées utilisent des sources décodées 10 bits en 1920 × 800 et 3840 × 1600, sorties vers un puits, initialisation exclue du temps `done`. Une passe d’échauffement puis quatre passes alternées permettent de comparer médianes et dispersion. Il s’agit de mesures de l’interpolation seule, pas d’un débit d’encodage complet.

Le premier contrôle de 1/2 a comparé 18 configurations : dimensions alignées/non alignées, 8/10 bits, ×2, ×3, cadence cible fractionnaire, NVOF activé/désactivé, UHD, TTA, Ultra, MC seul et ancienne règle hybride. Toutes les sorties comparées étaient identiques au binaire de référence.

Sur l’extrait réel 4K ×2,5, deux rendus du binaire de référence varient déjà légèrement : 144 échantillons sans NV et 335 avec NV, sur 829 440 000, d’un seul niveau 10 bits. Le candidat 1/2/5/6 de comparaison diffère respectivement sur 248 et 472 échantillons, également d’un seul niveau. Il ne faut donc pas assimiler une empreinte différente à une dégradation. Ce contrôle situe l’ordre de grandeur des écarts, sans établir une amélioration de qualité. Le binaire retenu après retrait de 5 et l’option 3 ont ensuite été contrôlés séparément face aux instants natifs connus, comme décrit ci-dessous.

Des tests de régression vérifient les instants communs entre ×2 et ×4, l’indépendance des paires et, spécialement pour ×2,5, l’égalité avec un ×5 échantillonné une image sur deux. Ce dernier contrôle exerce successivement les temps 0,4/0,8 puis 0,2/0,6. Il couvre les deux cadences usuelles, avec et sans NVOF et avec les deux backends.

La suite native complète passe : **149 tests**, avec Vulkan, NVOF et le plugin TensorRT disponibles, avant ajout du réglage CLI de réserve. Après cet ajout et celui du cas fp32, les **14 tests ciblés** de capacités et de cache optionnel passent également. Le contrôle privé de repli demande une réserve supérieure au budget : aucune tête n’est réutilisée et la sortie reste identique ; avec la réserve par défaut, le même contrôle observe 12 réutilisations. Il vérifie la décision de repli sans saturer physiquement la VRAM. Les tests d’option couvrent aussi UHD, fp32, TTA et Ultra.

Après généralisation aux autres modes et intégration de MC, la suite native 1.7.1 passe intégralement : **165 tests**, avec Vulkan et le plugin TensorRT 1.1.0. Elle ajoute TTA 4/8, les autres modèles RIFE dont Light, et les entrées inversées d’Ultra. Côté Muxiveo, **154 tests ciblés** de raccordement/interpolation/extensions passent ; Ruff et Mypy sont verts.

L’affectation des entrées TensorRT qui conserve l’image commune entre paires passe ensuite les **28 tests ciblés** TensorRT et ×2,5. Les comparaisons privées de rendus et d’instants natifs sont répétées avec cette affectation finale.

Un contrôle privé compare ensuite la 1.7.0, la 1.7.1 et la 1.7.1 avec cache dans **11 configurations pour chacun des deux backends** : Rapide, Équilibré, Qualité, Ultra, Light, ancienne règle hybride, MC seul, Qualité Fast et TTA 2/4/8. Les 66 rendus donnent des sorties identiques par configuration. En Vulkan, les compteurs confirment notamment 26 réutilisations en Ultra, 6 en Light et 92 en TTA 8 sur le petit contrôle fractionnaire ; ces nombres constatent l’activité du cache, sans mesurer son gain de vitesse.

## Cache optionnel et évaluation de la recherche MC

**3 — Têtes RIFE optionnelles.** `--feature-cache`, désactivé par défaut, conserve les têtes indépendantes de t (`154`, `166`) dans ncnn. Le moteur vérifie que chaque tête dépend seulement de sa source à travers trois convolutions et une déconvolution, puis vérifie sa forme pleine résolution à huit canaux (v4.15) ou quatre (Light). Les clés distinguent source, côté du graphe, miroir et réseau normal/demi-résolution : Ultra et TTA 2/4/8 réutilisent effectivement leurs caractéristiques. Les sources sorties de la paire courante sont libérées. Aucun partage de poids entre côtés n’est supposé. Une première inférence sans cache amorce l’allocateur, puis la décision demande de garder 1 Gio et 20 % du budget en réserve, plus le coût estimé des têtes manquantes. `--feature-cache-reserve <MiB>` permet d’augmenter la réserve minimale. Sans mesure fiable ou sans marge, le moteur recalcule normalement. v4.6 ne possède pas ces têtes indépendantes. L’extension des têtes au backend TensorRT demande un travail spécifique sur le graphe et son interface d’entrées.

Premier bilan ciblé en Vulkan, avant généralisation des caches à Ultra/TTA/Light et intégration de la recherche MC, 36 images sources, 23,976 → 59,94, médiane des quatre répétitions après échauffement :

| Dimensions | Référence 1.7.0 | 1/2/6, cache des têtes désactivé | Cache des têtes activé | Gain de débit total avec cache |
|---|---:|---:|---:|---:|
| 1920 × 800 | 1,700 s | 1,695 s | 1,570 s | +8,3 % |
| 3840 × 2160 | 13,185 s | 13,020 s | 12,105 s | +8,9 % |

Sans l’option, le gain de débit est respectivement d’environ 0,3 % et 1,3 %. Avec l’option, le gain supplémentaire par rapport au binaire déjà optimisé est d’environ 8,0 % et 7,6 %. Les résultats portent sur ces extraits et sur l’interpolation seule ; ils ne prédisent pas le gain de bout en bout d’un encodage.

Un contrôle 4K à ×2,5 utilise 21 instants natifs connus, dont 16 images générées. Pour chaque backend séparément, la référence répétée, le binaire retenu et celui avec cache donnent exactement les mêmes images, le même PSNR-Y moyen et le même pire 1 % des blocs 64 × 64 ; les sources sont conservées. Ce jeu sert à vérifier la conservation du calcul, pas à démontrer la qualité générale du modèle ni une supériorité entre backends. La validation matérielle couvre Linux et la RTX 4070 Ti SUPER ; le benchmark exhaustif et les autres GPU restent des travaux distincts.

Le même contrôle natif est ensuite répété avec la 1.7.1 et son cache activé, en Qualité puis en Ultra : sorties et métriques identiques à la 1.7.0 dans chaque backend, y compris après intégration de la recherche MC et des caches par miroir/côté.

**4 — Recherche MC.** Le prédicteur/médiane est calculé une fois par groupe. Les modes de recherche à petite liste dédupliquent uniquement les vecteurs de représentation binaire identique, en conservant le premier candidat, l’ordre des ex æquo et l’ordre des sommes. La recherche exhaustive garde son parcours courant. Le shader ajoute une barrière et 248 octets de mémoire partagée ; aucun raccourci approché n’est utilisé.

Les 18 configurations synthétiques donnent les mêmes sorties. Le contrôle natif 4K ×2,5 donne aussi exactement les mêmes images et métriques, dans chaque backend séparément. Sur le contrôle Vulkan MC seul en 3840 × 2160, médiane 1,110 → 1,015 s, soit environ +9,4 % de débit. En Qualité complet, les premiers contrôles courts donnent 1,690 → 1,675 s en 1920 × 800 et 13,055 → 13,000 s en 3840 × 2160. Sur le second extrait, 96 sources en 3840 × 1600 et six répétitions mesurées après échauffement, médiane 20,985 → 20,815 s, soit environ +0,8 %. Cependant les plages se chevauchent fortement (20,85–21,82 s contre 20,43–21,54 s), avec deux ralentissements sur six comparaisons appariées. Les comparaisons de cette piste utilisent le même ordonnancement expérimental 5 dans leurs deux binaires, afin d’isoler la modification du shader.

Le contrôle Équilibré en 2160p donne 8,320 → 8,295 s en médiane, soit environ +0,3 %, avec une comparaison plus lente, deux plus rapides et une égale. L’ancienne règle hybride donne 13,155 → 13,100 s, soit environ +0,4 %, avec quatre comparaisons mesurées toutes plus rapides.

**Décision révisée après demande utilisateur : intégrer 4 dans tous les chemins MC, y compris les préréglages hybrides et Ultra.** Le gain de MC seul est net ; les gains globaux médians sont positifs sur les contrôles Qualité, Équilibré et ancienne règle, mais restent modestes et variables. Leur faible amplitude n’est pas un motif de rejet : les calculs supprimés sont strictement redondants et les contrôles de conservation passent. Ces mesures ne démontrent pas un gain garanti sur chaque scène. Le benchmark exhaustif final devra quantifier leur contribution cumulée, y compris avec les têtes en cache.

## Portée des optimisations

| Optimisation | Modes concernés |
|---|---|
| Candidat MC direct | MC seul, Équilibré, Qualité, Ultra ; ancienne règle hybride conservée |
| Cartes NV et statistiques par paire | Équilibré, Qualité, Ultra avec sélecteur |
| Recherche MC | MC seul, Équilibré, Qualité, Ultra, ancienne règle hybride |
| Entrées partagées TensorRT | Tous les modèles compatibles avec le plugin, UHD et TTA compris ; inversion sans recopie pour Ultra/TTA |
| Têtes Vulkan optionnelles | RIFE v4.15, Qualité, Ultra, Light ; UHD et TTA 2/4/8 ; repli pour les graphes non compatibles |

## État du chantier

Version : mvo-rife **1.7.1**. Contrat, ABI TensorRT, modèles et poids inchangés. La case facultative « Cache des caractéristiques » de Muxiveo transmet `--feature-cache` aux pipelines FFmpeg et NVEncC ; elle est conservée dans les réglages/préréglages et reste disponible en Ultra. Un moteur trop ancien provoque une demande de mise à jour seulement si cette option est activée. Le benchmark exhaustif reste prévu à la fin des travaux d’interpolation.
