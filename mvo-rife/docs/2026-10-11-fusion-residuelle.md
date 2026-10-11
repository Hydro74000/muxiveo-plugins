# Fusion résiduelle — prototype du 11 octobre 2026

Dans les graphes v4.6 et v4.15, 32 additions résiduelles sont immédiatement
suivies d’une activation. Le prototype fusionne chaque paire en une seule passe
Vulkan : il économise un lancement et l’écriture/lecture d’un tampon intermédiaire.
Les poids, les canaux et les étapes d’estimation du mouvement restent identiques.

La transformation se fait en mémoire au chargement. Elle exige une addition à
deux entrées, un seul consommateur et une activation reconnue, sans paramètre
inconnu. Les graphes non reconnus restent inchangés. Une forme incompatible avec
le shader utilise les deux couches ncnn natives, avec leur broadcasting.
Le partage optionnel des têtes reste indépendant ; cette fusion bénéficie aussi
à v4.6, au cache désactivé et aux modes compatibles dont l’Ultra existant.

## Conservation de la précision

L’addition native écrit son résultat en FP16 avant l’activation. Supprimer cet
arrondi change le calcul : le premier prototype a été refusé pour cette raison.
La version corrigée reproduit explicitement l’arrondi IEEE 754 au plus proche,
avec égalités au pair, par opérations entières. Le chemin FP32 reste en FP32.
Une simple conversion aller-retour ne suffit pas : le compilateur/pilote pouvait
l’éliminer dans l’essai initial. Les sous-normaux et dépassements sont également
vérifiés par comparaison avec les couches natives.

## Premières mesures isolées

Linux, RTX 4070 Ti SUPER, Vulkan, NVOF automatique, 3840×1600, 36 images source.
Quatre répétitions alternées après amorçage. Référence : prototype avec partage
des têtes ; candidat : ce même prototype avec la fusion résiduelle.
Temps du traitement natif, hors initialisation et encodage final.

| Mode | Cadence | Médiane référence → fusion | Gain de débit | Comparaison des pixels |
|---|---|---|---:|---|
| Équilibré | ×2 | 3,455 → 3,325 s | +3,91 % | Identiques |
| Équilibré | ×2,5 | 6,350 → 6,095 s | +4,18 % | 294 valeurs différentes, maximum 1 code 10 bits ; analyse ouverte |
| Qualité | ×2 | 5,175 → 5,040 s | +2,68 % | Identiques ; plages de temps recouvrantes |
| Qualité | ×2,5 | 8,575 → 8,445 s | +1,54 % | Identiques |

Ces pourcentages ne s’additionnent pas aux gains du partage des têtes. Une mesure
combinée contre la version livrée est nécessaire.

## Validation en cours

- 214 tests du moteur passent sur Linux/NVIDIA, compléments TensorRT compris.
- 19 tests de transformation vérifient les refus, consommateurs et ordre des poids.
- Seize tests natifs comparent addition/activation fusionnées et séparées en
  FP16/FP32 avec quatre pentes d’activation, valeurs négatives et broadcasting,
  sur une conversion 24→60 en 10 bits. Ces 16 contrôles renforcés passent.
- Le contrôle élémentaire FP16/FP32 et six plages d’exposants reproduit les
  résultats natifs ; les rendus courts v4.6/v4.15 sont identiques.
- Comparaison combinée contre la 1.7.1 : 22/24 sorties identiques. Les deux
  cas 3840×1600 ×2,5 ont des écarts maximum d’un code 10 bits (Équilibré :
  265 valeurs ; Qualité : 597 sur 829 440 000). Analyse encore ouverte.
- Référence native, mesures alternées et autres GPU/OS encore à terminer. Aucun gain qualitatif supplémentaire démontré.

État : recherche, non livré. Les modèles installés ne sont pas modifiés.

## Correction après contrôle Windows physique

Sur GTX 1070, les 30 comparaisons des modes compatibles en ×2/×2,5 sont
identiques, y compris les contrôles 10 bits BT.2020. Sur Intel UHD 630, le
premier rendu Équilibré FP16 diverge. Le diagnostic isole la fusion : remettre
les additions et activations séparées dans le même binaire conserve exactement
la 1.7.1 pour v4.6/v4.15 ; les chemins FP32 sont également identiques.

La divergence FP16 atteint 14 codes 8 bits dans le cas v4.6 testé, et 2 avec
v4.15. Cette variante est refusée. Les tests élémentaires RGB passent pourtant :
ils ne remplacent pas les rendus complets sur matériel réel.

Le chemin fusionné FP16 est donc limité à NVIDIA, dont les contrôles natifs
passent. Les autres constructeurs conservent les deux opérations ncnn natives ;
ils peuvent toujours bénéficier du partage vérifié des têtes. La fusion FP32
reste disponible. Cette protection doit encore être vérifiée dans le paquet
Windows reconstruit et sur AMD avant adoption. Le diagnostic précis du pilote
Intel reste ouvert ; aucun mode d’arrondi différent n’est supposé sans preuve.

## Validation du correctif et gain cumulé

La garde du commit `3a1ce68` passe les contrôles sur matériel réel : Linux/NVIDIA
et AMD (27 tests ciblés chacun, 18 comparaisons AMD), Windows/GTX 1070 et
Intel UHD 630 (60 comparaisons identiques et 27 tests ciblés par GPU). La CI
`38130144582` passe les trois compilations : Linux 214 tests/16 ignorés,
Windows 58/172, macOS 83/147. macOS reste sans contrôle matériel Metal.

Les deux optimisations combinées sont mesurées contre la 1.7.1 avec cache activé
des deux côtés, 96 images source en 3840×1600, RTX 4070 Ti SUPER, Vulkan/NVOF auto,
six répétitions alternées après amorçage, traitement natif hors chargement/encodage :

| Mode | Cadence | Médiane 1.7.1 → candidat | Gain de débit |
|---|---|---|---:|
| Équilibré | ×2 | 8,250 → 8,010 s | +3,00 % |
| Équilibré | ×2,5 | 13,070 → 12,695 s | +2,95 % |
| Qualité | ×2 | 12,175 → 11,115 s | +9,54 % |
| Qualité | ×2,5 | 18,855 → 17,915 s | +5,25 % |

Le contrôle ciblé de la garde a des plages de temps recouvrantes, sans coût
établi : Équilibré 12,63–12,72 → 12,58–12,72 s, Qualité
17,83–17,89 → 17,76–18,10 s. Variations médianes : −0,16 % et −0,28 %.

Les séquences 3840×2160 avec images natives connues conservent pixels et
métriques moyens/pires blocs en ×2,5 (24→60) et ×2 complémentaire (30→60).
Le paquet Linux CI passe le contrôle natif ×2,5 et 32 encodages Muxiveo
FFmpeg/NVEncC : ×2/×2,5, cadences entières/fractionnaires, Équilibré/Qualité/
Ultra/Light, cadence, chronologie, décodage complet et audio copié conformes.

Les écarts d’un code sur l’autre séquence 3840×1600 ×2,5 restent conservés.
RIFE seul est identique dans les deux familles ; la référence hybride varie
également sur certains rendus. Les ablations n’isolent pas encore une cause
unique. Aucune amélioration qualitative supplémentaire n’est revendiquée.
Les poids installés et la réserve finale d’apprentissage restent inchangés.
