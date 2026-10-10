# Benchmark du workflow avec et sans FEL

Mesure locale du 10 octobre 2026, sur le remux Moana fourni par l'utilisateur.
Le fichier original est resté intact. Les extraits, sorties et journaux sont
hors dépôt ; aucun contenu du film n'est redistribué.

## Protocole

- Même extrait pour tous les passages : CRA à 479,521 s, 73 images à 24000/1001,
  soit environ 3,045 s. La découpe retire les RASL antérieures au CRA.
- Trois passages par configuration, exécutés séquentiellement. Ordre BL/FEL
  inversé au deuxième passage. Les résultats centraux sont des médianes.
- Ancien parcours : `bake_dovi_fel=False`, BL seul avec copie du RPU converti
  vers P8.1. Nouveau parcours : même configuration avec `bake_dovi_fel=True`,
  reconstruction FEL confirmée, puis encodage et RPU P8.1 sans résiduel.
- HEVC 10 bits, DV conservé, un seul passage d'encodage. Sans audio, sous-titres,
  interpolation, tone mapping ni génération NFO. Pas d'autre job du benchmark
  exécuté en parallèle.
- Temps du workflow complet : préparation DV, reconstruction éventuelle,
  encodage, injection et assemblage final. Découpe de l'extrait et contrôle
  externe des sorties exclus des deux chronométrages.
- Contrôles après chaque passage : 73 images, P8.1, RPU présent, EL absente,
  compatibilité HDR10, PQ 10 bits. Reconstruction réussie pour les six passages
  avec FEL, aucun repli BL. Dimensions identiques entre BL/FEL pour chaque
  encodeur : 3840×2160 pour x265 ; 3840×1600 pour NVEncC, selon le recadrage DV
  existant. Les deux encodeurs ne sont pas comparés en qualité entre eux.

Matériel : Ryzen 7 7800X3D, RTX 4070 Ti SUPER, pilote 615.71.09. Réglage de threads
transmis au workflow : 12 ; pools internes des encodeurs conservés. RAM buffer
Muxiveo désactivé. FFmpeg système 8.1.2, x265 4.1+1, NVEncC 9.36 r4153.
Moteur natif Release (`-O3 -DNDEBUG`), reconstruction CPU.

SHA-256 de la bibliothèque testée :
`9837ce4f22a410a34cf10907057f5df92373d381e898801442c65ed5e1a4a8d4`.

## Résultats

| Configuration | Sans FEL, médiane | Avec FEL, médiane | Temps ajouté | Hausse de durée | Rapport de durée |
|---|---:|---:|---:|---:|---:|
| x265 medium, CRF 18 | 14,28 s | 83,50 s | 69,22 s | +484,84 % | ×5,85 |
| NVEncC default, CQP 18:20:22 | 5,93 s | 80,19 s | 74,26 s | +1252,11 % | ×13,52 |

Calcul : `hausse = 100 × (médiane_avec / médiane_sans − 1)`.

Temps bruts, en secondes :

| Configuration | Passage 1 | Passage 2 | Passage 3 |
|---|---:|---:|---:|
| x265 sans FEL | 14,654 | 14,278 | 13,927 |
| x265 avec FEL | 83,669 | 83,220 | 83,502 |
| NVEncC sans FEL | 6,009 | 5,899 | 5,931 |
| NVEncC avec FEL | 80,187 | 79,769 | 80,201 |

Débit global, calculé par `73 / temps_médian`, préparation comprise :

| Configuration | Sans FEL | Avec FEL |
|---|---:|---:|
| x265 medium | 5,11 i/s | 0,874 i/s |
| NVEncC default | 12,31 i/s | 0,910 i/s |

Les durées avec FEL sont proches malgré les encodeurs différents, ce qui
confirme le poids du producteur CPU actuel. Le parcours NVEncC sans FEL utilise
le décodage matériel `avcuvid` ; avec FEL, il reçoit le flux reconstruit puis
préparé en pipe. Le producteur ne réalise aucun encodage compressé supplémentaire.

## Portée et reproduction

Ces chiffres concernent cet extrait court et ces réglages. Le démarrage des
processus, le remplissage des buffers et la préparation DV pèsent davantage sur
trois secondes que sur un film complet. Les i/s ci-dessus sont ceux du workflow,
pas la vitesse instantanée de l'encodeur. Aucune extrapolation au film complet
ni à d'autres presets ou machines n'est validée par cette mesure.

La [validation réelle](2026-10-10-real-fel-validation.md) décrit les contrôles
numériques et les limites de qualité encore ouvertes. Le benchmark mesure le
temps, sans modifier ces critères.

Depuis le dépôt Muxiveo :

```sh
QT_QPA_PLATFORM=offscreen python3 -m scripts.benchmark_fel_workflow /chemin/film.mkv \
  --library /chemin/libmvo_fel.so --work-dir /chemin/benchmarks \
  --start 480 --duration 3 --runs 3 --threads 12
```

`--codecs libx265` permet de mesurer sur une machine sans NVIDIA. Chaque
exécution crée un dossier neuf ; `results.json` contient les temps bruts,
médianes et informations source, et chaque passage possède son journal de
workflow. Pour cette mesure locale, `machine.json` consigne également
l'environnement utilisé.

Résultats locaux de cette exécution :
`/var/mnt/Drive2/muxiveo-fel-dev/real-moana/fel-benchmark-v59izo3p/`.
