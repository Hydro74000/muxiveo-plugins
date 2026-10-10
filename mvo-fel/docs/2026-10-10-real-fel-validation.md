# Validation locale sur une source FEL réelle

Date : 10 octobre 2026. Source fournie par l'utilisateur : remux Moana UHD,
conservé intact. Les extraits et résultats sont hors dépôt ; aucun contenu du
film ne fait partie du plugin ou des données redistribuées.

## Source et découpe

Le record annonce P7/BL+EL+RPU ; libdovi confirme FEL depuis le RPU original.
BL : HEVC 10 bits, 3840×2160 ; EL : 1920×1080 ; cadence 24000/1001.
Trois points demandés : 480, 1200 et 3600 secondes, avec environ une seconde
par extrait. Le premier démarre au CRA à 479,521 s et contient 25 images.

Une découpe naïve en stream copy conservait deux RASL antérieures au CRA :
63 RPU pour 61 images décodables. Le helper de test cherche le CRA avec ffprobe,
puis exclut les paquets dont le PTS est antérieur au début choisi (`noise` BSF).
L'extrait final contient un RPU par image. Cette préparation concerne les
extraits de test ; elle ne modifie ni le film original ni le workflow d'encodage.

## Comparaison numérique

Le moteur CPU produit RGB PQ 16 bits, sans Display Management. L'outil facultatif
`fel_reference_file` compare les images 0, 8 et 16 de chaque extrait, tous pixels
et trois canaux, après normalisation PQ 12 bits. Aucune tolérance n'a été relevée.

| Oracle | Extrait | Maximum observé, codes PQ12 | Résultat face au seuil 2 |
|---|---|---:|---|
| `kernel64` | ≈ 480 s | 0,065505 | Réussi |
| `kernel64` | ≈ 1200 s | 0,065928 | Réussi |
| `kernel64` | ≈ 3600 s | 0,059388 | Réussi |
| `gpu32` | ≈ 480 s | 2,278650 | Échec strict |

`gpu32` conserve les shaders de rééchantillonnage/mapping/NLQ/couleurs libplacebo
avec FBO float32. `kernel64` évalue les noyaux Spline16 par la fonction officielle
`pl_filter_sample` en double, indépendamment du sampler du moteur ; les shaders
de mapping/NLQ/couleurs restent ceux de libplacebo. Le résiduel est centré avant
son transfert float32, avec un offset NLQ nul équivalent, pour préserver son
signe sous un ulp de l'offset original.

Le premier essai GPU atteignait 3,494 codes. La reconstruction amplifiait parfois
un ulp de rééchantillonnage autour du zéro EL à cause du terme discontinu
`sign(d)*threshold`. Le moteur ignore désormais les écarts limités à 32 epsilon
double à ce zéro ; un test vérifie que les petits résiduels significatifs restent
appliqués. Cela ne constitue ni une analyse de bruit ni une détection de faux FEL.

Les différences restantes face au GPU apparaissent près de cette discontinuité,
où ses interpolations et la représentation float32 changent le signe d'un
résiduel minuscule. L'accord de l'oracle `kernel64` ne valide **pas** le critère
strict `gpu32`, qui reste ouvert. La variante GPU centrée essayée pendant le
diagnostic atteignait encore 2,185 ; elle n'a pas remplacé l'oracle GPU brut.

### Backend GPU corrigé

Le backend de production a ensuite remplacé son interpolation float32 de l'EL
par une interpolation centrée et compensée. Les instructions NLQ/mapping et
couleurs restent celles de libplacebo. Sur les mêmes trois scènes (neuf images
par carte), les maxima sont 0,312428 PQ12 sur AMD intégré et 1,374690 sur RTX
face au CPU ; 0,309435 et 1,359730 face à `kernel64`. Toutes ces comparaisons
respectent le seuil de 2 sans l'augmenter. Cela ne réécrit pas le résultat de
l'oracle brut `gpu32` ci-dessus. L'[investigation GPU](2026-10-10-performance-gpu.md)
détaille la cause, la correction et l'attestation du paquet prototype Linux.

Les six workflows réels décrits ci-dessous ont aussi été rejoués avec cette
bibliothèque statique GPU : six réussites, P8.1 sans instructions de résiduel,
horodatages et HDR10/SDR vérifiés, sans repli BL. La lecture sur matériel DV
reste à faire.

Une comparaison de signal avec le même mapping et le résiduel désactivé donne
des différences non nulles (jusqu'à environ 65 codes PQ12 sur une grille de pas
32). Cela vérifie que l'EL contribue aux pixels ; cela ne mesure pas un gain
perceptuel et ne pilote aucune décision automatique dans Muxiveo.

## Workflows et métadonnées

Tests dans Muxiveo : `tests/integration/test_encode_fel_real.py`, opt-in par
`MUXIVEO_TEST_FEL_SOURCE` et `MUXIVEO_TEST_FEL_LIBRARY`.

- x265 vers Dolby Vision P8.1, HDR10 et SDR ; comparaison BL seul.
- FFmpeg/NVENC et NVEncC vers P8.1, avec les règles de géométrie DV existantes.
- 25 images en sortie et écart d'horodatage maximal d'un tick Matroska (1 ms).
- Record P8.1 : BL/RPU présents, EL absente, compatibilité HDR10 1.
- RPU réellement extrait des sorties : sans EL, `disable_residual_flag=true`.
- HDR10 statique vérifié dans le flux : mastering BT.2020 conservé, MaxCLL/MaxFALL
  estimés depuis les L1 du RPU original (109/10 pour cet extrait), plutôt que
  recopiés du BL (356/165).
- Décodage complet des sorties sans traitement Dolby Vision ; aucune reprise BL
  silencieuse pour les tentatives demandant la reconstruction.

Ces contrôles et la comparaison BL/FEL ne remplacent pas une lecture Dolby
Vision sur téléviseur ou lecteur compatible, qui n'a pas été faite ici.

## Première mesure de coût

Ryzen 7 7800X3D (8 cœurs / 16 threads), RTX 4070 Ti SUPER, budget Muxiveo
12 threads, reconstruction CPU. Extrait de 25 images UHD, un seul passage par
configuration ; durée du workflow complet, préparation et assemblage inclus.

| Parcours | Durée |
|---|---:|
| x265 ultrafast, BL seul | 1,98 s |
| x265 ultrafast, FEL vers HDR | 26,89 s |
| x265 ultrafast, FEL vers P8.1 | 28,99 s |
| FFmpeg/NVENC, FEL vers P8.1 | 28,46 s |
| NVEncC, FEL vers P8.1 | 29,63 s |

Le premier couple correspond à environ +1255 % sur cet extrait et ce preset
rapide. Ce chiffre n'est ni une moyenne ni une estimation pour tous les films,
machines et presets. La reconstruction est actuellement le facteur dominant ;
les optimisations CPU restent nécessaires. Aucun double encodage compressé ni
intermédiaire vidéo décompressé complet n'explique ce coût.

La fonctionnalité reste désactivée par défaut, y compris en P5/P7.

Une [comparaison appairée avec trois passages](2026-10-10-fel-workflow-benchmark.md)
complète cette première mesure : RPU conservé dans les deux parcours, x265
medium CRF 18 et NVEncC default CQP 18:20:22. Sur 73 images, les médianes sont
14,28 → 83,50 s pour x265 et 5,93 → 80,19 s pour NVEncC.
