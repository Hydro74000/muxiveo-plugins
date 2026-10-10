# Deux optimisations FEL : pipe allégé et raccordement GPU direct

## Périmètre

Le changement concerne FELMerge et son entrée dans l'encodeur. Les réglages
de compression, l'interpolation RIFE, les autres filtres et le muxage restent
ceux du workflow. Une seule compression vidéo finale a lieu ; BL et EL sont
deux décodages, pas deux encodages.

## 1. Pipe NUT allégé

Le renderer écrit désormais trois plans R16 GBR sur GPU, puis les télécharge
directement dans le paquet NUT. Cela supprime le rapatriement de l'alpha RGBA,
le déinterleave CPU et la copie complète de l'image RGB dans un second tampon.
Une image UHD passe de 66,4 Mo à 49,8 Mo rapatriés. Les transferts des plans
sont soumis de façon asynchrone et attendus avant de libérer le paquet, même
sur erreur. La mémoire et le nombre d'images restent bornés.

Le parcours CPU écrit aussi directement dans le paquet. Aucun codec aval
n'est remplacé. Cette voie s'applique à x265, NVEncC, aux autres encodeurs,
aux filtres, au SDR et à RIFE existants.

Le journal expose des chronomètres natifs : ouverture, lecture, appels aux
décodeurs BL/EL, reconstruction/rapatriement, NUT et écriture du pipe. Le temps
du pipe est inclus dans NUT. Les décodeurs ont des workers : leurs frontières
d'appel ne représentent pas tout leur coût CPU. Ces compteurs ne doivent pas
être additionnés pour attribuer une durée de bout en bout.

Sur le même extrait de 73 images, trois répétitions par parcours, RTX 4070 Ti
SUPER et Ryzen 7800X3D, 12 threads :

| Workflow DV P8.1 | BL seul, médiane | FEL avec pipe allégé | Surcoût |
| --- | ---: | ---: | ---: |
| x265 medium CRF 18 | 13,619 s | 15,696 s | +15,3 % |
| NVEncC default CQP 18:20:22 | 5,688 s | 10,380 s | +82,5 % |

La précédente série FEL RTX mesurait respectivement 17,660 et 12,823 s.
La nouvelle série est environ 11,1 % et 19,1 % plus courte ; les références
BL ont été remesurées, donc ces comparaisons historiques ne constituent pas
une attribution isolée au code de copie.

## 2. FEL → Vulkan → CUDA → FFmpeg/NVENC

Le plugin fournit une interface pull de couches BL/EL/RPU : une image par
appel, dans l'ordre de présentation, avec vidange et validation strictes.
Un filtre source `mvo_fel`, compilé dans un FFmpeg **privé et épinglé**, appelle
ce moteur et reconstruit dans des AVFrame Vulkan. Le contexte Vulkan est
emprunté à FFmpeg, sans second périphérique logique indépendant.

Le filtre libplacebo convertit les pixels PQ RGB en P010 avec dithering et,
si nécessaire, recadre à dimensions identiques. Display Management,
réapplication du RPU, détection de pic et film grain sont désactivés. Le filtre
`mvo_fel_cuda` utilise le transfert GPU Vulkan/CUDA de FFmpeg : il existe une
copie **sur la carte**, mais aucune image reconstruite ne revient en RAM et
aucun NUT ne circule. Le décodage des couches reste CPU avec upload des couches.

L'encodeur `hevc_nvenc` n'est pas réécrit. Le workflow conserve son preset,
son contrôle de débit, sa qualité et son profil. Un éventuel `-pix_fmt` de
transport devient CUDA ; le contenu du pool reste P010 10 bits. L'injection
P8.1, le recadrage L5, les gardes de comptage et le muxage utilisent toujours
le parcours existant après l'encodage.

Conditions de ce prototype : Linux, NVIDIA avec pilote ≥ 570, même UUID
physique que l'encodeur, HEVC 10 bits, aucun RIFE/SDR/resize/filtre additionnel
non pris en charge. Une identité ambiguë en multi-GPU conserve NUT. Le choix
manuel CPU ou iGPU conserve également NUT. Auto reste dédié > iGPU > CPU ;
il ne déplace ni l'encodeur ni RIFE pour permettre le transport direct.

NVEncC n'accepte pas ces AVFrame d'un autre processus par son pipe standard.
Il conserve donc la première optimisation. Une voie directe NVEncC demanderait
un lecteur GPU intégré à cet outil ; un shader seul ne remplit pas ce rôle.
Les autres OS et constructeurs conservent le parcours portable, sans annoncer
une interopérabilité non testée.

La supervision normale du processus continue de gérer progression et
annulation. Les erreurs identifiées par le filtre FEL déclenchent le repli
complet BL existant ; les erreurs de source, de stockage, d'encodeur et les
annulations ne sont pas assimilées à une erreur FEL.

## Comparaison directe à réglages identiques

Même extrait de 73 images, sortie 3840×1600 suivant le recadrage DV actuel,
HEVC/NVENC p5 CQ 18, conservation du RPU P8.1, 12 threads, sans RIFE,
trois répétitions par parcours :

| Entrée FFmpeg/NVENC | Médiane du workflow | Écart avec BL |
| --- | ---: | ---: |
| BL seul | 6,413 s | référence |
| FEL en NUT allégé | 10,756 s | +67,7 % |
| FEL en GPU direct | 6,828 s | +6,5 % |

Le parcours direct est 36,5 % plus court que le parcours NUT FEL dans cette
série. Chaque résultat vérifie 73 images, P8.1 sans EL, RPU présent, PQ 10 bits
et PTS strictement croissants, avec une erreur maximale de 0,5 ms.

Le FFmpeg direct provient de la révision du plugin ; les deux autres parcours
utilisent le FFmpeg configuré. Les arguments d'encodage sont conservés, mais
ceci mesure le prototype complet, pas une expérience isolant chaque différence
entre révisions FFmpeg. Les coûts fixes dominent une séquence de trois secondes :
ces pourcentages ne prédisent pas la durée d'un film complet.

Une vérification plus longue, **721 images / environ 30 secondes**, un passage
par parcours sur le build final, donne :

| Entrée FFmpeg/NVENC | Workflow complet | Écart avec BL |
| --- | ---: | ---: |
| BL seul | 21,124 s | référence |
| FEL en NUT allégé | 61,242 s | +189,9 % |
| FEL en GPU direct | 24,161 s | +14,4 % |

Sur cette séquence, le direct réduit la durée FEL de **60,5 %** face à NUT,
et produit 29,84 images/s en incluant préparation/injection/muxage. Les 721
images, le RPU P8.1 et les horodatages sont vérifiés dans les trois parcours.
C'est une vérification de débit soutenu sur cette machine, avec une seule
répétition ; elle ne remplace pas une mesure longue répétée sur d'autres films.

## Fidélité et paquet

Les [mesures brutes](2026-10-10-fel-transport-measurements.json) conservent les
durées, comptages, débits, géométries et horodatages, sans les médias.

La bibliothèque, le renderer indépendant et le binaire direct exacts ont été
validés sur les images 0/8/16 de trois scènes, sur AMD iGPU et NVIDIA RTX.
Les empreintes GBRP16LE du NUT et du filtre Vulkan sont identiques à celles du
renderer de référence. Maxima PQ12 : CPU 0,312428 AMD / 1,374690 NVIDIA ; oracle
kernel64 0,309435 / 1,359730. Seuil inchangé : 2.

Les tests natifs couvrent aussi pull/EOF/B-frames/cadence rationnelle/arrêt ;
les tests du workflow couvrent RPU, cadences variables, chemins spéciaux,
RIFE, HDR10/SDR, deux passes et EL amputée avec reprise BL. Les tests de paquet
refusent une bibliothèque, un filtre, des sources ou des pixels non attestés.

Le paquet Linux expérimental contient `libmvo_fel.so` et, facultativement,
`ffmpeg-fel`, statiques hors libc/libm/pilotes. Les composants FFmpeg ne sont
pas ajoutés au PATH. Les sources, configurations, révisions et notices sont
redistribuées, y compris nv-codec-headers MIT. Les paquets CPU Linux/Windows/
macOS ont passé la CI ; le transport GPU direct reste un prototype Linux.

Pour activer la voie directe, installer un paquet qui l'annonce dans son
manifeste. Pour une expérience locale, fournir explicitement `direct_ffmpeg`
à `FelEngine` ou `--direct-ffmpeg` au benchmark. L'option FEL reste décochée
par défaut sur tous les profils.
