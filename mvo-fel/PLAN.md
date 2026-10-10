# Reconstruction FEL — extension mvo-fel

Révision du 9 octobre 2026 : le moteur devient une extension facultative du
dépôt `Hydro74000/muxiveo-plugins`, suivant l'organisation de `mvo-rife` et
`mvo-rife-trt`. Cette décision remplace le moteur toujours installé, sa présence
dans `Muxiveo/native/fel` et sa compilation obligatoire par l'installeur de
l'application. Les exigences de reconstruction, de qualité et de repli restent
applicables.

Ce document décrit la cible. L'état effectivement réalisé figure à la fin.

Révision du 10 octobre 2026 : activation **uniquement manuelle**, y compris
pour les sources P5/P7. Le profil inspecté ne coche plus la case automatiquement.

## 1. Répartition entre les dépôts

| Dépôt / composant | Responsabilité |
|---|---|
| `muxiveo-plugins/mvo-fel` | Bibliothèque native, ABI C, FFmpeg/libdovi privés, calculs, tests natifs et références numériques, construction, notices, paquets et CI de l'extension. |
| `muxiveo/core/plugins.py` | Registre `mvo-fel`, compatibilité ABI/capacités, installation atomique, sélection de version, vérification des archives et suppression. |
| `muxiveo/core/version.py` | Version minimale validée de l'extension et contrat attendu ; épinglage après publication du plugin. |
| `muxiveo/core/fel/` | Adaptateur `ctypes`, découverte de l'extension installée, contexte d'exécution et producteur interne. Aucun calcul natif dupliqué. |
| `muxiveo/core/workflows/encode/` | Politique par piste, préparation du RPU, routage des images, métadonnées, supervision et repli complet BL. |
| `muxiveo/ui/` | Case HDR/Dolby Vision, persistance, état du moteur, page Extensions et aperçu. |
| `muxiveo/tests/` | Tests des décisions, profils, adaptation ABI, pipelines, annulation et reprise. |

Le moteur n'est ni un exécutable utilisateur à configurer, ni une dépendance de
VapourSynth. L'extension reste indépendante de RIFE et de TensorRT : elle peut
fonctionner sans ces extensions, et les précède lorsqu'elles sont utilisées.
La découverte dynamique de nouvelles familles d'extensions reste hors périmètre ;
le registre existant est étendu explicitement.

## 2. Installation et disponibilité

- Identifiant : `mvo-fel` ; nom affiché : « Reconstruction Dolby Vision FEL ».
- Plateformes visées : Linux x86_64, Windows x64, macOS arm64, comme les paquets
  actuellement distribués. Chemin CPU portable, sans GPU obligatoire.
- Installation, mise à jour et suppression par la page Extensions et les
  commandes `plugins` existantes. Aucune nouvelle commande CLI d'encodage.
- Installation dans le dossier utilisateur des extensions, avec manifeste,
  sommes SHA-256 et bascule atomique de `current.json` existants.
- Aucun téléchargement au lancement ou pendant un encodage. La page HDR offre
  un accès à Extensions si le moteur manque ; elle ne lance pas une installation.
- Une case cochée exprime une intention, même sans plugin. L'absence, l'échec de
  chargement, l'ABI incompatible ou une capacité manquante sont annoncés avant
  traitement : reconstruction indisponible, parcours BL avec avertissement.
- Le choix mémorisé n'est pas décoché automatiquement après ce repli. L'extension
  installée ultérieurement sera utilisée au lancement suivant.
- Au début du job, figer le chemin et la version vérifiés. Empêcher la suppression
  des fichiers d'une version utilisée jusqu'à libération de son contexte ; ne
  pas changer de bibliothèque entre deux passes ou entre deux pistes du job.
- Les paquets minimaux de Muxiveo fonctionnent sans le moteur. Les paquets
  all-inclusive pourront préinstaller la version validée selon le mécanisme
  existant de RIFE ; ils ne compileront pas une autre copie du moteur.
- L'installation de Muxiveo depuis les sources n'impose plus de compiler FEL.
  La construction depuis les sources est portée par `mvo-fel`, avec préparation
  automatique de ses dépendances pendant le build.

## 3. Interface et persistance Muxiveo

Ajouter sous la copie du RPU, dans **Encodage → HDR → Dolby Vision**, une case
ordinaire à deux états :

> Intégrer la couche FEL à l’image avant encodage

Infobulle :

> Reconstruit l’image Dolby Vision avec la couche FEL avant les filtres et
> l’encodage. Fonctionne aussi sans conserver le RPU. La présence d’un FEL est
> confirmée au lancement. Nécessite l’extension mvo-fel.

Ajouter `bake_dovi_fel: bool | None = None` à `VideoEncodeSettings` et
`EncodePreset` : absent/`None` = désactivé par défaut pour le moment ; `True` =
demande explicite ; `False` = désactivation explicite.

- Tous les profils, P5/P7 inclus : case décochée par défaut.
  P7 ne prouve pas FEL ; le niveau, par exemple `06` de `dvhe.07.06`, ne permet
  pas de distinguer MEL et FEL.
- Garder `None` tant que l'utilisateur n'a pas modifié la case. Préserver les
  choix explicites par piste lors des changements de sélection, codec, HDR,
  installation ou suppression de l'extension.
- La copie du RPU et le tone mapping ne changent pas ce choix. Tous les codecs
  de réencodage restent accessibles selon les règles existantes.
- En Copy, case inactive mais choix mémorisé ; bandeau : réencodage nécessaire.
- « Appliquer à toutes » propage le choix selon les règles actuelles. Une piste
  sans choix explicite reste désactivée.
- Capture/restauration, profils sauvegardés et constructeurs mono/multipistes
  transportent le champ. Les anciens profils restent désactivés.
- Aucun chemin temporaire, version installée, handle natif ou résultat de
  confirmation n'est sérialisé dans les profils.
- `dovi_policy.py` partage la décision entre interface, aperçu et workflow.
  Distinguer « demandée, à confirmer », « moteur indisponible : repli BL » et
  le parcours qui retire l'EL. La normalisation P8.1 n'annonce pas une perte
  inévitable du FEL lorsqu'une intégration des pixels est demandée.

## 4. Contrat natif et producteur

Créer une ABI C versionnée, chargée par `ctypes`, sans liaison à une version
précise de CPython. Bibliothèques principales : `libmvo_fel.so`, `mvo_fel.dll`,
`libmvo_fel.dylib`. Le manifeste et la bibliothèque annoncent la même ABI.

Le contrat expose :

1. Version ABI, version moteur et capacités ; aucune allocation lourde à la sonde.
2. Confirmation du RPU original par libdovi, avec diagnostic structuré.
3. Création d'un contexte pour source, index global de piste, cadence rationnelle
   éventuelle, budget de threads/mémoire et paramètres immuables.
4. Production NUT vers un consommateur à mémoire bornée, avec contre-pression.
5. Progression, annulation thread-safe et destruction déterministe.
6. Codes distinguant erreur de reconstruction, annulation, fermeture du
   consommateur et erreur d'entrée/sortie. Aucune exception C++ ne traverse l'ABI.

L'application ne doit pas transformer un pipe fermé par l'encodeur en erreur FEL
récupérable. Les buffers et chaînes exposés ont une durée de vie documentée ;
les allocations sont libérées par le composant qui les a créées.

Les dépendances FFmpeg/libdovi sont privées au paquet du plugin, épinglées et
chargées depuis celui-ci. Éviter les collisions de symboles/SONAME avec Qt et
les bibliothèques de la machine. Ne jamais charger au hasard un FFmpeg système
pour exécuter la reconstruction.

## 5. Reconstruction des images

- C++ avec CPU portable, parallélisme borné et SIMD sélectionné à l'exécution.
- FFmpeg privé avec `dovi_split` : démuxage de la seule piste sélectionnée et
  décodage HEVC BL/EL. Le FFmpeg configuré par l'utilisateur reste l'encodeur.
- libdovi pour lire le RPU original. Port CPU des calculs libplacebo : mapping
  polynomial/MMR, déquantification NLQ et conversions colorimétriques.
- Spline16 pour EL/chroma, positionnement explicite des échantillons, dimensions
  BL/EL différentes et validation contre une référence indépendante.
- Calcul haute précision, sortie RGB PQ `gbrp16le`. Aucun Display Management,
  trim d'écran ni plafonnement arbitraire de luminosité.
- Association stricte BL/EL/RPU dans l'ordre de présentation : B-frames, vidange
  des décodeurs, données RPU précédentes, PTS, départ non nul et cadence
  rationnelle. Une erreur interdit tout appariement approximatif.
- Flux NUT en pipe avec horodatages et signalisation BT.2020/PQ. Aucun
  intermédiaire vidéo décompressé complet. Le transport des indications couleur
  jusqu'au consommateur FFmpeg doit être vérifié, pas seulement demandé au muxer.
- Ordre : reconstruction → filtres/géométrie → tone mapping → interpolation →
  encodage. Dithering lorsque la précision est réduite. Aucune instruction Dolby
  de reconstruction déjà appliquée ne suit les images produites.
- Aucune détection de « faux FEL », aucun score de bruit ou d'utilité perceptuelle.
  Tout FEL confirmé est traité lorsque l'option est active.

## 6. Workflows Muxiveo

Étendre `PipelineCommand` et les superviseurs avec un producteur natif interne
en amont des processus. Réutiliser progression, annulation, supervision des
processus et politique de ressources ; ne pas déguiser le plugin en commande
shell.

- Sans interpolation : FFmpeg consomme NUT puis filtre/encode.
- Un seul encodage compressé final : décoder BL/EL, reconstruire les pixels et
  convertir le format dans le pipe ne constituent pas un second encodage avec
  pertes. Un mode deux passes refait les calculs pour chacune des deux passes.
- Aucune dépendance à NVIDIA : calcul CPU, puis encodeur logiciel ou matériel
  existant de Muxiveo (Intel/AMD/Apple compris selon les capacités de FFmpeg).
- Avec RIFE : FFmpeg prépare son entrée après reconstruction et filtres.
- NVEncC : NUT non compressé horodaté après reconstruction et filtres ; y4m
  lorsque RIFE est utilisé (règles CFR existantes). RPU fourni séparément par
  fichier, aucun second encodage compressé.
- Mono sans HDR dynamique : passer aussi par le parcours vidéo séparé.
- Multipiste : contexte par piste et respect du budget global, pas de partage
  mutable du décodeur entre pistes.
- Deux passes : recréer le producteur depuis la même source et les mêmes
  paramètres pour chaque passe.
- Aperçu des commandes : décrire l'étape interne et la confirmation différée,
  sans proposer une fausse commande shell copiable.
- Aperçu d'image : même moteur et même décision après confirmation ; afficher
  explicitement un aperçu BL si la reconstruction est indisponible.

## 7. Confirmation et métadonnées

Au lancement, si la reconstruction est demandée et le moteur disponible :

1. Extraire le RPU original de la piste choisie avec les helpers existants.
2. Le valider et confirmer FEL/MEL avec libdovi, sans heuristique MediaInfo ni
   analyse de l'utilité de l'EL.
3. Vérifier les ressources et préparer le contexte uniquement pour un FEL confirmé.

MEL ou profil sans EL : opération sans effet annoncée, parcours existant.
Confirmation impossible/RPU invalide : avertissement et BL. Source et RPU
originaux restent intacts ; la conversion P7 actuelle avec `--discard` ne doit
pas précéder la reconstruction.

Si le RPU est conservé :

- Préparer une copie P8.1 distincte par `dovi_tool` mode 2, supprimant les
  instructions de reconstruction appliquées.
- Réutiliser corrections L5, synchronisation d'interpolation et gardes de
  comptage ; les comptages seuls ne prouvent pas l'alignement.
- Donner ce RPU en fichier à NVEncC, même sans interpolation.
- Sortie signalée P8.1, BL/RPU présents, EL absente, compatibilité HDR10 ;
  vérifier les métadonnées réellement présentes dans le flux final.
- Préserver les règles de géométrie NVENC, profondeur et compatibilité DV.

HDR10 statique : les saisies manuelles priment. Estimer automatiquement
MaxCLL/MaxFALL depuis les L1 du RPU original, avec provenance « estimation » ;
ne pas reprendre aveuglément BL ou L6 qui peuvent décrire le fallback source.
Les métadonnées de mastering suivent les règles existantes.

HDR10+ : conserver le contrôle par sa case. Avec FEL reconstruit, avertir que
les métadonnées proviennent du signal source et peuvent nécessiter une nouvelle
analyse. Pas de régénération dans ce chantier.

## 8. Repli BL

Sur erreur de reconstruction récupérable, avertir avec piste et cause, arrêter
producteur/processus de cette tentative, nettoyer seulement ses intermédiaires
possédés, puis recommencer la piste entière sur le BL. Refaire les deux passes
si nécessaire et recalculer les métadonnées automatiques pour le parcours BL.

Ne jamais mélanger des images FEL et BL au sein d'une tentative. Garder les
pistes déjà terminées du job multipiste. Un seul repli est autorisé ; l'échec
du BL échoue le job. Une annulation ne relance rien. Les erreurs indépendantes
d'encodage, stockage ou muxage suivent le traitement habituel.

Journal et résultat final distinguent reconstruction réussie, opération sans
effet, moteur indisponible et reconstruction en échec suivie d'un repli BL.

## 9. Publication et licences

- Version du plugin dans son `CMakeLists.txt`, ABI dans son en-tête public.
- CI propre `.github/workflows/mvo-fel.yml`, construisant et testant les trois
  plateformes. Pas de dépendance du build FEL au build RIFE.
- Releases `mvo-fel-vX.Y.Z`, jamais « latest » du dépôt.
- Archive racine `mvo-fel-<version>-<plateforme>` ; manifeste contenant nom,
  version, ABI, capacités, plateforme, bibliothèque, révisions des dépendances
  et SHA-256 de tous les fichiers.
- Réutiliser `scripts/update_feed.py` pour `extensions-feed/mvo-fel.json`.
- Chaque paquet contient bibliothèque, dépendances nécessaires, notices et
  sources/offre de sources conformes aux licences. Livrer les scripts de build
  et les éléments nécessaires au remplacement/reliage des composants LGPL.
- Code C++ adapté de libplacebo et FFmpeg : LGPL-2.1-or-later ; libdovi : MIT.
  Code Python de Muxiveo : MIT. Aucun code GPL de FelBaker/DoViBaker intégré.
- Publier et valider le plugin avant de relever le plancher dans Muxiveo ou
  d'exiger cette release dans ses paquets all-inclusive. Pas de version fictive
  installable pendant le développement.

## 10. Vérification et ordre de réalisation

1. **Moteur et références numériques dans `mvo-fel`** : polynômes/MMR, NLQ,
   plages, chroma, matrices, RPU réutilisé, absence de DM ; comparaison libplacebo
   avec rééchantillonnage identique, erreur maximale de deux codes PQ 12 bits.
2. **ABI et distribution locale** : chargement sans GPU, refus d'ABI inconnue,
   capacités, dépendances isolées, corruption de manifeste, budgets, annulation,
   libération et test de chargement/reconstruction d'une archive installée.
3. **Producteur et workflows Muxiveo** : ordre des filtres, horodatages, B-frames,
   GOP multiples, départ non nul, cadences fractionnaires, absence de perte ou
   duplication, deux passes et reprise par piste.
4. **Interface et persistance** : défaut désactivé P5/P7/autres/inconnus, choix manuel,
   Copy, indépendance RPU/SDR, multipiste, appliquer à toutes, anciens profils,
   extension absente/installée/incompatible/supprimée sans perte du choix.
5. **Bout en bout** : HEVC logiciel/matériel, NVEncC, HDR10 sans RPU, SDR, RIFE,
   crop DV, deux passes ; erreurs EL tronquée, RPU invalide, désalignement,
   annulation, nettoyage ; régressions option désactivée/P5/P8/MEL/passthrough.
6. **Validation de livraison** : extrait FEL réel comparé au BL, lecture P8.1 et
   fallback HDR10. Le badge DV et les comptages ne suffisent pas. Ruff, Mypy et
   tests Muxiveo concernés au vert, tests natifs/paquets/CI multiplateforme au
   vert, puis publication plugin et épinglage dans l'application.

## État du chantier au 10 octobre 2026

- Implémenté : bibliothèque native et ABI C, dépendances privées statiques,
  décodage BL/EL, association stricte des images/RPU, reconstruction CPU avec
  chemin AVX2 sélectionné à l'exécution et sortie NUT bornée.
- Implémenté côté Muxiveo : producteur interne, supervision/annulation, préparation
  des métadonnées, reprise complète BL, interface/persistance, aperçu et registre
  Extensions. L'activation reste exclusivement explicite.
- Validé localement sur séquence synthétique : x265 HDR10/DV/SDR, deux passes,
  FFmpeg/NVENC, NVEncC, RIFE ×2, piste sélectionnée avec B-frames/départ non nul,
  et repli après suppression d'une image EL en milieu de flux.
- Référence indépendante libplacebo : erreurs maximales observées de 0,046 code
  PQ 12 bits pour le mapping et 0,411 pour les images avec rééchantillonnage,
  contre une limite de 2. Les intermédiaires de référence sont en float32 pour
  éviter la quantification FBO 16 bits du renderer par défaut. Ces mesures
  concernent les cas synthétiques du test, pas un corpus de films FEL.
- Média réel fourni : P7 FEL confirmé par libdovi, BL 3840×2160, EL 1920×1080,
  24000/1001 i/s. Neuf images à trois moments du film concordent à moins de
  0,066 code PQ12 avec l'oracle à noyaux libplacebo évalués en double. Correction
  de la stabilité du zéro NLQ après le rééchantillonnage ; test de régression ajouté.
- La référence GPU float32 reste distincte : maximum de 2,279 codes PQ12 sur
  le premier extrait, avec quelques franchissements du seuil près du zéro NLQ.
  Cet oracle brut reste conservé. Le backend GPU de production corrigé passe
  ensuite sous deux codes face au CPU et à l'oracle indépendant `kernel64`,
  sur les deux cartes et trois scènes ; cela ne réécrit pas ce résultat brut.
- Sur le premier extrait réel : x265 DV/HDR10/SDR, FFmpeg/NVENC, NVEncC et BL
  seul, 25 images, P8.1 effectivement injecté sans résiduel, fallback HDR décodable,
  horodatages conservés à un tick Matroska près. Lecture sur téléviseur DV non validée.
- Construction, paquet Linux autonome et tests de chargement/reconstruction
  réalisés localement. CI Linux x86_64/Windows x64/macOS arm64 écrite ; les deux
  autres plateformes et les encodeurs Intel/AMD/Apple restent à valider sur leur
  matériel. Benchmark appairé DV sur 73 images, trois passages : x265 medium
  14,28 → 83,50 s (+485 %) ; NVEncC default 5,93 → 80,19 s (+1252 %), sur
  Ryzen 7800X3D/RTX 4070 Ti SUPER. Aucune extrapolation générale au film complet
  ou aux autres presets. Le moteur CPU reste coûteux.
- Release générale encore conditionnée à la lecture Dolby Vision sur matériel
  compatible et à un corpus élargi. La première préversion `mvo-fel-v0.1.0`
  devient installable via le flux Extensions après validation de la CI
  multiplateforme et du paquet GPU portable ; voir la publication ci-dessous.

Le [rapport de validation réelle](docs/2026-10-10-real-fel-validation.md) contient
les conditions et limites de ces mesures.
Le [benchmark appairé](docs/2026-10-10-fel-workflow-benchmark.md) compare les
workflows avec et sans FEL à réglages identiques pour chaque encodeur.

### Reconstruction GPU et placements après mesure

Le [prototype Vulkan](docs/2026-10-10-performance-gpu.md) corrige la divergence
NLQ par interpolation EL centrée et compensée. Sur neuf images de trois scènes
par carte, maximum PQ12 face au CPU : 0,312428 AMD et 1,374690 NVIDIA ; face à
l'oracle indépendant : 0,309435 et 1,359730. Limite maintenue à 2. Tests natifs,
six workflows réels et tests d'interface/persistance passent localement.

Le moteur et ses dépendances FFmpeg/libdovi/libplacebo/glslang sont privés et
statiques. Le chargeur Vulkan est facultatif au chargement ; CPU reste disponible.
Une attestation SHA-256 exacte est requise pour empaqueter le prototype Linux
x86_64. Installation dans un dossier de test par le gestionnaire Extensions
vérifiée ; aucune installation utilisateur modifiée pendant ces validations.

L'interface propose Auto / GPU détectés / CPU, dynamique et enregistrée par
UUID. Le choix est indépendant de celui de l'encodeur ou du RPU et conservé
entre pistes/presets/Copy. Auto observe les identités aval, la mémoire, les
réservations concurrentes et la charge si disponible. Les indices CUDA,
Vulkan et RIFE ne sont jamais considérés comme identiques par défaut.

Les mesures distinguent x265 medium et NVEncC default, avec/sans RIFE. Le
benchmark [de placements](docs/2026-10-10-gpu-pipeline-matrix.md) couvre CPU,
iGPU et GPU dédié FEL, RIFE sur son GPU habituel, et relève les charges
calcul/encodeur/décodeur. Une carte libre n'est pas automatiquement plus rapide :
sans RIFE, NVEncC + FEL RTX prend 12,79 s contre 16,31 s avec FEL iGPU.

La [réduction des copies et le transport direct](docs/2026-10-10-fel-transport-optimizations.md)
sont implémentés. NVEncC conserve le NUT allégé ; FFmpeg/NVENC peut recevoir
des images GPU sans retour RGB en RAM, sur Linux/NVIDIA compatible, sans RIFE.
La fidélité du filtre direct est attestée par les pixels qu'il produit réellement.
Windows, macOS/MoltenVK, Intel et plusieurs cartes dédiées restent à mesurer
sur leurs matériels. Les tests de politique simulés ne remplacent pas ces
benchmarks physiques. L'activation reste exclusivement explicite et désactivée
par défaut pour tous les profils.

### Première préversion installable

Le candidat du commit `1030741`, construit par la
[CI 38065556166](https://github.com/Hydro74000/muxiveo-plugins/actions/runs/38065556166),
porte les binaires Linux sur glibc 2.28. Sa bibliothèque et son FFmpeg privé
ont passé de nouveau la validation numérique sur les deux GPU physiques et
trois scènes réelles, avec vérification des images des deux transports.
La [validation du paquet portable](docs/2026-10-10-portable-prerelease.md)
documente les empreintes et les limites.

La préversion `mvo-fel-v0.1.0` livre Linux x86_64 CPU/Vulkan/direct NVIDIA,
Windows x86_64 CPU et macOS arm64 CPU. Le flux `extensions-feed/mvo-fel.json`
décrit les capacités par plateforme et permet l'installation depuis l'unstable
existante. Les variantes GPU Windows/macOS et la lecture sur appareil DV
restent à valider ; cette préversion ne constitue pas une release générale.
