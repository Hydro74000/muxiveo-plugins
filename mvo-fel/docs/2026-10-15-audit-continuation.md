# Audit de la continuation FEL — 15 octobre 2026

**Avis : corriger les quatre défauts ci-dessous avant de considérer la 0.2.0 prête à publier.** L'attestation GPU Linux réussit, mais elle ne couvre ni le nouveau calcul CPU optimisé, ni le réalignement L5 après réutilisation du RPU, ni le packaging Windows complet.

Périmètre relu, sans modification du code ni publication :

| Dépôt | Branche | Base locale | Dernier commit | Commits |
|---|---|---|---|---:|
| muxiveo-plugins | fel-continuation | main, `855d6ca` | `ead50d3` | 12 |
| Muxiveo (mediarecode) | devel-cli | origin/devel-cli, `7ce4c92` | `94f0539` | 8 |

Les deux arbres étaient propres au départ. Les références distantes sont celles présentes localement ; aucun fetch, push, tag ou changement de branche n'a été effectué. Le présent rapport est le seul ajout dans les dépôts.

**1. [P1] Le RPU réutilisé échappe au réalignement L5 après recadrage.**

Commit : Muxiveo `e60526b`, lot 1. Emplacement : [multi_video.py](/home/hydromel/dev/mediarecode/core/workflows/encode/runtime/multi_video.py:203).

La branche qui convertit `fel_context.original_rpu` en P8.1 se termine après l'ajout du fichier à la liste de nettoyage. L'appel à `crop_dovi_rpu` reste dans le `elif` réservé à l'extraction classique. Une reconstruction FEL réussie avec réutilisation du RPU ne réaligne donc plus les offsets de zone active sur l'image recadrée. Cela concerne le parcours FFmpeg de préparation vidéo ; le parcours NVEncC possède son propre réalignement après `source_rpu`.

Reproduction complète sur un extrait FEL réel, x265, copie Dolby Vision et recadrage de 8 pixels en haut et en bas :

| Mesure | Source | Attendu en sortie | Obtenu |
|---|---:|---:|---:|
| Hauteur | 2160 | 2144 | 2144 |
| L5 haut | 42 | 34 | 42 |
| L5 bas | 42 | 34 | 42 |

L'encodage est annoncé comme réussi. Les RPU extraits de la source et de la sortie confirment les offsets incorrects. La zone active déclarée au lecteur Dolby Vision ne correspond plus aux pixels.

Correction attendue : faire converger l'extraction et la réutilisation vers le même traitement de géométrie, avant l'expansion des métadonnées et leur injection. Ajouter un test de workflow FEL recadré qui inspecte le L5 réellement encodé.

Preuves locales : [reproduction](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/crop_check.py), [RPU source](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/crop/source.bin), [RPU de sortie](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/crop/encoded.bin), [journal](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/crop/log.json).

**2. [P1] Le chemin CPU optimisé ne respecte pas la borne de 2 codes PQ12 sur tout le domaine synthétique accepté.**

Commit : plugin `329df9e`, lot 8. Emplacement : [reconstruction.cpp](/home/hydromel/dev/muxiveo-plugins/mvo-fel/src/reconstruction.cpp:510), calcul de `FastReconstruction::rows` et nouvelles fonctions PQ float32.

Le remplacement du calcul en double par des intermédiaires et une PQ float32 introduit des écarts amplifiés lorsque les termes de la matrice LMS→RGB se compensent. Le seul test rapide ajouté utilise une graine et un mapping MMR particuliers ; il ne borne pas les autres mappings acceptés.

Reproduction à partir du test mathématique du dépôt : conserver son RPU synthétique à mapping polynomial identité et ses matrices identité, utiliser des images BL 64×32 / EL 32×16, positions `topleft`, et tirer les codes dans `[64, 940]` avec `std::mt19937`, graines 1 à 200. Comparer `FastReconstruction::rows` à `reconstruct_rows`, sur la même bibliothèque mathématique reconstruite en Release.

À la graine 66, pixel `(5, 28)`, plan R : référence **5715**, chemin rapide **4285**, soit **1430 codes 16 bits ≈ 89,35 codes PQ12**. Le signal non linéaire avant les matrices vaut environ `(0,790235 ; 0,828172 ; 0,425237)`. Les deux chemins terminent normalement ; aucune entrée malformée ni valeur non finie n'est nécessaire pour obtenir cet écart.

Ce résultat est synthétique. Il ne démontre pas un défaut visible sur les trois scènes réelles testées. Des essais complémentaires avec les matrices du test de référence et des plages de codes plus modérées n'ont pas dépassé la borne. Il démontre toutefois que la borne générale annoncée ne découle pas des mesures actuelles et n'est pas respectée pour tous les paramètres acceptés.

L'attestation GPU ne ferme pas ce point : [reference_file.cpp](/home/hydromel/dev/muxiveo-plugins/mvo-fel/tests/reference_file.cpp:111) et `reference_test.cpp` emploient toujours `reconstruct_rows`, le chemin en double, comme référence CPU.

Correction attendue : conserver suffisamment de précision, ou prévoir un repli lorsque l'erreur peut être amplifiée ; élargir le test rapide aux mappings polynomial/MMR et aux annulations entre canaux. Le cas ci-dessus doit rester sous 2 PQ12 avant de revendiquer cette borne pour le CPU optimisé.

Preuves locales : [source du stress](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/cpu_stress_limited.cpp), [résultats](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/cpu-stress-limited.log), [diagnostic du pixel](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/cpu_diagnostic.cpp). Le stress est dérivé de `tests/reconstruction_test.cpp`, sans modifier ce dernier.

**3. [P1] Le packaging GPU Windows appelle encore le contrôle ELF Linux.**

Commit : plugin `a13991a`, lot 9. Emplacement : [package.py](/home/hydromel/dev/muxiveo-plugins/mvo-fel/scripts/package.py:143).

Le nouveau code autorise `windows-x86_64` pour un moteur expérimental et contrôle ses imports avec `objdump`. Quelques lignes plus bas, le bloc commun à tous les moteurs expérimentaux lance néanmoins `readelf -d` et applique la liste des bibliothèques ELF Linux.

Sur une vraie DLL PE, `readelf` échoue avec « Not an ELF file ». Dans un environnement Windows dépourvu de `readelf`, le script échoue dès la recherche de l'exécutable. Un candidat GPU Windows attesté ne peut donc pas être empaqueté par cette voie.

La reproduction sur Linux remplace uniquement `load_capabilities`, afin de franchir le chargement impossible d'une DLL Windows dans ce processus Linux. Le contrôle d'imports et l'appel fautif portent sur la vraie DLL Windows disponible localement. Le script échoue exactement à la ligne 148.

Correction attendue : réserver le contrôle `readelf` aux paquets Linux et conserver le contrôle PE pour Windows. Tester `main()` sur le parcours de packaging Windows expérimental ; le test unitaire ajouté au lot 9 ne teste que `check_windows_imports`.

Preuve locale : [reproduction du parcours de packaging](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/package_windows_check.py).

**4. [P2] La position chroma produite par FEL n'est pas toujours celle signalée par NVEncC.**

Correctif incomplet dans Muxiveo `e60526b`, puis parcours direct au consommateur dans `2c09f99`. Emplacements : [pipeline.py](/home/hydromel/dev/mediarecode/core/fel/pipeline.py:177) et [nvencc.py](/home/hydromel/dev/mediarecode/core/workflows/encode/runtime/nvencc.py:517).

Le plugin sous-échantillonne désormais à la position du BL. Le constructeur NVEncC conserve cependant sa règle antérieure : imposer `--chromaloc 2` avec copie DV, et ne pas imposer de position dans l'autre cas. Les options placées sur l'étage FFmpeg intermédiaire ne corrigent pas la VUI de l'encodeur final, et cet étage peut maintenant être supprimé.

Deux workflows complets sur la fixture FEL synthétique reproduisent le défaut :

| Position du BL et du rendu FEL | Copie DV | Position de la sortie encodée |
|---|---|---|
| topleft | non | left |
| left | oui | topleft |

La seconde source est obtenue par changement de la VUI de la fixture, avec `hevc_metadata=chroma_sample_loc_type=0`. Le flux reste FEL et le workflow réussit. Le défaut est une signalisation incompatible avec la grille des échantillons, ce qui peut déplacer la chroma à la lecture. Il persiste malgré l'objectif de correction B3 du lot 1.

Correction attendue : traduire explicitement `fel_context.chroma_location` vers l'option NVEncC finale, pour les sorties HDR concernées, avec et sans copie DV et avec ou sans étage FFmpeg. Le test réel actuel utilise un seul cas de position chroma et ne couvre pas cette matrice.

Preuves locales : [reproduction](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/chroma_check.py), [sortie HDR10](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/chroma-hdr10/output.mkv), [sortie DV](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/chroma-dv/output.mkv).

**Écarts aux critères du plan et limites de validation.**

- Lot 4 : la règle livrée examine les filtres avant l'upload, pas tous les traitements aval demandés dans le plan. `final_output_format("format=p010,hwupload,scale_cuda=1920:1080", 10)` renvoie `yuv420p10le`. De même, `--vpp-nlmeans sigma=0,h=0.12` côté NVEncC laisse le producteur en 10 bits final. Les traitements matériels/VPP ne forcent donc pas la sortie 16 bits prévue. C'est un écart de conception confirmé par les commandes produites ; aucune perte de qualité supplémentaire n'a été quantifiée dans cet audit.
- Lot 4 : l'attestation stricte vérifie les pixels RGB reconstruits. `validate_direct.py` demande encore le format RGB par défaut ; le P010 final, son recadrage et son tramage ne sont pas attestés par cette comparaison d'empreintes. Les smoke tests YUV vérifient surtout le succès et les tailles. Les tests d'intégration vérifient le compte d'images, la profondeur, les tags et les métadonnées. Ils ne remplacent pas la comparaison numérique finale ni les métriques PSNR/SSIM/ΔE ITP demandées au plan.
- Lot 3 : le pipeline comporte bien trois threads/étages et des files bornées. Le transport NUT garde un `pl_gpu_finish` par image après les téléchargements. Le chevauchement GPU de plusieurs rendus et l'anneau de téléchargements envisagés au plan ne sont pas livrés. Cela n'invalide pas les gains mesurés antérieurement, mais borne l'affirmation de réalisation complète du lot.
- Lots 7 et 9 : abandon des cœurs tensoriels et du direct VAAPI/QSV explicitement documenté. Ce sont des décisions de périmètre, pas des défauts ajoutés à cette liste.
- Lot 10 : les benchmarks complets répétés de 30–60 s n'ont pas été rejoués dans cet audit. Les chiffres historiques des documents restent des mesures antérieures. La lecture sur un téléviseur DV, le candidat CI Windows MSYS2 et la sonde macOS restent à valider selon le propre bilan du dépôt.
- Version : CMake et `MVO_FEL_VERSION` indiquent encore `0.1.0`, conformément à l'absence de release. Le README « 0.2.0 prête » doit être interprété avec les défauts et validations restantes ci-dessus.

**Vérifications exécutées.**

Toutes les commandes de construction, de test et de reproduction ont tourné dans `my-distrobox`. La construction utilise les sources actuelles des dépôts et les dépendances privées déjà présentes ; elle est isolée des builds précédents.

| Vérification | Résultat |
|---|---|
| Tests Python FEL : moteur, appareils, direct, benchmarks, panneau | 116 réussis |
| Tests Python NVEncC, workflows et géométrie DV | 534 réussis |
| Intégration FEL synthétique, plugin reconstruit, direct activé | 14 réussis |
| Intégration FEL réel, extrait Toy Story 5 | 6 réussis |
| CTest natif : math, producteur, packaging, pull, références CPU/GPU | 6 suites réussies |
| Attestation stricte : 3 scènes réelles × 2 GPU, NUT et direct | réussie, 6 échantillons de 3 images |
| `git diff --check` sur les deux séries | réussi |
| Reproductions dédiées | 4 défauts ci-dessus confirmés |

Total Python : **670 tests réussis**. Les 14 tests d'intégration initialement ignorés avant configuration des chemins ont ensuite été exécutés avec succès. La suite complète de tous les modules Muxiveo n'a pas été lancée.

L'attestation a utilisé les extraits privés `spr-0700`, `spr-4000` et `ts-2500`, sur l'iGPU AMD Raphael et une RTX 4070 Ti SUPER. Maxima observés : **0,124971 code PQ12** face à la référence CPU en double, **0,106527** face à `kernel64`. Les empreintes du producteur NUT et du transport Vulkan direct correspondent à celles du renderer de référence. Décodage CPU sur AMD ; décodage Vulkan Video vérifié sur NVIDIA. Pour ces extraits courts, le contrôle a comparé toutes les images disponibles, soit 37 à 39 par couche.

Rapport brut : [validation.json](/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/validation.json). SHA-256 de la bibliothèque reconstruite et attestée : `b143069dcbf8c77d270129316ed1847510f652b32bf174a271d8f4e3b105005a`.

Les fichiers de reproduction, binaires, médias de sortie et résultats restent dans `/var/mnt/Drive2/muxiveo-fel-dev/codex-audit-20261015/`. Aucun média privé n'est ajouté au dépôt. Les essais Windows physiques et macOS n'ont pas été rejoués ; les résultats documentés par l'auteur n'ont pas été présentés comme des vérifications de cet audit.

**Couverture des 20 commits.**

« Aucun autre défaut identifié » désigne la relecture et les essais ci-dessus, sans garantir tous les pilotes, médias ou réglages possibles.

| Dépôt | Commit | Lot / objet | Résultat de relecture |
|---|---|---|---|
| plugin | `993b862` | 0, statistiques et plan | mesures murales cohérentes ; critères du plan pris comme référence |
| plugin | `1fcdaff` | 1, seek natif | aucun autre défaut identifié ; seek brut couvert par smoke |
| plugin | `10b4803` | 2, descripteur/HANDLE | aucun autre défaut identifié ; NUT callback/fd et aval fermé couverts |
| plugin | `6bea577` | 3, pipeline borné | tests et attestation réussis ; chevauchement GPU partiel signalé |
| plugin | `0bbb198` | 4, sorties YUV/P010 | intégration réussie ; attestation numérique des formats finaux à compléter |
| plugin | `580a820` | 5, crop et PTS | aucun autre défaut identifié ; crop et PTS exercés en intégration |
| plugin | `19eeb70` | 6, décodage Vulkan et entiers | attestation réelle AMD/NVIDIA réussie |
| plugin | `592498d` | 6, composition fusionnée | références synthétiques et réelles réussies |
| plugin | `e37cc20` | 7, cœurs tensoriels écartés | décision et conditions de reprise explicites ; performances non remesurées |
| plugin | `329df9e` | 8, CPU optimisé | défaut 2 : dépassement de la borne PQ12 |
| plugin | `a13991a` | 9, Windows/macOS | défaut 3 : packaging Windows bloqué ; CI non rejouée |
| plugin | `ead50d3` | 10, docs et notices | état « prêt » trop affirmatif au regard des défauts et critères restants |
| Muxiveo | `33e6e92` | 0, benchmark NUT/direct | aucun autre défaut identifié ; relecture des variantes et médianes |
| Muxiveo | `e60526b` | 1, seek/RPU/chroma | défauts 1 et 4 : L5 contourné et correction chroma incomplète |
| Muxiveo | `00598e0` | 0, contrôle PTS | aucun autre défaut identifié ; PTS fidèles ou réguliers acceptés explicitement |
| Muxiveo | `5a7ed19` | 2, pipe natif | aucun autre défaut identifié ; adaptation et intégration exercées |
| Muxiveo | `fc5251d` | 2, garde Windows | garde de plateforme relue ; Mypy non relancé |
| Muxiveo | `a7151ad` | 4, négociation finale | sorties exercées ; règle aval plus limitée que le plan |
| Muxiveo | `2c09f99` | 5, alimentation NVEncC | workflow exercé ; signalisation chroma finale incorrecte dans les cas du défaut 4 |
| Muxiveo | `94f0539` | 9, Auto macOS sur CPU | règle et test relus, test réussi ; matériel macOS non testé |
