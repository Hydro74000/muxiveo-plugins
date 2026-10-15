# Lot 9 — plateformes et raccordements matériels

État au 15/10/2026, après les lots 1 à 8.

## Windows x86_64 : GPU validé sur matériel réel

### Méthode

Le binaire GPU (`mvo_fel.dll`, Vulkan, décodage Vulkan Video contrôlé) a été produit par **compilation croisée MinGW sous Linux**. C'est un outil local de validation : les paquets publiés restent construits par la CI (MSYS2 UCRT64).

Il a été testé sur le PC Windows 11 de test : i7-8700K, GTX 1070 (pilote 581.57), Intel UHD 630.

### Résultats

- Test mathématique natif, smoke test complet et test pull : réussis. Le smoke test couvre l'écriture sur HANDLE Windows, le seek, le Y'CbCr, le CPU et le GPU.
- Attestation stricte `validate_gpu.py` : **réussie**. Six scènes FEL réelles (Moana ×3, Il faut sauver le soldat Ryan ×2, Toy Story 5), deux GPU, pixels du producteur identiques au renderer de référence.
  - Écart maximal face au CPU : 0,125 code PQ12.
  - Écart maximal face à `kernel64` : 0,092 code PQ12.
- La DLL n'importe que des DLL système (KERNEL32, msvcrt, bcrypt, ntdll, shlwapi, userenv, ws2_32). Le chargeur Vulkan est ouvert à l'exécution. `package.py` vérifie désormais ces imports pour un paquet GPU Windows.

Débits sur un extrait court de 36 images UHD, ouverture comprise :

| Moteur | Débit |
|---|---:|
| GTX 1070 (décodage CPU) | 22,6 i/s |
| Intel UHD 630 (décodage CPU) | 10,3 i/s |
| CPU (i7-8700K) | 5,6 i/s |

### Décodage Vulkan Video instable sur Pascal

Sur la GTX 1070, le décodage Vulkan Video HEVC Main10 **n'est pas déterministe**. D'un passage à l'autre, certaines images diffèrent du décodage logiciel.

Le premier contrôle (16 images, un décodage) est passé une fois. Les images 8 et 16 de l'encodage ont ensuite divergé, ce que l'attestation a détecté.

Le contrôle est désormais renforcé :

- 48 images par couche ;
- **deux décodages Vulkan indépendants**, tous deux comparés au logiciel ;
- résultat mis en cache par GPU, source et piste.

Sur la GTX 1070, il rejette le décodage Vulkan cinq fois sur cinq, et le décodage CPU donne des pixels conformes. Sur la RTX 4070 Ti SUPER (Linux), il est validé (2 × 48 images). L'iGPU AMD (RADV) et l'Intel UHD 630 n'exposent pas `VK_KHR_video_decode_h265`.

### Publication

La tâche CI `windows_gpu_candidate` (lancement manuel `prepare_gpu`) construit le candidat. Celui-ci est ensuite attesté sur matériel Windows avant le paquet, comme pour Linux.

La voie directe FFmpeg/NVENC reste réservée à Linux : `ffmpeg-fel` n'est construit que pour Linux.

## macOS arm64 : CPU par défaut, GPU en choix manuel

Seuls les runners de CI macOS sont disponibles. Aucun contenu FEL réel ne peut y valider le GPU (MoltenVK). Borne validée par l'utilisateur le 15/10/2026 :

- Le paquet macOS publié reste **CPU**. `package.py` refuse un paquet GPU sans attestation sur contenu réel.
- Muxiveo : Auto retient le CPU sous macOS (`automatic_gpu_allowed`). Un GPU explicitement choisi reste utilisable si une version GPU est installée.
- La tâche CI `macos_gpu_probe` construit la version GPU avec MoltenVK, énumère les GPU visibles et lance la comparaison synthétique GPU/CPU/libplacebo. Elle n'est pas bloquante. C'est la condition préalable avant toute proposition GPU sur macOS.

## Raccordement direct VAAPI / QSV : non livré

Essai sur l'iGPU AMD Raphael (RADV, pilote VA de l'hôte), FFmpeg 8.1.2 :

- L'encodage `hevc_vaapi` Main10 fonctionne.
- Le mappage d'images Vulkan vers VAAPI (`hwmap=derive_device=vaapi`) échoue : « Unable to export the image as a FD », puis le processus plante (double libération). Il faudrait inverser la conception : images allouées côté VAAPI puis importées en Vulkan pour le rendu FEL.
- Sur la seule machine AMD disponible, le rendu FEL est limité par le GPU (22,8 i/s). Le transport n'y est pas le goulot, donc aucun gain mesurable n'est possible.
- QSV ne se teste que sur le serveur Intel de production, où rien ne doit être installé.

Conformément au plan (gain mesuré obligatoire), ce raccordement n'est pas livré.

Conditions de reprise :

- un GPU AMD ou Intel **dédié**, où le pipe redeviendrait le goulot ;
- la conception VAAPI → Vulkan décrite ci-dessus ;
- liaison statique des répartiteurs libva/libvpl, comme le FFmpeg BtbN déjà embarqué par Muxiveo.
