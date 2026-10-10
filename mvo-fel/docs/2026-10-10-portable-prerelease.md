# Validation de la première préversion portable

La présence de l'extension dans Muxiveo ne suffisait pas à l'installer : aucune
release `mvo-fel-v0.1.0` ni entrée `extensions-feed/mvo-fel.json` n'avait été
publiée. Cette publication fournit ces deux éléments, sans nouvelle release
de l'application.

## Binaires validés

Sources natives : commit `10307412ad9bdfb08e830b196ad0e4d70b34738a`.
[CI 38065556166](https://github.com/Hydro74000/muxiveo-plugins/actions/runs/38065556166) :
builds CPU Linux/Windows/macOS, référence numérique Mesa et candidat GPU
Linux avec FFmpeg direct au vert. Le candidat Linux est construit dans
manylinux 2.28 ; les deux binaires exigent au maximum `GLIBC_2.28`, sans
dépendance dynamique aux bibliothèques FFmpeg, libdovi, libplacebo ou glslang.
Le chargeur Vulkan et le pilote restent facultatifs pour le CPU.

| Binaire Linux | SHA-256 |
|---|---|
| `libmvo_fel.so` | `aa02dee10df4045ab91548dcd771354c70fc3359e90de76783aac0983f47e018` |
| FFmpeg privé (`ffmpeg-fel`) | `60ffaf21f7d38597d5deaab915f9442f0e8947405c163108aae0324511f14776` |

## Comparaison numérique sur les binaires exacts

Trois scènes FEL réelles, images 0/8/16, sur les deux GPU locaux : 18 images.
La limite reste deux codes PQ 12 bits pour chaque comparaison.

| GPU | Écart maximal face au CPU | Écart maximal face à libplacebo `kernel64` |
|---|---:|---:|
| AMD Ryzen 7800X3D, iGPU RADV | 0,312428 | 0,309435 |
| NVIDIA RTX 4070 Ti SUPER | 1,374690 | 1,359730 |

Les empreintes GBRP16 des images effectivement produites par le pipe NUT
et par le filtre source du FFmpeg privé correspondent à celles du renderer
comparé aux références. Le packaging exige l'attestation de schéma 2 et
vérifie les empreintes des binaires ainsi que la limite glibc 2.28. Les tests
ABI, B-frames, annulation, lecture, fermeture aval et invalidation de piste
passent également sur le candidat portable.

## Périmètre de la préversion

Linux x86_64 : CPU, Vulkan AMD/NVIDIA et transport direct expérimental vers
FFmpeg `hevc_nvenc`, sous les conditions documentées dans le README.
Windows x86_64 et macOS arm64 : CPU. Les capacités et la présence du FFmpeg
direct sont décrites par plateforme dans le flux, avec le même contrat ABI 1.

L'activation reste manuelle et les installations de validation sont isolées
du dossier utilisateur. Aucun média commercial n'est distribué. La lecture
sur téléviseur Dolby Vision, les GPU des autres OS, Intel et plusieurs GPU
dédiés restent à valider.
