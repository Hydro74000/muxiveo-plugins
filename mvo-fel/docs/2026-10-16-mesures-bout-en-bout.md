# Mesures de bout en bout (lot M)

16 octobre 2026. Lot M du [plan de précision certifiée](2026-10-16-plan-precision-certifiee.md).

La question posée : le FEL ralentit-il un encodage réel, et qu'est-ce qui limite le débit ? Selon la réponse, on sait si le recouvrement ([P](2026-10-16-recouvrement.md)) ou les Tensor Cores (T) servent.

Cette note remplace une première version. Celle-ci tirait ses conclusions de la seule durée totale des workflows, sur des extraits de 10 s. Elle ne distinguait ni le calcul FEL, ni le transport, ni l'encodeur, ni les coûts fixes.

## Protocole

- **Workflow.** Muxiveo complet (`scripts/benchmark_fel_workflow.py`), FEL en mode Lent sur la RTX 4070 Ti SUPER, décodage Vulkan Video.
- **Extraits.** 20 s (environ 480 images UHD) de Saving Private Ryan (contenu MMR) et de Toy Story 5 (polynômes), en P7 FEL.
- **Encodeurs.** NVEncC HEVC 10 bits sur le même GPU :
  - preset par défaut, qui encode à environ 116 i/s ;
  - réglage « standard HQ » de l'utilisateur, à environ 50 i/s : VBR 34 Mbit/s, 2 passes au quart, UHQ, lookahead 24, AQ temporelle.
- **Mesures isolées (`--isolate`).** Elles reprennent les réglages exacts que le workflow transmet au producteur, ainsi que la commande exacte de l'encodeur :
  - **A** : FEL seul, écrit par le plugin sur `/dev/null` ;
  - **B** : encodeur seul, nourri par le même NUT via un pipe ;
  - **C** : workflow complet.
- **Chronologie.** Le banc horodate chaque étape et chaque ligne d'outil.

Les mesures « plugin seul » antérieures passaient par un lecteur Python, qui plafonnait vers 1,9 Go/s, soit environ 78 i/s en Y'CbCr 10 bits UHD. Le Standard et le Rapide y paraissaient limités vers 78 i/s ; c'était le lecteur, pas le plugin.

## Cause principale du ralentissement : l'analyse d'entrée de NVEncC

Les mesures horodatées montrent que NVEncC, nourri par un NUT d'images brutes, met 8 à 10 s avant sa première image encodée. Ensuite, il encode à environ 126 i/s.

Son analyse d'entrée par défaut porte sur 5 s. Elle accumule une centaine d'images de 25 Mo à environ 13 i/s, avec le FEL comme avec le même NUT servi depuis le cache. Pendant ce temps, le FEL reste bloqué sur le pipe. Une analyse jusqu'à 1 s évite ce comportement, alors qu'avec 2 s il réapparaît.

| Analyse d'entrée (`--input-analyze`) | Première image encodée | Total, 483 images |
|---|---:|---:|
| défaut | 8,4 s | 11,9 s |
| 0,05 à 1 s | 1,6 s | 6,2 à 7,3 s |
| 2 s | 10,3 s | 13,8 s |

**Correctif dans Muxiveo** (`2674d87`) :
- `--input-analyze 0.2` sur l'entrée NUT, car le NUT décrit déjà son flux ;
- `--fps` égal à la cadence de la source. Le NUT ne porte aucune cadence, et NVEncC la devinait : 24/1 ou 25/1 pour du 23,976. Les horodatages restent ceux du producteur (`--avsync vfr`), vérifié en cadence fixe et en cadence variable.

## Durées des workflows

Extraits de 20 s, médianes de 2 passages.

| Cas | BL | FEL avant correctif | FEL après correctif |
|---|---:|---:|---:|
| Ryan, NVEncC par défaut | 22,7 s | 34,9 s (+56 %) | 27,5 s (+21 %) |
| Ryan, NVEncC HQ | 28,5 s | 38,5 s (+37 %) | 30,8 s (+8 %) |
| Toy Story 5, NVEncC par défaut | 18,7 s | 28,0 s (+49 %) | 23,8 s (+27 %) |
| Toy Story 5, NVEncC HQ | 24,7 s | 36,1 s (+46 %) | 26,7 s (+8 %) |

**Ce qui reste :**

- **Coûts fixes (≈ 1,5 à 2 s par job).** Ouverture du plugin (1,1 à 1,4 s : vérification Vulkan Video, auto-test, compilation), puis démarrage de NVEncC.
- **NVEncC HQ (≈ 50 i/s).** L'encodeur limite le débit, avec ou sans FEL : sur un film entier, le surcoût du FEL est négligeable.
- **NVEncC par défaut (≈ 116 i/s).** Le FEL en Lent devient la limite, à 75–81 i/s dans le workflow. Sur un film entier, l'encodage dure environ 50 % de plus qu'en BL.

L'étape 3 de Muxiveo (détection DoVi/HDR10+ sur 240 images) coûtait 15 s par job, avec ou sans FEL, parce que ffprobe décode sur un seul fil. Avec `-threads 0` (`31b1ac4`), elle prend 2 s.

## Débit du FEL seul

Sortie Y'CbCr 4:2:0 10 bits finale vers `/dev/null`, environ 480 images UHD, décodage Vulkan Video. Débits en i/s, avec 1 puis 2 images en vol.

| GPU, contenu | Lent | Standard | Rapide |
|---|---:|---:|---:|
| RTX, Ryan (MMR) | 73,3 / 81,3 | 141,8 / 171,3 | 259,8 / 269,8 |
| RTX, Toy Story 5 (polynômes) | 82,4 / 94,9 | 170,3 / 207,1 | 262,9 / 269,7 |
| Raphael, Ryan | 2,3 / 2,3 | 6,1 / 6,1 | 14,6 / 15,1 |

Le Lent est limité par sa composition : 9,3 ms par image en virgule fixe sur Ryan avec une image en vol, 12,1 ms quand deux images se partagent le GPU. Il faut y ajouter 1,9 ms de sortie (Y'CbCr en entiers exacts). Le Standard et le Rapide dépassent largement tout encodeur NVENC.

## Conséquences

- **P (2 images en vol) reste par défaut** : +11 à +15 % sur le Lent RTX, sans effet sur Raphael.
- **T est rouvert.** Avec un encodeur rapide, le Lent RTX limite le débit. Mais les Tensor Cores ne reprendraient au plus que 1,3 ms par image sur Ryan, et rien sur un contenu à polynômes ; le Lent resterait sous les 116 i/s de NVEncC. Le lot L du plan profile donc d'abord la composition section par section, puis choisit ses leviers, T compris.
- **Avec RIFE** (24 → 59,94 i/s sur le même GPU), RIFE limite le débit, avec ou sans FEL. Muxiveo y garde une seule image en vol (`mvo_fel_set_render_workers`). Ce cas, marginal car très lourd, n'oriente pas les choix ci-dessus.
