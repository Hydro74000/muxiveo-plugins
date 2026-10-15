# Réponse à l'audit de la continuation FEL — 15 octobre 2026

Réponse point par point à l'[audit du 15/10/2026](2026-10-15-audit-continuation.md).

Méthode :

1. reproduire chaque défaut avant toute correction ;
2. corriger, puis ajouter un test qui échoue sans le correctif et passe avec ;
3. réauditer le point au-delà de l'emplacement signalé.

Les quatre défauts sont **avérés**. Le point 4 et l'attestation Y'CbCr ont chacun révélé un défaut supplémentaire, corrigé lui aussi.

## 1. [P1] L5 non réaligné quand le RPU FEL original est réutilisé — corrigé

**Muxiveo `e70852d`**

- **Reproduction.** Toy Story 5, recadrage de 8 lignes en haut et en bas. La sortie mesure bien 3840 × 2144, mais son L5 reste 42/42 au lieu de 34/34.
- **Cause.** Dans le pipeline multi-pistes, la branche du RPU FEL converti en P8.1 sautait `crop_dovi_rpu`.
- **Correction.** Le réalignement est appliqué après les deux branches, que le RPU soit extrait ou réutilisé.
- **Réaudit.**
  - Le chemin NVEncC réaligne déjà le RPU réutilisé (`routing.needs_rpu_alignment`).
  - Le chemin mono-piste n'est jamais emprunté par FEL : hors NVEncC, FEL passe toujours par le pipeline multi-pistes.
- **Tests.** Test réel avec recadrage pour x265 et pour NVEncC, ce dernier alimenté directement par le plugin. Assertion : L5 de sortie = L5 source − recadrage, borné à 0.

## 2. [P1] Chemin CPU rapide au-delà de 2 codes PQ12 — corrigé

**Plugin `7027483`**

### Cause

Deux amplifications s'enchaînent :

1. **PQ EOTF.** Le dénominateur `c2 − c3·p` approche zéro quand le signal approche 1. Il amplifie environ 85 fois l'erreur float32 de `p`.
2. **Matrice LMS→RGB.** Elle soustrait des termes d'environ 0,43 pour obtenir R ≈ 2·10⁻⁵ : un vert BT.2020 saturé vers 2 000 cd/m², le cas de la graine 66.

Le float32 ne suffit pas, même parfaitement arrondi : près de R = 0, PQ(3·10⁻⁸) vaut déjà environ 13 codes PQ12.

### Correction

- La PQ et la matrice passent en double, avec `log2`/`exp2` double sans branche.
- Les boucles restent vectorisées (AVX2+FMA choisi à l'exécution) ; la vectorisation est vérifiée dans le code machine.
- Aucun float32 ne subsiste dans le chemin CPU. La conversion Y'CbCr est déjà en double.

### Tests et mesures

- **Test natif.** Graines 1 à 200 (cas de l'audit : mapping identité, codes 64–940), plus 50 graines MMR. Résultat : ≤ 2 codes 16 bits, soit 0,125 code PQ12.
  - Avant correctif : échec dès la graine 1 (19 codes 16 bits).
- **Scènes réelles.** Sur les trois extraits, le chemin rapide est identique au bit près à la référence en double.
- **Coût.** Sur 300 images UHD (Ryzen 7800X3D), le débit passe de 11,3 à 8,1 i/s. L'objectif du lot 8 (≥ 5 i/s) reste tenu sur cette machine.
  - Le i7-8700K Windows sera remesuré avec l'attestation définitive.

### Limite GPU

Les shaders GPU calculent en float32, comme libplacebo. Ils partagent donc la même limite théorique sur le domaine synthétique.

Leur borne de 2 codes PQ12 reste celle mesurée par l'attestation sur contenu réel : 0,125 au maximum. Le README le précise désormais.

Le fp64 n'est pas une option : il n'est pas portable (MoltenVK) et il est 32 à 64 fois plus lent sur les GPU grand public.

## 3. [P1] Packaging GPU Windows bloqué par le contrôle ELF — corrigé

**Plugin `d1269a4`**

- **Cause.** Le bloc `readelf` commun aux moteurs expérimentaux s'appliquait aussi à la DLL PE.
- **Correction.**
  - Le contrôle ELF est réservé à Linux.
  - Le contrôle des imports PE couvre désormais toute DLL Windows, CPU comprise : elles sont toutes liées en `-static`.
- **Tests.** `main()` est testé sur le parcours Windows GPU dans une racine factice :
  - aucun appel à `readelf` ;
  - archive contenant la DLL, l'attestation, le manifeste et les sources ;
  - import étranger refusé avant toute écriture.

  Le test échoue sans le correctif.
- **Réel.** La DLL issue de la compilation croisée passe le vrai `objdump`.

## 4. [P2] Position chroma signalée par NVEncC — corrigé, avec un défaut SDR en plus

**Muxiveo `bc5b32c`**

### Côté NVEncC

- **Cause.** NVEncC gardait sa règle « `--chromaloc 2` si copie DV ».
- **Correction.** Il signale désormais la position du BL (`fel_context.chroma_location`), pour toute sortie FEL : HDR10, DV ou SDR.
- **Mesure.** Le tone mapping libplacebo de NVEncC produit les mêmes octets quels que soient le marquage des images et `--chromaloc` : la grille chroma reste celle du BL.

### Défaut supplémentaire en SDR

- **Cause.** Le tone mapping FEL sous-échantillonne déjà à la position du BL, mais zscale ne marque pas les images. La réduction 4:2:0 ajoutée ensuite supposait donc « left » et rééchantillonnait la chroma. Pour une source `topleft`, l'écart atteint 56 codes 8 bits.
- **Correction.**
  - Le tone mapping FEL marque ses images.
  - La seconde réduction est supprimée.

### Tests

- **Matrice de bout en bout** sur la séquence FEL : position (`topleft`, `left`) × encodeur (x265, NVEncC) × sortie (HDR10, DV, SDR). Douze cas.
  - Sans le correctif, trois échouent : les deux cas de l'audit et le SDR NVEncC.
- **Tests unitaires** de la signalisation et de la réduction unique.

## Écarts aux critères du plan

### Attestation des sorties finales (lot 4) — ajoutée

**Schéma 3.** Pour chaque GPU et chaque scène, `validate_yuv.py` compare la sortie Y'CbCr 4:2:0 10 bits du NUT livré à une conversion de référence en double (numpy). Cette référence est recalculée depuis le RGB déjà attesté du même producteur.

Deux cas sont contrôlés :

| Cas | Tolérance | Résultat CPU | Résultat RTX |
|---|---|---|---|
| `topleft`, recadrage (2, 4, 6, 8), sans tramage | ≤ 0,5 code (arrondi exact) | 0,5 | 0,5001 |
| `left`, tramage `blue` livré | < 1 code | 0,9999 | 0,9999 |

Sensibilité du contrôle : une position chroma fausse donne 3,94 codes, un recadrage décalé 16,5.

**Transport direct.** Ses images P010 (position, recadrage, tramage) doivent être identiques, octet pour octet, à ces sorties. C'est le cas.

**Faux écart écarté en chemin.** Mon premier essai faisait convertir le P010 par swscale : on observait jusqu'à 4,5 codes d'écart. Le P010 brut est en fait identique. La conversion swscale de P010 vers le 10 bits planaire n'est pas sans perte : l'attestation décale désormais elle-même les bits. Muxiveo transmet le P010 directement à NVENC, sans swscale.

**Attestation complète locale** (bibliothèque corrigée, iGPU AMD et RTX × six scènes, transport direct compris) : réussie.

- Reconstruction : 0,125 code PQ12 au plus face au CPU, 0,107 face à `kernel64`.
- Y'CbCr : 0,5001 au plus sans tramage, 0,9999 avec.
- P010 direct : identique dans les douze échantillons.

Elle ne remplace pas l'attestation de la bibliothèque définitive, qui reste à faire (Linux et Windows).

**Packaging.** `package.py` exige le schéma 3, les deux cas Y'CbCr dans leurs tolérances et, pour le direct, l'égalité P010. Le rapport réel ci-dessus est accepté ; tests ajoutés.

### Règle du lot 4 : quantification avant l'upload — décision documentée

`final_output_format` confie l'image finale au plugin quand rien ne modifie les pixels **avant** l'upload. Les filtres matériels situés après (`scale_cuda`, VPP NVEncC) reçoivent alors l'image 10 bits finale.

Ce n'est pas une régression : avant le lot 4, la réduction RGB → 10 bits était déjà insérée avant l'upload. Le point de quantification est inchangé. Un upload 16 bits ferait transiter deux fois plus de données, et `scale_cuda` ramènerait ensuite le P016 en P010 sans tramage. Le gain n'est pas établi. Le plan est amendé dans ce sens.

### Chevauchement GPU (lot 3) — partiel, documenté

Le pipeline a bien trois étages et des files bornées. En revanche, le rendu NUT garde un `pl_gpu_finish` par image : le chevauchement de plusieurs rendus et l'anneau de téléchargements ne sont pas livrés.

Cette limite ne coûte rien de mesurable à ce jour :

- **RTX** : le goulot est l'écriture du pipe (`render_blocked_s` dépasse le temps GPU cumulé de 1 ms par image).
- **iGPU AMD** : le rendu lui-même domine (34 ms par image).

**À rouvrir** si un GPU limité par les téléchargements apparaît dans les mesures.

### Métriques PSNR/SSIM/ΔE ITP

L'attestation borne désormais chaque étape en codes :

- reconstruction : ≤ 2 codes PQ12 ;
- conversion finale : ≤ 0,5 code arrondi, < 1 code tramé.

Les métriques de bout en bout après encodage relèvent de la série de mesures finale. Celle-ci sera lancée sur accord du mainteneur.

### README

La formulation « 0.2.0 prête » est remplacée par l'état réel : il reste l'attestation de la bibliothèque définitive (Linux, Windows), les mesures, la construction portable et le paquet.
