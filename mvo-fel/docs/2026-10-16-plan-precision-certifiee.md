# Plan : précision certifiée du rendu FEL

16 octobre 2026. Suite de la [proposition](2026-10-16-gpu-exact-proposition.md) et de la [mesure d'ambiguïté](2026-10-16-quantification-certifiee-mesure.md).

## Objectifs

1. **Une référence parfaite et figée.** Le code RGB16 de référence est l'arrondi mathématiquement exact de la formule documentée : sommes Spline16 entières, coefficients du RPU en rationnels, constantes en doubles. Il ne dépend plus des arrondis d'une implémentation.
2. **Le mode Lent ne se dégrade jamais.** Il produit exactement la référence. Toute variante plus rapide n'est livrée qu'avec 0 différence et une borne d'erreur démontrée.
3. **Rapprocher les autres modes de la référence,** au coût le plus faible possible, y compris sur iGPU et sur les GPU sans double (piste B).

Critère commun aux modes exacts : **un seul code entier possible**. Le choix de la précision intermédiaire ne joue que sur le coût et le nombre de reprises, jamais sur l'exactitude.

## Principe

1. Les décisions discontinues restent exactes en entiers (déjà le cas) : échantillonnage Spline16, segment de mapping, zéro du résiduel NLQ.
2. La valeur linéaire R de chaque composante est encadrée : R ∈ [R̃ − e, R̃ + e], où e découle d'une **borne démontrée** du calcul employé.
3. **L'OETF disparaît.** R est classé entre des seuils précalculés T(k + ½) = EOTF((k + ½)/65535). Ils sont stockés arrondis vers le bas ; la borne haute est le double suivant. Le code k est certain si l'encadrement tient dans la cellule.
4. Les composantes ambiguës sont recalculées en précision supérieure, de façon compacte (liste, et non branche dispersée).
5. La conversion Y'CbCr est faite en **arithmétique entière exacte** depuis les codes RGB16 certifiés. Toutes ses constantes sont rationnelles : BT.2020 2627/10000 et 593/10000, poids chroma en seizièmes, seuils de tramage en (k + ½)/64 ou /4096.

Conventions :

- arrondi au plus proche ; une valeur exactement égale à un seuil prend le code supérieur (comme `lround`) ;
- R ≤ 0 donne le code 0, R ≥ 1 le code 65535 ;
- un signal hors du domaine de l'EOTF est refusé (comportement actuel).

## Lots

| Lot | Contenu | Livrable | Critère |
|---|---|---|---|
| **C0** | Table des seuils PQ | Générateur `decimal` à 60 chiffres (sans dépendance) ; table de 65 535 doubles arrondis vers le bas, intégrée au plugin ; empreinte vérifiée par test | Chaque seuil vérifié : T_bas ≤ T exact < suivant(T_bas) |
| **C1** | CPU certifié (nouvelle référence) | Chemin CPU rapide : R en double + borne démontrée (marge large), classement par seuils ; reprises en double-double (TU compilée sans contraction, validée face à `decimal`) ; Y'CbCr en entiers exacts | Écarts avec le CPU actuel comptés sur le corpus (attendu : 0 ou rarissimes, uniquement près des frontières) ; tests natifs |
| **A1** | Lent GPU « double sans OETF » | Mapping, EOTF et matrices en double ; borne démontrée ; classement par seuils ; reprises résolues **par le CPU certifié lui-même** (liste rapatriée puis correctif téléversé) ; Y'CbCr en entiers exacts | **0 différence avec C1** : stress 3 × 200 graines, scènes réelles, attestation ; débit mesuré (attendu ≈ 2 fois le Lent actuel sur RTX) |
| **B0** | Banc du mapping en entiers multi-mots (Raphael) | Le bloc complet (signaux, produits, polynôme ou MMR, NLQ, matrice Y'CbCr→LMS') en virgule fixe sur plusieurs mots `uint32`, précision ≤ 10⁻¹² des termes, comparé au même bloc en double | **Go si au moins 2 fois plus rapide** que le double sur Raphael ; sinon arrêt de B |
| **B1** | Certificat entier (si B0 go) | EOTF et matrice en virgule fixe à borne démontrée, classement par seuils, reprises vers le CPU ; aucun `shaderFloat64` requis | 0 différence avec C1 ; débit iGPU face au Standard actuel ; exactitude possible sur Intel Arc et Mac |
| **R** | Rapide amélioré | Shader float32 Muxiveo à la place de libplacebo : pivots et zéro NLQ exacts, EOTF et OETF sans annulation | Écart au CPU mesuré (stress et réel), en baisse nette ; coût iGPU acceptable ; tous GPU |
| ~~A2~~ | ~~Étage float32 du Standard~~ | Remplacé par A2' (décision du 16/10) : l'erreur relative de L en float32 plafonne vers 10⁻⁶ (L = g^6,28 amplifie l'erreur de g = num/den), borne réaliste 6·10⁻⁶ à 1,5·10⁻⁵ ; 15 à 80 % des pixels resteraient à recalculer, gain ≈ 1,3× au mieux | — |
| **A2'** | Lent en virgule fixe multi-mots | Mapping (B0) **et EOTF** en entiers 64 bits : logarithme et exponentielle par table de 64 entrées et polynôme court, num = c₁·expm1(ln(x/x₀)/m2) sans annulation, inverse par Newton ; matrice LMS→RGB et seuils PQ en virgule fixe 96 bits ; erreurs de méthode bornées analytiquement, erreurs d'arrondi démontrées (Gappa, installé) ; même classement, mêmes reprises CPU et même auto-test qu'A1 ; variante choisie au démarrage là où elle bat le double (NVIDIA) | 0 différence avec C1 (stress, réel, attestation) ; débit Lent RTX ≈ 2× A1 |
| **P** | Recouvrement du pipeline de rendu GPU | Plusieurs images en vol : téléversement, calcul, rapatriement des reprises et téléchargement d'images successives se chevauchent (contextes de rendu par fil, décision d'auto-test partagée) | Lent RTX vers 70 i/s ou plus ; codes inchangés (attestation) ; mémoire GPU mesurée |
| **T0** | Banc Tensor Cores (RTX) | Mapping MMR et matrices en virgule fixe découpée en tranches INT8, produits partiels **exacts** accumulés en INT32 sur les unités matricielles (schéma d'Ozaki, `VK_KHR_cooperative_matrix`) ; pipeline de calcul hors libplacebo si l'extension GLSL ne peut pas y être activée ; même précision des termes que B0 (≤ 10⁻¹²), comparé au mapping en double d'A2 | **Go si au moins 2 fois plus rapide** que le mapping en double sur RTX ; sinon arrêt de T |
| **T1** | Mapping sur Tensor Cores (si T0 go) | Intégration dans le Lent et le Standard : mapping exact à erreur bornée sur unités matricielles, le reste inchangé (étage float32 d'A2, classement par seuils, reprises vers le CPU) ; repli double sans l'extension | 0 différence avec C1 ; débit face à A2 sur RTX ; exactitude sans `shaderFloat64` possible sur Intel Arc (XMX), non testable localement |
| **V** | Validation et livraison | Attestation de schéma 5 (référence certifiée, compteurs de reprises) ; documentation ; textes de Muxiveo (débits et précisions) | Attestation réussie ; tests Muxiveo et natifs verts |

Ordre d'exécution : C0 → C1 → A1 → B0 → R → A2' → B1 (arrêté) → P → V ; T0 → T1 reportés, sur mesure (décision du 16/10 : le mapping ne pèse plus qu'environ 2 ms sur 7,8 ms de composition, le goulot est la mise en série de l'étage de rendu). Chaque lot est commité séparément puis audité.

T vient après A2' : sa référence à battre est la virgule fixe multi-mots d'A2' sur RTX, et non plus le double.

Ce que les Tensor Cores ne couvrent pas :

- **pas de double.** Les GeForce n'ont pas d'unités matricielles FP64, et FP16, BF16, TF32 et FP8 sont moins précis que le float32 ;
- **pas l'EOTF.** Les fonctions transcendantes n'ont pas une forme matricielle ;
- **pas l'iGPU Raphael.** Le RDNA2 n'a pas d'instructions matricielles ;
- **pas TensorRT.** C'est de l'inférence non exacte.

## Points de décision

- **Après A2' (décidé le 16/10) :** les trois modes sont conservés. Lent : exact certifié (43–51 i/s sur RTX). Standard : ≤ 1 code 16 bits, plus rapide (67 i/s). Rapide : float32 sur tous GPU.
- **Après B0 :** go ou arrêt de B1, selon le critère du double débit.
- **Après T0 :** go ou arrêt de T1, selon le critère du double débit sur RTX. T0 se mesure sur la seule RTX : la GTX 1070 et les UHD 630 et 730 n'ont pas d'unités matricielles, et aucun Arc n'est disponible.
- **A2 :** l'installation de Gappa dans le distrobox de développement sera demandée avant ce lot.

## Risques

| Risque | Parade |
|---|---|
| Borne sous-estimée : code faux silencieux | Bornes démontrées avec marge ; mode diagnostic qui compare chaque image à C1 ; critère de 0 différence sur stress, réel et attestation |
| Optimisations du pilote GPU sur les transformations exactes | Pas de float-float sur GPU ; reprises fines sur CPU (compilation maîtrisée, sans contraction) |
| Synchronisation GPU → CPU pour les reprises | Liste minuscule (moins d'un pixel sur 10⁶ attendu en A1) ; rapatriement groupé par image ; coût mesuré |
| Écart entre l'ancienne référence et la référence certifiée | Écarts comptés et expliqués (proximité d'une frontière) avant de changer de référence |
| Extension `VK_KHR_cooperative_matrix` absente du contexte FFmpeg importé, ou non activable dans libplacebo | Activation à la création du périphérique autonome ; pipeline de calcul dédié ; repli sur le mapping en double |
| Taille de la table (512 Kio) | Intégrée au plugin et partagée par toutes les images ; générateur et empreinte versionnés |

## Avancement

| Lot | État |
|---|---|
| C0 | Fait : table des seuils (parties haute et basse), `scripts/generate_pq_thresholds.py` |
| C1 | Fait : [CPU certifié](2026-10-16-cpu-certifie.md) |
| A1 | Fait : [Lent GPU certifié](2026-10-16-gpu-exact-certifie.md), identique au CPU certifié ; débit +20 à 25 % (au lieu de ×2 : l'EOTF en double domine, cible d'A2) |
| B0 | Fait : [banc](2026-10-16-b0-entiers-multimots.md) ; Raphael 0,54× (critère 2× non atteint) : **B1 arrêté**. RTX : 2,64× plus rapide que le double, candidat pour le mapping d'A2 et référence de T0 |
| R | Fait : [Rapide amélioré](2026-10-16-rapide-ameliore.md) ; stress : 9 151 → 1 code 16 bits (matrices réelles), 1 026 → 112 (identité) ; réel : 1 code ; iGPU −12 % |
| A2' | Fait : [Lent en virgule fixe](2026-10-16-a2-virgule-fixe.md) ; RTX : composition 18,7 → 7,8 ms, Lent 34,4 → 43,6–51,4 i/s, 0 reprise sur scènes réelles ; iGPU : double conservé (choix par auto-test) |
