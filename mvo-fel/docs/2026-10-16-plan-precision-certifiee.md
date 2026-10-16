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
| **A2** | Étage float32 du Standard | Étage float32 avant A1 ; borne float32 **démontrée** (Gappa, à installer dans le distrobox de développement) ; liste compacte vers A1 | 0 différence avec C1 ; débit proche du Standard actuel sur GPU dédié |
| **V** | Validation et livraison | Attestation de schéma 5 (référence certifiée, compteurs de reprises) ; documentation ; textes de Muxiveo (débits et précisions) | Attestation réussie ; tests Muxiveo et natifs verts |

Ordre d'exécution : C0 → C1 → A1 → B0 → R → A2 → B1 (si go) → V. Chaque lot est commité séparément puis audité.

## Points de décision

- **Après A1 :** si A2 donne au Standard l'exactitude du Lent pour un débit proche, les modes Lent et Standard deviennent tous deux exacts. Faut-il les garder distincts (Lent sans étage float32, Standard avec), ou les fusionner ? La décision revient à l'utilisateur, au vu des mesures.
- **Après B0 :** go ou arrêt de B1, selon le critère du double débit.
- **A2 :** l'installation de Gappa dans le distrobox de développement sera demandée avant ce lot.

## Risques

| Risque | Parade |
|---|---|
| Borne sous-estimée : code faux silencieux | Bornes démontrées avec marge ; mode diagnostic qui compare chaque image à C1 ; critère de 0 différence sur stress, réel et attestation |
| Optimisations du pilote GPU sur les transformations exactes | Pas de float-float sur GPU ; reprises fines sur CPU (compilation maîtrisée, sans contraction) |
| Synchronisation GPU → CPU pour les reprises | Liste minuscule (moins d'un pixel sur 10⁶ attendu en A1) ; rapatriement groupé par image ; coût mesuré |
| Écart entre l'ancienne référence et la référence certifiée | Écarts comptés et expliqués (proximité d'une frontière) avant de changer de référence |
| Taille de la table (512 Kio) | Intégrée au plugin et partagée par toutes les images ; générateur et empreinte versionnés |

## Avancement

| Lot | État |
|---|---|
| C0 | Fait : table des seuils (parties haute et basse), `scripts/generate_pq_thresholds.py` |
| C1 | Fait : [CPU certifié](2026-10-16-cpu-certifie.md) |
