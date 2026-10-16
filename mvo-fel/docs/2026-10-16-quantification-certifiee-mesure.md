# Quantification certifiée : mesure du taux de valeurs ambiguës

16 octobre 2026. Première étape de la piste retenue dans la [proposition](2026-10-16-gpu-exact-proposition.md) : mesurer, avant toute implémentation, combien de valeurs un certificat laisserait indécidées.

## Méthode

Outil local `ambiguity.cpp`, qui n'est pas livré.

Pour chaque pixel et chaque composante, il calcule en double :

- la valeur linéaire de référence R ;
- son code k = round(PQ(R) · 65535) ;
- les seuils T(k ± ½) = EOTF((k ± ½)/65535).

Un certificat de précision relative ε encadre R à ± ε·S près, avec :

S = Σₖ |M_ck| · Lₖ · (1 + κₖ)

- M : matrice LMS→RGB, dont l'annulation est ainsi prise en compte ;
- κ : conditionnement relatif de l'EOTF.

Une composante est ambiguë si son encadrement chevauche un seuil.

Le corpus compte six scènes FEL réelles : Il faut sauver le soldat Ryan ×2, Toy Story 5 et Moana ×3, à raison de 3 images UHD par scène (images 0, 8 et 16).

## Résultats

| ε | Composantes ambiguës | Pixels | Blocs 8 × 4 touchés |
|---|---:|---:|---:|
| 10⁻⁵ | 18–50 % | 42–83 % | 75–100 % |
| 10⁻⁶ | 1,8–5,0 % | 5,4–14,2 % | 68–94 % |
| 10⁻⁷ | 0,18–0,49 % | 0,55–1,47 % | 16–37 % |
| 10⁻⁸ | 0,018–0,050 % | 0,05–0,15 % | 1,7–4,6 % |
| 10⁻⁹ | ≤ 0,005 % | ≤ 0,015 % | ≤ 0,48 % |
| 10⁻¹⁰ | ≤ 0,0005 % | ≤ 0,0016 % | ≤ 0,05 % |
| 10⁻¹² | ≈ 0 | ≈ 0 | ≤ 0,001 % |

La part ambiguë est proportionnelle à ε : environ 4·10⁴ · ε des composantes.

## Conséquences

1. **Un certificat float32 ne suffit pas seul.** Une borne float32 rigoureuse vaut environ ε ≈ 10⁻⁶ : elle laisse 5 à 14 % des pixels à recalculer.
2. **Les exceptions doivent passer par une liste compacte.** À ε = 10⁻⁶, 68 à 94 % des blocs de 32 pixels contiennent une exception : une branche dans le shader principal coûterait presque autant que le recalcul complet.
3. **Pour que les exceptions deviennent négligeables, il faut ε ≤ 10⁻⁸.** Cela suppose plus que la simple précision : entiers sur deux mots, ou double.
4. **Cette mesure exclut l'erreur du mapping.** Les mappings MMR réels présentent des annulations : en float32, le prototype standard recalculait 99,9 % des pixels de Ryan (borne du mapping). Le mapping reste donc en précision étendue (double ou entiers multi-mots). C'est lui qui coûte 56 ms par image sur l'iGPU Raphael.

## Architectures possibles

| Variante | Calcul | RTX (estimation) | iGPU Raphael (estimation) | GPU sans double |
|---|---|---|---|---|
| Lent actuel | double partout | 27,6 i/s mesurés | 2,1 i/s mesurés | CPU |
| A | mapping en double ; EOTF et matrice en float32 sous borne rigoureuse ; classement par seuils PQ ; liste compacte recalculée en double | proche du standard (≈ 65 i/s) | ≈ standard (≈ 8 i/s), mapping en double dominant | CPU |
| B | mapping et certificat en entiers multi-mots | à mesurer | à mesurer, gain possible | possible |

Les estimations de A reposent sur les mesures du prototype standard : environ 4 ms de composition sur la RTX, et une liste compacte de 5 à 14 % de pixels recalculés en double, soit environ 3 ms à 22 ms par image entière.

La variante A rendrait le mode Lent identique au bit près à la vitesse du standard sur GPU dédié, le principal cas d'usage. Elle exige :

- une analyse d'erreur rigoureuse de l'étage float32 (EOTF, matrice), de type Gappa ;
- la table des seuils PQ ;
- une passe compacte (comptage atomique et liste d'indices).

La variante B est la seule à pouvoir aider l'iGPU et les GPU sans double ; c'est une recherche plus longue.
