# Statistiques de l'image produite

Capacité `frame_statistics` (0.3.0), fonction `mvo_fel_set_frame_statistics(contexte, chemin)`,
appelée avant `mvo_fel_run` ou `mvo_fel_run_fd`. Le transport direct (`mvo_fel_next_layers`) la refuse.

Le plugin mesure chaque image reconstruite, telle qu'elle est encodée. Muxiveo s'en sert pour
recalculer le HDR10+ et les MaxCLL/MaxFALL d'une sortie dont le FEL est intégré : les
métadonnées de la source décrivent la BL, pas l'image reconstruite.

## Mesure

- Entrée : codes RGB PQ 16 bits (BT.2020) de l'image reconstruite, avant la conversion Y'CbCr,
  sur la fenêtre encodée (recadrage compris pour les sorties Y'CbCr, image entière en RGB),
  limitée à la zone active du RPU de l'image (L5) : les bandes noires sont exclues, y compris
  quand le format change en cours de film. Sans L5, ou si l'intersection est vide, fenêtre
  encodée entière.
- Lumière linéaire par une table EOTF PQ de 65 536 valeurs float (nits), calculée en double.
- maxRGB = max(R, G, B) ; luminance Y = 0,2627 R + 0,6780 G + 0,0593 B, en float, sans
  contraction FMA (`precise` sur GPU, `-ffp-contract=off` sur CPU).
- Histogramme de maxRGB : 1 024 cases, case = code PQ de maxRGB décalé de 6 bits.
- Histogramme de Y : 1 024 cases bornées par l'EOTF des codes 64·i (nits).
- Somme de maxRGB linéaire en float par ligne, colonnes dans l'ordre, puis en double entre lignes.

Le GPU et le CPU suivent exactement ce calcul : leurs fichiers sont identiques au bit près
pour des images RGB identiques (mode exact).

## Fichier (JSON Lines, UTF-8)

Première ligne :

```json
{"format":"mvo-fel-frame-statistics","version":1,"bins":1024,"width":3840,"height":1600}
```

Une ligne par image produite, dans l'ordre :

```json
{"frame":0,"scene_start":true,"max_nits":[1563.6,1054.1,921.5],"avg_nits":13.0,"y9999_nits":426.5,"y100":0.97,"maxrgb_percentiles":[0,0.05,4,14.4,29.2,42.4,593],"window":[0,280,3840,1600]}
```

- `scene_start` : drapeau de changement de plan du RPU Dolby Vision de l'image (faux pour une
  image sans RPU propre, dont la courbe est héritée).
- `max_nits` : maxima de R, G, B (nits) ; `avg_nits` : moyenne de maxRGB (nits).
- `y9999_nits` : borne haute de la case de Y atteinte à 99,99 % des pixels.
- `y100` : part des pixels de luminance ≤ 100 nits.
- `maxrgb_percentiles` : percentiles de maxRGB de l'image (nits, centre de la case atteinte)
  à 1, 25, 50, 75, 90, 95 et 99,98 %. Le HDR10+ en fait par plan la moyenne (1 à 95 %) et
  le maximum (99,98 %) : règle de l'analyseur de référence, vérifiée sur une source réelle.
- `window` : fenêtre mesurée (x, y, largeur, hauteur), en pixels de la source.

À la fin de chaque plan (changement de plan suivant ou fin du flux) :

```json
{"scene_first":0,"frames":120,"maxrgb_hist":[[0,1520],[312,88004]]}
```

`maxrgb_hist` liste les cases non vides `[case, nombre de pixels]` de l'histogramme de maxRGB
cumulé sur le plan. Le premier plan commence à la première image produite, même sans drapeau.
