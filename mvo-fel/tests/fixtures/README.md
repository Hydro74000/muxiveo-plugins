# Séquence synthétique

`synthetic-fel.hevc` contient 12 images grises, BL 256×144, EL 128×72,
24000/1001 i/s, GOP 12 avec trois B-frames, chroma centré.
Les couches ont été générées avec FFmpeg/libx265 (`color=gray`, `chromaloc=2`),
puis assemblées avec `dovi_tool inject-rpu` et `dovi_tool mux`.
Le RPU de test `assets/tests/fel_orig.bin` du dépôt dovi_tool de la révision libdovi
épinglée est répété pour les 12 images (sources sous MIT). Seuls les offsets L5
sont remis à zéro : ceux de l'image UHD d'origine dépassaient le cadre 256×144.
Le mapping FEL reste intact. `../make_fixture.py` reproduit ce fichier depuis
le RPU épinglé avec FFmpeg/libx265 et dovi_tool installés.

Cette séquence valide le transport et les associations de couches. Elle ne
remplace ni la référence numérique, ni la validation visuelle d'un FEL réel.

`synthetic-fel-mixed-gop.hevc` est la même séquence avec une BL sans B-frames
(`make_fixture.py --bl-bframes 0`) face à l'EL à trois B-frames : flux non
conforme dont les couches d'une même access unit ne sont pas affichées au même
instant. Le plugin les apparie par rang de présentation et doit produire
exactement l'image de `synthetic-fel.hevc`.
