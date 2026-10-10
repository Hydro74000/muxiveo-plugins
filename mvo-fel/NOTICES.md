# Bibliothèque FEL

Le code C++ de cette bibliothèque est distribué sous LGPL-2.1-or-later.
Le code Python de Muxiveo conserve sa licence MIT.

Les calculs de reconstruction sont un port CPU des fonctions de libplacebo :
`pl_shader_dovi_reshape`, `sh_dovi_compose_nlq`, `pl_shader_decode_color_ex`,
`pl_map_dovi_metadata`, `pl_color_repr_decode`, `spline16` et du positionnement
des couches dans `sample_el`. Copyright des contributeurs de libplacebo ;
licence LGPL-2.1-or-later. La révision est épinglée dans `dependencies.json`.
Sources : https://github.com/haasn/libplacebo.

FFmpeg : copyright des contributeurs de FFmpeg, LGPL-2.1-or-later ; aucune
option GPL/nonfree n'est activée dans la construction embarquée.
Sources : https://github.com/FFmpeg/FFmpeg.

libdovi (`dolby_vision`) : copyright quietvoid et contributeurs, MIT.
Sources : https://github.com/quietvoid/dovi_tool.

Runtimes GCC (libgcc/libstdc++, constructions GCC) : GPL-3.0 avec exception
GCC Runtime Library Exception 3.1 ; textes dans LICENSES/GPL-3.0.txt et
LICENSES/GCC-RUNTIME-EXCEPTION.txt. La construction Windows MinGW inclut aussi
winpthreads (notices dans LICENSES/MinGW-winpthreads.txt).

Aucun code de FelBaker ou DoViBaker n'est intégré.

Prototype Vulkan : libplacebo est liée statiquement, avec le compilateur GLSL
glslang (copyright Khronos Group et contributeurs). Notices complètes dans
LICENSES/glslang.txt ; ses parties optionnelles et SPIRV-Tools ne sont pas
construits. Le pilote Vulkan reste celui du système, chargé dynamiquement.
# Transport GPU direct expérimental

Le binaire privé `ffmpeg-fel` est construit depuis la même révision FFmpeg,
avec deux filtres propres à mvo-fel sous LGPL-2.1-or-later. Les en-têtes
nv-codec-headers 13.0.19 sont épinglés dans `dependencies.json`, sous MIT
(notice originale dans `LICENSES/nv-codec-headers.txt`). Aucun SDK binaire
NVIDIA propriétaire n'est redistribué ; le pilote fourni par la machine
reste nécessaire. Le binaire ne remplace aucun FFmpeg installé.
