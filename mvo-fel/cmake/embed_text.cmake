# Intègre un fichier texte en chaîne C terminée par un octet nul (INPUT → OUTPUT, symbole SYMBOL).
file(READ "${INPUT}" bytes HEX)
get_filename_component(name "${INPUT}" NAME)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${bytes}")
file(WRITE "${OUTPUT}" "/* Généré depuis ${name} : ne pas modifier. */\nconst char ${SYMBOL}[] = {${bytes}0x00};\n")
