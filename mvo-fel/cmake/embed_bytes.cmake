# Intègre un fichier binaire en tableau C (INPUT → OUTPUT, symbole SYMBOL).
file(READ "${INPUT}" bytes HEX)
get_filename_component(name "${INPUT}" NAME)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${bytes}")
file(WRITE "${OUTPUT}" "/* Généré depuis ${name} : ne pas modifier. */\nconst unsigned char ${SYMBOL}[] = {${bytes}};\n")
