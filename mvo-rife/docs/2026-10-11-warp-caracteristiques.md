# Warp des caractéristiques sans conversions intermédiaires

Prototype de recherche à partir de `b7ad9ca`, sans modification des poids.
Les caractéristiques des têtes v4.15 et Light restent compactées par quatre
autour des Warps Vulkan. Les coordonnées sont partagées entre quatre composantes,
mais chaque composante conserve le calcul scalaire du chemin précédent.

Une première version à calcul vectoriel introduisait de petits écarts locaux,
amplifiés par la suite du réseau, surtout en FP16. Elle est écartée. La version
scalaire conserve les 60 sorties brutes comparées sur RTX et les 60 sur AMD :
v4.6, v4.15-mvo1 et Light, FP16/FP32, normal/UHD, cinq positions temporelles.
Ces contrôles portent sur 128×64 ; ils ne remplacent pas les mesures vidéo 4K.

Le flux conserve son propre pas et reste scalaire. Un flux compacté fourni par
un autre graphe est décompacté avant lecture. Le chemin précédent est conservé
en l'absence de stockage FP16 uniforme ou si le nouveau pipeline ne peut pas
être créé. Le chemin CPU reste inchangé.

Seize tests à translation connue passent sur RTX, AMD et Lavapipe : huit
caractéristiques, deux groupes de canaux, sentinelles, flux à deux ou quatre
canaux, déplacements fractionnaires ou hors image, FP16/FP32. Ils couvrent la
conservation des composantes et le padding, indépendamment des réseaux appris.

L'optimisation concerne les Warps de caractéristiques v4.15/Light, y compris
UHD, TTA et Ultra existants. v4.6 conserve ses Warps RGB scalaires. TensorRT
conserve son graphe et son backend. Aucun gain de vitesse n'est encore annoncé :
références natives, ×2 et ×2,5, répétitions alternées et autres GPU/OS doivent
précéder une adoption. L'Ultra adaptatif reste un chantier ultérieur.
