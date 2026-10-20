# Un fragment shader raymarche tout le fond

Oui, Python permet un fond fait uniquement de formes SDF raymarchées en temps réel, parce que Python n'y fait presque rien : tout le calcul tient dans un fragment shader GLSL exécuté sur un triangle plein écran, et l'hôte n'envoie que quelques dizaines d'uniforms par frame, comme il le fait déjà pour les ~80 floats de features du modèle. La vraie contrainte est le GPU Apple (**M1 ≈ 2,6 TFLOPS FP32**) derrière un OpenGL **plafonné à 4.1 et posé sur Metal** ([Geeks3D](https://www.geeks3d.com/?p=10147)). L'intégration la plus simple et la plus sûre consiste à rendre le fond **à part** : une texture HDR non-MSAA à demi-résolution, insérée au composite *sous* la scène via son alpha, avant AgX. Le fond reste ainsi hors du MSAA et hors des trainées, et le modèle reste toujours devant sans écrire de profondeur. Les rayons se construisent à partir des **mêmes matrices P·V** que le modèle, ce qui donne une parallaxe exacte avec la caméra orbitale (T26). Un simple uniform bascule entre 2D écran, 2,5D, 3D fixe et 3D liée à la caméra. Les quatre familles demandées (fractales, organique, géométrique répété, paysages) se rangent en **un programme compilé par mode**, partageant le même raymarcher, l'éclairage et la sortie ; leurs sous-variantes passent par des uniforms et des poids de mélange, sans recompilation. Les paramètres, nombreux, s'organisent en trois niveaux (globaux, propres au mode, macros), avec un dé à seed, des règles de morphing par paramètre et un schéma JSON qui génère à la fois les uniforms, l'interface et les presets. La règle audio qui découle de toutes les sources : les signaux rapides (KICK, SNARE) ne touchent que des paramètres « exacts » (rotations, rayons, émissifs) ; les paramètres qui changent la topologie ne reçoivent que l'ÉNERGIE de section ou des macros. Il reste une inconnue majeure : **aucun benchmark de raymarching sur GPU Apple en OpenGL n'a été trouvé**, donc le budget doit être mesuré dans le projet avec `GL_TIME_ELAPSED` avant de figer les niveaux de qualité.

## Python n'envoie que des uniforms : le goulot est le GPU d'Apple

Les outils qui font tourner du raymarching type Shadertoy depuis Python ont tous la même architecture. `einarf/shadertoy` est une implémentation de Shadertoy sur moderngl ([GitHub](https://github-redirect.dependabot.com/einarf/shadertoy)). `wgpu-shadertoy`, chez pygfx, accepte du GLSL ou du WGSL, mais se déclare encore « en développement » et ne gère ni `sampler3D` ni `samplerCube` ([GitHub](https://github.com/pygfx/shadertoy)). À l'inverse, un raymarcher en NumPy/PIL reste hors du temps réel ([PashaWNN/raymarching](https://github.com/PashaWNN/raymarching)), et `fogleman/sdf` évalue ses fonctions de distance sur le CPU pour **produire des maillages STL** par Marching Cubes, pas un fond animé ([GitHub](https://github.com/fogleman/sdf)). La leçon est nette : tant que le calcul vit dans un fragment shader, le langage hôte n'est pas le facteur limitant. moderngl offre déjà ce qu'il faut pour piloter et mesurer ce shader : `ctx.query(time=True)` (requête `GL_TIME_ELAPSED`, temps GPU réel de la passe), `ctx.includes` pour factoriser des blocs GLSL, et un multisampling réglé par attachement ([moderngl, Context](https://moderngl.readthedocs.io/en/latest/reference/context.html)).

Le plafond technique vient de macOS. Sur Apple Silicon, `GL_VERSION` vaut « 4.1 Metal », avec seulement 43 extensions en contexte core ([Geeks3D](https://www.geeks3d.com/?p=10147)). OpenGL est déprécié depuis Mojave, et Apple n'y fait plus que des « changements mineurs » ([AppleInsider](https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave)). Les compute shaders datent de GL 4.3 : ils sont donc absents, ce qui est une déduction du plafond et pas une affirmation trouvée telle quelle. Tout passe donc par des fragment shaders et, si besoin, par des ping-pong de textures. Côté vitesse, la seule mesure disponible est un proxy : le shader lourd « Neural Stanford Bunny » tourne à **245 fps en 800×480 sur M1, contre 720 fps sur une RTX 2070** ([Geeks3D](https://www.geeks3d.com/?p=10147)). Cela fait environ 4 ms pour 0,38 Mpx, soit de l'ordre de 10 ns par pixel pour un gros shader. Ce n'est pas un raymarcher. L'extrapolation (un fond SDF simple coûterait 2 à 4 ms en demi-résolution Retina, et 8 à 16 ms en pleine résolution) est un calcul, pas une mesure, mais elle suffit à fixer le principe : **demi-résolution par défaut, pleine résolution seulement pour le 2D**. Des applications réelles gagnent 1,8 à 2× en passant d'OpenGL à Metal sur Apple Silicon ([CurseForge, Pure Metal](https://www.curseforge.com/minecraft/mc-mods/pure-metal)) : la marge existe, mais elle relève de T28 (wgpu-py), pas de T16.

Python garde trois rôles utiles. Le premier : calculer chaque frame les uniforms, dont les sorties du `Reactor`, les matrices caméra, les rotations Rodrigues des fractales et les positions des métaballes. Le deuxième : recharger à chaud les fichiers `.glsl` en recompilant dans un `try/except moderngl.Error`, sans remplacer le programme courant tant que le nouveau n'a pas compilé. C'est exactement la règle du projet « ne jamais remplacer la source courante avant que la nouvelle soit valide ». Le troisième : décrire les paramètres dans un module sans GL, donc testable. Deux pièges propres au pilote Apple justifient ce garde-fou. D'abord, les binaires de programme ne sont pas une option fiable sur macOS : le pilote peut n'annoncer **aucun format**, et il faut alors tout compiler depuis le source ([Khronos forum](https://community.khronos.org/t/binary-shaders-support-in-opengl/68264)). Ensuite, les compilateurs inlinent souvent quatre fois `map()` dans le calcul de normale. Écrire ces quatre appels dans une boucle bornée par un « zéro non constant » évite plantages et compilations lentes, sans coût à l'exécution ([iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/)).

## Le fond se glisse sous la scène, hors du MSAA et des trainées

Le tiquet actuel prévoit de dessiner le décor « avant le modèle, dans la passe scène (MSAA) » et d'écrire la profondeur. L'analyse du renderer conduit à une autre solution, moins chère et plus robuste. La boucle actuelle dessine le modèle dans `_accum` (MSAA 4×, f16), le résout, puis fusionne avec la couche trainées par `mix(max(t.rgb, s.rgb), s.rgb, s.a)` avec un alpha `max(t.a, s.a)`. Le bloom part de cette couche et le composite AgX ajoute ensuite `u_bg * (1 − a)` (`gfx/renderer.py`, `TRAIL_FS` et `COMPOSITE_FS`). Mettre le raymarch dans le MSAA aurait trois défauts. Il multiplierait par 4 la bande passante d'écriture et le resolve, sans antialiaser les bords SDF, puisque le fragment shader ne s'exécute qu'une fois par pixel sans *sample shading*. Le fond entrerait dans la couche trainées et « baverait » à chaque mouvement de caméra. Enfin, il imposerait `gl_FragDepth`, dont la formule est piégeuse : il faut projeter le point touché et faire la division perspective. Une normalisation linéaire de la profondeur ou un mélange d'espaces décale l'intersection ([three.js forum](https://discourse.threejs.org/t/raymarching-correct-gl-fragdepth/66033) ; [GameDev.net](https://gamedev.net/forums/topic/622531-matching-raymarcher-and-polygon-depth-values/)).

La solution recommandée ajoute une passe `draw_background()` dans une texture RGBA f16 non-MSAA, à `bg_scale × taille interne` (0,5 par défaut). Elle est insérée **en HDR linéaire avant AgX**, par `hdr = trail.rgb + bg.rgb * (1 − trail.a)`. La lecture du code montre que cette formule se comporte comme un « over » prémultiplié cohérent avec les trainées : la couleur et l'alpha d'une surface opaque qui s'éloigne décroissent au même rythme `decay^n`, si bien que la trace du modèle se fond proprement dans le fond, sans silhouette sombre. Les points additifs (alpha 0) brillent par-dessus le fond, et les surfaces opaques (alpha 1) le masquent. Pour que les émissifs du fond nourrissent le bloom, il suffit d'échantillonner la même texture dans le premier niveau de downsample, avec un gain dédié `u_bg_bloom`. La demi-résolution n'a pas besoin d'upscale « depth-aware », puisque le modèle est toujours dessiné par-dessus. On peut aussi dessiner le fond après le resolve, sauter les pixels déjà couverts par un modèle opaque (minimum de l'alpha sur les 4 pixels du bloc), et récupérer ainsi le coût caché derrière le modèle. Cette optimisation est déduite de la structure du pipeline et n'a pas été mesurée. L'option « formes SDF qui passent devant ou coupent le modèle » reste possible plus tard : on recopierait couleur et `gl_FragDepth` dans `_accum` avant le modèle. Mais elle réintroduit trainées, MSAA et z-fighting ; avec near = 10 et far = 8000 en profondeur 24 bits, la précision au loin est faible. Elle ne doit pas faire partie de T16.

La caméra partagée se règle en reconstruisant chaque rayon depuis l'inverse de la vue-projection du modèle :

```glsl
uniform mat4 u_inv_vp;  uniform vec3 u_eye;          // depuis P·V du renderer
vec3 ray_dir(vec2 uv) {                              // uv ∈ [0,1]
    vec4 w = u_inv_vp * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    return normalize(w.xyz / w.w - u_eye);
}
```

Avec cette forme, la « Secousse caméra » (qui modifie la translation de V) est prise en compte automatiquement. Le monde du projet est en « pixels » (cam_dist ≈ 869 pour h = 720). Le fond doit donc travailler dans ses propres unités, `p_fond = p / S`, en multipliant par S la distance retournée, sans jamais mélanger les deux échelles. L'epsilon de touche et celui des normales doivent suivre l'empreinte du pixel (∝ t) : un epsilon constant donne des normales bruitées et aliasées au loin ([iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/)).

Le même shader porte quatre rendus, sélectionnés par uniform, donc sans recompilation, puisque tous les pixels prennent la même branche. En **3D caméra**, les rayons viennent de la vue orbitale, ce qui donne une parallaxe réelle. En **3D fixe**, ils viennent de la vue de départ `translate(0,0,−cam_dist)` : le fond ne réagit plus à l'orbite. En **2,5D**, le rayon est intersecté analytiquement avec N plans portant une SDF 2D : on obtient de la parallaxe sans boucle de marche. En **2D**, la SDF 2D est évaluée directement sur les coordonnées écran, avec un antialiasing par `fwidth`. C'est quasi gratuit et c'est le seul rendu raisonnable en pleine résolution. Un paramètre « Parallaxe » entre 0 et 1 relie progressivement la caméra du fond à l'orbite. Pour l'interpolation, il faut mélanger les paramètres (lacet, tangage, distance), puis reconstruire `look_at`, et non mélanger des matrices, car un lerp de rotations n'est plus orthonormé. L'`OrbitCamera` du projet, avec son retour en douceur, fournit déjà la mécanique. Une bascule 2D↔3D se fait par un fondu `u_mix` de 0,5 à 1 s : les deux chemins ne sont payés que pendant le fondu.

## Quatre familles, une boucle de marche commune, un programme par mode

La question « uber-shader ou permutations » a une réponse claire pour cinq familles de coût très inégal. Un uber-shader paie les registres du pire chemin même quand un petit chemin s'exécute, ce qui réduit l'occupation du GPU ([gamedev.net](https://gamedev.net/forums/topic/715362-why-shader-permutation/)). De plus, une erreur GLSL dans un mode casse alors tous les modes. La voie médiane courante est de spécialiser à la compilation par macros et de ne garder que des branches sur uniforms ([gamedev.net](https://www.gamedev.net/forums/topic/512455-shader-permutation-problem--again-/?page=1)). Concrètement, chaque mode a sa fonction `map()` dans `fond/<mode>.glsl`, injectée par `#define MODE_x` dans un squelette commun : raymarcher sur-relaxé, normale tétraédrique, éclairage, palette, glow, atténuation de distance. Les programmes sont précompilés au démarrage ou étalés à raison d'un par frame, puisque le contexte GL est mono-thread. Les sous-variantes d'un mode restent des uniforms ou des poids `mix()`.

Ce découpage tient aussi parce que **le mélange de deux SDF 1-Lipschitz reste une borne valide**. Le gradient de `t·a + (1−t)·b` vaut au plus t·1 + (1−t)·1 = 1. La forme intermédiaire est donc un vrai morphing géométrique et non un fondu d'images, au prix de deux évaluations pendant la transition seulement. Toute la sécurité du sphere tracing repose sur cette propriété. hg_sdf en fait sa règle de conception, « rester Lipschitz continu », avec un gradient ≤ 1 ([hg_sdf](https://mercury.sexy/hg_sdf/hg_sdf.glsl)). Rotations, translations, échelle uniforme, arrondi, *onion* et élongation sont exacts ([iq, distance functions](https://iquilezles.org/articles/distfunctions/)). Twist, bend, déplacement et domain warping « distordent le champ » et imposent de réduire le pas en proportion de la compression de l'espace ([iq, distance functions](https://iquilezles.org/articles/distfunctions/)). Chaque mode doit donc accumuler un facteur L et retourner `d / L`.

### Fractales : un gabarit fold–scale–rotate couvre presque toutes les familles

Deux motifs de distance estimée couvrent presque tout le catalogue. Le motif « escape-time » (Mandelbulb, Julia quaternion) retourne `0.5·log(r)·r/dr`. Le motif « IFS / pliage » (Mandelbox, KIFS Menger, Sierpinski, octaèdre, icosaèdre, pseudo-Kleinian, Amazing Surf, Kalibox) suit un facteur d'échelle courant dans `p.w`. Les sources de Fragmentarium, écrites par Syntopia, Knighty, Kali et Rrrola, donnent les boucles exactes et des plages de curseurs éprouvées. Le Mandelbulb utilise Power 0–16 (8 standard), Bailout 5 et 9 à 14 itérations, avec une **rotation appliquée à chaque itération**, déjà utilisée comme animation (`RotAngle + time*10`) ([Mandelbulb.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbulb.frag)). Le Mandelbox utilise Scale −3…5 (3 donne un Menger parfait) et MinRad2 0–2 (défaut 0,25). Ses presets de référence sont Scale 2,39 ; −1,49 et 2,04 ([Mandelbox.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag)). Le Menger KIFS utilise Scale 0–4, un Offset dans [0,1]³ et 8 itérations ; un preset « symétrie brisée » obtient un tout autre objet avec une rotation de seulement **2,1°** ([Menger.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Menger.frag)). Pour l'icosaèdre, toute la chaîne rotation–translation–échelle–translation–rotation se replie côté CPU en **une seule `mat4`**, d'où une multiplication matricielle par itération ([Icosahedron.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kaleidoscopic%20IFS/Icosahedron.frag)). Le pseudo-Kleinian ne demande que **5 itérations** par défaut, ce qui en fait la fractale « riche » la moins chère ([PseudoKleinian.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Knighty%20Collection/PseudoKleinian.frag)).

Ces boucles partagent un squelette : rotation, plis, pli sphérique, échelle autour d'un offset ou ajout d'une constante de Julia, rotation. Un seul programme « Fractale » peut donc couvrir KIFS, Mandelbox, Kalibox, Amazing Surf et pseudo-Kleinian avec des **poids de pli entre 0 et 1**. Living KIFS montre l'étendue de cet espace avec un pli décalé sur 1, 2 ou 3 axes, `p*Sc+Julia`, une rotation dans la boucle et un terme « Fatness » qui épaissit la forme ([LivingKIFS.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Kali%27s%20Creations/LivingKIFS.frag)). Ce même preset exige pourtant un facteur de pas de **0,35 et 308 pas** : mélanger des plis casse l'exactitude de la distance, d'où un facteur de pas exposé en qualité. Le Mandelbulb, le plus cher (acos, atan, pow, sin et cos à chaque itération), mérite sa propre sous-variante, limitée à 6–8 itérations. La couleur est presque gratuite. Un orbit trap `min(trap, abs(vec4(p, r²)))` sur les premières itérations, passé dans la palette cosinus d'iq `a + b·cos(2π(c·t + d))`, suit la structure de la fractale, et le trap de l'origine simule une occlusion ambiante ([iq, Mandelbulb](https://iquilezles.org/articles/mandelbulb/) ; [iq, palettes](https://iquilezles.org/articles/palettes/)).

### Organique : métaballes, surfaces minimales triplement périodiques, corail

Trois familles organiques bon marché donnent des looks très différents. Les **métaballes** (6 à 16 sphères en `uniform vec4[16]`, orbites calculées en Python) se fusionnent de préférence par le smin exponentiel. C'est la seule variante qu'on peut appliquer dans n'importe quel ordre, alors que le smin quadratique, le défaut d'iq pour deux objets, dépend de l'ordre ([iq, smooth minimum](https://iquilezles.org/articles/smin/)). Grâce à la normalisation d'iq, **k vaut directement l'épaisseur du raccord en unités de distance**, ce qui en fait un curseur perceptuellement linéaire, idéal pour l'audio. Les **surfaces minimales triplement périodiques (TPMS)** remplissent l'espace sans artefacts de répétition. Le gyroïde, le diamant et Schwarz P ont un gradient borné par √3 ; cela se vérifie analytiquement et la note de recherche l'a confirmé numériquement. `abs(f(s·p) − bias)/(√3·s) − épaisseur` est donc une borne sûre, plus rapide que le `*0.5` courant sur Shadertoy. Le biais déplace la surface de niveau : près de 0, deux labyrinthes égaux ; à |biais| élevé, la surface se rompt en poches isolées. Un seul float change ainsi la topologie. Les familles de TPMS se déforment d'ailleurs continûment les unes dans les autres ([Wikipedia, TPMS](https://en.wikipedia.org/wiki/Triply_periodic_minimal_surface)). Enfin, le **fBM-SDF d'iq** (grilles de sphères aléatoires combinées octave par octave par smin/smax) ajoute un détail de corail ou de mousse qui **reste une SDF**, alors qu'ajouter du bruit à une distance « casse rapidement quand on pousse » ([iq, fBM SDF](https://iquilezles.org/articles/fbmsdf/)).

### Géométrique répété : la topologie de répétition fait la différence

La répétition de domaine produit une infinité de copies pour une seule évaluation, mais elle n'est correcte que pour une forme symétrique et identique dans chaque cellule. Toute variation par cellule ou toute forme asymétrique crée des trous, sauf si l'on évalue les voisins : **2, 4 ou 8 évaluations en 1D, 2D ou 3D** ([iq, domain repetition](https://iquilezles.org/articles/sdfrepetition/)). La répétition miroir garde une seule évaluation, et la répétition limitée se fait en bornant l'identifiant de cellule par `clamp`, pas en intersectant avec une boîte. hg_sdf fournit la grammaire complète. `pMod1/2/3`, `pModPolar` et `pModMirror` répètent le domaine. Côté combinaisons, `fOpUnionStairs` donne des terrasses et `fOpUnionColumns` des cannelures ; `fOpPipe` change une intersection en tubes, `fOpEngrave` creuse des rainures ([hg_sdf](https://mercury.sexy/hg_sdf/hg_sdf.glsl)). Sept recettes sortent de ces briques : treillis infini, champ de piliers en égaliseur 3D (une bande EQ par cellule), tunnel polaire répété le long de z, anneaux en gyroscope, cathédrale à colonnes, Truchet 3D et espace plié kaléidoscopique. L'ordre d'impact visuel est, d'après la note de recherche : la topologie de répétition, puis le type de primitive, le k du smin et les itérations de pli. Deux règles de lisibilité s'imposent derrière un modèle central. D'abord, l'atténuation de distance et un TMAX court suppriment le moiré lointain. Ensuite, la période de répétition ne doit jamais être modulée par l'audio, car le monde entier glisse autour de l'origine. On module le contenu des cellules et on fait avancer un défilement monotone.

### Paysages : un heightfield marché par pas proportionnels à la distance

Pour un « sol + ciel », un heightfield `y = h(x,z)` marché avec un pas ∝ t puis raffiné par interpolation linéaire est la voie la moins chère, parce que le détail à l'écran décroît à peu près en 1/distance ([iq, Terrain raymarching](https://iquilezles.org/articles/terrainmarching), lu via un extrait). Seascape montre une autre voie pour l'eau et les dunes. Une version bon marché de la hauteur (3 octaves) sert à la marche, une version détaillée (5 octaves) à l'ombrage, et la recherche par **fausse position ne prend que 8 itérations** ([Ultraviolet, Seascape](https://ultraviolet.indigoengine.io/shadertoy/seascape) ; [Godot Shaders](https://godotshaders.com/shader/seascape-shader/)). Les variantes visuelles ne sont que des fonctions `h` différentes sur la même boucle. Le **fBm** utilise un gain G = 2^−H, avec H = 1 pour des formes naturelles et une lacunarité « désaccordée » (1,99 ou 2,01) pour éviter l'alignement des octaves ([iq, fBM](https://iquilezles.org/articles/fbm/)). Le **fBm érodé** façon Elevated tire parti d'un bruit à dérivées analytiques, environ 5× moins cher que les différences centrales, et d'une ligne `a += b·n.x/(1 + dot(d,d))` qui amortit les octaves sur les pentes ([iq, More noise](https://iquilezles.org/articles/morenoise/)). Le **ridged multifractal** produit des crêtes vives ([Blender, Musgrave](https://docs.blender.org/manual/sl/3.6/modeling/geometry_nodes/texture/musgrave.html)). S'y ajoutent dunes, mer, canyons et un **synthwave** faisable presque sans marche. Le plus musical est le **spectrogramme 3D** : une texture d'historique de 64 bandes × N lignes, alimentée chaque frame par le `spectrum` existant, sert directement de heightfield. Les grottes et tunnels bruités exigent une SDF 3D complète et coûtent nettement plus ; ils doivent rester en qualité haute.

## Trois niveaux de paramètres, un dé à seed et des règles de morphing

Les outils de shaders live convergent tous vers la même structure. Un fichier déclaratif décrit chaque contrôle par un nom qui devient l'uniform, un type, un minimum, un maximum, un défaut et un groupe d'interface, puis l'hôte construit le panneau. C'est vrai pour l'en-tête JSON d'ISF, avec `NAME`, `TYPE`, `MIN`, `MAX`, `DEFAULT`, `VALUES` et `LABELS` ([ISF Spec](https://github.com/mrRay/ISF_Spec)). Synesthesia ajoute au type un **comportement temporel** (`slider smooth`, `slider speed` qui avance en continu, `bang`) et des transitions automatiques ([Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)). Fragmentarium place les annotations `slider[min,défaut,max]` et le verrou `Locked` directement dans le GLSL ([DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)). Synthclipse ne stocke dans un preset que les uniforms qui changent ([Fractal Forums](https://www.fractalforums.com/fragmentarium/slider-t22662)), comme le fait déjà T24 en remettant à leur défaut les réglages absents. Le projet doit reprendre ce modèle avec un schéma JSON par mode, lu par un module Python sans GL ni Qt. Ce module génère les déclarations d'uniforms, construit l'interface, fournit les clés de presets et les cibles de la matrice Physique, et se teste sous pytest.

Un nombre élevé de paramètres n'est pas un problème en soi. Le génome d'un mouton Electric Sheep compte **environ 130 paramètres** et reste évolutif parce que des humains sélectionnent les résultats ([Wikipedia, Electric Sheep](https://en.wikipedia.org/wiki/Electric_Sheep)). Picbreeder applique la même boucle : l'utilisateur choisit les images prometteuses, qui sont croisées et mutées ([UT Austin](https://www.cs.utexas.edu/~risto/cs378ne/private/slides/8.3-8.4.Interactive-4up.pdf)). La difficulté est l'accès aux paramètres, d'où trois niveaux. Le premier regroupe les **globaux du fond** : mode, rendu, qualité, palette, vitesse, intensité, seed. Le deuxième rassemble les **paramètres propres au mode**, déclarés par mode et stockés par mode, si bien qu'on retrouve ses réglages en revenant à un mode. Le troisième est fait de quatre **macros** toujours visibles : Complexité, Chaos, Densité et Organique↔Géométrique. Chaque paramètre déclare la plage qu'il parcourt quand une macro va de 0 à 1. Une macro est aussi une cible de routing : router l'ÉNERGIE vers « Complexité » remplace quinze routes. Aucune source n'a été trouvée sur la conception de macros dans un outil VJ, donc ce troisième niveau est une proposition à valider sur prototype.

Le dé tire chaque paramètre dans une plage `rand` plus étroite que min/max, celle des zones « belles », avec une distribution adaptée : uniforme, logarithmique pour les échelles et fréquences, ou choix pondéré pour les énumérations. Le tirage est déterministe à partir d'un **seed entier**, si bien qu'un preset peut se réduire à `{mode, seed}` plus quelques surcharges. Un cadenas par paramètre l'exclut du tirage. La mutation applique une perturbation gaussienne autour du preset courant ; une grille de 6 à 9 vignettes rendues hors écran par `tools/snapshot.py` reproduit la boucle Picbreeder. Le morphing obéit à une règle connue des synthés : **les paramètres discrets ne s'interpolent pas, ils sautent** ([SonicState](https://www.sonicstate.com/news/2026/02/26/using-snappysnap-with-synths/)). On peut aussi les geler pendant la transition ([Polyend forum](https://backstage.polyend.com/t/morph-between-two-presets/20141)). Chaque paramètre déclare donc un champ `interp` parmi `linear`, `log`, `angle`, `oklab`, `snap`, `freeze`, `crossfade` et `weights`. Synesthesia montre qu'un choix discret devient continu s'il est encodé en famille de poids fondus par `mix()` ([Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)). C'est exactement l'encodage des types de pli fractals et des types de TPMS. Les itérations deviennent continues en mélangeant les itérations n et n+1.

## La musique pilote des vitesses et des émissifs, pas des topologies

Synesthesia classe ses signaux audio en quatre familles qui correspondent à quatre usages. **Level** est un niveau lissé, pour l'échelle et la luminosité. **Hits** regroupe les transitoires, pour les réactions percussives. **Presence** suit les sections qui montent ou chutent, avec « moins de jitter que Hits ». **Time** est une horloge qui accélère avec la bande, pour le mouvement et la caméra. Synesthesia recommande aussi de vérifier la confiance du BPM avant de caler un mouvement sur le beat ([Synesthesia audio uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html)). Le projet couvre déjà trois familles : les bandes EQ lissées par l'enveloppe attack/release de T4 (Level), le ressort KICK et le flash SNARE (Hits), l'ÉNERGIE de section (Presence). Il manque **Time** : des phases intégrées côté Python, `φ += dt·(base + k·niveau)`. Le panneau vit dans un autre process, donc l'intégration en `dt` doit se faire dans le moteur. Il manque aussi un compteur de beats. Magic Music Visuals recommande de son côté un lissage exponentiel « avec modération », sinon la réaction devient léthargique, et un Gate ou un Clamp contre le jitter ([Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=11958)).

Toutes les sources mènent à la même discipline, avec trois règles. La première : un signal rapide ne va jamais vers un paramètre qui change la topologie (itérations, nombre de cellules, type de pli, biais TPMS proche d'un seuil, signe du scale du Mandelbox), sinon un saut plein écran crée un scintillement. Ces paramètres ne reçoivent que l'ÉNERGIE ou une macro, et changent de façon fractionnaire ou quantifiée au beat. La deuxième : on route une bande vers une **vitesse** intégrée, pas vers une position ou un angle, car une position modulée tremble alors qu'une vitesse modulée accélère en douceur. Le projet applique déjà ce principe à la rotation musicale. La troisième : les impulsions KICK et SNARE vont vers ce qui est exact pour la distance (angle de rotation dans la boucle, rayon, arrondi, épaisseur, k du smin) ou vers ce qui ne touche pas la géométrie (émissifs, phase de palette, glow). Un raymarching est sensible à l'aliasing temporel : un petit changement de paramètre fractal fait bouger du détail haute fréquence. Un `slew` (variation maximale par seconde) sur les paramètres de forme est donc indispensable.

Le fond doit enfin rester subordonné au modèle. iq recommande un albédo ≤ 0,2 et un éclairage extérieur à trois termes (soleil ombré, ciel × AO, rebond × AO), moins cher qu'un second Studio GGX ([iq, Outdoors lighting](https://iquilezles.org/articles/outdoorslighting/)). Ce modèle a un autre avantage : il évite que fond et modèle partagent le même éclairage et fusionnent visuellement. Le reste découle de la composition : intensité du fond appliquée avant le tonemap (défaut ≈ 0,35), désaturation, flou gratuit de la demi-résolution, atténuation de distance vers la couleur d'horizon, émissifs plafonnés juste au-dessus du seuil de bloom. On peut ajouter un halo sombre autour de la projection écran du modèle. Le brouillard abandonné comme type de décor survit sous cette forme minimale d'atténuation : elle masque la coupe TMAX et pousse le fond vers l'arrière, pour un coût nul.

## Spécification recommandée pour la réécriture de T16

**Titre proposé :** « T16 — Onglet Fond : formes SDF raymarchées (fractales, organique, géométrique, paysages) ». Priorité moyenne · **Effort : 4 sessions** (l'ancienne estimation de 2 sessions couvrait des décors plus simples) · Impact : profondeur de scène, identité visuelle. L'onglet doit dépendre de T27 (PySide6), placé avant lui dans l'index : comme l'interface est générée depuis le schéma, la construire directement dans le nouveau panneau évite de la faire deux fois.

### Architecture

| Brique | Rôle | Testable sans GL |
|---|---|---|
| `gfx/background/schema.py` | Charge et valide `modes.json` : noms uniques, défauts dans les bornes, `rand` ⊂ [min, max], `interp` connu. Produit les déclarations `uniform` à injecter. | Oui |
| `gfx/background/params.py` | État du fond : macros → valeurs, dé à seed, mutation, morphing (`interp`), `slew`, phases intégrées en `dt`, routing audio → paramètres. | Oui |
| `gfx/background/*.glsl` | `common.glsl` (rayons, marche sur-relaxée, normale tétraédrique anti-inlining, palette, glow, atténuation, éclairage extérieur) et `fractal.glsl`, `organic.glsl`, `geometric.glsl`, `landscape.glsl` (une `map()` par mode). | — |
| `gfx/background/pass_.py` (ou dans `Renderer`) | Programmes par mode précompilés, cache `{(mode, qualité): Program}`, texture f16 à `bg_scale`, requête `time=True`, hot-reload protégé. | Non |
| `Renderer` | Passe `draw_background()` après le resolve, échantillonnage dans le premier downsample du bloom (`u_bg_bloom`), insertion `hdr = trail.rgb + bg.rgb·(1 − trail.a)` dans `COMPOSITE_FS`. | Non |
| `app/` (dans `App`) | Un module « fond » branché comme les autres briques (T22). Erreur de compilation → `report()`, et le programme précédent est conservé. | En partie |

Ordre dans la frame : modèle (MSAA) → resolve → fond (peut sauter les pixels couverts) → trainées → bloom (+ fond émissif) → composite. Le fond « Aucun » ne lance aucune passe et laisse le composite actuel inchangé.

### Paramètres globaux du fond

| Paramètre | Plage / valeurs | Défaut | Interp | Audio |
|---|---|---|---|---|
| Mode | Aucun, Fractale, Organique, Géométrique, Paysage | Aucun | crossfade | — (changement de section en mode auto, T10) |
| Sous-mode | dépend du mode (voir tables) | 1er | weights / crossfade | — |
| Rendu | 2D, 2,5D, 3D fixe, 3D caméra | 3D caméra | fondu 0,5–1 s | — |
| Parallaxe | 0–1 (lien avec l'orbite) | 1 | linear | — |
| Qualité | Bas, Moyen, Haut, Auto | Moyen | snap (recompilation) | — |
| Résolution du fond | ⅓, ½, ¾, 1 | ½ | snap (debounce 0,5 s) | — |
| Intensité | 0–1 | 0,35 | linear | ÉNERGIE (faible) |
| Saturation | 0–1 | 0,6 | linear | — |
| Atténuation de distance | 0–2/TMAX | faible > 0 | linear | — |
| Palette (a, b, c, d) + teinte | presets iq + phase 0–1 | arc-en-ciel sombre | oklab / angle | phase ← beat_phase, saut ← SNARE |
| Glow | 0–2 | 0,4 | linear | KICK, RMS |
| Bloom du fond | 0–1 | 0,3 | linear | — |
| Vitesse globale | 0–4 (multiplie toutes les phases) | 1 | log | BPM / ÉNERGIE |
| Seed | entier | 0 | snap | re-seed optionnel au changement de section |
| Halo autour du modèle | −1 (sombre)…+1 (clair) | −0,3 | linear | — |
| Couleur de base (`u_bg`) | RVB | actuelle | oklab | — |

### Macros (communes, chaque mode déclare sa réponse)

| Macro | Fractale | Organique | Géométrique | Paysage |
|---|---|---|---|---|
| Complexité | itérations 5→12 | octaves du corail, nombre de boules | itérations de pli, voisins | octaves 4→9 |
| Chaos | angles dans la boucle, Julia | warp, biais TPMS | twist, désordre par cellule | warp 0→3 |
| Densité | échelle, offset | échelle TPMS, k | période (lente), rayon des poutres | fréquence, niveau de la mer |
| Organique↔Géométrique | poids box ↔ sphère | k du smin, épaisseur | arrondi, poids de primitive | érosion ↔ ridged |

### Mode Fractale

Sous-modes : KIFS (Menger, Tétra, Octa, Icosa), Mandelbox / Kalibox / Amazing Surf, Pseudo-Kleinian, Mandelbulb, Julia quaternion. Les plages proviennent de Fragmentarium ; les cibles audio sont des inférences à valider au banc.

| Paramètre | Plage | Défaut | Classe | Cible audio recommandée |
|---|---|---|---|---|
| Poids de pli (abs, box, sphère, tétra/tri) | 0–1 chacun | selon sous-mode | weights | macro Organique↔Géométrique |
| Itérations (fractionnaire) | 4–14 (Mandelbulb 6–8) | 8 | topologie | ÉNERGIE ou macro Complexité seulement |
| Scale | KIFS 1,5–3,5 ; Mandelbox −3…5, sans changer de signe | 3 / 2,39 | exact lent | ÉNERGIE ±0,1 |
| Offset (vec3) | [0,1]³ (KIFS) | (1,1,1) | continu | BASS (lent), impulsion KICK |
| Rotation dans la boucle (pré / post) | axe [−1,1]³, angle ±180° | 0–5° | exact | KICK ±2–8° (ressort), vitesse ← BPM |
| MinRad2 | 0,05–0,5 (borne 0–2) | 0,25 | continu | SUB / BASS |
| Julia (vec3) | ±2 | selon sous-mode | continu | LOW MID / MID (lent) |
| Power (Mandelbulb) | 2–16 | 8 | continu | ÉNERGIE ±0,3–0,5 |
| Fatness | 0–2 | 0 | exact | BASS |
| Placement | objet lointain / vol intérieur, distance, taille | objet lointain | — | défilement ← phase BPM |
| Facteur de pas (fudge) | 0,3–1 | 0,8 | qualité, verrouillé | — |

### Mode Organique

Sous-modes : Métaballes, TPMS (gyroïde, diamant, Schwarz P), Corail (fBM-SDF), Liquide (morph de deux formes), Orbe gyroïde.

| Paramètre | Plage | Défaut | Classe | Cible audio recommandée |
|---|---|---|---|---|
| Nombre de boules | 4–16 | 8 | entier (snap) | changement de section |
| k (smin exponentiel) | 0,1–1,0 | 0,4 | exact | ÉNERGIE (calme = gouttes, drop = masse) |
| Rayon de la boule i | 0,2–1,2 | 0,6 | exact | bande EQ i (mod 7) |
| Vitesse d'orbite | 0–2 (phase intégrée) | 0,3 | phase | BPM, ÉNERGIE |
| Écartement | 0–2 | 0 | exact | KICK (ressort : projette puis refusionne) |
| Type TPMS (float) | 0–2 (gyroïde → diamant → P) | 0 | weights | macro |
| Échelle TPMS | 0,3–3 | 1 | échelle du monde | jamais par beat |
| Biais TPMS | −1,2…+1,2 | 0 | topologie | ÉNERGIE lente avec `slew` |
| Épaisseur | 0,005–0,3 | 0,05 | exact | SUB / BASS |
| Warp (A, avec A·w ≤ 0,5) | 0–0,5 | 0 | borné (L) | RMS lissé |
| Octaves du corail | 3–5 | 3 | entier | macro Complexité |
| Lignes de contour émissives | 0–1 | 0,2 | émissif | SNARE |
| Irisation / fresnel | 0–1 | 0,3 | look | PRESENCE / TREBLE |

### Mode Géométrique

Sous-modes : Treillis, Piliers (égaliseur 3D), Tunnel, Anneaux, Cathédrale, Truchet 3D, Espace plié.

| Paramètre | Plage | Défaut | Classe | Cible audio recommandée |
|---|---|---|---|---|
| Période P | 2–8 | 4 | échelle du monde | jamais par beat |
| Rayon des poutres / tubes | 0,02–0,3·P | 0,08·P | exact | SUB / BASS |
| Rayon des nœuds | 0–0,4·P | 0,1·P | exact | KICK |
| Secteurs polaires N | 3–24 | 8 | entier | changement de section |
| Rayon du tunnel R | 3–10 | 6 | échelle | — |
| Période des anneaux L | 0,5–4 | 1,5 | échelle | — |
| Twist | 0–0,3 rad/unité, facteur 1/√(1+(twist·R)²) | 0 | borné | SNARE (impulsion qui décroît) |
| Défilement | 0–5 unités/s (intégré) | 1 | phase | BPM × ÉNERGIE, pas calé sur le beat (option) |
| Hauteur par cellule (piliers) | 0–1 × hash | — | exact (2×2 voisins) | 7 bandes EQ |
| Poids des primitives (boîte, sphère, tore, capsule) | 0–1, somme 1 | boîte | weights | macro |
| Arrondi, onion | 0–0,3·P | 0 | exact | BASS |
| Itérations / offset / angles de pli | 2–5 ; ±P ; radians | 3 | topologie / exact | offset ← BASS, angles ← MID/HIGH intégrés lentement |
| Émissif des joints / poursuite par secteur | 0–3 | 1 | émissif | KICK, compteur de beats |

### Mode Paysage

Sous-modes : Montagnes érodées, Crêtes (ridged), Dunes, Mer, Canyon, Synthwave, Spectrogramme 3D (et Grotte en qualité haute seulement).

| Paramètre | Plage | Défaut | Classe | Cible audio recommandée |
|---|---|---|---|---|
| Octaves (dernière fractionnaire) | 3–12 | 6 | topologie | macro Complexité |
| Lacunarité | 1,8–2,3 | 2,01 | continu lent | aucune |
| Gain (ou H) | 0,35–0,65 | 0,5 | continu | — |
| Amplitude | 0–3 | 1 | continu | **ÉNERGIE** (break = plaine, drop = montagnes) |
| Fréquence | 0,05–1 | 0,3 | échelle | jamais en direct |
| Warp | 0–4 | 0 | continu | MID |
| Érosion | 0–2 | 1 | continu | — |
| Netteté des crêtes | 1–4 | 2 | continu | — |
| Niveau de la mer | −1…+1 | −0,3 | continu lent | marée ← ÉNERGIE |
| Défilement | 0–5 unités/s (intégré, modulo une grande période) | 1 | phase | BPM, ÉNERGIE |
| Onde de choc au sol | 0–1 | 0,3 | exact (ressort) | KICK |
| Émissif des crêtes / lignes de grille | 0–4 | 1 | émissif | TREBLE, SNARE |
| Soleil synthwave (pulsation) | 0–1 | 0,5 | look | beat_phase |
| Historique du spectrogramme | 64 bandes × 128–256 lignes | 128 | texture | `spectrum` (64 mel) |
| Horizon TMAX | 20–200 | 80 | qualité | — |
| Ciel (zénith, horizon, direction du soleil) | couleurs HDR 0–2 | dégradé sombre | oklab | teinte ← palette |

### Niveaux de qualité

| | Bas | Moyen (défaut) | Haut |
|---|---|---|---|
| Résolution du fond | ⅓–½ | ½ | ½ (¾ si la machine suit) |
| MAX_STEPS (`#define`) | 48 | 80–96 | 128–160 (≤ 256, watchdog GPU) |
| Itérations fractales / octaves (marche / détail) | 6 / 3–5 | 8–10 / 4–7 | 12 / 5–9 |
| Normale | trap ou dérivées analytiques (0–4 éval.) | tétraèdre (4) | tétraèdre, ε ∝ t |
| AO | pas de marche ou hauteur (0) | 3 échantillons | 5 échantillons |
| Ombres douces | aucune (terme d'horizon) | classique, 16 pas | Aaltonen, 32–48 pas |
| Sur-relaxation | ω ≈ 1,5 avec repli | idem | idem |
| Cible GPU (à mesurer) | ≤ 0,5 ms | ~1–2 ms | 2–3 ms |

Les ombres douces coûtent une seconde marche ([iq, Soft shadows](https://iquilezles.org/articles/rmshadows/)) ; ce sont les premières sacrifiées. La sur-relaxation avec repli vient de Keinert et al. ([Eurographics DL](https://diglib.eg.org/handle/10.2312/stag.20141233.001-008)), et la littérature compare ω = 1,5 à ω = 0,88 ([Bán & Valasek 2023](https://diglib.eg.org/items/56b3a5fa-9c27-4036-a21f-7287dd83af11/full)). Les volumes englobants rapportent jusqu'à ×8 sur un objet coûteux ([iq, SDF bounding](https://iquilezles.org/articles/sdfbounding/)). Le niveau **Auto** baisse la résolution du fond par crans, avec hystérésis et un debounce de 0,5 s, quand `GL_TIME_ELAPSED` dépasse le budget, et l'affiche dans la barre de statut.

### Étapes de livraison

| Étape | Contenu | Sessions |
|---|---|---|
| 1 — Socle + Géométrique | Passe de fond séparée, insertion au composite et au bloom, rayons P·V, rendus 2D / 2,5D / 3D fixe / 3D caméra et fondu, qualité + mesure GPU, schéma JSON, dé, presets par mode, onglet Fond généré. Premier mode : Géométrique (le moins cher, valide la parallaxe). | 1,5 |
| 2 — Organique + Paysage | Métaballes, TPMS, corail ; heightfield, synthwave, spectrogramme 3D (texture d'historique) ; phases intégrées et compteur de beats dans le `Reactor`. | 1 |
| 3 — Fractales | Gabarit fold–scale–rotate, Mandelbulb, pseudo-Kleinian, orbit traps ; presets portés depuis Fragmentarium (Mandelbox +2,39 / −1,49, Menger brisé 3,1). | 1 |
| 4 — Morphing et mutation | `interp` par paramètre, fondu entre modes (deux textures pendant 0,5–2 s), grille de vignettes mutées via `tools/snapshot.py`. | 0,5 |
| Hors périmètre | Formes SDF devant le modèle (`gl_FragDepth`), mode « Composé » à génération de GLSL, accumulation temporelle. | — |

### Idées de critères d'acceptation

| Critère | Vérification |
|---|---|
| Fond « Aucun » + noir = rendu actuel au pixel près | `tools/snapshot.py` avant/après |
| Chaque mode par défaut tient le budget en Moyen, 60 fps conservés avec le mode le plus lourd | Requête `time=True`, affichage dans le HUD |
| 3D caméra : vraie parallaxe en orbitant ; 3D fixe et 2D : fond immobile à l'écran | `snapshot.py --orbit` |
| Bascule 2D ↔ 3D et changement de mode à chaud, sans saccade de plus d'une frame (précompilation) | Mesure du temps de frame |
| Erreur GLSL (hot-reload) → message `report()`, l'ancien programme reste affiché | Test manuel + journal |
| Le modèle reste toujours devant ; aucune bavure du fond dans les trainées | Visuel, orbite rapide |
| Même `{mode, seed}` = même image | Test pytest sur `params.py` + snapshot |
| Morph entre deux presets : aucun saut sur les paramètres continus, discrets selon `interp` | Test pytest |
| Silence → fond calme et stable, sans scintillement | Source silencieuse |
| Schéma validé : défauts dans les bornes, `rand` ⊂ bornes, uniforms présents | pytest, `ruff check .` |
| Ancien preset sans clé `fond` → chargé avec défauts ; clé inconnue → signalée, pas d'échec | Test `presets` |

### Risques

| Risque | Mitigation |
|---|---|
| Performance réelle inconnue sur GPU Apple en GL-over-Metal | Mesurer dès l'étape 1 et figer les niveaux après mesure ; demi-résolution par défaut |
| Saccade de compilation, pas de binaires de programme sur macOS | Précompilation au démarrage ou étalée, dessin « de chauffe » 1×1, boucle anti-inlining |
| Reset du GPU (watchdog) sur une boucle trop longue | MAX_STEPS ≤ 256 en `#define`, TMAX borné, `FrameGuard` |
| Trous et surestimation (plis mélangés, warp, twist) | Facteur L par mode, fudge en qualité, test « baisser le pas ne change pas la forme » ([incoherency.co.uk](https://incoherency.co.uk/notes/20250410.html)) |
| Scintillement temporel du détail fractal | `slew` sur la forme, topologie réservée aux sources lentes, jitter de départ + grain |
| Fond qui vole l'attention ou délave Points / Wireframe additifs | Intensité 0,35, désaturation, plafond de luminance sous les styles additifs, halo autour du modèle |
| Précision float32 du défilement sur une longue session | Phases modulo une grande période |
| Explosion du nombre de paramètres dans l'interface | Macros en tête, groupes repliables, dé par groupe, cadenas |
| Licences du code copié | Réimplémenter les formules ; Seascape et Shadertoy sous CC BY-NC-SA par défaut ([Ultraviolet](https://ultraviolet.indigoengine.io/shadertoy/seascape)), hg_sdf sous MIT ou CC-BY-NC ([hg_sdf](https://mercury.sexy/hg_sdf/)), Mandelbulber2 sous licence GPL à vérifier ([GitHub](https://github.com/buddhi1980/mandelbulber2)) |
| Dépendance au panneau T27 | Interface générée depuis le schéma, donc portable ; à défaut, générateur tkinter minimal |

## Conclusion

La recherche déplace la difficulté de T16. Le raymarching lui-même est maîtrisé ; Fragmentarium, iq et hg_sdf en fournissent les boucles exactes. Le vrai travail est un **système de paramètres** : distinguer ce qui est exact pour la distance de ce qui la déforme, ce qui est continu de ce qui change la topologie, et ce qui est rapide de ce qui est lent. Ces trois distinctions, écrites une fois dans le schéma JSON (`interp`, `topology`, `slew`, `routable`), décident à la fois de la sécurité du rendu, de la qualité du morphing et de la lisibilité de la réaction musicale. Le fond ne sera « vraiment différent » d'un preset à l'autre que si le dé tire d'abord la topologie (sous-mode, type de répétition, poids de pli) et ensuite seulement les valeurs.

Le second changement concerne la place du fond dans le pipeline. Le traiter comme une couche HDR indépendante, et non comme de la géométrie de scène, le sort du MSAA, des trainées et du Z-buffer. Il gagne au passage sa propre résolution, son propre budget mesurable et une bascule 2D/3D gratuite. Cette indépendance prépare aussi la suite : la même texture pourra être enregistrée à part par T5, reçue en entrée par un feedback vidéo (T12), ou portée telle quelle vers wgpu si T28 conclut à un changement de moteur.
