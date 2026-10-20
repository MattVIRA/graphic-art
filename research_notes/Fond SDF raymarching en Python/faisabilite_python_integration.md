# Fond SDF raymarché en Python (moderngl, OpenGL 3.3/4.1 sur macOS) — faisabilité, intégration, mode 2D/3D

*Notes de recherche, 2026-10-18. Contexte : renderer `gfx/renderer.py` du projet (lu pour cette note) — projection `_perspective(fov=45°, near=10, far=8000)` en unités « pixels logiques » (`cam_dist = (h/2)/tan(fov/2)`, ≈ 869 pour h = 720), scène dessinée dans un FBO MSAA 4× f16 + profondeur (`_accum`), résolu dans `_scene_tex`, puis couche trainées `max(trail×decay, scène)`, bloom 6 niveaux, composite AgX qui ajoute le fond `u_bg` en espace écran pondéré par `1 − alpha` de la scène. Les surfaces opaques écrivent alpha = 1, les points additifs alpha = 0. Caméra orbitale : `OrbitCamera.view_matrix()` = `look_at(eye, target)`.*

## 1. Python + moderngl est-il une voie viable pour du raymarching SDF temps réel ?

### Takeaway
Oui. Tout le coût est dans un fragment shader GLSL exécuté par le GPU sur un triangle plein écran ; Python ne fait qu'envoyer quelques dizaines d'uniforms par frame (comme le projet le fait déjà pour ~80 floats de features), donc le langage hôte n'est pas le goulot. La vraie contrainte est le GPU (Apple M1 ≈ 2,6 TFLOPS FP32) et le driver OpenGL 4.1-sur-Metal d'Apple, pas Python.

### Cited Findings
- Sur un Mac mini M1, OpenGL rapporte `GL_VERSION "4.1 Metal - 71.0.7"`, GLSL 4.10, `GL_RENDERER Apple M1` : le GL d'Apple Silicon est une couche au-dessus de Metal, plafonnée à 4.1 core — [Geeks3D, Apple Mac mini M1 Quick Test (OpenGL)](https://www.geeks3d.com/?p=10147)
- Même test : shader Shadertoy lourd « Neural Stanford Bunny 5kb » à 800×480 : M1 natif arm64 = 245 fps, RTX 2070 = 720 fps, GTX 750 = 176 fps ; l'auteur relie l'écart au débit FP32 théorique (M1 2,6 TFLOPS vs RTX 2070 7,4 TFLOPS). Ce n'est pas un raymarcher, seulement un proxy « gros fragment shader » — [Geeks3D](https://www.geeks3d.com/?p=10147)
- OpenGL et OpenCL sont dépréciés depuis macOS 10.14 Mojave (WWDC 2018) ; les applis continuent de tourner, Apple oriente vers Metal et ne fait plus que des « changements mineurs » sur GL — [Geeks3D, annonce de dépréciation](https://geeks3d.com/20180605/apple-announces-that-opengl-and-opencl-will-be-deprecated-in-macos-10-14-mojave) ; [AppleInsider](https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave)
- Aucune solution sur M1 ne dépassait le jeu de fonctionnalités OpenGL 4.1 sous macOS (les développeurs Asahi ont ensuite atteint GL 4.6 sur leur propre pilote Linux, pile distincte de macOS) — [Asahi Linux, Conformant OpenGL 4.6 on the M1 (fév. 2024)](https://asahilinux.org/2024/02/conformant-gl46-on-the-m1/)
- moderngl expose `Context.query(samples, any_samples, time, primitives)` où `time` = « Query GL_TIME_ELAPSED » (mesure GPU du coût du fond, utile pour une résolution dynamique), `Context.includes` (« Mapping used for include statements », pour factoriser des blocs GLSL), et le multisampling se règle sur les attachements (`texture()` / `renderbuffer()` param `samples`, « 0 means no multisample format ») — [moderngl docs, Context](https://moderngl.readthedocs.io/en/latest/reference/context.html)
- Projets Python existants qui font tourner des shaders type Shadertoy :
  - `einarf/shadertoy` : « a python implementation of shadertoy using moderngl » (état actuel non vérifié) — [github.com/einarf/shadertoy](https://github-redirect.dependabot.com/einarf/shadertoy)
  - `wgpu-shadertoy` (organisation pygfx, sur wgpu-py) : colle le code GLSL ou WGSL de Shadertoy, jusqu'à 4 canaux texture, rendu hors écran `snapshot()` ; le README dit « still in development », pas de `sampler3D` / `samplerCube` — [github.com/pygfx/shadertoy](https://github.com/pygfx/shadertoy) ; [PyPI wgpu-shadertoy](https://pypi.org/project/wgpu-shadertoy). Une ancienne version vivait dans `wgpu.utils.shadertoy` (docs v0.9.3/v0.10.0) — [wgpu-py utils v0.10.0](https://wgpu-py.readthedocs.io/en/v0.10.0/utils.html)
  - `NamorNiradnug/raymarching` : module Python qui **génère des shaders** de raymarching à partir d'une description de scène (exemple de génération de code) — [github.com/NamorNiradnug/raymarching](https://github.com/NamorNiradnug/raymarching)
  - `alexeykarnachev/py2glsl` : transpileur Python → GLSL (autre approche de génération) — [py2glsl (miroir)](https://backiee.wasmer.app/https_github_com/alexeykarnachev/py2glsl)
  - Contre-exemple : `PashaWNN/raymarching` fait le raymarching en NumPy/PIL (CPU), donc pas temps réel — [github.com/PashaWNN/raymarching](https://github.com/PashaWNN/raymarching)
- `fogleman/sdf` est d'une autre nature : il **génère des maillages** à partir de fonctions de distance (Marching Cubes, évaluation NumPy par lots en threads, export STL / meshio), sans rendu temps réel — [github.com/fogleman/sdf](https://github.com/fogleman/sdf). Il pourrait servir à produire un `.stl` chargé par le pipeline modèles 3D existant (T19), pas à faire un fond animé.
- Les exemples moderngl-window contiennent `geometry_quad_fs.py` et `geometry_quad_fs_mouse_scroll.py` (quad plein écran + fragment shader, base de tout effet type Shadertoy) ; aucun exemple explicitement « raymarching » ni « hot reload » n'a été trouvé dans la liste visible — [moderngl-window/examples](https://github.com/moderngl/moderngl-window/tree/master/examples)
- Licence : les shaders Shadertoy sont par défaut sous CC BY-NC-SA 3.0 sauf mention contraire de l'auteur — à vérifier avant de copier du code (affirmation remontée par la recherche web, source exacte parmi les résultats non isolée ; la règle figure dans les conditions de shadertoy.com) — [résultats de recherche, ex. DEV Community](https://dev.to/ramislicer/ray-marching-the-unappreciated-cousin-of-ray-tracing-1iad)

### Inferences
- **Rôle de Python** (adapté au projet) :
  1. *Uniforms par frame* : temps, features audio routées (le `Reactor` existant fournit déjà `fx`), matrices caméra, paramètres du panneau. Coût CPU négligeable (quelques `prog["u_x"].value = …`).
  2. *Hot-reload* : surveiller le `mtime` d'un fichier `.glsl` (ou d'un bloc dans `gfx/`), recompiler `ctx.program()` dans un `try/except moderngl.Error`, et **ne remplacer le programme courant que si la compilation réussit** — c'est exactement la règle du projet « ne jamais remplacer la source / le modèle courant avant que le nouveau soit valide » ; l'erreur va dans `App.report()`. L'appel doit rester sur le thread GL (un seul contexte), mais il ne coûte qu'à la modification.
  3. *Génération de code* : décrire le fond comme une liste Python de primitives + opérateurs (union lisse, répétition, twist…) et assembler `float map(vec3 p)` en chaîne GLSL. Changement de **structure** → recompilation (rare) ; changement de **valeur** → uniform (chaque frame). Le module de génération est sans GL → testable dans `tests/` (règle T22/T25).
- **Budget théorique** (calcul, pas mesure) : à 60 fps, 2 ms × 2,6 TFLOPS ≈ 5,2 GFLOP par frame pour le fond. Un framebuffer Retina interne de 2560×1600 = 4,1 Mpx → ≈ 1 270 FLOP/pixel en pleine résolution ; à demi-résolution (1280×800 = 1 Mpx) ≈ 5 000 FLOP/pixel. Une scène simple (quelques primitives, `map()` ≈ 30–80 FLOP) à 64 pas + normale (4 évaluations) + éclairage tient dans ≈ 3 000–6 000 FLOP : **réaliste à demi-résolution, serré en pleine résolution Retina, hors de portée pour un Mandelbulb à 200 pas + ombres en pleine résolution**. Le rendement réel d'un GPU est inférieur au pic théorique (divergence des boucles, latences), d'où la nécessité de mesurer avec `ctx.query(time=True)`.
- **Alternatives** :
  - *wgpu-py / pygfx* : backend Metal natif (pas de dépréciation, compute shaders disponibles), mais cela impose de réécrire tout le renderer (pas de partage de contexte avec moderngl) ; c'est le sujet du tiquet T28, pas du fond.
  - *Compute shaders OpenGL* : introduits en GL 4.3, donc absents d'un macOS plafonné à 4.1 (inférence à partir du plafond, voir Gaps). Tout doit se faire en fragment shaders (ping-pong de textures pour l'accumulation temporelle).
  - *Taichi* : a un backend Metal et des exemples de rendu SDF, mais c'est un deuxième runtime GPU à faire cohabiter avec le contexte GL (copie GPU→CPU→GPU ou interop) ; peu pertinent ici (non vérifié dans cette session).
- **Conclusion faisabilité** : Python ne limite pas ; la voie la plus simple et compatible avec l'existant est « un programme moderngl de plus » (fragment shader plein écran) branché dans `Renderer`, avec un module Python de génération/hot-reload testable.

### Gaps
- Aucun benchmark publié de raymarching SDF sur M1/M2/M3 sous OpenGL macOS (fps à 1080p / Retina) n'a été trouvé ; la seule donnée chiffrée est le shader « Neural Bunny » de Geeks3D. À mesurer sur la machine avec `ctx.query(time=True)` et `tools/snapshot.py`.
- Pas de source trouvée affirmant explicitement « pas de compute shaders sur macOS GL » ; c'est déduit du plafond 4.1 (compute = GL 4.3).
- État 2026 d'`einarf/shadertoy` et de `wgpu-shadertoy` (dernière release, support macOS/Metal, multipass) non confirmé ; la page GitHub de pygfx/shadertoy n'affichait pas de release.
- Temps de compilation GLSL sur le driver GL-over-Metal d'Apple pour un gros shader : aucune mesure trouvée.

## 2. Caméra partagée, parallaxe et profondeur : construire les rayons et cohabiter avec le modèle rasterisé

### Takeaway
On reconstruit le rayon de chaque pixel à partir des **mêmes** matrices que le modèle (inverse de P·V, ou œil + base caméra + tan(fov/2)), ce qui donne une parallaxe exacte avec l'orbite. Comme le modèle doit **toujours rester devant**, le plus simple et le moins cher est de ne pas écrire de profondeur du tout : rendre le fond dans sa propre texture (non-MSAA, demi-résolution possible) et l'insérer au composite sous la scène via l'alpha. `gl_FragDepth` ne devient nécessaire que si des formes SDF doivent passer devant / s'intersecter avec le modèle.

### Cited Findings
- Le buffer de profondeur matériel stocke le **z projeté** du point d'intersection, pas la distance le long du rayon ; il faut projeter le point touché par la matrice de projection, faire la division perspective, puis ramener en [0,1] pour `gl_FragDepth` — c'est ce qui permet à la géométrie raytracée de coexister dans le Z-buffer avec des polygones — [UAF CS481, Raytracing Multiple Objects](https://www.cs.uaf.edu/2010/spring/cs481/section/0/lecture/02_02_multiple.html)
- Formule utilisée en pratique (three.js) : `clip = P * MV * vec4(p,1); fragDepth = clip.z/clip.w; fragDepth = saturate((fragDepth+1)/2); gl_FragDepth = fragDepth;`. Piège rapporté : une première version normalisait linéairement `(depth−near)/(far−near)`, ce qui ne correspond pas à la profondeur non linéaire attendue ; autre piège : mélanger les espaces (appliquer une model-view à un point déjà en espace monde) ; le fil ne valide pas le cas du depth logarithmique — [three.js forum, Raymarching – Correct gl_FragDepth](https://discourse.threejs.org/t/raymarching-correct-gl-fragdepth/66033)
- Un raymarcher dont la courbe de profondeur ne coïncidait pas avec les polygones malgré la division clip.z/clip.w : corrigé en ajustant l'échelle du rayon en espace vue (le `t` du rayon doit correspondre à l'espace où l'on projette) — [GameDev.net, matching raymarcher and polygon depth values](https://gamedev.net/forums/topic/622531-matching-raymarcher-and-polygon-depth-values/)
- Écrire la profondeur de la géométrie proxy (boîte englobante) au lieu de celle de la surface marchée provoque des erreurs là où la géométrie normale intersecte le proxy — [iq, raypolys](https://iquilezles.org/articles/raypolys) (résumé via la recherche) ; Babylon.js (bloc FragDepth du Node Material Editor) et un shader Godot font la même écriture de profondeur pour que les maillages se coupent correctement avec le raymarché (Godot : GLES3 seulement, GLES2 n'a pas d'écriture de profondeur) — [Babylon.js, ray marching in the NME](https://babylonjs.medium.com/ray-marching-in-the-babylon-js-node-material-editor-967b5b8c269c) ; [Godot Shaders, Raymarching with depth writing](https://godotshaders.com/shader/raymarching-with-depth-writting/)
- Le renderer du projet (lu) : `_perspective()` OpenGL standard (`M[2,2]=(f+n)/(n−f)`, `M[2,3]=2fn/(n−f)`, `M[3,2]=−1`), near = 10, far = 8000 ; composite : `col = agx(hdr) ; col += u_bg * (1 − clamp(c.a,0,1))` — `gfx/renderer.py` lignes 27-35, 728, 877 et `COMPOSITE_FS` (code local, pas de lien web).

### Inferences
- **Rayons depuis les matrices du projet** (deux variantes équivalentes) :
  ```glsl
  // A) inverse de P·V envoyée par Python (np.linalg.inv((P @ V)), transposée pour GLSL column-major)
  uniform mat4 u_inv_vp;  uniform vec3 u_eye;   // u_eye = inverse(V)[:3,3]
  vec3 ray_dir(vec2 uv) {                       // uv dans [0,1]
      vec4 w = u_inv_vp * vec4(uv * 2.0 - 1.0, 1.0, 1.0);   // point sur le plan far
      return normalize(w.xyz / w.w - u_eye);
  }
  // B) base caméra (moins d'ALU, pas d'inverse 4×4) : lignes 0..2 de V = right, up, -forward
  uniform vec3 u_right, u_up, u_fwd; uniform float u_tan_half_fov, u_aspect;
  vec3 ray_dir(vec2 ndc) {
      return normalize(u_fwd + ndc.x * u_aspect * u_tan_half_fov * u_right
                             + ndc.y * u_tan_half_fov * u_up);
  }
  ```
  La « Secousse caméra » du projet modifie `V[0,3]`, `V[1,3]` : avec la variante A elle est prise en compte automatiquement (elle déplace l'œil) ; avec B, il faut dériver `u_eye` de `inverse(V)`.
- **Échelle** : le monde du projet est en « pixels » (cam_dist ≈ 869 pour h = 720, far = 8000). Soit on construit le fond dans ces unités (seuils `EPS ≈ 0.5–1`, `TMAX ≈ 6000`), soit on divise `p` par une constante `u_world_scale` (par ex. 100) à l'entrée de `map()` et on multiplie la distance retournée par la même constante — ne jamais mélanger les deux.
- **Profondeur depuis le `t` du raymarch** : avec un rayon normalisé `rd`, `z_eye = -t * dot(rd, u_fwd)` (distance le long de l'axe caméra, pas le long du rayon — c'est le piège du fil GameDev). Puis, avec la projection du projet :
  ```glsl
  float depth_from_eye_z(float ze) {            // ze < 0 (espace caméra)
      const float n = 10.0, f = 8000.0;         // passer en uniforms
      float ndc = ((f + n) / (n - f) * ze + 2.0 * f * n / (n - f)) / (-ze);
      return ndc * 0.5 + 0.5;                   // glDepthRange par défaut [0,1]
  }
  // ou plus robuste : vec4 c = u_vp * vec4(hit,1.0); gl_FragDepth = c.z/c.w*0.5+0.5;
  // pas de hit → gl_FragDepth = 1.0 (le fond reste derrière tout)
  ```
  Précision : avec n = 10, f = 8000 en depth 24 bits, la résolution de profondeur loin de la caméra est faible ; un fond SDF loin (z ≈ 3000–8000) qui doit s'intersecter précisément avec le modèle donnera du z-fighting — le cas « fond toujours derrière » l'évite par construction.
- **Architecture recommandée pour « le modèle reste devant » (option A, la moins chère)** :
  1. Nouvelle passe `draw_background()` **avant** `begin_frame()` ou indépendante : triangle plein écran dans une texture `f2` RGBA **non-MSAA** à `bg_scale × taille interne` (0,5 par défaut), `DEPTH_TEST` désactivé.
  2. Au composite (`COMPOSITE_FS`), lire `u_bgtex` et l'insérer **en HDR linéaire avant AgX** : `hdr = scene.rgb + bg.rgb * (1 − scene.a)` (les points additifs, alpha 0, brillent par-dessus le fond ; les surfaces opaques, alpha 1, le masquent). L'actuel `u_bg` (couleur unie ajoutée après AgX) peut rester comme teinte de base.
  3. Le fond n'entre ni dans le MSAA (pas de coût 4× en bande passante), ni dans la couche **trainées** (sinon `max(trail×decay, scène)` ferait baver le fond à chaque mouvement de caméra), ni obligatoirement dans le bloom (option : ajouter `bg` à l'entrée du premier downsample avec un gain pour faire briller les formes émissives).
  4. Limite : un modèle transparent additif (style Surface « transparent ») ou un trou dans le modèle laisse voir le fond, ce qui est l'effet voulu ; mais une forme SDF ne peut jamais passer devant le modèle.
- **Option B (formes SDF qui peuvent passer devant / couper le modèle)** : stocker aussi la profondeur du raymarch (canal alpha de la texture de fond ou texture `f4` R32F), puis dans `begin_frame()` dessiner un triangle plein écran dans `_accum` (MSAA) qui recopie couleur + `gl_FragDepth` (échantillonnage `NEAREST` pour ne pas interpoler les profondeurs entre bords), avant le modèle. Écrire `gl_FragDepth` désactive l'early-Z pour **ce** draw, sans conséquence puisqu'il est seul et premier. Le modèle se dessine ensuite avec `DEPTH_TEST` normal. Coût : un fetch par échantillon, mais le fond entre alors dans les trainées et le bloom comme la scène (à gérer : soit exclure via alpha, soit accepter).
- **MSAA** : sans `GL_ARB_sample_shading`, un fragment shader s'exécute une fois par pixel même en MSAA 4× ; le coût ALU du raymarch n'est pas multiplié par 4, mais la bande passante d'écriture et le resolve le sont, et le bord des formes SDF n'est pas antialiasé par le MSAA (il faut un AA analytique : `smoothstep` sur la distance / la taille de pixel). D'où l'intérêt de la texture séparée non-MSAA.

### Gaps
- Le fichier iq « raypolys » n'a pas été relu en entier (contenu repris du résumé de recherche).
- Aucune source n'a été trouvée sur le comportement précis du driver Apple GL-over-Metal quand on écrit `gl_FragDepth` dans un FBO MSAA (coût, early-Z) — à mesurer si l'option B est retenue.

## 3. Mode 2D (écran fixe) vs 3D (lié à la caméra) et bascule à chaud

### Takeaway
Un seul shader peut porter trois modes : **3D-caméra** (rayons depuis la vue orbitale), **3D-fixe** (rayons depuis une vue constante, le fond ne bouge plus quand on orbite) et **2D** (SDF 2D évaluée directement sur les coordonnées écran, sans marche — très bon marché). La bascule se fait par uniform (pas de recompilation) et un fondu (`u_mix`) ou une interpolation de la caméra (œil/cible, pas des matrices). Un mode intermédiaire « plans 2D en perspective » donne la parallaxe 3D au coût du 2D.

### Cited Findings
- La « pixel footprint » (taille d'un pixel à la distance t) doit piloter l'epsilon des normales et le niveau de détail : une géométrie plus petite qu'un pixel peut être ignorée (band-limiting) ; un `h` constant donne des normales bruitées et aliasées au loin — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/)

### Inferences
- **Structure d'un shader unique** (pseudo-GLSL) :
  ```glsl
  uniform float u_mode3d;      // 0 = 2D, 1 = 3D (animé par Python pour le fondu)
  uniform float u_cam_link;    // 0 = caméra fixe, 1 = caméra orbitale (interpolé)
  vec3 shade2d(vec2 p) {                        // p = (fragCoord - 0.5*res) / res.y
      float d = scene2d(p);                     // sdCircle, sdBox, sdStar… (iq 2D)
      float aa = fwidth(d);                     // AA analytique à 1 px
      return palette(d) * smoothstep(aa, -aa, d) + glow(d);
  }
  void main() {
      vec3 col = vec3(0.0);
      if (u_mode3d < 0.999) col += (1.0 - u_mode3d) * shade2d(p2);
      if (u_mode3d > 0.001) col += u_mode3d * shade3d(ray_from(u_inv_vp_mix));
      f_color = vec4(col, 1.0);
  }
  ```
  Les branches dépendent d'un uniform → tous les pixels prennent le même chemin, la branche est quasi gratuite ; pendant le fondu (≈ 0,5–1 s) les deux chemins sont payés, hors fondu un seul.
- **Caméra fixe vs liée** : Python calcule `V_fixed` (= la vue par défaut du projet, `translate(0,0,−cam_dist)`) et `V_orbit` (caméra courante). Pour une transition douce, interpoler **les paramètres** (yaw, pitch, distance, cible — ou slerp de quaternion) puis reconstruire `look_at`, plutôt que d'interpoler les matrices terme à terme (un lerp de matrices de rotation n'est plus orthonormé et déforme l'image). L'`OrbitCamera` du projet a déjà un retour en douceur (`_ease`, `reset()`), réutilisable : « mode fixe » = la caméra du fond suit une copie qui tend vers la vue de départ.
- **Mode « 2,5D » — plans SDF 2D en perspective** : intersecter le rayon caméra avec N plans (z = −z_k) analytiquement (`t = (z_k − ro.z)/rd.z`), évaluer une SDF 2D au point d'impact, composer de l'arrière vers l'avant. Aucune boucle de marche : coût ≈ N × coût 2D, et parallaxe réelle quand on orbite. Bon compromis pour « idéalement 3D pour la perspective » si le budget est serré.
- **Où sélectionner la caméra** : en 3D-caméra, appliquer la même transformation que les lumières Studio (espace caméra, T26) pour l'éclairage → normales SDF transformées par `mat3(V)` ; en 3D-fixe, le fond est éclairé comme s'il était vu de face, il ne réagit pas à l'orbite (cohérent avec un « décor d'écran »).
- **Paramètres audio** : utiliser les sorties du `Reactor` (KICK à ressort, ÉNERGIE de section…) comme uniforms (rayon, union lisse `k`, répétition, twist, couleur) — pas de recompilation, donc pas de saccade.

### Gaps
- Pas de source spécifique trouvée sur l'interpolation de caméra entre vue fixe et vue orbitale pour un fond raymarché ; les recommandations ci-dessus sont de l'ingénierie générale.
- Les pages iq « 2D distance functions » (iquilezles.org/articles/distfunctions2d) et « distance functions » (iquilezles.org/articles/distfunctions), référence standard des primitives SDF, n'ont pas été relues dans cette session ; les noms de primitives (sdCircle, sdBox…) viennent de connaissances antérieures.

## 4. Techniques de performance (et ordres de grandeur)

### Takeaway
Les leviers par ordre d'efficacité pour ce projet : (1) rendre le fond à **demi-résolution** (÷4 pixels) dans une texture séparée, (2) borner `MAX_STEPS` (48–96) avec epsilon proportionnel à la taille de pixel et `TMAX` serré, (3) volumes englobants dans `map()` (iq rapporte ×8 sur un personnage), (4) sur-relaxation (ω ≈ 1,5–1,7) avec repli sûr, (5) résolution dynamique pilotée par `GL_TIME_ELAPSED`. L'accumulation temporelle est possible en ping-pong mais délicate avec une caméra et un fond animés par l'audio.

### Cited Findings
- **Sphere tracing amélioré (Keinert et al., STAG 2014)** : méthode « sûre » de sur-relaxation pour accélérer le sphere tracing, contrôle dynamique des auto-intersections (compromis précision/perf), test plus rapide pour objets convexes dans des volumes englobants convexes, métrique écran pour choisir le meilleur candidat quand la marche ne converge pas, itération de point fixe contre les artefacts de discontinuité. Auteurs : B. Keinert, H. Schäfer, J. Korndörfer, U. Ganse, M. Stamminger — [Eurographics DL, Enhanced Sphere Tracing](https://diglib.eg.org/handle/10.2312/stag.20141233.001-008) ; [FAU LGDV](https://www.lgdv.tf.fau.de/publications/enhanced-sphere-tracing/). Version journal étendue : « Improved Ray Casting of Procedural Distance Bounds », JGT 17(4), 2015 — [CRIS FAU](https://cris.fau.de/publications/200539563/)
- **Bán & Valasek, « Automatic Step Size Relaxation in Sphere Tracing » (Eurographics 2023)** : adapte la taille de pas à partir des itérations précédentes, un seul hyperparamètre (comparable à un learning rate) moins sensible que les facteurs des variantes concurrentes, et beaucoup moins de pas de repli car moins de pas sur-relaxés invalides ; leur figure 1 compare enhanced sphere tracing ω = 0,88, relaxed ω = 1,5 et leur méthode — [Eurographics DL, egs20231014](https://diglib.eg.org/items/56b3a5fa-9c27-4036-a21f-7287dd83af11/full)
- Un autre article de 2023 utilise un facteur de sur-relaxation de 1,7 avec une borne de Lipschitz de 1,45 pour un rendu sans artefact — [arXiv 2304.09673, Synchronized-tracing of implicit surfaces](https://arxiv.org/pdf/2304.09673) (via résumé de recherche)
- **Volumes englobants SDF (iq)** : évaluer d'abord une borne bon marché (sphère, ou demi-espace pour un sol) et retourner tôt si elle est plus loin que la distance minimale courante ; rapport d'un « 8x speed up » sur un personnage coûteux ; les bornes peuvent être récursives mais chaque test et branche a un coût — [iq, SDF bounding volumes](https://iquilezles.org/articles/sdfbounding/)
- **Normales** : différences centrales = 6 évaluations ; avant = 4 (3 si f(p)≈0 est supposé) ; **technique du tétraèdre** = 4 évaluations aux sommets (1,−1,−1), (−1,−1,1), (−1,1,−1), (1,1,1), équivalente aux différences centrales ; écrire ces 4 appels dans une boucle dont la borne est un zéro inconnu à la compilation (`min(iFrame,0)`) empêche l'inlining de `map()` ×4, ce qui réduit plantages du compilateur et temps de compilation — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/)
- **Ombres douces** : une seule marche vers la lumière, `res = min(res, k*h/t)` (k = dureté) ; variante d'Aaltonen qui estime le point le plus proche entre deux échantillons (`y = h²/(2·ph)`, `d = sqrt(h²−y²)`, terme `d/(w·max(0,t−y))`) pour supprimer bandes et fuites ; boucle plafonnée (256 dans l'exemple) — coût ≈ une seconde marche par pixel — [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/)

### Inferences
- **Boucle de marche recommandée** (sur-relaxation avec repli, epsilon en pixels) :
  ```glsl
  const int MAX_STEPS = 80;
  float march(vec3 ro, vec3 rd, out int steps) {
      float t = u_tmin, w = 1.6, prev = 0.0, step = 0.0;
      for (int i = 0; i < MAX_STEPS; i++) {
          float d = map(ro + rd * t);
          if (w > 1.0 && abs(d) + prev < step) {   // sphères disjointes : sur-pas invalide
              t -= step - prev;                     // revenir au pas « sûr »
              step = prev; w = 1.0;                 // repasser en sphere tracing classique
              continue;
          }
          float eps = u_pix_angle * t;              // footprint pixel ≈ t·tan(fov/2)/(h/2)
          if (d < eps) { steps = i; return t; }
          if (t > u_tmax) break;
          prev = d; step = d * w; t += step;
      }
      steps = MAX_STEPS; return -1.0;
  }
  ```
  (Forme tirée de la description de Keinert et al. ; le détail exact de la condition de repli est à vérifier dans le PDF, voir Gaps.)
- **Demi / quart de résolution** : sur Retina interne 2560×1600, demi-résolution = 1 Mpx (coût ÷4). Le fond étant souvent doux (dégradés, glow, brouillard de couleur), un upscale bilinéaire suffit en option A ; si des arêtes nettes de formes SDF apparaissent floues, upscale « depth-aware » (4 taps + poids sur la différence de profondeur stockée en alpha) ou rendu 2D en pleine résolution (le 2D ne coûte presque rien). Le bloom et le grain du composite masquent une partie du flou.
- **Résolution dynamique** : mesurer le fond avec `q = ctx.query(time=True)` (`with q: draw_background()` ; lire `q.elapsed` en ns une ou deux frames plus tard pour ne pas bloquer le pipeline), et ajuster `bg_scale` ∈ {0,25 ; 0,33 ; 0,5 ; 0,75 ; 1} avec hystérésis (rebuild de la texture = cher → debounce comme le budget de points, 0,5 s).
- **LOD / early exit** : `TMAX` le plus court possible (le fond est à distance connue), epsilon ∝ t (footprint), sortir quand `t > tmax`, et ne pas calculer ombres/AO pour les pixels sans hit. Dans un fond, peu de pixels ont besoin d'ombres douces ; les remplacer par une AO « 5 échantillons le long de la normale » (technique iq classique, non relue ici) ou par rien.
- **Cône marching / multi-résolution** : marcher d'abord une passe très basse résolution (1/8) qui donne une distance de départ `t0` conservatrice par tuile, puis démarrer la marche fine à `t0` — utile seulement si les rayons parcourent de grandes distances vides ; pour un fond simple, la demi-résolution + bornes suffit (inférence, non mesurée).
- **Accumulation temporelle** : possible en ping-pong de textures (pas besoin de compute) avec reprojection par `prev_VP`, mais le fond est animé par l'audio (formes qui pulsent sur le kick) → l'historique devient faux à chaque transitoire, d'où ghosting ; le projet a déjà une couche trainées pour l'effet de persistance. À réserver à un jitter sous-pixel d'antialiasing léger, pas indispensable.
- **Ordres de grandeur attendus** (extrapolation du proxy Geeks3D, non mesuré) : le shader « Neural Bunny » tourne à 245 fps soit ≈ 4 ms pour 0,38 Mpx sur M1 → ≈ 10 ns/pixel. Un fond SDF simple (≈ 3–5× moins lourd par pixel que ce réseau de neurones en GLSL, hypothèse) à 1 Mpx (demi-résolution Retina) → de l'ordre de 2–4 ms ; à pleine résolution 4 Mpx → 8–16 ms, incompatible avec 60 fps en plus du modèle. **Conclusion pratique : viser demi-résolution par défaut, pleine résolution seulement pour le mode 2D.** À confirmer par mesure.

### Gaps
- Le PDF de Keinert et al. et celui de Bán & Valasek n'ont pas pu être lus (403 / pas de lien PDF) : pas de chiffres de gain (nombre de pas économisés) ni de valeur ω recommandée par Keinert ; seules les valeurs ω = 1,5 (relaxed), 0,88 (enhanced, dans la comparaison 2023) et 1,7 (autre article 2023) sont sourcées.
- Aucun benchmark M1/M2 de Mandelbulb ou de scène SDF simple en GL macOS trouvé.
- Pas de source consultée pour le cone marching (démoscène, « Fulcrum ») ni pour l'upscale bilatéral appliqué au raymarching.

## 5. Pièges spécifiques à macOS OpenGL (GLSL 330/410, compilation, watchdog, précision, variantes)

### Takeaway
Le GLSL 330 desktop n'a pas les limites de WebGL (boucles à borne non constante autorisées, pas de déroulage obligatoire), mais le driver Apple est figé (4.1, plus maintenu activement), la compilation d'un gros shader peut être lente et une boucle trop longue peut faire « redémarrer » le GPU. Stratégie : paramètres continus en **uniforms**, structure de scène en **#define/génération** précompilées hors de la boucle chaude, `MAX_STEPS` bornés, garde-fou Python autour de chaque compilation.

### Cited Findings
- Le GL macOS est une couche sur Metal (`"4.1 Metal - 71.0.7"`) avec 43 extensions en contexte 4.1 (133 en 2.1 legacy) — [Geeks3D, M1 OpenGL test](https://www.geeks3d.com/?p=10147)
- Apple : développement actif d'OpenGL arrêté, « minor changes » seulement — [AppleInsider](https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave)
- Les compilateurs peuvent inliner plusieurs fois `map()` (normales, ombres), ce qui peut dépasser des limites d'instructions ou faire planter le compilateur ; l'astuce de la boucle à zéro opaque (`min(iFrame,0)`) réduit plantages **et** temps de compilation — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/)
- Écart de performance mesuré entre OpenGL et Metal sur Apple Silicon (applications réelles, pas du raymarching) : Minecraft + mod sur M2 MacBook Air ≈ 1,8× plus de fps en Metal ; Blender Eevee ≈ 2× en changeant de backend (M3) — [résultats de recherche : CurseForge « Pure Metal »](https://www.curseforge.com/minecraft/mc-mods/pure-metal) ; [BlenderArtists](https://blenderartists.org/t/mac-m3-hardware-accelerated-rt-part-1/1264658?page=421) — indique qu'OpenGL macOS laisse des performances sur la table, mais ces chiffres mêlent coût CPU du driver et GPU.

### Inferences
- **Uniform vs #define** :
  - *Uniform* : bascule de mode 2D/3D, nombre de pas, paramètres de formes, couleurs, routing audio. Zéro saccade. Les branches sur uniform sont cohérentes (tout le front d'onde prend le même chemin).
  - *#define / génération de code* : changement du **type** de scène (liste de primitives), activation des ombres/AO (supprime du code mort, accélère). Coût : recompilation (non mesurée sur ce driver, typiquement dizaines à centaines de ms pour un gros shader) → saccade d'une frame. Mitigation : précompiler toutes les variantes connues au démarrage (ou à la sélection dans le panneau, quand une saccade est acceptable) et les garder en cache `dict[key] → Program`; un dessin « à blanc » 1×1 juste après la compilation force le driver à finaliser le pipeline (certains drivers compilent paresseusement au premier draw — comportement Apple non vérifié).
  - Le contexte GL étant mono-thread, compiler dans un thread séparé demanderait un contexte partagé GLFW — complexité non justifiée ici.
- **Watchdog** : une frame GPU qui dure plusieurs secondes (boucle quasi infinie, `MAX_STEPS` énorme, TMAX non borné en pleine résolution) peut provoquer un reset GPU et une perte de contexte ; toujours borner les boucles par des constantes, garder un `MAX_STEPS` ≤ 128–256, et laisser la `FrameGuard` du projet signaler les erreurs. La durée exacte du watchdog macOS n'est pas documentée dans les sources trouvées.
- **Précision** : float 32 bits partout (pas de `precision` à gérer en GLSL desktop) ; en unités « pixels » (valeurs jusqu'à 8000), l'epsilon de marche doit rester ≥ ~1e-3 × t pour éviter le bruit ; pour des fractales profondément zoomées, la limite FP32 est atteinte vite, et le FP64 n'est pas une option réaliste sur ce GPU / driver (non vérifié).
- **GLSL 330 suffit** : `fwidth`, `dFdx`, boucles dynamiques, `gl_FragDepth`, `textureLod`, `layout(location)` en sortie sont disponibles. Le projet compile déjà ses shaders en `#version 330` — rester cohérent (pas besoin de 410).
- **Uniforms éliminés** : le compilateur supprime les uniforms inutilisés (le projet a déjà le garde `_members`) ; c'est d'autant plus fréquent avec des variantes `#define` — réutiliser `_write_frame_uniforms`.

### Gaps
- Temps de compilation GLSL et durée du watchdog GPU sur macOS GL-over-Metal : aucune donnée sourcée.
- Comportement de compilation paresseuse (au premier draw) du driver Apple : non confirmé.

## 6. Éclairage et shading SDF dans ce pipeline (réutiliser le Studio PBR)

### Takeaway
Normales par tétraèdre (4 évaluations, boucle anti-inlining), éclairage en **espace caméra** en réutilisant les fonctions GLSL `env()` / `F_schlick` / `studio()` déjà présentes dans `SURFACE_FS`, sortie en **linéaire HDR** comme les autres shaders ; ombres douces et AO optionnelles et coûteuses (≈ +1 marche par pixel pour les ombres).

### Cited Findings
- Technique du tétraèdre et epsilon ∝ footprint : [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/)
- Ombres douces `k*h/t` + variante Aaltonen : [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/)
- Le renderer contient déjà `env(vec3 r, float rough)` (dégradé studio + softbox), `F_schlick`, `studio(n, v, albedo, ao)` (3 lumières GGX + environnement) et l'injection de blocs GLSL par `_glsl()` (`//%NOISE%`, `//%MODEL%`) — `gfx/renderer.py` lignes 113-156 et 286-300 (code local).

### Inferences
- **Normale** (forme anti-inlining) :
  ```glsl
  uniform int u_zero;   // Python envoie 0 : le compilateur ne peut pas dérouler
  vec3 calc_normal(vec3 p, float h) {          // h = u_pix_angle * t (footprint)
      vec3 n = vec3(0.0);
      for (int i = u_zero; i < 4; i++) {
          vec3 e = 0.5773 * (2.0 * vec3(((i + 3) >> 1) & 1, (i >> 1) & 1, i & 1) - 1.0);
          n += e * map(p + e * h);
      }
      return normalize(n);
  }
  ```
- **Réutilisation du Studio** : extraire `env`, `F_schlick`, `studio` dans un bloc `_STUDIO_GLSL` injecté (`//%STUDIO%`) dans `SURFACE_FS` **et** dans le shader de fond, pour que le fond et le modèle partagent les mêmes lumières (qui suivent la caméra depuis T26). Normales et vue en espace caméra : `n_cam = mat3(V) * n_world`, `v = -normalize((V*vec4(hit,1)).xyz)`. Sortie `to_linear()` comme les autres shaders, alpha = 1 si hit (ou 0 et couleur additive pour des formes « néon » émissives qui doivent rester derrière les points).
- **Esthétique d'un fond** : souvent plus efficace et moins cher qu'un PBR complet : couleur émissive par distance (`glow = exp(-k*d_min)` accumulé pendant la marche, gratuit car la marche calcule déjà `d`), brouillard exponentiel en fonction de `t` pour donner la profondeur, rim fresnel. Le bloom du projet transforme ces émissions en halos.
- **AO** : 5 évaluations de `map()` le long de la normale (technique iq classique) ≈ +5 évaluations par pixel touché — acceptable ; ombres douces ≈ une deuxième marche complète → à réserver à un réglage « qualité ».
- **Antialiasing des bords** : en 3D, utiliser la distance minimale du rayon à la surface et la footprint pour un alpha de couverture (`clamp(0.5 - dmin/footprint, 0, 1)`), car le MSAA du projet ne s'applique pas à une texture de fond séparée ; en 2D, `fwidth(d)`.

### Gaps
- La page iq sur l'occlusion ambiante SDF n'a pas été consultée ; la méthode « 5 échantillons le long de la normale » est citée de mémoire.
- Aucune mesure du coût de `studio()` (3 lumières GGX) par pixel sur M1.
