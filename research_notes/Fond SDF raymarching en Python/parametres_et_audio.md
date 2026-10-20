# Système de paramètres d'un fond SDF multi-modes réactif à la musique

*Notes de recherche, 2026-10-18. Contexte : moderngl / OpenGL 3.3 core sur macOS, panneau tkinter → PySide6, presets JSON (global + onglet), matrice « Physique » (7 bandes EQ + BEAT, KICK, SNARE, ÉNERGIE × montant → effet). Les éléments marqués « connaissance générale, non vérifiée ici » viennent de ma connaissance antérieure et n'ont pas été confirmés par une source pendant cette session.*

## Q1 — Comment les outils existants structurent paramètres et modes (Synesthesia, ISF, Fragmentarium, Mandelbulber, TouchDesigner/RayTK, Magic, hg_sdf, fogleman/sdf)

### Takeaway
Tous les outils de shaders « live » convergent vers le même modèle : **un fichier de description déclaratif (JSON ou annotations dans le GLSL) qui liste chaque contrôle avec nom = nom de l'uniform, type, min/max/défaut, groupe d'UI, libellé**, et l'hôte construit le panneau automatiquement. Synesthesia ajoute au type un **comportement temporel** (`smooth`, `speed`, `bang`) et des **transitions automatiques au changement de morceau** ; les outils fractals (Mandelbulber, Fragmentarium) ajoutent des **groupes, verrous et presets dans le fichier** ; les outils nodaux (RayTK, fogleman/sdf, hg_sdf) structurent plutôt la *forme* comme un graphe d'opérateurs (primitives, booléens lissés, opérateurs de domaine).

### Cited Findings
**Synesthesia (SSF — Synesthesia Scene Format)**
- Une scène est un dossier `.synScene` contenant `main.glsl` (fragment shader rendu chaque frame), `scene.json` (métadonnées, contrôles du panneau, passes, transitions, textures), un `script.js` optionnel exécuté avant le shader chaque frame et qui passe des valeurs calculées comme uniforms, une vignette et un dossier `images/` — [Synesthesia SSF](https://app.synesthesia.live/docs/ssf/ssf.html)
- Types de contrôle : `"slider"`, `"slider smooth"`, `"slider speed"`, `"knob"` (+ smooth/speed), `"toggle"`, `"toggle smooth"`, `"bang"`, `"bang smooth"`, `"bang counter"`, `"bang counter smooth"`, `"xy"` (+ smooth/speed), `"color"`, `"color smooth"`, `"dropdown"`, `"dropdown smooth"` — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)
- Champs d'un contrôle : `TYPE`, `NAME` (devient un uniform ; `_` affichés comme espaces), `DEFAULT` (index pour un dropdown, tableau pour multi-dim., `DEF_COLOR` pour les couleurs), `MIN`/`MAX`, `UI_GROUP` (groupe créé automatiquement), `DESCRIPTION` (tooltip), `PARAMS` (vitesse de lissage, défaut 0,01, transition ≈ 1/PARAMS frames), `LABELS`/`VALUES` pour les dropdowns — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)
- Les types « speed » produisent une valeur qui avance en continu à la vitesse réglée (« mouvement infini ») ; « bang smooth » produit une brève enveloppe dont la durée est réglée par `PARAMS` — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)
- Transitions automatiques : `HARD_TRANSITIONS` (`UNIFORM` + `VALUES`, une nouvelle valeur tirée à chaque changement de morceau détecté) et `SMOOTH_TRANSITIONS` (`UNIFORMS` = famille dont un seul membre est « ON » à la fois, `DURATION` en secondes, « se combine bien avec `mix()` ») — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)
- Multi-passes : `PASSES` avec `TARGET`, `WIDTH`/`HEIGHT`, `FLOAT`, `DOUBLE_BUFFER` (lire la frame précédente) ; `PASSINDEX` pour brancher par passe — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)

**ISF (Interactive Shader Format, VIDVOX — VDMX, Resolume, etc.)**
- L'en-tête est un dictionnaire JSON dans un commentaire `/* */` en tête du shader ; sans en-tête valide le fichier ne charge pas ; `ISFVSN` = version de la spec (ex. "2.1"), absent → 1.0 ; `DESCRIPTION`, `CATEGORIES` — [ISF Spec](https://github.com/mrRay/ISF_Spec)
- `INPUTS` = tableau de dicts affichés dans l'ordre ; `NAME` (sans espace, devient la variable du shader), `TYPE` parmi `event`, `bool`, `long`, `float`, `point2D`, `color`, `image`, `audio`, `audioFFT` ; `LABEL`, `DEFAULT`, `MIN`, `MAX`, `IDENTITY` optionnels ; `VALUES`/`LABELS` parallèles pour `long` (menu) ; pour `audio`/`audioFFT`, `MAX` = nombre d'échantillons, données passées en image (une ligne par canal) — [ISF Spec](https://github.com/mrRay/ISF_Spec)
- Uniforms automatiques : `TIME`, `TIMEDELTA`, `FRAMEINDEX`, `RENDERSIZE`, `DATE`, `PASSINDEX`, `isf_FragNormCoord` ; `PASSES` avec `TARGET`, `PERSISTENT`, `FLOAT`, `WIDTH`/`HEIGHT` en expressions (`$WIDTH`…) — [ISF Spec](https://github.com/mrRay/ISF_Spec)

**Fragmentarium (et Synthclipse, compatible)**
- Annotations directement après la déclaration d'uniform dans le GLSL : `#group Raytracer`, `uniform float Detail;slider[-7,-2.3,0];` (min, défaut, max), `uniform vec2 SpotLightDir; slider[(-1,-1),(0.1,0.1),(1,1)]`, `uniform vec3 BackgroundColor; color[0.6,0.6,0.45]`, `uniform bool CycleColors; checkbox[false]`, et le modificateur `Locked` (`slider[0,1,10] Locked`) ; composition par `#include "3D.frag"` — [DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)
- Synthclipse : format `slider[Min, Value, Max]` pour int/uint/float/double/vec2-4 (`uniform float FloatSlider; //! slider[-5, 1, 7]`) ; blocs de presets `<preset></preset>` ; le preset par défaut initialise tout au chargement, les autres presets ne contiennent que les uniforms qu'ils changent — [Synthclipse commands](https://synthclipse.sourceforge.net/user_guide/fragx/commands.html), [Synthclipse uniform controls](https://synthclipse.sourceforge.net/user_guide/fragx/uniform_controls.html), [Fractal Forums](https://www.fractalforums.com/fragmentarium/slider-t22662)

**Mandelbulber / Mandelbulber2**
- Mode « hybrid » : empilement de jusqu'à cinq formules fractales dans un même système (article ancien, version 1) — [Linux Journal](https://www.linuxjournal.com/node/1233881)
- Un hybride à 5 slots de composants Mandelbox dépend de 24 paramètres réglés individuellement (expérience d'un utilisateur) — [Fractal Forums](https://www.fractalforums.com/mandelbulber/mandelbulber-v2-animation)
- « Key-frame animation for all parameters with different interpolations » ; formules trigonométriques, hypercomplexes, Mandelbox, IFS et bien d'autres — [Mandelbulber2 GitHub](https://github.com/buddhi1980/mandelbulber2)

**TouchDesigner / RayTK**
- RayTK : bibliothèque libre de composants pour construire des scènes de raymarching sans écrire de GLSL, opérateurs câblés comme des nœuds TD (menu ALT+R) ; inspirée de RTK de Patrik Lechner ; SDF, transformations, matériaux, éclairage, répétition infinie, fractals, réflexions, motifs 2D, échantillonnage volumétrique ; opérateur `customRender` pour écrire son propre shader en appelant les opérateurs RayTK comme fonctions ; addons raytkVolumes / raytkAbstractions (Patreon) — [Derivative RayTK](https://derivative.ca/community-post/asset/raytk-raymarching-masses/63620), [RayTK releases](https://github.com/t3kt/raytk/releases/latest), [Workshop 2024](https://derivative.ca/workshop/raymarching-shader-programming-raytk/70463)
- T3D (Josef Pelz) : famille d'opérateurs 3D textures, add-on payant séparé, souvent associé à RayTK — [Derivative workshop T3D](https://derivative.ca/workshop/raymarching-3d-textures-t3d/70464)

**Magic Music Visuals** (sources : forum, pas le manuel officiel — page du manuel en 403)
- Lissage = filtre exponentiel une fois par frame, force 0–1 (1 = garde la valeur précédente, 0 = suit l'entrée) — [Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=1468)
- Modificateurs Smooth, Peak, Average « à utiliser avec modération sinon la réaction est léthargique » ; masquer l'amplitude avec Gate ou Clamp réduit le jitter ; combiner masquage fréquence+amplitude et lissage — [Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=11958)
- Rampe lente ou « Increase » = façon la plus simple de moduler dans le temps ; enveloppes Triangle/Sine/Linear ; Globals + expressions ; depuis 2.11 les modificateurs Expression peuvent utiliser les Globals (ex. aléatoire appliqué seulement quand le beat dépasse un seuil) — [Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=13554), [Magic 2.11](https://magicmusicvisuals.com/forums/viewtopic.php?p=6596)

**Bibliothèques SDF**
- hg_sdf (Mercury, utilisé en intros 64k) : préfixes `f` (distance), `p` (opérateur de domaine : `pMod1`, `pModSingle1`, `pModInterval1`, `pModPolar`, `pMod2`, `pModMirror2`, `pMod3`), `fOp` (combinaison : `fOpUnionRound`, `fOpUnionChamfer`, `fOpUnionColumns`, `fOpUnionStairs` + variantes intersection/différence, `fOpPipe`, `fOpEngrave`, `fOpGroove`, `fOpTongue`) ; principe : « rester Lipschitz continu » (gradient ≤ 1), ne pas multiplier les distances pour corriger ; licence MIT OU CC-BY-NC-4.0 (2021-07-28) — [hg_sdf](https://mercury.sexy/hg_sdf/)
- fogleman/sdf : API Python chaînable (`sphere(1) & box(1.5)`, `|`, `&`, `-`, lissage `.k(0.25)`, `translate/rotate/scale/twist/bend/repeat/elongate/shell/dilate/erode`, `extrude`/`revolve`/`slice`) mais **évaluée sur CPU (numpy, threads) pour produire un maillage STL** ; ne génère pas de GLSL — [fogleman/sdf](https://github.com/fogleman/sdf)

### Inferences
- **Le modèle « une déclaration = un uniform + un widget » (ISF / SSF / Fragmentarium) est directement transposable** : un fichier `fond_modes.json` (ou un en-tête JSON par fichier `.glsl` de mode, façon ISF) lu par Python, qui (a) génère les déclarations `uniform` à injecter dans le shader, (b) construit les widgets PySide6 groupés par `group`, (c) donne la liste des clés pour les presets et la matrice Physique. Parser un en-tête JSON dans un commentaire `/* */` est trivial (`json.loads` sur le texte entre le premier `/*` et `*/`) ; parser des annotations Fragmentarium (`slider[a,b,c]`) demande une regex mais reste simple. Le JSON est préférable : validable, testable sans GL (règle du projet : logique sans GL → module + tests).
- Les variantes `smooth` / `speed` / `bang` de Synesthesia montrent qu'il faut **séparer la valeur d'un contrôle de son comportement temporel** : un champ `smoothing` (constante de temps) et un type `phase` (intégré, `value += speed·dt`) évitent que chaque mode réinvente ces filtres en GLSL. Dans ce projet, l'intégration en `dt` doit se faire côté Python (le panneau vit dans un autre process).
- `SMOOTH_TRANSITIONS` (famille d'uniforms dont un seul est ON, fondu de N s) est exactement le motif utile pour des **sous-variantes d'un même mode** (ex. 4 types de pliage d'un fractal) : envoyer des poids 0–1 qui somment à 1 et faire `mix()` dans le shader, plutôt qu'un entier.
- `HARD_TRANSITIONS` au changement de morceau = idée simple pour un **mode auto** (lien avec T10) : retirer certains paramètres discrets au changement de section (`section` du projet) plutôt qu'au changement de morceau.
- hg_sdf fournit la grammaire de base pour le mode « géométrique répété » (opérateurs `pMod*`) et des combinaisons non triviales (`fOpUnionStairs`, colonnes, chanfrein) qui donnent des formes « vraiment différentes » à coût faible ; la contrainte Lipschitz implique que les paramètres audio qui déforment le domaine (twist, bend, bruit) doivent s'accompagner d'un facteur de pas réduit (« step factor »/« fudge ») exposé comme paramètre qualité.
- fogleman/sdf est utile comme **modèle d'API** pour un générateur GLSL côté Python (mêmes opérateurs, mais émission de code GLSL au lieu d'évaluation numpy) ; je n'ai pas trouvé de bibliothèque Python maintenue qui fasse déjà cette génération GLSL (voir Gaps).

### Gaps
- Manuel officiel de Magic Music Visuals inaccessible (HTTP 403) : liste exacte des modificateurs non vérifiée, seulement des posts de forum.
- Pas de source trouvée pour la syntaxe `#preset … #endpreset` de Fragmentarium (seulement l'équivalent Synthclipse `<preset>`). Statut de maintenance de Fragmentarium (fork FragM) non vérifié — l'original Syntopia semble ancien, à considérer comme potentiellement daté.
- Notch, hydra, cables.gl, Kodelife/Bonzomatic, VDMX non étudiés en détail faute de budget d'appels. Connaissance générale, non vérifiée ici : hydra chaîne des fonctions JS (`osc().rotate().kaleid().out()`) où tout argument peut être une fonction `() => a.fft[0]` ; Shadertoy fournit `iTime`, `iResolution`, `iMouse`, `iChannel0..3`, et une entrée audio sous forme de texture 512×2 (ligne FFT + ligne forme d'onde).
- Mandelbulber2 : types d'interpolation de keyframes et format exact des slots hybrides non documentés dans les pages consultées (renvoi à `mandelbulber_doc`).
- Je n'ai pas trouvé de bibliothèque « SDF → GLSL codegen » en Python maintenue en 2026 (recherche non faite faute de budget ; ne pas en citer sans vérifier).

## Q2 — Hiérarchie des paramètres, macros, aléatoire, mutation, morphing de presets

### Takeaway
Structurer en **trois niveaux** : globaux du fond (mode, 2D/3D, qualité, palette, vitesse, intensité), paramètres propres au mode (déclarés par mode), et **macros** (Complexité, Chaos, Densité, Organique↔Géométrique) qui pilotent plusieurs paramètres bas niveau via des courbes déclarées. Le dé (« dice ») tire dans des **plages contraintes par paramètre** à partir d'un **seed** sauvegardé ; la mutation (Electric Sheep / Picbreeder) perturbe autour du preset courant ; le morphing interpole uniquement les paramètres continus, les discrets étant **gelés, basculés au milieu, ou remplacés par un fondu enchaîné (crossfade d'images)**.

### Cited Findings
- Electric Sheep : le génotype (« DNA ») d'un mouton ≈ 130 paramètres contrôlant couleur, mouvement et forme ; les nouveaux moutons naissent par croisement des paramètres de deux parents bien notés + mutations aléatoires ; votes des spectateurs = fonction de fitness ; des utilisateurs avancés soumettent leurs propres paramètres, des « shepherds » guident l'accouplement — [Wikipedia Electric Sheep](https://en.wikipedia.org/wiki/Electric_Sheep)
- Picbreeder : à chaque génération l'utilisateur choisit les images prometteuses, qui sont croisées et mutées ; images = CPPN évoluées par NEAT ; pas de spéciation car les reproducteurs sont choisis directement par l'utilisateur ; branchement possible depuis le travail publié d'autres utilisateurs — [UT Austin slides](https://www.cs.utexas.edu/~risto/cs378ne/private/slides/8.3-8.4.Interactive-4up.pdf), [arXiv 1704.05143](https://arxiv.org/pdf/1704.05143)
- Morphing de presets (synthés) : les paramètres discrets (on/off, menus) ne s'interpolent pas, ils sautent et provoquent des clics ; solution = ne morpher que les potentiomètres/curseurs continus — [SonicState 2026](https://www.sonicstate.com/news/2026/02/26/using-snappysnap-with-synths/)
- Proposition sur forum : maintenir les paramètres discrets fixes pendant un morph plutôt que de les laisser basculer au milieu — [Polyend forum](https://backstage.polyend.com/t/morph-between-two-presets/20141)
- Morphing ≠ crossfade : le crossfade fait disparaître un son pendant qu'un autre apparaît ; le morphing interpole les paramètres eux-mêmes — [Yamaha](https://yamahasynth.com/?p=3339)
- Recherche : interpolation de presets complets dans un espace latent (autoencodeur bimodal, ICASSP 2023 ; TASLP 2024 ; thèse 2025) donne des interpolations plus douces que le mélange linéaire paramètre par paramètre — [arXiv 2210.16984](https://www.arxiv.org/abs/2210.16984), [TASLP 2024](https://orbi.umons.ac.be/bitstream/20.500.12907/49507/1/taslp24_accepted.pdf)
- Synesthesia `"dropdown smooth"` et `SMOOTH_TRANSITIONS` : même un choix discret est rendu continu en le transformant en famille de poids mélangés par `mix()` sur une durée — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)
- Fragmentarium `Locked` : marquer un paramètre pour qu'il ne soit pas modifié (utilisé pour les paramètres techniques du raytracer) — [DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)
- Synthclipse : un preset ne contient que les uniforms qu'il change, le preset par défaut initialise le reste — [Fractal Forums](https://www.fractalforums.com/fragmentarium/slider-t22662) (cohérent avec le comportement T24 du projet : « charger un preset remet à leur défaut les réglages qu'il ne contient pas »)

### Inferences
**Hiérarchie recommandée**
1. *Globaux du fond* (communs à tous les modes) : `mode`, `dim` (2D/3D), `quality` (pas max, epsilon, résolution relative du fond — ex. rendu à ½ résolution puis upsample), `palette` + `hue`, `speed` (multiplie toutes les phases), `intensity`/`exposure`, `fog`, `seed`, `mix` avec le modèle (opacité/profondeur du fond).
2. *Paramètres du mode* : déclarés dans la description du mode (ex. fractal : `power`, `iterations`, `fold_limit`, `scale`, `offset`, `fold_type` ; métaballes : `count`, `radius`, `smooth_k`, `orbit_speed` ; répété : `cell`, `primitive`, `twist`, `mirror` ; paysage : `octaves`, `height`, `ridge`, `scroll`).
3. *Macros* (4 à 6 curseurs « artistiques » toujours visibles en haut de l'onglet) : chaque paramètre du mode déclare comment il répond à une macro (`"macros": {"complexity": [0.2, 1.0]}` = plage atteinte pour macro 0→1, ou une courbe). Une macro est un point de routage comme une bande audio : elle s'additionne au paramètre, ce qui permet aussi de **router l'audio vers une macro** (ÉNERGIE → Complexité) au lieu de 15 paramètres — gros gain d'utilisabilité quand les paramètres sont « très nombreux ».
   - Complexité : itérations, octaves, nombre d'objets, détail.
   - Chaos : amplitude de bruit de domaine, twist, désordre de répétition, vitesse.
   - Densité : taille de cellule, nombre de métaballes, brouillard.
   - Organique↔Géométrique : `smooth_k` des unions, arrondi, poids bruit vs pliage.
- **Dé (« dice ») contraint** : chaque paramètre déclare une plage aléatoire `rand` plus étroite que `min/max` (les zones « belles ») et une distribution (`uniform`, `log` pour échelles/fréquences, `choice` pour les énumérations avec poids). Le tirage est déterministe à partir d'un `seed` entier stocké dans le preset → un preset peut n'être que `{mode, seed}` + surcharges. Prévoir des boutons « dé global », « dé de ce groupe », et un cadenas par paramètre (équivalent `Locked` de Fragmentarium) exclu du tirage.
- **Mutation / évolution** : « Muter » = perturbation gaussienne autour du preset courant avec un taux 0–1 (σ = taux × largeur de `rand`), discrets changés avec probabilité = taux. Une grille de 6 à 9 variantes mutées (vignettes rendues hors écran, cf. `tools/snapshot.py`) où l'utilisateur clique la meilleure reprend la boucle Picbreeder ; le croisement à la Electric Sheep = pour chaque paramètre, valeur d'un parent au hasard ou interpolation. Électric Sheep montre qu'un génome de ~130 paramètres reste évolutif s'il est sélectionné par l'humain.
- **Morphing** (lien T10) : classer chaque paramètre via un champ `interp` :
  - `linear` (défaut floats) ; `log` (échelles, fréquences, itérations fractionnaires) ; `angle` (chemin le plus court, modulo 2π) ; `color` (interpoler en Oklab/linéaire, pas en sRGB) ;
  - `snap` (entiers sans sens continu : bascule à t = 0,5) ; `freeze` (garde la valeur de départ jusqu'à la fin) ;
  - `crossfade` (le `mode` lui-même, `dim`, la primitive de base) : impossible d'interpoler → rendre les deux modes dans deux FBO et fondre les images (ou, en 3D, `mix` des distances `d = mix(dA, dB, t)` si les deux modes sont dans le même programme : morphing géométrique réel, plus spectaculaire mais double le coût par pas).
  - Les itérations d'un fractal peuvent devenir continues en interpolant entre l'itération `n` et `n+1` (fraction), sinon `snap`.

### Gaps
- Pas de source spécifique sur la conception de **macros** dans un outil VJ (les « Abstractions » de RayTK et les « macros » de Notch n'ont pas été documentées dans les pages consultées).
- Mécanisme exact de transition entre moutons d'Electric Sheep (« edges » interpolées) non confirmé par la page Wikipedia consultée.
- Pas de mesure publiée trouvée sur le nombre « idéal » de paramètres exposés ; la recommandation 4–6 macros est une inférence.

## Q3 — Changer de mode à l'exécution : uber-shader, un programme par mode, génération de GLSL ; saccade de compilation ; fondu entre modes

### Takeaway
Pour ~5 familles de modes, le compromis recommandé est **un programme compilé par mode (spécialisé par `#define`) précompilé au démarrage**, avec des branches uniformes seulement pour les sous-variantes légères ; la génération de GLSL depuis un graphe Python est puissante mais réserve la compilation à l'édition structurelle. Sur macOS, **les binaires de programme ne sont pas disponibles** : il faut tout compiler depuis le source au lancement (ou en tâche de fond, un mode par frame) pour éviter la saccade au premier affichage.

### Cited Findings
- Branches sur uniform : moins chères que des branches sur attribut de sommet, elles-mêmes moins chères que sur valeur interpolée ou échantillonnée ; le compilateur peut parfois transformer des branches dynamiques en statiques — [Khronos forum](https://community.khronos.org/t/ubershader-and-branching-cost/108571/3), [gamedev.net](https://gamedev.net/forums/topic/659145-what-is-a-uber-shader/)
- Coût caché de l'uber-shader : on paie les registres nécessaires au code complet même si une petite partie s'exécute → occupation réduite → perte de performance — [gamedev.net](https://gamedev.net/forums/topic/715362-why-shader-permutation/)
- Permutations : chaque combinaison est compilée séparément, explosion combinatoire possible ; voie médiane courante = spécialiser à la compilation par macros et ne garder que des branches uniformes — [gamedev.net](https://www.gamedev.net/forums/topic/512455-shader-permutation-problem--again-/?page=1), [Khronos forum](https://community.khronos.org/t/multiple-shaders-vs-dynamic-branching/57239)
- Binaires de programme : optionnels et dépendants du pilote, le pilote peut annoncer zéro format ; les pilotes Intel macOS n'en supportent aucun ; interroger `GL_NUM_PROGRAM_BINARY_FORMATS` et compiler depuis le source si 0 ; un binaire peut être refusé après un changement matériel/logiciel — [Khronos forum](https://community.khronos.org/t/binary-shaders-support-in-opengl/68264), [O'Reilly OpenGL 4 Shading](https://www.oreilly.com/library/view/opengl-4-shading/9781789342253/bae34249-478a-44d2-b0e6-17b3ce90f067.xhtml), [Khronos wiki](https://wikis.khronos.org/opengl/GlProgramBinary)
- `glProgramBinary` est core depuis GL 4.1 — [Khronos refpage](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glProgramBinary.xhtml) (le projet est en contexte 3.3 core ; macOS plafonne à 4.1 — connaissance générale)
- RayTK : les scènes sont des graphes d'opérateurs TouchDesigner qui génèrent le shader (« build and render raymarching shaders from ordinary TouchDesigner networks ») — [Derivative RayTK](https://derivative.ca/community-post/asset/raytk-raymarching-masses/63620)
- Synesthesia `SMOOTH_TRANSITIONS` + `mix()` : fondu entre variantes dans un même shader — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)

### Inferences
- **Option A — uber-shader unique** (`if (u_mode == 0) … else if …`) : changement instantané, morphing de distance `mix(dA, dB, t)` possible entre deux modes ; mais registres du pire cas pour tous, compilation longue unique, et un bug GLSL d'un mode casse tous les modes. Acceptable si les modes sont légers, risqué avec un fractal à 10+ itérations + paysage à octaves de bruit.
- **Option B — un programme par mode** (même squelette de raymarcher/éclairage, la fonction `map()` du mode injectée par `#define MODE_x` ou concaténation de fichiers `fond/<mode>.glsl`) : chaque programme ne paie que son coût ; erreur de compilation isolée à un mode (s'intègre à `App.report()` / `UserError` : le mode courant reste affiché tant que le nouveau n'est pas compilé, conformément à la règle « ne jamais remplacer le modèle courant avant que le nouveau soit valide »). **Recommandé.**
- **Précompilation** : compiler tous les programmes au démarrage (≈ 5 programmes, typiquement quelques dizaines à quelques centaines de ms chacun — ordre de grandeur non mesuré ici, à profiler) ou en tâche de fond un programme par frame (le contexte GL est mono-thread, donc pas de vrai thread de compilation avec moderngl ; on étale). Ne pas compter sur `glProgramBinary` sur macOS. Un rendu « de chauffe » d'un quad 1×1 après compilation force le pilote à finaliser le programme (pratique courante, non sourcée ici).
- Les **sous-variantes** d'un mode (type de pliage, primitive répétée) restent des branches uniformes ou des poids `mix()` dans le programme du mode : coût modéré, morphing possible.
- **Option C — génération de GLSL depuis un graphe Python** (API à la fogleman/sdf, briques hg_sdf) : maximise la variété (« vraiment différentes formes » : union lissée de primitives tirées au sort, chaîne d'opérateurs de domaine tirée au sort), et le dé peut alors tirer aussi la *structure*. Chaque changement de structure = recompilation (saccade possible) → recompiler hors du chemin critique et n'installer qu'une fois compilé ; les paramètres numériques du graphe restent des uniforms (tableau `uniform float u_p[64]` indexé par le générateur) pour que l'audio et les sliders ne recompilent jamais. À réserver à un « mode Composé » plutôt qu'à tous les modes.
- **Fondu entre modes** : rendre l'ancien et le nouveau mode dans deux FBO du fond pendant 0,5–2 s et mélanger (coût doublé pendant la transition, éventuellement à demi-résolution) ; ou, si le fond a déjà une couche trainées, simplement basculer et laisser la persistance lisser. En 3D, une dissolution par bruit ou un fondu en profondeur (brouillard qui monte puis redescend) masque le changement de silhouette.

### Gaps
- Aucun temps de compilation mesuré pour un raymarcher fractal sur pilote Apple (M1/M2/Intel) trouvé ; à mesurer dans le projet.
- Comportement exact de moderngl (compilation synchrone) non vérifié par une source ici.

## Q4 — Mapping audio pour un fond : enveloppes lentes, impulsions, phase calée sur le beat, énergie de section ; anti-scintillement ; courbes

### Takeaway
La pratique Synesthesia formalise quatre familles de signaux qui correspondent à quatre usages : **Level** (suivi continu lissé → échelle, luminosité), **Hits** (transitoires → déformation ponctuelle), **Presence** (montée/descente de section → complexité, structure), **Time** (horloges qui accélèrent avec le volume → mouvement, défilement, caméra) ; plus des signaux de **beat/BPM** pour les mouvements calés. Un fond doit surtout utiliser Presence/Time/phase de beat, et réserver les impulsions à quelques paramètres, avec lissage asymétrique et seuils (gate) pour éviter le scintillement.

### Cited Findings
- Familles d'uniforms audio Synesthesia : **Level** (lissé, normalisé 0–1 : `syn_Level`, `syn_BassLevel`, `syn_MidLevel`, `syn_MidHighLevel`, `syn_HighLevel`), **Hits** (pics sur transitoires isolés : `syn_BassHits`…), **Time** (horloges non normalisées qui avancent plus vite quand la bande est forte : `syn_Time`, `syn_BassTime`… ; `syn_CurvedTime` accélère beaucoup plus fort), **Presence** (activité soutenue montante/descendante, ignore les sons individuels) — [Synesthesia audio uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html)
- Beat : `syn_OnBeat` (saute à 1 puis revient vite à 0), `syn_ToggleOnBeat` (alterne 0/1 à chaque beat avec transitions douces), `syn_RandomOnBeat` (nouvelle valeur aléatoire lissée à chaque beat), `syn_BeatTime` (+1 par beat) ; BPM : `syn_BPM` (50–220), `syn_BPMConfidence`, `syn_BPMTwitcher` (`floor()` = compteur propre, `fract()` = jitter par beat), `syn_BPMSin/Sin2/Sin4/Sin8`, `syn_BPMTri…` ; grandes structures : `syn_FadeInOut`, `syn_Intensity` (monte lentement vers 1 selon l'intensité globale du morceau) — [Synesthesia audio uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html)
- Recommandations Synesthesia : Level pour échelle/luminosité qui suit le volume ; Hits pour réactions percussives ; **Presence pour les sections qui montent/chutent, avec moins de jitter que Hits** ; Time pour un mouvement qui accélère avec la musique (« caméra qui pousse en avant à chaque kick ») ; BPM pour un mouvement régulier sur le groove ; **vérifier `syn_BPMConfidence` avant de se fier au mouvement calé BPM** ; `syn_Spectrum` (canal « Juiced » recommandé) pour les formes ; `syn_LevelTrail` pour les traînées — [Synesthesia audio uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html)
- Magic : lissage exponentiel par frame ; Smooth/Peak/Average avec modération sinon réaction « léthargique » ; Gate ou Clamp sur l'amplitude contre le jitter ; rampes lentes / Increase pour moduler dans le temps — [Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=1468), [Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=11958), [Magic forum](https://magicmusicvisuals.com/forums/viewtopic.php?p=13554)
- Synesthesia types `speed` : sortie qui avance en continu (mouvement infini) — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)

### Inferences
**Correspondance avec les sources du projet** (`app/routing.py`) :

| Famille Synesthesia | Équivalent projet | Usage recommandé dans le Fond |
|---|---|---|
| Level | bandes EQ lissées (Envelope T4) | luminosité, couleur, épaisseur, échelle douce |
| Hits | KICK (ressort), SNARE (flash) | déformation ponctuelle : gonflement des métaballes, onde de choc sur le paysage, pli fractal qui « claque » puis revient |
| Presence | ÉNERGIE de section | complexité (itérations, octaves, nombre d'objets), densité, brouillard — via les macros |
| Time | **à ajouter** : phase intégrée `φ += dt·(base + k·niveau)` | défilement du paysage, avance de la caméra 2D, rotation du domaine, évolution du bruit |
| BeatTime / BPMSin | `beat_phase` (+ compteur de beats à ajouter) | rotation par quart de tour sur le beat, pulsation calée, `ToggleOnBeat` pour alterner deux variantes |
| RandomOnBeat | à ajouter (seed changé au beat, lissé) | variation de position/couleur au beat sans scintillement |

- **Règle principale anti-jitter : ne jamais router un signal rapide vers un paramètre qui change la topologie** (itérations entières, nombre de cellules, type de pliage) : le saut produit un scintillement plein écran. Ces paramètres ne reçoivent que des sources lentes (ÉNERGIE, macros) et sont soit fractionnaires (interpolation n/n+1), soit quantifiés au beat (changent seulement sur un beat → rythmique et non bruyant).
- **Intégrer plutôt que moduler les positions** : router une bande sur une *vitesse* (phase intégrée, type Time/`speed`) et non sur une *position/angle* directement ; une position modulée tremble, une vitesse modulée produit une accélération fluide. Ce principe existe déjà dans le projet pour la rotation musicale (densité de transients → vitesse).
- **Lissage asymétrique** (attack court, release long — déjà Envelope T4) + **gate** (seuil sous lequel la source vaut 0, cf. Gate/Clamp de Magic) + **courbe** par route (`pow(x, γ)` : γ > 1 rend la réaction plus « punchy » et coupe le bruit de fond ; `smoothstep(lo, hi, x)` pour une plage utile). Proposer dans la matrice Physique, par route ou par paramètre, `curve ∈ {lin, pow2, pow0.5, smoothstep}` et `slew` (vitesse max de variation).
- **Les fonds en raymarching sont sensibles à l'aliasing temporel** : un petit changement de paramètre fractal produit des changements de détail haute fréquence → limiter la vitesse de variation (`slew`) des paramètres de forme et préférer la couche trainées / un accumulateur temporel pour le fond.
- **Confiance BPM** : comme Synesthesia, les animations calées sur le beat devraient se replier sur une phase libre quand la confiance du tracker T2 est faible (exposer une confiance si `LiveAnalyzer` peut la calculer).
- **Fond vs objet** : le fond doit rester plus lent que le modèle 3D (sinon il vole l'attention) ; recommander des montants par défaut faibles sur les sources impulsionnelles et un « Dynamique » propre au fond.

### Gaps
- Pas de documentation officielle Magic consultée (403) sur ses recommandations de mapping.
- Pas trouvé de guide publié spécifique au mapping audio d'un *fond* (vs. sujet principal) ; les recommandations ci-dessus sont des inférences.
- Formule exacte de `syn_Presence` / `syn_Intensity` non documentée (comportement seulement).

## Q5 — Format concret de liste de paramètres (JSON) adapté aux presets

### Takeaway
Un schéma inspiré d'ISF (`NAME/TYPE/MIN/MAX/DEFAULT/LABEL/VALUES/LABELS`) et de SSF (`UI_GROUP`, `DESCRIPTION`, comportement `smooth`/`speed`), enrichi des champs propres au projet : `routable` (entre dans la matrice Physique), `interp` (règle de morphing), `rand` (plage du dé), `macros` (réponse aux macros), `topology` (paramètre qui change la structure → sources lentes seulement), `locked`. Les presets ne stockent que des valeurs (et éventuellement un `seed`), jamais le schéma.

### Cited Findings
- ISF : `NAME`, `TYPE` (`event`, `bool`, `long`, `float`, `point2D`, `color`…), `LABEL`, `DEFAULT`, `MIN`, `MAX`, `IDENTITY`, `VALUES`/`LABELS` — [ISF Spec](https://github.com/mrRay/ISF_Spec)
- SSF : `UI_GROUP`, `DESCRIPTION`, `PARAMS` (lissage), types `smooth`/`speed`/`bang`, `HARD_TRANSITIONS`/`SMOOTH_TRANSITIONS` — [Synesthesia JSON](https://app.synesthesia.live/docs/ssf/json.html)
- Fragmentarium : `#group`, `slider[min,défaut,max]`, `color[...]`, `checkbox[...]`, `Locked` — [DE-Raytracer.frag](https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/DE-Raytracer.frag)
- Presets partiels (seuls les uniforms changés) + défaut pour le reste — [Fractal Forums](https://www.fractalforums.com/fragmentarium/slider-t22662)

### Inferences
**Schéma proposé** (`gfx/fond/modes.json` ou en-tête JSON `/* */` en tête de chaque `fond/<mode>.glsl`, à la ISF) :

```json
{
  "version": 1,
  "globals": [
    {"name": "mode", "type": "enum", "values": ["fractal", "metaballs", "repeat", "landscape"],
     "labels": ["Fractale", "Organique", "Géométrique", "Paysage"], "default": "fractal",
     "group": "Fond", "routable": false, "interp": "crossfade", "rand": {"weights": [3, 2, 2, 1]}},
    {"name": "dim", "type": "enum", "values": ["2d", "3d"], "default": "3d",
     "group": "Fond", "routable": false, "interp": "crossfade"},
    {"name": "speed", "type": "float", "min": 0, "max": 4, "default": 1, "group": "Fond",
     "routable": true, "interp": "log", "rand": {"range": [0.3, 1.5], "dist": "log"}},
    {"name": "seed", "type": "int", "min": 0, "max": 2147483647, "default": 0,
     "group": "Fond", "routable": false, "interp": "snap"}
  ],
  "macros": ["complexity", "chaos", "density", "organic"],
  "modes": {
    "fractal": {
      "label": "Fractale",
      "glsl": "fond/fractal.glsl",
      "params": [
        {"name": "iterations", "type": "float", "min": 2, "max": 14, "default": 8, "step": 0.01,
         "group": "Forme", "label": "Itérations", "uniform": "u_f_iter",
         "routable": true, "topology": true, "interp": "linear",
         "macros": {"complexity": [4, 12]}, "rand": {"range": [5, 10]},
         "description": "Fractionnaire : mélange l'itération n et n+1, pas de saut."},
        {"name": "fold", "type": "enum", "values": ["box", "sphere", "kifs", "menger"],
         "default": "box", "group": "Forme", "uniform": "u_f_fold_w",
         "encoding": "weights", "smoothing": 1.5,
         "routable": false, "interp": "crossfade_weights", "rand": {"weights": [2, 1, 2, 1]}},
        {"name": "scale", "type": "float", "min": -3, "max": 3, "default": 2,
         "group": "Forme", "routable": true, "interp": "linear",
         "macros": {"chaos": [1.8, 2.6]}, "rand": {"range": [1.5, 2.8]}},
        {"name": "rotation", "type": "phase", "min": 0, "max": 2, "default": 0.1,
         "group": "Mouvement", "label": "Vitesse de rotation",
         "routable": true, "interp": "linear",
         "description": "Type phase : la valeur est une vitesse, Python intègre l'angle en dt."},
        {"name": "punch", "type": "float", "min": 0, "max": 1, "default": 0.3,
         "group": "Réaction", "routable": true, "prefer_sources": ["KICK", "SNARE"],
         "interp": "linear", "slew": 8.0},
        {"name": "tint", "type": "color", "default": [0.9, 0.5, 0.2],
         "group": "Couleur", "routable": false, "interp": "oklab"},
        {"name": "step_factor", "type": "float", "min": 0.3, "max": 1, "default": 0.8,
         "group": "Qualité", "locked": true, "rand": null, "routable": false, "interp": "snap"}
      ]
    }
  }
}
```

Champs et rôle :
- `name` (clé de preset et de routing, stable), `uniform` (nom GLSL, défaut `u_<mode>_<name>`), `label`, `group`, `description` (tooltip) — comme ISF/SSF.
- `type` : `float`, `int`, `bool`, `enum` (menu ; `encoding: "index"` → `int` uniform, ou `"weights"` → `vec4` de poids lissés à la SSF `SMOOTH_TRANSITIONS`), `color`, `vec2`/`vec3` (xy, direction), `phase` (vitesse intégrée côté Python), `event` (bang → impulsion décroissante, `smoothing` = durée).
- `min`/`max`/`default`/`step` ; `rand` = plage + `dist` (`uniform`/`log`/`choice` + `weights`), `null` = exclu du dé ; `locked` = exclu par défaut du dé et de la mutation (cadenas modifiable dans l'UI).
- `routable` : apparaît comme cible dans la matrice Physique ; `prefer_sources` = suggestion d'UI (ex. `topology: true` → seules ÉNERGIE et macros proposées, ou les sources rapides quantifiées au beat).
- `topology` : le paramètre change la structure → `slew` obligatoire, sources rapides interdites ou quantifiées au beat.
- `macros` : plage atteinte pour macro 0→1 (additif autour de la valeur de base, ou courbe `[[0,a],[0.5,b],[1,c]]`).
- `interp` : `linear` | `log` | `angle` | `oklab` | `snap` | `freeze` | `crossfade` | `crossfade_weights`.
- `smoothing` (s, lissage de la valeur envoyée, à la SSF `PARAMS`) et `slew` (variation max par seconde).

**Format de preset** (valeurs uniquement, compatible T24 : défaut pour ce qui manque) :

```json
{"fond": {"mode": "fractal", "dim": "3d", "seed": 48213,
          "globals": {"speed": 0.8},
          "macros": {"complexity": 0.6, "chaos": 0.2},
          "params": {"fractal": {"iterations": 9.5, "fold": "kifs"}},
          "routing": {"fractal.punch": {"KICK": 0.7}, "macro.complexity": {"ENERGIE": 0.5}}}}
```
- Stocker les paramètres **par mode** (`params.<mode>`) : changer de mode puis revenir conserve les réglages de chaque mode ; un preset « onglet Fond » ne touche que `fond`.
- Les clés de routing préfixées par mode (`fractal.punch`) évitent les collisions entre modes ; les macros sont des cibles de routing comme les autres.
- Versionner (`version`) et ignorer en le signalant (`report`) un paramètre inconnu, plutôt que d'échouer : un preset ancien reste chargeable quand un mode gagne des paramètres.
- Validation du schéma au démarrage dans un module sans GL ni Qt (tests pytest : défaut dans `[min,max]`, `rand` ⊂ `[min,max]`, noms uniques, uniforms présents dans le GLSL compilé — moderngl expose les uniforms actifs, attention : un uniform inutilisé est éliminé par le compilateur, d'où le « guard `_members` » déjà utilisé par le renderer).
- Construction automatique de l'UI PySide6 : un widget par `type`, regroupé par `group` (QGroupBox repliable), macros en tête, cadenas + double-clic reset (déjà la convention T7), bouton dé par groupe ; changement de `mode` = reconstruction de la zone « paramètres du mode » seulement.

### Gaps
- Aucun outil existant trouvé qui combine dans un même schéma `interp` + `rand` + `routable` ; le schéma ci-dessus est une synthèse, à valider sur un prototype.
- Le nombre d'uniforms par programme n'est pas un problème en pratique (GL 3.3 garantit au moins 1024 composantes de fragment uniform — connaissance générale, non vérifiée ici) ; à confirmer si un mode « Composé » utilise un grand tableau `u_p[]`.
