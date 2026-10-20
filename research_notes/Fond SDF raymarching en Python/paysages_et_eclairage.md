# Fond SDF : paysages raymarchés (A) et couche d'éclairage / look commune (B)

> Notes de recherche pour T16 (onglet Fond, étape 2 « Tunnel / terrain et formes SDF »).
> Contexte projet : moderngl, OpenGL 3.3 core (macOS), fragment shader plein écran,
> caméra orbitale T26, scène HDR f16 + bloom Jimenez 6 niveaux + AgX + dither déjà en
> place ([architecture.md](../../dev-instructions/architecture.md)), budget fond
> < 1 ms/frame visé par le tiquet, demi-résolution autorisée
> ([tiquet_16.md](../../dev-instructions/tiquets/tiquet_16.md)).
>
> Avertissement sur les sources : iquilezles.org a répondu de façon instable pendant la
> recherche (articles `morenoise`, `rmshadows`, `normalsSDF`, `outdoorslighting`,
> `palettes`, `fbm` lus ; `terrainmarching`, `warp`, l'index et Shadertoy en 403/404).
> Ce qui vient de ma mémoire et n'a pas pu être relu est marqué **[non vérifié]** et
> placé dans « Inferences » ou « Gaps », jamais dans « Cited Findings ».

## A1 — Heightfield ou SDF complet ? Les familles de terrains (fbm, érosion, ridged, warp, dunes, eau, canyons, grottes, piliers, synthwave, spectrogramme 3D)

### Takeaway
Pour un paysage « sol + ciel », un heightfield `y = h(x,z)` marché avec un pas proportionnel à la distance puis raffiné (interpolation linéaire / bissection) est le chemin le moins cher et le mieux documenté ; un SDF 3D complet n'est nécessaire que pour les surplombs (grottes, tunnels, arches). Les variantes visuelles (érodé, ridged, warpé, dunes, mer, piliers, synthwave, spectrogramme « Joy Division ») ne sont que des fonctions `h(x,z)` différentes branchées sur la même boucle.

### Cited Findings
**Marche de heightfield**
- Méthode de base d'iq : avancer le long du rayon par petits incréments et tester si l'échantillon est au-dessus ou au-dessous de la hauteur du terrain ; le pas règle directement vitesse et qualité (gros pas = rapide mais terrain sous-échantillonné) — [iq, Terrain raymarching](https://iquilezles.org/articles/terrainmarching) (contenu via extrait de recherche, page non relue).
- L'erreur compte moins loin de la caméra, le détail à l'écran décroissant à peu près en 1/distance : on peut lier le pas **linéairement à la distance** (`dt ∝ t`) — [iq, Terrain raymarching](https://iquilezles.org/articles/terrainmarching) (extrait).
- Raffinement : approximer le terrain linéairement entre le dernier échantillon au-dessus et le premier au-dessous, puis résoudre l'intersection analytiquement — [iq, Terrain raymarching](https://iquilezles.org/articles/terrainmarching) (extrait).
- Un pas trop grand saute des intersections, un pas trop petit gaspille du GPU — [Cambridge, Further Graphics : Ray Marching and SDF](https://www.cl.cam.ac.uk/teaching/1920/FGraphics/1.%20Ray%20Marching%20and%20Signed%20Distance%20Fields.pdf).

**Seascape (TDM, 2014) — mer en heightfield, référence pour les « plans d'eau »**
- Licence CC BY-NC-SA 3.0, original shadertoy.com/view/Ms2SD1 ; `map(p)` renvoie `p.y - h` ; une version `map` bon marché (ITER_GEOMETRY octaves) sert à la marche et `map_detailed` (ITER_FRAGMENT octaves) à l'ombrage ; original : `NUM_STEPS = 8`, ITER_GEOMETRY = 3, ITER_FRAGMENT = 5 — [Unreal forum / Ultraviolet](https://ultraviolet.indigoengine.io/shadertoy/seascape). Le portage Godot utilise 14 / 6 / 9 (valeurs différentes, probablement réglées pour la qualité) — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).
- Marche par **bissection / fausse position** entre `tm` (au-dessus) et `tx` (au-dessous) : `tmid = mix(tm, tx, hm/(hm-hx))`, puis on resserre l'intervalle selon le signe de `map(p)` ; 8 à 14 itérations seulement — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).
- Octave de vague : `uv += noise(uv); wv = 1-abs(sin(uv)); swv = abs(cos(uv)); wv = mix(wv,swv,wv); return pow(1-pow(wv.x*wv.y,0.65), choppy);`, rotation-échelle entre octaves `mat2(1.6,1.2,-1.2,1.6)` ; constantes SEA_HEIGHT 1.2 (portage), SEA_CHOPPY ~4-6, SEA_FREQ ~0.1-0.16, SEA_SPEED — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).
- Ciel : `e.y = (max(e.y,0)*0.8+0.2)*0.8; return vec3(pow(1-e.y,2), 1-e.y, 0.6+(1-e.y)*0.4)*1.1;` ; Fresnel de l'eau : `pow(clamp(1-dot(n,-eye),0,1), 3)*0.5`, mix réfracté/réfléchi — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).
- Normale à epsilon dépendant de la distance : `getNormal(p, dot(dist,dist)*EPSILON_NRM)` (epsilon ∝ distance² × taille de pixel) — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).

**fbm**
- fBm = somme d'octaves de bruit ; amplitude par octave `a = f^-H` (H de Hurst ∈ [0,1]) ; gain `G = 2^-H` ∈ [0.5, 1] ; G = 0.5 (H = 1) est la valeur codée en dur par la plupart et convient aux formes naturelles isotropes (montagnes, nuages ; profils de montagnes mesurés ≈ −9 dB/octave, cohérent avec H = 1) — [iq, fBM](https://iquilezles.org/articles/fbm/).
- Fréquence ×2 par octave, à « désaccorder » (2.01, 1.99) pour que les pics des octaves ne s'alignent pas ; en 2D, tourner légèrement le domaine à chaque octave pour éviter des motifs artificiels ; la construction octave par octave permet de **limiter la bande** pour le LOD et l'anti-aliasing ; 24 octaves suffisent à couvrir la Terre entière — [iq, fBM](https://iquilezles.org/articles/fbm/).
- Le fbm à base de sinus est rapide mais répétitif et donne de mauvais paysages — [iq, fBM](https://iquilezles.org/articles/fbm/).

**fbm « érodé » à dérivées (style Elevated)**
- Bruit de valeur à dérivées analytiques : lissage quintique `u = 6x⁵−15x⁴+10x³`, `u' = 30x²(x²−2x+1)` ; valeur + gradient en une passe, environ **5× moins cher** que les différences centrales — [iq, More noise](https://iquilezles.org/articles/morenoise/).
- Terrain : boucle de 15 octaves, amplitude b = 1 divisée par 2 à chaque octave, accumulation des dérivées `d += n.yz`, ligne clé `a += b*n.x/(1.0+dot(d,d));` → les pentes fortes amortissent les octaves suivantes : zones planes + zones rugueuses (aspect « érodé ») ; matrice d'octave 2D `mat2(0.8,-0.6,0.6,0.8)` (rotation ≈ 37°) × 2.0 ; variante 3D f = 1.98, s = 0.49 — [iq, More noise](https://iquilezles.org/articles/morenoise/).
- Le lissage cubique produit un artefact de continuité (dérivée seconde discontinue) : utiliser le quintique ; une mise en forme non dérivable (`abs()`) casse la règle de dérivation — [iq, More noise](https://iquilezles.org/articles/morenoise/).
- Ces dérivées ont servi au paysage de la démo 4k Elevated ; shaders de référence : shadertoy MdX3Rr (terrain érodé), XsXfRH (bruit à dérivées), 4ttSWf (paysage), XttSz2 (roche) — [iq, More noise](https://iquilezles.org/articles/morenoise/).

**Ridged multifractal (Musgrave)**
- Blender décrit le ridged multifractal comme créant des pics nets : valeur absolue du bruit (« canyons ») puis surface retournée ; les fonctions Musgrave (hybrid, hetero terrain) prennent H, lacunarité, octaves, **offset** (place le terrain au-dessus du « niveau de la mer ») et **gain** (poids des octaves hautes) ; un portage GLSL existe dans Blender (Cycles/Eevee) — [Blender manual, Musgrave](https://docs.blender.org/manual/sl/3.6/modeling/geometry_nodes/texture/musgrave.html), [Blender commit GLSL Musgrave](https://projects.blender.org/archive/blender-archive/commit/2f978656ab7d076706ef458ddd7674a6f653cad8) (code du ridged non lu, page en 403).

### Inferences
**Boucle heightfield recommandée (GLSL 330, esquisse)** — combine pas ∝ t et raffinement :
```glsl
// h(xz) : hauteur du terrain (fbm, ridged, dunes, spectrogramme...)
float marchTerrain(vec3 ro, vec3 rd, float tmin, float tmax, out float steps) {
    float t = tmin, lastT = t, lastD = 1.0;
    steps = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {          // 64-160 selon la qualité
        vec3 p = ro + rd * t;
        float d = p.y - terrainH(p.xz, t);         // t passé pour le LOD d'octaves
        if (d < 0.002 * t || t > tmax) break;
        lastT = t; lastD = d;
        t += max(0.45 * d, 0.004 * t);             // facteur < 1 : h n'est pas 1-Lipschitz
        steps += 1.0;
    }
    // raffinement : interpolation linéaire entre le dernier point au-dessus et l'actuel
    float d = (ro + rd*t).y - terrainH((ro + rd*t).xz, t);
    return mix(lastT, t, lastD / max(lastD - d, 1e-5));
}
```
  - `p.y - h(xz)` n'est pas une vraie distance : elle surestime la distance sur les pentes. Le facteur de relaxation 0.3-0.6 (plus petit si l'amplitude ou la pente max augmente ; borne théorique `1/sqrt(1+|∇h|max²)`) évite de traverser les crêtes. Le seuil de sortie `0.002*t` est un « cône de pixel » qui stoppe tôt au loin. **[non vérifié — pratique courante des shaders d'iq, pas relu dans l'article]**
  - Alternative Seascape (bissection) quand `h` est cher mais lisse (mer, dunes) : chercher d'abord `tx` tel que `map(ro+rd*tx) < 0` (ex. `tx = 1000`), puis 8-14 itérations de fausse position. Très bon marché mais suppose que la première racine est la seule (pas de crêtes fines devant) → bien pour l'eau et les dunes, mauvais pour les montagnes ridged.
- **Dunes** : `h = A * pow(1 - abs(sin(dot(xz, dir)*f + warp(xz))), k)` avec un warp fbm basse fréquence pour casser la régularité, + petites rides (2e sinus haute fréquence, faible amplitude) ; asymétrie vent/abri via `smoothstep` sur la phase. Profil voisin de `sea_octave` de Seascape (même famille 1−|sin| à puissance).
- **Montagnes ridged** (esquisse Musgrave, constantes typiques offset ≈ 1.0, gain ≈ 2.0, H ≈ 1.0, lacunarité ≈ 2.0 — **[non vérifié]**) :
```glsl
float ridged(vec2 p, int oct) {
    float sum = 0.0, amp = 0.5, prev = 1.0;
    for (int i = 0; i < oct; i++) {
        float n = 1.0 - abs(noise(p) * 2.0 - 1.0);   // crête en 1-|n|
        n = n * n;                                    // netteté de la crête (pow k)
        sum += n * amp * prev;                        // pondéré par l'octave précédente
        prev = clamp(n * 2.0, 0.0, 1.0);              // "gain" de Musgrave
        p = m2 * p * 2.03; amp *= 0.5;
    }
    return sum;
}
```
  Attention : `abs()` casse les dérivées analytiques (cf. iq) → normales par différences finies pour ce mode.
- **Terrain domain-warpé** : `h(xz) = fbm(xz + W * vec2(fbm(xz + o1), fbm(xz + o2)))` ; W = 0 → fbm classique, W 0.5-4 → formes coulées / « marbrées ». Coût ×3 en fbm → réserver au mode « abstrait » et réduire les octaves du warp (2-3). (Article `warp` d'iq non relu : page en 404 pendant la recherche.)
- **Canyons** : partir d'un fbm érodé et creuser `h -= D * (1 - smoothstep(0, w, abs(xz.x - path(xz.y))))` (lit sinueux suivant `path(z) = A sin(z*f)`), ou terraces `h = floor(h*n)/n + smoothstep(0,s,fract(h*n))/n` pour des strates. Le chemin sinueux sert aussi de trajectoire de caméra (vol dans le canyon).
- **Niveau de la mer** : `h' = max(h, sea)` pour la géométrie + matériau eau (Fresnel Seascape, réflexion du ciel seulement, pas de second raymarch) si `h < sea`. Animer `sea` = marée lente.
- **Grottes / tunnels** (SDF 3D, plus cher) : tunnel `d = r(z) - length(p.xy - path(p.z)) + A*fbm3(p)` (intérieur d'un tube déformé, caméra qui avance sur `path`) ; grotte « éponge » `d = (fbm3(p) - seuil) * s` avec facteur de relaxation 0.5-0.7 (le fbm 3D n'est pas Lipschitz). Ici on perd les avantages du heightfield : tous les termes d'ombrage (normales, AO, ombres) coûtent des fbm 3D.
- **Piliers / voxels** : répétition de domaine sur `xz` (`cell = floor(p.xz/s)`, hauteur par cellule = hash ou bande du spectre), SDF de boîte arrondie dans la cellule ; tester aussi la cellule voisine dans la direction du rayon ou limiter le pas à la demi-cellule pour éviter les trous. Variante très bon marché : DDA sur grille 2D (Amanatides-Woo) au lieu du sphere tracing — coût fixe par cellule traversée. (Détail des répétitions : voir la note « géométrie répétée ».)
- **Synthwave** : plan `y = h` avec `h = 0` au centre et montagnes sur les côtés (`h = A*fbm(xz) * smoothstep(w0, w1, abs(x))`), lignes de grille émissives `g = 1 - smoothstep(0, fwidth(u)*1.5, abs(fract(u)-0.5)*...)` sur `u = xz/cell + vec2(0, scroll)` (anti-aliasées par `fwidth`, sinon moiré à l'horizon), soleil = disque + bandes horizontales `step(fract(y*k), seuil)`, ciel en dégradé magenta→violet. Peut même être fait sans raymarching (intersection rayon-plan analytique + heightfield peu marché) → mode le moins cher, idéal en « 2D » du tiquet.
- **Spectrogramme 3D « Joy Division »** : texture d'historique `R16F` de taille `64 bandes × N lignes` (N = 128-256), alimentée chaque frame par `features['spectrum']` (64 mel, contrat existant) dans une ligne tournante (`head = frame % N`, `texture.write` d'une seule ligne via viewport/`write(..., viewport=...)`) ; `h(x,z) = A * texture(hist, vec2(bandeDe(x), fract(z/L + head/N))).r` avec filtrage linéaire ; défilement continu en décalant `z` d'une fraction de ligne pour éviter les sauts. Style « Unknown Pleasures » : afficher seulement des lignes (fond noir, `smoothstep` autour de `fract(z*N)`) avec occlusion des lignes arrière gratuite (c'est un heightfield opaque). Remap des bandes en log et miroir autour de x = 0 pour un relief symétrique.
- **Défilement infini** (caméra qui vole) : ne pas déplacer la caméra (l'orbite T26 doit rester autour du modèle) mais **décaler le domaine** `xz += vec2(0, u_scroll)` où `u_scroll` est intégré côté Python (`scroll += speed*dt`) ; garder `u_scroll` modulo une grande période (ex. 1000 × période du bruit) pour éviter la perte de précision float32 après de longues sessions.

### Gaps
- Code exact de l'article `terrainmarching` d'iq (pas initial, facteurs `0.01*t`, version `castRay`) : page en 404 pendant la recherche, seul un extrait de moteur de recherche a été lu.
- Code exact du ridged multifractal de Blender/Musgrave (page Blender en 403) ; constantes offset/gain citées de mémoire.
- Code d'Elevated / Rainforest (iq) et du shader « Canyon » non consultés (Shadertoy en 403) ; aucune analyse publique d'Elevated trouvée.
- Aucun shader de référence vérifié pour le « spectrogramme 3D » ou le synthwave : les esquisses ci-dessus sont des propositions de conception.

## A2 — Paramètres exposables (plages utiles, effet visuel, animables ou non)

### Takeaway
Une douzaine de paramètres suffisent (octaves, lacunarité, gain/H, amplitude, fréquence, warp, érosion, netteté des crêtes, niveau de la mer, vitesse de défilement, distance d'horizon, dégradé de ciel) ; tous sauf le nombre d'octaves s'animent en continu sans saut, à condition que l'audio module des quantités lisses (enveloppes T4) et que le défilement soit intégré en `dt`.

### Cited Findings
- G = 2^−H ∈ [0.5, 1] ; H = 1 (G = 0.5) pour des formes naturelles ; lacunarité 2 désaccordée (1.99 / 2.01) — [iq, fBM](https://iquilezles.org/articles/fbm/).
- Variante de l'érosion : division de l'amplitude d'octave par `1 + dot(d,d)` ; 15 octaves dans l'exemple ; f = 1.98, s = 0.49 — [iq, More noise](https://iquilezles.org/articles/morenoise/).
- Seascape : hauteur, « choppy », vitesse, fréquence et nombre d'octaves séparés pour la géométrie et le détail — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).
- Les enveloppes attack/release (20 ms / 180 ms) et le routing par bande existent déjà côté moteur (`app/routing.py`) — [architecture.md](../../dev-instructions/architecture.md).

### Inferences
| Paramètre (uniform) | Plage utile | Effet | Animable en continu ? |
|---|---|---|---|
| `octaves` | 3-12 (log : 4/6/9 par qualité) | détail fin | **Non** (saut de détail) ; rendre la dernière octave fractionnaire `amp *= fract(oct)` pour un fondu |
| `lacunarity` | 1.8-2.3 | 2.0 = classique ; < 2 détail plus « collé » | Oui, lentement (tout le motif glisse) |
| `gain` (ou H) | 0.35-0.65 | rugosité : 0.35 lisse, 0.65 rocheux | Oui |
| `amplitude` | 0-3 × échelle de scène | hauteur des reliefs | **Oui — meilleure cible audio** |
| `frequency` (échelle) | 0.05-1 | taille des collines | Oui mais « zoome » le paysage ; à moduler peu |
| `warp` | 0-4 | 0 = fbm, >1 formes fluides | Oui (effet psychédélique fort) |
| `erosion` (poids du 1/(1+k·dot(d,d))) | k = 0-2 | 0 = fbm brut, 1 = Elevated | Oui |
| `ridge` (exposant de 1-|n|) | 1-4 | crêtes douces → arêtes vives | Oui |
| `sea_level` | −1 à +1 (relatif) | noie les vallées | Oui (marée lente) |
| `scroll_speed` | 0-5 unités/s | vol vers l'avant | Oui, **intégré en dt** (sinon saut quand la vitesse change) |
| `horizon` (tmax) | 20-200 × échelle | distance de coupe | Non (coût) ; masquer la coupe par le fondu de distance |
| `sky_top`, `sky_horizon`, `sun_dir` | couleurs HDR 0-2 | ambiance | Oui (teinte via palette cosinus) |
| `cam_height` / pitch du terrain | — | rapport sol/ciel à l'écran | Oui |

Mapping audio proposé (cohérent avec le routing existant, valeurs lissées seulement) :
- ÉNERGIE de section → `amplitude` (breaks = plaine, drops = montagnes) et `scroll_speed` ;
- SUB/BASS (enveloppe) → amplitude d'une onde radiale `h += A*kick*exp(-r²)*sin(r*f - t*w)` partant du centre de l'écran (onde de choc au sol sur KICK, ressort T21) ;
- MID → `warp` ; TREBLE / transients → intensité émissive des crêtes ou des lignes de grille (pas la géométrie : les hautes fréquences en géométrie = scintillement) ;
- beat_phase → pulsation des lignes synthwave / du soleil ;
- spectre 64 → mode spectrogramme 3D (A1).
- Ne jamais router un signal brut sur `frequency` ou `lacunarity` : tout le paysage « glisse » et le cerveau le lit comme un bug.

### Gaps
- Pas de source mesurant la perception de ces mappings ; plages proposées issues de la pratique, à valider au banc `tools/snapshot.py`.

## A3 — Coût du raymarching de terrain en temps réel, LOD par distance, demi-résolution

### Takeaway
Le coût ≈ (nombre de pixels) × (pas de marche) × (octaves par évaluation) + l'ombrage ; les leviers les plus efficaces sont le LOD d'octaves avec la distance (fbm à bande limitée), une version « géométrie » à peu d'octaves séparée de la version « détail » pour les normales (schéma Seascape), le pas ∝ t, et le rendu en demi-résolution. Aucun chiffre mesuré sur Apple Silicon n'a été trouvé.

### Cited Findings
- Séparer une fonction de hauteur bon marché pour la marche (3 octaves dans Seascape original) d'une version détaillée pour l'ombrage (5 octaves) ; seulement 8 pas de bissection — [Ultraviolet, Seascape](https://ultraviolet.indigoengine.io/shadertoy/seascape).
- La construction par octaves permet de limiter la bande pour le LOD et l'anti-aliasing — [iq, fBM](https://iquilezles.org/articles/fbm/).
- Les dérivées analytiques donnent valeur + normale en une évaluation ; les différences centrales sont ~5× plus lentes ; jusqu'à 6× plus rapide pour des nuages volumétriques — [iq, More noise](https://iquilezles.org/articles/morenoise/).
- Epsilon de normale ≈ empreinte du pixel au point (∝ t) : sert aussi de LOD / filtrage, les falaises lointaines sont beaucoup plus stables — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/).
- Évaluer le bruit de nombreuses fois par rayon rend le terrain naïf lent, d'où l'optimisation agressive des démos — [gamedev.net (fil forum, source faible)](https://www.gamedev.net/topic/591926-marching-cubes-octree-and-lod-seams/?page=2).
- Budget du tiquet : < 1 ms/frame par type de fond en Retina, raymarching SDF autorisé en demi-résolution puis mis à l'échelle — [tiquet_16.md](../../dev-instructions/tiquets/tiquet_16.md).

### Inferences
- **LOD d'octaves** : `float oct = clamp(OCT_MAX - log2(1.0 + t * lodK), OCT_MIN, OCT_MAX);` puis boucle `for (i < OCT_MAX)` avec `if (float(i) >= oct) break;` et dernière octave pondérée par `fract(oct)` (pas de saut). Loin, 2-3 octaves suffisent puisqu'elles sont sous le pixel.
- **Ordre de grandeur (estimation, à mesurer)** : fenêtre Retina 2560×1600 → demi-rés. 1280×800 ≈ 1 M pixels. Avec ~60 pas moyens × 5 octaves = 300 bruits 2D par pixel ≈ 3·10⁸ bruits/frame, chaque bruit de valeur ≈ 4 hash + interpolation (~30-50 ALU) → ~10¹⁰ ALU/frame : à la limite de quelques ms sur un M1 de base, confortable sur M-Pro/Max. D'où : pas de MAX_STEPS > 100 en « bas », bruit de valeur par **texture de bruit** (une lecture `texture()` d'une 256² RG8 aléatoire remplace 4 hash : c'est le choix d'iq dans ses terrains Shadertoy **[non vérifié]**), et le ciel ne coûte rien (rayons qui sortent par le haut : test `rd.y > 0 && ro.y > hmax` → ciel direct).
- **Bornes** : couper le rayon par le plan `y = hmax` (au-dessus, aucun terrain possible) avant de marcher ; démarrer à `t` où le rayon passe sous `hmax` → gros gain pour les rayons rasants vers le ciel.
- **Demi-résolution** : rendre fond + profondeur dans un FBO ½ (ou ⅓ en « bas »), upscale bilinéaire ; le flou induit est bienvenu (fond subordonné, cf. B4). Si le fond écrit une profondeur pour l'occlusion du modèle, faire l'upscale *depth-aware* ou plus simple : le fond reste toujours derrière (ce que demande T16 : « profondeur écrite pour que le modèle reste devant ») → pas besoin d'upscale de profondeur, le modèle est dessiné par-dessus.
- **Réutilisation temporelle** (option) : le fond change lentement → rendu à 30 Hz alterné en damier (checkerboard) avec reprojection simple, ou rendu d'une frame sur deux quand la caméra est immobile. Complexe avec l'orbite T26 ; à garder en réserve.
- **macOS / GL 3.3** : la boucle de marche doit avoir une borne constante (`for (int i = 0; i < MAX_STEPS; i++)`) avec `break`, et MAX_STEPS en `#define` injecté à la compilation par preset de qualité (recompilation du programme moderngl au changement de qualité) plutôt qu'un uniform, ce qui laisse le compilateur dérouler/optimiser. Éviter l'inlining de 4-6 `map()` dans la normale (astuce boucle iq, cf. B1) pour limiter le temps de compilation.
- **Temps de compilation** : sur macOS le compilateur GLSL→Metal peut être lent sur de gros shaders ; compiler les variantes (mode × qualité) à la volée au premier choix, dans un message « compilation… », ou au lancement en tâche de fond. **[non vérifié — aucune mesure trouvée]**

### Gaps
- Aucun benchmark publié de terrain raymarché sur GPU Apple (M1-M4) en OpenGL 3.3 (couche de traduction GL→Metal) : chiffres ci-dessus purement estimés.
- Pas de source sur le coût réel de la couche GL→Metal d'Apple pour des boucles longues ; à mesurer avec `tools/snapshot.py` + requêtes de timer GL (`ctx.query(time=True)` de moderngl).

## B1 — Termes d'ombrage SDF : normales, ombres douces, AO, glow par pas, rim / Fresnel

### Takeaway
Normales tétraédriques (4 évaluations, epsilon ∝ t), ombre douce d'iq/Aaltonen (8-32 évaluations, k ≈ 8-16), AO à 5 échantillons le long de la normale (5 évaluations), glow par nombre de pas (gratuit) et rim/Fresnel (gratuit) forment le kit standard ; pour un fond, l'ombre douce est le seul terme vraiment cher et le premier à sacrifier.

### Cited Findings
**Normales**
- Différences centrales : 6 évaluations ; différences avant : 4 (3 si on suppose f(p) = 0, ce qui est numériquement faux) ; **tétraèdre** : 4 évaluations avec la qualité des différences centrales :
  `normalize(k.xyy*f(p+k.xyy*h) + k.yyx*f(p+k.yyx*h) + k.yxy*f(p+k.yxy*h) + k.xxx*f(p+k.xxx*h))`, `k = vec2(1,-1)` — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/).
- h idéalement ≈ empreinte du pixel au point (∝ distance t) : LOD + filtrage anti-aliasing, normales lointaines stables ; h constant trop grand perd le détail proche — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/).
- Les 4 appels inlinés peuvent dépasser les limites d'instructions ou planter le compilateur ; astuce : boucle sur un « zéro non constant » (`#define ZERO (min(iFrame,0))`) avec `e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0); n += e*map(pos+e*h).x;` — compilation beaucoup plus rapide, même coût d'exécution — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/).
- Heightfield : normale analytique gratuite si le bruit fournit ses dérivées (`n = normalize(vec3(-dh/dx, 1, -dh/dz))`) — [iq, More noise](https://iquilezles.org/articles/morenoise/).

**Ombres douces**
- Classique : `res = min(res, k*h/t)` pendant la marche vers la lumière ; k grand = ombre dure ; k ~ inverse de la taille angulaire de la lumière ; coût arithmétique négligeable devant `map()` ; un seul rayon remplace des dizaines — [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/).
- Piège : la marche discrète peut rater la pénombre la plus sombre → bandes / fuites de lumière près des coins vifs — [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/).
- Version améliorée (Aaltonen) : `y = h*h/(2.0*ph); d = sqrt(h*h-y*y); res = min(res, d/(w*max(0.0,t-y))); ph = h;` → estime le point le plus proche entre deux échantillons, corrige bandes et fuites (surtout coins vifs) — [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/).
- Version « pénombre intérieure » : on continue dans la matière, pas `clamp(h, 0.005, 0.50)` (dépend de l'échelle de scène), sortie si `res < -1` ou `t > maxt`, remap `0.25*(1+res)*(1+res)*(2-res)` → proche de la référence physique pour petites et grandes lumières — [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/).

**Rim / Fresnel**
- Seascape : `fresnel = pow(clamp(1.0 - dot(n,-eye), 0.0, 1.0), 3.0) * 0.5` pour mélanger réfraction et réflexion du ciel — [Godot Shaders, Seascape](https://godotshaders.com/shader/seascape-shader/).
- Le renderer a déjà un rim Fresnel et un Studio GGX pour le modèle — [architecture.md](../../dev-instructions/architecture.md).

### Inferences
- **AO SDF 5 échantillons** (forme d'iq dans « Raymarching - Primitives », shadertoy Xds3zN — **[non vérifié, de mémoire ; page en 403]**) :
```glsl
float calcAO(vec3 pos, vec3 nor) {
    float occ = 0.0, sca = 1.0;
    for (int i = ZERO; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i) / 4.0;   // rayons 0.01 → 0.13 (échelle scène)
        float d = map(pos + h * nor);
        occ += (h - d) * sca;                       // h-d > 0 = géométrie proche
        sca *= 0.95;
        if (occ > 0.35) break;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * nor.y);  // + terme « ciel »
}
```
  Coût 5 `map()`. Pour un terrain, `h` en unités de scène (0.05-1) car les creux sont grands ; 3 échantillons suffisent en « moyen ».
- **AO gratuite pour heightfield** : `occ ≈ clamp(0.5 + (h(p) - fbmBasseFréquence(p))*k, 0, 1)` (cavité = point plus bas que la moyenne locale) ou simplement la hauteur normalisée (vallées sombres) ; 0 évaluation supplémentaire si on garde des sommes partielles d'octaves.
- **Glow par nombre de pas** (gratuit) : `glow = steps / MAX_STEPS` ou accumuler `acc += exp(-d * k)` dans la boucle principale → halo près des surfaces frôlées, parfait pour les fractales et pour un look « néon » ; normaliser car il dépend du MAX_STEPS de la qualité (sinon le look change entre presets). Pour le terrain, plutôt accumuler un glow de crête : `acc += exp(-d*40) * ridgeMask`.
- **Ombres pour un fond** : soleil rasant sur terrain = ombres longues très belles mais 16-32 `map()` supplémentaires par pixel ; en « bas », remplacer par un terme d'horizon bon marché (`sha = smoothstep(-0.1, 0.3, dot(n, sun))` + AO-hauteur), en « moyen » 12-16 pas avec la version classique (k = 8), en « haut » 24-48 pas version Aaltonen (k = 12-16), `mint` ≈ 0.01-0.05 × échelle pour éviter l'auto-ombrage, `maxt` = distance où plus aucune géométrie n'est possible (couper au plan `hmax`).
- **Rim** pour les fonds : `rim = pow(1 - max(dot(n, -rd), 0), 3-5)` × couleur de ciel → détache les silhouettes du ciel ; gratuit. Sur un terrain, rim fort = crêtes lumineuses qui « pulsent » bien sur la caisse claire.

### Gaps
- Code exact de `calcAO` et de la boucle d'éclairage du shader Xds3zN non relus (Shadertoy en 403).
- Pas trouvé de source chiffrant le glow par nombre de pas ni de méthode « officielle » (pratique de démoscène).

## B2 — Modèle d'éclairage et couleur : 1-2 lumières vs Studio GGX, matcap, ciel, palettes cosinus, orbit trap, émissif, atténuation de distance, dither

### Takeaway
Pour le fond, un modèle « extérieur » d'iq à 3 termes (soleil ombré + ciel × AO + rebond × AO, albédo ≤ 0.2) est moins cher et plus lisible qu'un second Studio GGX complet ; la couleur vient d'une palette cosinus indexée par hauteur / orbit trap / pas, avec des parties émissives HDR pour nourrir le bloom existant ; l'atténuation de distance vers la couleur du ciel reste utile même sans « brouillard » comme type de fond.

### Cited Findings
**Recette extérieure d'iq**
- Soleil : `lin += sun*vec3(1.64,1.27,0.99)*pow(vec3(sha),vec3(1.0,1.2,1.5));` (pénombre teintée en élevant l'ombre à une puissance par canal) ; **ne pas** moduler le soleil par l'AO — [iq, Outdoors lighting](https://iquilezles.org/articles/outdoorslighting/).
- Ciel (lumière d'appoint verticale, pas une intégrale d'hémisphère) : `lin += sky*vec3(0.16,0.20,0.28)*occ;` ; l'AO est réservée aux lumières d'appoint — [iq, Outdoors lighting](https://iquilezles.org/articles/outdoorslighting/).
- Rebond : lumière horizontale opposée au soleil `normalize(sunDir*vec3(-1.0,0.0,-1.0))`, `lin += ind*vec3(0.40,0.28,0.20)*occ;` — [iq, Outdoors lighting](https://iquilezles.org/articles/outdoorslighting/).
- Matériaux : albédo ≤ 0.2 en règle générale ; `color = material * lin;` puis brouillard coloré (pas de brouillard de couleur constante), gamma en dernier — [iq, Outdoors lighting](https://iquilezles.org/articles/outdoorslighting/).
- Brouillard de base : `f = 1 - exp(-t*b)` puis mix vers la couleur de brouillard ; brouillard teinté vers le jaune quand le rayon s'aligne avec le soleil (`max(dot(rd, sunDir), 0)`) ; variante en hauteur plus délicate à mettre en œuvre — [iq, Fog](https://iquilezles.org/articles/fog/) (via résultats de recherche), [Khronos forum](https://community.khronos.org/t/question-about-fog/74465), [LYGIA fog.glsl](https://git.gay/kaneraZh/lygia/src/branch/main/lighting/raymarch/fog.hlsl).

**Palettes cosinus**
- `color(t) = a + b*cos(6.283185*(c*t + d))` ; a = luminosité, b = contraste, c = fréquence (valeurs entières ou demi-entières → cycle exact sur 0..1, entières → continuité C1), d = phase par canal — [iq, Palettes](https://iquilezles.org/articles/palettes/).
- Jeux d'exemple (a, b, c, d) : arc-en-ciel `(.5,.5,.5)(.5,.5,.5)(1,1,1)(0,.33,.67)` ; `(…)(1,1,1)(0,.10,.20)` ; `(…)(1,1,1)(.3,.2,.2)` ; `(…)(1,1,.5)(.8,.9,.3)` ; `(…)(1,.7,.4)(0,.15,.2)` ; `(…)(2,1,0)(.5,.2,.25)` ; `(.8,.5,.4)(.2,.4,.2)(2,1,1)(0,.25,.25)` — [iq, Palettes](https://iquilezles.org/articles/palettes/).

**Pipeline existant**
- Scène HDR f16, MSAA 4×, couleurs converties en linéaire en sortie de chaque shader, bloom Jimenez à seuil soft-knee 6 niveaux, composite AgX + fond (`u_bg`) + grading + vignette + grain + **dither** ; le fond est attendu dans la passe scène, alpha 0 = émissif, sans laisser de trainées sur le modèle — [architecture.md](../../dev-instructions/architecture.md), [tiquet_16.md](../../dev-instructions/tiquets/tiquet_16.md).

### Inferences
- **Studio GGX vs modèle simple** : le Studio du modèle (3 lumières + environnement, GGX) reste en espace caméra ; le réutiliser tel quel sur le fond coûte ~3× les termes spéculaires et surtout donne au fond **le même éclairage que le modèle**, ce qui les fait fusionner visuellement. Recommandation : bloc GLSL partagé `bg_light(n, rd, p, sha, occ)` = recette extérieure d'iq (soleil + ciel + rebond, diffus Lambert, spéculaire Blinn ou GGX unique optionnel sur l'eau) avec une **direction de soleil propre au fond** (en espace monde, pour que l'orbite montre l'éclairage tourner = parallaxe), éventuellement alignée sur la lumière clé du Studio en option « cohérence ».
- **Matcap sur SDF** : `uv = (n_vue.xy * 0.5 + 0.5)` avec `n_vue = mat3(u_view) * n` → réutilise `gfx/matcaps.py` à coût nul (une lecture de texture), sans ombres ; bon pour les modes fractals/organiques « argile » et pour le niveau « bas ». Peu adapté au terrain (aspect plastique).
- **Ciel / dégradé** : `sky = mix(horizonCol, zenithCol, pow(max(rd.y,0), 0.4..0.6))` + halo solaire `sunCol*pow(max(dot(rd,sun),0), 8..64)` + disque `pow(…, 512+)` ; sous l'horizon, couleur de sol sombre. La couleur de l'horizon doit **être** la couleur d'atténuation de distance pour fondre la coupe `tmax`. Le dégradé est en espace monde → bouge avec la caméra (mode 3D) ; en mode 2D, figer `rd` à la caméra fixe.
- **Coloration par position** : terrain → palette cosinus indexée par `h` normalisée + pente (`n.y`) (roche sur les pentes `n.y < 0.7`, « neige »/émissif sur les crêtes) ; fractales → **orbit trap** (min de `|z|`, distance à un plan ou un point pendant l'itération) indexant la palette ; métaballs → indice du blob dominant ; répétitions → hash de la cellule. Tout ça est gratuit (déjà calculé dans `map`), à condition que `map` puisse retourner un `vec2(distance, matériau)` seulement lors de l'évaluation finale (version `mapMat` séparée, pour ne pas alourdir les évaluations de marche).
- **Émissif pour le bloom** : lignes de grille, crêtes, piliers « actifs », fissures → couleur HDR 2-8 (au-dessus du seuil soft-knee du bloom) ; le reste du fond albédo ≤ 0.2 sous le seuil → seul l'émissif fleurit, contrôle précis du bloom. Les émissifs sont la meilleure cible des transients/SNARE (flash) car ils ne changent pas la géométrie.
- **Atténuation de distance (optionnelle, pas un « type » de fond)** : `col = mix(col, horizonCol, 1 - exp(-t * density))`, density réglable 0 (off) à ~2/tmax, éventuellement teintée soleil (iq). Rôle : masquer la coupe `tmax`, donner la profondeur atmosphérique et **pousser le fond vers l'arrière** (B4). Coût nul.
- **Dither** : un fond en dégradés sombres lisses (ciel, brouillard) bande en 8 bits ; le composite a déjà un dither après AgX — vérifier qu'il s'applique aussi au fond (il est dans la même chaîne) ; si le fond est rendu dans un FBO RGBA8 demi-résolution, passer en `f2`/`f4` (f16) pour éviter de quantifier avant le tonemap. Ajouter un jitter de départ de rayon `t0 += hash(fragCoord + frame) * pas` supprime les bandes de pas (« wood grain ») du glow et de l'ombre au prix d'un léger bruit que le grain existant masque.

### Gaps
- Article `fog` d'iq non relu en entier (formule exacte du brouillard en hauteur non confirmée).
- Pas de source spécifique sur le matcap appliqué aux SDF ou sur l'orbit trap (la note « fractales » d'un autre chercheur couvre probablement l'orbit trap).

## B3 — Coût de chaque terme (évaluations SDF supplémentaires) et échelle de qualité bas / moyen / haut

### Takeaway
En évaluations de `map()` par pixel touché : marche 30-160, normale 4 (0 si dérivées analytiques), AO 3-5, ombre douce 12-48, glow/rim/palette/brouillard 0 ; une échelle de qualité qui fait varier la résolution, le MAX_STEPS, les octaves et la présence des ombres couvre un facteur ~10 de coût.

### Cited Findings
- Normales : 6 (centrales), 4 (avant ou tétraèdre) évaluations — [iq, Normals for an SDF](https://iquilezles.org/articles/normalsSDF/).
- Ombre douce : un rayon de marche complet vers la lumière (boucles jusqu'à 256 pas dans les exemples), arithmétique supplémentaire négligeable — [iq, Soft shadows](https://iquilezles.org/articles/rmshadows/).
- Dérivées analytiques ≈ 5× moins chères que les différences centrales — [iq, More noise](https://iquilezles.org/articles/morenoise/).
- Géométrie à peu d'octaves pour la marche, plus d'octaves pour l'ombrage (Seascape 3 → 5) — [Ultraviolet, Seascape](https://ultraviolet.indigoengine.io/shadertoy/seascape).

### Inferences
| Terme | Évaluations `map()` / pixel | Remarque |
|---|---|---|
| Marche primaire | 30-160 (moy. réelle souvent 40-60 % du max) | dominant ; ciel = ~0 si rejeté par le plan `hmax` |
| Normale | 4 (tétraèdre) ; 0 (heightfield à dérivées) | utiliser la version détaillée (plus d'octaves) |
| AO | 3-5 | ou 0 (AO par hauteur sur terrain) |
| Ombre douce | 12-48 | le plus cher après la marche ; premier sacrifié |
| Glow par pas, rim, Fresnel, palette, ciel, atténuation, dither | 0 | « look » quasi gratuit |
| Réflexion (eau) | 0 si ciel seul ; +marche complète si vraie réflexion | ne jamais faire de 2e marche pour un fond |

Échelle de qualité proposée (uniform/define `u_bg_quality`, sauvée dans les presets) :

| | Bas | Moyen (défaut) | Haut |
|---|---|---|---|
| Résolution du fond | ⅓-½ | ½ | ½ (¾ sur M-Pro/Max) |
| MAX_STEPS | 48 | 96 | 160 |
| Octaves marche / détail | 3 / 5 | 4 / 7 | 5 / 9 |
| LOD octaves par distance | oui, agressif | oui | oui |
| Normale | dérivées analytiques ou tétraèdre | tétraèdre | tétraèdre, ε ∝ t |
| AO | hauteur (0 éval.) | 3 échantillons | 5 échantillons |
| Ombres | terme d'horizon (0 éval.) | classique, 16 pas, k = 8 | Aaltonen, 32-48 pas, k = 12-16 |
| Jitter de départ + dither | oui | oui | oui |
| Coût visé (estimation) | ≤ 0,5 ms | ~1 ms | 2-3 ms |

- Brancher un **auto-ajustement** facultatif : si le `FrameClock` mesure > 16 ms sur 1 s, descendre d'un cran (et l'afficher dans la barre de statut), comme le debounce du budget de points existant.
- Mesurer avec des requêtes de temps GPU moderngl (`ctx.query(time=True)`) autour de la passe de fond, pas avec le temps CPU.

### Gaps
- Aucun chiffre mesuré : la colonne « coût visé » est un objectif, pas une mesure. À établir par un banc sur la machine de l'utilisateur.

## B4 — Garder le fond subordonné au modèle (contraste, saturation, flou, assombrissement)

### Takeaway
Le fond doit perdre sur tous les axes de saillance face au modèle : plus sombre (albédo ≤ 0.2, exposition propre), moins saturé, moins contrasté, plus flou (demi-résolution + flou), plus atténué avec la distance, et réactif par des émissifs lents plutôt que par des flashs plein écran ; le tout réglable par un unique slider « Intensité fond » déjà prévu par T16.

### Cited Findings
- Albédo des matériaux ≤ 0.2 et AO réservée aux lumières d'appoint (fond naturellement sombre et doux) — [iq, Outdoors lighting](https://iquilezles.org/articles/outdoorslighting/).
- Le tiquet prévoit sliders « Intensité fond », « Vitesse », réactivité par défaut calme (luminosité ∝ énergie de section lissée), et alerte : fond clair + points / wireframe additifs = rendu délavé — [tiquet_16.md](../../dev-instructions/tiquets/tiquet_16.md).
- Le modèle est rendu en Studio PBR avec rim fresnel, dans une scène HDR suivie d'un bloom à seuil et d'AgX — [architecture.md](../../dev-instructions/architecture.md).

### Inferences
- **Hiérarchie de luminance** : multiplier la radiance du fond par `u_bg_intensity` (0-1, défaut ~0.35) **avant** le tonemap ; viser un fond dont la luminance moyenne reste 3-5× sous celle du modèle. Les émissifs du fond plafonnés juste au-dessus du seuil de bloom (pas de halo qui mange le modèle).
- **Désaturation / contraste** : `col = mix(vec3(luma(col)), col, u_bg_sat)` (défaut 0.5-0.7) et contraste réduit autour du gris moyen ; décaler la teinte du fond à l'opposé (ou en analogue sombre) de la teinte dominante du modèle, p. ex. phase `d` de la palette cosinus pilotée en opposition.
- **Profondeur de champ simulée** : rendre en ½ (ou ⅓) résolution puis upscale bilinéaire = flou gratuit ; aller plus loin avec un flou gaussien 2 passes (réutiliser les shaders de flou du bloom à un niveau de mip) contrôlé par `u_bg_blur` ; un vrai DoF par profondeur n'est pas nécessaire puisque le fond est toujours derrière.
- **Vignette / masque autour du modèle** : assombrir le fond dans un disque doux centré sur la projection écran du modèle (centre et rayon fournis par Python depuis la sphère englobante fixe de `mesh.py`) → « halo sombre » qui détache la silhouette ; inverse possible (halo clair) pour les modèles sombres.
- **Atténuation de distance** (B2) toujours un peu active (density > 0) : le fond s'éloigne visuellement.
- **Mouvement** : le fond bouge plus lentement que le modèle (scroll en enveloppes longues, release ≥ 500 ms) ; réserver les impulsions (KICK, SNARE) aux émissifs et à l'onde de choc au sol, avec amplitude limitée ; pas de flash plein écran issu du fond (le flash de beat existe déjà côté composite).
- **Lisibilité des styles additifs** : pour Points / Wireframe (blend additif), forcer un plafond de luminance du fond plus bas (ou multiplier par `1 - alpha_modèle` via la couverture), sinon délavage (point d'attention du tiquet).
- **Contrôle unique** : un preset « Discret » (intensité 0.25, saturation 0.5, flou fort, atténuation forte) et un preset « Scène » (fond protagoniste quand aucun modèle n'est chargé ou pour les breaks) ; « Dynamique » (existant) peut aussi atténuer le fond dans les sections calmes.

### Gaps
- Aucune source académique ou de production consultée sur la hiérarchie visuelle avant-plan / fond en temps réel (règles ci-dessus = pratique de composition, à valider visuellement au banc `tools/snapshot.py --orbit`).
