# Architecture — Graphic Art (visualiseur audio GPU)

*Mis à jour : juillet 2026*

## Vue d'ensemble

```
                    ┌─────────────────────────────────────────────┐
                    │              main.py (boucle 60 fps)         │
                    │  events pygame · routing Physique · caméra   │
                    └──────┬──────────────┬──────────────┬────────┘
        features (dict)    │              │              │  params / actions
   ┌───────────────────────┴──┐   ┌───────┴────────┐   ┌─┴──────────────────┐
   │ SOURCE AUDIO (au choix)  │   │  PARTICULES    │   │  PANNEAU (tkinter) │
   │ audio/analyzer.py+player │   │  visuals/      │   │  gfx/controls.py   │
   │   → fichier, pré-analysé │   │  particles.py  │   │  process séparé,   │
   │ audio/live.py            │   │  géométrie +   │   │  Queue → params &  │
   │   → sounddevice, temps   │   │  uniforms GPU  │   │  actions           │
   │     réel, ring buffer    │   └───────┬────────┘   └────────────────────┘
   └──────────────────────────┘           │ static VBO (1×) + uniforms (par frame)
                                  ┌───────┴────────┐
                                  │  gfx/renderer  │
                                  │  moderngl      │
                                  └────────────────┘
```

Règle de dépendance : `audio/` ne connaît pas `visuals/` ni `gfx/` ;
`main.py` fait tout le lien.

## Modules

### `main.py` — orchestration
- Fenêtre pygame + contexte OpenGL 3.3 core, boucle à 60 fps.
- Gestion des sources à chaud : `start_file()` / `start_live()` / `stop_source()`
  (actions venant de l'onglet Source). Sans source : features silencieuses,
  la fenêtre reste ouverte.
- **Routing Physique** : chaque frame, les 8 bandes EQ (+BEAT) des features
  sont multipliées par le montant choisi et accumulées par effet
  (`fx = {effet: Σ bande×montant}`), puis appliquées : rotation → particles,
  décalages RGB/grain/luminosité → post-processing, trainées → fade, etc.
  Les effets par particule (Taille, Luminosité, Turbulence) peuvent cibler
  le sujet ou le décor seul (`band<i>_scope`) → uniforms fg/bg du shader.
- **Sliders couleur scopés** : Rouge/Vert/Bleu/Contraste/Négatif/Luminosité
  ont un scope (tout/objet/décor). Scope « tout » → composite (post) ;
  sinon appliqué par particule (`renderer.set_scoped_colors`), valeur
  neutre côté post pour éviter la double application.
- **Budget de points** : le slider « Nombre de points » déclenche
  `particles.set_budget()` avec un debounce de 0,5 s.
- **Monitoring (T8)** : paquet 15 Hz vers le panneau (bandes post-enveloppe,
  rms, bpm, fps, source) via `panel.send_monitor()`.
- Caméra : zoom (slider × molette), plein écran (`F`) avec détection de
  redimensionnement par frame (l'animation macOS est asynchrone) →
  `renderer.resize()`.
- Précession de l'axe de rotation (accumulateur en °/s réels).

### `audio/analyzer.py` — analyse hors-ligne (mode fichier)
- Pré-analyse complète avec librosa : STFT, 4 bandes larges + 3 bandes fines
  (10–14 kHz), 7 bandes EQ (SUB→TREBLE), mel 64, RMS, centroid, beats/onsets.
- **Densité de transients** : onsets HF (>2 kHz) comptés sur une fenêtre
  glissante de 2 beats, lissés → pilote la vitesse de rotation.
- Interface : `get_features(frame_idx) -> dict`.

### `audio/player.py` — lecture synchronisée
- `pygame.mixer.music` ; position de lecture → index de frame d'analyse.

### `audio/live.py` — analyse temps réel (mode live)
- `sounddevice.InputStream` (blocs 512) → ring buffer 1 s thread-safe.
- Mêmes clés de features que l'analyzer, calculées par FFT à chaque frame de
  rendu. **Normalisation adaptative** (pics glissants à décroissance lente),
  gate de silence, transients par flux spectral HF + densité 1,5 s.
- Capture d'une appli : router l'appli vers BlackHole / Background Music et
  sélectionner ce périphérique (CoreAudio n'expose pas la sortie par appli).

### `visuals/particles.py` — géométrie des particules
Trois représentations, reconstruites à la demande (`rev` incrémenté →
`main` re-upload le buffer statique) :
- **Étoile** (défaut) : sphère de Fibonacci, 1 pic par bande mel.
- **Image points** : chaque pixel → particule, grille **densifiée** (≤2×)
  pour remplir le budget (1 M par défaut, `set_budget()` à chaud) ;
  détourage **BiRefNet** (repli isnet → u2net) + nettoyage morphologique
  OpenCV ; sujet recentré sur son barycentre (x,y,z) ; sujet devant (z+)
  avec **relief = luminance floutée** (heightmap propre), fond = plan plat
  derrière (z−) ; **carte de contours** (Canny + silhouette) → attribut
  `edge` (points de bord accentués).
- **Image courbes** : silhouette du masque + contours Canny auto-calibrés
  sur image bilatérale-filtrée → polylignes lissées, rééchantillonnées à
  longueur d'arc uniforme ; chaque courbe liée à une bande mel.

Deux chemins d'exécution :
- **GPU (défaut, tiquet T1)** : `build_static()` → (N,11) float32
  [pos3, color3, aux4, fg1] uploadé une fois ; `gpu_uniforms()` → dict
  d'uniforms par frame (features + angle de rotation lissé).
- **CPU (fallback)** : `update()` → buffer (N,7) recalculé en numpy
  (budget 90k).

### `gfx/renderer.py` — pipeline de rendu moderngl
Par frame :
1. **Fade** du buffer d'accumulation HDR (float16) vers le fond → trainées
2. **Grille de sol** + **axe de rotation** (optionnels, espace monde)
3. **Particules** : points additifs à falloff gaussien.
   Programme GPU (`GPU_POINT_VS`) : physique stateless par mode
   (0 = points image, 1 = courbes, 2 = étoile) — vague diagonale
   traversante + drift flownoise réduit, épaisseur z (`u_depth`), contours
   accentués (`u_edge` × attribut edge), ondulation par normale, shimmer,
   hide_bg cullé hors clip. Grading par scope : tint/contraste/négatif et
   multiplicateurs taille/luminosité/turbulence séparés sujet/décor
   (uniforms `*_fg` / `*_bg`, sélection par l'attribut `in_fg`). Les
   uniforms sont écrits avec un guard `_members` (le compilateur GLSL
   élimine les uniforms inutilisés).
4. **Bloom multi-échelle (T6)** : pyramide 3 niveaux (1/2, 1/4, 1/8), flou
   gaussien séparable par niveau, mélange piloté par « Rayon bloom »
5. **Composite** : tonemap Reinhard + post-traitement — c'est aussi ici que
   la scène **super-échantillonnée** (1–2×, slider « Anti-aliasing », T6)
   est rééchantillonnée vers la fenêtre
   (exposition, décalages UV par canal R/V/B = aberration chromatique,
   gains RGB, contraste, négatif, grain animé)
6. **HUD** : quad texturé en bande haute

Transform : `MVP = P · V(zoom) · R_axe(angle, Rodrigues) · T(orbite) ·
R_x(inclinaison objet) · S(pulse basse)`. L'axe (inclinaison/azimut[+précession])
et la distance sont fixes (panneau) ; seuls l'angle et le pulse sont musicaux.

`resize(w, h)` reconstruit accum/bloom/projection/HUD (plein écran, Retina à
venir — tiquet T6).

### `gfx/controls.py` — panneau de contrôle
- Process **séparé** (tkinter doit avoir son main thread sur macOS),
  communication par `multiprocessing.Queue` :
  `("param", clé, valeur)` et `("action", nom, payload)` ; en retour une
  queue de **monitoring** main→panel (15 Hz : bandes, rms, bpm, fps,
  source) alimente les VU-mètres de l'onglet Physique et la barre de
  statut (T8).
- **Thème ttk sombre** (T7) : valeur live + double-clic reset par slider,
  groupes LabelFrame, tooltips, géométrie de fenêtre mémorisée
  (`presets.json` clé `_window`, debounce 400 ms).
- **Barre presets** au-dessus des onglets : presets globaux nommés dans
  `presets.json` + autosave `_autosave` (écrit par le process principal à la
  fermeture, réappliqué au lancement). Source et image exclues du snapshot.
- 5 onglets scrollables : Source, Vidéo, Image, Rotation, Physique.
- `ControlPanel.poll()` draine la queue chaque frame ; `pop_actions()`
  pour les événements (chargement fichier/image, live, stop).

### Traitements transverses dans `main.py`
- **Enveloppes attack/release (T4)** : classe `Envelope` — features scalaires,
  spectre mel et bandes EQ lissés (attack 20 ms / release 180 ms ÷ slider
  « Réactivité ») ; `beat_phase`, `is_beat`, `transients` restent bruts.
- **Beat** : flash de luminosité sur `is_beat` (decay ×0.86/frame) ; bande
  routable « BEAT » = `(1 − beat_phase)²`.

## Dictionnaire des features (contrat analyzer/live → visuels)

| clé | contenu |
|-----|---------|
| `bass, low_mid, mid, high` | bandes larges normalisées 0–1 |
| `hz_10_11, hz_11_12, hz_12_14` | bandes fines (turbulence, étirement, shimmer) |
| `eq_bands` | (7,) SUB→TREBLE pour le routing Physique |
| `spectrum` | (64,) mel normalisé |
| `rms, centroid, transients` | énergie, brillance, densité de transients 0–1 |
| `is_beat, is_onset` | booléens (fichier ; approximés en live) |
| `bpm, beat_phase` | tempo + phase 0→1 entre beats (T2 : streaming en live) |

## Fichiers annexes
- `context.md` — vision, phases, état d'avancement
- `tiquet.md` — tiquets détaillés (fait : T1–T4, T6–T8, T11 ; reste :
  T5 capture vidéo, T9 MIDI, T10 morphing)
- `presets.json` — presets utilisateur + `_autosave` + `_window` (T3/T7)
