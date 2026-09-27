# Graphic Art — Visualiseur Audio Python

## Vision du projet

Script Python qui prend un fichier audio (MP3/WAV) en entrée et génère une animation visuelle temps réel synchronisée avec la musique : fréquences, oscilloscope, jeux de lumière, et animation d'une image importée par l'utilisateur.

---

## Stack technique

| Lib | Rôle |
|-----|------|
| `librosa` | Analyse audio — FFT, BPM, détection de beats, bandes spectrales |
| `numpy` | Traitement vectoriel des samples audio |
| `pygame` | Rendu temps réel 60fps, fenêtre, événements, playback audio |
| `Pillow` | Chargement et manipulation d'images |
| `opencv-python` | Effets visuels avancés sur image (phase 2) |
| `sounddevice` | Capture microphone en live (phase 3) |
| `moviepy` | Export vidéo de l'animation (phase 4) |

---

## Architecture de données

```
Fichier MP3/WAV
    └─► Décodage audio (librosa / soundfile)
            └─► Analyse spectrale (FFT, beat tracking, onset detection)
                    └─► Features par frame : [bass, mid, high, bpm, onset]
                            └─► Boucle de rendu pygame (60 fps)
                                    └─► Effets visuels (waveform, spectre, particules, image)
```

---

## Structure de fichiers cible

```
graphic-art/
├── context.md           ← ce fichier
├── main.py              ← point d'entrée, boucle principale
├── requirements.txt     ← dépendances pip
├── audio/
│   ├── analyzer.py      ← FFT, features spectrales, beat detection
│   └── player.py        ← playback synchronisé avec l'analyse
├── visuals/
│   ├── waveform.py      ← oscilloscope
│   ├── spectrum.py      ← barres fréquentielles
│   ├── particles.py     ← système de particules (phase 2)
│   └── effects.py       ← jeux de lumière, flash, halos
└── assets/              ← images importées par l'utilisateur
```

---

## Plan de développement

---

### Phase 1 — Core MVP (visualisation audio) ✅ TERMINÉE

**Objectif :** lire un fichier audio et afficher une animation visuelle synchronisée.

#### 1.1 — Mise en place de l'environnement ✅
- [x] Créer `requirements.txt` avec les dépendances phase 1
- [x] Initialiser la structure de dossiers (`audio/`, `visuals/`, `assets/`)
- [x] Tester l'import de toutes les libs

#### 1.2 — Chargement et analyse audio (`audio/analyzer.py`) ✅
- [x] Charger un fichier MP3 ou WAV avec `librosa.load()`
- [x] Calculer la FFT frame par frame (n_fft=2048, hop_length=512)
- [x] Séparer en 3 bandes fréquentielles : basses (20–250 Hz), médiums (250–4k Hz), aigus (4k–20k Hz)
- [x] Détecter les beats / onsets (`librosa.beat.beat_track`, `librosa.onset.onset_detect`)
- [x] Calculer l'énergie RMS par frame pour l'amplitude globale
- [x] Pré-analyser le fichier entier avant la lecture (tableau de features indexé par frame)
- [x] Mel spectrogram 64 bandes normalisé → utilisé pour le spectre radial
- [x] Centroid spectral normalisé → pilote la teinte des couleurs

#### 1.3 — Playback synchronisé (`audio/player.py`) ✅
- [x] Lancer la lecture audio avec `pygame.mixer.music`
- [x] Synchroniser la position via `pygame.mixer.music.get_pos()` → frame index
- [x] Exposer `get_current_features()` → retourne les features + waveform de la frame courante
- [x] `toggle_pause()` — pause/reprise propre

#### 1.4 — Fenêtre pygame (`main.py`) ✅
- [x] Initialiser une fenêtre pygame 1280×720
- [x] Boucle principale event + draw à 60 fps (`clock.tick(60)`)
- [x] SPACE pour pause/reprise, ESC pour quitter
- [x] Effet de persistance / trail (surface semi-transparente par-dessus chaque frame)
- [x] HUD minimaliste (BPM, RMS, bass en temps réel)

#### 1.5 — Oscilloscope (`visuals/waveform.py`) ✅
- [x] Waveform brute (signal temporel) sur bande horizontale en bas de l'écran
- [x] Couleur réactive : bleu froid → cyan électrique → blanc selon RMS
- [x] Épaisseur du trait variable selon les aigus
- [x] Passe de glow (trait plus épais, couleur atténuée) en arrière-plan

#### 1.6 — Spectre de fréquences (`visuals/spectrum.py`) ✅
- [x] **Spectre radial circulaire** (360°) — les barres rayonnent depuis le centre
- [x] 64 barres mel-spectrogram, longueur = énergie par bande
- [x] Teinte par position fréquentielle + décalage par centroid spectral (hue shift live)
- [x] Saturation variable selon l'amplitude de chaque bande
- [x] Peak dots blancs avec decay progressif
- [x] Glow semi-transparent sur chaque barre

#### 1.7 — Jeux de lumière (`visuals/effects.py`) ✅
- [x] Flash plein écran sur chaque beat (+ flash léger sur les onsets)
- [x] 4 anneaux concentriques pulsants sur les basses (rayon ∝ énergie basse)
- [x] Halo ambiant centré dont la teinte suit le centroid spectral (avec inertie)
- [x] Light leaks dans les 4 coins (effet aberration chromatique)
- [x] Fondu progressif du fond entre les beats (trail / persistence via `fade_surf.set_alpha()`)

---

### Phase 2 — Particules GPU (image animée + champ génératif) ✅ TERMINÉE

**Pivot technique :** le rendu passe sur GPU via `moderngl`. pygame garde la fenêtre,
le contexte OpenGL, l'audio et les events ; tout le rendu se fait en shaders.
L'affichage littéral de l'oscilloscope et du spectre radial (anciens modules pygame
`waveform.py` / `spectrum.py` / `effects.py`) est retiré du pipeline — ces fichiers
sont désormais legacy (non importés).

#### 2.0 — Pipeline de rendu GPU 3D (`gfx/renderer.py`) ✅
- [x] Contexte OpenGL 3.3 core via pygame (`OPENGL | DOUBLEBUF`) + `moderngl.create_context()`
- [x] Particules **3D** rendues en `GL_POINTS` additifs (falloff gaussien = glow par point)
- [x] **Matrice MVP** : perspective + rotation Y/X + zoom ; taille des points ∝ profondeur
- [x] Buffer d'accumulation HDR (float16) → traînées / persistance
- [x] Bloom : downsample + flou gaussien séparable (ping-pong demi-résolution)
- [x] Composite avec tone mapping Reinhard
- [x] HUD rendu en texture (quad en bande haute)
- [x] Cible de sortie configurable (`output_fbo`) → prêt pour l'export vidéo (phase 4)

#### 2.1 — Drag & drop d'image (`main.py`) ✅
- [x] Détecter `pygame.DROPFILE`, formats PNG/JPG/WEBP/BMP/GIF
- [x] Charger l'image avec Pillow, la redimensionner/centrer à la fenêtre
- [x] Invitation "drop an image" dans le HUD ; touche `U` pour revenir au mode étoile

#### 2.2 — Détourage + 3D (`visuals/particles.py`) ✅
- [x] **Détourage rembg** (U2-Net) : masque du sujet ; fallback automatique si échec
- [x] Mode **image** : l'image reste reconnaissable (particules à leur (x,y) d'origine) ;
      sujet poussé vers la caméra (z+), fond en arrière (z−), + relief par luminance,
      le tout tourne en 3D → parallaxe
- [x] Mode **étoile / oursin** (défaut, remplace l'ancien cercle) : sphère de Fibonacci,
      1 pic par bande mel, longueur du pic = gain de la bande, rotation 3D
- [x] Physique vectorisée numpy, buffer de sortie (N,7) `[x,y,z,r,g,b,size]`

#### 2.3 — Mapping fréquences → effets ✅
- [x] **bass** 20–250 Hz → taille des éléments (+ zoom pulse global)
- [x] **low-mid** 250–800 Hz → séparation de profondeur 3D (sujet/fond)
- [x] **mid** 800–4k Hz → contraste des couleurs
- [x] **high** 4k–20k Hz → vitesse de rotation
- [x] **10–11 kHz** → vitesse des particules (turbulence)
- [x] **11–12 kHz** → distance (écartement radial)
- [x] **12–14 kHz** → scintillement (shimmer)

---

### Phase 3 — Live (micro / capture d'application) ✅ TERMINÉE

**Objectif :** remplacer le fichier audio par une source temps réel.

#### 3.1 — Capture audio live ✅
- [x] `audio/live.py` — `sounddevice.InputStream` + ring buffer thread-safe
- [x] FFT 2048 par frame de rendu, normalisation adaptative (pics glissants)
- [x] Détection de transients HF en streaming (remplace le beat tracking offline)

#### 3.2 — Intégration dans l'architecture ✅
- [x] Sélection de la source dans le panneau (onglet Source), pas en CLI :
      fichier audio OU périphérique d'entrée (micro, BlackHole, Background Music)
- [x] `LiveAnalyzer` expose la même interface `get_current_features()`
- [x] Tous les modules visuels fonctionnent sans modification
- [x] Capture d'une appli : routage via BlackHole / périphérique multi-sorties
      (détail dans tiquet.md)

---

### Phase 4 — Interface de paramètres

**Objectif :** panneau de contrôle pour régler l'animation en temps réel.

#### 4.1 — Panneau de contrôle ✅ (fenêtre tkinter en process séparé)
- [x] 5 onglets scrollables : **Source / Vidéo / Image / Rotation / Physique**
- [x] Vidéo : grain, gains RGB, contraste, négatif, luminosité, bloom,
      trainées, taille points, zoom, grille du décor
- [x] Image : upload natif, suppression du fond, mode courbes, turbulence
- [x] Rotation : axe 3D paramétrable (inclinaison/azimut/distance/précession),
      affichage de l'axe, presets
- [x] Physique : routing 8 bandes EQ (+BEAT) → effets (rotation, décalages
      RGB, trainées, zoom, turbulence, luminosité, taille) + presets
      + cible Tout/Objet/Décor par bande + VU-mètres temps réel (T8)
- [x] Presets utilisateur persistants (JSON) → tiquet T3 ✅
- [x] Refonte visuelle ttk sombre, valeurs live, double-clic reset,
      fenêtre mémorisée, tooltips (T7) + barre de statut BPM/RMS/fps (T8)

#### 4.2 — Export vidéo
- [ ] Capturer chaque frame pygame comme image numpy
- [ ] Encoder en MP4 avec `moviepy.editor.ImageSequenceClip`
- [ ] Muxer la vidéo + l'audio source
- [ ] Bouton "Enregistrer" dans l'interface

---

## Commandes de lancement

```bash
# Sans argument : la fenêtre + le panneau s'ouvrent, tout se pilote
# depuis l'onglet Source (fichier ou live)
python main.py

# Lecture directe d'un fichier
python main.py fichier.mp3

# Options
python main.py fichier.mp3 --image photo.png --particles 80000
```

Raccourcis fenêtre : `SPACE` pause · `F` plein écran · molette zoom ·
`U` retour étoile · drag & drop d'image · `ESC` quitter.

---

## État actuel (juillet 2026) — au-delà du plan initial

- **Rendu image v2 (T11, 2026-07-05)** : 1 M de points (densification ≤2× de
  la grille), contours tracés (Canny + silhouette, slider « Intensité
  contours »), relief propre (heightmap de luminance floutée sur le sujet,
  décor plat), vague diagonale traversante (remplace le drift aléatoire),
  slider « Épaisseur » (plage z), slider « Nombre de points » (rebuild à
  chaud), taille des points découplée de la basse.
- **Cibles Objet/Décor** : les sliders couleur (Vidéo) et les effets par
  particule du routing Physique (Taille, Luminosité, Turbulence) peuvent
  viser le sujet détouré seul, le décor seul, ou les deux — appliqué par
  particule dans le vertex shader via le masque rembg.
- **Panneau v2 (T7/T8)** : thème ttk sombre, valeur live + double-clic reset
  par slider, groupes, tooltips, géométrie mémorisée, VU-mètres des 8 bandes
  dans l'onglet Physique (queue main→panel 15 Hz), barre de statut
  `BPM · RMS · fps · source`.

- **Physique GPU stateless (tiquet T1)** : toute la physique tourne dans le
  vertex shader ; attributs uploadés une fois, ~80 floats de features par
  frame. ≈ 520 000 particules à 1,9 ms/frame — chaque pixel de l'image est
  une particule. Turbulence par champ de bruit 3D. Fallback CPU conservé.
- **Détourage BiRefNet** (rembg) : segmentation neuronale état de l'art,
  repli isnet → u2net. Recentrage du sujet sur son barycentre.
- **Mode courbes** : contours (silhouette + Canny auto-calibré), polylignes
  lissées/rééchantillonnées, ondulation par bande mel → art abstrait au trait.
- **Rotation 3D paramétrable** : axe arbitraire (Rodrigues), orbite,
  précession de l'axe, visualisation de l'axe, grille de sol du décor.
- **Rotation musicale** : vitesse pilotée par la densité de transients HF
  (lente dans les breaks, rapide sur les drops).
- **Post-traitement** : aberration chromatique R/V/B routable par bande,
  grain, négatif, gains RGB, contraste, exposition, **bloom multi-échelle**
  (3 niveaux, slider Rayon), **super-échantillonnage 1–2×** (anti-aliasing).
- **Beat tracking temps réel (T2)** : autocorrélation + prior log-tempo dans
  `LiveAnalyzer` — BPM live au HUD, flash sur beat, bande « BEAT » routable.
- **Presets utilisateur (T3)** : `presets.json`, snapshot global des réglages,
  autosave restauré au lancement (source et image exclues).
- **Enveloppes attack/release (T4)** : features lissées (attack 20 ms,
  release 180 ms), slider « Réactivité ».
- La suite du travail est trackée dans **tiquet.md** (reste : T5 capture
  vidéo, T9 MIDI, T10 morphing de presets) ; l'architecture est décrite
  dans **architecture.md**.

---

## Contraintes et règles de développement

- Pas d'over-engineering : chaque phase doit être fonctionnelle seule avant de passer à la suivante
- `numpy` vectorisé pour toutes les opérations sur tableaux (pas de boucles Python sur les samples)
- 60 fps cible — profiler si une frame dépasse 16ms
- Pré-analyse complète du fichier avant la lecture (pas d'analyse en streaming pour la phase 1)
- Code modulaire : `audio/` ne connaît pas `visuals/`, `main.py` fait le lien
