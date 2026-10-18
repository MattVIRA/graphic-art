# Graphic Art

Un visualiseur audio temps réel en Python et OpenGL. La musique (un fichier ou le son d'une appli) fait vibrer des centaines de milliers de particules : une image, un modèle 3D, ou une simple étoile.

## Fonctionnalités

- **Trois sources visuelles** : étoile (par défaut), image (chaque pixel devient une particule, avec détourage du sujet) et modèle 3D (`.glb .gltf .obj .stl .ply`) en Points, Surface ou Wireframe.
- **Audio au choix** : fichier MP3/WAV analysé à l'avance, ou capture live (micro, BlackHole…) avec BPM et détection de beats en temps réel.
- **Physique sur GPU** : jusqu'à 1 M de particules à 60 fps, avec turbulence, vagues et relief.
- **Rendu soigné** : bloom multi-échelle, traînées, aberration chromatique, grain, anti-aliasing.
- **Panneau de contrôle** : sliders, VU-mètres, rotation 3D paramétrable et routing libre des bandes de fréquences vers les effets.
- **Presets** : tes réglages sont sauvegardés et restaurés au lancement.

## Installation

Il te faut Python 3.13 et un GPU compatible OpenGL 3.3.

```bash
git clone <url-du-repo>
cd graphic-art
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour développer : outils qualité (ruff, pytest, pre-commit) et contrôle
automatique à chaque `git commit` :

```bash
pip install -r requirements-dev.txt
pre-commit install
```

`ruff check .` vérifie le style et la gestion des exceptions (lint seulement,
pas de reformatage) ; `pytest` lance les tests (`-m "not slow"` pour sauter
les tests lents).

## Utilisation

```bash
# Lance la fenêtre et le panneau, tout se pilote depuis l'onglet Source
python main.py

# Lecture directe d'un fichier
python main.py morceau.mp3

# Avec une image ou un modèle 3D
python main.py morceau.mp3 --visual photo.png
python main.py --visual modele.glb

# Forme de test (icosphere, torus, box)
python main.py --visual primitive:torus
```

**Raccourcis** : `Espace` pause · `F` plein écran · `U` retour au tore · `Échap` quitter. Tu peux aussi glisser-déposer un modèle 3D dans la fenêtre.

**Caméra** : `⌘`/`Ctrl` + glisser (ou bouton du milieu) pour tourner autour du modèle · `Maj` + glisser pour se déplacer · molette pour zoomer · `C` ou double-clic pour revenir à la vue de départ. La case « Rotation musicale » (onglet Rotation) arrête le modèle pour l'explorer.

### Capturer le son d'une appli

macOS n'expose pas la sortie d'une appli précise. Route-la vers [BlackHole](https://github.com/ExistentialAudio/BlackHole) ou Background Music, puis choisis ce périphérique dans l'onglet Source.

## Structure

```
main.py        boucle principale, routing, caméra
audio/         analyse du fichier, lecture, analyse live
visuals/       particules, chargement des modèles 3D
gfx/           rendu moderngl, panneau tkinter, matcaps
presets.default.json   presets de départ (presets.json, local, n'est pas versionné)
tests/         tests pytest
pyproject.toml           règles ruff ; .pre-commit-config.yaml : contrôle au commit
```

`audio/` ne connaît ni `visuals/` ni `gfx/` : `main.py` fait le lien.

## Feuille de route

- Enregistrement vidéo
- Contrôle MIDI
- Morphing de presets
- Nouveaux effets : feedback, palettes, kaléidoscope, glitch
- Formes procédurales (SDF)
