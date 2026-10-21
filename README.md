# Graphic Art — V1

Un visualiseur audio temps réel en Python et OpenGL. Un modèle 3D se déforme sur
la musique, dans un paysage généré : chaque bande de fréquences, chaque kick
et chaque beat peut faire tourner, gonfler, briller ou secouer l'image.

La musique vient d'un fichier (MP3, WAV) ou d'une source audio de l'ordinateur 
en direct (micro, ahut-parleurs, etc). Un panneau de contrôle 
permet de personnaliser et paramètrer l'outil

## Installation

Prérequis : Python 3.13 et une carte graphique compatible OpenGL 3.3
(développé sur macOS).

```bash
git clone <url-du-repo>
cd graphic-art
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py
```

Deux fenêtres s'ouvrent : le rendu, et le panneau de contrôle. Il suffit
ensuite de choisir une musique dans l'onglet **Source**.

On peut aussi tout donner au lancement :

```bash
python main.py morceau.mp3                    # lire un fichier
python main.py morceau.mp3 --visual modele.glb  # avec un modèle 3D
python main.py --visual primitive:box         # forme intégrée : torus, icosphere, box
```

### Raccourcis

| Action | Geste |
|---|---|
| Pause / reprise | `Espace` |
| Plein écran | `F` |
| Revenir au tore | `U` |
| Quitter | `Échap` |
| Charger un modèle 3D | glisser-déposer le fichier dans la fenêtre |
| Tourner autour du modèle | `⌘`/`Ctrl` + glisser, ou bouton du milieu |
| Se déplacer | `Maj` + glisser |
| Zoomer | molette |
| Revenir à la vue de départ | `C` ou double-clic |

### Modèles 3D

Formats acceptés : `.glb` (le plus simple : un seul fichier, couleurs et
textures comprises), `.gltf`, `.obj`, `.stl` et `.ply`. On en trouve
beaucoup de gratuits sur internet (Sketchfab, Poly Pizza…), et on peut en
créer ou en convertir depuis [Blender](https://www.blender.org/)
(*Fichier → Exporter → glTF 2.0*).

### Capturer le son d'une appli

> **macOS**
>
> macOS ne donne pas accès au son d'une appli précise. Il faut l'envoyer
> vers [BlackHole](https://github.com/ExistentialAudio/BlackHole) (ou
> Background Music), puis choisir ce périphérique dans l'onglet **Source**.

## Fonctionnalités de la V1

**Le son**
- Fichier MP3/WAV analysé à l'avance, ou capture en direct.
- Détection du tempo (BPM), des beats, des kicks, des caisses claires et de
  l'énergie du morceau (calme dans un break, forte sur un drop).
- 7 bandes de fréquences, du sub-grave aux aigus.

**Le modèle 3D**
- Formats `.glb .gltf .obj .stl .ply`, chargés sans bloquer l'affichage.
- Trois styles : **Points**, **Surface** (éclairage studio réaliste) et
  **Wireframe** (fil de fer).
- Rotation autour d'un axe réglable, qui peut lui-même tourner ; vitesse
  pilotée par la musique (à désactiver pour explorer le modèle).

**La réaction à la musique** (onglet Physique)
- Chaque source (7 bandes, BEAT, KICK, SNARE, ÉNERGIE) se branche sur un
  effet : rotation, zoom, gonflement, turbulence, luminosité, traînées,
  décalage des couleurs, secousse de caméra…
- VU-mètres en direct pour voir ce qui bouge.
- Presets intégrés (Studio, Chromatique, Psyché) et réglage « Dynamique »
  qui calme tout pendant les passages doux.

**Le fond** (onglet Fond, masqué par défaut)
- Un paysage 3D généré : montagnes, dunes, canyons, terrasses, grottes,
  arches… L'objet voyage au fond d'une vallée, le relief peut passer devant
  lui et reçoit son ombre.
- Tout se tire au dé, groupe par groupe, avec un cadenas pour garder ce qui
  plaît ; 15 ambiances prêtes à l'emploi.
- Une texture décorative en chrome à peindre sur le relief : 8 motifs
  (volutes, coulures, gouttes, cellules, strates…).

**L'image**
- Rendu net sur écran Retina, anti-crénelage, bloom, traînées, grain,
  aberration chromatique, réglages de couleur et de contraste (onglet
  Vidéo).
- Plein écran, caméra libre à la souris.

**Le confort**
- Presets enregistrés : globaux (tous les onglets) ou par onglet ; les
  derniers réglages sont restaurés au lancement.
- Pas de plantage sur une erreur : un message s'affiche dans le panneau et
  dans la fenêtre, le détail est écrit dans `logs/graphic-art.log`.

## Prévu pour la V2

- Enregistrement vidéo d'une session
- Nouveau panneau de contrôle, plus confortable
- Explosion du modèle sur le beat
- Effet d'écho (tunnel, spirales)
- Contrôle par un contrôleur MIDI
- Transitions douces entre presets et mode automatique
- Étude d'un moteur de rendu plus moderne

## Pour développer

```bash
pip install -r requirements-dev.txt
pre-commit install
```

`ruff check .` vérifie le code, `pytest` lance les tests ; les deux tournent
à chaque `git commit`.

```
main.py        assemblage : fenêtre, boucle, actions
app/           briques de l'appli : sources, chargement, réaction à la musique…
audio/         analyse du son (fichier et direct), lecture
visuals/       chargement et préparation des modèles 3D
gfx/           rendu GPU, panneau de contrôle, presets
background/    le paysage du fond et ses shaders
tests/         tests pytest
```
