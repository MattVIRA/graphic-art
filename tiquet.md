# Tiquet — Source audio au choix : fichier MP3 ou capture live d'une application

## Contexte

Jusqu'ici le visualiseur se lance en ligne de commande avec un fichier audio
obligatoire (`python main.py fichier.mp3`) et pré-analyse tout le fichier.
On veut pouvoir visualiser **le son qui sort d'une application ouverte**
(DAW, Spotify, navigateur…) en temps réel.

## Analyse de faisabilité (macOS)

- ✅ **Capture live d'un périphérique d'entrée** : possible avec `sounddevice`
  (PortAudio) — micro, ou n'importe quel périphérique virtuel.
- ⚠️ **Capture directe de la sortie d'une appli précise** : CoreAudio ne
  l'expose pas aux processus tiers. La voie standard est un **périphérique de
  bouclage virtuel** vers lequel on route l'application :
  - **Background Music** (déjà installé sur cette machine) — routage par appli,
    le périphérique « Background Music » apparaît comme entrée capturable.
  - **BlackHole** (gratuit) — sortie système → entrée virtuelle.
  - (Alternative native : ScreenCaptureKit par appli, macOS 13+, mais demande
    des bindings pyobjc lourds — hors scope de ce tiquet.)
- ➜ **Décision** : l'onglet Source liste les périphériques d'entrée ; pour
  capter une appli, l'utilisateur la route vers Background Music / BlackHole
  et sélectionne ce périphérique. L'analyse est identique quel que soit le
  périphérique choisi.

## Spécification

1. `python main.py` se lance **sans argument** : la fenêtre s'ouvre (mode
   étoile silencieux) avec le panneau de contrôles ; l'argument fichier
   devient optionnel (compat descendante conservée).
2. Nouvel onglet **Source** (premier onglet du panneau) :
   - bouton « Choisir un fichier audio… » → dialogue natif, lecture + analyse
     pré-calculée comme avant ;
   - liste déroulante des **périphériques d'entrée** détectés (micro,
     Background Music, BlackHole…) + bouton « Démarrer l'analyse live » ;
   - bouton « Stop » pour couper la source courante.
3. Dès qu'une source live est sélectionnée, la visualisation tourne **même en
   silence** (features à zéro, l'image reste affichée) et réagit dès que du
   son arrive.
4. L'analyse live reproduit l'interface `get_current_features()` du mode
   fichier : mêmes clés (bass, mid, eq_bands, spectrum, transients…), donc
   **aucun changement** dans les visuels ni le routing Physique.

## Implémentation

- [x] `audio/live.py` — `LiveAnalyzer` : `sounddevice.InputStream` (callback →
      ring buffer thread-safe), FFT 2048 par frame de rendu, bandes / mel /
      centroid / RMS calculés en direct, **normalisation adaptative** (pics
      glissants avec décroissance, pas de pré-analyse possible), détection de
      transients HF par flux spectral + densité glissante 1,5 s.
- [x] `gfx/controls.py` — onglet Source (fichier, périphériques, live, stop).
- [x] `main.py` — argument audio optionnel, boucle sans source (silence),
      bascule fichier ↔ live à chaud, HUD adapté (BPM `--` en live).
- [x] `requirements.txt` — ajout de `sounddevice`.

## Limitations connues

- La sélection se fait par **périphérique**, pas par appli : pour isoler une
  appli il faut la router (Background Music le fait par appli nativement).
- Pas de BPM affiché en live (le beat-tracking hors-ligne de librosa n'est pas
  applicable en streaming) — la densité de transients le remplace pour la
  physique de rotation.
- Latence ≈ taille de bloc (512 échantillons ≈ 12 ms) + fenêtre FFT.

---
---

# Backlog court terme — amélioration de l'application (usage personnel)

Priorisé par ratio impact / effort. Chaque tiquet est autonome.

---

## T1 — Physique des particules en compute shader (GPU) ✅ TERMINÉ

**Priorité : haute · Effort : 2-3 sessions · Impact : majeur**

> **Réalisé (2026-07-04)** — implémenté en **vertex shader stateless** (option
> recommandée, macOS plafonnant OpenGL à 4.1). Attributs statiques (11 floats/
> particule) uploadés une fois ; chaque frame le CPU n'envoie que ~80 floats de
> features. Mesuré : **518 400 particules à 1,9 ms/frame** (~527 fps de marge) —
> le mode image utilise désormais chaque pixel, sans sous-échantillonnage.
> Turbulence = champ de bruit 3D dérivant (flownoise). Les 3 styles portés
> (points, courbes, étoile). `hide_bg` cullé dans le shader. Fallback CPU
> conservé (budget 90k) si le shader ne compile pas.

### Contexte
La physique (ondulations, turbulence, jitter) tourne en numpy sur CPU, et le
buffer complet (N,7) est re-uploadé vers le GPU à chaque frame. C'est le
plafond actuel : ~90 000 particules max, et le CPU sature avant le GPU.

### Spécification
- Positions de repos, couleurs, normales, bandes → buffers GPU **persistants**
  (SSBO), uploadés une seule fois au chargement de l'image.
- Un **compute shader** (moderngl `compute_shader`, GLSL 430) calcule chaque
  frame les positions déplacées à partir d'un petit uniform buffer de features
  audio (bass, spectrum 64, transients… ≈ 80 floats).
- Le vertex shader lit directement le SSBO → **zéro transfert CPU↔GPU** par
  frame en dehors des features.
- Portage des trois styles : points image, courbes, étoile.
- Turbulence améliorée au passage : curl noise 3D (simplex) au lieu du jitter
  aléatoire — mouvement organique type "fumée".

### Critères d'acceptation
- [ ] 1 000 000 particules à 60 fps en mode image (Apple Silicon).
- [ ] Comportement visuel identique à paramètres égaux (points/courbes/étoile).
- [ ] Le slider Turbulence pilote le curl noise.
- [ ] Fallback CPU conservé si le contexte ne supporte pas les compute shaders.

### Points d'attention
- macOS plafonne OpenGL à 4.1 → pas de compute en GL. Solutions : passer le
  calcul en **transform feedback** (GL 4.1) OU faire le déplacement
  entièrement dans le **vertex shader** (stateless, déplacements = f(pos_repos,
  features, temps)) — recommandé : vertex shader stateless, plus simple et
  suffisant pour nos effets.

---

## T2 — Beat tracking temps réel (BPM en live) ✅ TERMINÉ

**Priorité : haute · Effort : 1 session · Impact : fort**

> **Réalisé (2026-07-04)** — aubio ne compile pas sur Python 3.13 →
> implémentation maison dans `LiveAnalyzer` : enveloppe de flux d'énergie sur
> la grille des blocs (86 Hz), tempo par **autocorrélation + prior log-tempo
> 120 BPM** (Ellis) ré-estimé toutes les 2 s avec raffinement parabolique,
> phase par alignement de peigne. Testé : 128 BPM synthétique → 128,4 estimé.
> `bpm`, `beat_phase`, `is_beat` exposés dans les deux modes (fichier :
> interpolation des beats librosa). HUD BPM live, flash de luminosité sur
> chaque beat, et 8ᵉ bande **BEAT** (impulsion décroissante) routable dans
> l'onglet Physique.

### Contexte
En mode live le BPM affiche 0 et rien n'est calé sur le beat : librosa ne
fait du beat tracking qu'en hors-ligne. La densité de transients pilote la
rotation mais il n'y a ni pulsation ni flash synchronisés.

### Spécification
- Intégrer **aubio** (`pip install aubio`, léger, temps réel natif) dans
  `LiveAnalyzer` : un objet `aubio.tempo` alimenté par les blocs de 512
  échantillons du callback.
- Exposer dans les features : `bpm` (estimation glissante), `is_beat`
  (True sur la frame du beat), `beat_phase` (0→1 entre deux beats).
- Utilisations immédiates :
  - HUD : BPM live affiché ;
  - flash léger sur beat (réutiliser la logique fichier) ;
  - `beat_phase` disponible comme 8e « bande » dans l'onglet Physique
    (ex. zoom qui respire exactement sur le beat).

### Critères d'acceptation
- [ ] BPM stable (±2) sur un morceau au tempo constant via BlackHole.
- [ ] `is_beat` déclenche un flash visible caléé sur le kick.
- [ ] Mode fichier inchangé (librosa reste la source hors-ligne).

---

## T3 — Presets utilisateur sauvegardés (JSON) ✅ TERMINÉ

**Priorité : haute · Effort : ½ session · Impact : confort quotidien**

> **Réalisé (2026-07-04)** — `presets.json` à la racine ; barre presets en
> haut du panneau (combobox + Charger / Sauver… / ✕). Un preset = snapshot
> **global** de tous les onglets (37 paramètres) ; la **source et l'image
> sont exclues** (choix utilisateur). Autosave `_autosave` écrit à la
> fermeture par le process principal, restauré au lancement (panneau et
> moteur synchronisés). Les presets par onglet (Rotation/Physique) restent
> comme raccourcis de composition.

### Contexte
Tous les réglages (12+ sliders, routing Physique, rotation) sont perdus à
chaque fermeture. Les presets actuels sont codés en dur.

### Spécification
- `presets.json` à la racine : `{"nom": {param: valeur, ...}, ...}` — un
  preset = snapshot complet du dict params.
- Dans le panneau, en haut (hors onglets) : combobox des presets + boutons
  **Sauver** (demande un nom), **Supprimer**, **Charger**.
- Sauvegarde automatique de l'état courant en quittant → restauré au
  lancement (`_autosave` comme preset réservé).
- Les presets codés en dur (rotation, physique) deviennent des entrées
  par défaut du fichier, modifiables.

### Critères d'acceptation
- [ ] Fermer/relancer l'app restaure tous les réglages.
- [ ] Créer, recharger et supprimer un preset nommé fonctionne.
- [ ] Fichier JSON lisible/éditable à la main.

---

## T4 — Lissage attack/release des features (qualité du mouvement) ✅ TERMINÉ

**Priorité : moyenne · Effort : ½ session · Impact : rendu nettement plus pro**

> **Réalisé (2026-07-04)** — classe `Envelope` dans main.py (scalaires et
> tableaux numpy, coefficients exacts en dt réel). Attack 20 ms (57 % en une
> frame), release 180 ms (retombée ~320 ms mesurée). Lissés : 8 features
> scalaires + spectre mel 64 + bandes EQ du routing. Non lissés (voulu) :
> beat_phase/is_beat, transients. Slider « Réactivité » (0.25–3) dans
> l'onglet Vidéo : scale le release de ~1 s (planant) à ~117 ms (percussif).

### Contexte
Les features brutes par frame font « vibrer » les effets (la taille des
points saute à chaque frame de 16 ms). Les visualiseurs qui semblent
« organiques » lissent tout avec des enveloppes asymétriques.

### Spécification
- Classe `Envelope(attack, release)` : montée rapide, descente lente
  (EMA à deux coefficients), appliquée par feature **côté main.py** avant le
  routing Physique.
- Réglages par défaut : attack 20 ms (réactif au kick), release 180 ms
  (retombée douce). Un slider global « Réactivité » dans l'onglet Vidéo
  scale les deux.
- Appliquer aussi au `fx` routé (les 7 bandes de l'onglet Physique).

### Critères d'acceptation
- [ ] Plus de scintillement de taille/luminosité sur bruit de fond.
- [ ] Un kick isolé produit une impulsion franche qui retombe en ~0,2 s.

---

## T5 — Enregistrement vidéo (capture de session)

**Priorité : moyenne · Effort : 1 session · Impact : partage/archive**

### Contexte
Aucun moyen de garder une trace d'un rendu réussi (phase 4 du context.md,
jamais faite). Usage personnel : capturer 30 s pour l'envoyer à un ami.

### Spécification
- Touche **R** (et bouton dans l'onglet Vidéo) : démarre/arrête la capture.
- Le renderer rend déjà dans un FBO → `glReadPixels` du composite final
  (éviter l'écran pour rester correct en plein écran), push dans une queue,
  **thread d'encodage** ffmpeg (`imageio-ffmpeg` ou subprocess ffmpeg,
  H.264, CRF 18).
- Mode fichier : muxer l'audio source dans le MP4 (offset = frame de départ).
  Mode live : vidéo muette (ou capture parallèle du device sounddevice si
  simple).
- Indicateur ● REC dans le HUD. Sortie : `captures/YYYY-MM-DD_HHMMSS.mp4`.

### Critères d'acceptation
- [ ] 60 fps de capture sans faire chuter le rendu sous 50 fps (1280×720).
- [ ] MP4 lisible avec audio synchronisé en mode fichier.

---

## T6 — Rendu Retina + bloom multi-échelle ✅ TERMINÉ (adapté)

**Priorité : basse · Effort : 1 session · Impact : finition visuelle**

> **Réalisé (2026-07-04)** — le Retina natif est **impossible avec pygame 2.6**
> (le flag SDL `ALLOW_HIGHDPI` n'est pas exposé pour les fenêtres OpenGL ;
> vérifié : drawable == taille fenêtre). Remplacé par du
> **super-échantillonnage interne** : la scène est rendue à 1×–2× la taille
> de la fenêtre (slider « Anti-aliasing (×) », défaut 1.5) puis rééchantillonnée
> au composite → supprime l'aliasing des courbes et points fins.
> **Bloom multi-échelle** : pyramide 3 niveaux (1/2, 1/4, 1/8) avec flou
> gaussien séparable par niveau, mélangés par le slider « Rayon bloom »
> (0 = glow serré, 1 = larges halos doux). Mesuré : 500 000 particules à
> 1,5× de super-échantillonnage = 4,3 ms/frame (60 fps large).

### Contexte
La fenêtre est rendue en résolution logique (non-HiDPI) → flou léger sur
écran Retina. Le bloom actuel est mono-échelle (un seul rayon de flou).

### Spécification
- Activer le mode HiDPI de SDL (`SDL_WINDOW_ALLOW_HIGHDPI` via pygame) et
  utiliser la taille du *drawable* (≠ taille fenêtre) pour tous les FBO —
  le `resize()` existant fait déjà le gros du travail.
- Bloom multi-échelle : 3 niveaux de downsample (1/2, 1/4, 1/8) blurés puis
  sommés avec poids décroissants → halos larges et doux au lieu du glow
  uniforme actuel.
- Slider « Rayon bloom » dans l'onglet Vidéo (poids relatif des niveaux).

### Critères d'acceptation
- [ ] Points nets au pixel près sur écran Retina.
- [ ] Un point brillant isolé produit un halo doux et étendu, sans bandes.
- [ ] 60 fps conservés en 2560×1600 drawable.

---

## Ordre suggéré (mis à jour 2026-07-05)

~~T1 T2 T3 T4 T6~~ ✅ faits. Reste dans ce backlog : `T5 (capture vidéo)`.

---
---

# Backlog interface de contrôle (T7 → T10)

Objectif : rendre le panneau agréable, lisible et informatif — aujourd'hui il
est fonctionnel mais brut (tkinter par défaut, aucun retour visuel sur ce que
« entend » le moteur).

---

## T7 — Refonte visuelle du panneau ✅ TERMINÉ

**Priorité : haute · Effort : 1 session · Impact : confort permanent**

> **Réalisé (2026-07-04)** — thème clam sombre complet (#15151c, accents
> cyan/violet) appliqué à tous les widgets ttk ; `tk.Scale` → `ttk.Scale`
> avec **label de valeur en direct** (décimales adaptées au pas) ;
> **double-clic = retour au défaut** ; groupes `ttk.LabelFrame` (Couleur,
> Trainées & bloom, Rendu, presets…) ; fenêtre redimensionnable (min
> 340×480), géométrie envoyée au main (debounce 400 ms) et sauvée dans
> `presets.json` clé `_window`, restaurée au lancement ; tooltips sur
> Réactivité, Rayon bloom, Anti-aliasing, Trainées et le routing Physique.
> Protocole queue params/actions inchangé.

### Contexte
Le panneau est du tkinter brut : sliders gris hétérogènes, pas de valeurs
lisibles, sections tassées, largeur fixe 380 px, look incohérent avec
l'esthétique de l'app.

### Spécification
- Passer tous les widgets sur **ttk avec un thème sombre cohérent**
  (style "clam" personnalisé : fond #15151c, accents cyan/violet assortis au
  rendu) ; supprimer les `tk.Scale` au profit de `ttk.Scale` + **label de
  valeur en direct** à droite de chaque slider (les tk.Scale affichent la
  valeur au-dessus, illisible).
- **Double-clic sur un slider = retour à sa valeur par défaut** (réglage fin
  au quotidien).
- Groupes visuels avec `ttk.LabelFrame` (« Couleur », « Rendu », « Trainées &
  bloom »…) au lieu d'une liste plate.
- Fenêtre redimensionnable proprement (min 340 px), position mémorisée dans
  `presets.json` (clé réservée `_window`).
- Tooltips courts sur les contrôles non évidents (Réactivité, Rayon bloom,
  routing Physique).

### Critères d'acceptation
- [ ] Chaque slider affiche sa valeur numérique en permanence.
- [ ] Double-clic remet la valeur par défaut.
- [ ] Le panneau rouvre à sa dernière position/taille.
- [ ] Aucun changement de protocole (queue params/actions inchangée).

---

## T8 — Monitoring audio dans le panneau (VU-mètres) ✅ TERMINÉ

**Priorité : haute · Effort : 1-2 sessions · Impact : le routing devient intuitif**

> **Réalisé (2026-07-04)** — queue main→panel (`mp.Queue(maxsize=4)`,
> `put_nowait` avec drop) ; le main envoie 15×/s les 8 bandes
> post-enveloppe (incl. impulsion BEAT), rms, bpm, fps et le label de
> source. Onglet Physique : **VU-mètre canvas vertical** à gauche de chaque
> bande (redessin ~15 Hz, décroissance douce entre paquets). Barre de
> statut en bas : `BPM 128 · RMS ▓▓▓░░░ · 60 fps · LIVE`.

### Contexte
Pour régler l'onglet Physique il faut deviner ce que contient chaque bande.
Aucun retour visuel : on route « HIGH MID → rotation » à l'aveugle.

### Spécification
- Le process principal envoie ~15×/s un petit paquet de monitoring au panneau
  (nouvelle queue main→panel, déjà nécessaire pour T7 position fenêtre) :
  niveaux des 8 bandes (post-enveloppe), rms, bpm, fps.
- Dans l'onglet Physique : **VU-mètre vertical à côté de chaque bande**
  (canvas tkinter, barre qui suit le niveau) → on voit immédiatement quelle
  bande s'active sur quel élément du morceau.
- Barre de statut en bas du panneau : `BPM 128 · RMS ▓▓▓░ · 60 fps · LIVE`.
- Léger : canvas redessiné à 15 Hz max, aucune image, pas de matplotlib.

### Critères d'acceptation
- [ ] Les 8 VU-mètres bougent en musique dans l'onglet Physique.
- [ ] FPS du rendu inchangé (60) pendant le monitoring.
- [ ] La barre de statut affiche BPM/RMS/fps/source en continu.

---

## T9 — Contrôle MIDI (mapper la FLX4 ou tout contrôleur)

**Priorité : moyenne · Effort : 1-2 sessions · Impact : contrôle physique en live**

### Contexte
Une FLX4 est branchée : ses potards/faders sont des contrôleurs MIDI
standard. Tourner un vrai potard pour le zoom ou le rayon bloom est bien plus
agréable qu'une souris — surtout pendant qu'on écoute.

### Spécification
- `pip install mido python-rtmidi` ; écoute des messages CC dans un thread du
  process principal (pas de dépendance au panneau).
- **MIDI learn** : bouton « MIDI » à côté de chaque slider (T7) → le prochain
  CC reçu est mappé sur ce paramètre ; mapping sauvé dans `presets.json`
  (clé `_midi`).
- Les valeurs MIDI (0-127) sont normalisées vers la plage du slider ; le
  panneau reflète la valeur reçue (via la queue main→panel de T8).
- Périphérique MIDI choisi dans l'onglet Source (liste + rafraîchir, comme
  l'audio).

### Critères d'acceptation
- [ ] Un potard de la FLX4 mappé sur « Zoom » zoome en direct.
- [ ] Les mappings survivent au redémarrage.
- [ ] Débrancher le contrôleur ne fait pas planter l'app.

---

## T10 — Morphing de presets & mode aléatoire

**Priorité : basse · Effort : 1 session · Impact : fun / variété**

### Contexte
Charger un preset est instantané et brutal ; pour une écoute prolongée on
veut que le visuel évolue tout seul.

### Spécification
- **Transition douce** : au chargement d'un preset, interpoler tous les
  paramètres continus sur ~2 s (côté main, easing cosinus) au lieu du saut.
  Les paramètres discrets (routing, cases) basculent à mi-transition.
- **Bouton « Aléatoire »** dans la barre presets : génère un état plausible
  (valeurs tirées dans des plages sûres par paramètre, pas un chaos total).
- **Mode « Auto »** (case à cocher) : enchaîne un morphing vers un preset
  sauvegardé (ou aléatoire) toutes les N mesures — utilise `bpm`/`beat_phase`
  (T2) pour changer pile sur un début de mesure.

### Critères d'acceptation
- [ ] Charger un preset = fondu de 2 s, pas de saut visuel.
- [ ] Le mode Auto tourne 10 min sans intervention avec des états variés.
- [ ] Désactivable instantanément (la case décochée fige l'état courant).

---

## Ordre suggéré interface

~~T7~~ ✅ ~~T8~~ ✅ faits (2026-07-04). Reste : `T9 (MIDI learn s'appuie
sur les widgets de T7) → T10 (morphing de presets)`

---
---

# T11 — Améliorations du rendu image (session 2026-07-04/05) ✅ TERMINÉ

Série de demandes hors backlog, réalisées dans la foulée :

- **Plus de grossissement automatique des points** : les termes bass/rms de la
  taille sont supprimés (modes points et courbes). La taille reste pilotable
  par le slider « Taille points » et le routing Physique.
- **Vague au lieu du décalage** : le drift flownoise est réduit à 25 % ; une
  vague sinusoïdale diagonale traverse l'image en continu (z + léger y,
  amplitude ∝ rms/aigus × slider Turbulence).
- **Épaisseur (relief)** : slider 0–2 (onglet Image, défaut 0.5) qui contrôle
  toute la plage z des points — remplace l'étirement auto par les low-mid.
- **Densification** : l'image est suréchantillonnée (≤2×) pour que la grille
  utilise tout le budget → 1 000 000 points (mesuré), définition nettement
  meilleure. Coordonnées recompensées (taille écran inchangée).
- **Contours tracés** : Canny + silhouette rembg → les points de bord sont
  plus brillants/gros (uniform `u_edge`). Slider « Intensité contours » 0–2.
- **Nombre de points** : slider 50k–1M (onglet Image), rebuild débouncé 0,5 s
  (`ParticleSystem.set_budget`, masque rembg en cache).
- **Relief propre** : le sujet reçoit une carte de hauteur = luminance floutée
  (gaussienne σ=8) → surface embossée cohérente ; le décor est un plan plat
  (plus d'éparpillement z derrière).
- **Cibles Objet/Décor** :
  - onglet Vidéo : Rouge/Vert/Bleu/Contraste/Négatif/Luminosité ont un
    sélecteur « Décor + objet / Objet / Décor » (`<clé>_scope`). Scope global
    → composite (comme avant) ; scope objet/décor → par particule dans le
    vertex shader (uniforms `u_tint/u_con/u_neg` fg/bg).
  - onglet Physique : sélecteur « Tout / Objet / Décor » par bande
    (`band<i>_scope`) pour les effets par particule : Taille points,
    Luminosité, Turbulence (uniforms `u_szm/u_brm/u_tbm` fg/bg). Les effets
    plein écran (décalages, trainées, zoom, rotation) restent globaux.
    Fallback CPU : scopes couleur appliqués, scopes par bande ignorés.
- **Fixes** : uniforms optimisés out par le compilateur GLSL (guard
  `_members`) ; scroll des onglets (plus d'espace fantôme, pas de scroll
  quand le contenu tient) ; **presets qui ne chargeaient pas** (2026-07-05 :
  `var.set()` ne déclenche pas le `command` d'un ttk.Scale → les valeurs
  n'atteignaient jamais le moteur ; corrigé via `Scale.set()`, répare aussi
  le double-clic reset et la restauration d'autosave dans le panneau).

---
---

# Backlog effets visuels (T12 → T18)

Objectif : enrichir le vocabulaire visuel — aujourd'hui tout repose sur les
particules + bloom + décalages chromatiques. Chaque tiquet ajoute une
famille d'effets distincte, pilotable par le routing Physique quand ça a
du sens.

---

## T12 — Feedback vidéo (écho zoom / rotation, style tunnel)

**Priorité : haute · Effort : 1 session · Impact : transforme le rendu**

### Contexte
Le buffer d'accumulation ne fait que s'estomper (fade). Les visualiseurs
hypnotiques (MilkDrop, projectM) réinjectent l'image précédente avec une
légère transformation → tunnels infinis, spirales, fractales de trainées.

### Spécification
- Dans `begin_frame`, remplacer le simple fade par un **resample de l'accum**
  vers lui-même (ping-pong) avec une transformation UV : zoom léger
  (0.98–1.02), rotation (±1°/frame), décalage radial.
- Sliders (onglet Vidéo, groupe « Feedback ») : Zoom écho (-1 à +1, 0 = off),
  Rotation écho, Intensité (mix avec le fade actuel).
- Nouveaux effets routables : « Écho zoom » et « Écho rotation » (la bande
  module la transformation) → tunnel qui respire sur la basse.
- Le HUD et la grille restent hors feedback (déjà dessinés après).

### Critères d'acceptation
- [ ] Zoom écho positif = tunnel qui aspire vers le centre ; négatif = explosion.
- [ ] Aucune perte de fps (un seul pass fullscreen supplémentaire).
- [ ] Tout à zéro = comportement actuel exact (fade simple).

---

## T13 — Palettes de couleurs (gradient mapping)

**Priorité : haute · Effort : 1 session · Impact : identité visuelle immédiate**

### Contexte
Les couleurs viennent de l'image source ou du HSV de l'étoile ; les gains
RGB permettent de teinter mais pas de re-colorer. Un mapping
luminance → dégradé (comme les LUT cinéma) donne des identités fortes :
purple world, incandescent, glace, thermique…

### Spécification
- Au composite : `col = mix(col, gradient(lum(col)), montant)` où
  `gradient` est une rampe de 4-5 couleurs clés interpolées (uniform
  vec3[5], pas de texture nécessaire).
- ~8 palettes prédéfinies (Purple world, Incandescent, Glace, Thermique,
  Toxique, Sépia, Néon rose, Mono cyan) + combobox et slider « Montant »
  dans l'onglet Vidéo.
- **Rotation de teinte** optionnelle : le centroid spectral (ou une bande
  routée « Teinte ») fait tourner la palette → la couleur suit le timbre.
- Compatible scopes : appliqué au composite quand « Décor + objet »,
  par particule sinon (réutiliser l'infra fg/bg de T11).

### Critères d'acceptation
- [ ] Changer de palette re-colore instantanément tout le rendu.
- [ ] Montant à 0 = couleurs d'origine exactes.
- [ ] La palette est sauvée dans les presets.

---

## T14 — Explosion sur beat (physique à ressort)

**Priorité : haute · Effort : 1 session · Impact : le beat devient physique**

### Contexte
Le beat ne produit qu'un flash de luminosité. L'impact viscéral d'un kick
mérite une réponse spatiale : les particules projetées puis qui reviennent
élastiquement à leur place.

### Spécification
- Vertex shader (stateless) : déplacement radial
  `d = dir_depuis_centre × A × exp(-t/τ) × sin(ω·t)` où `t` = temps depuis
  le dernier beat (uniform), A piloté par l'intensité — un ressort amorti
  sans état par particule.
- Variantes par un hash du gl_VertexID : chaque particule part avec une
  amplitude/phase légèrement différente → l'explosion a du grain.
- Nouvel effet routable « Explosion » (la bande BEAT est la candidate
  naturelle, mais SUB sur un drop est aussi intéressant) + scope
  Objet/Décor (réutiliser T11).
- Slider « Élasticité » (τ : retour sec 100 ms → rebond long 800 ms).

### Critères d'acceptation
- [ ] Sur un kick isolé : projection + retour avec 1-2 oscillations.
- [ ] L'image redevient parfaitement nette entre les beats.
- [ ] Fonctionne en mode étoile, points et courbes.

---

## T15 — Toile / constellations (relier les points)

**Priorité : moyenne · Effort : 2 sessions · Impact : rendu unique**

### Contexte
Idée de la session T11 : relier les points proches pour dessiner une toile.
Un maillage de fils sur le sujet donne un look « wireframe organique »
qu'aucun slider actuel ne peut produire.

### Spécification
- Au build de l'image : échantillonner ~2-5k points du sujet (poisson-disque
  sur le masque), **triangulation de Delaunay** (scipy) → liste d'arêtes
  uploadée une fois en `GL_LINES` (indices vers les mêmes attributs que les
  points → les fils suivent la vague/explosion automatiquement).
- Alpha d'un fil = f(longueur) (les fils étirés s'éteignent) × bande routée
  « Toile » → la toile scintille en musique.
- Case « Toile » + slider densité dans l'onglet Image ; se superpose aux
  points (pas un mode exclusif).
- Les fils passent par le même pipeline (bloom les fait rougeoyer).

### Critères d'acceptation
- [ ] La toile épouse le sujet et suit toutes ses déformations.
- [ ] 60 fps conservés avec toile + 1M de points.
- [ ] Densité 0 = aucune ligne, rendu actuel exact.

---

## T16 — Fond génératif « nébuleuse »

**Priorité : moyenne · Effort : 1 session · Impact : profondeur de scène**

### Contexte
Derrière l'objet il n'y a que le noir et la grille. Un fond vivant mais
discret donnerait une scène complète, surtout en mode « sujet seul »
(hide_bg) où le décor d'origine disparaît.

### Spécification
- Pass fullscreen AVANT les particules : nébuleuse en fbm (fractal brownian
  motion, 4 octaves de value noise — le `vnoise` du shader existe déjà),
  teintée par la palette T13, dérive lente 3D + **parallaxe** avec la
  rotation de la caméra.
- Réactivité douce : luminosité globale ∝ rms lissé, éclaircissements
  locaux sur les basses (les « nuages » pulsent).
- Étoiles : ~300 points fixes lointains, scintillement par hash, parallaxe
  plus forte (elles sont « derrière » la nébuleuse).
- Combobox « Fond » dans l'onglet Vidéo : Aucun / Grille / Nébuleuse /
  Grille + nébuleuse, + slider intensité.

### Critères d'acceptation
- [ ] Le fond tourne avec la caméra (cohérence spatiale), en retrait du sujet.
- [ ] Coût < 1 ms/frame en 1280×720 à 1,5×.
- [ ] « Aucun » = noir actuel exact.

---

## T17 — Glitch / strobe sur transients

**Priorité : basse · Effort : 1 session · Impact : énergie sur les drops**

### Contexte
Les moments violents d'un morceau (drop, fill de batterie) méritent une
rupture visuelle : glitch numérique, image qui « casse » une frame.

### Spécification
- Au composite, déclenché par les transients HF ou une bande routée
  « Glitch » : décalage horizontal par tranches (8-16 bandes, offsets
  hashés re-seedés par déclenchement), séparation RGB amplifiée,
  quantification des couleurs 1-2 frames.
- Mode « Strobe » optionnel : 1 frame blanche/négative pile sur le beat
  (utilise `is_beat`), **plafonné à 8 Hz** (sécurité photosensibilité).
- Slider « Glitch » (seuil/intensité) dans l'onglet Vidéo ; 0 = off.
- Chaque glitch dure 2-4 frames max puis retour net (pas d'accumulation).

### Critères d'acceptation
- [ ] Un snare fort déclenche un glitch d'une ~frame, lisible mais bref.
- [ ] Le strobe ne dépasse jamais 8 flashs/seconde.
- [ ] À 0, aucun coût ni artefact.

---

## T18 — Kaléidoscope / symétries

**Priorité : basse · Effort : ½ session · Impact : hypnotique à peu de frais**

### Contexte
Une symétrie radiale transforme n'importe quel rendu en mandala — l'effet
« beaucoup pour pas cher » par excellence, très fort combiné au feedback T12.

### Spécification
- Au composite : repli des UV en coordonnées polaires sur N secteurs
  (miroir alterné pour la continuité), N ∈ {off, 2, 4, 6, 8, 12}.
- Combobox « Symétrie » + case « Rotation lente » (le motif tourne à
  vitesse routable — réutilise « Écho rotation » de T12 si présent).
- Le HUD reste hors symétrie (dessiné après, déjà le cas).

### Critères d'acceptation
- [ ] N=6 sur le mode étoile = mandala net, sans couture entre secteurs.
- [ ] Combiné au feedback T12 : spirales kaléidoscopiques stables.
- [ ] Off = rendu actuel exact.

---

## Ordre suggéré effets visuels

`T13 (palettes — impact/effort imbattable) → T12 (feedback — change la
nature du rendu) → T14 (explosion beat) → T18 (kaléidoscope, ½ session
bonus) → T16 (nébuleuse) → T15 (toile) → T17 (glitch)`

T12 + T13 + T18 combinés ≈ un moteur type MilkDrop avec ton identité.
