"""Déclaration de tous les paramètres du fond (T16).

Une entrée `Param` décrit à la fois le réglage du panneau (libellé, plage,
widget, groupe), l'uniform du shader, la sous-plage tirée par le dé, la
réponse aux macros et la réaction à la musique. Le panneau, l'état du fond
et les shaders sont générés depuis cette table : ajouter un réglage = une
ligne ici (+ son usage dans le GLSL).

Classes (`kind`) — ce que la musique et les macros ont le droit de faire :
- `exact` : rotation, rayon, position… ne déforme pas l'espace, sûr pour
  les impulsions (KICK, SNARE, BEAT) ;
- `deform` : déforme l'espace (réduit le pas de marche) ;
- `topology` : change la structure (période, nombre de cellules) :
  seulement ÉNERGIE ou macro, variation lente (`slew`) ;
- `int` : saute d'une valeur entière à l'autre, ne glisse pas ;
- `color` / `emissive` : ne touchent pas la géométrie ;
- `phase` : une vitesse, intégrée en dt ; la musique la multiplie ;
- `setting` : réglage de l'utilisateur, ni musique ni dé.
"""

import math
import random
from dataclasses import dataclass, field

KINDS = ("exact", "deform", "topology", "int", "color", "emissive", "phase",
         "setting")
# sources de la réaction musicale : 7 bandes EQ, puis la couche percussive
AUDIO_SOURCES = ("sub", "bass", "lowmid", "mid", "highmid", "presence",
                 "treble", "beat", "kick", "snare", "energy", "rms")
FAST_SOURCES = ("beat", "kick", "snare")
FAST_OK = ("exact", "color", "emissive")      # classes que KICK/SNARE peuvent viser
SLOW_ONLY = ("topology", "int")               # seulement ÉNERGIE
# variation maximale par défaut, en fraction de la plage par seconde
DEFAULT_SLEW = {"topology": 0.4, "exact": 4.0, "deform": 2.0, "color": 4.0,
                "emissive": 8.0}

QUALITIES = ["Bas", "Moyen", "Haut", "Auto"]
RESOLUTIONS = [("¼", 0.25), ("⅓", 1.0 / 3.0), ("½", 0.5), ("¾", 0.75), ("1", 1.0)]

GROUPS = [
    ("top", "Fond"),
    ("macro_amb", "Macros d'ambiance"),
    ("macro_form", "Macros de forme"),
    ("nw", "Générateur et monde de bruit"),
    ("sp", "Espace"),
    ("dx", "Distorsion"),
    ("sh", "Formes"),
    ("fu", "Fusion"),
    ("col", "Couleur"),
    ("mat", "Matière et lumière"),
    ("mov", "Mouvement"),
    ("comp", "Composition"),
    ("solid", "Couleur unie et grille"),
]

MACROS_AMB = [
    ("energy", "Énergie", "calme → agité"),
    ("warmth", "Chaleur", "froid → chaud"),
    ("glow", "Éclat", "mat → néon"),
    ("daylight", "Nuit ↔ Jour", "sombre → lumineux"),
    ("depth", "Profondeur", "proche, serré → lointain, abyssal"),
    ("tension", "Tension", "les formes collent au modèle → s'en écartent"),
    ("fluid", "Fluidité", "raide → liquide"),
    ("symmetry", "Symétrie", "libre → ordonné"),
    ("strange", "Étrangeté", "réaliste → onirique"),
    ("contrast", "Contraste", "doux → dur"),
]
MACROS_FORM = [
    ("complexity", "Complexité", "simple → chargé"),
    ("chaos", "Chaos", "régulier → désordonné"),
    ("density", "Densité", "aéré → serré"),
    ("organic", "Organique ↔ Géométrique", "arrondi → anguleux"),
    ("scale", "Échelle", "micro → macro"),
    ("fill", "Vide ↔ Plein", "ajouré → massif"),
    ("crystal", "Cristal ↔ Chair", "arêtes vives → matière douce"),
]


@dataclass(frozen=True)
class Param:
    """Un réglage du fond. `default`, `lo`, `hi` dans l'unité affichée ;
    pour une liste (`options`), la valeur est l'index de l'option."""
    key: str
    label: str
    group: str
    lo: float
    hi: float
    default: float
    step: float = 0.01
    kind: str = "exact"
    law: str = "lin"               # lin, log (échelles, fréquences), angle (boucle)
    options: tuple = None          # liste → combo
    check: bool = False            # case à cocher
    rand: tuple = None             # sous-plage tirée par le dé (None = jamais)
    audio: tuple = None            # (source, montant) réaction par défaut
    macros: dict = field(default_factory=dict)   # macro → écart à macro = 1
    slew: float = None             # fraction de la plage par seconde
    shader: bool = True            # envoyé tel quel comme uniform
    tip: str = None
    weights: tuple = None          # dé : poids de chaque valeur de la plage `rand`
    zero: float = 0.0              # dé : probabilité de tirer 0 (effet éteint)

    @property
    def uniform(self) -> str:
        return "u_" + self.key[3:]

    @property
    def max_slew(self) -> float:
        """Variation maximale (unités par seconde), None = immédiat."""
        rate = self.slew if self.slew is not None else DEFAULT_SLEW.get(self.kind)
        return None if rate is None else rate * (self.hi - self.lo)


def _group(group, *rows):
    return [Param(key, label, group, *rest, **kw) for key, label, *rest, kw in rows]


def _macro_params():
    out = []
    for group, rows in (("macro_amb", MACROS_AMB), ("macro_form", MACROS_FORM)):
        for name, label, tip in rows:
            out.append(Param(f"bg_m_{name}", label, group, 0.0, 1.0, 0.5,
                             kind="setting", shader=False, tip=tip + "\n(0,5 = neutre)"))
    return out


PALETTE_MODES = ("Cosinus", "Dégradé 2 couleurs", "Dégradé 3 couleurs",
                 "Monochrome", "Spectre", "Néon", "Pastel", "Thermique")
COSINE_PRESETS = ("Arc-en-ciel", "Aurore", "Braise", "Lagune", "Coucher",
                  "Acide", "Corail")
COLOR_SOURCES = ("Hauteur", "Profondeur", "Normale", "Structure", "Instance",
                 "Nappes de bruit", "Creux et bosses")
HARMONIES = ("Libre", "Même teinte", "Complémentaire", "Triadique", "Analogue")
MATERIAL_NAMES = ("Argile", "Chrome", "Verre", "Néon", "Velours", "Nacre",
                  "Personnalisé")
LIGHT_MODES = ("Lumière du modèle", "Soleil du fond")
BREATH_BEATS = ((0.5, "½ beat"), (1.0, "1 beat"), (2.0, "2 beats"),
                (4.0, "4 beats"), (8.0, "8 beats"))

TOP = [
    Param("bg_visible", "Fond visible", "top", 0, 1, 0, check=True,
          kind="setting", shader=False,
          tip="Masqué = rendu d'avant T16, sans aucun coût"),
    Param("bg_quality", "Qualité", "top", 0, 3, 1, step=1,
          options=tuple(QUALITIES), kind="setting", shader=False,
          tip="Bas / Moyen / Haut : nombre de pas, ombres, occlusion.\n"
              "Auto = Moyen + résolution baissée si le fond dépasse son budget"),
    Param("bg_res", "Résolution du fond", "top", 0, len(RESOLUTIONS) - 1, 1,
          step=1, options=tuple(r[0] for r in RESOLUTIONS), kind="setting",
          shader=False, tip="Fraction de la taille de rendu : ⅓ ≈ 1,5 ms sur un M3"),
    Param("bg_temporal", "Lissage temporel", "top", 0.0, 1.0, 0.7, kind="setting",
          shader=False,
          tip="Mélange les frames successives (rayon décalé d'une fraction de\n"
              "pixel à chaque frame) : contours nets, plus d'escaliers ni de\n"
              "zébrures au loin. 0 = chaque frame seule"),
    Param("bg_seed", "Seed", "top", 0, 9999, 0, step=1, kind="setting",
          shader=False, tip="Numéro du dernier tirage « 🎲 Nouvelle forme »"),
]

# ── chaîne d'étages (passe 2) : Espace → Distorsion → Formes → Fusion ──
SPACE_MODES = ("Objet seul", "Grille 3D", "Grille au sol",
               "Anneaux le long du trajet", "Miroirs", "Pliage KIFS")
PRIMITIVES = ("Sphère", "Boîte", "Boîte arrondie", "Boîte en fil de fer", "Tore",
              "Octaèdre", "Capsule", "Cylindre", "Croix (3 barres)", "Gyroïde (coque)")
COMBINATIONS = ("Union douce", "Creusement doux", "Intersection douce")
DEFORM_AXES = ("Le long du trajet", "Vertical")
MAX_SATELLITES = 8
MAX_FOLDS = 8

# ── monde de bruit (passe 4) : un monde continu en couches de bruit ──
GENERATORS = ("Formes répétées", "Monde de bruit")
WORLD_MODES = ("Caverne", "Nappes liquides", "Paysage")

NW = _group(
    "nw",
    ("bg_nw_gen", "Générateur", 0, len(GENERATORS) - 1, 0,
     dict(step=1, options=GENERATORS, kind="int", rand=(0, 1), weights=(45, 55),
          tip="Formes répétées : la chaîne Espace → Distorsion → Formes → Fusion.\n"
              "Monde de bruit : un monde continu (caverne, nappes, paysage) fait de\n"
              "couches de bruit ; la Distorsion s'y applique aussi")),
    ("bg_nw_mode", "Monde", 0, len(WORLD_MODES) - 1, 0,
     dict(step=1, options=WORLD_MODES, kind="int", rand=(0, 2), weights=(45, 35, 20))),
    ("bg_nw_scale", "Taille du monde", 4.0, 60.0, 16.0,
     dict(step=0.5, law="log", kind="topology", rand=(8.0, 35.0),
          macros={"density": -0.4, "scale": 0.6},
          tip="Taille des grandes formes du monde (unités du fond)")),
    ("bg_nw_fill", "Remplissage", 0.2, 0.8, 0.5,
     dict(kind="topology", rand=(0.3, 0.65), macros={"fill": 0.15}, shader=False,
          tip="Caverne : part de l'espace occupée par la roche")),
    ("bg_nw_breath", "Respiration", 0.0, 0.3, 0.05,
     dict(step=0.005, rand=(0.0, 0.15),
          tip="Les parois gonflent et dégonflent lentement")),
    ("bg_nw_warp", "Déformation", 0.0, 2.0, 0.6,
     dict(kind="deform", rand=(0.2, 1.6), macros={"chaos": 0.6, "fluid": 0.4},
          tip="Le bruit déforme le bruit : formes liquides, organiques")),
    ("bg_nw_warp_size", "Taille de la déformation", 0.3, 2.0, 0.8,
     dict(kind="deform", rand=(0.4, 1.5), tip="En tailles du monde")),
    ("bg_nw_detail", "Couche moyenne", 0.0, 1.0, 0.35,
     dict(kind="deform", rand=(0.0, 0.7), macros={"complexity": 0.3},
          tip="Formes trois fois plus petites ajoutées au monde")),
    ("bg_nw_fine", "Couche fine", 0.0, 0.5, 0.1,
     dict(kind="deform", rand=(0.0, 0.3), zero=0.3, macros={"complexity": 0.15},
          tip="Détail fin (s'efface au loin : pas de grain)")),
    ("bg_nw_flow", "Flux (unités/s)", 0.0, 1.0, 0.12,
     dict(kind="phase", rand=(0.02, 0.4), audio=("energy", 0.4),
          macros={"energy": 0.15}, shader=False,
          tip="Les couches glissent lentement, chacune dans sa direction")),
    ("bg_nw_thick", "Épaisseur des nappes", 0.02, 0.5, 0.12,
     dict(rand=(0.04, 0.3), macros={"fill": 0.08})),
    ("bg_nw_gyroid", "Gyroïde dans les nappes", 0.0, 1.0, 0.4,
     dict(kind="deform", rand=(0.0, 0.9), macros={"symmetry": 0.4},
          tip="Mêle aux nappes un gyroïde (surface minimale) : plus ordonné")),
    ("bg_nw_relief", "Hauteur du relief (Paysage)", 0.2, 4.0, 1.5,
     dict(kind="topology", rand=(0.6, 3.0), tip="En tailles du monde × 0,3")),
    ("bg_nw_vault", "Voûte (Paysage)", 0, 1, 0,
     dict(step=1, check=True, kind="int", rand=(0, 1), tip="Un plafond : canyon")),
    ("bg_nw_soft", "Douceur du tunnel", 0.0, 1.0, 0.5,
     dict(rand=(0.2, 0.9),
          tip="Le couloir du trajet est creusé dans le monde ; 1 = bords très doux")),
)

SP = _group(
    "sp",
    ("bg_sp_mode", "Mode", 0, len(SPACE_MODES) - 1, 1,
     dict(step=1, options=SPACE_MODES, kind="int", rand=(0, 5),
          weights=(10, 25, 15, 20, 15, 15),
          tip="Comment le monde se répète autour du trajet de l'objet")),
    ("bg_sp_period", "Période", 2.0, 16.0, 6.0,
     dict(step=0.1, law="log", kind="topology", rand=(3.0, 11.0),
          macros={"density": -0.5, "scale": 0.6},
          tip="Taille d'une cellule (unités du fond ; le modèle a un rayon de 1).\n"
              "Objet seul : la grande forme fait 2,5 périodes ; KIFS : taille du motif")),
    ("bg_sp_along", "Période le long du trajet", 2.0, 16.0, 6.0,
     dict(step=0.1, kind="topology", rand=(3.0, 12.0),
          tip="Anneaux : écart entre deux anneaux ; Miroirs, KIFS : longueur du motif\n"
              "répété en miroir ; Objet seul : une forme tous les 2 × cette période")),
    ("bg_sp_count", "Instances par anneau", 3, 24, 8,
     dict(step=1, kind="int", rand=(3, 14))),
    ("bg_sp_dist", "Écart au couloir (Objet seul)", 0.0, 30.0, 3.0,
     dict(step=0.5, rand=(0.0, 12.0),
          tip="Distance entre le couloir du trajet et la grande forme")),
    ("bg_sp_iter", "Itérations de pliage (KIFS)", 1, MAX_FOLDS, 4,
     dict(step=1, kind="int", rand=(2, 6), macros={"complexity": 3})),
    ("bg_sp_angle", "Angle de pliage", 0.0, 90.0, 20.0,
     dict(step=0.5, rand=(0.0, 60.0), slew=0.05, shader=False,
          macros={"chaos": 25.0},
          tip="Rotation à chaque pliage (KIFS) ; varie lentement")),
    ("bg_sp_fscale", "Échelle de pliage", 1.2, 3.0, 2.0,
     dict(kind="topology", rand=(1.4, 2.6))),
    ("bg_sp_off_x", "Décalage de pliage x", 0.0, 2.0, 1.0,
     dict(rand=(0.2, 1.5), shader=False)),
    ("bg_sp_off_y", "Décalage de pliage y", 0.0, 2.0, 1.0,
     dict(rand=(0.2, 1.5), shader=False)),
    ("bg_sp_off_z", "Décalage de pliage z", 0.0, 2.0, 1.0,
     dict(rand=(0.0, 1.5), shader=False)),
    ("bg_sp_mirror_x", "Miroir X (en travers)", 0, 1, 1,
     dict(step=1, check=True, kind="int", rand=(0, 1))),
    ("bg_sp_mirror_y", "Miroir Y (haut / bas)", 0, 1, 0,
     dict(step=1, check=True, kind="int", rand=(0, 1))),
    ("bg_sp_mirror_z", "Miroir Z (le long du trajet)", 0, 1, 0,
     dict(step=1, check=True, kind="int", rand=(0, 1))),
    ("bg_sp_disorder", "Désordre par instance", 0.0, 1.0, 0.0,
     dict(rand=(0.0, 0.8), zero=0.5, macros={"chaos": 0.45, "symmetry": -0.3},
          tip="Rotation et taille propres à chaque instance\n"
              "(0 = instances identiques : rendu plus net et plus rapide)")),
    ("bg_sp_missing", "Instances manquantes", 0.0, 0.8, 0.0,
     dict(kind="topology", rand=(0.0, 0.5), zero=0.4, macros={"fill": -0.35})),
    ("bg_sp_eq", "Hauteur ← EQ (Grille au sol)", 0.0, 1.0, 0.0,
     dict(rand=(0.0, 0.8), zero=0.5,
          tip="Égaliseur 3D : une bande EQ par rangée étire les formes vers le haut")),
)

DX = _group(
    "dx",
    ("bg_dx_twist", "Torsion (tours par unité)", 0.0, 0.05, 0.0,
     dict(step=0.001, kind="deform", rand=(0.0, 0.03), zero=0.5,
          tip="Le monde tourne autour de l'axe des déformations en avançant")),
    ("bg_dx_bend", "Courbure", -0.05, 0.05, 0.0,
     dict(step=0.001, kind="deform", rand=(-0.03, 0.03), zero=0.5,
          tip="Les rangées se courbent en s'éloignant du trajet")),
    ("bg_dx_wave_amp", "Ondes : amplitude", 0.0, 0.5, 0.0,
     dict(step=0.005, kind="deform", rand=(0.0, 0.3), zero=0.5,
          audio=("bass", 0.1),
          tip="Fraction de la période")),
    ("bg_dx_wave_len", "Ondes : longueur", 1.0, 20.0, 8.0,
     dict(step=0.1, kind="deform", rand=(4.0, 16.0), tip="Unités du fond")),
    ("bg_dx_wave_speed", "Ondes : vitesse", 0.0, 0.2, 0.02,
     dict(step=0.005, kind="phase", rand=(0.0, 0.1),
          shader=False, tip="Tours par seconde : les ondes glissent lentement")),
    ("bg_dx_warp_amp", "Warp : amplitude", 0.0, 3.0, 0.0,
     dict(step=0.01, kind="deform", rand=(0.0, 2.0), zero=0.5,
          macros={"chaos": 1.0},
          tip="Bruit 3D (texture) qui déplace l'espace (unités du fond)")),
    ("bg_dx_warp_freq", "Warp : fréquence", 0.03, 0.5, 0.15,
     dict(step=0.005, law="log", kind="deform", rand=(0.05, 0.3),
          tip="Nombre de bosses par unité")),
    ("bg_dx_warp_oct", "Warp : octaves", 1, 3, 1,
     dict(step=1, kind="int", rand=(1, 3), macros={"complexity": 1.5})),
    ("bg_dx_axis", "Axe des déformations", 0, len(DEFORM_AXES) - 1, 0,
     dict(step=1, options=DEFORM_AXES, kind="int", rand=(0, 1),
          weights=(70, 30))),
)

SH = _group(
    "sh",
    ("bg_sh_a", "Primitive A", 0, len(PRIMITIVES) - 1, 3,
     dict(step=1, options=PRIMITIVES, kind="int", rand=(0, 9))),
    ("bg_sh_b", "Primitive B", 0, len(PRIMITIVES) - 1, 5,
     dict(step=1, options=PRIMITIVES, kind="int", rand=(0, 9))),
    ("bg_sh_morph", "Morphing A → B", 0.0, 1.0, 0.0,
     dict(rand=(0.0, 1.0), zero=0.35, audio=("energy", 0.3),
          tip="Mélange des deux distances : passage continu, sans trou")),
    ("bg_sh_size", "Taille", 0.05, 0.45, 0.2,
     dict(step=0.005, rand=(0.12, 0.4), macros={"density": 0.1, "fill": 0.1},
          tip="Fraction de la période (réduite si la forme déborderait de sa cellule)")),
    ("bg_sh_sx", "Étirement x", 0.3, 3.0, 1.0,
     dict(law="log", rand=(0.5, 2.0))),
    ("bg_sh_sy", "Étirement y", 0.3, 3.0, 1.0,
     dict(law="log", rand=(0.5, 2.5))),
    ("bg_sh_sz", "Étirement z", 0.3, 3.0, 1.0,
     dict(law="log", rand=(0.5, 2.0))),
    ("bg_sh_round", "Arrondi", 0.0, 0.3, 0.0,
     dict(step=0.005, rand=(0.0, 0.15), macros={"organic": -0.12, "fluid": 0.08})),
    ("bg_sh_shell", "Coque (0 = plein)", 0.0, 0.2, 0.0,
     dict(step=0.005, rand=(0.0, 0.12), zero=0.6, macros={"fill": -0.1},
          tip="Épaisseur de la paroi : se voit quand la fusion creuse la forme")),
    ("bg_sh_thick", "Épaisseur des barres", 0.01, 0.2, 0.04,
     dict(step=0.005, rand=(0.02, 0.12), macros={"fill": 0.04},
          tip="Fil de fer, croix, gyroïde (fraction de la période)")),
    ("bg_sh_rot", "Rotation de l'instance", 0.0, 360.0, 0.0,
     dict(step=1, law="angle", rand=(0.0, 360.0), shader=False)),
    ("bg_sh_spin", "Rotation lente (tours/min)", -0.1, 0.1, 0.0,
     dict(step=0.005, kind="phase", rand=(-0.06, 0.06), zero=0.4,
          shader=False)),
    ("bg_sh_wave", "Vague", 0.0, 1.0, 0.0,
     dict(rand=(0.0, 0.6), zero=0.5, audio=("beat", 0.25),
          tip="Les instances gonflent l'une après l'autre, en vague, au tempo")),
)

FU = _group(
    "fu",
    ("bg_fu_count", "Satellites", 0, MAX_SATELLITES, 0,
     dict(step=1, kind="int", rand=(1, MAX_SATELLITES), zero=0.4,
          macros={"complexity": 4})),
    ("bg_fu_combo", "Combinaison", 0, len(COMBINATIONS) - 1, 0,
     dict(step=1, options=COMBINATIONS, kind="int", rand=(0, 2),
          weights=(60, 25, 15))),
    ("bg_fu_k", "Viscosité", 0.05, 1.0, 0.3,
     dict(rand=(0.1, 0.8), audio=("energy", 0.2),
          macros={"organic": -0.3, "fluid": 0.2},
          tip="Plus haut : les satellites fondent dans la forme comme un liquide")),
    ("bg_fu_rad", "Rayon des satellites", 0.1, 1.0, 0.4,
     dict(rand=(0.2, 0.8), tip="Fraction de la taille de la forme")),
    ("bg_fu_spread", "Écartement", 0.0, 2.0, 1.2,
     dict(rand=(0.6, 1.8), audio=("kick", 0.15),
          tip="Distance des satellites, en tailles de la forme (le KICK les écarte)")),
    ("bg_fu_orbit", "Orbite (tours/min)", 0.0, 0.5, 0.05,
     dict(step=0.005, kind="phase", rand=(0.0, 0.3),
          shader=False)),
    ("bg_fu_glow", "Satellites lumineux", 0.0, 3.0, 0.6,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), audio=("kick", 0.2),
          macros={"glow": 1.0})),
)

COL = _group(
    "col",
    ("bg_col_mode", "Palette", 0, len(PALETTE_MODES) - 1, 1,
     dict(step=1, options=PALETTE_MODES, kind="int", rand=(0, 7), shader=False)),
    ("bg_col_preset", "Palette cosinus", 0, len(COSINE_PRESETS) - 1, 0,
     dict(step=1, options=COSINE_PRESETS, kind="int", rand=(0, 6), shader=False,
          tip="Utilisée quand la palette est « Cosinus »")),
    ("bg_col_a_hue", "Couleur A : teinte", 0.0, 1.0, 0.62,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_a_sat", "Couleur A : saturation", 0.0, 1.0, 0.7,
     dict(kind="color", rand=(0.3, 1.0), shader=False)),
    ("bg_col_a_lum", "Couleur A : luminosité", 0.0, 2.0, 0.25,
     dict(kind="color", rand=(0.1, 0.8), shader=False)),
    ("bg_col_b_hue", "Couleur B : teinte", 0.0, 1.0, 0.52,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_b_sat", "Couleur B : saturation", 0.0, 1.0, 0.8,
     dict(kind="color", rand=(0.3, 1.0), shader=False)),
    ("bg_col_b_lum", "Couleur B : luminosité", 0.0, 2.0, 0.8,
     dict(kind="color", rand=(0.4, 1.2), shader=False)),
    ("bg_col_c_hue", "Couleur C : teinte", 0.0, 1.0, 0.88,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_c_sat", "Couleur C : saturation", 0.0, 1.0, 0.6,
     dict(kind="color", rand=(0.2, 1.0), shader=False)),
    ("bg_col_c_lum", "Couleur C : luminosité", 0.0, 2.0, 1.0,
     dict(kind="color", rand=(0.5, 1.5), shader=False)),
    ("bg_col_source", "Couleur selon", 0, len(COLOR_SOURCES) - 1, 5,
     dict(step=1, options=COLOR_SOURCES, kind="int", rand=(0, 6),
          weights=(10, 8, 10, 12, 20, 30, 10),
          tip="Ce qui choisit la couleur dans la palette (lue d'un bout à\n"
              "l'autre, sans répétition : pas de bandes)")),
    ("bg_col_spread", "Étendue de la palette", 0.0, 1.0, 0.7,
     dict(kind="color", rand=(0.3, 1.0), macros={"strange": 0.3},
          tip="0 = une seule couleur (le milieu de la palette), 1 = toute la palette")),
    ("bg_col_inst", "Variation par instance", 0.0, 1.0, 0.3,
     dict(kind="color", rand=(0.0, 0.7),
          tip="Chaque instance décale un peu sa couleur")),
    ("bg_col_noise", "Taille des nappes de couleur", 3.0, 80.0, 18.0,
     dict(step=0.5, law="log", kind="color", rand=(6.0, 50.0),
          tip="Nappes de bruit : taille des taches de couleur (unités du fond)")),
    ("bg_col_hue", "Décalage de teinte", 0.0, 1.0, 0.0,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_cycle", "Cycle de teinte (tours/min)", -1.0, 1.0, 0.0,
     dict(kind="phase", rand=(-0.3, 0.3), audio=("energy", 0.5),
          macros={"strange": 0.3}, shader=False)),
    ("bg_col_sat", "Saturation", 0.0, 1.0, 0.6,
     dict(kind="color", rand=(0.3, 1.0), audio=("energy", 0.15),
          macros={"glow": 0.3})),
    ("bg_col_contrast", "Contraste du fond", 0.5, 2.0, 1.0,
     dict(kind="color", rand=(0.8, 1.5), macros={"contrast": 0.6})),
    ("bg_col_steps", "Paliers (postérisation)", 0, 12, 0,
     dict(step=1, kind="int", macros={"contrast": 3},
          tip="0 = dégradé continu ; sinon nombre de paliers de couleur")),
    ("bg_col_tint", "Température des couleurs", -1.0, 1.0, 0.0,
     dict(kind="color", macros={"warmth": 0.8}, shader=False,
          tip="−1 = froid (bleuté) … +1 = chaud (ambré)")),
    ("bg_col_emit_hue", "Couleur des émissifs", 0.0, 1.0, 0.5,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_emit", "Force des émissifs", 0.0, 4.0, 1.0,
     dict(step=0.05, kind="emissive", rand=(0.3, 2.0), macros={"glow": 1.5},
          shader=False)),
    ("bg_col_shadow_hue", "Couleur des ombres", 0.0, 1.0, 0.66,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_shadow", "Teinte des ombres", 0.0, 1.0, 0.3,
     dict(kind="color", rand=(0.0, 0.7), shader=False)),
    ("bg_col_hor_hue", "Couleur d'horizon", 0.0, 1.0, 0.72,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_col_hor_lum", "Luminosité d'horizon", 0.0, 1.0, 0.16,
     dict(kind="color", rand=(0.03, 0.25), macros={"daylight": 0.35}, shader=False,
          tip="Lueur de l'horizon dans laquelle le lointain se fond")),
    ("bg_col_harmony", "Harmonie avec le modèle", 0, len(HARMONIES) - 1, 0,
     dict(step=1, options=HARMONIES, kind="int", shader=False,
          tip="Cale la teinte de la palette sur le rim du modèle\n"
              "(onglet Objet) : même teinte, complémentaire…")),
    ("bg_col_music", "Teinte musicale", 0.0, 1.0, 0.0,
     dict(kind="color", rand=(0.0, 0.5), shader=False,
          tip="Aigus et basses font tourner la teinte")),
    ("bg_col_flash", "Flash de couleur", 0.0, 1.0, 0.0,
     dict(kind="emissive", rand=(0.0, 0.6), shader=False,
          tip="Les formes s'illuminent de la couleur des émissifs sur la caisse claire")),
    ("bg_col_irid", "Irisation", 0.0, 1.0, 0.3,
     dict(kind="color", rand=(0.0, 0.8), macros={"strange": 0.5},
          tip="Les reflets changent de couleur avec l'angle de vue (film mince,\n"
              "bulle de savon)")),
    ("bg_col_irid_freq", "Épaisseur du film d'irisation", 1.0, 10.0, 3.0,
     dict(step=0.1, kind="color", rand=(1.0, 6.0),
          tip="Plus épais : plus de couleurs entre le centre et le bord")),
)

MAT = _group(
    "mat",
    ("bg_mat_preset", "Matière", 0, len(MATERIAL_NAMES) - 1, 0,
     dict(step=1, options=MATERIAL_NAMES, kind="setting", shader=False,
          tip="Préréglage : règle d'un coup rugosité, métal,\n"
              "translucidité, émission et fresnel")),
    ("bg_mat_rough", "Rugosité", 0.02, 1.0, 0.6,
     dict(kind="color", rand=(0.1, 0.9), macros={"glow": -0.3})),
    ("bg_mat_metal", "Métal", 0.0, 1.0, 0.0,
     dict(kind="color", rand=(0.0, 1.0))),
    ("bg_mat_transl", "Translucidité", 0.0, 1.0, 0.0,
     dict(kind="color", rand=(0.0, 0.7), macros={"crystal": 0.5},
          tip="Fausse diffusion sous la surface : la lumière traverse")),
    ("bg_mat_emit", "Émission propre", 0.0, 4.0, 0.0,
     dict(step=0.05, kind="emissive", rand=(0.0, 1.0), audio=("rms", 0.15),
          macros={"glow": 1.0})),
    ("bg_mat_edges", "Arêtes lumineuses", 0.0, 3.0, 0.3,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), audio=("treble", 0.2),
          macros={"crystal": -1.0, "glow": 0.8}, shader=False,
          tip="Les arêtes vives brillent (envoyé au shader comme u_edges)")),
    ("bg_mat_chase", "Poursuite au beat", 0.0, 1.0, 0.0,
     dict(kind="emissive", rand=(0.0, 1.0), zero=0.6,
          tip="Une instance sur quatre s'allume, à tour de rôle sur chaque beat")),
    ("bg_mat_glow", "Halo des formes", 0.0, 2.0, 0.3,
     dict(kind="emissive", rand=(0.0, 1.0), macros={"glow": 0.8, "energy": 0.3},
          tip="Lueur autour des formes (frôlements du rayon)")),
    ("bg_mat_fresnel", "Fresnel / contre-jour", 0.0, 2.0, 0.5,
     dict(kind="color", rand=(0.0, 1.5), audio=("treble", 0.1))),
    ("bg_mat_ao", "Occlusion des creux", 0.0, 2.0, 1.0,
     dict(kind="color", rand=(0.5, 1.5), macros={"contrast": 0.5})),
    ("bg_mat_env", "Reflets de l'environnement", 0.0, 2.0, 1.0,
     dict(kind="color", rand=(0.3, 1.5), macros={"crystal": -0.4},
          tip="Les formes reflètent le ciel et l'horizon, nets si la matière\n"
              "est lisse, flous si elle est rugueuse")),
    ("bg_mat_rim", "Contre-jour", 0.0, 2.0, 0.6,
     dict(kind="color", rand=(0.0, 1.5), macros={"glow": 0.3},
          tip="Le bord des formes s'éclaire de la couleur de la brume :\n"
              "elles se détachent du fond")),
    ("bg_mat_bump", "Relief de surface", 0.0, 1.0, 0.3,
     dict(kind="color", rand=(0.0, 0.8), macros={"crystal": 0.4},
          tip="Bosses fines dessinées par la lumière (ne coûte presque rien)")),
    ("bg_mat_bump_size", "Taille du relief", 0.05, 2.0, 0.4,
     dict(step=0.01, law="log", kind="color", rand=(0.1, 1.0),
          tip="Unités du fond (le modèle a un rayon de 1)")),
    ("bg_mat_grain", "Grain de la matière", 0.0, 1.0, 0.25,
     dict(kind="color", rand=(0.0, 0.6), tip="Couleur marbrée, comme une pierre")),
    ("bg_mat_veins", "Veines lumineuses", 0.0, 3.0, 0.0,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), zero=0.6,
          audio=("energy", 0.3), macros={"glow": 0.6},
          tip="Des veines de la couleur des émissifs parcourent les formes")),
    ("bg_mat_veins_width", "Finesse des veines", 0.01, 0.2, 0.05,
     dict(step=0.005, kind="emissive", rand=(0.02, 0.12),
          tip="Petit = veines fines")),
    ("bg_mat_veins_size", "Taille des veines", 0.5, 30.0, 5.0,
     dict(step=0.1, law="log", kind="emissive", rand=(1.0, 15.0),
          tip="Écart entre les veines (unités du fond)")),
    ("bg_light_mode", "Lumière", 0, len(LIGHT_MODES) - 1, 0,
     dict(step=1, options=LIGHT_MODES, kind="setting", shader=False,
          tip="Lumière du modèle : le fond est éclairé par la lumière principale\n"
              "du Studio, comme le modèle (même espace) ; Soleil du fond :\n"
              "un soleil réglé ci-dessous")),
    ("bg_light_az", "Soleil : azimut", 0.0, 360.0, 40.0,
     dict(step=1, law="angle", rand=(0.0, 360.0), shader=False)),
    ("bg_light_el", "Soleil : hauteur", 0.0, 90.0, 35.0,
     dict(step=1, rand=(10.0, 70.0), shader=False)),
    ("bg_light_int", "Lumière : intensité", 0.0, 4.0, 1.5,
     dict(step=0.05, kind="color", rand=(0.5, 2.5), macros={"daylight": 1.2},
          shader=False)),
    ("bg_light_temp", "Soleil : température (K)", 2000.0, 12000.0, 5500.0,
     dict(step=50, kind="color", rand=(3000.0, 9000.0), macros={"warmth": -4000.0},
          shader=False)),
    ("bg_light_sky", "Ciel : intensité", 0.0, 2.0, 0.4,
     dict(kind="color", rand=(0.1, 1.0), macros={"daylight": 0.6}, shader=False)),
    ("bg_light_sky_hue", "Ciel : teinte", 0.0, 1.0, 0.6,
     dict(law="angle", kind="color", rand=(0.0, 1.0), shader=False)),
    ("bg_light_shadow", "Dureté des ombres", 0.0, 1.0, 0.0,
     dict(kind="color", rand=(0.0, 1.0), macros={"contrast": 0.4},
          tip="0 = pas d'ombre portée (plus léger), 1 = ombres nettes")),
    ("bg_light_center", "Le modèle éclaire le fond", 0.0, 4.0, 0.5,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), audio=("rms", 0.2),
          shader=False, tip="Lumière au centre de la scène, couleur du rim du modèle")),
    ("bg_light_flash", "Flash de lumière", 0.0, 1.0, 0.0,
     dict(kind="emissive", rand=(0.0, 0.6), shader=False,
          tip="Le soleil s'allume sur la caisse claire")),
)

MOV = _group(
    "mov",
    ("bg_mov_speed", "Vitesse", 0.05, 4.0, 1.0,
     dict(law="log", kind="phase", rand=(0.4, 2.0), audio=("energy", 0.3),
          macros={"energy": 1.0}, shader=False,
          tip="Multiplie tous les mouvements propres du fond")),
    ("bg_mov_rot_az", "Rotation : azimut de l'axe", 0.0, 360.0, 30.0,
     dict(step=1, law="angle", rand=(0.0, 360.0), shader=False)),
    ("bg_mov_rot_tilt", "Rotation : inclinaison de l'axe", 0.0, 90.0, 25.0,
     dict(step=1, rand=(0.0, 60.0), shader=False)),
    ("bg_mov_rot", "Rotation du décor (tours/min)", -1.0, 1.0, 0.0,
     dict(kind="phase", audio=("mid", 0.1), shader=False,
          tip="0 = décor fixe (l'objet voyage dedans)")),
    ("bg_mov_travel", "Vitesse du voyage (unités/s)", 0.0, 6.0, 0.6,
     dict(step=0.05, kind="phase", audio=("bass", 0.15),
          macros={"energy": 0.4}, shader=False,
          tip="L'objet avance dans le décor fixe, la caméra le suit\n"
              "(le modèle a un rayon de 1 unité)")),
    ("bg_mov_dir_az", "Direction du voyage : azimut", 0.0, 360.0, 0.0,
     dict(step=1, law="angle", rand=(0.0, 360.0), shader=False,
          tip="0 = droit devant, vers le fond de l'écran (vue de départ)")),
    ("bg_mov_dir_el", "Direction du voyage : hauteur", -90.0, 90.0, 0.0,
     dict(step=1, rand=(-20.0, 20.0), shader=False,
          tip="Monter ou descendre ; ignoré en Grille au sol et en Paysage\n"
              "(le sol reste au sol)")),
    ("bg_mov_breath", "Respiration", 0.0, 0.3, 0.0,
     dict(step=0.005, rand=(0.0, 0.03), shader=False,
          tip="Le fond grandit et rétrécit au tempo")),
    ("bg_mov_breath_beats", "Respiration : période", 0, len(BREATH_BEATS) - 1, 3,
     dict(step=1, options=tuple(b[1] for b in BREATH_BEATS), kind="int",
          rand=(1, 4), shader=False)),
    ("bg_mov_drift", "Dérive", 0.0, 2.0, 0.1,
     dict(kind="phase", rand=(0.0, 0.4), macros={"fluid": 0.3}, shader=False,
          tip="Flottement lent du fond, sans motif visible")),
    ("bg_mov_sync", "Synchro au tempo", 0.0, 1.0, 0.0,
     dict(kind="setting", shader=False,
          tip="1 = les vitesses suivent le BPM (référence 120)")),
    ("bg_mov_inertia", "Inertie", 0.0, 1.0, 0.8,
     dict(kind="setting", macros={"fluid": 0.4}, shader=False,
          tip="Lissage des réactions à la musique")),
    ("bg_mov_music", "Réaction à la musique", 0.0, 1.0, 0.15,
     dict(kind="setting", macros={"energy": 0.3}, shader=False,
          tip="0 = le fond ne fait que son mouvement propre")),
    ("bg_mov_freeze", "Gel du fond", 0, 1, 0,
     dict(check=True, kind="setting", shader=False, tip="Fige le temps du fond")),
)

COMP = _group(
    "comp",
    ("bg_comp_intensity", "Intensité", 0.0, 1.0, 0.35,
     dict(kind="color", audio=("energy", 0.08),
          macros={"daylight": 0.15, "glow": 0.1})),
    ("bg_comp_scale", "Échelle globale", 0.25, 4.0, 1.0,
     dict(law="log", rand=(0.6, 1.8), macros={"scale": 0.7, "depth": -0.3},
          shader=False)),
    ("bg_comp_far", "Éloignement (effet)", 0.0, 0.95, 0.0,
     dict(step=0.01,
          tip="0 = même espace que le modèle (les formes peuvent passer devant\n"
              "lui). Au-dessus : le fond glisse moins quand la caméra bouge,\n"
              "mais il n'est plus dans le même espace (plus d'occlusion)")),
    ("bg_comp_off_x", "Décalage horizontal", -10.0, 10.0, 0.0,
     dict(step=0.05, shader=False)),
    ("bg_comp_off_y", "Décalage vertical", -10.0, 10.0, 0.0,
     dict(step=0.05, shader=False)),
    ("bg_comp_roll", "Inclinaison de l'horizon", -45.0, 45.0, 0.0,
     dict(step=0.5, rand=(-12.0, 12.0), shader=False)),
    ("bg_comp_void", "Vide central", 0.0, 20.0, 6.0,
     dict(step=0.1, macros={"tension": 3.0},
          tip="Rayon sans formes autour du modèle (le modèle a un rayon de 1) :\n"
              "des cellules entières disparaissent, aucune forme n'est tranchée")),
    ("bg_comp_halo", "Halo autour du modèle", -1.0, 1.0, -0.3,
     dict(kind="color", audio=("rms", 0.1),
          tip="−1 = assombrit le fond derrière le modèle, +1 = l'éclaire")),
    ("bg_comp_fog_tint", "Brume teintée par la palette", 0.0, 1.0, 0.6,
     dict(kind="color", rand=(0.2, 1.0), shader=False,
          tip="L'horizon et la brume prennent la couleur moyenne de la palette")),
    ("bg_comp_fog_glow", "Lueur de la lumière dans la brume", 0.0, 2.0, 0.5,
     dict(kind="color", rand=(0.0, 1.2), macros={"glow": 0.3},
          tip="La brume s'éclaire dans la direction de la lumière")),
    ("bg_comp_fade", "Brume de distance", 0.05, 2.0, 0.85,
     dict(kind="color", macros={"depth": -0.25},
          tip="Le lointain se fond dans la couleur d'horizon")),
    ("bg_comp_horizon", "Horizon (distance max)", 20.0, 300.0, 150.0,
     dict(step=1, kind="setting", macros={"depth": 80.0},
          tip="Au-delà, tout est dans la brume ; la marche s'arrête plus tôt\n"
              "quand la brume est opaque (pas de coût inutile)")),
    ("bg_comp_parallax", "Parallaxe", 0.0, 1.0, 1.0,
     dict(kind="setting", shader=False,
          tip="1 = le fond suit la caméra (orbite), 0 = il garde la vue de départ")),
    ("bg_comp_bloom", "Lueur du fond dans le bloom", 0.0, 1.0, 0.3,
     dict(kind="emissive", macros={"glow": 0.3}, shader=False)),
)

SOLID = [
    Param(f"bg_{c}", f"Couleur unie : {name}", "solid", 0, 255, v, step=1,
          kind="setting", shader=False)
    for c, name, v in (("r", "rouge", 4), ("g", "vert", 4), ("b", "bleu", 10))
]

PARAMS = TOP + _macro_params() + NW + SP + DX + SH + FU + COL + MAT + MOV + COMP + SOLID
BY_KEY = {p.key: p for p in PARAMS}
MACRO_KEYS = [f"bg_m_{n}" for n, *_ in MACROS_AMB + MACROS_FORM]
# groupes qui ont un dé et un cadenas
DICE_GROUPS = [g for g, _ in GROUPS if any(p.rand for p in PARAMS if p.group == g)]
LOCK_KEYS = {g: f"bg_lock_{g}" for g in DICE_GROUPS}

# préréglages de matière, appliqués par le panneau (valeurs des sliders)
MATERIALS = {
    "Argile": {"bg_mat_rough": 0.85, "bg_mat_metal": 0.0, "bg_mat_transl": 0.0,
               "bg_mat_emit": 0.0, "bg_mat_fresnel": 0.3, "bg_mat_env": 0.4},
    "Chrome": {"bg_mat_rough": 0.12, "bg_mat_metal": 1.0, "bg_mat_transl": 0.0,
               "bg_mat_emit": 0.0, "bg_mat_fresnel": 1.0, "bg_mat_env": 1.6},
    "Verre": {"bg_mat_rough": 0.05, "bg_mat_metal": 0.0, "bg_mat_transl": 0.8,
              "bg_mat_emit": 0.0, "bg_mat_fresnel": 1.6, "bg_mat_env": 1.3},
    "Néon": {"bg_mat_rough": 0.4, "bg_mat_metal": 0.0, "bg_mat_transl": 0.2,
             "bg_mat_emit": 1.5, "bg_mat_fresnel": 0.5, "bg_mat_env": 0.6},
    "Velours": {"bg_mat_rough": 1.0, "bg_mat_metal": 0.0, "bg_mat_transl": 0.3,
                "bg_mat_emit": 0.0, "bg_mat_fresnel": 1.4, "bg_mat_env": 0.3},
    "Nacre": {"bg_mat_rough": 0.3, "bg_mat_metal": 0.3, "bg_mat_transl": 0.4,
              "bg_mat_emit": 0.0, "bg_mat_fresnel": 1.0, "bg_col_irid": 0.8, "bg_mat_env": 1.0},
}

# ambiances intégrées : presets de l'onglet Fond (T24 : jamais copiés dans
# presets.json) ; les réglages absents reprennent leur défaut
AMBIANCES = {
    "Ferrofluide": {
        "bg_visible": 1.0, "bg_sp_mode": 0.0, "bg_sp_period": 6.0,
        "bg_sp_along": 6.0, "bg_sp_dist": 2.0, "bg_sh_a": 0.0, "bg_sh_size": 0.28,
        "bg_fu_count": 6.0, "bg_fu_k": 0.6, "bg_fu_rad": 0.45, "bg_fu_spread": 1.1,
        "bg_fu_orbit": 0.1, "bg_fu_glow": 0.0, "bg_mat_preset": 6.0,
        "bg_mat_rough": 0.18, "bg_mat_metal": 0.6, "bg_mat_fresnel": 1.2,
        "bg_col_mode": 3.0, "bg_col_a_hue": 0.62, "bg_col_a_lum": 0.6,
        "bg_col_irid": 0.6, "bg_light_shadow": 0.5,
        "bg_comp_fade": 0.3},
    "Cathédrale KIFS": {
        "bg_visible": 1.0, "bg_sp_mode": 5.0, "bg_sp_period": 16.0,
        "bg_sp_along": 12.0, "bg_sp_iter": 3.0, "bg_sp_angle": 12.0,
        "bg_sp_fscale": 2.0, "bg_sp_off_x": 1.0, "bg_sp_off_y": 1.0,
        "bg_sp_off_z": 0.6, "bg_sh_a": 8.0, "bg_sh_size": 0.4, "bg_sh_thick": 0.06,
        "bg_mat_preset": 0.0, "bg_col_mode": 1.0, "bg_col_a_hue": 0.08,
        "bg_col_a_sat": 0.3, "bg_col_b_hue": 0.11, "bg_col_b_sat": 0.25,
        "bg_col_source": 0.0, "bg_mat_edges": 0.6, "bg_light_shadow": 0.6},
    "Récif tordu": {
        "bg_visible": 1.0, "bg_sp_mode": 1.0, "bg_sp_period": 5.0,
        "bg_dx_warp_amp": 1.5, "bg_dx_warp_freq": 0.12, "bg_dx_warp_oct": 2.0,
        "bg_sh_a": 5.0, "bg_sh_b": 0.0, "bg_sh_morph": 0.5, "bg_sh_size": 0.3,
        "bg_sp_disorder": 0.5, "bg_col_mode": 0.0, "bg_col_preset": 6.0,
        "bg_col_source": 4.0, "bg_mat_preset": 4.0, "bg_mat_rough": 1.0,
        "bg_mat_transl": 0.3, "bg_mat_fresnel": 1.4},
    "Hélices": {
        "bg_visible": 1.0, "bg_sp_mode": 3.0, "bg_sp_period": 3.5,
        "bg_sp_along": 2.5, "bg_sp_count": 6.0, "bg_dx_twist": 0.03,
        "bg_sh_a": 6.0, "bg_sh_sz": 2.0, "bg_sh_size": 0.3, "bg_col_mode": 5.0,
        "bg_mat_preset": 3.0, "bg_mat_emit": 1.5, "bg_mat_rough": 0.4,
        "bg_mat_chase": 0.7, "bg_mov_travel": 1.5, "bg_comp_void": 4.0},
    "Kaléidoscope": {
        "bg_visible": 1.0, "bg_sp_mode": 4.0, "bg_sp_period": 4.0,
        "bg_sp_along": 8.0, "bg_sp_mirror_x": 1.0, "bg_sp_mirror_y": 1.0,
        "bg_sp_mirror_z": 1.0, "bg_sp_disorder": 0.6, "bg_sp_missing": 0.4,
        "bg_sh_a": 3.0, "bg_sh_b": 4.0, "bg_sh_morph": 0.3, "bg_sh_spin": 0.03,
        "bg_col_mode": 4.0, "bg_col_source": 4.0, "bg_mat_preset": 5.0,
        "bg_mat_rough": 0.3, "bg_mat_metal": 0.3, "bg_mat_transl": 0.4,
        "bg_col_irid": 0.8},
    "Jardin de gyroïdes": {
        "bg_visible": 1.0, "bg_sp_mode": 2.0, "bg_sp_period": 7.0,
        "bg_sh_a": 9.0, "bg_sh_size": 0.4, "bg_sh_sy": 1.5, "bg_sh_thick": 0.03,
        "bg_sp_eq": 0.4, "bg_dx_wave_amp": 0.12, "bg_dx_wave_len": 14.0,
        "bg_mat_preset": 2.0, "bg_mat_rough": 0.05, "bg_mat_transl": 0.8,
        "bg_mat_fresnel": 1.6, "bg_col_mode": 2.0, "bg_col_source": 0.0},
    "Essaim": {
        "bg_visible": 1.0, "bg_sp_mode": 1.0, "bg_sp_period": 9.0,
        "bg_sp_missing": 0.5, "bg_sh_a": 0.0, "bg_sh_size": 0.12,
        "bg_fu_count": 8.0, "bg_fu_k": 0.15, "bg_fu_spread": 1.8, "bg_fu_rad": 0.5,
        "bg_fu_orbit": 0.3, "bg_fu_glow": 1.5, "bg_dx_wave_amp": 0.15,
        "bg_col_mode": 7.0, "bg_col_source": 4.0, "bg_mat_preset": 3.0,
        "bg_mat_emit": 0.8, "bg_mat_rough": 0.4},
    "Monolithes": {
        "bg_visible": 1.0, "bg_sp_mode": 0.0, "bg_sp_period": 4.0,
        "bg_sp_along": 8.0, "bg_sp_dist": 1.0, "bg_sh_a": 1.0, "bg_sh_size": 0.3,
        "bg_sh_sx": 0.6, "bg_sh_sy": 3.0, "bg_sh_sz": 0.6, "bg_sh_shell": 0.05,
        "bg_fu_count": 4.0, "bg_fu_combo": 1.0, "bg_fu_rad": 0.6,
        "bg_fu_spread": 0.7, "bg_fu_glow": 0.0, "bg_light_mode": 1.0,
        "bg_light_el": 12.0, "bg_light_temp": 3200.0, "bg_light_shadow": 0.7,
        "bg_mat_preset": 0.0, "bg_col_mode": 3.0, "bg_col_a_hue": 0.05,
        "bg_comp_fade": 0.3},
    "Grotte de cristal": {
        "bg_visible": 1.0, "bg_nw_gen": 1.0, "bg_nw_mode": 0.0, "bg_nw_scale": 14.0,
        "bg_nw_fill": 0.5, "bg_nw_warp": 0.8, "bg_nw_detail": 0.4, "bg_nw_fine": 0.12,
        "bg_mat_preset": 2.0, "bg_mat_rough": 0.05, "bg_mat_transl": 0.8,
        "bg_mat_fresnel": 1.6, "bg_mat_env": 1.3, "bg_col_mode": 2.0,
        "bg_col_irid": 0.7, "bg_mat_veins": 1.2, "bg_mat_veins_size": 6.0,
        "bg_col_emit_hue": 0.55, "bg_light_center": 1.5},
    "Entrailles": {
        "bg_visible": 1.0, "bg_nw_gen": 1.0, "bg_nw_mode": 0.0, "bg_nw_scale": 10.0,
        "bg_nw_fill": 0.58, "bg_nw_warp": 1.5, "bg_nw_warp_size": 0.6,
        "bg_nw_detail": 0.3, "bg_nw_breath": 0.12, "bg_nw_flow": 0.2,
        "bg_mat_preset": 4.0, "bg_mat_rough": 0.35, "bg_mat_transl": 0.6,
        "bg_mat_fresnel": 1.2, "bg_col_mode": 7.0, "bg_col_hue": 0.95,
        "bg_mat_bump": 0.6, "bg_mat_bump_size": 0.3, "bg_mat_veins": 0.8,
        "bg_mat_veins_size": 3.0, "bg_col_emit_hue": 0.02},
    "Océan de mercure": {
        "bg_visible": 1.0, "bg_nw_gen": 1.0, "bg_nw_mode": 1.0, "bg_nw_scale": 20.0,
        "bg_nw_warp": 0.9, "bg_nw_gyroid": 0.6, "bg_nw_thick": 0.18,
        "bg_nw_flow": 0.18, "bg_mat_preset": 1.0, "bg_mat_rough": 0.1,
        "bg_mat_metal": 1.0, "bg_mat_fresnel": 1.0, "bg_mat_env": 1.8,
        "bg_col_mode": 3.0, "bg_col_a_hue": 0.6, "bg_col_a_lum": 0.7,
        "bg_mat_bump": 0.0, "bg_mat_grain": 0.0, "bg_col_hor_lum": 0.25},
    "Membranes néon": {
        "bg_visible": 1.0, "bg_nw_gen": 1.0, "bg_nw_mode": 1.0, "bg_nw_scale": 12.0,
        "bg_nw_warp": 1.3, "bg_nw_gyroid": 0.2, "bg_nw_thick": 0.07,
        "bg_nw_detail": 0.5, "bg_mat_preset": 3.0, "bg_mat_transl": 0.7,
        "bg_mat_emit": 0.8, "bg_mat_rough": 0.4, "bg_col_mode": 5.0,
        "bg_col_source": 5.0, "bg_mat_veins": 1.5, "bg_mat_veins_width": 0.03},
    "Planète inconnue": {
        "bg_visible": 1.0, "bg_nw_gen": 1.0, "bg_nw_mode": 2.0, "bg_nw_scale": 30.0,
        "bg_nw_relief": 2.5, "bg_nw_warp": 1.0, "bg_nw_detail": 0.45,
        "bg_nw_fine": 0.15, "bg_mat_preset": 0.0, "bg_col_mode": 1.0,
        "bg_col_a_hue": 0.05, "bg_col_b_hue": 0.55, "bg_col_source": 0.0,
        "bg_light_mode": 1.0, "bg_light_el": 18.0, "bg_light_temp": 4200.0,
        "bg_light_shadow": 0.6, "bg_comp_fade": 0.4, "bg_col_hor_lum": 0.3,
        "bg_comp_fog_glow": 1.2},
    "Canyon voûté": {
        "bg_visible": 1.0, "bg_nw_gen": 1.0, "bg_nw_mode": 2.0, "bg_nw_scale": 18.0,
        "bg_nw_relief": 1.8, "bg_nw_vault": 1.0, "bg_nw_warp": 1.2,
        "bg_nw_detail": 0.4, "bg_mat_preset": 4.0, "bg_mat_rough": 0.9,
        "bg_col_mode": 2.0, "bg_col_source": 6.0, "bg_mat_veins": 1.0,
        "bg_mat_veins_size": 8.0, "bg_col_emit_hue": 0.5, "bg_light_center": 2.0},
}


def defaults() -> dict:
    """Valeurs par défaut de tous les réglages du fond (+ cadenas)."""
    d = {p.key: float(p.default) for p in PARAMS}
    d.update({key: 0.0 for key in LOCK_KEYS.values()})
    return d


def params_of(group: str) -> list:
    return [p for p in PARAMS if p.group == group]


def glsl_declarations() -> str:
    """Les uniforms des paramètres envoyés tels quels au shader."""
    return "".join(f"uniform float {p.uniform};\n" for p in PARAMS if p.shader)


def _check_range(p: Param) -> list:
    errors = []
    if p.kind not in KINDS:
        errors.append(f"classe inconnue {p.kind}")
    if p.law not in ("lin", "log", "angle"):
        errors.append(f"loi inconnue {p.law}")
    if not p.lo <= p.default <= p.hi:
        errors.append("défaut hors plage")
    if p.law == "log" and p.lo <= 0:
        errors.append("loi log avec une borne ≤ 0")
    if p.options is not None and len(p.options) != int(p.hi - p.lo) + 1:
        errors.append("plage ≠ nombre d'options")
    if p.rand is not None and not (p.lo <= p.rand[0] <= p.rand[1] <= p.hi):
        errors.append("plage du dé hors de [min, max]")
    if p.rand is not None and p.kind == "setting":
        errors.append("un réglage ne se tire pas au dé")
    if p.weights is not None and (p.rand is None
                                  or len(p.weights) != int(p.rand[1] - p.rand[0]) + 1):
        errors.append("un poids par valeur de la plage du dé")
    if p.zero > 0.0 and not p.lo <= 0.0 <= p.hi:
        errors.append("extinction (zero) alors que 0 est hors plage")
    errors += [f"macro inconnue {m}" for m in p.macros if f"bg_m_{m}" not in BY_KEY]
    return errors


def _check_audio(p: Param) -> list:
    """Les règles de la musique : rien de rapide sur la structure."""
    if p.audio is None:
        return []
    src = p.audio[0]
    errors = []
    if src not in AUDIO_SOURCES:
        errors.append(f"source audio inconnue {src}")
    if p.kind in SLOW_ONLY and src != "energy":
        errors.append("structure pilotée par une source rapide")
    if src in FAST_SOURCES and p.kind not in FAST_OK:
        errors.append(f"{src} sur un paramètre {p.kind}")
    if p.kind == "setting":
        errors.append("un réglage ne réagit pas à la musique")
    return errors


def validate() -> list:
    """Incohérences de la table (vide = tout va bien) ; vérifié par les tests."""
    errors = []
    keys = [p.key for p in PARAMS]
    errors += [f"{k} : clé en double" for k in set(keys) if keys.count(k) > 1]
    for p in PARAMS:
        if not p.key.startswith("bg_"):
            errors.append(f"{p.key} : la clé doit commencer par bg_")
        errors += [f"{p.key} : {e}" for e in _check_range(p) + _check_audio(p)]
    for key in MACRO_KEYS:
        if not any(key[5:] in p.macros for p in PARAMS):
            errors.append(f"{key} : macro sans effet")
    for name, vals in list(MATERIALS.items()) + list(AMBIANCES.items()):
        for k, v in vals.items():
            p = BY_KEY.get(k)
            if p is None or not p.lo <= v <= p.hi:
                errors.append(f"préréglage {name} : {k} inconnu ou hors plage")
    return errors


def _draw(p: Param, rng: random.Random) -> float:
    a, b = p.rand
    # le premier nombre tiré décide de l'extinction : changer `zero` ne
    # déplace pas les autres tirages de la clé
    if rng.random() < p.zero:
        return 0.0
    if p.options is not None or p.kind == "int":
        values = list(range(int(a), int(b) + 1))
        if p.weights is not None:
            return float(rng.choices(values, weights=p.weights)[0])
        return float(rng.choice(values))
    if p.law == "log":
        v = math.exp(rng.uniform(math.log(a), math.log(b)))
    else:
        v = rng.uniform(a, b)
    return round(round((v - p.lo) / p.step) * p.step + p.lo, 6)


def random_values(seed: int, groups=None, locked=()) -> dict:
    """Tirage du dé : chaque paramètre tiré dans sa plage `rand`, par un
    générateur propre à (seed, clé) → même seed = mêmes valeurs, quel que
    soit le groupe tiré ou l'ordre de la table. Les modes suivent leurs
    poids (`weights`), un effet peut être éteint (`zero`). `groups` limite le tirage ;
    les groupes de `locked` (cadenas) ne sont jamais tirés."""
    out = {}
    for p in PARAMS:
        if p.rand is None or p.group in locked:
            continue
        if groups is not None and p.group not in groups:
            continue
        out[p.key] = _draw(p, random.Random(f"{int(seed)}:{p.key}"))
    return out
