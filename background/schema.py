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
- `setting` : réglage de l'utilisateur, ni musique ni dé ;
- `terrain` : dans les cartes de hauteur précalculées (background/terrain.py) :
  le changer relance le calcul ; ni musique ni glissement, pas d'uniform.
"""

import math
import random
from dataclasses import dataclass, field

KINDS = ("exact", "deform", "topology", "int", "color", "emissive", "phase",
         "setting", "terrain")
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
    ("macro_form", "Macros du paysage"),
    ("ld", "Paysage"),
    ("vo", "Volume : distorsion, cavités, bulles"),
    ("col", "Couleur"),
    ("mat", "Matière et lumière"),
    ("tex", "Texture du relief"),
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
    ("tension", "Tension", "vallée étroite → large"),
    ("fluid", "Fluidité", "raide → liquide"),
    ("symmetry", "Symétrie", "libre → ordonné"),
    ("strange", "Étrangeté", "réaliste → onirique"),
    ("contrast", "Contraste", "doux → dur"),
]
MACROS_FORM = [
    ("complexity", "Complexité", "lisse → détaillé (couches)"),
    ("chaos", "Chaos", "régulier → tourmenté (déformation)"),
    ("scale", "Échelle", "micro → macro"),
    ("relief", "Relief", "plaine → montagnes à crêtes"),
    ("hollow", "Cavités", "plein → creusé (grottes, bulles)"),
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
    only: tuple = None             # (clé d'un menu, valeurs) : réglage propre à ces choix

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
          shader=False, tip="Numéro du dernier tirage « 🎲 Nouveau paysage »"),
]

# ── paysage (passe finale) : le seul générateur ──────────────────────
# Les réglages de classe `terrain` sont dans les cartes de hauteur
# précalculées (background/terrain.py) : les changer relance le calcul
# (≈ 0,4 s, en arrière-plan), ils ne glissent pas et la musique n'y touche pas.
LD = _group(
    "ld",
    ("bg_ld_seed", "Graine du relief", 0, 9999, 0,
     dict(step=1, kind="terrain", rand=(0, 9999), shader=False,
          tip="Un autre tirage du même genre de relief")),
    ("bg_ld_scale", "Taille du monde", 4.0, 60.0, 20.0,
     dict(step=0.5, law="log", kind="topology", rand=(10.0, 35.0),
          macros={"scale": 0.6},
          tip="Taille des grandes formes du relief (unités du fond ; le modèle a\n"
              "un rayon de 1). Ne recalcule rien")),
    ("bg_ld_relief", "Hauteur du relief", 0.2, 5.0, 2.0,
     dict(law="log", kind="topology", rand=(0.8, 4.0), macros={"relief": 0.7},
          tip="En tailles du monde × 0,1 : 1 = collines, 4 = montagnes.\n"
              "Ne recalcule rien")),
    ("bg_ld_octaves", "Couches (octaves)", 1, 10, 6,
     dict(step=1, kind="terrain", rand=(4, 9), macros={"complexity": 3.0},
          shader=False,
          tip="Chaque couche ajoute un relief plus fin (texture). Les couches\n"
              "plus fines qu'≈ 1/128 de la taille du monde sont laissées au\n"
              "relief de surface (onglet Matière)")),
    ("bg_ld_rough", "Rugosité", 0.3, 0.7, 0.5,
     dict(kind="terrain", rand=(0.38, 0.6), shader=False,
          tip="Part de chaque couche dans la suivante : haut = relief accidenté")),
    ("bg_ld_lacun", "Lacunarité", 1.6, 3.0, 2.0,
     dict(kind="terrain", rand=(1.8, 2.5), shader=False,
          tip="Finesse d'une couche à la suivante : haut = détail très fin vite")),
    ("bg_ld_ridges", "Crêtes", 0.0, 1.0, 0.3,
     dict(kind="terrain", rand=(0.0, 1.0), macros={"relief": 0.4}, shader=False,
          tip="Montagnes à arêtes vives")),
    ("bg_ld_erosion", "Érosion", 0.0, 1.0, 0.4,
     dict(kind="terrain", rand=(0.0, 0.9), macros={"fluid": 0.3}, shader=False,
          tip="Pentes et fonds de vallée lisses, sommets détaillés")),
    ("bg_ld_warp", "Déformation", 0.0, 2.0, 0.6,
     dict(kind="terrain", rand=(0.1, 1.5), macros={"chaos": 0.8, "fluid": 0.4},
          shader=False, tip="Le bruit déforme le relief : formes liquides, vallées sinueuses")),
    ("bg_ld_warp_size", "Taille de la déformation", 0.3, 2.0, 0.8,
     dict(kind="terrain", rand=(0.4, 1.5), shader=False, tip="En tailles du monde")),
    ("bg_ld_terrace", "Terrasses", 0.0, 1.0, 0.0,
     dict(kind="terrain", rand=(0.0, 0.8), zero=0.6, macros={"symmetry": 0.4},
          shader=False, tip="Strates : plateaux en gradins")),
    ("bg_ld_terrace_n", "Nombre de terrasses", 2, 16, 6,
     dict(step=1, kind="terrain", rand=(3, 10), shader=False)),
    ("bg_ld_valley", "Largeur de la vallée", 1.5, 30.0, 6.0,
     dict(step=0.1, rand=(3.0, 12.0), macros={"tension": 3.0},
          tip="Demi-largeur du fond plat de la vallée où voyage l'objet (unités)")),
    ("bg_ld_vault", "Voûte (canyon)", 0, 1, 0,
     dict(step=1, check=True, kind="int", rand=(0, 1), weights=(75, 25),
          tip="Un plafond rocheux au-dessus de la vallée")),
)

# ── volume (passe finale, 2e temps) : ce qu'un champ de hauteur ne fait pas ──
# La Distorsion déforme l'espace avant le relief (surplombs, espace tordu),
# les cavités et les bulles creusent ou ajoutent de la matière en 3D. Tout
# s'éteint près du trajet de l'objet (couloir libre), sauf la torsion autour
# du trajet, qui le laisse en place.
CAVE_MODES = ("Creusées dans le relief", "Roche dans tout l'espace")
BUBBLE_MODES = ("Creusées", "Pleines", "Les deux")

VO = _group(
    "vo",
    ("bg_vo_twist", "Torsion (tours / 100 unités)", 0.0, 3.0, 0.0,
     dict(step=0.05, kind="deform", rand=(0.0, 1.5), zero=0.7,
          tip="L'espace tourne autour du trajet en avançant : le sol devient\n"
              "une spirale près du trajet, et redevient un sol au loin")),
    ("bg_vo_bend", "Courbure", -3.0, 3.0, 0.0,
     dict(step=0.05, kind="deform", rand=(-2.0, 2.0), zero=0.6,
          tip="Le relief se relève (ou plonge) en s'éloignant du trajet")),
    ("bg_vo_wave_amp", "Ondes : amplitude", 0.0, 8.0, 0.0,
     dict(step=0.05, kind="deform", rand=(0.0, 5.0), zero=0.6, tip="Unités du fond")),
    ("bg_vo_wave_len", "Ondes : longueur", 4.0, 80.0, 30.0,
     dict(step=0.5, law="log", kind="deform", rand=(10.0, 60.0), tip="Unités du fond")),
    ("bg_vo_warp", "Warp 3D", 0.0, 1.0, 0.0,
     dict(kind="deform", rand=(0.0, 0.7), zero=0.4, macros={"chaos": 0.3},
          tip="Le bruit 3D déplace l'espace : surplombs, formes liquides\n"
              "(en tailles du monde)")),
    ("bg_vo_warp_size", "Warp 3D : taille", 0.3, 3.0, 1.0,
     dict(law="log", kind="deform", rand=(0.5, 2.0), tip="En tailles du monde")),
    ("bg_vo_warp_oct", "Warp 3D : couches", 1, 3, 2,
     dict(step=1, kind="int", rand=(1, 3))),
    ("bg_vo_caves", "Cavités", 0.0, 1.0, 0.0,
     dict(kind="topology", rand=(0.2, 0.9), zero=0.5, macros={"hollow": 0.5},
          tip="Grottes, arches et trous creusés en 3D (0 = aucune)")),
    ("bg_vo_cave_mode", "Cavités : genre", 0, len(CAVE_MODES) - 1, 0,
     dict(step=1, options=CAVE_MODES, kind="int", rand=(0, 1), weights=(65, 35),
          tip="Creusées dans le relief : arches, surplombs, grottes dans les\n"
              "montagnes. Roche dans tout l'espace : une caverne où l'objet\n"
              "voyage dans un tunnel")),
    ("bg_vo_cave_size", "Cavités : taille", 0.2, 2.0, 0.6,
     dict(law="log", kind="topology", rand=(0.3, 1.2), macros={"hollow": 0.4},
          tip="En tailles du monde")),
    ("bg_vo_cave_detail", "Cavités : détail", 0.0, 1.0, 0.4,
     dict(kind="deform", rand=(0.0, 0.8), macros={"complexity": 0.2},
          tip="Parois plus tourmentées (couche de bruit trois fois plus fine)")),
    ("bg_vo_bubbles", "Bulles", 0.0, 1.0, 0.0,
     dict(kind="topology", rand=(0.2, 0.9), zero=0.6, macros={"hollow": 0.3},
          tip="Part des cellules qui reçoivent une bulle (0 = aucune)")),
    ("bg_vo_bubble_mode", "Bulles : genre", 0, len(BUBBLE_MODES) - 1, 0,
     dict(step=1, options=BUBBLE_MODES, kind="int", rand=(0, 2), weights=(45, 30, 25),
          tip="Creusées : alvéoles dans le relief ; pleines : bulles qui se\n"
              "fondent au relief et flottent au-dessus")),
    ("bg_vo_bubble_size", "Bulles : taille", 0.05, 1.0, 0.25,
     dict(law="log", kind="topology", rand=(0.08, 0.5),
          tip="Taille des cellules (en tailles du monde) ; une bulle en occupe\n"
              "au plus 90 %")),
    ("bg_vo_bubble_soft", "Bulles : fusion", 0.0, 1.0, 0.4,
     dict(rand=(0.0, 0.9), tip="Raccord doux, comme un liquide, avec le relief")),
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
    ("bg_col_hor_lum", "Luminosité d'horizon", 0.0, 1.0, 0.3,
     dict(kind="color", rand=(0.12, 0.45), macros={"daylight": 0.35}, shader=False,
          tip="Lueur de l'horizon dans laquelle le lointain se fond : plus\n"
              "claire que le relief, le lointain s'éclaircit (profondeur)")),
    ("bg_col_harmony", "Harmonie avec le modèle", 0, len(HARMONIES) - 1, 0,
     dict(step=1, options=HARMONIES, kind="int", shader=False,
          tip="Cale la teinte de la palette sur le rim du modèle\n"
              "(onglet Objet) : même teinte, complémentaire…")),
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
     dict(kind="color", rand=(0.0, 0.7),
          tip="Fausse diffusion sous la surface : la lumière traverse")),
    ("bg_mat_emit", "Émission propre", 0.0, 4.0, 0.0,
     dict(step=0.05, kind="emissive", rand=(0.0, 1.0), audio=("rms", 0.15),
          macros={"glow": 1.0})),
    ("bg_mat_edges", "Arêtes lumineuses", 0.0, 3.0, 0.3,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), audio=("treble", 0.2),
          macros={"glow": 0.8}, shader=False,
          tip="Les arêtes vives brillent (envoyé au shader comme u_edges)")),
    ("bg_mat_glow", "Halo des formes", 0.0, 2.0, 0.3,
     dict(kind="emissive", rand=(0.0, 1.0), macros={"glow": 0.8, "energy": 0.3},
          tip="Lueur autour des formes (frôlements du rayon)")),
    ("bg_mat_fresnel", "Fresnel / contre-jour", 0.0, 2.0, 0.5,
     dict(kind="color", rand=(0.0, 1.5), audio=("treble", 0.1))),
    ("bg_mat_ao", "Occlusion des creux", 0.0, 2.0, 1.0,
     dict(kind="color", rand=(0.5, 1.5), macros={"contrast": 0.5})),
    ("bg_mat_env", "Reflets de l'environnement", 0.0, 2.0, 1.0,
     dict(kind="color", rand=(0.3, 1.5),
          tip="Les formes reflètent le ciel et l'horizon, nets si la matière\n"
              "est lisse, flous si elle est rugueuse")),
    ("bg_mat_rim", "Contre-jour", 0.0, 2.0, 0.6,
     dict(kind="color", rand=(0.0, 1.5), macros={"glow": 0.3},
          tip="Le bord des formes s'éclaire de la couleur de la brume :\n"
              "elles se détachent du fond")),
    ("bg_mat_bump", "Relief de surface", 0.0, 1.0, 0.3,
     dict(kind="color", rand=(0.0, 0.8),
          tip="Bosses fines dessinées par la lumière (ne coûte presque rien)")),
    ("bg_mat_bump_size", "Taille du relief", 0.05, 2.0, 0.4,
     dict(step=0.01, law="log", kind="color", rand=(0.1, 1.0),
          tip="Unités du fond (le modèle a un rayon de 1)")),
    ("bg_mat_grain", "Grain de la matière", 0.0, 1.0, 0.25,
     dict(kind="color", rand=(0.0, 0.6), tip="Couleur marbrée, comme une pierre")),
    ("bg_mat_veins", "Veines lumineuses", 0.0, 3.0, 0.0,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), zero=0.6,
          audio=("energy", 0.3), macros={"glow": 0.6},
          tip="Des veines de la couleur des émissifs parcourent le relief (figées)")),
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
    ("bg_light_shadow", "Dureté des ombres", 0.0, 1.0, 0.6,
     dict(kind="color", rand=(0.0, 1.0), macros={"contrast": 0.4},
          tip="Ombres portées du relief et de l'objet sur le relief.\n"
              "0 = pas d'ombre portée (plus léger), 1 = ombres nettes")),
    ("bg_light_center", "Le modèle éclaire le fond", 0.0, 4.0, 0.5,
     dict(step=0.05, kind="emissive", rand=(0.0, 2.0), audio=("rms", 0.2),
          shader=False, tip="Lumière au centre de la scène, couleur du rim du modèle")),
    ("bg_light_flash", "Flash de lumière", 0.0, 1.0, 0.0,
     dict(kind="emissive", rand=(0.0, 0.6), shader=False,
          tip="Le soleil s'allume sur la caisse claire")),
)

# ── texture du relief (T29) : une décoration peinte sur le relief, fixe ──
# Le motif dit quelle forme dessine la texture, rendue en chrome
# (main.glsl) ; les réglages communs déforment le motif selon le relief
# (comme l'outil de réaction-diffusion de Karl Sims). Ni musique ni
# macros : la texture est collée au décor (validate()). Motif « Aucun » =
# la matière du groupe « Matière et lumière », image d'avant T29.
TEX_MOTIFS = ("Aucun", "Volutes", "Vermicelles", "Coulures", "Gouttes", "Cellules",
              "Strates", "Isolignes", "Veines")
TEX_ORIENTS = ("Selon les courbes de niveau", "Fixe")
TEX_FIELDS = ("Aucun", "Altitude", "Pente", "Creux", "Crêtes")
TEX_ESSENTIAL = ("bg_tex_scale", "bg_tex_density", "bg_tex_width", "bg_tex_sharp",
                 "bg_tex_shine")

TEX = _group(
    "tex",
    ("bg_tex_motif", "Motif", 0, len(TEX_MOTIFS) - 1, 0,
     dict(step=1, options=TEX_MOTIFS, kind="int", rand=(0, 8),
          weights=(20, 10, 10, 10, 10, 10, 10, 10, 10),
          tip="La forme que la texture dessine sur le relief.\n"
              "Aucun = la matière du groupe « Matière et lumière »")),
    # essentiel
    ("bg_tex_scale", "Échelle", 0.3, 20.0, 3.0,
     dict(law="log", kind="color", rand=(1.0, 8.0),
          tip="Taille du motif (unités du fond ; le modèle a un rayon de 1)")),
    ("bg_tex_density", "Densité", 0.5, 4.0, 1.5,
     dict(law="log", kind="color", rand=(0.8, 3.0),
          tip="Nombre de traits ou de volutes dans une forme du motif")),
    ("bg_tex_width", "Épaisseur", 0.05, 0.95, 0.5,
     dict(kind="color", rand=(0.25, 0.75),
          tip="Part des formes face à l'entre-formes : traits fins ↔ épais")),
    ("bg_tex_sharp", "Netteté des bords", 0.0, 1.0, 0.6,
     dict(kind="color", rand=(0.3, 1.0), tip="Bord flou ↔ tranché")),
    ("bg_tex_shine", "Éclat", 0.0, 1.0, 0.8,
     dict(kind="color", rand=(0.5, 1.0), tip="Chrome mat ↔ poli comme un miroir")),
    # échelle variable
    # (la densité varie, pas la taille des formes : une taille variable
    # déchirerait le motif loin de l'origine du décor)
    ("bg_tex_scale_alt", "Densité selon l'altitude", -1.0, 1.0, 0.0,
     dict(kind="color", rand=(-0.6, 0.6), zero=0.4,
          tip="> 0 : traits plus serrés en haut ; < 0 : plus serrés en bas")),
    ("bg_tex_scale_slope", "Densité selon la pente", -1.0, 1.0, 0.0,
     dict(kind="color", rand=(-0.6, 0.6), zero=0.4,
          tip="> 0 : traits plus serrés sur les pentes ; < 0 : plus lâches")),
    ("bg_tex_scale_rand", "Densité aléatoire", 0.0, 1.0, 0.0,
     dict(kind="color", rand=(0.0, 0.6), zero=0.4,
          tip="La densité varie par grandes nappes de bruit")),
    # flux : déformation du motif par le relief
    ("bg_tex_flow_slope", "Flux : suivre la pente", -2.0, 2.0, 0.8,
     dict(kind="color", rand=(-1.5, 1.5),
          tip="Le motif est entraîné dans le sens de la pente : il coule")),
    ("bg_tex_flow_iso", "Flux : suivre les courbes de niveau", -2.0, 2.0, 0.0,
     dict(kind="color", rand=(-1.5, 1.5), zero=0.3,
          tip="Le motif s'enroule autour des collines")),
    ("bg_tex_flow_swirl", "Flux : tourbillons", 0.0, 2.0, 0.5,
     dict(kind="color", rand=(0.0, 1.5), tip="Turbulence en spirales")),
    ("bg_tex_flow_bubble", "Flux : bulles", -2.0, 2.0, 0.0,
     dict(kind="color", rand=(-1.5, 1.5), zero=0.5,
          tip="Le motif gonfle autour des sommets (> 0) ou des creux (< 0)")),
    # orientation
    ("bg_tex_stretch", "Étirement", 0.0, 4.0, 0.0,
     dict(kind="color", rand=(0.0, 3.0), zero=0.3, tip="Motif rond ↔ en longs traits")),
    ("bg_tex_orient", "Direction de l'étirement", 0, len(TEX_ORIENTS) - 1, 0,
     dict(step=1, options=TEX_ORIENTS, kind="int", rand=(0, 1), weights=(70, 30),
          tip="Les traits suivent les courbes de niveau du relief, ou une\n"
              "direction fixe (angle ci-dessous)")),
    ("bg_tex_angle", "Angle (direction fixe)", 0.0, 180.0, 0.0,
     dict(step=1, law="angle", kind="color", rand=(0.0, 180.0),
          only=("bg_tex_orient", (1,)))),
    ("bg_tex_disorder", "Désordre de l'orientation", 0.0, 1.0, 0.2,
     dict(kind="color", rand=(0.0, 0.8), tip="Les traits ondulent au lieu d'aller droit")),
    # variation de la forme
    ("bg_tex_vary", "Variation selon", 0, len(TEX_FIELDS) - 1, 0,
     dict(step=1, options=TEX_FIELDS, kind="int", rand=(0, 4), weights=(30, 20, 20, 15, 15),
          tip="Le champ du relief qui fait varier l'épaisseur des traits")),
    ("bg_tex_vary_amt", "Force de la variation", -1.0, 1.0, 0.5,
     dict(kind="color", rand=(-1.0, 1.0))),
    # couleurs (la palette, la teinte, la saturation et le contraste sont
    # ceux du groupe Couleur)
    ("bg_tex_invert", "Inverser les couleurs", 0, 1, 0,
     dict(step=1, check=True, kind="int", rand=(0, 1))),
    ("bg_tex_freq", "Fréquence de la palette", 0.5, 4.0, 1.0,
     dict(law="log", kind="color", rand=(0.5, 2.5),
          tip="Combien de fois la palette se répète sur le motif")),
    ("bg_tex_shift", "Décalage dans la palette", 0.0, 1.0, 0.0,
     dict(law="angle", kind="color", rand=(0.0, 1.0))),
    ("bg_tex_bright", "Luminosité", 0.2, 2.0, 1.0,
     dict(law="log", kind="color", rand=(0.7, 1.4))),
    # style
    ("bg_tex_metal", "Teinte du métal", -1.0, 1.0, 0.0,
     dict(kind="color", rand=(-1.0, 1.0), tip="−1 = acier froid … +1 = or chaud")),
    ("bg_tex_strips", "Reflets : bandes", 0, 8, 3,
     dict(step=1, kind="int", rand=(0, 6),
          tip="Bandes de lumière du studio que la surface reflète")),
    ("bg_tex_strip_sharp", "Reflets : netteté", 0.0, 1.0, 0.6,
     dict(kind="color", rand=(0.2, 1.0))),
    ("bg_tex_strip_int", "Reflets : intensité", 0.0, 4.0, 2.0,
     dict(kind="color", rand=(1.0, 3.0))),
    ("bg_tex_horizon", "Reflets : ligne d'horizon", 0.0, 2.0, 0.8,
     dict(kind="color", rand=(0.0, 1.5))),
    ("bg_tex_strip_orient", "Reflets : orientation", 0.0, 360.0, 0.0,
     dict(step=1, law="angle", kind="color", rand=(0.0, 360.0))),
    ("bg_tex_smooth", "Lissage du relief", 0.0, 1.0, 0.7,
     dict(kind="color", rand=(0.4, 1.0),
          tip="Gomme le relief fin du terrain : reflets plus nets")),
    ("bg_tex_emboss", "Gravure", -1.0, 1.0, 0.0,
     dict(kind="color", rand=(-0.6, 0.6), zero=0.5,
          tip="Le motif en creux (< 0) ou en relief (> 0) à la lumière ;\n"
              "0 = texture plate")),
)

MOV = _group(
    "mov",
    ("bg_mov_speed", "Vitesse", 0.05, 4.0, 1.0,
     dict(law="log", kind="phase", rand=(0.4, 2.0), shader=False,
          tip="Multiplie les mouvements propres du fond (rotation, dérive,\n"
              "respiration) ; le relief, lui, reste fixe")),
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
    ("bg_mov_breath", "Respiration", 0.0, 0.3, 0.0,
     dict(step=0.005, shader=False,
          tip="Le fond grandit et rétrécit au tempo")),
    ("bg_mov_breath_beats", "Respiration : période", 0, len(BREATH_BEATS) - 1, 3,
     dict(step=1, options=tuple(b[1] for b in BREATH_BEATS), kind="int",
          rand=(1, 4), shader=False)),
    ("bg_mov_drift", "Dérive", 0.0, 2.0, 0.0,
     dict(kind="phase", shader=False,
          tip="Flottement lent du fond, sans motif visible (0 = décor fixe)")),
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
          tip="Le lointain se fond dans la couleur d'horizon ; la brume est\n"
              "plus dense au fond des vallées")),
    ("bg_comp_fog_height", "Hauteur de la brume", 0.2, 4.0, 1.0,
     dict(law="log", kind="color", rand=(0.5, 2.0),
          tip="En hauteurs du relief : bas = brume couchée au fond des vallées,\n"
              "sommets nets ; haut = brume partout")),
    ("bg_comp_horizon", "Horizon (distance max)", 20.0, 400.0, 220.0,
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
] + _group(
    # grille posée sur le relief (« Grille du décor » cochée, fond visible) ;
    # fond masqué, c'est l'ancienne grille plate qui est dessinée
    "solid",
    ("bg_grid_size", "Grille : taille des mailles", 0.5, 20.0, 3.0,
     dict(step=0.1, kind="setting",
          tip="Unités du fond (le modèle a un rayon de 1) ;\n"
              "les lignes suivent le relief et le trajet")),
    ("bg_grid_width", "Grille : épaisseur (pixels)", 0.5, 4.0, 1.2,
     dict(step=0.1, kind="setting")),
    ("bg_grid_int", "Grille : intensité", 0.0, 4.0, 1.0,
     dict(step=0.05, kind="setting", shader=False)),
    ("bg_grid_hue", "Grille : teinte", 0.0, 1.0, 0.64,
     dict(law="angle", kind="setting", shader=False)),
)

PARAMS = TOP + _macro_params() + LD + VO + COL + MAT + TEX + MOV + COMP + SOLID
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
    "Planète inconnue": {
        "bg_visible": 1.0, "bg_ld_scale": 30.0, "bg_ld_relief": 3.0,
        "bg_ld_warp": 1.0, "bg_ld_octaves": 7.0, "bg_ld_erosion": 0.4,
        "bg_mat_preset": 0.0, "bg_col_mode": 1.0,
        "bg_col_a_hue": 0.05, "bg_col_b_hue": 0.55, "bg_col_source": 0.0,
        "bg_light_mode": 1.0, "bg_light_el": 18.0, "bg_light_temp": 4200.0,
        "bg_light_shadow": 0.6, "bg_comp_fade": 0.4, "bg_col_hor_lum": 0.3,
        "bg_comp_fog_glow": 1.2},
    "Canyon voûté": {
        "bg_visible": 1.0, "bg_ld_scale": 18.0, "bg_ld_relief": 2.6,
        "bg_ld_vault": 1.0, "bg_ld_warp": 1.2, "bg_ld_erosion": 0.6,
        "bg_ld_terrace": 0.3, "bg_mat_preset": 4.0, "bg_mat_rough": 0.9,
        "bg_col_mode": 2.0, "bg_col_source": 6.0, "bg_mat_veins": 1.0,
        "bg_mat_veins_size": 8.0, "bg_col_emit_hue": 0.5, "bg_light_center": 2.0},
    "Montagnes à crêtes": {
        "bg_visible": 1.0, "bg_ld_scale": 28.0, "bg_ld_relief": 4.0,
        "bg_ld_octaves": 8.0, "bg_ld_rough": 0.52, "bg_ld_lacun": 2.1,
        "bg_ld_ridges": 0.9, "bg_ld_erosion": 0.7, "bg_ld_warp": 0.4,
        "bg_ld_valley": 8.0, "bg_mat_preset": 0.0, "bg_col_mode": 3.0,
        "bg_col_a_hue": 0.6, "bg_col_a_lum": 0.6, "bg_col_source": 0.0,
        "bg_light_shadow": 0.8, "bg_col_hor_lum": 0.22, "bg_comp_fog_tint": 0.4},
    "Dunes": {
        "bg_visible": 1.0, "bg_ld_scale": 14.0, "bg_ld_relief": 1.3,
        "bg_ld_octaves": 4.0, "bg_ld_rough": 0.35, "bg_ld_lacun": 2.4,
        "bg_ld_ridges": 0.7, "bg_ld_erosion": 0.0, "bg_ld_warp": 1.4,
        "bg_ld_warp_size": 1.6, "bg_mat_preset": 4.0, "bg_mat_rough": 1.0,
        "bg_col_mode": 1.0, "bg_col_a_hue": 0.07, "bg_col_a_sat": 0.6,
        "bg_col_a_lum": 0.5, "bg_col_b_hue": 0.1, "bg_col_b_sat": 0.4,
        "bg_col_b_lum": 1.0, "bg_col_source": 2.0, "bg_light_mode": 1.0,
        "bg_light_el": 14.0, "bg_light_temp": 3800.0, "bg_light_shadow": 0.9,
        "bg_mat_bump": 0.15, "bg_col_hor_hue": 0.07, "bg_col_hor_lum": 0.25},
    "Plateaux en terrasses": {
        "bg_visible": 1.0, "bg_ld_scale": 22.0, "bg_ld_relief": 2.8,
        "bg_ld_octaves": 6.0, "bg_ld_terrace": 0.85, "bg_ld_terrace_n": 7.0,
        "bg_ld_erosion": 0.3, "bg_ld_warp": 0.7, "bg_mat_preset": 0.0,
        "bg_col_mode": 2.0, "bg_col_source": 0.0, "bg_mat_grain": 0.4,
        "bg_light_shadow": 0.7},
    "Canyon érodé": {
        "bg_visible": 1.0, "bg_ld_scale": 20.0, "bg_ld_relief": 3.6,
        "bg_ld_valley": 9.0, "bg_ld_octaves": 7.0, "bg_ld_erosion": 1.0,
        "bg_ld_ridges": 0.5, "bg_ld_terrace": 0.35, "bg_ld_terrace_n": 10.0,
        "bg_ld_warp": 0.9, "bg_mat_preset": 4.0, "bg_mat_rough": 1.0,
        "bg_col_mode": 0.0, "bg_col_preset": 2.0, "bg_col_source": 6.0,
        "bg_light_shadow": 0.8},
    "Collines": {
        "bg_visible": 1.0, "bg_ld_scale": 26.0, "bg_ld_relief": 1.2,
        "bg_ld_octaves": 6.0, "bg_ld_rough": 0.45, "bg_ld_ridges": 0.0,
        "bg_ld_erosion": 0.2, "bg_ld_warp": 0.5, "bg_ld_valley": 5.0,
        "bg_mat_preset": 0.0, "bg_col_mode": 1.0, "bg_col_a_hue": 0.28,
        "bg_col_b_hue": 0.18, "bg_light_mode": 1.0, "bg_light_el": 35.0,
        "bg_light_temp": 5800.0, "bg_light_shadow": 0.6, "bg_col_hor_lum": 0.3,
        "bg_light_sky": 0.8},
    "Grotte de cristal": {
        "bg_visible": 1.0, "bg_ld_scale": 16.0, "bg_ld_relief": 2.4,
        "bg_ld_vault": 1.0, "bg_ld_octaves": 7.0, "bg_ld_erosion": 0.5,
        "bg_vo_caves": 0.7, "bg_vo_cave_mode": 1.0, "bg_vo_cave_size": 0.8,
        "bg_vo_warp": 0.3, "bg_mat_preset": 2.0, "bg_mat_rough": 0.05,
        "bg_mat_transl": 0.8, "bg_mat_fresnel": 1.6, "bg_mat_env": 1.3,
        "bg_col_mode": 2.0, "bg_col_irid": 0.7, "bg_mat_veins": 1.2,
        "bg_mat_veins_size": 6.0, "bg_col_emit_hue": 0.55, "bg_light_center": 1.5},
    "Arches et alvéoles": {
        "bg_visible": 1.0, "bg_ld_scale": 22.0, "bg_ld_relief": 3.0,
        "bg_ld_octaves": 7.0, "bg_ld_ridges": 0.4, "bg_ld_erosion": 0.6,
        "bg_vo_caves": 0.8, "bg_vo_cave_size": 0.5, "bg_vo_cave_detail": 0.5,
        "bg_vo_bubbles": 0.6, "bg_vo_bubble_mode": 0.0, "bg_vo_bubble_size": 0.2,
        "bg_vo_warp": 0.4, "bg_mat_preset": 0.0, "bg_col_mode": 0.0,
        "bg_col_preset": 4.0, "bg_col_source": 6.0, "bg_light_shadow": 0.8},
    "Entrailles": {
        "bg_visible": 1.0, "bg_ld_scale": 12.0, "bg_ld_relief": 2.0,
        "bg_ld_vault": 1.0, "bg_ld_warp": 1.4, "bg_vo_warp": 0.6,
        "bg_vo_warp_size": 0.8, "bg_vo_twist": 0.6, "bg_vo_bubbles": 0.7,
        "bg_vo_bubble_mode": 2.0, "bg_vo_bubble_size": 0.3, "bg_vo_bubble_soft": 0.8,
        "bg_mat_preset": 4.0, "bg_mat_rough": 0.35, "bg_mat_transl": 0.6,
        "bg_mat_fresnel": 1.2, "bg_col_mode": 7.0, "bg_col_hue": 0.95,
        "bg_mat_bump": 0.6, "bg_mat_bump_size": 0.3, "bg_mat_veins": 0.8,
        "bg_mat_veins_size": 3.0, "bg_col_emit_hue": 0.02},
}


# recettes de la texture du relief (T29) : un paysage ci-dessus + un motif
# et ses réglages (le reste de la texture à son défaut)
TEX_AMBIANCES = {
    "Chrome fondu": ("Dunes", {
        "bg_tex_motif": 1.0, "bg_tex_scale": 4.0, "bg_tex_flow_swirl": 0.8,
        "bg_tex_shine": 0.95, "bg_tex_metal": -0.4}),
    "Coulées de métal": ("Canyon érodé", {
        "bg_tex_motif": 3.0, "bg_tex_scale": 2.5, "bg_tex_density": 2.0,
        "bg_tex_flow_swirl": 0.2, "bg_tex_metal": 0.6}),
    "Cartes d'altitude": ("Collines", {
        "bg_tex_motif": 7.0, "bg_tex_density": 2.5, "bg_tex_width": 0.6,
        "bg_tex_strip_int": 2.6}),
    "Ruche chromée": ("Canyon voûté", {
        "bg_tex_motif": 5.0, "bg_tex_scale": 2.0, "bg_tex_width": 0.75,
        "bg_tex_shine": 0.9, "bg_tex_metal": 0.3}),
    "Corail": ("Dunes", {
        "bg_tex_motif": 2.0, "bg_tex_scale": 2.0, "bg_tex_density": 1.6,
        "bg_tex_disorder": 0.5, "bg_tex_metal": 0.8, "bg_tex_sharp": 0.85}),
}
AMBIANCES.update({name: {**AMBIANCES[base], **tex}
                  for name, (base, tex) in TEX_AMBIANCES.items()})


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


def _check_kind(p: Param) -> list:
    errors = []
    if p.kind not in KINDS:
        errors.append(f"classe inconnue {p.kind}")
    if p.rand is not None and p.kind == "setting":
        errors.append("un réglage ne se tire pas au dé")
    if p.kind == "terrain" and p.shader:
        errors.append("relief précalculé : pas d'uniform")
    return errors


def _check_range(p: Param) -> list:
    errors = []
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
    if p.kind in ("setting", "terrain"):
        errors.append(f"un paramètre {p.kind} ne réagit pas à la musique")
    return errors


def _check_texture(p: Param) -> list:
    """La texture du relief est fixe (T29) : ni musique ni macros ; un
    réglage propre (`only`) vise un menu de la texture et ses valeurs."""
    errors = []
    if p.key.startswith("bg_tex_") and p.audio is not None:
        errors.append("la texture est fixe : pas de musique")
    if p.key.startswith("bg_tex_") and p.macros:
        errors.append("la texture est fixe : pas de macros")
    if p.only is not None:
        menu = BY_KEY.get(p.only[0])
        if menu is None or menu.options is None:
            errors.append(f"only : {p.only[0]} n'est pas un menu")
        elif any(not menu.lo <= v <= menu.hi for v in p.only[1]):
            errors.append("only : valeur hors du menu")
    return errors


def validate() -> list:
    """Incohérences de la table (vide = tout va bien) ; vérifié par les tests."""
    errors = []
    keys = [p.key for p in PARAMS]
    errors += [f"{k} : clé en double" for k in set(keys) if keys.count(k) > 1]
    for p in PARAMS:
        if not p.key.startswith("bg_"):
            errors.append(f"{p.key} : la clé doit commencer par bg_")
        errors += [f"{p.key} : {e}" for e in _check_kind(p) + _check_range(p) + _check_audio(p)
                   + _check_texture(p)]
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

