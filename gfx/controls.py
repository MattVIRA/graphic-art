"""
Control panel — a second window with live controls, organized in tabs.

Tkinter must run in a process' main thread on macOS, so the panel lives in its
own process (multiprocessing) and streams messages back through a Queue:
    ("param", key, float_value)   slider / checkbox change
    ("action", name, payload)     e.g. ("action", "load_visual", "/path/img.png")
The main loop polls `ControlPanel.poll()` once per frame.

T8: a second queue goes the other way (main → panel) with small monitoring
packets (~15 Hz): band levels, rms, bpm, fps, source label. The panel draws
VU meters in the Physique tab and a status bar at the bottom.
"""

import json
import multiprocessing as mp
import os
import queue
import random

from gfx.matcaps import NAMES as MATCAP_NAMES
from visuals.mesh import UP_AXES as MESH_UP_AXES

# presets.json at the project root; "_autosave" is a reserved hidden entry,
# "_window" stores the panel geometry (T7)
PRESETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets.json")


def _load_presets_file() -> dict:
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as fh:
            d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_presets_file(d: dict):
    try:
        with open(PRESETS_PATH, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[warning] preset save failed: {e}")


# Presets propres à un onglet (Vidéo, Physique) : {onglet: {nom: {clé: val}}}
# sous la clé réservée "_tab_presets". Les presets codés en dur ne servent que
# de graines à la première ouverture, ensuite tout est modifiable.
TAB_PRESETS_KEY = "_tab_presets"


def _load_tab_presets(tab: str, builtin: dict) -> dict:
    d = _load_presets_file()
    store = d.get(TAB_PRESETS_KEY)
    if not isinstance(store, dict):
        store = d[TAB_PRESETS_KEY] = {}
    if not isinstance(store.get(tab), dict):
        store[tab] = {n: dict(v) for n, v in builtin.items()}
        _save_presets_file(d)
    return store[tab]


def _save_tab_presets(tab: str, presets: dict):
    d = _load_presets_file()
    if not isinstance(d.get(TAB_PRESETS_KEY), dict):
        d[TAB_PRESETS_KEY] = {}
    d[TAB_PRESETS_KEY][tab] = presets
    _save_presets_file(d)


DEFAULTS = {
    # ── Vidéo ──
    "grain":       0.0,    # film grain amount            0 .. 0.5
    "red":         1.0,    # per-channel gain             0 .. 2
    "green":       1.0,
    "blue":        1.0,
    "contrast":    1.0,    # post contrast                0.3 .. 2.5
    "negative":    0.0,    # invert mix                   0 .. 1
    "brightness":  1.0,    # pre-tonemap exposure         0.2 .. 2.5
    "bloom":       0.9,    # bloom intensity              0 .. 3
    "bloom_radius": 0.5,   # T6: 0 = glow serré, 1 = larges halos
    "bloom_threshold": 0.8,  # luminance (linéaire) où le glow commence
    "vignette":    0.25,   # assombrit les bords de l'image   0 .. 1
    "render_scale": 1.5,   # T6: densité de rendu (px internes / px logique) ;
                           # sans effet sur Retina (déjà 2×), MSAA 4× en plus
    "trail":       0.30,   # fade per frame (high = short trails)
    "point_size":  1.0,    # particle size multiplier
    "zoom":        1.0,    # camera zoom (also mouse wheel)   0.3 .. 3
    "reactivity":  1.0,    # T4: release scaling — high = percussive, low = floaty
    "show_grid":   1.0,    # 1 = draw the static world grid
    # ── Objet : modèle 3D (T19) ──
    "turbulence":  1.0,    # dose of the routed Turbulence effect
    "n_points":    1000.0, # point budget in thousands (style Points)
    "model_style": 1.0,    # index in MODEL_STYLES (1 = Surface)
    "model_up":    0.0,    # index in visuals.mesh.UP_AXES
    "shading":     0.0,    # index in SHADING_MODES (0 = Studio PBR)
    "matcap":      0.0,    # index in gfx.matcaps.NAMES (shading = Matcap)
    "roughness":   0.45,   # studio: 0 = miroir, 1 = mat
    "metal":       0.0,    # studio: 0 = diélectrique, 1 = métal
    "ao":          1.0,    # occlusion des creux (cavités)       0 .. 2
    "wire_width":  1.2,    # épaisseur des traits du wireframe (px)
    "inflate":     0.0,    # constant inflation along the normals   0 .. 2
                           # (0 = model untouched; animate via Physique)
    "rim":         1.0,    # fresnel rim intensity                  0 .. 3
    "rim_hue":     0.52,   # rim color hue                          0 .. 1
    "orig_color":  1.0,    # 0 = gris neutre / matcap pur, 1 = couleurs du modèle
    "mesh_transparent": 0.0,   # surface drawn additive (hologram)
    "wire_sharp":  0.0,    # wireframe: sharp edges only
    "wire_hidden": 0.0,    # wireframe: hide the edges behind the surface
    # ── Rotation (fixed scene parameters — music only drives the spin) ──
    "rot_gain":     1.0,   # spin speed multiplier         0 .. 3
    "axis_tilt":    15.0,  # rotation axis tilt from vertical (deg)  0 .. 90
    "axis_azimuth": 0.0,   # rotation axis azimuth (deg)             0 .. 360
    "orbit_dist":   0.0,   # object distance to the axis (px) → orbit radius
    "obj_tilt":     10.0,  # object's own inclination (deg)         -60 .. 60
    "show_axis":    0.0,   # 1 = draw the rotation axis (green line)
    "axis_spin":       0.0,   # 1 = the axis itself precesses around the world's
                              #     vertical axis at the origin (0,0) of the scene
    "axis_spin_speed": 20.0,  # precession speed (deg/s), negative = reverse
}

# ── Physique : routing bandes de fréquences → effets ──────────────────
EQ_BANDS = ["SUB", "BASS", "LOW MID", "MID", "HIGH MID", "PRESENCE", "TREBLE",
            "BEAT",     # impulsion sur chaque beat, décroissante (T2)
            "KICK",     # ressort amorti lancé par chaque grosse caisse (rebond)
            "SNARE",    # impulsion courte sur caisse claire / clap
            "ÉNERGIE"]  # énergie de la section : ~0 dans les breaks, ~1 sur un drop

EFFECTS = [
    "Aucun",
    "Vitesse rotation",
    "Décalage rouge",
    "Décalage vert",
    "Décalage bleu",
    "Trainées",
    "Zoom (pulse)",
    "Turbulence",
    "Luminosité",
    "Taille points",
    "Gonflement",     # T19: 3D model inflation along the normals
    "Lumière",        # 3D model: intensity of the studio lights
    "Rim",            # 3D model: fresnel rim boost
    "Secousse caméra",
]

DEFAULTS["dynamics"] = 0.0   # 0 = off, 1 = effets très calmes dans les breaks

for _i in range(len(EQ_BANDS)):
    DEFAULTS[f"band{_i}_fx"] = 0.0    # effect index in EFFECTS (0 = Aucun)
    DEFAULTS[f"band{_i}_amt"] = 1.0   # effect amount 0..2


def _preset(mapping):
    """mapping: {band_index: (effect_name, amount)} → param dict."""
    out = {}
    for i in range(len(EQ_BANDS)):
        fx, amt = mapping.get(i, ("Aucun", 1.0))
        out[f"band{i}_fx"] = float(EFFECTS.index(fx))
        out[f"band{i}_amt"] = amt
    return out


PHYSICS_PRESETS = {
    # neutral: no routing (only the music-driven rotation speed)
    "Défaut": _preset({}),
    "Chromatique": _preset({
        1: ("Décalage rouge", 1.2),      # BASS
        3: ("Décalage vert", 0.8),       # MID
        6: ("Décalage bleu", 1.2),       # TREBLE
        4: ("Vitesse rotation", 1.0),    # HIGH MID
    }),
    "Psyché": _preset({
        0: ("Trainées", 1.5),            # SUB
        1: ("Zoom (pulse)", 1.0),        # BASS
        5: ("Turbulence", 1.0),          # PRESENCE
        6: ("Décalage bleu", 0.8),       # TREBLE
    }),
    # 3D model, "music video" look: the kick makes the model breathe with a
    # spring rebound, the snare flashes the rim, the section drives the lights
    "Studio": _preset({
        0: ("Zoom (pulse)", 0.25),       # SUB
        4: ("Vitesse rotation", 0.6),    # HIGH MID
        8: ("Gonflement", 0.6),          # KICK
        9: ("Rim", 1.0),                 # SNARE
        10: ("Lumière", 0.7),            # ÉNERGIE
    }),
    "Drum machine": _preset({
        0: ("Zoom (pulse)", 1.5),        # SUB
        1: ("Taille points", 1.2),       # BASS
        4: ("Vitesse rotation", 1.5),    # HIGH MID
        6: ("Turbulence", 1.0),          # TREBLE
    }),
}

# ── Vidéo : sliders regroupés par LabelFrame (T7) ──────────────────────
VIDEO_GROUPS = [
    ("Couleur", [
        ("Rouge",      "red",        0.0, 2.0,  0.01, None),
        ("Vert",       "green",      0.0, 2.0,  0.01, None),
        ("Bleu",       "blue",       0.0, 2.0,  0.01, None),
        ("Contraste",  "contrast",   0.3, 2.5,  0.01, None),
        ("Négatif",    "negative",   0.0, 1.0,  0.01, None),
        ("Luminosité", "brightness", 0.2, 2.5,  0.01, None),
    ]),
    ("Trainées & bloom", [
        ("Bloom",           "bloom",        0.0,  3.0, 0.05, None),
        ("Rayon bloom",     "bloom_radius", 0.0,  1.0, 0.05,
         "0 = glow serré, 1 = larges halos doux"),
        ("Seuil bloom",     "bloom_threshold", 0.0, 2.0, 0.05,
         "Seules les zones plus lumineuses que ce seuil\n"
         "brillent (0 = tout l'écran glow, voile laiteux)"),
        ("Trainées (fade)", "trail",        0.05, 0.9, 0.01,
         "Effacement par frame : bas = longues trainées"),
    ]),
    ("Rendu", [
        ("Grain",             "grain",        0.0,  0.5, 0.01, None),
        ("Anti-aliasing (×)", "render_scale", 1.0,  2.0, 0.25,
         "Densité de rendu (px internes par px).\nSans effet sur un écran Retina "
         "(déjà 2×) :\nle MSAA 4× lisse les arêtes dans tous les cas"),
        ("Vignette",          "vignette",     0.0,  1.0, 0.01,
         "Assombrit doucement les bords de l'image"),
        ("Taille points",     "point_size",   0.3,  3.0, 0.05, None),
        ("Zoom",              "zoom",         0.3,  3.0, 0.05, None),
        ("Réactivité",        "reactivity",   0.25, 3.0, 0.05,
         "Haut = percussif (retombée courte),\nbas = planant (retombée ~1 s)"),
    ]),
]

VIDEO_SLIDERS = [s[:5] for _, grp in VIDEO_GROUPS for s in grp]

# Tirage aléatoire : plages « sûres » (pas d'écran noir ou saturé d'office).
# Les réglages absents prennent toute leur plage ; render_scale est exclu
# (c'est un réglage de performance, pas d'esthétique).
RANDOM_VIDEO_RANGES = {
    "red": (0.5, 1.5), "green": (0.5, 1.5), "blue": (0.5, 1.5),
    "contrast": (0.7, 1.6), "brightness": (0.6, 1.6),
    "bloom": (0.3, 2.2), "trail": (0.1, 0.6), "grain": (0.0, 0.2),
    "point_size": (0.5, 2.0), "zoom": (0.6, 1.6),
}
RANDOM_VIDEO_SKIP = {"render_scale"}


def _snap(v: float, lo: float, step: float) -> float:
    return round(round((v - lo) / step) * step + lo, 6)


def random_video_values() -> dict:
    vals = {}
    for _label, key, lo, hi, step in VIDEO_SLIDERS:
        if key in RANDOM_VIDEO_SKIP:
            continue
        a, b = RANDOM_VIDEO_RANGES.get(key, (lo, hi))
        vals[key] = _snap(random.uniform(a, b), lo, step)
    if random.random() < 0.8:       # le négatif reste l'exception
        vals["negative"] = 0.0
    return vals


def random_physics_values() -> dict:
    vals = {}
    for i in range(len(EQ_BANDS)):
        # ~1 bande sur 3 reste muette : un routing lisible plutôt qu'un chaos
        fx = 0 if random.random() < 0.3 else random.randrange(1, len(EFFECTS))
        vals[f"band{i}_fx"] = float(fx)
        vals[f"band{i}_amt"] = _snap(random.uniform(0.3, 1.8), 0.0, 0.05)
    return vals

OBJECT_SLIDERS = [
    ("Turbulence", "turbulence", 0.0, 3.0, 0.05,
     "Dose l'effet Turbulence routé dans Physique\n(aucun effet seul)"),
    ("Nombre de points (milliers)", "n_points", 50.0, 1000.0, 25.0,
     "Style Points : nombre de points sur la surface\n"
     "— reconstruit ~0,5 s après le réglage"),
]

MODEL_STYLES = ["Points", "Surface", "Wireframe"]

MODEL_SLIDERS = [
    ("Gonflement", "inflate", 0.0, 2.0, 0.05,
     "Gonflement permanent le long des normales\n(0 = modèle intact). Pour l'animer sur\n"
     "la musique : effet « Gonflement » dans Physique"),
    ("Rim (contour lumineux)", "rim", 0.0, 3.0, 0.05,
     "Liseré lumineux sur les bords (fresnel)"),
    ("Teinte du rim", "rim_hue", 0.0, 1.0, 0.01, None),
    ("Couleur d'origine", "orig_color", 0.0, 1.0, 0.05,
     "Surface : 0 = gris neutre (studio) / matcap pur,\n"
     "1 = couleurs du modèle éclairées"),
    ("Rugosité", "roughness", 0.05, 1.0, 0.01,
     "Studio : 0 = reflets nets (vernis, chrome),\n1 = mat (argile, plâtre)"),
    ("Métal", "metal", 0.0, 1.0, 0.01,
     "Studio : 1 = les reflets prennent la couleur\ndu modèle (or, cuivre…)"),
    ("Occlusion des creux", "ao", 0.0, 2.0, 0.05,
     "Assombrit les plis et cavités → volume, détails"),
    ("Épaisseur traits", "wire_width", 0.5, 4.0, 0.1,
     "Wireframe : largeur des arêtes en pixels"),
]

SHADING_MODES = ["Studio (PBR)", "Matcap"]

IMPORT_TYPES = [
    ("Modèles 3D", "*.glb *.gltf *.obj *.stl *.ply"),
    ("Tous les fichiers", "*.*"),
]

ROTATION_SLIDERS = [
    ("Vitesse rotation (× musique)", "rot_gain",     0.0,  3.0,   0.05, None),
    ("Inclinaison de l'axe (°)",     "axis_tilt",    0.0,  90.0,  1.0,  None),
    ("Angle de l'axe (°)",           "axis_azimuth", 0.0,  360.0, 1.0,  None),
    ("Distance à l'origine (px)",    "orbit_dist",   0.0,  600.0, 5.0,  None),
    ("Inclinaison de l'objet (°)",   "obj_tilt",   -60.0,  60.0,  1.0,  None),
    ("Vitesse rotation de l'axe (°/s)", "axis_spin_speed", -120.0, 120.0, 1.0,
     None),
]

ROTATION_PRESETS = {
    "Sur soi (0,0,0)": {"rot_gain": 1.0, "axis_tilt": 0,  "axis_azimuth": 0,
                        "orbit_dist": 0,   "obj_tilt": 0},
    "Sur place":       {"rot_gain": 1.0, "axis_tilt": 15, "axis_azimuth": 0,
                        "orbit_dist": 0,   "obj_tilt": 10},
    "Orbite inclinée": {"rot_gain": 1.0, "axis_tilt": 60, "axis_azimuth": 45,
                        "orbit_dist": 250, "obj_tilt": 15},
    "Toupie penchée":  {"rot_gain": 1.6, "axis_tilt": 30, "axis_azimuth": 90,
                        "orbit_dist": 0,   "obj_tilt": -25},
    "Satellite":       {"rot_gain": 0.8, "axis_tilt": 80, "axis_azimuth": 0,
                        "orbit_dist": 420, "obj_tilt": -15},
}


def list_input_devices() -> list:
    """[(index, label), …] of capturable input devices."""
    try:
        import sounddevice as sd
        return [(i, f"{d['name']}")
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


# ── Thème sombre (T7) ───────────────────────────────────────────────────
BG      = "#15151c"   # window background
CARD    = "#1b1b25"   # group / row background
TROUGH  = "#2a2a38"
FG      = "#c8c8d8"
MUTED   = "#71718a"
ACCENT  = "#5fd7d7"   # cyan — values, VU meters
ACCENT2 = "#8a7fd7"   # violet — selection, active elements
VU_BG   = "#101018"

FONT      = ("Menlo", 11)
FONT_SM   = ("Menlo", 10)
FONT_BOLD = ("Menlo", 11, "bold")


def _run_panel(q: mp.Queue, mon_q: mp.Queue, devices: list,
               init_params: dict = None, init_geometry: str = None):
    import tkinter as tk
    from tkinter import ttk, filedialog

    root = tk.Tk()
    root.title("Graphic Art — Contrôles")
    root.geometry(init_geometry or "380x760")
    root.minsize(340, 480)
    root.configure(bg=BG)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=BG, foreground=FG, font=FONT,
                    fieldbackground=TROUGH, bordercolor=TROUGH,
                    lightcolor=BG, darkcolor=BG)
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 6), font=FONT,
                    background=CARD, foreground=MUTED)
    style.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", ACCENT)])
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED,
                    font=FONT_SM)
    style.configure("Value.TLabel", background=BG, foreground=ACCENT,
                    font=FONT_SM)
    style.configure("Band.TLabel", background=BG, foreground="#8fb",
                    font=FONT_BOLD)
    style.configure("TLabelframe", background=BG, bordercolor=TROUGH,
                    relief="groove")
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT2,
                    font=FONT_BOLD)
    style.configure("TButton", background=CARD, foreground=FG, font=FONT,
                    borderwidth=1, focuscolor=BG, padding=(8, 4))
    style.map("TButton", background=[("active", TROUGH)])
    style.configure("TCheckbutton", background=BG, foreground=FG, font=FONT,
                    indicatorcolor=TROUGH, focuscolor=BG)
    style.map("TCheckbutton",
              background=[("active", BG)],
              indicatorcolor=[("selected", ACCENT)])
    style.configure("Horizontal.TScale", background=BG, troughcolor=TROUGH,
                    lightcolor=ACCENT2, darkcolor=ACCENT2, bordercolor=TROUGH)
    style.configure("TCombobox", fieldbackground=TROUGH, background=CARD,
                    foreground=FG, arrowcolor=ACCENT, selectbackground=TROUGH,
                    selectforeground=FG)
    style.map("TCombobox", fieldbackground=[("readonly", TROUGH)])
    style.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED)
    # dropdown list of the comboboxes (plain tk widget)
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT2)
    root.option_add("*TCombobox*Listbox.font", FONT_SM)

    # ── Tooltips (T7) ─────────────────────────────────────────────────
    class Tooltip:
        def __init__(self, widget, text):
            self.widget, self.text = widget, text
            self.tip = None
            self._after = None
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<Button>", self._hide, add="+")

        def _schedule(self, _e=None):
            self._after = self.widget.after(600, self._show)

        def _show(self):
            if self.tip is not None:
                return
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self.tip, text=self.text, bg="#26263a", fg=FG,
                     font=FONT_SM, justify="left", padx=8, pady=4,
                     highlightthickness=1,
                     highlightbackground=ACCENT2).pack()

        def _hide(self, _e=None):
            if self._after is not None:
                self.widget.after_cancel(self._after)
                self._after = None
            if self.tip is not None:
                self.tip.destroy()
                self.tip = None

    # ── Barre presets utilisateur (globale, au-dessus des onglets) ─────
    # Un preset = snapshot de TOUS les réglages (source et modèle exclus :
    # ce ne sont pas des paramètres esthétiques).
    preset_bar = ttk.Frame(root)
    preset_bar.pack(fill="x", padx=10, pady=(10, 2))
    preset_combo = ttk.Combobox(preset_bar, state="readonly", width=14,
                                font=FONT_SM)
    preset_combo.pack(side="left", fill="x", expand=True)
    btn_load = ttk.Button(preset_bar, text="Charger", width=8)
    btn_save = ttk.Button(preset_bar, text="Sauver…", width=8)
    btn_del = ttk.Button(preset_bar, text="✕", width=2)
    for b in (btn_load, btn_save, btn_del):
        b.pack(side="left", padx=(4, 0))

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=6, pady=6)

    # ── Barre de statut (T8) ──────────────────────────────────────────
    status_lbl = tk.Label(root, text="—", bg=CARD, fg=MUTED, font=FONT_SM,
                          anchor="w", padx=10, pady=3)
    status_lbl.pack(fill="x", side="bottom")

    scales = {}   # key → slider proxy (.get()/.set())
    checks = {}   # key → BooleanVar
    tab_canvases = {}   # notebook tab id → its scrollable canvas

    def make_tab(title):
        """Create a notebook tab whose content scrolls vertically."""
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        def _sync_scrollregion(_e=None):
            bbox = canvas.bbox("all")
            if bbox is None:
                return
            # anchored at (0,0); when the content fits, pin the view to the
            # top and collapse the scrollregion so no dead space can appear
            h = max(bbox[3], 1)
            canvas.configure(scrollregion=(0, 0, bbox[2], h))
            if h <= canvas.winfo_height():
                canvas.yview_moveto(0.0)

        inner.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>",
                    lambda e: (canvas.itemconfigure(win, width=e.width),
                               _sync_scrollregion()))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tab_canvases[str(outer)] = canvas
        return inner

    def _on_mousewheel(event):
        canvas = tab_canvases.get(nb.select())
        if canvas is None:
            return
        bbox = canvas.bbox("all")
        if bbox is None or bbox[3] <= canvas.winfo_height():
            canvas.yview_moveto(0.0)   # everything fits: no scrolling
            return
        canvas.yview_scroll(-event.delta, "units")

    root.bind_all("<MouseWheel>", _on_mousewheel)

    def send(key, val):
        q.put(("param", key, float(val)))

    class SliderProxy:
        """ttk.Scale + live value label; .set() drives the scale (fires the
        command → queue), .get() returns the step-quantized value.
        NB: goes through Scale.set(), NOT var.set() — setting the linked
        variable does not fire the ttk command (presets would silently not
        reach the engine)."""

        def __init__(self, scale, var, quantize):
            self._scale, self._var, self._q = scale, var, quantize

        def get(self):
            return self._q(self._var.get())

        def set(self, v):
            # a greyed-out scale (inactive Objet group) must still accept
            # preset / autosave values
            disabled = self._scale.instate(["disabled"])
            if disabled:
                self._scale.state(["!disabled"])
            self._scale.set(float(v))
            if disabled:
                self._scale.state(["disabled"])

    def build_slider(parent, label, key, lo, hi, step, tip=None):
        """Return a (not yet packed) frame: label + live value + ttk.Scale.
        Double-click resets the slider to its default (T7)."""
        row = ttk.Frame(parent)
        row.columnconfigure(0, weight=1)
        decimals = 0 if step >= 1 else (1 if step >= 0.1 else 2)

        def quantize(raw):
            return round(round((raw - lo) / step) * step + lo, 6)

        ttk.Label(row, text=label).grid(row=0, column=0, sticky="w")
        val_lbl = ttk.Label(row, text="", style="Value.TLabel")
        val_lbl.grid(row=0, column=2, sticky="e")
        var = tk.DoubleVar(value=DEFAULTS[key])

        def on_move(_v=None, k=key):
            v = quantize(var.get())
            val_lbl.configure(text=f"{v:.{decimals}f}")
            send(k, v)

        s = ttk.Scale(row, from_=lo, to=hi, variable=var, command=on_move)
        s.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 2))
        s.bind("<Double-Button-1>",
               lambda e, k=key, w=s: (w.set(DEFAULTS[k]), "break")[1])
        val_lbl.configure(text=f"{quantize(DEFAULTS[key]):.{decimals}f}")
        if tip:
            Tooltip(s, tip)
        scales[key] = SliderProxy(s, var, quantize)
        return row

    def add_sliders(parent, sliders):
        for label, key, lo, hi, step, *rest in sliders:
            tip = rest[0] if rest else None
            build_slider(parent, label, key, lo, hi, step, tip)\
                .pack(fill="x", padx=14, pady=(6, 0))

    def add_check(parent, text, key, **pack_kw):
        var = tk.BooleanVar(value=DEFAULTS[key] >= 0.5)
        checks[key] = var
        ttk.Checkbutton(parent, text=text, variable=var,
                        command=lambda: send(key, 1.0 if var.get() else 0.0)
                        ).pack(anchor="w", padx=14, **pack_kw)
        return var

    combos = {}   # key → (Combobox, values list)

    def add_combo(parent, label, key, options, tip=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=14, pady=(6, 0))
        ttk.Label(row, text=label).pack(side="left")
        c = ttk.Combobox(row, values=options, state="readonly", width=15,
                         font=FONT_SM)
        c.current(int(DEFAULTS[key]))
        c.pack(side="right")
        c.bind("<<ComboboxSelected>>",
               lambda e: send(key, float(c.current())))
        combos[key] = (c, options)
        if tip:
            Tooltip(c, tip)

    def apply_values(values: dict):
        for key, val in values.items():
            if key in scales:
                scales[key].set(val)   # .set() fires the callback → queue
            elif key in checks:
                checks[key].set(val >= 0.5)
                send(key, 1.0 if val >= 0.5 else 0.0)
            elif key in combos:
                combo, options = combos[key]
                combo.current(int(val))
                send(key, float(int(val)))

    # per-tab reset: each tab claims the controls built inside it, so its
    # button only resets those (never the other tabs)
    tab_keys = {}

    def claim_keys(tab):
        taken = set().union(*tab_keys.values())
        tab_keys[tab] = (set(scales) | set(checks) | set(combos)) - taken

    def reset_button(parent, tab):
        ttk.Button(parent, text="↺  Reset de cet onglet",
                   command=lambda: apply_values(
                       {k: DEFAULTS[k] for k in tab_keys.get(tab, ())})
                   ).pack(fill="x", padx=10, pady=(12, 4))

    def random_button(parent, make_values):
        ttk.Button(parent, text="🎲  Paramètres aléatoires",
                   command=lambda: apply_values(make_values())
                   ).pack(fill="x", padx=10, pady=(0, 4))

    def tab_presets_ui(parent, tab, builtin, **pack_kw):
        """Presets propres à un onglet : liste + Charger / Sauver… / ✕.
        Un preset = snapshot des réglages de l'onglet seulement."""
        from tkinter import messagebox, simpledialog
        grp = ttk.LabelFrame(parent, text="Presets")
        grp.pack(fill="x", padx=10, **pack_kw)
        combo = ttk.Combobox(grp, state="readonly", width=14, font=FONT_SM)
        combo.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=8)
        btns = {t: ttk.Button(grp, text=t, width=w) for t, w in
                (("Charger", 8), ("Sauver…", 8), ("✕", 2))}
        for c, b in enumerate(btns.values(), start=1):
            b.grid(row=0, column=c, padx=(0, 4 if c < 3 else 8), pady=8)
        grp.columnconfigure(0, weight=1)
        presets = _load_tab_presets(tab, builtin)

        def refresh(select=None):
            names = sorted(presets)
            combo.configure(values=names or ["(aucun preset)"])
            combo.set(select if select in presets else
                      (names[0] if names else "(aucun preset)"))

        def load():
            vals = presets.get(combo.get())
            if vals:
                apply_values({k: v for k, v in vals.items() if k in DEFAULTS})

        def save():
            name = simpledialog.askstring(
                "Preset", "Nom du preset (un nom existant le remplace) :",
                initialvalue=combo.get() if combo.get() in presets else "",
                parent=root)
            name = (name or "").strip()
            if not name:
                return
            if name in presets and not messagebox.askyesno(
                    "Preset", f"Remplacer « {name} » ?", parent=root):
                return
            presets[name] = {k: float(v) for k, v in snapshot().items()
                             if k in tab_keys.get(tab, ())}
            _save_tab_presets(tab, presets)
            refresh(name)

        def delete():
            if combo.get() in presets:
                del presets[combo.get()]
                _save_tab_presets(tab, presets)
                refresh()

        btns["Charger"].configure(command=load)
        btns["Sauver…"].configure(command=save)
        btns["✕"].configure(command=delete)
        refresh()

    # ── Onglet Source ─────────────────────────────────────────────────
    tab_src = make_tab("Source")

    def choose_audio():
        path = filedialog.askopenfilename(
            title="Choisir un fichier audio",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                       ("Tous les fichiers", "*.*")])
        if path:
            q.put(("action", "load_audio", path))

    grp_file = ttk.LabelFrame(tab_src, text="Fichier")
    grp_file.pack(fill="x", padx=10, pady=(14, 4))
    ttk.Button(grp_file, text="🎵  Choisir un fichier audio…",
               command=choose_audio).pack(fill="x", padx=8, pady=8)

    grp_live = ttk.LabelFrame(tab_src, text="Live (appli / micro)")
    grp_live.pack(fill="x", padx=10, pady=(10, 4))
    ttk.Label(grp_live,
              text="Pour capter une application, route-la vers\n"
                   "Background Music ou BlackHole puis choisis\n"
                   "ce périphérique ci-dessous.",
              style="Muted.TLabel", justify="left"
              ).pack(anchor="w", padx=8, pady=(4, 4))

    dev_state = {"devices": list(devices)}
    dev_combo = ttk.Combobox(grp_live, state="readonly", font=FONT_SM)
    dev_combo.pack(fill="x", padx=8)

    def refresh_devices():
        # re-scan in this process: devices can appear after launch (routing,
        # plugged interface) and the parent list may be stale or empty
        devs = list_input_devices() or dev_state["devices"]
        dev_state["devices"] = devs
        labels = [f"{i} · {name}" for i, name in devs] or ["(aucun périphérique)"]
        dev_combo.configure(values=labels)
        dev_combo.current(0)

    refresh_devices()

    def start_live():
        if not dev_state["devices"]:
            return
        q.put(("action", "start_live",
               dev_state["devices"][dev_combo.current()][0]))

    ttk.Button(grp_live, text="🔄  Rafraîchir la liste",
               command=refresh_devices).pack(fill="x", padx=8, pady=(6, 0))
    ttk.Button(grp_live, text="▶  Démarrer l'analyse live",
               command=start_live).pack(fill="x", padx=8, pady=(6, 8))
    ttk.Button(tab_src, text="■  Stop source",
               command=lambda: q.put(("action", "stop_source", None))
               ).pack(fill="x", padx=10, pady=(8, 10))

    # ── Onglet Vidéo ──────────────────────────────────────────────────
    tab_video = make_tab("Vidéo")
    reset_button(tab_video, "Vidéo")
    random_button(tab_video, random_video_values)
    for group_title, group_sliders in VIDEO_GROUPS:
        grp = ttk.LabelFrame(tab_video, text=group_title)
        grp.pack(fill="x", padx=10, pady=(8, 2))
        for label, key, lo, hi, step, tip in group_sliders:
            build_slider(grp, label, key, lo, hi, step, tip)\
                .pack(fill="x", padx=8, pady=(4, 0))
        ttk.Frame(grp).pack(pady=2)   # bottom breathing room

    add_check(tab_video, "Grille du décor", "show_grid", pady=(8, 10))
    claim_keys("Vidéo")
    tab_presets_ui(tab_video, "Vidéo", {}, pady=(4, 12))

    # ── Onglet Objet (modèle 3D, T19) ─────────────────────────────────
    tab_obj = make_tab("Objet")
    reset_button(tab_obj, "Objet")

    def import_visual():
        path = filedialog.askopenfilename(
            title="Importer un modèle 3D", filetypes=IMPORT_TYPES)
        if path:
            q.put(("action", "load_visual", path))

    ttk.Button(tab_obj, text="📁  Importer un modèle 3D…",
               command=import_visual).pack(fill="x", padx=10, pady=(14, 4))
    ttk.Button(tab_obj, text="↩  Retour au tore",
               command=lambda: q.put(("action", "default_visual", None))
               ).pack(fill="x", padx=10, pady=(2, 4))
    prim_row = ttk.Frame(tab_obj)
    prim_row.pack(fill="x", padx=10, pady=(2, 2))
    ttk.Label(prim_row, text="Formes test :", style="Muted.TLabel"
              ).pack(side="left")
    for label, kind in (("Sphère", "icosphere"), ("Tore", "torus"),
                        ("Cube", "box")):
        ttk.Button(prim_row, text=label, width=6,
                   command=lambda k=kind: q.put(
                       ("action", "load_visual", f"primitive:{k}"))
                   ).pack(side="left", padx=(4, 0))
    add_sliders(tab_obj, OBJECT_SLIDERS)

    grp_model = ttk.LabelFrame(tab_obj, text="Modèle 3D")
    grp_model.pack(fill="x", padx=10, pady=(10, 8))
    add_combo(grp_model, "Style", "model_style", MODEL_STYLES,
              "Points : nuage sur la surface\nSurface : objet plein éclairé\n"
              "Wireframe : arêtes du maillage")
    add_combo(grp_model, "Axe haut", "model_up", MESH_UP_AXES,
              "Les STL et les scans sont souvent en Z-up")
    add_combo(grp_model, "Éclairage", "shading", SHADING_MODES,
              "Studio : lumières + reflets physiques (rugosité, métal)\n"
              "Matcap : sphère pré-éclairée (look stylisé)")
    add_combo(grp_model, "Matcap", "matcap", MATCAP_NAMES,
              "Utilisé quand l'éclairage est en mode Matcap")
    add_check(grp_model, "Surface transparente (hologramme)",
              "mesh_transparent", pady=(8, 0))
    add_check(grp_model, "Wireframe : arêtes vives seulement", "wire_sharp",
              pady=(2, 0))
    add_check(grp_model, "Wireframe : masquer les arêtes cachées",
              "wire_hidden", pady=(2, 0))
    add_sliders(grp_model, MODEL_SLIDERS)
    ttk.Frame(grp_model).pack(pady=2)

    claim_keys("Objet")

    # ── Onglet Rotation ───────────────────────────────────────────────
    tab_rot = make_tab("Rotation")
    reset_button(tab_rot, "Rotation")
    ttk.Label(tab_rot,
              text="Axe et position fixes — la musique\n"
                   "ne pilote que la vitesse de rotation.",
              style="Muted.TLabel", justify="left"
              ).pack(anchor="w", padx=14, pady=(0, 2))
    add_sliders(tab_rot, ROTATION_SLIDERS)

    add_check(tab_rot, "Rotation de l'axe autour du (0,0) du décor",
              "axis_spin", pady=(8, 0))
    add_check(tab_rot, "Afficher l'axe de rotation (vert)", "show_axis",
              pady=(8, 2))

    grp_rotp = ttk.LabelFrame(tab_rot, text="Presets")
    grp_rotp.pack(fill="x", padx=10, pady=(12, 8))
    for i, (name, values) in enumerate(ROTATION_PRESETS.items()):
        ttk.Button(grp_rotp, text=name,
                   command=lambda v=values: apply_values(v)
                   ).grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=3)
    grp_rotp.columnconfigure(0, weight=1)
    grp_rotp.columnconfigure(1, weight=1)
    claim_keys("Rotation")

    # ── Onglet Physique (avec VU-mètres, T8) ──────────────────────────
    tab_phys = make_tab("Physique")
    reset_button(tab_phys, "Physique")
    random_button(tab_phys, random_physics_values)
    ttk.Label(tab_phys,
              text="Chaque bande de fréquence pilote\nl'effet de son choix.",
              style="Muted.TLabel", justify="left"
              ).pack(anchor="w", padx=14, pady=(10, 4))
    build_slider(tab_phys, "Dynamique (breaks → drops)", "dynamics", 0.0, 1.0,
                 0.05, "Atténue tous les effets dans les passages calmes\n"
                       "et les libère sur les drops (bande ÉNERGIE).\n0 = off")\
        .pack(fill="x", padx=14, pady=(2, 4))

    VU_W, VU_H = 12, 46
    vu_canvases = []   # (canvas, bar item) per band

    for i, band in enumerate(EQ_BANDS):
        blk = ttk.Frame(tab_phys)
        blk.pack(fill="x", padx=14, pady=(6, 0))
        blk.columnconfigure(1, weight=1)
        vu = tk.Canvas(blk, width=VU_W, height=VU_H, bg=VU_BG,
                       highlightthickness=1, highlightbackground=TROUGH)
        vu.grid(row=0, column=0, rowspan=2, padx=(0, 8), sticky="ns")
        bar = vu.create_rectangle(1, VU_H, VU_W, VU_H, fill=ACCENT, width=0)
        vu_canvases.append((vu, bar))
        Tooltip(vu, f"Niveau de la bande {band} (post-enveloppe)")

        head = ttk.Frame(blk)
        head.grid(row=0, column=1, sticky="ew")
        ttk.Label(head, text=band, style="Band.TLabel", width=9,
                  anchor="w").pack(side="left")
        combo = ttk.Combobox(head, values=EFFECTS, state="readonly",
                             width=16, font=FONT_SM)
        combo.current(int(DEFAULTS[f"band{i}_fx"]))
        combo.pack(side="left", fill="x", expand=True, padx=(4, 0))
        combo.bind("<<ComboboxSelected>>",
                   lambda e, k=f"band{i}_fx", c=combo: send(k, float(c.current())))
        combos[f"band{i}_fx"] = (combo, EFFECTS)
        Tooltip(combo, "Effet piloté par cette bande")

        build_slider(blk, "Intensité", f"band{i}_amt", 0.0, 2.0, 0.05)\
            .grid(row=1, column=1, sticky="ew", pady=(2, 0))

    claim_keys("Physique")
    tab_presets_ui(tab_phys, "Physique", PHYSICS_PRESETS, pady=(12, 12))

    # ── Monitoring main → panel (T8) : VU-mètres + barre de statut ─────
    vu_levels = [0.0] * len(EQ_BANDS)

    def poll_monitor():
        pkt = None
        while True:   # drain: only the freshest packet matters
            try:
                pkt = mon_q.get_nowait()
            except queue.Empty:
                break
        if pkt is not None:
            bands = pkt.get("bands", ())
            for j, (vu, bar) in enumerate(vu_canvases):
                lvl = min(max(float(bands[j]) if j < len(bands) else 0.0,
                              0.0), 1.0)
                # small decay so the bars fall smoothly between packets
                vu_levels[j] = max(lvl, vu_levels[j] * 0.7)
                vu.coords(bar, 1, VU_H - vu_levels[j] * (VU_H - 2),
                          VU_W, VU_H)
            rms = min(max(pkt.get("rms", 0.0), 0.0), 1.0)
            blocks = "▓" * round(rms * 6) + "░" * (6 - round(rms * 6))
            bpm = pkt.get("bpm", 0.0)
            status_lbl.configure(text=(
                f"{'BPM %.0f' % bpm if bpm > 0 else 'BPM --'} · "
                f"RMS {blocks} · {pkt.get('fps', 0.0):.0f} fps · "
                f"{pkt.get('source', '—')}"))
        root.after(66, poll_monitor)   # ~15 Hz

    poll_monitor()

    # ── Position/taille de fenêtre mémorisées (T7) ─────────────────────
    geom_state = {"after": None}

    def push_geometry():
        geom_state["after"] = None
        q.put(("action", "_window", root.geometry()))

    def on_configure(_e):
        # debounce: only send once the user stops moving/resizing
        if geom_state["after"] is not None:
            root.after_cancel(geom_state["after"])
        geom_state["after"] = root.after(400, push_geometry)

    root.bind("<Configure>", on_configure)

    # ── Presets utilisateur : snapshot global de tous les onglets ──────
    def snapshot() -> dict:
        vals = {}
        for k, s in scales.items():
            vals[k] = float(s.get())
        for k, v in checks.items():
            vals[k] = 1.0 if v.get() else 0.0
        for k, (c, _) in combos.items():
            vals[k] = float(c.current())
        return vals

    def refresh_preset_list(select: str = None):
        names = sorted(n for n in _load_presets_file()
                       if not n.startswith("_"))
        preset_combo.configure(values=names or ["(aucun preset)"])
        if select and select in names:
            preset_combo.set(select)
        elif names:
            preset_combo.current(0)
        else:
            preset_combo.set("(aucun preset)")

    def save_preset():
        from tkinter import simpledialog
        name = simpledialog.askstring("Preset", "Nom du preset :", parent=root)
        if not name or name.strip() == "" or name.strip().startswith("_"):
            return
        d = _load_presets_file()
        d[name.strip()] = snapshot()
        _save_presets_file(d)
        refresh_preset_list(name.strip())

    def load_preset():
        d = _load_presets_file()
        name = preset_combo.get()
        if name in d:
            apply_values({k: v for k, v in d[name].items() if k in DEFAULTS})

    def delete_preset():
        d = _load_presets_file()
        name = preset_combo.get()
        if name in d:
            del d[name]
            _save_presets_file(d)
            refresh_preset_list()

    btn_load.configure(command=load_preset)
    btn_save.configure(command=save_preset)
    btn_del.configure(command=delete_preset)
    refresh_preset_list()

    # restore the autosaved state (pushes the values back to main via the queue)
    if init_params:
        apply_values({k: v for k, v in init_params.items() if k in DEFAULTS})

    root.mainloop()


class ControlPanel:
    MONITOR_INTERVAL = 1.0 / 15   # seconds between monitoring packets (T8)

    def __init__(self):
        self.params = dict(DEFAULTS)
        # restore the last session's settings (source and model excluded)
        presets = _load_presets_file()
        auto = presets.get("_autosave", {})
        self.params.update({k: float(v) for k, v in auto.items() if k in DEFAULTS})
        self.window_geometry = presets.get("_window") or None
        self.actions = []            # pending ("name", payload) actions
        self._q = mp.Queue()
        self._mon_q = mp.Queue(maxsize=4)   # main → panel monitoring (T8)
        self._proc = mp.Process(target=_run_panel,
                                args=(self._q, self._mon_q,
                                      list_input_devices(), auto,
                                      self.window_geometry),
                                daemon=True)
        self._proc.start()

    def save_autosave(self):
        """Persist the current settings; restored on the next launch."""
        d = _load_presets_file()
        d["_autosave"] = {k: float(v) for k, v in self.params.items()}
        if self.window_geometry:
            d["_window"] = self.window_geometry
        _save_presets_file(d)

    def send_monitor(self, packet: dict):
        """Push a monitoring packet to the panel (drops if the queue is full,
        the panel only cares about the freshest one)."""
        try:
            self._mon_q.put_nowait(packet)
        except queue.Full:
            pass

    def poll(self) -> dict:
        """Drain pending messages; return the parameter dict.
        Actions (e.g. model loads) are queued in `self.actions`."""
        while True:
            try:
                kind, key, val = self._q.get_nowait()
            except queue.Empty:
                break
            if kind == "param":
                self.params[key] = val
            elif key == "_window":
                self.window_geometry = val   # T7: saved with the autosave
            else:
                self.actions.append((key, val))
        return self.params

    def pop_actions(self) -> list:
        acts, self.actions = self.actions, []
        return acts

    def close(self):
        if self._proc.is_alive():
            self._proc.terminate()
