"""État du fond d'une frame à l'autre (T1.16), sans GL.

`BackgroundState.step()` prend les réglages du panneau, les features audio
lissées et les sources du routing, et rend tout ce que la passe GL envoie
au shader : macros appliquées, réaction musicale dosée par « Réaction à la
musique », variation bornée (`slew`), voyage de l'objet et mouvements
propres intégrés en dt (rotation, dérive, respiration), horloge des beats,
couleurs dérivées et palette, et les réglages des cartes de relief
(`Frame.terrain`, calculées par background/terrain.py). Pour le Volume, il
calcule ce qui ne change pas d'un pas de marche à l'autre : étirement dû à
la Distorsion, seuil des cavités (quantile réel du bruit). Le relief est
fixe : rien dans le paysage ne bouge ni ne change de couleur avec le temps.

Horloge des beats : comptée ici (retour de `beat_phase` à 0) plutôt que dans
le `Reactor` ; elle ne suit les vrais beats que si la musique a le droit
d'agir (dose > 0 et BPM connu), sinon elle tourne seule à 120 BPM : à dose
nulle, le fond ne dépend plus de l'audio.
"""

import math
from dataclasses import dataclass

import numpy as np

from background import macros, noise, palettes, temporal, terrain
from background.schema import BREATH_BEATS, PARAMS, RESOLUTIONS

# ordre des sources dans `Routed.bands` (app/routing.py) : 7 bandes EQ,
# BEAT, KICK, SNARE, ÉNERGIE
BAND_SOURCES = ("sub", "bass", "lowmid", "mid", "highmid", "presence",
                "treble", "beat", "kick", "snare", "energy")
FREE_BPM = 120.0          # horloge des beats sans tempo
DRIFT_AMP = 0.3           # amplitude de la dérive (unités du fond)
VALLEY_EDGE = 0.6         # = landscape.glsl : flanc de vallée, extinction de la Distorsion
# = landscape.glsl : déplacement du warp / (warp × taille) ; à 1, l'air
# devenait roche au-dessus de la caméra
WARP_GAIN = 0.3
BUBBLE_MAX = 0.45         # = landscape.glsl : rayon + écart d'une bulle (cellule = 1)
NOISE_QUANTILE = 99.0     # pente du bruit 3D retenue pour borner les pas (voir noise)
AUDIO_PARAMS = [p for p in PARAMS if p.audio is not None]
SLEW_PARAMS = [p for p in PARAMS if p.max_slew is not None]
SHADER_PARAMS = [p for p in PARAMS if p.shader]
# réglages qui sautent d'une valeur à l'autre : quand l'un change, le décor
# n'est plus le même et l'historique du lissage temporel est jeté
STRUCTURE_KEYS = [p.key for p in PARAMS if p.kind == "int"] + ["bg_seed"]
PALETTE_KEYS = ("bg_col_mode", "bg_col_preset",
                "bg_col_a_hue", "bg_col_a_sat", "bg_col_a_lum",
                "bg_col_b_hue", "bg_col_b_sat", "bg_col_b_lum",
                "bg_col_c_hue", "bg_col_c_sat", "bg_col_c_lum")


def rotation(axis, angle: float) -> np.ndarray:
    """Rotation 3×3 (Rodrigues) autour d'un axe unitaire."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + s * k + (1.0 - c) * np.outer(axis, axis)


def axis_from(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    a, t = math.radians(azimuth_deg), math.radians(tilt_deg)
    return np.array([math.sin(t) * math.cos(a), math.cos(t), math.sin(t) * math.sin(a)])


def travel_dir(v: dict) -> np.ndarray:
    """Direction du voyage (unitaire, toujours à l'horizontale : le sol reste
    au sol) ; azimut 0 = vers le fond de l'écran de la vue de départ."""
    az = math.radians(v["bg_mov_dir_az"])
    return np.array([math.sin(az), 0.0, -math.cos(az)])


def volume_stretch(v: dict) -> float:
    """Étirement maximal dû aux ondes et au warp de la Distorsion (les pas de
    marche sont divisés d'autant) ; torsion et courbure dépendent du point :
    le shader les calcule. Les termes en 1/W viennent de l'extinction près
    du trajet (sur un flanc de vallée)."""
    S = v["bg_ld_scale"]
    W = VALLEY_EDGE * S
    A = v["bg_vo_wave_amp"]
    # deux ondulations enchaînées : leurs étirements se multiplient
    ka = 2.0 * math.pi * A / v["bg_vo_wave_len"]
    edge = 1.5 * A / W
    wave = (1.0 + ka + edge) * (1.0 + 0.73 * ka + edge)
    warp = 1.0
    if v["bg_vo_warp"] > 0.0:
        octaves = range(int(v["bg_vo_warp_oct"]))
        amp = 2.0 * WARP_GAIN * v["bg_vo_warp"] * S
        texels = noise.SIZE / (noise.CELLS * v["bg_vo_warp_size"] * S)   # par unité
        # pente réaliste du bruit (99e centile), sans le facteur √3 du pire
        # cas où les trois axes seraient au plus raide ensemble : la borne
        # stricte divisait les pas par 3, les rayons s'épuisaient
        slope = noise.gradient_norm(NOISE_QUANTILE) * texels * sum(1.015 ** i for i in octaves)
        reach = 0.5 * sum(0.5 ** i for i in octaves)       # plus grand |n| par axe
        warp += amp * (slope + reach * 1.5 / W)
    return wave * warp


def cave_threshold(v: dict) -> float:
    """Seuil des cavités sur le bruit centré : creusées, la part creusée vaut
    la moitié de « Cavités » ; roche partout, la roche occupe de 20 à 60 %
    de l'espace."""
    if int(v["bg_vo_cave_mode"]) == 0:
        return noise.centered_quantile(1.0 - 0.5 * v["bg_vo_caves"])
    return noise.centered_quantile(1.0 - (0.2 + 0.4 * v["bg_vo_caves"]))


def lin_hsv(h, s, v) -> tuple:
    return tuple(float(c) for c in palettes.to_linear(palettes.hsv(h, s, v)))


@dataclass
class Frame:
    """Ce qu'il faut pour dessiner le fond cette frame."""
    visible: bool
    quality: int
    res_index: int
    uniforms: dict
    palette: np.ndarray       # (SIZE, 3) linéaire
    palette_rev: int          # change quand la palette change
    bloom: float
    solid: tuple              # couleur unie sous les formes (0–1)
    parallax: float
    light_model: bool         # éclairé par la lumière principale du modèle
    light_int: float          # intensité de cette lumière (1 = celle du modèle)
    same_space: bool          # même espace que le modèle → profondeur partagée
    history: float = 0.0      # poids maximal de l'historique (0 = sans lissage)
    structure: tuple = ()     # change quand le décor change de structure
    terrain: dict = None      # réglages des cartes de relief (background/terrain.py)


class BackgroundState:
    def __init__(self):
        self.values = None                         # après macros + musique + slew
        self.sources = dict.fromkeys(BAND_SOURCES + ("rms",), 0.0)
        self.angle = 0.0                           # rotation propre (rad)
        self.travel = np.zeros(3)                  # chemin parcouru par l'objet (unités)
        self.drift = 0.0                           # phase de la dérive
        self.clock = 0.0                           # horloge des beats (beats)
        self.beats = 0
        self._last_phase = None
        self._palette_key = None
        self.palette = None
        self.palette_hue = 0.0
        self.palette_rev = 0

    # ── étapes ────────────────────────────────────────────────────────
    def _listen(self, features: dict, bands, dt: float, inertia: float):
        """Sources audio lissées (constante de temps ∝ « Inertie »)."""
        raw = dict(zip(BAND_SOURCES, list(bands) + [0.0] * len(BAND_SOURCES),
                       strict=False))
        raw["rms"] = features.get("rms", 0.0)
        k = 1.0 - math.exp(-dt / (0.02 + 0.5 * inertia))
        for name, v in raw.items():
            self.sources[name] += (float(v) - self.sources[name]) * k

    def _react(self, v: dict) -> dict:
        """Réaction musicale par défaut de chaque paramètre, dosée."""
        dose = v["bg_mov_music"]
        if dose <= 0.0:
            return v
        out = dict(v)
        for p in AUDIO_PARAMS:
            src, amount = p.audio
            x = self.sources[src] * amount * dose
            if p.kind == "phase":
                out[p.key] = v[p.key] * (1.0 + 3.0 * x)
            else:
                out[p.key] = macros.bound(p, v[p.key] + x * (p.hi - p.lo))
        return out

    def _slew(self, target: dict, dt: float) -> dict:
        if self.values is None:
            return dict(target)
        out = dict(target)
        for p in SLEW_PARAMS:
            cur, want = self.values[p.key], target[p.key]
            delta = want - cur
            if p.law == "angle":   # par le plus court chemin
                span = p.hi - p.lo
                delta = (delta + span / 2.0) % span - span / 2.0
            limit = p.max_slew * dt
            out[p.key] = macros.bound(p, cur + min(max(delta, -limit), limit))
        return out

    def _advance(self, v: dict, features: dict, dt: float):
        """Voyage de l'objet, mouvements propres et horloge des beats (la
        respiration du décor la suit)."""
        if v["bg_mov_freeze"] >= 0.5:
            return
        # l'objet avance dans le décor fixe (float64 : des heures de voyage
        # restent précises une fois envoyées en float32)
        self.travel += travel_dir(v) * v["bg_mov_travel"] * dt
        bpm = float(features.get("bpm", 0.0) or 0.0)
        speed = v["bg_mov_speed"]
        if bpm > 0.0:
            speed *= 1.0 + v["bg_mov_sync"] * (bpm / FREE_BPM - 1.0)
        self.angle = (self.angle + 2.0 * math.pi * v["bg_mov_rot"] / 60.0 * speed * dt) \
            % (2.0 * math.pi)
        self.drift += v["bg_mov_drift"] * speed * dt
        phase = features.get("beat_phase")
        if v["bg_mov_music"] > 0.0 and bpm > 0.0 and phase is not None:
            if self._last_phase is not None and phase < self._last_phase - 0.5:
                self.beats += 1
            self._last_phase = phase
            self.clock = self.beats + float(phase)
        else:
            self._last_phase = None
            self.clock += dt * FREE_BPM / 60.0 * speed
            self.beats = int(self.clock)

    def _update_palette(self, v: dict):
        key = tuple(round(v[k], 4) for k in PALETTE_KEYS)
        if key == self._palette_key:
            return
        self._palette_key = key
        cols = [(v[f"bg_col_{c}_hue"], v[f"bg_col_{c}_sat"], v[f"bg_col_{c}_lum"])
                for c in "abc"]
        rgb = palettes.build(int(v["bg_col_mode"]), int(v["bg_col_preset"]), *cols)
        self.palette_hue = palettes.mean_hue(rgb)
        self.palette = palettes.to_linear(rgb)
        self.palette_rev += 1

    def _derived(self, v: dict, model_hue: float, grid: bool) -> dict:
        """Uniforms calculés (matrices, couleurs, horloges) ; grid : « Grille
        du décor » cochée."""
        s = self.sources
        dose = v["bg_mov_music"]
        roll = rotation((0.0, 0.0, 1.0), math.radians(v["bg_comp_roll"]))
        rot = roll @ rotation(axis_from(v["bg_mov_rot_az"], v["bg_mov_rot_tilt"]),
                              self.angle)
        d = self.drift
        drift = DRIFT_AMP * np.array([math.sin(d * 0.71) + 0.5 * math.sin(d * 1.93 + 1.0),
                                      math.sin(d * 0.53 + 2.0) * 0.6,
                                      math.sin(d * 0.37 + 4.0) + 0.5 * math.sin(d * 1.31)])
        offset = self.travel + np.array([v["bg_comp_off_x"], v["bg_comp_off_y"], 0.0])
        period = BREATH_BEATS[int(v["bg_mov_breath_beats"])][0]
        breath = 1.0 + v["bg_mov_breath"] * math.sin(2.0 * math.pi * self.clock / period)
        az, el = math.radians(v["bg_light_az"]), math.radians(v["bg_light_el"])
        sun_dir = (math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az))
        snare = s["snare"] * dose
        sun = palettes.to_linear(palettes.kelvin(v["bg_light_temp"])) * v["bg_light_int"] \
            * (1.0 + 3.0 * v["bg_light_flash"] * snare)
        # teinte fixe : ni cycle ni teinte musicale (des couleurs qui
        # glissaient sur le décor, amplifiées par la macro Énergie)
        hue = (v["bg_col_hue"]
               + palettes.harmony_shift(int(v["bg_col_harmony"]), self.palette_hue,
                                        model_hue))
        hue_rot = 2.0 * math.pi * (hue % 1.0)
        scale0 = v["bg_comp_scale"]
        scale = scale0 * breath
        # trajet de l'objet dans l'espace des formes : une droite fixe (la
        # dérive l'en écarte un peu sans la déplacer) ; repère (u, v, d)
        path_d = travel_dir(v)
        path_u = np.cross(path_d, (0.0, 1.0, 0.0))
        if np.linalg.norm(path_u) < 1e-3:
            path_u = np.array([1.0, 0.0, 0.0])
        path_u /= np.linalg.norm(path_u)
        path_m = np.array([path_u, np.cross(path_d, path_u), path_d])
        path_o = offset / scale0
        # la respiration se fait autour de l'objet : ps = R·p / scale +
        # position de l'objet / scale0. Rapportée à l'origine du décor, elle
        # le faisait aller et venir d'autant plus que l'objet avait voyagé
        # (±6 unités au tempo après 5 min)
        offset = (offset + drift) * breath
        # torsion et ondes : le shader compte l'abscisse le long du trajet
        # depuis l'objet (précise en float32), la phase rend l'abscisse
        # depuis l'origine du décor (float64). Sans elle, la spirale suivait
        # l'objet : le décor tournait sous lui pendant le voyage
        along = float(np.dot(path_o, path_d))
        twist_ph = 2.0 * math.pi * v["bg_vo_twist"] * 0.01 * along
        wave_ph = 2.0 * math.pi / v["bg_vo_wave_len"] * along
        sky = lin_hsv(v["bg_light_sky_hue"], 0.45, 1.0)
        return {
            "u_path_o": tuple(path_o),
            "u_path_m": tuple(path_m.T.flatten()),        # colonne-major pour GL
            "u_rot": tuple(rot.T.flatten()),              # colonne-major pour GL
            "u_offset": tuple(offset),
            "u_scale": scale,
            "u_detail": self._detail(v),
            "u_sun_dir": sun_dir,
            "u_sun_col": tuple(float(c) for c in sun),
            # ciel et brume accordés à la palette (même dosage)
            "u_sky_col": tuple(float(c) * v["bg_light_sky"] for c in palettes.fog_color(
                sky, self.palette, v["bg_comp_fog_tint"], hue_rot)),
            "u_hor_col": tuple(float(c) for c in palettes.fog_color(
                lin_hsv(v["bg_col_hor_hue"], 0.6, v["bg_col_hor_lum"]), self.palette,
                v["bg_comp_fog_tint"], hue_rot)),
            "u_shadow_col": tuple(c * v["bg_col_shadow"]
                                  for c in lin_hsv(v["bg_col_shadow_hue"], 0.8, 1.0)),
            "u_emit_col": tuple(c * v["bg_col_emit"]
                                for c in lin_hsv(v["bg_col_emit_hue"], 0.9, 1.0)),
            "u_center_col": tuple(c * v["bg_light_center"]
                                  * (1.0 + 2.0 * s["rms"] * dose)
                                  for c in lin_hsv(model_hue, 0.75, 1.0)),
            "u_flash": v["bg_col_flash"] * snare,
            "u_tint": tuple(float(c) for c in palettes.tint(v["bg_col_tint"])),
            "u_hue_rot": hue_rot,
            "u_edges": v["bg_mat_edges"],
            "u_vo_lip": volume_stretch(v),
            "u_vo_twist_ph": twist_ph % (2.0 * math.pi),
            "u_vo_wave_ph": wave_ph % (2.0 * math.pi),
            "u_grid_col": tuple(c * v["bg_grid_int"] * grid
                                for c in lin_hsv(v["bg_grid_hue"], 0.6, 1.0)),
            "u_vo_cave_thr": cave_threshold(v),
            # 2 (centré) × 8 texels / unité
            "u_vo_G": 16.0 * noise.gradient_norm(NOISE_QUANTILE),
        }

    @staticmethod
    def _detail(v: dict) -> float:
        """Plus petit détail visible (unités du fond) : en dessous de ~2
        pixels il se fond dans la brume au lieu de scintiller."""
        return v["bg_ld_scale"] * 0.15 * v["bg_comp_scale"]

    def step(self, p: dict, features: dict, bands, dt: float,
             model_hue: float = 0.0) -> Frame:
        """p : réglages du panneau ; bands : `Routed.bands` (11 sources) ;
        model_hue : teinte du rim du modèle (harmonie, lumière centrale)."""
        dt = max(float(dt), 1e-4)
        base = macros.apply(p)
        self._listen(features, bands, dt, base["bg_mov_inertia"])
        self.values = v = self._slew(self._react(base), dt)
        self._advance(v, features, dt)
        self._update_palette(v)
        uniforms = {q.uniform: float(v[q.key]) for q in SHADER_PARAMS}
        uniforms.update(self._derived(v, model_hue, p.get("show_grid", 0.0) >= 0.5))
        return Frame(
            visible=v["bg_visible"] >= 0.5,
            quality=int(v["bg_quality"]),
            res_index=min(max(int(v["bg_res"]), 0), len(RESOLUTIONS) - 1),
            uniforms=uniforms,
            palette=self.palette,
            palette_rev=self.palette_rev,
            bloom=v["bg_comp_bloom"],
            solid=(v["bg_r"] / 255.0, v["bg_g"] / 255.0, v["bg_b"] / 255.0),
            parallax=v["bg_comp_parallax"],
            light_model=int(v["bg_light_mode"]) == 0,
            light_int=v["bg_light_int"] / 1.5,
            same_space=v["bg_comp_far"] < 1e-3,
            history=temporal.max_weight(v["bg_temporal"]),
            structure=tuple(round(v[k], 3) for k in STRUCTURE_KEYS),
            terrain={k: v[k] for k in terrain.SETTINGS},
        )
