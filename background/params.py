"""État du fond d'une frame à l'autre (T16), sans GL.

`BackgroundState.step()` prend les réglages du panneau, les features audio
lissées et les sources du routing, et rend tout ce que la passe GL envoie
au shader : macros appliquées, réaction musicale dosée par « Réaction à la
musique », variation bornée (`slew`), mouvements propres intégrés en dt
(rotation, défilement, dérive, cycle de teinte), horloge des beats,
couleurs dérivées et palette. Pour la chaîne d'étages (passe 2), il
calcule aussi ce qui ne change pas d'un pas de marche à l'autre : rotation
de pliage KIFS, directions des satellites, portée des formes (bornée à
leur cellule), étirement dû aux déformations (pas de marche raccourcis).

Horloge des beats : comptée ici (retour de `beat_phase` à 0) plutôt que dans
le `Reactor` ; elle ne suit les vrais beats que si la musique a le droit
d'agir (dose > 0 et BPM connu), sinon elle tourne seule à 120 BPM : à dose
nulle, le fond ne dépend plus de l'audio.
"""

import math
from dataclasses import dataclass

import numpy as np

from background import macros, noise, palettes
from background.schema import BREATH_BEATS, MAX_SATELLITES, PARAMS, RESOLUTIONS

# ordre des sources dans `Routed.bands` (app/routing.py) : 7 bandes EQ,
# BEAT, KICK, SNARE, ÉNERGIE
BAND_SOURCES = ("sub", "bass", "lowmid", "mid", "highmid", "presence",
                "treble", "beat", "kick", "snare", "energy")
FREE_BPM = 120.0          # horloge des beats sans tempo
DRIFT_AMP = 0.3           # amplitude de la dérive (unités du fond)
GROUND = 2                # mode d'Espace « Grille au sol » (voyage à l'horizontale)
KIFS = 5                  # mode d'Espace « Pliage KIFS »
KIFS_AXIS = np.array([1.0, 1.6, 0.7]) / np.linalg.norm([1.0, 1.6, 0.7])
KIFS_SIZE = 2.0           # taille des formes dans l'espace plié (× Taille)
SINGLE = 0                # mode d'Espace « Objet seul »
SINGLE_SIZE = 2.5         # Objet seul : la grande forme fait 2,5 périodes
ORBIT_AXIS = np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
# portée d'une primitive de « rayon » 1 (voir shape() dans chain.glsl) ;
# fil de fer et croix y ajoutent l'épaisseur des barres
PRIM_REACH = (1.0, 1.39, 1.39, 1.39, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0)
THIN_PRIMS = (3, 8, 9)
NOISE_GRAD = None         # plus forte pente du bruit 3D, calculée au premier usage
AUDIO_PARAMS = [p for p in PARAMS if p.audio is not None]
SLEW_PARAMS = [p for p in PARAMS if p.max_slew is not None]
SHADER_PARAMS = [p for p in PARAMS if p.shader]
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
    """Direction du voyage (unitaire) ; azimut 0 = vers le fond de l'écran
    de la vue de départ. En Grille au sol, le voyage reste à l'horizontale."""
    az, el = math.radians(v["bg_mov_dir_az"]), math.radians(v["bg_mov_dir_el"])
    if int(v["bg_sp_mode"]) == GROUND:
        el = 0.0
    return np.array([math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)])


def lin_hsv(h, s, v) -> tuple:
    return tuple(float(c) for c in palettes.to_linear(palettes.hsv(h, s, v)))


def fibonacci_dirs(n: int) -> np.ndarray:
    """n directions réparties régulièrement sur la sphère (n, 3)."""
    if n <= 0:
        return np.zeros((0, 3))
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    a = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(a), y, r * np.sin(a)], axis=1)


def kifs_rotation(angle_deg: float) -> np.ndarray:
    """Rotation appliquée à chaque pliage KIFS (axe fixe, angle réglé)."""
    return rotation(KIFS_AXIS, math.radians(angle_deg))


def noise_gradient() -> float:
    global NOISE_GRAD
    if NOISE_GRAD is None:
        NOISE_GRAD = noise.max_gradient(noise.texture())
    return NOISE_GRAD


def shape_reach(v: dict) -> tuple:
    """Portée d'une instance en fractions de période, avant réduction :
    (toutes directions, horizontale, verticale). Le shader la borne à la
    demi-cellule et au couloir du trajet : aucune forme n'est tranchée."""
    r = v["bg_sh_size"] * (1.0 + 0.4 * v["bg_sp_disorder"]) * (1.0 + 0.5 * v["bg_sh_wave"])
    prims = (int(v["bg_sh_a"]), int(v["bg_sh_b"]))
    ext = max(PRIM_REACH[i] for i in prims) * r
    if any(i in THIN_PRIMS for i in prims):
        ext += v["bg_sh_thick"]
    ext += (v["bg_sh_round"] + 0.5 * v["bg_sh_shell"]) * r
    sx, sy, sz = v["bg_sh_sx"], v["bg_sh_sy"], v["bg_sh_sz"]
    sy_max = sy * (1.0 + 3.0 * v["bg_sp_eq"]) if int(v["bg_sp_mode"]) == GROUND else sy
    n = int(v["bg_fu_count"])
    sat = 0.0
    if n > 0:
        sat = (v["bg_fu_spread"] + v["bg_fu_rad"]) * r
        if int(v["bg_fu_combo"]) == 0:          # l'union douce gonfle un peu
            sat += 0.5 * v["bg_fu_k"] * r * math.log(n + 1.0)
        else:                                   # creuser / couper : dans la forme
            sat = 0.0
    full = max(ext * max(sx, sy_max, sz), sat)
    return full, max(ext * max(sx, sz), sat), max(ext * sy_max, sat)


def deform_stretch(v: dict) -> float:
    """Étirement maximal dû aux ondes et au warp (les pas de marche sont
    divisés d'autant) ; torsion et courbure dépendent du point : le shader
    les calcule. Le terme en 1/période vient de l'extinction des
    déformations près du couloir (sur une période)."""
    P = v["bg_sp_period"]
    amp = v["bg_dx_wave_amp"] * P
    wave = 1.0 + 2.0 * math.pi * amp / v["bg_dx_wave_len"] * 1.42 + 1.5 * amp / P
    warp_amp = v["bg_dx_warp_amp"]
    texels = noise.SIZE * v["bg_dx_warp_freq"] / noise.CELLS     # texels par unité
    warp = 1.0
    if warp_amp > 0.0:
        warp += (2.0 * warp_amp * noise_gradient() * texels * int(v["bg_dx_warp_oct"])
                 * math.sqrt(3.0) + 1.5 * warp_amp / P)
    return wave * warp


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


class BackgroundState:
    def __init__(self):
        self.values = None                         # après macros + musique + slew
        self.sources = dict.fromkeys(BAND_SOURCES + ("rms",), 0.0)
        self.angle = 0.0                           # rotation propre (rad)
        self.travel = np.zeros(3)                  # chemin parcouru par l'objet (unités)
        self.drift = 0.0                           # phase de la dérive
        self.hue_phase = 0.0                       # cycle de teinte (tours)
        self.spin = 0.0                            # rotation lente des instances (tours)
        self.orbit = 0.0                           # orbite des satellites (tours)
        self.wave_phase = 0.0                      # phase des ondes (tours)
        self.time = 0.0                            # temps du fond (s), figé par « Gel »
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
        """Voyage de l'objet, mouvements propres et horloge des beats."""
        if v["bg_mov_freeze"] >= 0.5:
            return
        # l'objet avance dans le décor fixe (float64 : des heures de voyage
        # restent précises une fois envoyées en float32)
        self.travel += travel_dir(v) * v["bg_mov_travel"] * dt
        bpm = float(features.get("bpm", 0.0) or 0.0)
        speed = v["bg_mov_speed"]
        if bpm > 0.0:
            speed *= 1.0 + v["bg_mov_sync"] * (bpm / FREE_BPM - 1.0)
        self.time += dt * speed
        self.angle = (self.angle + 2.0 * math.pi * v["bg_mov_rot"] / 60.0 * speed * dt) \
            % (2.0 * math.pi)
        self.drift += v["bg_mov_drift"] * speed * dt
        self.hue_phase = (self.hue_phase + v["bg_col_cycle"] / 60.0 * speed * dt) % 1.0
        self.spin = (self.spin + v["bg_sh_spin"] / 60.0 * speed * dt) % 1.0
        self.orbit = (self.orbit + v["bg_fu_orbit"] / 60.0 * speed * dt) % 1.0
        self.wave_phase = (self.wave_phase + v["bg_dx_wave_speed"] * speed * dt) % 1.0
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

    def _derived(self, v: dict, model_hue: float) -> dict:
        """Uniforms calculés (matrices, couleurs, horloges)."""
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
        hue = (v["bg_col_hue"] + self.hue_phase
               + palettes.harmony_shift(int(v["bg_col_harmony"]), self.palette_hue,
                                        model_hue)
               + v["bg_col_music"] * (s["treble"] - s["bass"]) * 0.25 * dose)
        eq_gain = min(1.0, 3.0 * dose)
        scale = v["bg_comp_scale"] * breath
        # trajet de l'objet dans l'espace des formes : une droite fixe (la
        # dérive l'en écarte un peu sans la déplacer) ; repère (u, v, d)
        path_d = travel_dir(v)
        path_u = np.cross(path_d, (0.0, 1.0, 0.0))
        if np.linalg.norm(path_u) < 1e-3:
            path_u = np.array([1.0, 0.0, 0.0])
        path_u /= np.linalg.norm(path_u)
        path_m = np.array([path_u, np.cross(path_d, path_u), path_d])
        path_o = offset / scale
        offset = offset + drift
        n_sat = min(int(v["bg_fu_count"]), MAX_SATELLITES)
        sats = np.zeros((MAX_SATELLITES, 3), dtype="f4")
        if n_sat:
            sats[:n_sat] = fibonacci_dirs(n_sat) @ rotation(ORBIT_AXIS,
                                                            2.0 * math.pi * self.orbit).T
        reach, reach_h, reach_v = shape_reach(v)
        fs = v["bg_sp_fscale"]
        return {
            "u_path_o": tuple(path_o),
            "u_path_m": tuple(path_m.T.flatten()),        # colonne-major pour GL
            "u_rot": tuple(rot.T.flatten()),              # colonne-major pour GL
            "u_offset": tuple(offset),
            "u_scale": scale,
            "u_detail": self._detail(v),
            "u_beat": self.clock,
            "u_time": self.time,
            "u_sun_dir": sun_dir,
            "u_sun_col": tuple(float(c) for c in sun),
            "u_sky_col": tuple(c * v["bg_light_sky"]
                               for c in lin_hsv(v["bg_light_sky_hue"], 0.45, 1.0)),
            "u_hor_col": lin_hsv(v["bg_col_hor_hue"], 0.6, v["bg_col_hor_lum"]),
            "u_shadow_col": tuple(c * v["bg_col_shadow"]
                                  for c in lin_hsv(v["bg_col_shadow_hue"], 0.8, 1.0)),
            "u_emit_col": tuple(c * v["bg_col_emit"]
                                for c in lin_hsv(v["bg_col_emit_hue"], 0.9, 1.0)),
            "u_center_col": tuple(c * v["bg_light_center"]
                                  * (1.0 + 2.0 * s["rms"] * dose)
                                  for c in lin_hsv(model_hue, 0.75, 1.0)),
            "u_flash": v["bg_col_flash"] * snare,
            "u_tint": tuple(float(c) for c in palettes.tint(v["bg_col_tint"])),
            "u_hue_rot": 2.0 * math.pi * (hue % 1.0),
            "u_eq": np.array([s[b] * eq_gain for b in BAND_SOURCES[:7]], dtype="f4"),
            "u_edges": v["bg_mat_edges"],
            "u_kifs_rot": tuple(kifs_rotation(v["bg_sp_angle"]).T.flatten()),
            "u_kifs_off": tuple(np.array([v["bg_sp_off_x"], v["bg_sp_off_y"],
                                          v["bg_sp_off_z"]]) * (fs - 1.0)),
            "u_kifs_inv": 1.0 / fs ** int(v["bg_sp_iter"]),
            "u_sat": sats.flatten(),
            "u_reach": reach,
            "u_reach_h": reach_h,
            "u_reach_v": reach_v,
            "u_dx_lip": deform_stretch(v),
            "u_dx_phase": 2.0 * math.pi * self.wave_phase,
            "u_sh_angle": math.radians(v["bg_sh_rot"]) + 2.0 * math.pi * self.spin,
        }

    @staticmethod
    def _detail(v: dict) -> float:
        """Plus petit détail des formes (unités du fond) : en dessous de ~2
        pixels il se fond dans la brume au lieu de scintiller."""
        P = v["bg_sp_period"]
        if int(v["bg_sp_mode"]) == SINGLE:
            P *= SINGLE_SIZE
        r = v["bg_sh_size"] * P
        prims = (int(v["bg_sh_a"]), int(v["bg_sh_b"]))
        size = 0.5 * r
        if any(i in THIN_PRIMS for i in prims):
            size = min(size, v["bg_sh_thick"] * P)
        if 4 in prims:
            size = min(size, 0.28 * r)
        if int(v["bg_fu_count"]) > 0:
            size = min(size, v["bg_fu_rad"] * r)
        if int(v["bg_sp_mode"]) == KIFS:     # copies réduites à chaque pliage
            size *= KIFS_SIZE / v["bg_sp_fscale"] ** int(v["bg_sp_iter"])
        return max(size, 1e-3) * v["bg_comp_scale"]

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
        uniforms.update(self._derived(v, model_hue))
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
        )
