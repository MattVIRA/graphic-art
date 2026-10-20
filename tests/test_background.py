"""Fond SDF (T16) : schéma des paramètres, chaîne d'étages, macros, dé,
recettes, état d'une frame à l'autre, bruit 3D, palettes, résolution
automatique — tout ce qui se teste sans GL."""

import math
import os

import numpy as np
import pytest

from background import macros, noise, palettes, params, quality, schema
from background.params import BackgroundState, travel_dir
from background.pass_ import SHADER_DIR, SHADER_FILES, fragment_source
from gfx.camera import OrbitCamera, look_at

QUIET = {"rms": 0.0, "bpm": 0.0, "beat_phase": 0.0}
LOUD = {"rms": 0.9, "bpm": 128.0, "beat_phase": 0.4}


def settings(**kw) -> dict:
    p = schema.defaults()
    p.update({"bg_visible": 1.0, **kw})
    return p


def run(state, p, features=QUIET, bands=None, dt=1 / 60, steps=1):
    bands = [0.0] * 11 if bands is None else bands
    for _ in range(steps):
        frame = state.step(p, features, bands, dt)
    return frame


def same_uniforms(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        np.allclose(np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float))
        for k in a)


# ── schéma ────────────────────────────────────────────────────────────
def test_schema_is_consistent():
    assert schema.validate() == []


def test_every_shader_param_is_used_in_glsl():
    """Un réglage envoyé au shader doit y servir (sinon slider mort)."""
    glsl = "".join(open(os.path.join(SHADER_DIR, n), encoding="utf-8").read()
                   for n in os.listdir(SHADER_DIR))
    unused = [p.key for p in schema.PARAMS if p.shader and p.uniform not in glsl]
    assert unused == []


def test_fragment_source_has_defines_and_declarations():
    for level in range(len(quality.LEVELS)):
        src = fragment_source(level)
        assert src.startswith("#version 330")
        assert f"#define MAX_STEPS {quality.LEVELS[level]['MAX_STEPS']}" in src
        assert "uniform float u_sp_period;" in src


def test_shader_files_are_the_chain():
    """Un seul shader de forme : la chaîne ; les 4 scènes de la passe 1 sont parties."""
    assert sorted(os.listdir(SHADER_DIR)) == sorted(SHADER_FILES)
    assert "geometric.glsl" not in SHADER_FILES


def test_python_and_glsl_constants_agree():
    chain = open(os.path.join(SHADER_DIR, "chain.glsl"), encoding="utf-8").read()
    assert f"const float KIFS_SIZE = {params.KIFS_SIZE};" in chain
    assert f"const float SINGLE_SIZE = {params.SINGLE_SIZE};" in chain
    assert f"const float NOISE_CELLS = {float(noise.CELLS)};" in chain
    assert f"u_sat[{schema.MAX_SATELLITES}]" in chain
    assert f"i < {schema.MAX_FOLDS}" in chain


# ── chaîne d'étages ───────────────────────────────────────────────────
def test_old_scenes_are_gone():
    keys = schema.defaults()
    assert not any(k.startswith("bg_geo_") for k in keys)
    assert "bg_type" not in keys
    assert not hasattr(schema, "TYPES") and not hasattr(schema, "GEO_VARIANTS")


@pytest.mark.parametrize("group, mode", [("sp", "bg_sp_mode"), ("dx", "bg_dx_axis"),
                                         ("sh", "bg_sh_a"), ("fu", "bg_fu_combo")])
def test_each_stage_has_a_mode_dice_and_lock(group, mode):
    assert schema.BY_KEY[mode].group == group and schema.BY_KEY[mode].options
    assert group in schema.DICE_GROUPS and group in schema.LOCK_KEYS
    assert len(schema.params_of(group)) >= 3


def test_panel_defaults_include_background():
    from gfx.controls import DEFAULTS
    for key, value in schema.defaults().items():
        assert DEFAULTS[key] == value
    assert DEFAULTS["bg_visible"] == 0.0          # ancien preset → fond masqué
    assert DEFAULTS["show_grid"] == 1.0           # la grille garde sa clé
    assert (DEFAULTS["bg_r"], DEFAULTS["bg_g"], DEFAULTS["bg_b"]) == (4, 4, 10)


# ── macros ────────────────────────────────────────────────────────────
def test_neutral_macros_keep_values():
    p = schema.defaults()
    out = macros.apply(p)
    for q in schema.PARAMS:
        assert out[q.key] == pytest.approx(macros.bound(q, p[q.key]))


@pytest.mark.parametrize("key", schema.MACRO_KEYS)
def test_each_macro_moves_something_and_stays_in_range(key):
    base = macros.apply(schema.defaults())
    for m in (0.0, 1.0):
        out = macros.apply(dict(schema.defaults(), **{key: m}))
        for q in schema.PARAMS:
            assert q.lo <= out[q.key] <= q.hi
        assert any(out[k] != base[k] for k in out)


def test_all_macros_at_extremes_stay_bounded():
    for m in (0.0, 1.0):
        p = dict(schema.defaults(), **dict.fromkeys(schema.MACRO_KEYS, m))
        out = macros.apply(p)
        for q in schema.PARAMS:
            assert q.lo <= out[q.key] <= q.hi


def test_log_macro_is_a_factor():
    p = schema.BY_KEY["bg_comp_scale"]
    out = macros.apply(dict(schema.defaults(), bg_m_scale=1.0))
    assert out["bg_comp_scale"] == pytest.approx(p.default * math.exp(p.macros["scale"]))


# ── dé ────────────────────────────────────────────────────────────────
def test_same_seed_same_values():
    assert schema.random_values(42) == schema.random_values(42)
    assert schema.random_values(42) != schema.random_values(43)


def test_draws_stay_in_their_range():
    for seed in range(30):
        for key, v in schema.random_values(seed).items():
            p = schema.BY_KEY[key]
            a, b = p.rand
            assert a - 1e-9 <= v <= b + 1e-9 or (p.zero > 0 and v == 0.0), key


def test_same_seed_same_chain_modes_included():
    a, b = schema.random_values(123), schema.random_values(123)
    for key in ("bg_sp_mode", "bg_sh_a", "bg_sh_b", "bg_fu_combo", "bg_dx_axis"):
        assert a[key] == b[key]


@pytest.mark.parametrize("key", ["bg_sp_mode", "bg_fu_combo", "bg_dx_axis"])
def test_mode_weights_are_respected(key):
    p = schema.BY_KEY[key]
    n = 1000
    draws = [schema.random_values(s, groups=[p.group])[key] for s in range(n)]
    total = sum(p.weights)
    for i, w in enumerate(p.weights):
        share = draws.count(float(p.rand[0] + i)) / n
        assert share == pytest.approx(w / total, abs=0.05), (key, i)


def test_effects_are_off_about_half_the_time():
    n = 1000
    for key in ("bg_dx_twist", "bg_dx_warp_amp", "bg_dx_wave_amp", "bg_dx_bend"):
        off = sum(schema.random_values(s, groups=["dx"])[key] == 0.0 for s in range(n))
        assert off / n == pytest.approx(0.5, abs=0.05), key
    no_sat = sum(schema.random_values(s, groups=["fu"])["bg_fu_count"] == 0.0
                 for s in range(n))
    assert no_sat / n == pytest.approx(0.4, abs=0.05)


def test_stage_lock_keeps_the_stage():
    locked = schema.random_values(9, locked=["sp", "sh"])
    assert not any(schema.BY_KEY[k].group in ("sp", "sh") for k in locked)
    assert any(schema.BY_KEY[k].group == "fu" for k in locked)


def test_group_draw_matches_full_draw_and_respects_locks():
    full = schema.random_values(7)
    col = schema.random_values(7, groups=["col"])
    assert col and all(schema.BY_KEY[k].group == "col" for k in col)
    assert all(full[k] == v for k, v in col.items())
    locked = schema.random_values(7, locked=["col", "geo"])
    assert not any(schema.BY_KEY[k].group in ("col", "geo") for k in locked)


def test_settings_and_macros_are_never_drawn():
    keys = schema.random_values(1)
    assert "bg_visible" not in keys and "bg_quality" not in keys
    assert not any(k in keys for k in schema.MACRO_KEYS)


# ── état d'une frame à l'autre ────────────────────────────────────────
def test_no_music_means_no_dependence_on_audio():
    p = settings(bg_mov_music=0.0)
    a = run(BackgroundState(), p, QUIET, [0.0] * 11, steps=30)
    b = run(BackgroundState(), p, LOUD, [1.0, 0.8, 0.5, 0.5, 0.4, 0.3, 0.3, 1, 1, 1, 1],
            steps=30)
    assert same_uniforms(a.uniforms, b.uniforms)


def test_music_reacts_but_kick_never_touches_structure():
    p = settings(bg_mov_music=1.0)
    calm = run(BackgroundState(), p, LOUD, [0.0] * 11, steps=30)
    kick = [0.0] * 8 + [1.0, 1.0, 0.0]          # KICK + SNARE, pas d'ÉNERGIE
    hit = run(BackgroundState(), p, LOUD, kick, steps=30)
    assert hit.uniforms["u_fu_spread"] > calm.uniforms["u_fu_spread"]  # exact : réagit
    for q in schema.PARAMS:
        if q.kind in schema.SLOW_ONLY and q.shader:
            assert hit.uniforms[q.uniform] == calm.uniforms[q.uniform], q.key


def test_slew_limits_structural_jumps():
    st = BackgroundState()
    run(st, settings())
    dt = 0.1
    run(st, settings(bg_sp_period=12.0), dt=dt)
    p = schema.BY_KEY["bg_sp_period"]
    moved = st.values["bg_sp_period"] - p.default
    assert 0.0 < moved <= p.max_slew * dt + 1e-9


def test_first_frame_is_not_slewed():
    st = BackgroundState()
    run(st, settings(bg_sp_period=12.0))
    assert st.values["bg_sp_period"] == 12.0


def test_motion_stays_bounded_over_a_long_session():
    st = BackgroundState()
    p = settings(bg_mov_speed=4.0, bg_mov_rot=1.0, bg_col_cycle=1.0)
    run(st, p, dt=1.0, steps=3600)
    assert 0.0 <= st.angle < 2.0 * math.pi
    assert 0.0 <= st.hue_phase < 1.0


def test_the_object_travels_through_a_fixed_decor():
    st = BackgroundState()
    p = settings(bg_mov_travel=2.0, bg_mov_drift=0.0)
    first = run(st, p)
    later = run(st, p, dt=0.5, steps=10)                 # 5 s plus tard
    assert np.allclose(st.travel, travel_dir(p) * 2.0 * (5.0 + 1 / 60))
    # le trajet (donc le couloir vide du décor) ne bouge pas
    d = np.asarray(first.uniforms["u_path_m"]).reshape(3, 3).T[2]
    moved = np.asarray(later.uniforms["u_path_o"]) - np.asarray(first.uniforms["u_path_o"])
    assert np.allclose(np.cross(moved, d), 0.0, atol=1e-9)
    assert np.allclose(np.asarray(later.uniforms["u_rot"]), np.eye(3).T.flatten())  # décor fixe


def test_default_direction_goes_into_the_screen():
    assert np.allclose(travel_dir(schema.defaults()), (0.0, 0.0, -1.0))
    climbing = dict(schema.defaults(), bg_mov_dir_el=45.0)
    assert travel_dir(climbing)[1] > 0.5
    ground = dict(climbing, bg_sp_mode=2.0)            # Grille au sol : le sol reste au sol
    assert travel_dir(ground)[1] == 0.0


def test_beats_are_counted_when_music_may_act():
    st = BackgroundState()
    p = settings(bg_mov_music=0.5)
    for phase in (0.2, 0.5, 0.8, 0.05, 0.4, 0.9, 0.1):
        run(st, p, {"rms": 0.5, "bpm": 120.0, "beat_phase": phase})
    assert st.beats == 2
    assert st.clock == pytest.approx(2.1)


def test_freeze_stops_time():
    st = BackgroundState()
    run(st, settings(), steps=10)
    t, clock = st.time, st.clock
    run(st, settings(bg_mov_freeze=1.0), steps=10)
    assert (st.time, st.clock) == (t, clock)


def test_hidden_frame_and_solid_color():
    f = BackgroundState().step(schema.defaults(), QUIET, [0.0] * 11, 1 / 60)
    assert not f.visible
    assert f.solid == (4 / 255.0, 4 / 255.0, 10 / 255.0)


def test_palette_is_rebuilt_only_when_it_changes():
    st = BackgroundState()
    rev = run(st, settings()).palette_rev
    assert run(st, settings()).palette_rev == rev
    assert run(st, settings(bg_col_mode=7.0)).palette_rev == rev + 1


def test_same_space_and_model_light_by_default():
    f = run(BackgroundState(), settings())
    assert f.same_space and f.light_model
    assert not run(BackgroundState(), settings(bg_comp_far=0.5)).same_space


# ── recettes ──────────────────────────────────────────────────────────
def test_recipes_are_new_chains_in_range():
    assert len(schema.AMBIANCES) >= 8
    modes = set()
    for name, vals in schema.AMBIANCES.items():
        for k, v in vals.items():
            p = schema.BY_KEY[k]
            assert p.lo <= v <= p.hi, (name, k)
        assert vals["bg_visible"] == 1.0
        modes.add(vals.get("bg_sp_mode", schema.BY_KEY["bg_sp_mode"].default))
    assert len(modes) == len(schema.SPACE_MODES)       # chaque Espace a sa recette


def test_recipes_render_without_error():
    for vals in schema.AMBIANCES.values():
        f = run(BackgroundState(), settings(**vals), steps=3)
        assert all(np.isfinite(np.asarray(v, dtype=float)).all()
                   for v in f.uniforms.values())


# ── macros de forme sur les étages ────────────────────────────────────
@pytest.mark.parametrize("name", [n for n, *_ in schema.MACROS_FORM])
def test_form_macros_move_a_stage(name):
    base = macros.apply(schema.defaults())
    stage = {"sp", "dx", "sh", "fu", "mat"}
    for m in (0.0, 1.0):
        out = macros.apply(dict(schema.defaults(), **{f"bg_m_{name}": m}))
        moved = [k for k in out if out[k] != base[k] and schema.BY_KEY[k].group in stage]
        assert moved, (name, m)


def test_kick_and_snare_never_reach_structure():
    for p in schema.PARAMS:
        if p.audio and p.audio[0] in schema.FAST_SOURCES:
            assert p.kind not in ("topology", "int", "deform"), p.key


def test_zero_dose_means_no_audio_on_the_chain():
    p = settings(**schema.AMBIANCES["Essaim"], bg_mov_music=0.0)
    a = run(BackgroundState(), p, QUIET, [0.0] * 11, steps=20)
    b = run(BackgroundState(), p, LOUD, [1.0] * 11, steps=20)
    for key in ("u_fu_spread", "u_sh_morph", "u_dx_wave_amp", "u_fu_k", "u_sh_wave"):
        assert a.uniforms[key] == b.uniforms[key], key


# ── portée et étirement (aucune forme tranchée, pas raccourcis) ───────
def test_reach_grows_with_size_stretch_and_satellites():
    v = macros.apply(schema.defaults())
    base = params.shape_reach(v)[0]
    assert params.shape_reach(dict(v, bg_sh_sx=2.5))[0] > base
    assert params.shape_reach(dict(v, bg_sh_size=0.4))[0] > base
    assert params.shape_reach(dict(v, bg_fu_count=6.0, bg_fu_spread=2.0))[0] > base
    full, horiz, vert = params.shape_reach(dict(v, bg_sp_mode=2.0, bg_sp_eq=1.0))
    assert vert > horiz          # l'égaliseur étire vers le haut seulement


def test_deform_stretch_is_one_without_deformation_and_grows():
    v = macros.apply(schema.defaults())
    assert params.deform_stretch(v) == 1.0
    assert params.deform_stretch(dict(v, bg_dx_wave_amp=0.3)) > 1.0
    assert params.deform_stretch(dict(v, bg_dx_warp_amp=1.5)) > \
        params.deform_stretch(dict(v, bg_dx_warp_amp=0.5)) > 1.0


def test_kifs_rotation_is_orthogonal():
    for angle in (0.0, 20.0, 47.5, 90.0):
        r = params.kifs_rotation(angle)
        assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)


def test_kifs_copy_centre_is_found_through_the_folds():
    """Le centre d'une copie calculé comme dans chain.glsl : replié, il
    tombe exactement sur la forme (q = 0)."""
    rot, s = params.kifs_rotation(15.0), 2.0
    off = np.array([1.0, 1.0, 0.6]) * (s - 1.0)

    def fold(p, n=4):
        q, t = p.copy(), np.eye(3)
        for _ in range(n):
            sign = np.where(q >= 0.0, 1.0, -1.0)
            q = np.abs(q)
            t = (t * sign[None, :]) @ rot.T
            q = s * (rot @ q) - off
        return q, t

    rng = np.random.default_rng(3)
    for _ in range(200):
        p = rng.uniform(-2.0, 2.0, 3)
        q, t = fold(p)
        centre = p - t @ q / s ** 4
        assert np.allclose(fold(centre)[0], 0.0, atol=1e-9)


def test_satellites_orbit_and_stay_unit():
    st = BackgroundState()
    p = settings(bg_fu_count=5.0, bg_fu_orbit=0.5)
    a = np.asarray(run(st, p).uniforms["u_sat"]).reshape(-1, 3)
    b = np.asarray(run(st, p, dt=1.0, steps=10).uniforms["u_sat"]).reshape(-1, 3)
    assert np.allclose(np.linalg.norm(a[:5], axis=1), 1.0)
    assert np.allclose(a[5:], 0.0)
    assert not np.allclose(a[:5], b[:5])


# ── bruit 3D ──────────────────────────────────────────────────────────
def test_noise_texture_is_deterministic_bounded_and_periodic():
    tex = noise.texture()
    assert tex.shape == (noise.SIZE,) * 3 + (3,) and tex.dtype == np.float32
    assert np.array_equal(tex, noise.texture())
    assert tex.min() >= 0.0 and tex.max() <= 1.0
    # raccord : du dernier texel au premier, le pas ressemble aux autres
    inner = np.abs(np.diff(tex, axis=2)).max()
    seam = np.abs(tex[:, :, 0] - tex[:, :, -1]).max()
    assert seam <= inner * 1.05
    assert 0.0 < noise.max_gradient(tex) < 0.5          # lisse


# ── palettes ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", range(len(schema.PALETTE_MODES)))
def test_palettes_are_finite(mode):
    rgb = palettes.build(mode, 0, (0.6, 0.7, 0.3), (0.5, 0.8, 0.8), (0.9, 0.6, 1.0))
    assert rgb.shape == (palettes.SIZE, 3)
    assert np.isfinite(palettes.to_linear(rgb)).all()


def test_harmony_targets_the_model_hue():
    assert palettes.harmony_shift(0, 0.3, 0.7) == 0.0                  # Libre
    assert palettes.harmony_shift(1, 0.3, 0.7) == pytest.approx(0.4)   # même teinte
    assert palettes.harmony_shift(2, 0.3, 0.7) == pytest.approx(0.9)   # complémentaire
    red = palettes.hsv(np.zeros(8), np.ones(8), np.linspace(0.2, 1, 8))
    assert palettes.mean_hue(red) == pytest.approx(0.0, abs=1e-6)


def test_palette_is_read_without_mirror_or_repeat():
    """Passe 3 : plus de « Répétition de la palette » ni de lecture en
    miroir (les bandes de dégradé qui alternaient)."""
    assert "bg_col_repeat" not in schema.BY_KEY
    main = open(os.path.join(SHADER_DIR, "main.glsl"), encoding="utf-8").read()
    assert "x = clamp(x, 0.0, 1.0);" in main and "fract(x * 0.5)" not in main
    assert schema.BY_KEY["bg_col_steps"].rand is None      # paliers jamais tirés
    assert "Nappes de bruit" in schema.COLOR_SOURCES
    assert schema.COLOR_SOURCES[int(schema.BY_KEY["bg_col_source"].default)] == \
        "Nappes de bruit"


def test_hue_rotate_matches_the_shader_and_keeps_grey():
    grey = np.array([0.3, 0.3, 0.3])
    assert np.allclose(palettes.hue_rotate(grey, 1.3), grey)
    red = np.array([1.0, 0.0, 0.0])
    assert np.allclose(palettes.hue_rotate(red, 2.0 * math.pi), red)
    assert np.allclose(palettes.hue_rotate(red, 2.0 * math.pi / 3.0), [0.0, 1.0, 0.0],
                       atol=1e-9)


def test_fog_takes_the_palette_colour_at_the_same_luminance():
    horizon = np.array([0.02, 0.02, 0.05])
    pal = np.tile([0.8, 0.3, 0.1], (palettes.SIZE, 1))     # palette orangée
    assert np.allclose(palettes.fog_color(horizon, pal, 0.0, 0.0), horizon)
    full = palettes.fog_color(horizon, pal, 1.0, 0.0)
    assert full @ palettes.LUMA == pytest.approx(horizon @ palettes.LUMA)
    assert full[0] > full[2]                                 # orangé, plus bleu
    st = BackgroundState()
    a = run(st, settings(bg_comp_fog_tint=0.0)).uniforms["u_hor_col"]
    b = run(BackgroundState(), settings(bg_comp_fog_tint=1.0)).uniforms["u_hor_col"]
    assert not np.allclose(a, b)


def test_wave_stretch_accounts_for_the_chained_ripples():
    v = macros.apply(schema.defaults())
    w = dict(v, bg_dx_wave_amp=0.2, bg_dx_wave_len=4.0)
    ka = 2.0 * math.pi * 0.2 * v["bg_sp_period"] / 4.0
    assert params.deform_stretch(w) >= (1.0 + ka) * (1.0 + 0.73 * ka)


def test_materials_set_their_reflections():
    assert schema.MATERIALS["Chrome"]["bg_mat_env"] > schema.MATERIALS["Argile"]["bg_mat_env"]


def test_light_temperature_and_tint():
    assert np.allclose(palettes.kelvin(6600.0), 1.0, atol=0.02)
    warm, cold = palettes.kelvin(2500.0), palettes.kelvin(10000.0)
    assert warm[0] > warm[2] and cold[2] > cold[0]
    lum = np.array([0.2126, 0.7152, 0.0722])
    for amount in (-1.0, 0.0, 1.0):
        assert palettes.tint(amount) @ lum == pytest.approx(1.0)


# ── résolution automatique ────────────────────────────────────────────
def test_auto_resolution_steps_down_then_back_up():
    auto = quality.AutoResolution(budget_ms=2.0)
    assert auto.step(2, True, None, 0.1) == 2
    for _ in range(4):
        auto.step(2, True, 5.0, 0.1)
    assert auto.step(2, True, 5.0, 0.1) == 1          # 0,5 s au-dessus du budget
    for _ in range(30):
        idx = auto.step(2, True, 5.0, 0.1)
    assert idx == 0                                    # jamais sous ¼
    for _ in range(25):
        idx = auto.step(2, True, 0.5, 0.1)
    assert idx >= 1                                    # de la marge : on remonte


def test_manual_quality_keeps_the_chosen_resolution():
    auto = quality.AutoResolution()
    assert auto.step(3, False, 50.0, 1.0) == 3
    assert quality.level(quality.AUTO) == 1


# ── parallaxe (caméra du fond) ────────────────────────────────────────
def test_parallax_amount_blends_the_camera():
    cam = OrbitCamera()
    cam.load_state({"yaw": 0.8, "pitch": 0.3, "target": [10.0, 0.0, 0.0], "zoom": 1.5})
    full = cam.view_matrix(800.0, 1.2)
    assert np.allclose(cam.view_matrix(800.0, 1.2, amount=1.0), full)
    fixed = cam.view_matrix(800.0, 1.2, amount=0.0)
    assert np.allclose(fixed, look_at((0.0, 0.0, 800.0 / 1.2), (0.0, 0.0, 0.0)))
    half = cam.view_matrix(800.0, 1.2, amount=0.5)
    assert np.allclose(half[:3, :3] @ half[:3, :3].T, np.eye(3))   # rotation pure
