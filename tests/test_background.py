"""Fond SDF (T1.16) : schéma des paramètres, paysage, macros, dé, recettes,
état d'une frame à l'autre, bruit 3D, palettes, résolution automatique —
tout ce qui se teste sans GL (les cartes du relief : tests/test_terrain.py)."""

import math
import os

import numpy as np
import pytest

from background import macros, noise, palettes, params, quality, schema, terrain
from background.params import BackgroundState, travel_dir
from background.pass_ import RESOLVE_FILE, SHADER_DIR, SHADER_FILES, fragment_source
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
        assert "uniform float u_ld_valley;" in src


def test_shader_files_are_the_landscape():
    """Un seul générateur : le paysage ; la chaîne d'étages et les mondes de
    bruit sont partis (+ la passe de résolution, passe 5)."""
    assert sorted(os.listdir(SHADER_DIR)) == sorted(SHADER_FILES + [RESOLVE_FILE])
    assert "landscape.glsl" in SHADER_FILES and "chain.glsl" not in SHADER_FILES


def test_python_and_glsl_constants_agree():
    src = open(os.path.join(SHADER_DIR, "landscape.glsl"), encoding="utf-8").read()
    assert f"const float NOISE_CELLS = {float(noise.CELLS)};" in src
    assert f"const float TILE = {terrain.TILE};" in src
    assert f"const float DETAIL_TILE = {terrain.DETAIL_TILE};" in src
    assert f"const float VALLEY_EDGE = {params.VALLEY_EDGE};" in src
    assert f"const float WARP_GAIN = {params.WARP_GAIN};" in src
    assert f"const float BUBBLE_MAX = {params.BUBBLE_MAX};" in src
    # uniforms posés par la passe GL (cartes) et le renderer (ombre du modèle)
    glsl = src + open(os.path.join(SHADER_DIR, "common.glsl"), encoding="utf-8").read()
    for name in ("u_ld_res", "u_ld_slope", "u_ld_dslope", "u_ld_dgain", "u_ld_damp",
                 "u_terrain", "u_terrain_fine", "u_model_c", "u_model_r",
                 "u_vo_lip", "u_vo_cave_thr", "u_vo_G"):
        assert f" {name};" in glsl or f" {name}; " in glsl, name


def test_decor_does_not_move_with_time_in_the_shaders():
    """Décor statique : plus d'horloge dans les shaders (flux, veines qui
    glissaient, poursuite au beat, vague)."""
    glsl = "".join(open(os.path.join(SHADER_DIR, n), encoding="utf-8").read()
                   for n in SHADER_FILES)
    assert "u_time" not in glsl and "u_beat" not in glsl


# ── paysage seul ──────────────────────────────────────────────────────
def test_only_the_landscape_is_left():
    keys = schema.defaults()
    for prefix in ("bg_geo_", "bg_sp_", "bg_dx_", "bg_sh_", "bg_fu_", "bg_nw_"):
        assert not any(k.startswith(prefix) for k in keys), prefix
    for key in ("bg_type", "bg_col_cycle", "bg_col_music", "bg_mat_chase",
                "bg_comp_void", "bg_mov_dir_el", "bg_ld_water", "bg_ld_ripple",
                "bg_ld_clarity"):
        assert key not in keys
    assert [g for g, _ in schema.GROUPS if g in ("sp", "dx", "sh", "fu", "nw")] == []


def test_landscape_and_volume_groups_have_dice_and_lock():
    for group in ("ld", "vo"):
        assert group in schema.DICE_GROUPS and group in schema.LOCK_KEYS
        assert len(schema.params_of(group)) >= 12


def test_volume_stretch_is_one_without_distortion_and_grows():
    v = macros.apply(schema.defaults())
    assert params.volume_stretch(v) == 1.0
    assert params.volume_stretch(dict(v, bg_vo_wave_amp=3.0)) > 1.0
    assert params.volume_stretch(dict(v, bg_vo_warp=0.8)) > \
        params.volume_stretch(dict(v, bg_vo_warp=0.3)) > 1.0
    # deux ondulations enchaînées : les étirements se multiplient
    w = dict(v, bg_vo_wave_amp=2.0, bg_vo_wave_len=10.0)
    ka = 2.0 * math.pi * 2.0 / 10.0
    assert params.volume_stretch(w) >= (1.0 + ka) * (1.0 + 0.73 * ka)


def test_caves_leave_the_asked_share_of_space():
    values = (noise.texture()[..., 0].astype(float) - 0.5) * 2.0
    v = macros.apply(schema.defaults())
    for caves in (0.2, 0.6, 1.0):
        carved = params.cave_threshold(dict(v, bg_vo_caves=caves, bg_vo_cave_mode=0.0))
        assert (values > carved).mean() == pytest.approx(0.5 * caves, abs=0.01)
        rock = params.cave_threshold(dict(v, bg_vo_caves=caves, bg_vo_cave_mode=1.0))
        assert (values > rock).mean() == pytest.approx(0.2 + 0.4 * caves, abs=0.01)


def test_noise_slope_quantile_is_below_the_maximum():
    assert 0.0 < noise.gradient_norm(99.0) < noise.gradient_norm(100.0) < 0.5


def test_terrain_params_are_the_terrain_settings():
    """Les réglages de classe `terrain` sont exactement ceux des cartes :
    aucun n'est envoyé au shader, aucun n'est oublié par le calcul."""
    kinds = {p.key for p in schema.PARAMS if p.kind == "terrain"}
    assert kinds == set(terrain.SETTINGS)
    assert set(terrain.DETAIL_SETTINGS) <= kinds
    assert not any(schema.BY_KEY[k].shader or schema.BY_KEY[k].audio for k in kinds)


def test_frame_carries_the_terrain_settings_after_macros():
    f = run(BackgroundState(), settings(bg_m_complexity=1.0))
    assert set(f.terrain) == set(terrain.SETTINGS)
    assert f.terrain["bg_ld_octaves"] > schema.BY_KEY["bg_ld_octaves"].default


def test_energy_macro_no_longer_touches_the_decor():
    """L'Énergie amplifiait le flux et accélérait l'horloge du fond : elle ne
    touche plus ni le relief ni la vitesse des mouvements propres."""
    base = macros.apply(schema.defaults())
    for m in (0.0, 1.0):
        out = macros.apply(dict(schema.defaults(), bg_m_energy=m))
        moved = {k for k in out if out[k] != base[k]}
        assert moved and not any(schema.BY_KEY[k].group == "ld" for k in moved)
        assert "bg_mov_speed" not in moved


def test_decor_is_static_without_music_and_travel():
    st = BackgroundState()
    p = settings(bg_mov_travel=0.0, bg_mov_music=0.0)
    a = run(st, p, LOUD, [1.0] * 11)
    b = run(st, p, LOUD, [0.3] * 11, dt=0.5, steps=20)
    assert same_uniforms(a.uniforms, b.uniforms)


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


def test_same_seed_same_landscape():
    a, b = schema.random_values(123), schema.random_values(123)
    assert {k: a[k] for k in terrain.SETTINGS} == {k: b[k] for k in terrain.SETTINGS}
    assert a["bg_ld_seed"] == float(int(a["bg_ld_seed"]))       # graine entière


@pytest.mark.parametrize("key", ["bg_ld_vault", "bg_col_source"])
def test_mode_weights_are_respected(key):
    p = schema.BY_KEY[key]
    n = 1000
    draws = [schema.random_values(s, groups=[p.group])[key] for s in range(n)]
    total = sum(p.weights)
    for i, w in enumerate(p.weights):
        share = draws.count(float(p.rand[0] + i)) / n
        assert share == pytest.approx(w / total, abs=0.05), (key, i)


def test_terraces_and_volume_are_often_off():
    n = 1000
    for key, group, share in (("bg_ld_terrace", "ld", 0.6), ("bg_vo_caves", "vo", 0.5),
                              ("bg_vo_bubbles", "vo", 0.6), ("bg_vo_twist", "vo", 0.7)):
        off = sum(schema.random_values(s, groups=[group])[key] == 0.0 for s in range(n))
        assert off / n == pytest.approx(share, abs=0.05), key


def test_group_lock_keeps_the_group():
    locked = schema.random_values(9, locked=["ld", "col"])
    assert not any(schema.BY_KEY[k].group in ("ld", "col") for k in locked)
    assert any(schema.BY_KEY[k].group == "mat" for k in locked)


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
    p = settings(bg_mov_music=1.0, bg_col_flash=1.0)
    calm = run(BackgroundState(), p, LOUD, [0.0] * 11, steps=30)
    kick = [0.0] * 8 + [1.0, 1.0, 0.0]          # KICK + SNARE, pas d'ÉNERGIE
    hit = run(BackgroundState(), p, LOUD, kick, steps=30)
    assert hit.uniforms["u_flash"] > calm.uniforms["u_flash"]    # flash sur la caisse claire
    assert hit.terrain == calm.terrain
    for q in schema.PARAMS:
        if q.kind in schema.SLOW_ONLY and q.shader:
            assert hit.uniforms[q.uniform] == calm.uniforms[q.uniform], q.key


def test_slew_limits_structural_jumps():
    st = BackgroundState()
    run(st, settings())
    dt = 0.1
    run(st, settings(bg_ld_scale=50.0), dt=dt)
    p = schema.BY_KEY["bg_ld_scale"]
    moved = st.values["bg_ld_scale"] - p.default
    assert 0.0 < moved <= p.max_slew * dt + 1e-9


def test_first_frame_is_not_slewed():
    st = BackgroundState()
    run(st, settings(bg_ld_scale=50.0))
    assert st.values["bg_ld_scale"] == pytest.approx(50.0)


@pytest.mark.slow
def test_motion_stays_bounded_over_a_long_session():
    st = BackgroundState()
    p = settings(bg_mov_speed=4.0, bg_mov_rot=1.0)
    run(st, p, dt=1.0, steps=3600)
    assert 0.0 <= st.angle < 2.0 * math.pi


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


def test_twist_and_waves_stay_put_while_the_object_travels():
    # le shader compte l'abscisse depuis l'objet (a.z) + la phase : un point
    # fixe du décor doit garder le même angle de torsion et d'ondes
    st = BackgroundState()
    p = settings(bg_mov_travel=3.0, bg_mov_drift=0.0, bg_vo_twist=2.0,
                 bg_vo_wave_amp=2.0, bg_vo_wave_len=17.0, bg_mov_dir_az=40.0)
    q = np.array([5.0, -2.0, 11.0])                      # point du décor

    def angles(frame):
        u = frame.uniforms
        d = np.asarray(u["u_path_m"]).reshape(3, 3).T[2]
        z = float(np.dot(q - np.asarray(u["u_path_o"]), d))
        tw = 2.0 * math.pi * 0.02 * z + u["u_vo_twist_ph"]
        wa = 2.0 * math.pi / 17.0 * z + u["u_vo_wave_ph"]
        return np.array([math.cos(tw), math.sin(tw), math.cos(wa), math.sin(wa)])

    first = angles(run(st, p))
    later = angles(run(st, p, dt=0.5, steps=40))        # 20 s, 60 unités plus loin
    assert np.allclose(first, later, atol=1e-6)


def test_grid_follows_the_checkbox_and_intensity():
    on = run(BackgroundState(), settings(show_grid=1.0)).uniforms["u_grid_col"]
    assert max(on) > 0.0
    for kw in ({"show_grid": 0.0}, {"show_grid": 1.0, "bg_grid_int": 0.0}):
        assert run(BackgroundState(), settings(**kw)).uniforms["u_grid_col"] == (0.0,) * 3


def test_flat_grid_only_when_the_decor_is_hidden():
    from unittest.mock import MagicMock

    from app.frame import draw
    from gfx.controls import DEFAULTS

    for visible, flat in ((0.0, True), (1.0, False)):
        r = MagicMock(has_mesh=False)
        draw(r, dict(DEFAULTS, show_grid=1.0, bg_visible=visible),
             {"Rim": 0.0, "Lumière": 0.0}, {"angle": 0.0, "scale": 1.0}, 0.0)
        assert r.draw_grid.called == flat


def test_default_direction_goes_into_the_screen():
    assert np.allclose(travel_dir(schema.defaults()), (0.0, 0.0, -1.0))
    for az in (0.0, 90.0, 217.0):                      # le sol reste au sol
        d = travel_dir(dict(schema.defaults(), bg_mov_dir_az=az))
        assert d[1] == 0.0 and np.linalg.norm(d) == pytest.approx(1.0)


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
    travel, clock = st.travel.copy(), st.clock
    run(st, settings(bg_mov_freeze=1.0), steps=10)
    assert np.array_equal(st.travel, travel) and st.clock == clock


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
def test_recipes_are_landscapes_in_range():
    assert len(schema.AMBIANCES) >= 8
    assert {"Planète inconnue", "Canyon voûté"} <= set(schema.AMBIANCES)
    for name, vals in schema.AMBIANCES.items():
        for k, v in vals.items():
            p = schema.BY_KEY[k]
            assert p.lo <= v <= p.hi, (name, k)
        assert vals["bg_visible"] == 1.0
    vals = list(schema.AMBIANCES.values())
    assert any(v.get("bg_vo_cave_mode", 0.0) == 1.0 for v in vals)  # une caverne
    assert any(v.get("bg_vo_bubbles", 0.0) > 0 for v in vals)       # des bulles
    assert any(v.get("bg_ld_vault", 0.0) > 0 for v in vals)        # un canyon voûté
    assert any(v.get("bg_ld_terrace", 0.0) > 0.5 for v in vals)    # des terrasses


def test_recipes_render_without_error():
    for vals in schema.AMBIANCES.values():
        f = run(BackgroundState(), settings(**vals), steps=3)
        assert all(np.isfinite(np.asarray(v, dtype=float)).all()
                   for v in f.uniforms.values())


# ── macros du paysage ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", [n for n, *_ in schema.MACROS_FORM])
def test_landscape_macros_move_the_landscape(name):
    base = macros.apply(schema.defaults())
    for m in (0.0, 1.0):
        out = macros.apply(dict(schema.defaults(), **{f"bg_m_{name}": m}))
        moved = [k for k in out if out[k] != base[k]
                 and schema.BY_KEY[k].group in ("ld", "vo")]
        assert moved, (name, m)


def test_kick_and_snare_never_reach_structure():
    for p in schema.PARAMS:
        if p.audio and p.audio[0] in schema.FAST_SOURCES:
            assert p.kind not in ("topology", "int", "deform"), p.key


def test_landscape_travels_horizontally_and_detail_follows_the_size():
    assert travel_dir(dict(schema.defaults(), bg_mov_dir_az=33.0))[1] == 0.0
    small = run(BackgroundState(), settings(bg_ld_scale=8.0))
    big = run(BackgroundState(), settings(bg_ld_scale=40.0))
    assert big.uniforms["u_detail"] > small.uniforms["u_detail"]


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
    assert 0.0 < inner < 0.5                            # lisse


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


def _sway(state, p, secs, fps=15):
    """Écart d'un point fixe devant la caméra (10 unités) à sa trajectoire
    rectiligne dans l'espace des formes, sur les 4 dernières secondes."""
    pts = []
    for i in range(int(secs * fps)):
        ph = (i / fps * 2.0) % 1.0
        f = state.step(p, {"rms": 0.6, "bpm": 120.0, "beat_phase": ph}, [0.0] * 11, 1 / fps)
        u = f.uniforms
        rot = np.asarray(u["u_rot"]).reshape(3, 3).T
        pts.append((rot @ np.array([0.0, 0.0, -10.0]) + np.asarray(u["u_offset"])) / u["u_scale"])
    pts = np.array(pts[-4 * fps:])
    t = np.arange(len(pts))[:, None]
    a = np.hstack([t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(a, pts, rcond=None)
    return float(np.abs(pts - a @ coef).max())


@pytest.mark.slow
def test_breath_is_centred_on_the_object():
    """La respiration fait grandir le décor autour de l'objet : l'objet
    reste sur son trajet, et l'aller-retour ne grandit pas avec le chemin
    parcouru (avant : ±6 unités au tempo après 5 min)."""
    p = settings(bg_mov_breath=0.03, bg_mov_breath_beats=1.0, bg_mov_drift=0.0,
                 bg_mov_travel=2.0)
    state = BackgroundState()
    gaps = []
    for _ in range(60):          # l'objet (p = 0) reste à la même place de son trajet
        u = run(state, p, features=LOUD).uniforms
        gaps.append(np.asarray(u["u_offset"]) / u["u_scale"] - np.asarray(u["u_path_o"]))
    assert np.ptp(np.array(gaps), axis=0).max() < 1e-9
    early = _sway(BackgroundState(), p, 20)
    late = _sway(BackgroundState(), p, 200)
    assert late < 0.5 and late < early * 1.5 + 0.05
