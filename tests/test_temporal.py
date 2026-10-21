"""Accumulation temporelle du fond (T1.16, passe 5) : décalages de Halton,
reprojection d'un point du décor, règles de rejet, mélange, poids maximal,
remises à zéro — la logique que resolve.glsl applique, sans GL."""

import os
import re

import numpy as np
import pytest

from background import schema, temporal
from background.params import BackgroundState, rotation
from background.pass_ import RESOLVE_FILE, SHADER_DIR
from background.temporal import NO_HIT, History, Pose
from gfx.camera import look_at
from gfx.renderer import FAR, NEAR, _perspective

UNIT = 120.0                     # px du monde par unité du fond (rayon du modèle)
AXIS = np.array([0.2, 1.0, 0.1]) / np.linalg.norm([0.2, 1.0, 0.1])
PROJ = _perspective(np.radians(45.0), 16 / 9, NEAR, FAR)


def pose(yaw=0.0, pitch=0.0, dist=870.0, travel=(0.0, 0.0, 0.0), angle=0.0,
         scale=1.0, far=0.0) -> Pose:
    eye = dist * np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                           np.cos(yaw) * np.cos(pitch)])
    view = look_at(eye, (0.0, 0.0, 0.0))
    return Pose(vp=PROJ @ view, eye=eye, fwd=-view[2, :3], unit=UNIT, far=far,
                rot=rotation(AXIS, angle), offset=np.asarray(travel, float), scale=scale)


# ── décalages ─────────────────────────────────────────────────────────
def test_halton_first_terms():
    assert [temporal.halton(i, 2) for i in range(1, 5)] == [0.5, 0.25, 0.75, 0.125]
    assert temporal.halton(1, 3) == pytest.approx(1 / 3)
    assert temporal.halton(2, 3) == pytest.approx(2 / 3)


def test_jitter_stays_in_the_pixel_and_is_centred():
    js = np.array([temporal.jitter(f) for f in range(temporal.CYCLE)])
    assert np.all(js >= -0.5) and np.all(js < 0.5)
    assert np.abs(js.mean(axis=0)).max() < 0.05
    assert len({tuple(j) for j in js}) == temporal.CYCLE            # tous distincts
    assert temporal.jitter(3) == temporal.jitter(3 + temporal.CYCLE)


def test_jitter_visits_every_sub_pixel_at_a_third():
    # en ⅓, un pixel du fond couvre 3×3 pixels de l'écran : un cycle les
    # visite tous (sinon certains ne recevraient jamais de rayon au centre)
    cells = {tuple(np.floor((np.array(temporal.jitter(f)) + 0.5) * 3).astype(int))
             for f in range(temporal.CYCLE)}
    assert len(cells) == 9


# ── reprojection ──────────────────────────────────────────────────────
def check_reprojection(cur: Pose, prev: Pose, ps):
    """Un point fixe du décor (espace des formes `ps`) vu cette frame est
    ramené là où la frame précédente le voyait."""
    uv_cur, d_cur = cur.project(cur.from_shapes(ps))
    uv_prev, d_prev = prev.project(prev.from_shapes(ps))
    uv, d = temporal.reproject(cur, prev, uv_cur, d_cur)
    assert np.allclose(uv, uv_prev, atol=1e-6)
    assert d == pytest.approx(d_prev, rel=1e-6)


@pytest.mark.parametrize("ps", [(3.0, 1.0, -8.0), (-6.0, -2.0, -20.0), (1.0, 4.0, 3.0)])
def test_reprojection_follows_the_camera(ps):
    check_reprojection(pose(yaw=0.31, pitch=0.12), pose(yaw=0.30, pitch=0.11), ps)


def test_reprojection_follows_the_travel_rotation_and_breath():
    # l'objet voyage (offset), le décor tourne et respire : le point reste
    # fixe dans l'espace des formes
    cur = pose(yaw=0.2, travel=(0.0, 0.0, 0.61), angle=0.051, scale=1.02)
    prev = pose(yaw=0.2, travel=(0.0, 0.0, 0.60), angle=0.050, scale=1.01)
    check_reprojection(cur, prev, (2.0, -1.0, -12.0))


def test_reprojection_with_far_effect():
    # « Éloignement » : les rayons partent d'un œil rapproché du centre
    check_reprojection(pose(yaw=0.4, far=0.6), pose(yaw=0.38, far=0.6), (4.0, 0.5, -30.0))


def test_still_camera_reprojects_in_place():
    p = pose(yaw=0.1)
    for uv, d in (((0.3, 0.6), 5000.0), ((0.7, 0.2), NO_HIT)):
        out, _ = temporal.reproject(p, p, uv, d)
        assert np.allclose(out, uv, atol=1e-6)


def test_sky_reprojects_with_the_camera_only():
    # le ciel est à l'infini : le voyage et la rotation du décor ne le
    # déplacent pas, la rotation de la caméra si
    cur = pose(yaw=0.2, travel=(0.0, 0.0, 5.0), angle=0.4)
    prev = pose(yaw=0.2)
    uv, d = temporal.reproject(cur, prev, (0.4, 0.7), NO_HIT)
    assert np.allclose(uv, (0.4, 0.7), atol=1e-6) and d == NO_HIT
    uv, _ = temporal.reproject(pose(yaw=0.25), prev, (0.4, 0.7), NO_HIT)
    assert uv[0] < 0.4                   # la caméra tourne : le ciel glisse


def test_point_at_matches_the_recorded_depth():
    p = pose(yaw=0.3, pitch=0.2)
    point = np.array([2.0, -1.0, -6.0])
    uv, d = p.project(point)
    assert np.allclose(temporal.point_at(p, uv, d), point, atol=1e-6)


# ── rejet et mélange ──────────────────────────────────────────────────
def test_depth_confidence_rules():
    dc = temporal.depth_confidence
    assert dc(1000.0, 1010.0) == pytest.approx(1.0)       # 1 % : gardé
    assert dc(1000.0, 1200.0) == pytest.approx(0.0)       # 17 % : rejeté
    assert 0.0 < dc(1000.0, 1050.0) < 1.0                 # entre les deux
    assert dc(NO_HIT, NO_HIT) == 1.0                      # ciel contre ciel
    assert dc(1000.0, NO_HIT) == 0.0 and dc(NO_HIT, 1000.0) == 0.0
    assert dc(1000.0, temporal.MASKED) == 0.0             # masqué par le modèle
    # monotone avec l'écart
    rel = np.linspace(0.0, 0.2, 50)
    assert np.all(np.diff(dc(1000.0, 1000.0 * (1 + rel))) <= 1e-12)


def test_history_survives_an_edge_but_not_a_disocclusion():
    # contour forme / ciel : l'échantillon le plus proche change de côté,
    # mais la surface de l'historique est toujours dans le voisinage
    edge = [NO_HIT, NO_HIT, 1000.0, NO_HIT, 1002.0, 1001.0, 999.0, 1003.0, NO_HIT]
    assert temporal.history_confidence(1000.0, edge) == pytest.approx(1.0)
    assert temporal.history_confidence(NO_HIT, edge) == 1.0
    # la forme est partie (découverte) : rejeté
    sky = [NO_HIT] * 9
    assert temporal.history_confidence(1000.0, sky) == 0.0
    # une autre surface, plus loin : rejeté
    assert temporal.history_confidence(1000.0, [1500.0] * 9) == 0.0
    # masqués exclus, historique masqué rejeté
    assert temporal.history_confidence(1000.0, [temporal.MASKED] * 9) == 0.0
    assert temporal.history_confidence(temporal.MASKED, edge) == 0.0
    # le point s'est rapproché de 30 px : la profondeur gardée suit
    assert temporal.history_confidence(1030.0, [1000.0] * 9, shift=-30.0) == pytest.approx(1.0)


def test_motion_depth_follows_the_nearest_surface():
    assert temporal.motion_depth([NO_HIT, 1200.0, 900.0, temporal.MASKED]) == 900.0
    assert temporal.motion_depth([NO_HIT, temporal.MASKED]) == NO_HIT


def test_clip_confidence():
    assert temporal.clip_confidence(0.0) == 1.0
    assert temporal.clip_confidence(1.0) < temporal.clip_confidence(0.2) < 1.0


def test_accumulate_converges_and_is_bounded():
    col, w = np.zeros(3), 0.0
    for _ in range(200):
        col, w = temporal.accumulate(col, w, np.ones(3), 1.0, 16.0)
    assert np.allclose(col, 1.0) and w == 16.0
    # sans historique : l'échantillon courant tel quel
    out, w = temporal.accumulate(np.zeros(3), 0.0, np.full(3, 0.5), 0.7, 16.0)
    assert np.allclose(out, 0.5) and w == pytest.approx(0.7)
    # lissage 0 : jamais d'historique
    out, w = temporal.accumulate(np.zeros(3), 10.0, np.ones(3), 1.0, 0.0)
    assert np.allclose(out, 1.0) and w == 0.0


def test_accumulate_averages_the_jittered_samples():
    # 16 décalages d'une arête : l'historique tend vers la couverture moyenne
    # (anti-crénelage), pas vers le dernier échantillon
    col, w = np.zeros(1), 0.0
    for f in range(64):
        x, _ = temporal.jitter(f)
        col, w = temporal.accumulate(col, w, np.array([1.0 if x < 0.1 else 0.0]), 1.0, 32.0)
    assert 0.45 < col[0] < 0.75


def test_history_size():
    # ⅓ de 2560×1440 → ½ ; une résolution du fond plus fine plafonne à la taille de rendu
    assert temporal.history_size((853, 480), (2560, 1440)) == (1280, 720)
    assert temporal.history_size((1920, 1080), (2560, 1440)) == (2560, 1440)
    assert temporal.history_size((2, 2), (2560, 1440)) == (8, 8)


# ── poids maximal ─────────────────────────────────────────────────────
def test_max_weight():
    assert temporal.max_weight(0.0) == 0.0
    assert temporal.max_weight(1.0) == temporal.MAX_HISTORY
    assert temporal.max_weight(0.7) == pytest.approx(16.19)
    ws = [temporal.max_weight(s) for s in np.linspace(0.05, 1.0, 20)]
    assert all(b > a for a, b in zip(ws, ws[1:], strict=False))


def test_frame_carries_the_history_weight():
    p = schema.defaults()
    p.update(bg_visible=1.0, bg_temporal=0.7)
    f = BackgroundState().step(p, {"rms": 0.0, "bpm": 0.0, "beat_phase": 0.0},
                               [0.0] * 11, 1 / 60)
    assert f.history == pytest.approx(temporal.max_weight(0.7))
    p["bg_temporal"] = 0.0
    f = BackgroundState().step(p, {"rms": 0.0, "bpm": 0.0, "beat_phase": 0.0},
                               [0.0] * 11, 1 / 60)
    assert f.history == 0.0


def test_structure_key_changes_with_a_new_draw():
    s = BackgroundState()
    quiet = {"rms": 0.0, "bpm": 0.0, "beat_phase": 0.0}
    p = schema.defaults()
    p["bg_visible"] = 1.0
    a = s.step(p, quiet, [0.0] * 11, 1 / 60).structure
    assert s.step(p, quiet, [0.0] * 11, 1 / 60).structure == a
    p.update(schema.random_values(7))
    assert s.step(p, quiet, [0.0] * 11, 1 / 60).structure != a


# ── remises à zéro ────────────────────────────────────────────────────
def test_history_resets():
    h = History()
    a, b = pose(), pose(yaw=0.01)
    prev, f1 = h.step("k", a)
    assert prev is None                                  # première frame
    prev, f2 = h.step("k", b)
    assert prev is a and f2 == f1 + 1
    prev, _ = h.step("autre", a)                         # taille, qualité, dé…
    assert prev is None
    h.forget()                                           # une frame sans fond
    prev, _ = h.step("autre", a)
    assert prev is None


# ── accord avec les shaders ───────────────────────────────────────────
def glsl(name: str) -> str:
    with open(os.path.join(SHADER_DIR, name), encoding="utf-8") as f:
        return f.read()


def test_resolve_shader_uses_the_same_constants():
    src = glsl(RESOLVE_FILE)
    assert f"const float NO_HIT = {NO_HIT};" in src
    assert f"SIGMA_N = {temporal.SIGMA_NARROW}, SIGMA_W = {temporal.SIGMA_WIDE};" in src
    lo, hi = temporal.DEPTH_TOL
    assert f"DEPTH_LO = {lo}, DEPTH_HI = {hi};" in src
    assert f"const float CLIP_K = {temporal.CLIP_K};" in src
    main = glsl("main.glsl")
    assert f"const float NO_HIT = {NO_HIT};" in main
    assert f"const float MASKED = {temporal.MASKED};" in main


def test_resolve_uniforms_are_declared():
    src = glsl(RESOLVE_FILE)
    declared = set(re.findall(r"\bu_\w+", " ".join(
        line for line in src.splitlines() if line.startswith("uniform"))))
    names = temporal.resolve_uniforms(pose(), pose(yaw=0.1)).keys()
    assert set(names) <= declared, set(names) - declared
