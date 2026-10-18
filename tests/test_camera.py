"""Caméra orbitale (T26) : vue par défaut, bornes, inertie, retour, autosave."""

import math

import numpy as np

from gfx.camera import PITCH_LIMIT, OrbitCamera, look_at

D = 870.0   # distance de la vue par défaut (≈ renderer.cam_dist en 720p)


def settle(cam, seconds=3.0, dt=1 / 60):
    for _ in range(int(seconds / dt)):
        cam.update(dt)


def eye_of(view):
    """Position de la caméra dans le monde (inverse de la vue)."""
    return np.linalg.inv(view)[:3, 3]


def test_default_view_is_the_old_fixed_view():
    V = OrbitCamera().view_matrix(D)
    expected = np.eye(4)
    expected[2, 3] = -D
    assert np.allclose(V, expected)


def test_look_at_is_a_rigid_transform():
    V = look_at((300.0, 200.0, 500.0), (10.0, -20.0, 5.0))
    R = V[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    # la cible est droit devant (axe -z de la caméra)
    t = V @ np.array([10.0, -20.0, 5.0, 1.0])
    assert abs(t[0]) < 1e-9 and abs(t[1]) < 1e-9 and t[2] < 0


def test_drag_right_moves_the_camera_left():
    cam = OrbitCamera()
    cam.begin_orbit()
    cam.orbit(100, 0)
    cam.end_orbit()
    settle(cam)
    assert eye_of(cam.view_matrix(D))[0] < -100   # passée à gauche du modèle


def test_drag_down_looks_from_above_and_never_flips():
    cam = OrbitCamera()
    cam.begin_orbit()
    for _ in range(200):          # glisser très loin vers le bas
        cam.orbit(0, 40)
        cam.update(1 / 60)
    cam.end_orbit()
    settle(cam)
    assert cam.pitch <= PITCH_LIMIT + 1e-9
    V = cam.view_matrix(D)
    assert eye_of(V)[1] > 0.9 * D            # au-dessus
    assert V[1, 1] > 0                        # le haut du monde reste en haut


def test_distance_stays_constant_while_orbiting():
    cam = OrbitCamera()
    cam.orbit(250, -120)
    settle(cam)
    assert np.isclose(np.linalg.norm(eye_of(cam.view_matrix(D))), D)


def test_glide_after_release_then_stop():
    cam = OrbitCamera()
    cam.begin_orbit()
    for _ in range(10):           # geste rapide vers la droite
        cam.orbit(20, 0)
        cam.update(1 / 60)
    goal_at_release = cam.state()["yaw"]
    cam.end_orbit()
    settle(cam, 0.3)
    mid = cam.state()["yaw"]
    settle(cam, 4.0)
    end = cam.state()["yaw"]
    assert mid < goal_at_release              # la caméra glisse encore…
    assert abs(cam.yaw - end) < 1e-3          # …puis s'arrête
    settle(cam, 1.0)
    assert cam.state()["yaw"] == end


def test_no_glide_when_released_without_motion():
    cam = OrbitCamera()
    cam.begin_orbit()
    cam.orbit(50, 0)
    cam.update(1 / 60)
    for _ in range(20):           # on tient la souris immobile avant de lâcher
        cam.update(1 / 60)
    goal = cam.state()["yaw"]
    cam.end_orbit()
    settle(cam)
    assert abs(cam.state()["yaw"] - goal) < 0.01


def test_reset_takes_the_short_way_and_returns_home():
    cam = OrbitCamera()
    cam.orbit(-5 * 2 * math.pi / 0.006 - 30, 60)   # 5 tours + un peu
    cam.pan(40, 10)
    cam.zoom_by(8)
    settle(cam)
    cam.reset()
    assert abs(cam.yaw) <= math.pi            # pas 5 tours à dérouler
    settle(cam)
    assert cam.is_default
    expected = np.eye(4)
    expected[2, 3] = -D
    assert np.allclose(cam.view_matrix(D), expected, atol=1e-3)


def test_pan_follows_the_cursor_at_any_zoom():
    for zoom in (1.0, 2.0):
        cam = OrbitCamera()
        cam.view_matrix(D, extra_zoom=zoom)   # mémorise l'échelle courante
        cam.pan(100, 0)                        # glisser vers la droite
        settle(cam)
        # le décor suit le curseur : la cible part à gauche, de 100 px
        # écran = 100 / zoom unités monde
        assert np.allclose(cam.target, [-100 / zoom, 0, 0], atol=1e-3)


def test_zoom_is_clamped():
    cam = OrbitCamera()
    cam.zoom_by(500)
    settle(cam)
    assert np.isclose(cam.zoom, 4.0)


def test_state_round_trip_and_invalid_state():
    cam = OrbitCamera()
    cam.orbit(80, 30)
    cam.pan(10, -5)
    cam.zoom_by(3)
    s = cam.state()
    other = OrbitCamera()
    other.load_state(s)
    assert other.state() == s
    settle(cam)   # restaurée = position finale, sans animation
    assert np.allclose(other.view_matrix(D), cam.view_matrix(D))
    for bad in (None, {}, {"yaw": "x"}, {"yaw": 0, "pitch": 0, "target": [0],
                                         "zoom": 1}, {"yaw": float("nan"),
                "pitch": 0, "target": [0, 0, 0], "zoom": 1}):
        fresh = OrbitCamera()
        fresh.load_state(bad)
        assert fresh.is_default
