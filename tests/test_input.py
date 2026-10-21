"""Gestes souris → caméra (T1.13, app/input.py) : orbite, déplacement,
double-clic, sans fenêtre GLFW."""

import glfw

from app.input import DOUBLE_CLICK_S, MouseGestures
from gfx.camera import OrbitCamera

LEFT, MIDDLE, RIGHT = (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_MIDDLE,
                       glfw.MOUSE_BUTTON_RIGHT)


def drag(g, button, mods, start=(100.0, 100.0), end=(160.0, 130.0), t=1.0):
    g.button(button, glfw.PRESS, mods, start, t)
    g.move(end)
    g.button(button, glfw.RELEASE, mods, end, t + 0.5)


def test_cmd_drag_orbits():
    cam = OrbitCamera()
    drag(MouseGestures(cam), LEFT, glfw.MOD_SUPER)
    assert not cam.is_default
    assert cam.state()["yaw"] != 0.0
    assert cam.state()["target"] == [0.0, 0.0, 0.0]


def test_middle_drag_orbits_and_shift_drag_pans():
    cam = OrbitCamera()
    drag(MouseGestures(cam), MIDDLE, 0)
    assert cam.state()["yaw"] != 0.0
    cam = OrbitCamera()
    drag(MouseGestures(cam), LEFT, glfw.MOD_SHIFT)
    assert cam.state()["yaw"] == 0.0
    assert any(cam.state()["target"])


def test_plain_left_drag_does_nothing():
    cam = OrbitCamera()
    drag(MouseGestures(cam), LEFT, 0)
    assert cam.is_default


def test_release_of_another_button_keeps_the_drag():
    g = MouseGestures(OrbitCamera())
    g.button(LEFT, glfw.PRESS, glfw.MOD_SUPER, (0.0, 0.0), 1.0)
    g.button(RIGHT, glfw.RELEASE, 0, (0.0, 0.0), 1.1)
    assert g.drag == ("orbit", LEFT)
    g.button(LEFT, glfw.RELEASE, 0, (0.0, 0.0), 1.2)
    assert g.drag is None


def click(g, pos, t):
    g.button(LEFT, glfw.PRESS, 0, pos, t)
    g.button(LEFT, glfw.RELEASE, 0, pos, t + 0.05)


def test_double_click_resets_the_view():
    cam = OrbitCamera()
    g = MouseGestures(cam)
    drag(g, LEFT, glfw.MOD_SUPER)
    click(g, (50.0, 50.0), 10.0)
    click(g, (52.0, 51.0), 10.0 + DOUBLE_CLICK_S / 2)
    assert cam.is_default


def test_slow_or_distant_clicks_are_not_a_double_click():
    cam = OrbitCamera()
    g = MouseGestures(cam)
    drag(g, LEFT, glfw.MOD_SUPER)
    click(g, (50.0, 50.0), 10.0)
    click(g, (50.0, 50.0), 10.0 + DOUBLE_CLICK_S * 2)   # trop lent
    click(g, (90.0, 50.0), 10.0 + DOUBLE_CLICK_S * 2.2)  # trop loin
    assert not cam.is_default


def test_third_click_starts_a_new_double_click():
    cam = OrbitCamera()
    g = MouseGestures(cam)
    click(g, (50.0, 50.0), 10.0)
    click(g, (50.0, 50.0), 10.1)          # double-clic → retour
    drag(g, LEFT, glfw.MOD_SUPER, t=10.15)
    click(g, (50.0, 50.0), 10.2)          # 3e clic : ne compte pas comme double
    assert not cam.is_default
