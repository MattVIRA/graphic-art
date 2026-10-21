"""Gestes souris → caméra orbitale (T1.13, sorti de main.py par T1.15).

Cmd/Ctrl + glisser ou bouton du milieu = orbite, Maj + glisser =
déplacement, double-clic = vue de départ. Les événements arrivent déjà
décodés (bouton, action, modificateurs, position) : la logique ne dépend
que des constantes GLFW, pas d'une fenêtre ouverte — testée dans
tests/test_input.py.
"""

import glfw

DOUBLE_CLICK_S = 0.35    # two clicks closer than this = double-click
DOUBLE_CLICK_PX = 6      # …and closer than this on screen (logical px)


class MouseGestures:
    """Suit le glisser en cours et le dernier clic ; pilote `camera`."""

    def __init__(self, camera):
        self.camera = camera
        self.drag = None              # (mode "orbit" | "pan", mouse button) while dragging
        self.cursor = None            # last cursor position (logical px)
        self.last_click = (-1.0, 0.0, 0.0)   # time, x, y of the last plain left click

    def button(self, button: int, action: int, mods: int, pos, t: float):
        """Appui / relâchement d'un bouton à la position `pos`, à l'instant `t` (s)."""
        self.cursor = pos
        if action == glfw.PRESS and self.drag is None:
            shift = mods & glfw.MOD_SHIFT
            cmd = mods & (glfw.MOD_SUPER | glfw.MOD_CONTROL)
            if button == glfw.MOUSE_BUTTON_MIDDLE or (
                    button == glfw.MOUSE_BUTTON_LEFT and (shift or cmd)):
                self.drag = ("pan" if shift else "orbit", button)
                if self.drag[0] == "orbit":
                    self.camera.begin_orbit()
            elif button == glfw.MOUSE_BUTTON_LEFT:
                last_t, last_x, last_y = self.last_click
                if (t - last_t < DOUBLE_CLICK_S
                        and abs(pos[0] - last_x) < DOUBLE_CLICK_PX
                        and abs(pos[1] - last_y) < DOUBLE_CLICK_PX):
                    self.camera.reset()
                    t = -1.0   # a third click starts over
                self.last_click = (t, *pos)
        elif (action == glfw.RELEASE and self.drag is not None
                and button == self.drag[1]):
            if self.drag[0] == "orbit":
                self.camera.end_orbit()
            self.drag = None

    def move(self, pos):
        """Déplacement du curseur : fait avancer le glisser en cours."""
        if self.drag is not None and self.cursor is not None:
            dx, dy = pos[0] - self.cursor[0], pos[1] - self.cursor[1]
            if self.drag[0] == "orbit":
                self.camera.orbit(dx, dy)
            else:
                self.camera.pan(dx, dy)
        self.cursor = pos
