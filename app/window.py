"""Fenêtre GLFW + contexte OpenGL 3.3 core, entrées et cadence (T21, T22).

Les callbacks GLFW ne font qu'empiler des événements dans `events` ; la
boucle principale les traite une fois par frame.
"""

import time

import glfw


class FrameClock:
    """dt + fps measurement and a frame-rate cap. L'animation suit dt
    depuis T22 ; le cap évite de dessiner plus d'images qu'utile."""

    def __init__(self, fps: int):
        self.period = 1.0 / fps
        self._last = time.perf_counter()
        self.dt = self.period
        self.fps = float(fps)

    def tick(self):
        now = time.perf_counter()
        wait = self.period - (now - self._last)
        if wait > 0.002:
            time.sleep(wait - 0.001)
        while time.perf_counter() - self._last < self.period:
            pass
        now = time.perf_counter()
        self.dt = now - self._last
        self._last = now
        self.fps += (1.0 / max(self.dt, 1e-4) - self.fps) * 0.1


class Window:
    """Fenêtre GLFW avec framebuffer Retina natif. Événements empilés dans
    `events` : ("key", touche), ("wheel", dy), ("button", (bouton, action,
    modificateurs, position)), ("cursor", position), ("drop", chemin)."""

    def __init__(self, width: int, height: int, title: str):
        glfw.init_hint(glfw.COCOA_CHDIR_RESOURCES, glfw.FALSE)
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
        self.handle = glfw.create_window(width, height, title, None, None)
        if not self.handle:
            glfw.terminate()
            raise RuntimeError("could not create the OpenGL window")
        glfw.make_context_current(self.handle)
        glfw.swap_interval(1)

        self.events = []
        w = self.handle
        glfw.set_key_callback(w, lambda _w, key, _sc, action, mods:
                              self.events.append(("key", key))
                              if action == glfw.PRESS else None)
        glfw.set_scroll_callback(w, lambda _w, dx, dy: self.events.append(("wheel", dy)))
        glfw.set_mouse_button_callback(w, lambda _w, button, action, mods:
                                       self.events.append(
                                           ("button", (button, action, mods,
                                                       glfw.get_cursor_pos(_w)))))
        glfw.set_cursor_pos_callback(w, lambda _w, x, y:
                                     self.events.append(("cursor", (x, y))))
        glfw.set_drop_callback(w, lambda _w, paths:
                               self.events.extend(("drop", p) for p in paths))
        self._windowed_geom = None     # (x, y, w, h) to restore after fullscreen

    def poll(self) -> list:
        """Traite les événements système ; renvoie (et vide) la file."""
        glfw.poll_events()
        events, self.events = self.events, []
        return events

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    @property
    def size(self) -> tuple:
        """Taille logique (points, pas pixels Retina)."""
        return tuple(glfw.get_window_size(self.handle))

    def pixel_ratio(self) -> float:
        ww, _ = glfw.get_window_size(self.handle)
        fw, _ = glfw.get_framebuffer_size(self.handle)
        return fw / max(ww, 1)

    def set_title(self, title: str):
        glfw.set_window_title(self.handle, title)

    def toggle_fullscreen(self):
        if self._windowed_geom is None:
            x, y = glfw.get_window_pos(self.handle)
            self._windowed_geom = (x, y, *glfw.get_window_size(self.handle))
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            glfw.set_window_monitor(self.handle, mon, 0, 0, mode.size.width,
                                    mode.size.height, mode.refresh_rate)
        else:
            glfw.set_window_monitor(self.handle, None, *self._windowed_geom, 0)
            self._windowed_geom = None

    def swap(self):
        glfw.swap_buffers(self.handle)

    @staticmethod
    def time() -> float:
        """Secondes depuis l'ouverture de GLFW."""
        return glfw.get_time()

    @staticmethod
    def close():
        glfw.terminate()
