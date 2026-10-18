"""
Graphic Art — GPU Audio Visualizer
Usage: python main.py [audio_file] [--visual FILE]

Visual: a 3D model (.glb .gltf .obj .stl .ply), also via drag & drop / panel,
drawn as points, surface or wireframe (style chosen in the panel).
  none                      → the built-in torus (DEFAULT_VISUAL)
  --visual model.glb        → that model
  --visual primitive:box    → another built-in shape (icosphere, torus, box)

Keys: SPACE pause/resume  |  ESC quit  |  U back to the torus  |  F fullscreen
Mouse (T26): Cmd/Ctrl + drag or middle drag = orbit  |  Shift + drag = pan  |
             wheel = zoom  |  C or double-click = back to the default view

The window is a GLFW window (native Retina framebuffer); pygame only provides
the audio playback (mixer) and the HUD font.

Assemblage seulement (T22) : chaque responsabilité vit dans un module de
`app/` — sources audio, chargement des modèles, routing, gestes souris,
dessin d'une frame, fenêtre, HUD.
"""

import argparse
import os
import sys

import glfw
import moderngl
import pygame

from app import frame
from app.hud import Hud
from app.input import MouseGestures
from app.loader import DEFAULT_VISUAL, ModelLoader
from app.routing import Reactor
from app.sources import SourceManager
from app.window import FrameClock, Window
from gfx.camera import OrbitCamera
from gfx.controls import ControlPanel
from gfx.renderer import GLRenderer
from visuals.model_source import ModelSource

# ── Config ──────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (4, 4, 10)

FADE = 0.30            # trail decay per frame (higher = shorter trails)
BLOOM_INTENSITY = 0.9

MAX_POINTS = 1_000_000   # upper bound of the « Nombre de points » slider
BUDGET_DEBOUNCE_S = 0.5  # rebuild the points this long after the slider settles
HUD_PERIOD_S = 0.2       # HUD text refreshed at most 5×/s


class App:
    """La fenêtre, le moteur de rendu, le panneau et la source audio, reliés
    par la boucle `run()` (une itération = `frame()`)."""

    def __init__(self, visual: str):
        # ── audio (pygame mixer) + font — no pygame window ──
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.font.init()

        self.window = Window(WIDTH, HEIGHT, "Graphic Art")
        self.clock = FrameClock(FPS)
        ctx = moderngl.create_context()
        self.renderer = GLRenderer(ctx, (WIDTH, HEIGHT), bg_color=BG_COLOR,
                                   fade=FADE, bloom_intensity=BLOOM_INTENSITY)
        self.renderer.set_pixel_ratio(self.window.pixel_ratio())
        self.model = ModelSource(WIDTH, HEIGHT, max_points=MAX_POINTS)
        self.gpu_rev = -1      # last model revision uploaded to the GPU

        self.sources = SourceManager(on_title=self.window.set_title)
        self.hud = Hud(WIDTH, self.renderer.dpr)
        self.panel = ControlPanel()   # second window with live sliders
        self.loader = ModelLoader(self.model)
        self.load_visual(visual)

        self.camera = OrbitCamera()   # T26: orbit / pan / zoom with the mouse
        if self.panel.camera_state:
            self.camera.load_state(self.panel.camera_state)
        self.gestures = MouseGestures(self.camera)
        self.reactor = Reactor()      # enveloppes + couche percussive + routing

        self.running = True
        self.axis_precession = 0.0    # accumulated axis spin around the world vertical
        self.next_monitor = 0.0       # T8: next time a monitoring packet is due
        self.budget_wanted, self.budget_t = None, 0.0   # debounced n_points rebuild
        self.hud_next = 0.0           # next HUD text refresh

    # ── actions ───────────────────────────────────────────────────────
    def load_visual(self, path: str):
        self.loader.load(path, int(self.panel.params["model_up"]))

    def handle_event(self, kind: str, val):
        """Un événement de la fenêtre (touche, souris, fichier déposé)."""
        if kind == "key":
            if val == glfw.KEY_ESCAPE:
                self.running = False
            elif val == glfw.KEY_SPACE:
                self.sources.toggle_pause()
            elif val == glfw.KEY_U:
                self.load_visual(DEFAULT_VISUAL)
            elif val == glfw.KEY_F:
                self.window.toggle_fullscreen()
            elif val == glfw.KEY_C:
                self.camera.reset()
        elif kind == "wheel":
            self.camera.zoom_by(val)
        elif kind == "button":
            button, action, mods, pos = val
            self.gestures.button(button, action, mods, pos, self.window.time())
        elif kind == "cursor":
            self.gestures.move(val)
        elif kind == "drop":
            self.load_visual(val)

    def handle_action(self, name: str, payload):
        """Une action du panneau (onglets Source et Objet)."""
        if name == "load_visual" and payload:
            self.load_visual(payload)
        elif name == "default_visual":
            self.load_visual(DEFAULT_VISUAL)
        elif name == "load_audio" and payload:
            self.sources.start_file(payload)
        elif name == "start_live":
            try:
                self.sources.start_live(int(payload))
            except Exception as e:  # noqa: BLE001 — frontière : action du panneau
                print(f"[error] live capture failed: {e}")
        elif name == "stop_source":
            self.sources.stop()

    # ── boucle ────────────────────────────────────────────────────────
    def run(self):
        while self.running:
            self.frame()
        self.shutdown()

    def frame(self):
        """Une frame : entrées, état, routing, dessin, cadence."""
        if self.window.should_close:
            self.running = False
        for kind, val in self.window.poll():
            self.handle_event(kind, val)
        self.loader.update()
        self.sources.update()
        self._follow_window_size()

        features = self.sources.features()
        p = self.panel.poll()
        for name, payload in self.panel.pop_actions():
            self.handle_action(name, payload)
        self._update_budget(p)

        routed = self.reactor.step(features, p, self.clock.dt)
        features, fx = routed.features, routed.fx
        self._send_monitor(routed)

        t = self.window.time()
        renderer = self.renderer
        frame.apply_look(renderer, p, fx, routed.beat_env, t)
        # ── GPU draw pass (T1: stateless physics, the CPU only sends uniforms) ──
        rp = self.model.gpu_uniforms(features, frame.model_params(p, fx), time_s=t)
        if self.model.rev != self.gpu_rev:
            renderer.upload_particle_static(*self.model.build_static())
            renderer.upload_mesh(self.model.model_mesh)
            self.gpu_rev = self.model.rev
        rp["scale"] *= frame.zoom_pulse(fx)
        # axis precession: the rotation axis itself spins around the scene's
        # vertical axis at the origin when enabled
        if p["axis_spin"] >= 0.5:
            self.axis_precession += p["axis_spin_speed"] * frame.DEG * self.clock.dt
        self.camera.update(self.clock.dt)   # before set_transform (view matrix)
        renderer.set_view(self.camera.view_matrix(renderer.cam_dist, p["zoom"],
                                                  frame.shake(fx, t)))
        frame.draw(renderer, p, fx, rp, self.axis_precession)
        self._draw_hud(features)

        self.window.swap()
        self.clock.tick()

    def _follow_window_size(self):
        # the window size (and the pixel ratio, when it moves to another
        # screen) can change at any time: watch both every frame
        win_size = self.window.size
        dpr = self.window.pixel_ratio()
        r = self.renderer
        if (win_size[0] > 0 and win_size[1] > 0
                and (win_size != (r.w, r.h) or abs(dpr - r.dpr) > 1e-3)):
            r.set_pixel_ratio(dpr)
            r.resize(*win_size)
            self.hud = Hud(win_size[0], dpr)

    def _update_budget(self, p: dict):
        """Particle budget: rebuild ~0.5 s after the slider settles."""
        t_now = self.window.time()
        want = min(int(p["n_points"]) * 1000, MAX_POINTS)
        if want != self.model.max_points:
            if self.budget_wanted != want:
                self.budget_wanted, self.budget_t = want, t_now
            elif t_now - self.budget_t > BUDGET_DEBOUNCE_S:
                self.model.set_budget(want)
        else:
            self.budget_wanted = None

    def _send_monitor(self, routed):
        """T8: monitoring packet for the panel (VU meters + status bar)."""
        now = self.window.time()
        if now < self.next_monitor:
            return
        self.next_monitor = now + ControlPanel.MONITOR_INTERVAL
        f = routed.features
        self.panel.send_monitor({
            "bands": [float(b) for b in routed.bands],
            "rms": float(f["rms"]),
            "bpm": float(f.get("bpm", 0.0) or self.sources.tempo),
            "fps": self.clock.fps,
            "source": self.sources.label,
        })

    def _draw_hud(self, features: dict):
        # the HUD texture is re-uploaded at most 5×/s and only when its text
        # changed: writing a texture the GPU still reads stalls the pipeline
        now = self.window.time()
        hud, hud_bytes = self.hud, None
        if now >= self.hud_next:
            self.hud_next = now + HUD_PERIOD_S
            last = hud._last
            surf = hud.build(features, features.get("bpm", 0.0) or self.sources.tempo,
                             self.sources.is_paused, self.model.mode, self.model.n)
            if hud._last != last or not self.renderer.has_overlay(hud.width, hud.height):
                hud_bytes = hud.to_bytes(surf)
        self.renderer.draw_overlay(hud_bytes, hud.width, hud.height)

    def shutdown(self):
        """Arrêt : source, thread des modèles, autosave, panneau, fenêtre."""
        self.sources.stop()
        self.loader.shutdown()
        self.panel.camera_state = self.camera.state()
        self.panel.save_autosave()   # settings + camera — audio source and model are not saved
        self.panel.close()
        self.window.close()
        pygame.quit()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graphic Art — GPU Audio Visualizer")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Audio file (optional — sources can be picked in the panel)")
    parser.add_argument("--visual", metavar="FILE", default=DEFAULT_VISUAL,
                        help="3D model to load (default: primitive:torus)")
    args = parser.parse_args()

    if args.audio and not os.path.isfile(args.audio):
        print(f"[error] File not found: {args.audio}")
        sys.exit(1)

    try:
        app = App(args.visual)
    except RuntimeError as e:   # fenêtre / contexte OpenGL impossible
        print(f"[error] {e}")
        sys.exit(1)
    if args.audio:
        app.sources.start_file(args.audio)
    else:
        print("\nNo source — pick a file or a live input in the Source tab.\n")
    app.run()


if __name__ == "__main__":
    main()
