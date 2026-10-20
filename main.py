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
dessin d'une frame, fenêtre, HUD, journal, erreurs.

Frontières d'erreur (T22) : chaque action de l'utilisateur passe par
`App.dispatch()`, chaque frame par la garde de `App.run()` ; une erreur est
journalisée (logs/graphic-art.log) et affichée dans le panneau et le HUD,
l'appli continue. L'arrêt sauvegarde les réglages dans tous les cas.
"""

import argparse
import logging
import sys

import glfw
import moderngl
import pygame

from app import frame
from app import log as applog
from app.decor import Decor
from app.errors import FrameGuard, UserError
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

log = logging.getLogger("main")

# ── Config ──────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (4, 4, 10)

FADE = 0.30            # trail decay per frame (higher = shorter trails)
BLOOM_INTENSITY = 0.9

MAX_POINTS = 1_000_000   # upper bound of the « Nombre de points » slider
BUDGET_DEBOUNCE_S = 0.5  # rebuild the points this long after the slider settles
HUD_PERIOD_S = 0.2       # HUD text refreshed at most 5×/s
NOTICE_SECS = 5.0        # durée d'affichage d'une erreur dans le HUD
MAX_FRAME_ERRORS = 30    # frames en erreur d'affilée avant l'arrêt (≈ 0,5 s)

# libellé de chaque action dans les messages d'erreur
ACTION_LABELS = {
    "load_visual": "Chargement du modèle",
    "default_visual": "Retour au tore",
    "load_audio": "Lecture audio",
    "start_live": "Capture live",
    "stop_source": "Arrêt de la source",
    "drop": "Chargement du modèle",
    "key": "Raccourci clavier",
}


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

        self.hud = Hud(WIDTH, self.renderer.dpr)
        self.notice = None            # (message, fin d'affichage) : erreur au HUD
        self.panel = ControlPanel()   # second window with live sliders
        self.decor = Decor(ctx, self.report)   # T16 : fond en formes SDF
        self.panel_alive = True
        self.sources = SourceManager(on_title=self.window.set_title,
                                     report=self.report)
        self.loader = ModelLoader(self.model, report=self._model_failed)

        self.camera = OrbitCamera()   # T26: orbit / pan / zoom with the mouse
        if self.panel.camera_state:
            self.camera.load_state(self.panel.camera_state)
        self.gestures = MouseGestures(self.camera)
        self.reactor = Reactor()      # enveloppes + couche percussive + routing
        self.guard = FrameGuard(MAX_FRAME_ERRORS)

        self.running = True
        self.axis_precession = 0.0    # accumulated axis spin around the world vertical
        self.next_monitor = 0.0       # T8: next time a monitoring packet is due
        self.budget_wanted, self.budget_t = None, 0.0   # debounced n_points rebuild
        self.hud_next = 0.0           # next HUD text refresh

        self.keys = {
            glfw.KEY_ESCAPE: self.quit,
            glfw.KEY_SPACE: self.sources.toggle_pause,
            glfw.KEY_U: lambda: self.load_visual(DEFAULT_VISUAL),
            glfw.KEY_F: self.window.toggle_fullscreen,
            glfw.KEY_C: self.camera.reset,
        }
        self.actions = {
            "load_visual": lambda path: path and self.load_visual(path),
            "default_visual": lambda _: self.load_visual(DEFAULT_VISUAL),
            "load_audio": lambda path: path and self.sources.start_file(path),
            "start_live": lambda device: self.sources.start_live(int(device)),
            "stop_source": lambda _: self.sources.stop(),
        }

        # un modèle invalide au lancement (CLI) : le tore à la place
        if not self.dispatch("Chargement du modèle", self.load_visual, visual):
            self.load_visual(DEFAULT_VISUAL)

    # ── erreurs ───────────────────────────────────────────────────────
    def report(self, message: str, exc: BaseException = None):
        """Signale un incident : journal (avec la trace s'il y en a une),
        barre de statut du panneau et ligne temporaire du HUD."""
        if exc is None:
            log.warning("%s", message)
        else:
            log.error("%s", message, exc_info=exc)
        self.panel.report(message)
        self.notice = (message, self.window.time() + NOTICE_SECS)

    def dispatch(self, what: str, fn, *args) -> bool:
        """Frontière d'erreur d'une action de l'utilisateur : `fn(*args)`,
        l'erreur éventuelle est signalée et l'appli continue. True si l'action
        a réussi."""
        try:
            fn(*args)
        except UserError as e:
            self.report(str(e))
            return False
        except Exception as e:  # noqa: BLE001 — frontière : action de l'utilisateur (panneau, touche, dépôt)
            self.report(f"{what} impossible ({type(e).__name__}: {e})", e)
            return False
        return True

    def _model_failed(self, message: str, exc: BaseException):
        """Échec d'un chargement de modèle (thread) : sans modèle à l'écran
        (premier chargement), le tore prend la place."""
        self.report(message, exc)
        if self.model.mode is None and not self.loader.busy:
            self.load_visual(DEFAULT_VISUAL)

    # ── actions ───────────────────────────────────────────────────────
    def quit(self):
        self.running = False

    def load_visual(self, path: str):
        self.loader.load(path, int(self.panel.params["model_up"]))

    def handle_event(self, kind: str, val):
        """Un événement de la fenêtre (touche, souris, fichier déposé)."""
        if kind == "key":
            action = self.keys.get(val)
            if action is not None:
                action()
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
        action = self.actions.get(name)
        if action is None:
            log.warning("action inconnue du panneau : %s", name)
        else:
            action(payload)

    # ── boucle ────────────────────────────────────────────────────────
    def run(self):
        """Boucle principale ; l'arrêt (réglages sauvegardés, panneau et audio
        fermés) a lieu dans tous les cas, même sur une exception."""
        try:
            while self.running:
                self._guarded_frame()
        finally:
            self.shutdown()

    def _guarded_frame(self):
        """Une frame en erreur est sautée ; trop d'erreurs d'affilée → arrêt."""
        try:
            self.frame()
        except Exception as e:  # noqa: BLE001 — frontière : une frame (la suivante repart de zéro)
            new, stop = self.guard.failed(e)
            if new:
                self.report(f"Erreur pendant une frame ({type(e).__name__}: {e}) : "
                            "frame sautée", e)
            if stop:
                self.report(f"{self.guard.limit} frames en erreur d'affilée : arrêt "
                            "(réglages sauvegardés, détail dans logs/graphic-art.log)")
                self.running = False
            self.clock.tick()   # garder la cadence : pas de boucle folle sur une erreur
        else:
            self.guard.ok()

    def frame(self):
        """Une frame : entrées, état, routing, dessin, cadence."""
        if self.window.should_close:
            self.running = False
        for kind, val in self.window.poll():
            self.dispatch(ACTION_LABELS.get(kind, "Souris"), self.handle_event, kind, val)
        self.loader.update()
        self.sources.update()
        self._follow_window_size()

        features = self.sources.features()
        p = self.panel.poll()
        for name, payload in self.panel.pop_actions():
            self.dispatch(ACTION_LABELS.get(name, name), self.handle_action, name, payload)
        self._update_budget(p)

        dt = self.clock.dt
        routed = self.reactor.step(features, p, dt)
        features, fx = routed.features, routed.fx
        self._send_monitor(routed)

        t = self.window.time()
        renderer = self.renderer
        frame.apply_look(renderer, p, fx, routed.beat_env, t)
        # ── GPU draw pass (T1: stateless physics, the CPU only sends uniforms) ──
        rp = self.model.gpu_uniforms(features, frame.model_params(p, fx),
                                     time_s=t, dt=dt)
        if self.model.rev != self.gpu_rev:
            renderer.upload_particle_static(*self.model.build_static())
            renderer.upload_mesh(self.model.model_mesh)
            self.gpu_rev = self.model.rev
        rp["scale"] *= frame.zoom_pulse(fx)
        # axis precession: the rotation axis itself spins around the scene's
        # vertical axis at the origin when enabled
        if p["axis_spin"] >= 0.5:
            self.axis_precession += p["axis_spin_speed"] * frame.DEG * dt
        self.camera.update(dt)   # before set_transform (view matrix)
        shake = frame.shake(fx, t)
        renderer.set_view(self.camera.view_matrix(renderer.cam_dist, p["zoom"], shake))
        # T16 : le fond suit la caméra selon « Parallaxe »
        bg_view = self.camera.view_matrix(renderer.cam_dist, p["zoom"], shake,
                                          amount=p["bg_comp_parallax"])
        self.decor.frame(renderer, p, features, routed.bands, dt, bg_view,
                         self.model.model_radius, t)
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
        if self.panel_alive and not self.panel.alive:
            self.panel_alive = False
            self.report("Panneau de contrôle fermé : les réglages restent tels "
                        "quels, relancer l'appli pour le retrouver")
        f = routed.features
        self.panel.send_monitor({
            "bands": [float(b) for b in routed.bands],
            "rms": float(f["rms"]),
            "bpm": float(f.get("bpm", 0.0) or self.sources.tempo),
            "fps": self.clock.fps,
            "source": self.sources.label,
            "bg_ms": self.decor.gpu_ms,
        })

    def _draw_hud(self, features: dict):
        # the HUD texture is re-uploaded at most 5×/s and only when its text
        # changed: writing a texture the GPU still reads stalls the pipeline
        now = self.window.time()
        hud, hud_bytes = self.hud, None
        if now >= self.hud_next:
            self.hud_next = now + HUD_PERIOD_S
            if self.notice is not None and now > self.notice[1]:
                self.notice = None
            last = hud._last
            surf = hud.build(features, features.get("bpm", 0.0) or self.sources.tempo,
                             self.sources.is_paused, self.model.mode, self.model.n,
                             notice=self.notice[0] if self.notice else None)
            if hud._last != last or not self.renderer.has_overlay(hud.width, hud.height):
                hud_bytes = hud.to_bytes(surf)
        self.renderer.draw_overlay(hud_bytes, hud.width, hud.height)

    def _save_settings(self):
        self.panel.camera_state = self.camera.state()
        self.panel.save_autosave()   # settings + camera — audio source and model are not saved

    def shutdown(self):
        """Arrêt, réglages d'abord : une étape en échec n'empêche pas les
        suivantes."""
        steps = (("sauvegarde des réglages", self._save_settings),
                 ("arrêt de la source", self.sources.stop),
                 ("thread des modèles", self.loader.shutdown),
                 ("fermeture du panneau", self.panel.close),
                 ("fermeture de la fenêtre", self.window.close),
                 ("pygame", pygame.quit))
        for what, step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 — frontière : arrêt, chaque étape doit être tentée
                log.exception("arrêt : %s en échec", what)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    applog.setup()
    parser = argparse.ArgumentParser(description="Graphic Art — GPU Audio Visualizer")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Audio file (optional — sources can be picked in the panel)")
    parser.add_argument("--visual", metavar="FILE", default=DEFAULT_VISUAL,
                        help="3D model to load (default: primitive:torus)")
    args = parser.parse_args()

    try:
        app = App(args.visual)
    except RuntimeError as e:   # fenêtre / contexte OpenGL impossible
        log.critical("%s", e)
        sys.exit(1)
    if args.audio:
        app.dispatch("Lecture audio", app.sources.start_file, args.audio)
    else:
        log.info("Aucune source : choisir un fichier ou une entrée live dans l'onglet Source.")
    try:
        app.run()
    except KeyboardInterrupt:   # Ctrl+C dans le terminal : arrêt déjà fait par run()
        log.info("interrompu")


if __name__ == "__main__":
    main()
