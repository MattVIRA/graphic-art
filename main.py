"""
Graphic Art — GPU Audio Visualizer
Usage: python main.py [audio_file] [--visual FILE]

Visual: a 3D model (.glb .gltf .obj .stl .ply), also via drag & drop / panel,
drawn as points, surface or wireframe (style chosen in the panel).
  none                      → the built-in torus (DEFAULT_VISUAL)
  --visual model.glb        → that model
  --visual primitive:box    → another built-in shape (icosphere, torus, box)

Keys: SPACE pause/resume  |  ESC quit  |  U back to the torus  |  F fullscreen

The window is a GLFW window (native Retina framebuffer); pygame only provides
the audio playback (mixer) and the HUD font.
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import glfw
import pygame
import moderngl

from audio.analyzer import AudioAnalyzer
from audio.player import AudioPlayer
from gfx.renderer import GLRenderer
from gfx.controls import ControlPanel, EFFECTS, EQ_BANDS
from visuals import mesh as meshlib
from visuals.particles import ParticleSystem

# ── Config ──────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (4, 4, 10)

FADE = 0.30            # trail decay per frame (higher = shorter trails)
BLOOM_INTENSITY = 0.9

DEFAULT_VISUAL = "primitive:torus"   # shown at launch and on U / « Retour au tore »


# ── T4: attack/release envelopes ────────────────────────────────────────────

class Envelope:
    """Asymmetric smoother for audio features: fast attack (a kick still hits
    instantly), slow release (the effect breathes out instead of flickering).
    Works on scalars and numpy arrays, one state per key."""

    def __init__(self, attack_ms: float = 20.0, release_ms: float = 180.0):
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self._state = {}

    @staticmethod
    def _coef(ms: float, dt: float) -> float:
        import math
        return 1.0 - math.exp(-dt / max(ms / 1000.0, 1e-4))

    def smooth(self, key: str, value, dt: float, reactivity: float = 1.0):
        """reactivity scales the release: >1 = percussive, <1 = floaty."""
        import numpy as np
        prev = self._state.get(key)
        if prev is None or np.shape(prev) != np.shape(value):
            self._state[key] = np.array(value, dtype=np.float32) \
                if not np.isscalar(value) else float(value)
            return value
        ka = self._coef(self.attack_ms, dt)
        kr = self._coef(self.release_ms / max(reactivity, 0.1), dt)
        if np.isscalar(prev):
            k = ka if value > prev else kr
            new = prev + (value - prev) * k
        else:
            v = np.asarray(value, dtype=np.float32)
            k = np.where(v > prev, ka, kr).astype(np.float32)
            new = prev + (v - prev) * k
        self._state[key] = new
        return new


class Spring:
    """Damped oscillator kicked by impulses: snaps out, overshoots a little
    and settles — an elastic response instead of a linear follow."""

    def __init__(self, freq_hz: float = 3.0, damping: float = 0.35):
        self.w = 2.0 * math.pi * freq_hz
        self.z = damping
        self.x = self.v = 0.0

    def kick(self, amount: float):
        # velocity such that the first peak ≈ amount
        self.v += amount * self.w * 1.5

    def step(self, dt: float) -> float:
        n = max(1, int(dt / 0.004))          # stable sub-steps
        h = dt / n
        for _ in range(n):
            a = -self.w * self.w * self.x - 2.0 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


SMOOTHED_KEYS = ("bass", "low_mid", "mid", "high",
                 "hz_10_11", "hz_11_12", "hz_12_14", "rms")


class FrameClock:
    """dt + fps measurement and a frame-rate cap (the physics is per frame,
    so a 120 Hz display must not double the animation speed)."""

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


# ── HUD ──────────────────────────────────────────────────────────────────────

class Hud:
    """Renders a one-line status bar to a pygame surface → uploaded as a GL texture.
    Drawn at the framebuffer resolution (scale = pixel ratio) for sharp text."""

    HEIGHT = 40     # logical px

    def __init__(self, width: int, scale: float = 1.0):
        self.scale = scale
        self.width = int(width * scale)
        self.height = int(self.HEIGHT * scale)
        self.font = pygame.font.SysFont("monospace", int(14 * scale))
        self.surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._last = None

    def build(self, features, tempo, paused, mode, n):
        line = (f"BPM {tempo:.0f}  |  RMS {features['rms']:.2f}  |  "
                f"BASS {features['bass']:.2f}  |  {(mode or '…').upper()}  |  {n} pts")
        hint = ("SPACE pause   F plein écran   molette zoom   ESC quitter"
                "   U retour au tore   dépose un modèle 3D")
        if paused:
            line += "   [PAUSED]"
        key = (line, hint)
        if key == self._last:
            return self.surf
        self._last = key
        self.surf.fill((0, 0, 0, 0))
        k = self.scale
        self.surf.blit(self.font.render(line, True, (120, 120, 150)),
                       (int(12 * k), int(4 * k)))
        self.surf.blit(self.font.render(hint, True, (70, 60, 80)),
                       (int(12 * k), int(21 * k)))
        return self.surf

    def to_bytes(self, surf):
        # flipped=True → GL bottom-up origin so the strip displays upright
        return pygame.image.tostring(surf, "RGBA", True)


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

    # ── audio (pygame mixer) + font — no pygame window ──────────────────────
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.font.init()

    # ── GLFW window + OpenGL 3.3 core context (Retina framebuffer) ──────────
    glfw.init_hint(glfw.COCOA_CHDIR_RESOURCES, glfw.FALSE)
    if not glfw.init():
        print("[error] GLFW init failed")
        sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
    window = glfw.create_window(WIDTH, HEIGHT, "Graphic Art", None, None)
    if not window:
        glfw.terminate()
        print("[error] could not create the OpenGL window")
        sys.exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    clock = FrameClock(FPS)

    # input → event queue, drained in the main loop
    events = []
    glfw.set_key_callback(window, lambda w, key, sc, action, mods:
                          events.append(("key", key)) if action == glfw.PRESS else None)
    glfw.set_scroll_callback(window, lambda w, dx, dy: events.append(("wheel", dy)))
    glfw.set_drop_callback(window, lambda w, paths:
                           events.extend(("drop", p) for p in paths))
    windowed_geom = None     # (x, y, w, h) to restore after fullscreen

    def toggle_fullscreen():
        nonlocal windowed_geom
        if windowed_geom is None:
            x, y = glfw.get_window_pos(window)
            windowed_geom = (x, y, *glfw.get_window_size(window))
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            glfw.set_window_monitor(window, mon, 0, 0, mode.size.width,
                                    mode.size.height, mode.refresh_rate)
        else:
            glfw.set_window_monitor(window, None, *windowed_geom, 0)
            windowed_geom = None

    def pixel_ratio():
        ww, _ = glfw.get_window_size(window)
        fw, _ = glfw.get_framebuffer_size(window)
        return fw / max(ww, 1)

    ctx = moderngl.create_context()

    # ── GPU renderer + 3D model ──────────────────────────────────────────────
    renderer = GLRenderer(ctx, (WIDTH, HEIGHT), bg_color=BG_COLOR, fade=FADE,
                          bloom_intensity=BLOOM_INTENSITY)
    renderer.set_pixel_ratio(pixel_ratio())
    max_points = 1_000_000       # upper bound of the « Nombre de points » slider
    particles = ParticleSystem(WIDTH, HEIGHT, max_points=max_points)
    gpu_rev = -1     # last model revision uploaded to the GPU

    # ── Audio source (file player or live analyzer), switchable at runtime ──
    source = None          # object exposing get_current_features/toggle_pause/stop
    hud_tempo = 0.0

    SILENT_FEATURES = {
        "bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high": 0.0,
        "hz_10_11": 0.0, "hz_11_12": 0.0, "hz_12_14": 0.0,
        "rms": 0.0, "transients": 0.0, "centroid": 0.0,
        "is_beat": False, "is_onset": False, "frame_idx": 0,
        "bpm": 0.0, "beat_phase": 0.0,
        "spectrum": __import__("numpy").zeros(AudioAnalyzer.N_MELS, dtype="f4"),
        "eq_bands": __import__("numpy").zeros(7, dtype="f4"),
        "kick_id": 0, "kick_amp": 0.0, "snare_id": 0, "snare_amp": 0.0,
        "section": 0.0,
    }

    def stop_source():
        nonlocal source, hud_tempo
        if source is not None:
            source.stop()
        source = None
        hud_tempo = 0.0

    def start_file(path):
        """Analyze then play an audio file. On any failure (missing, empty or
        undecodable file) print why and keep the current source playing."""
        nonlocal source, hud_tempo
        print(f"\nLoading: {path}")
        if not os.path.isfile(path):
            print(f"[error] audio file not found: {path}")
            return
        if os.path.getsize(path) == 0:
            print(f"[error] audio file is empty (0 bytes): {path}")
            return
        try:
            analyzer = AudioAnalyzer(path)
            player = AudioPlayer(path, analyzer)
        except Exception as e:
            print(f"[error] cannot read audio file {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e or 'format not recognised'}")
            return
        stop_source()    # only once the new file is known to be readable
        player.play()
        source = player
        hud_tempo = analyzer.tempo
        glfw.set_window_title(window, f"Graphic Art  ·  {os.path.basename(path)}")

    def start_live(device):
        nonlocal source
        stop_source()
        from audio.live import LiveAnalyzer
        print(f"\nLive capture on input device {device}")
        source = LiveAnalyzer(device=device)
        glfw.set_window_title(window, "Graphic Art  ·  LIVE")

    hud = Hud(WIDTH, renderer.dpr)
    panel = ControlPanel()   # second window with live sliders

    # ── Visual: a 3D model, parsed in a worker thread so the animation keeps
    # running (the current model stays on screen until the new one is ready) ──
    model_loader = ThreadPoolExecutor(max_workers=1)
    pending_model = None     # (path, up, Future) of the model being prepared

    def load_visual(path):
        nonlocal pending_model
        if not path.startswith("primitive:") and not os.path.isfile(path):
            print(f"[warning] File not found: {path}")
            return
        if not meshlib.is_model_path(path):
            ext = os.path.splitext(path)[1].lower()
            print(f"[warning] Unsupported format: {ext or path} — 3D models "
                  f"{sorted(meshlib.MODEL_EXT)}")
            return
        print(f"  Loading model: {path} …")
        up = int(panel.params["model_up"])
        # the up axis travels with the job: if the panel changes it
        # meanwhile, gpu_uniforms() sees the mismatch and rebuilds
        pending_model = (path, up, model_loader.submit(
            meshlib.prepare, path, up,
            particles.model_radius, particles.max_points))

    load_visual(args.visual)

    # ── Play ─────────────────────────────────────────────────────────────────
    if args.audio:
        start_file(args.audio)
    else:
        print("\nNo source — pick a file or a live input in the Source tab.\n")

    running = True
    wheel_zoom = 1.0     # multiplied with the panel's Zoom slider
    axis_precession = 0.0   # accumulated axis spin around the world vertical
    beat_env = 0.0          # beat flash envelope (spikes on beat, fast decay)
    envelope = Envelope()   # T4: attack/release smoothing of the features
    kick_spring = Spring()  # KICK band: elastic response to each kick
    last_kick = last_snare = 0
    snare_env = 0.0
    next_monitor = 0.0      # T8: next time a monitoring packet is due
    budget_wanted, budget_t = None, 0.0   # debounced n_points rebuild
    hud_next = 0.0          # next HUD text refresh
    while running:
        glfw.poll_events()
        if glfw.window_should_close(window):
            running = False
        for kind, val in events:
            if kind == "key":
                if val == glfw.KEY_ESCAPE:
                    running = False
                elif val == glfw.KEY_SPACE:
                    if source is not None:
                        source.toggle_pause()
                elif val == glfw.KEY_U:
                    load_visual(DEFAULT_VISUAL)
                elif val == glfw.KEY_F:
                    toggle_fullscreen()
            elif kind == "wheel":
                wheel_zoom = max(0.2, min(4.0, wheel_zoom * (1.06 ** val)))
            elif kind == "drop":
                load_visual(val)
        events.clear()

        if pending_model is not None and pending_model[2].done():
            path, up, fut = pending_model
            pending_model = None
            try:
                particles.set_model(fut.result(), up)
            except Exception as e:
                print(f"[error] model load failed ({path}): {e}")

        if source is not None and source.is_done:
            stop_source()    # file ended: keep the window open, go silent

        # the window size (and the pixel ratio, when it moves to another
        # screen) can change at any time: watch both every frame
        win_size = glfw.get_window_size(window)
        dpr = pixel_ratio()
        if (win_size[0] > 0 and win_size[1] > 0
                and (win_size != (renderer.w, renderer.h)
                     or abs(dpr - renderer.dpr) > 1e-3)):
            renderer.set_pixel_ratio(dpr)
            renderer.resize(*win_size)
            hud = Hud(win_size[0], dpr)

        features = (source.get_current_features()
                    if source is not None else SILENT_FEATURES)
        p = panel.poll()
        for name, payload in panel.pop_actions():
            if name == "load_visual" and payload:
                load_visual(payload)
            elif name == "default_visual":
                load_visual(DEFAULT_VISUAL)
            elif name == "load_audio" and payload:
                start_file(payload)
            elif name == "start_live":
                try:
                    start_live(int(payload))
                except Exception as e:
                    print(f"[error] live capture failed: {e}")
            elif name == "stop_source":
                stop_source()

        # ── particle budget: rebuild ~0.5 s after the slider settles ──
        t_now = glfw.get_time()
        want_budget = min(int(p["n_points"]) * 1000, max_points)
        if want_budget != particles.max_points:
            if budget_wanted != want_budget:
                budget_wanted, budget_t = want_budget, t_now
            elif t_now - budget_t > 0.5:
                particles.set_budget(want_budget)
        else:
            budget_wanted = None

        # ── T4: smooth the features (fast attack, slow release) ──
        dt = max(clock.dt, 1e-3)
        react = p["reactivity"]
        features = dict(features)
        for k in SMOOTHED_KEYS:
            features[k] = float(envelope.smooth(k, features[k], dt, react))
        features["spectrum"] = envelope.smooth("spectrum", features["spectrum"],
                                               dt, react)
        features["eq_bands"] = envelope.smooth("eq_bands", features["eq_bands"],
                                               dt, react)
        # (beat_phase / is_beat / transients stay sharp on purpose)

        # beat flash: spikes on each beat then decays quickly
        if features.get("is_beat"):
            beat_env = 1.0
        else:
            beat_env *= 0.86

        # ── percussive layer: kick spring, snare flash, section energy ──
        kid, sid = features.get("kick_id", 0), features.get("snare_id", 0)
        if kid != last_kick:
            if kid > last_kick:
                kick_spring.kick(features.get("kick_amp", 0.0))
            last_kick = kid
        if sid != last_snare:
            if sid > last_snare:
                snare_env = max(snare_env, features.get("snare_amp", 0.0))
            last_snare = sid
        kick_x = kick_spring.step(dt)
        snare_env *= math.exp(-dt / 0.12)
        section = float(envelope.smooth("section", features.get("section", 0.0),
                                        dt, 0.15))

        # ── Physics routing: each EQ band drives its chosen effect ──
        fx = {name: 0.0 for name in EFFECTS}
        # "BEAT": 1.0 on the beat, decaying to 0 until the next one
        beat_pulse = ((1.0 - features.get("beat_phase", 0.0)) ** 2
                      if features.get("bpm", 0.0) > 0 else 0.0)
        bands = list(features["eq_bands"]) + [beat_pulse, kick_x, snare_env,
                                              section]
        # macro-dynamics: every routed effect is damped in calm sections
        dyn_gain = 1.0 - p.get("dynamics", 0.0) * 0.75 * (1.0 - section)
        for i in range(len(EQ_BANDS)):
            eff = EFFECTS[int(p[f"band{i}_fx"])]
            fx[eff] += float(bands[i]) * p[f"band{i}_amt"] * dyn_gain

        # ── T8: monitoring packet for the panel (VU meters + status bar) ──
        now = glfw.get_time()
        if now >= next_monitor:
            next_monitor = now + ControlPanel.MONITOR_INTERVAL
            if source is None:
                src_label = "AUCUNE SOURCE"
            elif type(source).__name__ == "LiveAnalyzer":
                src_label = "LIVE"
            else:
                src_label = "FICHIER"
            panel.send_monitor({
                "bands": [float(b) for b in bands],
                "rms": float(features["rms"]),
                "bpm": float(features.get("bpm", 0.0) or hud_tempo),
                "fps": clock.fps,
                "source": src_label,
            })

        # apply live control-panel parameters (+ routed effects)
        renderer.fade = p["trail"] * max(0.12, 1.0 - min(fx["Trainées"], 1.2) * 0.75)
        renderer.bloom_intensity = p["bloom"]
        renderer.bloom_radius = p["bloom_radius"]
        renderer.bloom_threshold = p["bloom_threshold"]
        renderer.vignette = p["vignette"]
        renderer.set_render_scale(p["render_scale"])
        renderer.set_size_mult(p["point_size"] * (1.0 + fx["Taille points"] * 0.6))
        sh = 0.012
        renderer.set_post_params(
            brightness=p["brightness"] * (1.0 + fx["Luminosité"] * 0.8)
                       * (1.0 + beat_env * 0.18),   # subtle flash on each beat
            gain=(p["red"], p["green"], p["blue"]),
            contrast=p["contrast"],
            invert=p["negative"],
            grain=p["grain"],
            time=glfw.get_time(),
            shift_r=(fx["Décalage rouge"] * sh, 0.0),
            shift_g=(0.0, fx["Décalage vert"] * sh),
            shift_b=(-fx["Décalage bleu"] * sh, 0.0),
        )

        # ── GPU draw pass (T1: stateless physics, the CPU only sends uniforms) ──
        p2 = dict(p)
        p2["fx_rot"] = fx["Vitesse rotation"]
        p2["inflate"] = p["inflate"] + fx["Gonflement"]
        # no motion unless Turbulence is routed in Physique; the Objet tab
        # slider only scales that routed amount
        p2["model_turb"] = p["turbulence"] * fx["Turbulence"]
        rp = particles.gpu_uniforms(features, p2, time_s=glfw.get_time())
        if particles.rev != gpu_rev:
            renderer.upload_particle_static(*particles.build_static())
            renderer.upload_mesh(particles.model_mesh)
            gpu_rev = particles.rev
        rp["scale"] *= 1.0 + fx["Zoom (pulse)"] * 0.25
        deg = 3.14159265 / 180.0
        # axis precession: the rotation axis itself spins around the scene's
        # vertical axis at the origin when enabled
        if p["axis_spin"] >= 0.5:
            axis_precession += p["axis_spin_speed"] * deg * clock.dt

        t_now = glfw.get_time()
        sk = fx["Secousse caméra"] * 14.0
        shake = (sk * 0.5 * (math.sin(t_now * 23.1) + math.sin(t_now * 31.7 + 1.3)),
                 sk * 0.5 * (math.sin(t_now * 27.3 + 0.7) + math.sin(t_now * 19.9)))
        # before set_transform (view matrix)
        renderer.set_zoom(p["zoom"] * wheel_zoom, shake)
        renderer.set_transform(
            rp["angle"], rp["scale"],
            axis_tilt=p["axis_tilt"] * deg,
            axis_azimuth=p["axis_azimuth"] * deg + axis_precession,
            orbit_dist=p["orbit_dist"],
            obj_tilt=p["obj_tilt"] * deg,
        )
        renderer.begin_frame()
        if p["show_grid"] >= 0.5:
            renderer.draw_grid()
        if p["show_axis"] >= 0.5:
            renderer.draw_axis()
        model_style = int(p["model_style"])   # 0 points, 1 surface, 2 wireframe
        rim = p["rim"] + max(fx["Rim"], 0.0) * 1.5
        light = max(0.15, 1.0 + fx["Lumière"] * 0.8)
        if model_style > 0 and renderer.has_mesh:
            renderer.set_model_look(matcap=p["matcap"], rim=rim,
                                    rim_hue=p["rim_hue"], orig=p["orig_color"],
                                    transparent=p["mesh_transparent"] >= 0.5,
                                    shading=p["shading"], roughness=p["roughness"],
                                    metal=p["metal"], ao=p["ao"],
                                    wire_width=p["wire_width"], light=light)
            renderer.draw_model_mesh(rp, model_style,
                                     sharp_only=p["wire_sharp"] >= 0.5,
                                     hide_hidden=p["wire_hidden"] >= 0.5)
        else:
            renderer.set_model_look(rim=rim, rim_hue=p["rim_hue"], light=light)
            renderer.draw_particles_gpu(rp)
        renderer.end_frame()

        # the HUD texture is re-uploaded at most 5×/s and only when its text
        # changed: writing a texture the GPU still reads stalls the pipeline
        now = glfw.get_time()
        hud_bytes = None
        if now >= hud_next:
            hud_next = now + 0.2
            last = hud._last
            surf = hud.build(features, features.get("bpm", 0.0) or hud_tempo,
                             source.is_paused if source is not None else False,
                             particles.mode, particles.n)
            if hud._last != last or not renderer.has_overlay(hud.width, hud.height):
                hud_bytes = hud.to_bytes(surf)
        renderer.draw_overlay(hud_bytes, hud.width, hud.height)

        glfw.swap_buffers(window)
        clock.tick()

    stop_source()
    model_loader.shutdown(wait=False, cancel_futures=True)
    panel.save_autosave()   # settings only — audio source and model are not saved
    panel.close()
    glfw.terminate()
    pygame.quit()


if __name__ == "__main__":
    main()
