"""
Graphic Art — GPU Audio Visualizer
Usage: python main.py [audio_file] [--visual FILE] [--particles N]

Visual source (picked from the file extension, also via drag & drop / panel):
  none                    → generative star (particles encode the FFT)
  --visual photo.png      → image: particles on the picture, react to the music
  --visual model.glb      → 3D model (.glb .gltf .obj .stl .ply): points,
                            surface or wireframe (style chosen in the panel)
  --visual primitive:torus  → built-in test shape (icosphere, torus, box)
  (--image is kept as an alias of --visual)

Keys: SPACE pause/resume  |  ESC quit  |  U back to the star
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pygame
import moderngl

from audio.analyzer import AudioAnalyzer
from audio.player import AudioPlayer
from gfx.renderer import GLRenderer
from gfx.controls import ControlPanel, EFFECTS, EQ_BANDS, SCOPABLE_EFFECTS
from visuals import mesh as meshlib
from visuals.particles import ParticleSystem

# ── Config ──────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (4, 4, 10)

FADE = 0.30            # trail decay per frame (higher = shorter trails)
BLOOM_INTENSITY = 0.9

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


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


SMOOTHED_KEYS = ("bass", "low_mid", "mid", "high",
                 "hz_10_11", "hz_11_12", "hz_12_14", "rms")


# ── HUD ──────────────────────────────────────────────────────────────────────

class Hud:
    """Renders a one-line status bar to a pygame surface → uploaded as a GL texture."""

    HEIGHT = 40

    def __init__(self, width: int):
        self.width = width
        self.font = pygame.font.SysFont("monospace", 14)
        self.surf = pygame.Surface((width, self.HEIGHT), pygame.SRCALPHA)
        self._last = None

    def build(self, features, tempo, paused, mode, n):
        line = (f"BPM {tempo:.0f}  |  RMS {features['rms']:.2f}  |  "
                f"BASS {features['bass']:.2f}  |  {mode.upper()}  |  {n} pts")
        hint = ("SPACE pause   F fullscreen   molette zoom   ESC quit"
                + ("   U back to star" if mode != "star"
                   else "   drop an image or a 3D model"))
        if paused:
            line += "   [PAUSED]"
        key = (line, hint)
        if key == self._last:
            return self.surf
        self._last = key
        self.surf.fill((0, 0, 0, 0))
        self.surf.blit(self.font.render(line, True, (120, 120, 150)), (12, 4))
        self.surf.blit(self.font.render(hint, True, (70, 60, 80)), (12, 21))
        return self.surf

    def to_bytes(self, surf):
        # flipped=True → GL bottom-up origin so the strip displays upright
        return pygame.image.tostring(surf, "RGBA", True)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graphic Art — GPU Audio Visualizer")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Audio file (optional — sources can be picked in the panel)")
    parser.add_argument("--visual", "--image", dest="visual", metavar="FILE",
                        default=None,
                        help="Image or 3D model to load (or primitive:torus)")
    parser.add_argument("--particles", type=int, default=50000, metavar="N",
                        help="Number of particles in generative mode (default 50000)")
    args = parser.parse_args()

    if args.audio and not os.path.isfile(args.audio):
        print(f"[error] File not found: {args.audio}")
        sys.exit(1)

    # ── pygame + OpenGL context ──────────────────────────────────────────────
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Graphic Art")
    clock = pygame.time.Clock()

    ctx = moderngl.create_context()

    # ── GPU renderer + particles ─────────────────────────────────────────────
    max_particles = max(args.particles, 90000)
    renderer = GLRenderer(ctx, (WIDTH, HEIGHT), max_particles,
                          bg_color=BG_COLOR, fade=FADE, bloom_intensity=BLOOM_INTENSITY)
    # T1: with GPU physics every image pixel can be a particle (≤ ~1M);
    # the CPU fallback keeps the old 90k budget
    max_image = 1_000_000 if renderer.gpu_prog is not None else 90_000
    particles = ParticleSystem(WIDTH, HEIGHT, n_generative=args.particles,
                               max_image=max_image, n_mels=AudioAnalyzer.N_MELS)
    gpu_rev = -1     # last particle revision uploaded to the GPU

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
    }

    def stop_source():
        nonlocal source, hud_tempo
        if source is not None:
            source.stop()
        source = None
        hud_tempo = 0.0

    def start_file(path):
        nonlocal source, hud_tempo
        stop_source()
        print(f"\nLoading: {path}")
        analyzer = AudioAnalyzer(path)
        player = AudioPlayer(path, analyzer)
        player.play()
        source = player
        hud_tempo = analyzer.tempo
        pygame.display.set_caption(f"Graphic Art  ·  {os.path.basename(path)}")

    def start_live(device):
        nonlocal source
        stop_source()
        from audio.live import LiveAnalyzer
        print(f"\nLive capture on input device {device}")
        source = LiveAnalyzer(device=device)
        pygame.display.set_caption("Graphic Art  ·  LIVE")

    hud = Hud(WIDTH)
    panel = ControlPanel()   # second window with live sliders

    # ── Visual source: star / image / 3D model, chosen by file extension ──
    # Models are parsed in a worker thread so the animation keeps running.
    model_loader = ThreadPoolExecutor(max_workers=1)
    pending_model = None     # (path, Future) of the model being prepared

    def load_visual(path):
        nonlocal pending_model
        ext = os.path.splitext(path)[1].lower()
        is_primitive = path.startswith("primitive:")
        if not is_primitive and not os.path.isfile(path):
            print(f"[warning] File not found: {path}")
            return
        if ext in SUPPORTED_IMAGE_EXT:
            pending_model = None         # a late model must not replace it
            particles.load_image(path)
        elif meshlib.is_model_path(path):
            if renderer.gpu_prog is None:
                print("[warning] 3D models need the GPU particle path")
                return
            print(f"  Loading model: {path} …")
            up = int(panel.params["model_up"])
            # the up axis travels with the job: if the panel changes it
            # meanwhile, gpu_uniforms() sees the mismatch and rebuilds
            pending_model = (path, up, model_loader.submit(
                meshlib.prepare, path, up,
                particles.model_radius, particles.max_image))
        else:
            print(f"[warning] Unsupported format: {ext or path} — images "
                  f"{sorted(SUPPORTED_IMAGE_EXT)}, models {sorted(meshlib.MODEL_EXT)}")

    def unload_visual():
        nonlocal pending_model
        pending_model = None
        particles.unload(n_generative=args.particles)

    if args.visual:
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
    next_monitor = 0.0      # T8: next time a monitoring packet is due
    budget_wanted, budget_t = None, 0.0   # debounced n_points rebuild
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if source is not None:
                        source.toggle_pause()
                elif event.key == pygame.K_u:
                    unload_visual()
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
            elif event.type == pygame.MOUSEWHEEL:
                wheel_zoom = max(0.2, min(4.0, wheel_zoom * (1.06 ** event.y)))
            elif event.type == pygame.DROPFILE:
                load_visual(event.file)

        if pending_model is not None and pending_model[2].done():
            path, up, fut = pending_model
            pending_model = None
            try:
                particles.set_model(fut.result(), up)
            except Exception as e:
                print(f"[error] model load failed ({path}): {e}")

        if source is not None and source.is_done:
            stop_source()    # file ended: keep the window open, go silent

        # macOS animates the fullscreen transition, so the window size changes
        # some frames AFTER toggle_fullscreen(): watch it every frame instead.
        win_size = pygame.display.get_window_size()
        if win_size != (renderer.w, renderer.h):
            renderer.resize(*win_size)
            hud = Hud(win_size[0])

        features = (source.get_current_features()
                    if source is not None else SILENT_FEATURES)
        p = panel.poll()
        for name, payload in panel.pop_actions():
            if name == "load_visual" and payload:
                load_visual(payload)
            elif name == "unload_image":
                unload_visual()
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
        t_now = pygame.time.get_ticks() * 0.001
        want_budget = min(int(p["n_points"]) * 1000, max_image)
        if want_budget != particles.max_image:
            if budget_wanted != want_budget:
                budget_wanted, budget_t = want_budget, t_now
            elif t_now - budget_t > 0.5:
                particles.set_budget(want_budget)
        else:
            budget_wanted = None

        # ── T4: smooth the features (fast attack, slow release) ──
        dt = max(clock.get_time() / 1000.0, 1e-3)
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

        # ── Physics routing: each EQ band drives its chosen effect ──
        fx = {name: 0.0 for name in EFFECTS}
        # 8th band "BEAT": 1.0 on the beat, decaying to 0 until the next one
        beat_pulse = ((1.0 - features.get("beat_phase", 0.0)) ** 2
                      if features.get("bpm", 0.0) > 0 else 0.0)
        bands = list(features["eq_bands"]) + [beat_pulse]
        # scopable effects can target the subject (fx_fg) or the décor (fx_bg)
        fx_fg = {name: 0.0 for name in SCOPABLE_EFFECTS}
        fx_bg = {name: 0.0 for name in SCOPABLE_EFFECTS}
        for i in range(len(EQ_BANDS)):
            eff = EFFECTS[int(p[f"band{i}_fx"])]
            val = float(bands[i]) * p[f"band{i}_amt"]
            sc_i = int(p.get(f"band{i}_scope", 0.0))
            if eff in SCOPABLE_EFFECTS and sc_i == 1:
                fx_fg[eff] += val
            elif eff in SCOPABLE_EFFECTS and sc_i == 2:
                fx_bg[eff] += val
            else:
                fx[eff] += val
        renderer.set_scoped_fx(
            size_fg=1.0 + fx_fg["Taille points"] * 0.6,
            size_bg=1.0 + fx_bg["Taille points"] * 0.6,
            bright_fg=1.0 + fx_fg["Luminosité"] * 0.8,
            bright_bg=1.0 + fx_bg["Luminosité"] * 0.8,
            turb_fg=1.0 + fx_fg["Turbulence"],
            turb_bg=1.0 + fx_bg["Turbulence"],
        )

        # ── T8: monitoring packet for the panel (VU meters + status bar) ──
        now = pygame.time.get_ticks() * 0.001
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
                "fps": clock.get_fps(),
                "source": src_label,
                "visual": particles.mode,   # greys out the unused Objet group
            })

        # ── scoped color sliders: full frame (composite), subject-only or
        # background-only (per-particle in the shader) ──
        def scoped(key, neutral=1.0):
            s = int(p.get(f"{key}_scope", 0.0))
            v = p[key]
            return (v if s == 0 else neutral,     # composite (post)
                    v if s == 1 else neutral,     # subject
                    v if s == 2 else neutral)     # background
        r_post, r_fg, r_bg = scoped("red")
        g_post, g_fg, g_bg = scoped("green")
        b_post, b_fg, b_bg = scoped("blue")
        br_post, br_fg, br_bg = scoped("brightness")
        con_post, con_fg, con_bg = scoped("contrast")
        neg_post, neg_fg, neg_bg = scoped("negative", 0.0)
        tint_fg = (r_fg * br_fg, g_fg * br_fg, b_fg * br_fg)
        tint_bg = (r_bg * br_bg, g_bg * br_bg, b_bg * br_bg)
        renderer.set_scoped_colors(tint_fg, tint_bg, con_fg, con_bg,
                                   neg_fg, neg_bg)

        # apply live control-panel parameters (+ routed effects)
        renderer.fade = p["trail"] * max(0.12, 1.0 - min(fx["Trainées"], 1.2) * 0.75)
        renderer.bloom_intensity = p["bloom"]
        renderer.bloom_radius = p["bloom_radius"]
        renderer.set_render_scale(p["render_scale"])
        renderer.set_size_mult(p["point_size"] * (1.0 + fx["Taille points"] * 0.6))
        sh = 0.012
        renderer.set_post_params(
            brightness=br_post * (1.0 + fx["Luminosité"] * 0.8)
                       * (1.0 + beat_env * 0.18),   # subtle flash on each beat
            gain=(r_post, g_post, b_post),
            contrast=con_post,
            invert=neg_post,
            grain=p["grain"],
            time=pygame.time.get_ticks() * 0.001,
            shift_r=(fx["Décalage rouge"] * sh, 0.0),
            shift_g=(0.0, fx["Décalage vert"] * sh),
            shift_b=(-fx["Décalage bleu"] * sh, 0.0),
        )

        # ── GPU draw pass ──
        p2 = dict(p)
        p2["turbulence"] = p["turbulence"] + fx["Turbulence"]
        p2["fx_rot"] = fx["Vitesse rotation"]
        p2["inflate"] = p["inflate"] + fx["Gonflement"]
        # 3D model: no motion unless Turbulence is routed in Physique; the
        # Objet tab slider only scales that routed amount
        p2["model_turb"] = p["turbulence"] * (fx["Turbulence"]
                                              + fx_fg["Turbulence"])
        if renderer.gpu_prog is not None:
            # T1: stateless physics in the vertex shader — CPU only sends uniforms
            rp = particles.gpu_uniforms(features, p2,
                                        time_s=pygame.time.get_ticks() * 0.001)
            if particles.rev != gpu_rev:
                renderer.upload_particle_static(*particles.build_static())
                renderer.upload_mesh(particles.model_mesh
                                     if particles.mode == "model" else None)
                gpu_rev = particles.rev
            out, count = None, 0
        else:
            import numpy as _np
            p2["_scope"] = (_np.array(tint_fg, dtype=_np.float32),
                            _np.array(tint_bg, dtype=_np.float32),
                            con_fg, con_bg, neg_fg, neg_bg)
            out, count, rp = particles.update(features, p2)
        rp["scale"] *= 1.0 + fx["Zoom (pulse)"] * 0.25
        deg = 3.14159265 / 180.0
        # axis precession: the rotation axis itself spins around the scene's
        # vertical axis at the origin when enabled
        if p["axis_spin"] >= 0.5:
            axis_precession += p["axis_spin_speed"] * deg * clock.get_time() / 1000.0

        renderer.set_zoom(p["zoom"] * wheel_zoom)   # before set_transform (view matrix)
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
        if (particles.mode == "model" and model_style > 0
                and renderer.has_mesh):
            renderer.set_model_look(matcap=p["matcap"], rim=p["rim"],
                                    rim_hue=p["rim_hue"], orig=p["orig_color"],
                                    transparent=p["mesh_transparent"] >= 0.5)
            renderer.draw_model_mesh(rp, model_style,
                                     sharp_only=p["wire_sharp"] >= 0.5,
                                     hide_hidden=p["wire_hidden"] >= 0.5)
        elif renderer.gpu_prog is not None:
            renderer.set_model_look(rim=p["rim"], rim_hue=p["rim_hue"])
            renderer.draw_particles_gpu(rp)
        else:
            renderer.draw_particles(out, count)
        renderer.end_frame()

        surf = hud.build(features, features.get("bpm", 0.0) or hud_tempo,
                         source.is_paused if source is not None else False,
                         particles.mode, particles.n)
        renderer.draw_overlay(hud.to_bytes(surf), hud.width, Hud.HEIGHT)

        pygame.display.flip()
        clock.tick(FPS)

    stop_source()
    model_loader.shutdown(wait=False, cancel_futures=True)
    panel.save_autosave()   # settings only — source and image are not saved
    panel.close()
    pygame.quit()


if __name__ == "__main__":
    main()
