"""
Banc de rendu hors écran : rend quelques scènes de référence en PNG pour
comparer la qualité d'image avant / après une modification du renderer.

    python tools/snapshot.py                       # toutes les scènes
    python tools/snapshot.py --visual primitive:torus --tag avant
    python tools/snapshot.py --visual inputs/3D_models/x/scene.gltf
    python tools/snapshot.py --orbit 40,25 --grid   # caméra orbitale (T1.13)
    python tools/snapshot.py --bg-recipe Dunes     # fond (T1.16) : un paysage intégré
    python tools/snapshot.py --seed 42 --style surface   # paysage tiré au dé
    python tools/snapshot.py --bg-recipe Collines --bg-set bg_vo_caves=0.6   # + un réglage
    python tools/snapshot.py --bg-recipe Dunes --orbit 30,10 --orbit-speed 12,3 \
        --frames 120 --save-every 30      # caméra qui tourne, plusieurs frames (passe 5)

Sortie : captures/bench/<tag>/<scène>.png (+ temps GPU moyen par frame).
Les features audio sont synthétiques et déterministes (silence / fort) ;
chaque modèle est rendu dans les 3 styles (points, surface, wireframe).
"""

import argparse
import os
import sys
import time

import moderngl
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.decor import Decor  # noqa: E402
from background import schema as bg_schema  # noqa: E402
from gfx.camera import OrbitCamera  # noqa: E402
from gfx.controls import DEFAULTS  # noqa: E402
from gfx.renderer import GLRenderer  # noqa: E402
from visuals import mesh as meshlib  # noqa: E402
from visuals.model_source import ModelSource  # noqa: E402

W, H = 1280, 720
STYLES = ("points", "surface", "wire")
N_MELS = 64


def features(level: float, t: float) -> dict:
    """Synthetic features: a 120 BPM kick + a sloped spectrum, scaled by level."""
    phase = (t * 2.0) % 1.0
    kick = level * np.exp(-phase * 6.0)
    spec = level * np.clip(1.0 - np.linspace(0, 1, N_MELS) * 0.7, 0, 1).astype("f4")
    eq = np.array([kick, kick * 0.8, level * 0.5, level * 0.5,
                   level * 0.4, level * 0.3, level * 0.3], dtype="f4")
    return {
        "bass": kick, "high": level * 0.3,
        "rms": level * 0.6, "transients": level * 0.3,
        "is_beat": phase < 1 / 60, "frame_idx": 0,
        "bpm": 120.0 if level > 0 else 0.0, "beat_phase": phase,
        "spectrum": spec, "eq_bands": eq,
    }


def grab(renderer) -> Image.Image:
    fbo = renderer.output_fbo
    img = Image.frombytes("RGB", fbo.size, fbo.read(components=3))
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def render_scene(ctx, renderer, particles, p, level, camera, grid=False,
                 frames=45, decor=None, orbit=None, stem=None, save_every=0):
    """Render `frames` frames (trails settle) and return the last one + ms/frame.
    orbit = (lacet, tangage, vitesse lacet, vitesse tangage) en degrés et
    degrés/s : la caméra tourne pendant le rendu ; save_every > 0 :
    enregistre aussi une frame sur N (`<stem>_fNNN.png`)."""
    times = []
    for i in range(frames):
        t = i / 60.0
        if orbit is not None:
            yaw, pitch, dyaw, dpitch = orbit
            camera.load_state({"yaw": np.radians(yaw + dyaw * t),
                               "pitch": np.radians(pitch + dpitch * t),
                               "target": [0.0, 0.0, 0.0], "zoom": 1.0})
        f = features(level, t)
        rp = particles.gpu_uniforms(f, dict(p, model_turb=0.0, fx_rot=0.0),
                                    time_s=t)
        renderer.set_model_look(matcap=p["matcap"], rim=p["rim"],
                                rim_hue=p["rim_hue"], orig=p["orig_color"],
                                transparent=False, shading=p["shading"],
                                roughness=p["roughness"], metal=p["metal"],
                                ao=p["ao"], wire_width=p["wire_width"])
        renderer.bloom_threshold = p["bloom_threshold"]
        renderer.vignette = p["vignette"]
        renderer.set_view(camera.view_matrix(renderer.cam_dist, p["zoom"]))
        if decor is not None:   # T1.16 : mêmes sources que le routing (BEAT, KICK…)
            bands = list(f["eq_bands"]) + [f["bass"], f["bass"], 0.0, level]
            decor.frame(renderer, p, f, bands, 1.0 / 60.0,
                        camera.view_matrix(renderer.cam_dist, p["zoom"],
                                           amount=p["bg_comp_parallax"]),
                        particles.model_radius, t)
        deg = np.pi / 180.0
        renderer.set_transform(0.6 + t * 0.4, rp["scale"],
                               axis_tilt=p["axis_tilt"] * deg,
                               obj_tilt=p["obj_tilt"] * deg)
        t0 = time.perf_counter()
        renderer.begin_frame()
        if grid and decor is None:   # fond visible : grille posée sur le relief
            renderer.draw_grid()
        style = int(p["model_style"])
        if style > 0 and renderer.has_mesh:
            renderer.draw_model_mesh(rp, style)
        else:
            renderer.draw_particles_gpu(rp)
        renderer.end_frame()
        ctx.finish()
        times.append((time.perf_counter() - t0) * 1000.0)
        if save_every > 0 and (i + 1) % save_every == 0 and i < frames - 1:
            grab(renderer).save(f"{stem}_f{i + 1:03d}.png")
    return grab(renderer), float(np.median(times[5:]))


def make_camera(args):
    """Caméra orbitale du banc ; orbit = None si elle ne bouge pas."""
    camera = OrbitCamera()
    if not (args.orbit or args.orbit_speed):
        return camera, None
    yaw, pitch = (float(v) for v in (args.orbit or "0,0").split(","))
    dyaw, dpitch = (float(v) for v in (args.orbit_speed or "0,0").split(","))
    camera.load_state({"yaw": np.radians(yaw), "pitch": np.radians(pitch),
                       "target": [0.0, 0.0, 0.0], "zoom": 1.0})
    return camera, (yaw, pitch, dyaw, dpitch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", action="append",
                    help="primitive:torus, un .glb… (répétable). Défaut : le tore")
    ap.add_argument("--tag", default="current", help="sous-dossier de sortie")
    ap.add_argument("--dpr", type=float, default=2.0,
                    help="densité de pixels de la sortie (2 = Retina)")
    ap.add_argument("--scale", type=float, default=None,
                    help="super-échantillonnage (défaut : celui du panneau)")
    ap.add_argument("--orbit", default=None, metavar="LACET,TANGAGE",
                    help="position de la caméra orbitale en degrés (T1.13)")
    ap.add_argument("--orbit-speed", default=None, metavar="LACET,TANGAGE",
                    help="la caméra tourne pendant le rendu (degrés/s)")
    ap.add_argument("--frames", type=int, default=45, help="frames rendues par image")
    ap.add_argument("--save-every", type=int, default=0, metavar="N",
                    help="enregistre aussi une frame toutes les N (_fNNN.png)")
    ap.add_argument("--grid", action="store_true", help="dessiner la grille")
    ap.add_argument("--bg-recipe", default=None, metavar="NOM",
                    help="fond (T1.16) : un paysage intégré ("
                         + ", ".join(bg_schema.AMBIANCES) + ")")
    ap.add_argument("--seed", type=int, default=None,
                    help="fond : paysage tiré au dé (après la recette s'il y en a une)")
    ap.add_argument("--bg-quality", type=int, default=1, help="0 Bas, 1 Moyen, 2 Haut")
    ap.add_argument("--bg-set", action="append", default=[], metavar="CLÉ=VALEUR",
                    help="un réglage du fond (répétable), ex. bg_ld_valley=3")
    ap.add_argument("--style", action="append", choices=STYLES, default=None,
                    help="styles rendus (répétable) ; défaut : les trois")
    args = ap.parse_args()

    camera, orbit = make_camera(args)

    out_dir = os.path.join("captures", "bench", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    ctx = moderngl.create_standalone_context(require=330)
    ow, oh = int(W * args.dpr), int(H * args.dpr)
    out_fbo = ctx.framebuffer(color_attachments=[ctx.texture((ow, oh), 4)])
    renderer = GLRenderer(ctx, (W, H), bg_color=(4, 4, 10),
                          fade=DEFAULTS["trail"], bloom_intensity=DEFAULTS["bloom"],
                          output_fbo=out_fbo)
    renderer.set_pixel_ratio(args.dpr)
    renderer.bloom_radius = DEFAULTS["bloom_radius"]
    renderer.set_render_scale(args.scale or DEFAULTS["render_scale"])

    p = dict(DEFAULTS)
    decor = None
    if args.bg_recipe is not None or args.seed is not None or args.bg_set:
        if args.bg_recipe is not None:
            names = {n.lower(): n for n in bg_schema.AMBIANCES}
            name = names.get(args.bg_recipe.lower())
            if name is None:
                ap.error(f"recette inconnue : {args.bg_recipe}")
            p.update(bg_schema.AMBIANCES[name])
        if args.seed is not None:
            p.update(bg_schema.random_values(args.seed))
        p.update(bg_visible=1.0, bg_quality=float(args.bg_quality),
                 show_grid=1.0 if args.grid else 0.0)
        for item in args.bg_set:
            key, value = item.split("=")
            p[key] = float(value)
        decor = Decor(ctx, report=lambda msg, exc=None: print(msg, exc or ""),
                      sync_terrain=True)
    for vis in args.visual or ["primitive:torus"]:
        particles = ModelSource(W, H)
        name = os.path.splitext(os.path.basename(vis.replace("primitive:", "")))[0]
        particles.set_model(meshlib.prepare(vis, 0, particles.model_radius,
                                            particles.max_points), 0)
        renderer.upload_particle_static(*particles.build_static())
        renderer.upload_mesh(particles.model_mesh)
        for style in [STYLES.index(s) for s in args.style or STYLES]:
            for level, lname in ((0.0, "silence"), (1.0, "fort")):
                p["model_style"] = float(style)
                renderer.fade = p["trail"]
                sname = STYLES[style]
                stem = os.path.join(out_dir, f"{name}_{sname}_{lname}")
                img, ms = render_scene(ctx, renderer, particles, p, level,
                                       camera, args.grid, frames=args.frames,
                                       decor=decor, orbit=orbit, stem=stem,
                                       save_every=args.save_every)
                path = stem + ".png"
                img.save(path)
                bg = f", fond {decor.gpu_ms:.2f} ms" if decor and decor.gpu_ms else ""
                print(f"  {path}  ({ms:.2f} ms/frame{bg})")


if __name__ == "__main__":
    main()
