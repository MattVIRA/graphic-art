"""
Banc de rendu hors écran : rend quelques scènes de référence en PNG pour
comparer la qualité d'image avant / après une modification du renderer.

    python tools/snapshot.py                       # toutes les scènes
    python tools/snapshot.py --visual primitive:torus --tag avant
    python tools/snapshot.py --visual inputs/3D_models/x/scene.gltf

Sortie : captures/bench/<tag>/<scène>.png (+ temps GPU moyen par frame).
Les features audio sont synthétiques et déterministes (silence / fort) ;
chaque modèle est rendu dans les 3 styles (points, surface, wireframe).
"""

import argparse
import os
import sys
import time

import numpy as np
import moderngl
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gfx.controls import DEFAULTS            # noqa: E402
from gfx.renderer import GLRenderer          # noqa: E402
from visuals import mesh as meshlib          # noqa: E402
from visuals.particles import ParticleSystem  # noqa: E402

W, H = 1280, 720
N_MELS = 64


def features(level: float, t: float) -> dict:
    """Synthetic features: a 120 BPM kick + a sloped spectrum, scaled by level."""
    phase = (t * 2.0) % 1.0
    kick = level * np.exp(-phase * 6.0)
    spec = level * np.clip(1.0 - np.linspace(0, 1, N_MELS) * 0.7, 0, 1).astype("f4")
    eq = np.array([kick, kick * 0.8, level * 0.5, level * 0.5,
                   level * 0.4, level * 0.3, level * 0.3], dtype="f4")
    return {
        "bass": kick, "low_mid": level * 0.5, "mid": level * 0.5, "high": level * 0.3,
        "hz_10_11": level * 0.2, "hz_11_12": level * 0.2, "hz_12_14": level * 0.2,
        "rms": level * 0.6, "transients": level * 0.3, "centroid": 0.4,
        "is_beat": phase < 1 / 60, "is_onset": False, "frame_idx": 0,
        "bpm": 120.0 if level > 0 else 0.0, "beat_phase": phase,
        "spectrum": spec, "eq_bands": eq,
    }


def render_scene(ctx, renderer, particles, p, level, frames=45):
    """Render `frames` frames (trails settle) and return the last one + ms/frame."""
    times = []
    for i in range(frames):
        t = i / 60.0
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
        renderer.set_zoom(p["zoom"])
        deg = np.pi / 180.0
        renderer.set_transform(0.6 + t * 0.4, rp["scale"],
                               axis_tilt=p["axis_tilt"] * deg,
                               obj_tilt=p["obj_tilt"] * deg)
        t0 = time.perf_counter()
        renderer.begin_frame()
        style = int(p["model_style"])
        if style > 0 and renderer.has_mesh:
            renderer.draw_model_mesh(rp, style)
        else:
            renderer.draw_particles_gpu(rp)
        renderer.end_frame()
        ctx.finish()
        times.append((time.perf_counter() - t0) * 1000.0)
    fbo = renderer.output_fbo
    img = Image.frombytes("RGB", fbo.size, fbo.read(components=3))
    return img.transpose(Image.FLIP_TOP_BOTTOM), float(np.median(times[5:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", action="append",
                    help="primitive:torus, un .glb… (répétable). Défaut : le tore")
    ap.add_argument("--tag", default="current", help="sous-dossier de sortie")
    ap.add_argument("--dpr", type=float, default=2.0,
                    help="densité de pixels de la sortie (2 = Retina)")
    ap.add_argument("--scale", type=float, default=None,
                    help="super-échantillonnage (défaut : celui du panneau)")
    args = ap.parse_args()

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
    for vis in args.visual or ["primitive:torus"]:
        particles = ParticleSystem(W, H)
        name = os.path.splitext(os.path.basename(vis.replace("primitive:", "")))[0]
        particles.set_model(meshlib.prepare(vis, 0, particles.model_radius,
                                            particles.max_points), 0)
        renderer.upload_particle_static(*particles.build_static())
        renderer.upload_mesh(particles.model_mesh)
        for style in (0, 1, 2):
            for level, lname in ((0.0, "silence"), (1.0, "fort")):
                p["model_style"] = float(style)
                renderer.fade = p["trail"]
                img, ms = render_scene(ctx, renderer, particles, p, level)
                sname = ["points", "surface", "wire"][style]
                path = os.path.join(out_dir, f"{name}_{sname}_{lname}.png")
                img.save(path)
                print(f"  {path}  ({ms:.2f} ms/frame)")


if __name__ == "__main__":
    main()
