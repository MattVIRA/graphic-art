"""
Images d'aperçu des motifs de la texture du relief (T29), affichées sur
les boutons de l'onglet Fond : un rendu du banc (tools/snapshot.py) par
motif, vu de dessus sur le paysage « Dunes », recadré hors du modèle.

    python tools/texture_thumbs.py

Sortie : gfx/assets/textures/motif_<n>.png (versionnées, à refaire quand
un motif change).
"""

import os
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from background import schema  # noqa: E402

OUT_DIR = os.path.join(ROOT, "gfx", "assets", "textures")
SIZE = (112, 63)
CROP = (0.0, 0.25, 0.32, 0.56)   # fraction de l'image : à gauche du modèle


def render(index: int) -> str:
    tag = f"thumb_{index}"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "snapshot.py"),
                    "--bg-recipe", "Dunes", "--style", "surface", "--orbit", "0,65",
                    "--frames", "30", "--tag", tag, "--bg-set", f"bg_tex_motif={index}",
                    # motif plus petit et plus dense : plusieurs formes dans la vignette
                    "--bg-set", "bg_tex_scale=1.2", "--bg-set", "bg_tex_density=2.5",
                    "--bg-set", "bg_tex_sharp=0.85"],
                   cwd=ROOT, check=True, capture_output=True)
    return os.path.join(ROOT, "captures", "bench", tag, "torus_surface_silence.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, name in enumerate(schema.TEX_MOTIFS):
        im = Image.open(render(i)).convert("RGB")
        w, h = im.size
        box = (int(CROP[0] * w), int(CROP[1] * h), int(CROP[2] * w), int(CROP[3] * h))
        path = os.path.join(OUT_DIR, f"motif_{i}.png")
        im.crop(box).resize(SIZE, Image.LANCZOS).save(path, optimize=True)
        print(f"  {path}  ({name})")


if __name__ == "__main__":
    main()
