"""Paramètres du panneau + effets routés → réglages du renderer, puis passe
de dessin (sorti de main.py par T1.15).

Les fonctions pures (`model_params`, `zoom_pulse`, `shake`) sont testées
dans tests/test_routing.py ; `apply_look` et `draw` ne font qu'appeler le
renderer.
"""

import math

DEG = math.pi / 180.0
SHIFT = 0.012        # amplitude des décalages R/V/B (fraction de l'écran)


def apply_look(renderer, p: dict, fx: dict, beat_env: float, t: float):
    """Post-traitement : trainées, bloom, couleurs, décalages R/V/B."""
    renderer.fade = p["trail"] * max(0.12, 1.0 - min(fx["Trainées"], 1.2) * 0.75)
    renderer.bloom_intensity = p["bloom"]
    renderer.bloom_radius = p["bloom_radius"]
    renderer.bloom_threshold = p["bloom_threshold"]
    renderer.vignette = p["vignette"]
    renderer.set_render_scale(p["render_scale"])
    renderer.set_size_mult(p["point_size"] * (1.0 + fx["Taille points"] * 0.6))
    renderer.set_post_params(
        brightness=p["brightness"] * (1.0 + fx["Luminosité"] * 0.8)
                   * (1.0 + beat_env * 0.18),   # subtle flash on each beat
        gain=(p["red"], p["green"], p["blue"]),
        contrast=p["contrast"],
        invert=p["negative"],
        grain=p["grain"],
        time=t,
        shift_r=(fx["Décalage rouge"] * SHIFT, 0.0),
        shift_g=(0.0, fx["Décalage vert"] * SHIFT),
        shift_b=(-fx["Décalage bleu"] * SHIFT, 0.0),
    )


def model_params(p: dict, fx: dict) -> dict:
    """Paramètres passés à `ModelSource.gpu_uniforms()` : ceux du panneau
    + les effets routés sur le modèle."""
    p2 = dict(p)
    p2["fx_rot"] = fx["Vitesse rotation"]
    if p["music_spin"] < 0.5:   # T1.13: freeze the model to explore it
        p2["rot_gain"] = 0.0
    p2["inflate"] = p["inflate"] + fx["Gonflement"]
    # no motion unless Turbulence is routed in Physique; the Objet tab
    # slider only scales that routed amount
    p2["model_turb"] = p["turbulence"] * fx["Turbulence"]
    return p2


def zoom_pulse(fx: dict) -> float:
    """Facteur d'échelle du modèle donné par l'effet « Zoom (pulse) »."""
    return 1.0 + fx["Zoom (pulse)"] * 0.25


def shake(fx: dict, t: float) -> tuple:
    """Décalage écran (px) de la « Secousse caméra » : somme de sinus
    incommensurables, pour un tremblement sans motif visible."""
    sk = fx["Secousse caméra"] * 14.0
    return (sk * 0.5 * (math.sin(t * 23.1) + math.sin(t * 31.7 + 1.3)),
            sk * 0.5 * (math.sin(t * 27.3 + 0.7) + math.sin(t * 19.9)))


def draw(renderer, p: dict, fx: dict, rp: dict, axis_precession: float):
    """Transformation du modèle, puis grille, axe et modèle dans le style
    choisi. La vue (caméra) doit être posée avant (`renderer.set_view`)."""
    renderer.set_transform(
        rp["angle"], rp["scale"],
        axis_tilt=p["axis_tilt"] * DEG,
        axis_azimuth=p["axis_azimuth"] * DEG + axis_precession,
        orbit_dist=p["orbit_dist"],
        obj_tilt=p["obj_tilt"] * DEG,
    )
    renderer.begin_frame()
    if p["show_grid"] >= 0.5 and p["bg_visible"] < 0.5:
        renderer.draw_grid()      # fond visible : la grille est posée sur le relief
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
