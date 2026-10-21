// Fond SDF (T1.16) — uniforms calculés en Python (background/params.py) et
// fonctions de distance de base. Assemblé par background/pass_.py :
// #version + #define de qualité + uniforms du schéma + ce fichier +
// landscape.glsl + main.glsl.

uniform mat4  u_inv_vp;     // inverse(P · V) de la caméra du fond (px du monde)
uniform vec3  u_eye;        // position de la caméra (px du monde)
uniform float u_unit;       // px du monde par unité du fond (rayon du modèle)
uniform float u_pix;        // taille angulaire d'un pixel du fond (rad)
uniform vec2  u_center_uv;  // projection écran du centre du modèle
uniform float u_aspect;     // largeur / hauteur
uniform sampler2D u_palette;
uniform int   u_zero;       // toujours 0 : empêche le compilateur de dérouler map()

uniform mat3  u_rot;        // rotation propre + inclinaison de l'horizon
uniform vec3  u_offset;     // voyage + décalage + dérive
uniform float u_scale;      // échelle globale × respiration
uniform vec3  u_sun_dir, u_sun_col, u_sky_col, u_hor_col, u_shadow_col;
uniform vec3  u_emit_col, u_center_col, u_tint;
uniform float u_flash, u_hue_rot, u_edges;
uniform vec3  u_path_o;     // un point du trajet de l'objet (fixe dans le décor)
uniform mat3  u_path_m;     // repère du trajet : x, y en travers, z le long
uniform float u_detail;     // plus petit détail visible (unités du fond)
uniform vec3  u_model_c;    // centre du modèle (unités du fond, monde)
uniform float u_model_r;    // rayon de l'ombre du modèle (unités du fond)
uniform float u_vo_twist_ph; // phase de la torsion, des ondes : abscisse de
uniform float u_vo_wave_ph;  // l'objet sur le trajet (décor fixe en voyage)
uniform vec3  u_grid_col;   // grille sur le relief (0 = sans grille)

const float PI = 3.14159265;

// œil du fond dans l'espace des formes (posé par main() avant la marche) :
// le paysage choisit le niveau de détail de ses cartes selon la distance
vec3 g_eye_s = vec3(0.0);

// position dans le repère du trajet : xy = écart au trajet, z = le long
vec3 path_local(vec3 p) {
    return u_path_m * (p - u_path_o);
}
