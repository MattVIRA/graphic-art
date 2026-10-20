// Fond SDF (T16) — uniforms calculés en Python (background/params.py) et
// fonctions de distance de base. Assemblé par background/pass_.py :
// #version + #define de qualité + uniforms du schéma + ce fichier +
// chain.glsl + main.glsl.

uniform mat4  u_inv_vp;     // inverse(P · V) de la caméra du fond (px du monde)
uniform vec3  u_eye;        // position de la caméra (px du monde)
uniform float u_unit;       // px du monde par unité du fond (rayon du modèle)
uniform float u_pix;        // taille angulaire d'un pixel du fond (rad)
uniform vec2  u_center_uv;  // projection écran du centre du modèle
uniform float u_aspect;     // largeur / hauteur
uniform sampler2D u_palette;
uniform int   u_zero;       // toujours 0 : empêche le compilateur de dérouler map()

uniform mat3  u_rot;        // rotation propre + inclinaison de l'horizon
uniform vec3  u_offset;     // défilement + décalage + dérive
uniform float u_scale;      // échelle globale × respiration
uniform float u_beat;       // horloge des beats (beats)
uniform float u_time;       // temps du fond (s)
uniform vec3  u_sun_dir, u_sun_col, u_sky_col, u_hor_col, u_shadow_col;
uniform vec3  u_emit_col, u_center_col, u_tint;
uniform float u_flash, u_hue_rot, u_edges;
uniform float u_eq[7];      // bandes EQ lissées × dose musicale
uniform vec3  u_path_o;     // un point du trajet de l'objet (fixe dans le décor)
uniform mat3  u_path_m;     // repère du trajet : x, y en travers, z le long
uniform float u_detail;     // plus petit détail des formes (unités du fond)

const float PI = 3.14159265;

// œil du fond dans l'espace des formes (posé par main() avant la marche) :
// le monde de bruit efface ses couches fines au loin
vec3 g_eye_s = vec3(0.0);

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float hash12(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float sd_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sd_torus(vec3 p, float R, float r) {     // dans le plan XZ
    return length(vec2(length(p.xz) - R, p.y)) - r;
}

float sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// position dans le repère du trajet : xy = écart au trajet, z = le long
vec3 path_local(vec3 p) {
    return u_path_m * (p - u_path_o);
}

// distance au trajet de l'objet : le couloir de voyage reste vide
float path_dist(vec3 p) {
    return length(path_local(p).xy);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}
