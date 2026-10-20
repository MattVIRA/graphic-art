// Paysage du fond (T16, passe finale) : le seul générateur. Un terrain
// h(x, z) sous le trajet de l'objet, lu dans deux cartes de hauteur
// périodiques précalculées en Python (background/terrain.py) : la carte
// (grandes couches, déformées, érodées, en terrasses ; creux ; poids du
// détail) et le détail fin. Quelques lectures de texture par pas, quel que
// soit le nombre de couches ; au loin, les mipmaps lissent le relief (pas
// de scintillement) et effacent le détail.
//
// Le terrain s'abaisse en vallée le long du trajet (« Largeur de la
// vallée ») ; la Voûte ajoute un plafond (canyon). Le Volume fait ce qu'un
// champ de hauteur ne sait pas faire : la Distorsion déforme l'espace avant
// le relief (surplombs, espace tordu), les cavités et les bulles creusent
// ou ajoutent de la matière en 3D. Tout s'éteint près du trajet (le
// couloir de l'objet reste libre), sauf la torsion autour du trajet, qui
// le laisse en place (elle s'éteint au loin). Le relief est fixe : rien ne
// bouge avec le temps ni avec le voyage.
// Unités du fond : le modèle a un rayon de 1.

uniform sampler3D u_noise;        // bruit lisse périodique (background/noise.py)
uniform sampler2D u_terrain;      // carte : hauteur ≈ −1…1, creux, poids du détail
uniform sampler2D u_terrain_fine; // détail fin (une taille du monde)
uniform vec2  u_ld_res;           // texels de la carte, du détail
uniform float u_ld_slope;         // pente de la carte (par taille du monde)
uniform float u_ld_dslope;        // pente du détail (par taille du monde)
uniform float u_ld_dgain;         // détail → hauteur de la carte (0 = sans détail)
uniform float u_ld_damp;          // plus grand écart du détail (hauteur de la carte)
uniform float u_vo_lip;           // étirement dû aux ondes et au warp (Python)
uniform float u_vo_cave_thr;      // seuil des cavités (quantile réel du bruit)
uniform float u_vo_G;             // pente du bruit 3D centré par unité (taille 1)

const float NOISE_CELLS = 8.0;    // = noise.CELLS
const float TILE = 12.0;          // = terrain.TILE
const float DETAIL_TILE = 1.0;    // = terrain.DETAIL_TILE
const float FLOOR = 2.2;          // fond de la vallée sous le trajet (unités)
const float VALLEY_EDGE = 0.6;    // flanc de la vallée (tailles du monde)
const float VAULT_SCALE = 0.7;    // la voûte est lue plus petite que le sol
const float TERRAIN_SLOPE = 6.0;  // pente maximale prise en compte (pas de marche)
const float VALLEY_RELIEF = 0.3;  // part des grandes couches gardée au fond de la vallée
const float RELIEF_K = 0.1;       // hauteur (tailles du monde) par unité de Hauteur
const float WARP_GAIN = 0.3;      // = params.WARP_GAIN : déplacement / (warp × taille)
const float BUBBLE_MAX = 0.45;    // = params.BUBBLE_MAX : rayon + écart ≤ 45 % de la cellule
const float TWIST_REACH = 0.25;   // torsion pleine jusqu'à vallée + ¼ de flanc…
const float TWIST_FADE = 1.0;     // … puis éteinte sur un flanc
const float FAR = 1e5;

// quatre nombres au hasard par cellule en un seul hachage (présence, rayon,
// position, genre d'une bulle : six hachages coûtaient le tiers du fond)
vec4 hash43(vec3 p) {
    vec4 q = fract(vec4(p.xyzx) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    q += dot(q, q.wzxy + 33.33);
    return fract((q.xxyz + q.yzzw) * q.zywx);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}

float smax_poly(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

float smin_poly(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// B-spline cubique en 4 lectures bilinéaires : de près, la hauteur est
// lisse et les normales ne dessinent pas les texels
vec4 bspline(sampler2D s, vec2 uv, float res) {
    vec2 x = uv * res - 0.5;
    vec2 i = floor(x);
    vec2 f = x - i;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1, g1 = 1.0 - g0;
    // chaque paire de texels voisins = une lecture bilinéaire bien placée
    vec2 a = (i - 0.5 + w1 / g0) / res;
    vec2 b = (i + 1.5 + w3 / g1) / res;
    return g0.y * (g0.x * textureLod(s, a, 0.0) + g1.x * textureLod(s, vec2(b.x, a.y), 0.0))
         + g1.y * (g0.x * textureLod(s, vec2(a.x, b.y), 0.0) + g1.x * textureLod(s, b, 0.0));
}

// niveau de mipmap pour un pixel de taille `foot` sur des texels de `texel`
float lod_of(float foot, float texel) {
    return max(log2(foot / texel), 0.0);
}

// carte au point xz (unités des formes) : (hauteur ≈ −1…1, creux, hauteur
// sans le détail, part du détail présente) ; foot = taille d'un pixel ici
vec4 terrain_map(vec2 xz, float foot) {
    float S = u_ld_scale;
    float lod = lod_of(foot, S * TILE / u_ld_res.x);
    vec2 uv = xz / (S * TILE);
    vec3 c = lod < 1.0 ? bspline(u_terrain, uv, u_ld_res.x).rgb
                       : textureLod(u_terrain, uv, lod).rgb;
    vec4 m = vec4(c.x, c.y, c.x, 0.0);
    // détail fin : de près seulement (au loin, ses mipmaps tendent vers 0)
    float dlod = lod_of(foot, S * DETAIL_TILE / u_ld_res.y);
    float fade = 1.0 - smoothstep(2.0, 4.0, dlod);
    if (u_ld_dgain > 0.0 && fade > 0.0) {
        float d = textureLod(u_terrain_fine, xz / (S * DETAIL_TILE), dlod).r;
        m.x += d * c.z * u_ld_dgain * fade;
        m.w = fade;
    }
    return m;
}

float pixel_at(vec3 q) {
    return max(length(q - g_eye_s) * u_pix, 1e-4);
}

// amplitude du relief (la carte vaut ≈ −1…1) : 0,1 taille du monde par
// unité de Hauteur ; à 0,3, la pente médiane dépassait 60° et les flancs
// presque verticaux d'un champ de hauteur tombent en rideaux
float relief_height() {
    return u_ld_relief * RELIEF_K * u_ld_scale;
}

float floor_level() {
    return u_path_o.y - FLOOR / u_scale;
}

// (0 au fond de la vallée, le long du trajet, 1 hors de la vallée ; pente
// de ce passage par unité, nulle hors du flanc) ; p : point non déformé
vec2 valley_at(vec3 p) {
    float rv = u_ld_valley / u_scale;
    float w = VALLEY_EDGE * u_ld_scale;
    float x = clamp((abs(path_local(p).x) - rv) / w, 0.0, 1.0);
    return vec2(x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x) / w);
}

// hauteur du sol : grandes couches `base` (écrasées au fond de la vallée,
// toujours sous l'objet) + détail `det` (partout : c'est la texture que le
// premier plan montre) ; v = valley_at
float ground_at(vec2 v, float base, float det) {
    float fl = floor_level();
    float H = relief_height();
    float h = fl + H * base;
    float low = fl + clamp((h - fl) * VALLEY_RELIEF, -3.0 / u_scale, 0.6 / u_scale);
    return mix(low, h, v.x) + H * det;
}

// hauteur du sol ; m : la carte en ce point (terrain_map)
float ground_height(vec2 v, vec4 m) {
    return ground_at(v, m.z, m.x - m.z);
}

float ceiling_height(vec3 q, vec2 v, float foot) {
    vec4 m2 = terrain_map(q.zx / VAULT_SCALE + 37.0 * u_ld_scale, foot / VAULT_SCALE);
    float H = relief_height();
    float top = floor_level() + 2.2 * H + 6.0 / u_scale - H * m2.x;
    // au-dessus de la vallée, la voûte reste loin de l'objet
    return mix(max(top, u_path_o.y + u_ld_valley / u_scale), top, v.x);
}

// ── Volume 1 : Distorsion (avant le relief : tout l'espace se déforme) ─
// rend le point déformé ; lip est multiplié par l'étirement local. Éteinte
// dans le couloir du trajet (rayon = vallée), sur un flanc de vallée.
// clear : distance au bord de la zone déformée (> 0 dans le couloir, où
// rien n'est déformé : le pas peut aller jusqu'à ce bord sans étirement)
vec3 distort(vec3 p, inout float lip, out float clear) {
    clear = -1.0;
    if (u_vo_twist <= 0.0 && u_vo_bend == 0.0 && u_vo_wave_amp <= 0.0 && u_vo_warp <= 0.0)
        return p;
    vec3 a = path_local(p);
    float rv = u_ld_valley / u_scale;
    float W = VALLEY_EDGE * u_ld_scale;
    float rc = length(a.xy);
    float fade = smoothstep(rv, rv + W, rc);
    float dfade = (rc > rv && rc < rv + W) ? 1.5 / W : 0.0;   // pente de l'extinction
    clear = rv - rc;
    if (u_vo_twist > 0.0) {
        // autour du trajet : le couloir reste en place. Au loin, la torsion
        // s'éteint : sinon l'air tournait avec le sol et, passé un quart de
        // tour, il n'y avait plus de sol sous le trajet (un vide sans fond,
        // noyé de brume). On mélange les points, pas les angles : l'angle
        // grandit sans fin le long du trajet, sa pente en travers n'aurait
        // pas de borne ; l'écart |R·a − a| ≤ 2 rc, lui, est borné
        float w = 2.0 * PI * u_vo_twist * 0.01;
        float r0 = rv + TWIST_REACH * W, rw = TWIST_FADE * W;
        float g = smoothstep(r0, r0 + rw, rc);
        float dg = (rc > r0 && rc < r0 + rw) ? 1.5 / rw : 0.0;
        a.xy = mix(rot2(w * a.z + u_vo_twist_ph) * a.xy, a.xy, g);
        lip *= (1.0 - g) * sqrt(1.0 + w * w * rc * rc) + g + 2.0 * rc * dg;
    }
    if (u_vo_bend != 0.0) {
        float k = u_vo_bend * 0.01;
        float x = clamp(a.x, -60.0, 60.0);
        a.y += k * x * x * fade;
        lip *= 1.0 + abs(k) * (2.0 * abs(x) * fade + x * x * dfade);
    }
    if (u_vo_wave_amp > 0.0) {
        float A = u_vo_wave_amp * fade;
        float k = 2.0 * PI / u_vo_wave_len;
        a.x += A * sin(k * a.z + u_vo_wave_ph);
        a.y += A * sin(k * 0.73 * a.x + 1.7);
    }
    if (u_vo_warp > 0.0) {
        float S = u_ld_scale;
        vec3 uvw = a / (u_vo_warp_size * S * NOISE_CELLS);
        vec3 n = vec3(0.0);
        float amp = 1.0;
        for (int i = u_zero; i < 3; i++) {
            if (float(i) >= u_vo_warp_oct) break;
            n += (texture(u_noise, uvw).rgb - 0.5) * amp;
            uvw = uvw * 2.03 + 0.37;
            amp *= 0.5;
        }
        a += n * 2.0 * WARP_GAIN * u_vo_warp * S * fade;
    }
    lip *= u_vo_lip;
    return u_path_o + transpose(u_path_m) * a;
}

// ── Volume 2 : cavités, un champ de bruit 3D (deux couches) ───────────
// F ≈ −1…1 ; lip = sa pente maximale par unité
float cave_field(vec3 q, out float lip) {
    float C = u_vo_cave_size * u_ld_scale;
    float F = (texture(u_noise, q / (C * NOISE_CELLS)).r - 0.5) * 2.0;
    // couche fine effacée quand elle ferait moins de ~8 px (pas de grain)
    float w2 = u_vo_cave_detail * 0.35 * smoothstep(4.0, 12.0, C * 0.33 / pixel_at(q));
    if (w2 > 0.001)
        F += w2 * (texture(u_noise, q / (C * 0.33 * NOISE_CELLS) + 0.5).g - 0.5) * 2.0;
    lip = u_vo_G / C * (1.0 + w2 / 0.33);
    return F;
}

// ── Volume 3 : bulles, une par cellule au plus ────────────────────────
// (distance à la plus proche, creusée ? 1 : 0) ; rayon + écart du centre
// ≤ 45 % de la cellule : les 8 cellules autour du point contiennent toutes
// les bulles à moins d'une demi-cellule, au-delà la borne suffit
vec2 bubbles(vec3 q) {
    float c = u_vo_bubble_size * u_ld_scale;
    vec3 base = floor(q / c - 0.5);
    float best = 0.5 * c, carved = 0.0;
    float rv = u_ld_valley / u_scale;
    for (int i = u_zero; i < 8; i++) {
        vec3 id = base + vec3(float(i & 1), float((i >> 1) & 1), float((i >> 2) & 1));
        vec4 h = hash43(id + 7.0);
        if (h.x > u_vo_bubbles) continue;
        float r = 0.18 + 0.22 * h.y;
        vec3 off = (vec3(h.z, h.w, fract(h.x * 17.3 + h.y)) - 0.5) * 2.0 * (BUBBLE_MAX - r);
        vec3 center = (id + 0.5 + off) * c;
        int mode = int(u_vo_bubble_mode + 0.5);
        float hollow = mode == 0 ? 1.0 : mode == 1 ? 0.0 : step(0.5, fract(h.y * 31.7));
        // une bulle pleine n'entre pas dans le couloir de l'objet
        if (hollow < 0.5 && length(path_local(center).xy) < rv + 2.0 * r * c) continue;
        float d = length(q - center) - r * c;
        if (d < best) { best = d; carved = hollow; }
    }
    return vec2(best, carved);
}

// (distance, altitude 0–1, 0) ; m : la carte au point déformé
vec3 landscape(vec3 p, out vec4 m) {
    float S = u_ld_scale;
    float H = relief_height();
    float lip = 1.0, clear;
    vec3 q = distort(p, lip, clear);
    vec2 v = valley_at(p);
    float foot = pixel_at(q);
    m = terrain_map(q.xz, foot);
    // un champ de hauteur n'est pas une distance : l'écart vertical est
    // divisé par la pente (carte, flanc de la vallée là où il est), bornée
    // (au-delà, les pas s'épuisaient devant la caméra). Deux bornes, on
    // garde la plus grande : avec la pente du détail (de près), ou sans
    // elle mais au-dessus du plus haut détail possible — le détail est
    // raide mais minuscule, sa pente ne raccourcit les pas que tout près
    float k = H / S;
    float flank = v.y * H * (abs(m.z) + 1.0);
    float slope = min(k * u_ld_slope + flank, TERRAIN_SLOPE);
    float d = (q.y - ground_at(v, m.z, u_ld_damp * m.w)) / sqrt(1.0 + slope * slope);
    if (m.w > 0.0) {
        float sd = min(slope + k * m.w * u_ld_dslope * u_ld_dgain, TERRAIN_SLOPE);
        d = max(d, (q.y - ground_height(v, m)) / sqrt(1.0 + sd * sd));
    }
    if (u_ld_vault > 0.5) {
        float s2 = min(k * u_ld_slope / VAULT_SCALE + flank * 2.0, TERRAIN_SLOPE);
        d = min(d, (ceiling_height(q, v, foot) - q.y) / sqrt(1.0 + s2 * s2));
    }
    if (u_vo_caves > 0.0) {
        float lipF;
        float F = cave_field(q, lipF);
        float C = u_vo_cave_size * S;
        if (u_vo_cave_mode < 0.5) {
            // creusées : le relief perd ce qui dépasse le seuil (arches, grottes)
            d = smax_poly(d, (F - u_vo_cave_thr) / lipF, 0.08 * C);
        } else {
            // roche dans tout l'espace : une caverne, le trajet y est un tunnel
            d = min(d, (u_vo_cave_thr - F) / lipF);
            float rv = max(u_ld_valley / u_scale, 2.0 / u_scale);
            d = smax_poly(d, rv - length(path_local(p).xy), 0.5 * rv);
        }
    }
    if (u_vo_bubbles > 0.0) {
        // les bulles restent près du sol : une bulle pleine s'arrête, en
        // douceur, à 1,5 cellule au-dessus de lui (écart vertical : la
        // distance au relief sous-estime la hauteur sur les pentes, des
        // bulles flottaient dans le ciel ; coupe arrondie : pas de dôme
        // tranché), une bulle creusée n'agit que dans la roche. Plus
        // haut, l'écart à cette couche, ramené par la pente, borne la
        // distance : on ne parcourt pas les cellules du ciel
        float c = u_vo_bubble_size * S;
        float kb = max(u_vo_bubble_soft * 0.3 * c, 1e-3);
        bool full = u_vo_bubble_mode > 0.5;
        float layer = (q.y - ground_height(v, m) - 1.5 * c) / sqrt(1.0 + slope * slope);
        if (layer > c) {
            if (full) d = min(d, layer);
        } else if (full || d < kb) {
            vec2 b = bubbles(q);
            d = b.y > 0.5 ? smax_poly(d, -b.x, kb)
                          : smin_poly(d, smax_poly(b.x, layer, 0.5 * c), kb);
        }
    }
    // dans le couloir, la Distorsion est éteinte (torsion exceptée) : sans
    // étirement jusqu'au bord de la zone déformée, étiré au-delà
    float dl = clear > 0.0 ? min(d / (lip / u_vo_lip), clear + d / lip) : d / lip;
    return vec3(dl, clamp(0.5 + 0.5 * (q.y - floor_level()) / max(H, 1e-3), 0.0, 1.0),
                0.0);
}

// occlusion précalculée des creux (canal « creux » de la carte), au point
// p de la surface : le sol est sombre dans ses creux, la voûte dans ses
// bosses ; les parois des cavités et des bulles n'en reçoivent pas
float terrain_occlusion(vec3 p) {
    float lip = 1.0, clear;
    vec3 q = distort(p, lip, clear);
    vec2 v = valley_at(p);
    float foot = pixel_at(q);
    vec4 m = terrain_map(q.xz, foot);
    float H = relief_height();
    float cav = m.y;
    float dg = abs(q.y - ground_height(v, m));
    if (u_ld_vault > 0.5) {
        float dc = abs(ceiling_height(q, v, foot) - q.y);
        if (dc < dg) {
            dg = dc;
            cav = -terrain_map(q.zx / VAULT_SCALE + 37.0 * u_ld_scale, foot / VAULT_SCALE).y;
        }
    }
    if (dg > 0.05 * H + 0.2) return 1.0;
    // creux sans unité × pente moyenne du relief (Hauteur / Taille)
    return clamp(1.0 - 1.5 * u_ld_relief * RELIEF_K * max(cav, 0.0), 0.3, 1.0);
}

float map(vec3 p) {
    vec4 m;
    return landscape(p, m).x;
}

// (structure = altitude, hasard = grandes nappes de bruit, 0) ; le hasard
// n'est lu qu'ici, pas à chaque pas de marche
vec3 material(vec3 p) {
    vec4 m;
    vec3 l = landscape(p, m);
    float h = smoothstep(0.25, 0.75,
                         texture(u_noise, p / (u_ld_scale * 4.0 * NOISE_CELLS)).b);
    return vec3(l.y, h, 0.0);
}
