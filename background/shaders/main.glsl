// Fond SDF (T16) — raymarcher commun : rayon de la caméra, marche
// sur-relaxée, normale, ombre, occlusion, couleur, éclairage « extérieur ».
// Sortie : HDR linéaire, alpha 1 ; le composite le glisse sous la scène.
// MAX_STEPS, SHADOW_STEPS, AO_SAMPLES : #define du niveau de qualité.

in vec2 v_uv;
out vec4 f_color;

// point du monde (unités du fond) → distance, après rotation propre,
// défilement et échelle ; le vide central est fait d'instances réduites
// (voir chain.glsl), jamais d'une forme tranchée
float scene(vec3 pw) {
    vec3 q = u_rot * pw + u_offset;
    return map(q / u_scale) * u_scale;
}

// brume de distance (perspective atmosphérique) ; un détail plus fin que
// ~2 pixels s'y fond aussi : pas de grain ni de moiré qui scintille au loin
float haze(float t) {
    float fog = 1.0 - exp(-u_comp_fade * t * 0.06);
    float px = u_detail / max(t * u_pix, 1e-6);
    return max(fog, 1.0 - smoothstep(1.5, 5.0, px));
}

// fond du monde : un horizon lumineux qui donne la profondeur, le ciel au-dessus
vec3 sky(vec3 rd) {
    return u_hor_col * (1.0 + 2.5 * exp(-abs(rd.y) * 5.0))
         + u_sky_col * 0.06 * max(rd.y, 0.0);
}

vec3 scene_material(vec3 pw) {
    return material((u_rot * pw + u_offset) / u_scale);
}

// normale par 4 échantillons en tétraèdre (iq) ; la somme des échantillons
// mesure la courbure → arêtes lumineuses. Boucle bornée par u_zero : le
// compilateur ne copie pas map() quatre fois.
vec4 normal_curv(vec3 p, float e) {
    vec3 n = vec3(0.0);
    float s = 0.0;
    for (int i = u_zero; i < 4; i++) {
        vec3 k = 0.5773 * (2.0 * vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        float f = scene(p + e * k);
        n += k * f;
        s += f;
    }
    return vec4(normalize(n), abs(s) / e);
}

float soft_shadow(vec3 o, vec3 l, float k) {
#if SHADOW_STEPS > 0
    float res = 1.0, t = 0.05;
    for (int i = u_zero; i < SHADOW_STEPS; i++) {
        float h = scene(o + l * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.05, 0.8);
        if (res < 0.01 || t > 12.0) break;
    }
    return clamp(res, 0.0, 1.0);
#else
    return 1.0;
#endif
}

float occlusion(vec3 p, vec3 n, float steps_frac) {
#if AO_SAMPLES > 0
    float occ = 0.0, sca = 1.0;
    for (int i = u_zero; i < AO_SAMPLES; i++) {
        float h = 0.05 + 0.45 * float(i) / float(AO_SAMPLES);
        occ += (h - scene(p + n * h)) * sca;
        sca *= 0.8;
    }
    return clamp(1.0 - 1.2 * occ, 0.0, 1.0);
#else
    return 1.0 - 0.8 * steps_frac;     // Bas : le nombre de pas suffit
#endif
}

vec3 palette(float x) {
    x = abs(fract(x * 0.5) * 2.0 - 1.0);           // lecture en miroir
    return texture(u_palette, vec2(x * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
}

vec3 hue_rotate(vec3 c, float a) {
    const vec3 k = vec3(0.57735);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 albedo(vec3 pos, vec3 n, vec3 rd, float t, float tmax, float steps_frac, vec3 m,
            float sharp) {
    int src = int(u_col_source + 0.5);
    float x = src == 0 ? pos.y * 0.15 + 0.5
            : src == 1 ? t / tmax
            : src == 2 ? dot(n, vec3(0.3, 0.55, 0.78)) * 0.5 + 0.5
            : src == 3 ? m.x
            : src == 4 ? m.y
            : src == 5 ? steps_frac * 2.0
            : length(pos) * 0.12;
    float fres = 1.0 - clamp(dot(n, -rd), 0.0, 1.0);
    x = x * u_col_repeat + u_col_irid * u_col_irid_freq * fres * 0.3 * sharp;
    if (u_col_steps >= 2.0)
        x = floor(x * u_col_steps) / u_col_steps;
    vec3 c = hue_rotate(palette(x), u_hue_rot);
    c = max(mix(vec3(luma(c)), c, u_col_sat), 0.0);
    c = pow(c / 0.25, vec3(u_col_contrast)) * 0.25;
    return c * u_tint;
}

vec3 shade(vec3 pos, vec3 rd, float t, float tmax, float steps_frac) {
    float e = max(u_pix * t, 1e-4);
    // détail de quelques pixels : sa normale est bruitée → arêtes, reflets
    // et irisation s'effacent au lieu de scintiller
    float sharp = smoothstep(2.0, 8.0, u_detail / max(t * u_pix, 1e-6));
    vec4 nc = normal_curv(pos, e);
    vec3 n = nc.xyz;
    vec3 m = scene_material(pos);
    vec3 alb = albedo(pos, n, rd, t, tmax, steps_frac, m, sharp);
    vec3 L = normalize(u_sun_dir);
    vec3 V = -rd;
    float ndl = dot(n, L);
    float sh = 1.0;
    if (u_light_shadow > 0.01 && ndl > 0.0)
        sh = mix(1.0, soft_shadow(pos + n * 2.0 * e, L, mix(2.0, 24.0, u_light_shadow)),
                 min(u_light_shadow * 2.0, 1.0));
    float ao = pow(occlusion(pos, n, steps_frac), u_mat_ao);
    float w = u_mat_transl;
    float dif = clamp((ndl + w) / (1.0 + w), 0.0, 1.0) * sh;
    float sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0) * ao;
    float bounce = clamp(0.5 - 0.5 * n.y, 0.0, 1.0) * ao * 0.25;
    float fres = pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 5.0);
    float sp = exp2(10.0 * (1.0 - u_mat_rough) + 1.0);
    float spec = pow(clamp(dot(n, normalize(L + V)), 0.0, 1.0), sp) * (sp + 8.0) / 25.0;
    vec3 F = mix(vec3(0.04), alb, u_mat_metal);
    F += (1.0 - F) * fres;
    vec3 diff = alb * (1.0 - u_mat_metal);
    vec3 col = diff * (u_sun_col * dif + u_sky_col * (sky + bounce))
             + diff * u_shadow_col * (1.0 - sh) * clamp(ndl, 0.0, 1.0) * 0.6
             + F * u_sun_col * spec * sh * (1.0 - u_mat_rough * 0.8) * sharp
             + F * u_sky_col * u_mat_fresnel * fres * ao * 2.0 * mix(0.3, 1.0, sharp);
    // le modèle, au centre, éclaire le fond
    vec3 lc = -pos;
    float dc = length(lc);
    col += diff * u_center_col * clamp(dot(n, lc / dc), 0.0, 1.0) / (1.0 + dc * dc * 0.08);
    // émission : matière propre, joints, arêtes (courbure), flash
    float edge = smoothstep(0.15, 0.8, nc.w) * u_edges * sharp;
    col += alb * u_mat_emit * 2.0 + u_emit_col * (m.z + edge + u_flash);
    return col;
}

uniform sampler2D u_scene_alpha;   // scène de la frame (alpha 1 = surface opaque)
uniform sampler2D u_scene_depth;   // profondeur de la scène (0–1, fenêtre)
uniform vec2 u_texel;              // taille d'un pixel du fond (uv)
uniform vec2 u_near_far;           // plans de la projection (px du monde)
uniform vec3 u_fwd;                // axe de visée de la caméra (monde)
uniform float u_same_space;        // 1 = même espace que le modèle (Éloignement 0)

const float NO_HIT = 60000.0;      // profondeur « rien touché » (alpha du fond)

// profondeur fenêtre → distance le long de l'axe de visée (px du monde)
float eye_depth(float d) {
    float n = u_near_far.x, f = u_near_far.y;
    return 2.0 * n * f / (f + n - (d * 2.0 - 1.0) * (f - n));
}

vec4 taps(sampler2D s, vec2 uv, vec2 o, int c) {
    return vec4(texture(s, uv - o)[c], texture(s, uv + vec2(o.x, -o.y))[c],
                texture(s, uv + vec2(-o.x, o.y))[c], texture(s, uv + o)[c]);
}

void main() {
    vec2 o = u_texel * 0.35;
    float limit_px = 1e9;      // distance de visée où le modèle arrête le rayon
    if (u_same_space > 0.5) {
        // même espace : la marche s'arrête à la surface du modèle (ce qui est
        // derrière est caché) ; la plus lointaine des 4 profondeurs du pixel
        // du fond : au bord du modèle, la marche reste complète
        vec4 d4 = taps(u_scene_depth, v_uv, o, 0);
        float d = max(max(d4.x, d4.y), max(d4.z, d4.w));
        if (d < 1.0)
            limit_px = eye_depth(d);
    } else {
        // espaces séparés : pixel entièrement caché par une surface opaque
        // (marge d'un pixel et demi : pas de liseré sombre au contour)
        vec4 a4 = taps(u_scene_alpha, v_uv, u_texel * 1.5, 3);
        if (min(min(a4.x, a4.y), min(a4.z, a4.w)) > 0.999) {
            f_color = vec4(0.0, 0.0, 0.0, NO_HIT);
            return;
        }
    }
    vec4 w = u_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 eye = u_eye / u_unit;
    vec3 rd = normalize(w.xyz / w.w / u_unit - eye);
    // Éloignement : l'œil du fond est rapproché du centre ; le fond tourne
    // avec la caméra mais ne glisse presque plus quand elle se déplace,
    // comme un paysage lointain
    vec3 ro = eye * (1.0 - u_comp_far);
    float tmin = 0.0;
    float tmax = u_comp_horizon;
    float view = max(dot(rd, u_fwd), 1e-3);       // px de visée par unité de rayon / unit
    float t_model = limit_px / (view * u_unit);
    tmax = min(tmax, t_model);

    // sphere tracing sur-relaxé avec repli (Keinert et al. 2014)
    float t = tmin, omega = 1.3, prev = 0.0, step_len = 0.0, glow = 0.0;
    bool hit = false;
    int i;
    for (i = 0; i < MAX_STEPS; i++) {
        float r = scene(ro + rd * t);
        float ar = abs(r);
        bool fail = omega > 1.0 && (ar + prev) < step_len;
        if (fail) {
            step_len -= omega * step_len;
            omega = 1.0;
        } else {
            step_len = r * omega;
        }
        prev = ar;
        glow += 0.01 / (1.0 + ar * ar * 400.0);
        if (!fail && ar < HIT_PIX * u_pix * t) { hit = true; break; }
        t += step_len;
        if (t < tmin) { t = tmin; hit = true; break; }   // départ dans une forme
        if (t > tmax || haze(t) > 0.96) break;     // perdu dans la brume
    }
    float steps_frac = float(i) / float(MAX_STEPS);
    // pas épuisés (rayon rasant, détail lointain) : le pixel prend la
    // couleur de la brume, proche de celle des formes lointaines — le
    // traiter comme touché donnait des stries

    vec3 horizon = sky(rd);
    vec3 col = horizon;
    // pas épuisés tout près d'une surface : elle est touchée
    if (!hit && i >= MAX_STEPS && prev < u_pix * t * 12.0)
        hit = true;
    if (!hit && i >= MAX_STEPS && t < tmax) {
        // pas épuisés (rayon rasant) : une teinte moyenne des formes, fondue
        // dans la brume à cette distance — ni strie ni trou noir
        vec3 avg = hue_rotate(palette(0.5), u_hue_rot) * u_tint * (u_sky_col + 0.3 * u_sun_col);
        col = mix(avg * 0.5, horizon, haze(t));
    }
    if (hit) {
        col = shade(ro + rd * t, rd, t, u_comp_horizon, steps_frac);
        // pas de coupe nette à l'horizon
        float f = max(haze(t), smoothstep(0.75 * u_comp_horizon, u_comp_horizon, t));
        col = mix(col, horizon, f);
    }
    col += normalize(u_emit_col + 1e-4) * glow * u_mat_glow;

    vec2 dc = (v_uv - u_center_uv) * vec2(u_aspect, 1.0);
    float halo = exp(-dot(dc, dc) / 0.12);
    col *= max(1.0 + min(u_comp_halo, 0.0) * halo, 0.0);
    col += max(u_comp_halo, 0.0) * halo * (horizon + u_sky_col * 0.1 + 0.02);
    // alpha = profondeur de visée de la forme touchée (px), pour que le
    // composite sache si elle passe devant le modèle
    float depth = hit ? t * u_unit * view : NO_HIT;
    f_color = vec4(max(col, 0.0) * u_comp_intensity, depth);
}
