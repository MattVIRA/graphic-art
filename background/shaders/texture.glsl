// Texture du relief (T29) — réglages bg_tex_* (background/schema.py).
// Une décoration peinte sur le relief, fixe : calculée dans shade()
// (main.glsl) une fois par pixel, après la marche, jamais dans map() — la
// forme du terrain ne change pas. Elle est lue au point déformé du terrain
// (comme la grille) : elle suit la torsion, les ondes, le warp, et reste
// collée au décor pendant le voyage. Aucune horloge, aucune musique.
//
// Chaîne : point du relief → Flux (le relief déforme le motif) → MOTIF
// (Volutes, Vermicelles, Coulures, Gouttes, Cellules, Strates, Isolignes,
// Veines) → masque des formes + coordonnée de palette → STYLE (main.glsl).

// définie plus loin (main.glsl)
float spread01(float n);

// ce que le relief dit d'un point de la surface
struct Relief {
    vec2 xz;       // position dans le plan du terrain (unités des formes)
    float h;       // hauteur de la carte sans le détail (≈ −1…1)
    float alt;     // altitude 0–1
    vec2 g;        // pente de la carte (hauteur de la carte par unité)
    float slope;   // pente réelle 0–1 (0 = plat)
    float creux;   // 0–1 dans les creux
    float crest;   // 0–1 sur les bosses et les crêtes
    float foot;    // taille d'un pixel à cet endroit (unités)
};

Relief relief_at(vec3 ps) {
    float lip = 1.0, clear;
    vec3 q = distort(ps, lip, clear);
    Relief r;
    r.xz = q.xz;
    r.foot = pixel_at(q);
    vec4 m = terrain_map(q.xz, r.foot);
    // pente des grandes formes : différences sur la carte sans le détail
    float d = max(u_ld_scale * 0.02, r.foot * 2.0);
    float hx = terrain_map(q.xz + vec2(d, 0.0), r.foot).z;
    float hz = terrain_map(q.xz + vec2(0.0, d), r.foot).z;
    r.h = m.z;
    r.alt = clamp(0.5 + 0.5 * m.z, 0.0, 1.0);
    r.g = vec2(hx - m.z, hz - m.z) / d;
    r.slope = 1.0 - exp(-1.5 * relief_height() * length(r.g));
    // creux précalculés : ≈ ±1,5 sur les paysages (terrain.py)
    r.creux = smoothstep(0.0, 1.5, m.y);
    r.crest = smoothstep(0.0, 1.5, -m.y);
    return r;
}

// bruit lisse 0–1 dans le plan du motif (p en cellules du bruit) ; w
// choisit une tranche de la texture 3D, c un canal
float tnoise(vec2 p, float w, int c) {
    return spread01(texture(u_noise, vec3(p.x, w, p.y) / NOISE_CELLS)[c]);
}

// trois couches, tournées l'une par rapport à l'autre (pas d'alignement
// sur la grille de la texture)
float tfbm(vec2 p, float w) {
    const mat2 R = mat2(1.6, 1.2, -1.2, 1.6);
    float s = 0.5 * tnoise(p, w, 0);
    p = R * p;
    s += 0.3 * tnoise(p, w + 0.21, 1);
    p = R * p;
    s += 0.2 * tnoise(p, w + 0.43, 2);
    return s;
}

// aller-retour 0 → 1 → 0 (période 1) : une couleur sans saut
float mirror01(float x) {
    return 1.0 - abs(2.0 * fract(x) - 1.0);
}

// hachage sans sin() (Dave Hoskins) : sin() d'un grand nombre est imprécis
// sur GPU, les cellules loin de l'origine devenaient informes
vec2 thash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// cellules de Voronoi : (distance au point le plus proche, au deuxième,
// identifiant 0–1 de la cellule)
vec3 cells(vec2 x) {
    vec2 n = floor(x), f = fract(x);
    float d1 = 8.0, d2 = 8.0, id = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = thash(n + g);
            vec2 rr = g + o - f;
            float d = dot(rr, rr);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                id = o.x;
            } else if (d < d2) {
                d2 = d;
            }
        }
    return vec3(sqrt(d1), sqrt(d2), id);
}

// gouttes : un disque par cellule, de rayon tiré au hasard et plus grand en
// altitude ; (forme 0–1, identifiant). Le maximum des disques voisins, pas
// le point le plus proche : une goutte qui déborde n'est pas tranchée net
vec2 drops(vec2 x, float alt) {
    vec2 n = floor(x), f = fract(x);
    float best = 0.0, id = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = thash(n + g);
            float rr = mix(0.14, 0.3, fract(o.y * 7.13)) * mix(0.8, 1.2, alt);
            float v = clamp(1.0 - 0.5 * length(g + o - f) / rr, 0.0, 1.0);
            if (v > best) {
                best = v;
                id = o.x;
            }
        }
    return vec2(best, id);
}

// le champ du relief choisi par « Variation selon » (0,5 = neutre)
float vary_field(Relief r) {
    int k = int(u_tex_vary + 0.5);
    return k == 1 ? r.alt : k == 2 ? r.slope : k == 3 ? r.creux : k == 4 ? r.crest : 0.5;
}

// Flux : le relief déplace le motif (déplacement borné : aucune
// déchirure loin de l'origine). p en unités du motif ; gs : pente en
// tailles du monde
vec2 tex_flow(vec2 p, Relief r) {
    vec2 gs = r.g * u_ld_scale * 0.25;
    p += u_tex_flow_slope * gs;
    p += u_tex_flow_iso * vec2(-gs.y, gs.x);
    p -= u_tex_flow_bubble * gs * (r.alt - 0.5) * 2.0;
    if (u_tex_flow_swirl > 0.0) {
        vec2 sw = vec2(tnoise(p * 0.35, 0.11, 0), tnoise(p * 0.35, 0.67, 1)) - 0.5;
        p += u_tex_flow_swirl * 1.6 * sw;
    }
    return p;
}

// le motif au point du relief : (masque des formes 0–1, coordonnée de
// palette 0–1) ; xz décalé de dxz (différences de la gravure)
vec2 tex_motif(Relief r, vec2 dxz) {
    float scale = u_tex_scale;
    vec2 p = tex_flow((r.xz + dxz) / scale, r);
    // densité locale : le nombre de traits varie, pas la taille des formes
    float kd = u_tex_scale_alt * (r.alt - 0.5) * 2.0
             + u_tex_scale_slope * (r.slope - 0.5) * 2.0;
    if (u_tex_scale_rand > 0.0)
        kd += u_tex_scale_rand * 2.5 * (tnoise(r.xz / (scale * 6.0), 0.31, 2) - 0.5);
    float dens = u_tex_density * exp2(kd);
    // orientation : une rampe que les traits suivent (courbes de niveau,
    // ou direction fixe), ondulée par « Désordre »
    float a = radians(u_tex_angle);
    float ramp = int(u_tex_orient + 0.5) == 0
               ? r.h * relief_height() / scale
               : dot(p, vec2(cos(a), sin(a)));
    ramp += u_tex_disorder * 2.0 * (tnoise(p * 0.25, 0.83, 0) - 0.5);
    int motif = int(u_tex_motif + 0.5);
    float w = clamp(u_tex_width + u_tex_vary_amt * 0.35 * (vary_field(r) - 0.5) * 2.0,
                    0.02, 0.98);
    float soft = mix(0.35, 0.015, u_tex_sharp);
    // s : la forme (0–1, le motif est là où s dépasse 1 − épaisseur) ;
    // c : coordonnée de palette ; kf : traits par unité du motif ÷ densité
    // (l'écart réel entre deux traits, de 2 à 6 fois plus petit qu'une
    // forme selon le motif) ; px : écart entre deux traits en pixels
    float s, c;
    float kf = 1.0;
    float px = -1.0;   // calculé à la fin, sauf strates et isolignes
    // pente de la hauteur de la carte par unité (strates, isolignes)
    float gh = max(length(r.g), 1e-4);
    if (motif == 1) {
        // Volutes : marbrure, un bruit déformé par lui-même, bandes douces
        vec2 q = p + 1.2 * vec2(tfbm(p * 0.8, 0.05), tfbm(p * 0.8 + 5.2, 0.55));
        float f = 5.0 * tfbm(q * 1.3, 0.25) * dens + u_tex_stretch * ramp;
        s = 0.5 + 0.5 * sin(2.0 * PI * f);
        c = s;
        kf = 3.0;
    } else if (motif == 2) {
        // Vermicelles : traits de largeur égale le long des lignes d'un
        // bruit (aspect réaction-diffusion), coupés en gouttes là où un
        // second bruit l'emporte
        vec2 q = p + 0.5 * vec2(tfbm(p * 0.6, 0.15), tfbm(p * 0.6 + 3.7, 0.75));
        float f = 6.0 * tfbm(q, 0.35) * dens + u_tex_stretch * 0.5 * ramp;
        s = 1.0 - abs(sin(PI * 2.0 * f));
        s *= mix(1.0, smoothstep(0.15, 0.6, tnoise(q * 1.7, 0.95, 1)), 0.6);
        c = mirror01(f * 0.25);
        kf = 6.0;
    } else if (motif == 3) {
        // Coulures : un bruit moyenné le long de la pente (cinq points vers
        // le bas) → des traînées qui descendent ; sur le plat, la direction
        // fixe. Moyenne locale : aucune déchirure loin de l'origine
        vec2 fixed_dir = vec2(cos(a), sin(a));
        vec2 down = -r.g / gh;
        vec2 d = normalize(mix(fixed_dir, down, smoothstep(0.02, 0.15, gh * u_ld_scale)));
        // traînées : sept points espacés de L, un bruit trois fois plus fin
        // en travers → des coulures au moins cinq fois plus longues que larges
        float L = 0.6 / dens * (1.0 + 0.5 * u_tex_stretch);
        float acc = 0.0;
        for (int i = -3; i <= 3; i++)
            acc += tnoise((p + d * L * float(i)) * dens * 3.0, 0.45, 0);
        acc /= 7.0;
        // la moyenne resserre le bruit autour de 0,5 : on l'étale
        s = smoothstep(0.42, 0.58, acc);
        c = mirror01(acc * 1.5 + 0.3 * tnoise(p * 0.3, 0.52, 1));
        kf = 3.0;
    } else if (motif == 4 || motif == 5) {
        if (motif == 4) {
            // gouttes lues avant le flux : il les étirait (des formes molles)
            vec2 dr = drops(r.xz / scale * dens * 0.8, r.alt);
            s = dr.x;
            c = dr.y;
            kf = 1.6;
        } else {
            // Cellules : des membranes bombées, séparées par des parois
            vec3 v = cells(p * dens * 0.8);
            s = sqrt(smoothstep(0.0, 0.5, v.y - v.x));
            c = v.z;
            kf = 2.0;
        }
    } else if (motif == 6) {
        // Strates : des bandes qui suivent l'altitude, ondulées
        float k = dens * 4.0;
        float f = (0.5 + 0.5 * r.h) * k
                + (0.4 + 1.2 * u_tex_disorder) * (tfbm(p * 0.5, 0.6) - 0.5)
                + u_tex_stretch * 0.1 * ramp;
        s = 0.5 + 0.5 * cos(2.0 * PI * f);
        c = clamp(r.alt + 0.15 * (tfbm(p * 0.5, 0.6) - 0.5), 0.0, 1.0);
        px = 1.0 / max(k * 0.5 * gh * r.foot, 1e-6);
    } else if (motif == 7) {
        // Isolignes : des lignes sur les courbes de niveau, et sur les crêtes
        float k = dens * 6.0;
        float f = 0.5 * r.h * k + 0.15 * (tfbm(p * 0.7, 0.66) - 0.5);
        // lignes fines : l'épaisseur vaut un quart de celle des autres
        // motifs (0,5 → un huitième de l'écart entre deux lignes)
        float line = 1.0 - 2.0 * abs(fract(f) - 0.5);
        line = 1.0 - (1.0 - line) * 4.0;
        s = max(line, smoothstep(0.85, 0.97, r.crest));
        c = r.alt;
        px = 1.0 / max(k * 0.5 * gh * r.foot, 1e-6);
    } else {
        // Veines : deux réseaux de crêtes du bruit (fin et large), plus
        // marqués dans les creux
        vec2 q = p + 0.4 * vec2(tfbm(p * 0.5, 0.18), tfbm(p * 0.5 + 2.9, 0.88));
        float v1 = tfbm(q * dens * 0.8, 0.28);
        float v2 = tfbm(q * dens * 1.9 + 7.1, 0.48);
        float ridge = max(1.0 - smoothstep(0.0, 0.16, abs(v1 - 0.5)),
                          0.8 * (1.0 - smoothstep(0.0, 0.1, abs(v2 - 0.5))));
        s = ridge * mix(0.85, 1.25, r.creux);
        c = mirror01(v1 * 2.0);
        kf = 3.0;
    }
    float mask = smoothstep(1.0 - w - soft, 1.0 - w + soft, s);
    // traits serrés à moins de ~8 pixels : fondus dans leur moyenne (ni
    // moiré ni scintillement au loin)
    if (px < 0.0)
        px = scale / max(dens * kf * (1.0 + u_tex_stretch) * r.foot, 1e-6);
    float fade = smoothstep(3.0, 8.0, px);
    mask = mix(w, mask, fade);
    c = mix(0.5, c, fade);
    if (u_tex_invert > 0.5) {
        mask = 1.0 - mask;
        c = 1.0 - c;
    }
    return vec2(mask, c);
}

// coordonnée de palette : décalée, répétée « Fréquence » fois en
// aller-retour (aucun saut de couleur)
float tex_palette_x(float c, float mask) {
    float x = u_tex_shift + u_tex_freq * mix(0.15, 0.85, c) * mix(0.7, 1.0, mask);
    x = fract(x * 0.5) * 2.0;
    return x > 1.0 ? 2.0 - x : x;
}

// gravure : la normale suit la pente du masque (deux évaluations de plus,
// seulement si « Gravure » ≠ 0) ; effacée quand le motif est fin
vec3 tex_emboss(Relief r, vec3 n, float mask, float sharp) {
    if (abs(u_tex_emboss) < 0.001) return n;
    float e = max(u_tex_scale * 0.02, r.foot);
    float mx = tex_motif(r, vec2(e, 0.0)).x;
    float mz = tex_motif(r, vec2(0.0, e)).x;
    vec2 gm = vec2(mx - mask, mz - mask) / e;
    float k = u_tex_emboss * u_tex_scale * 0.02 * sharp
            * smoothstep(3.0, 10.0, u_tex_scale / max(u_tex_density * r.foot, 1e-6));
    vec3 ns = u_rot * n;
    vec3 gv = vec3(gm.x, 0.0, gm.y) * k;
    ns = normalize(ns - (gv - dot(gv, ns) * ns));
    return transpose(u_rot) * ns;
}

// reflets du studio, d'après le prototype (t29/modes_liquides.html) : un
// sol très sombre, une ligne d'horizon fine, un ciel moyen teinté par celui
// du fond, des bandes de lumière fixes dans le décor (le relief fait bouger
// les reflets) et un plafond lumineux ; une surface rugueuse les fond dans
// leur moyenne. Le ciel du fond, avec sa large lueur d'horizon, aplatissait
// le chrome
vec3 studio(vec3 r, float rough) {
    vec3 rs = u_rot * r;
    float y = rs.y;
    float peak = max(u_sun_col.r, max(u_sun_col.g, u_sun_col.b));
    vec3 light = mix(vec3(1.0), u_sun_col / max(peak, 1e-3), 0.25) * u_tex_strip_int;
    vec3 sky_tint = mix(vec3(0.6, 0.7, 1.0),
                        u_sky_col / max(max(u_sky_col.r, max(u_sky_col.g, u_sky_col.b)), 1e-3),
                        0.5);
    vec3 c = vec3(mix(0.01, 0.03, smoothstep(-0.4, 0.0, y)));
    c = mix(c, sky_tint * 0.35 * u_tex_strip_int, smoothstep(0.0, 0.6, y));
    float n = floor(u_tex_strips + 0.5);
    float wid = mix(0.7, 0.15, u_tex_strip_sharp);
    float band = 0.0;
    if (n > 0.5) {
        float a = atan(rs.x, rs.z) + radians(u_tex_strip_orient);
        band = smoothstep(1.0 - wid, 1.0 - 0.25 * wid, cos(a * n))
             * smoothstep(0.03, 0.2, y) * (1.0 - smoothstep(0.6, 0.85, y));
    }
    float top = smoothstep(0.9, 0.99, y);
    float hor = u_tex_horizon * exp(-abs(y - 0.02) * mix(15.0, 60.0, u_tex_strip_sharp));
    vec3 sharp_lit = light * (1.4 * band + 1.5 * top + hor);
    vec3 avg = light * ((n > 0.5 ? 0.35 * wid : 0.0) + 0.08 * u_tex_horizon + 0.05);
    return c + mix(sharp_lit, avg, clamp(rough * 2.5, 0.0, 1.0));
}

// teinte du métal : acier froid (−1) … or chaud (+1)
vec3 metal_tint() {
    float t = u_tex_metal * 0.5 + 0.5;
    return mix(vec3(0.62, 0.72, 0.9), vec3(1.0, 0.78, 0.5), t) * 0.9
         + vec3(0.08) * (1.0 - abs(u_tex_metal));
}
