// Chaîne d'étages du fond (T16, passe 2) : Distorsion → Espace → Formes →
// Fusion. Chaque étage choisit son opération par un uniform (tous les
// pixels prennent la même branche) : changer de mode ne recompile rien.
// Unités du fond : le modèle a un rayon de 1.
//
// Le décor est fixe ; l'objet y voyage le long d'une droite (son trajet),
// entourée d'un couloir vide de rayon « Vide central ». Règle commune à
// tous les Espaces : une forme n'est jamais tranchée. Chaque instance
// connaît son centre ; elle rapetisse jusqu'à disparaître quand sa portée
// atteindrait le couloir, et elle est réduite pour tenir dans sa cellule
// (la voisine ne la coupe pas). Les déformations s'éteignent près du
// couloir (sur une période), sauf la torsion autour du trajet, qui le
// laisse en place.
//
// Calculés en Python (background/params.py) : rotation de pliage KIFS,
// directions des satellites, portée des formes, étirement des ondes et du
// warp (u_dx_lip).

uniform mat3  u_kifs_rot;     // rotation appliquée à chaque pliage
uniform vec3  u_kifs_off;     // décalage de pliage × (échelle − 1)
uniform float u_kifs_inv;     // 1 / échelle^itérations
uniform vec3  u_sat[8];       // directions des satellites (orbite comprise)
uniform float u_reach;        // portée d'une instance (fraction de période)
uniform float u_reach_h;      // … à l'horizontale (Grille au sol)
uniform float u_reach_v;      // … à la verticale, EQ au maximum (Grille au sol)
uniform float u_dx_lip;       // étirement dû aux ondes et au warp
uniform float u_dx_phase;     // phase des ondes (rad)
uniform float u_sh_angle;     // rotation fixe + rotation lente des instances (rad)
uniform sampler3D u_noise;    // bruit lisse périodique (background/noise.py)

const float KIFS_SIZE = 2.0;  // = params.KIFS_SIZE
const float SINGLE_SIZE = 2.5;  // = params.SINGLE_SIZE : Objet seul, en périodes
const float NOISE_CELLS = 8.0;  // = noise.CELLS
const float FIT = 0.48;       // une instance tient dans 48 % de sa cellule
const float FAR = 1e5;

float void_radius() {
    return u_comp_void / u_scale;
}

// ── 1. Distorsion (avant la répétition : toute la grille se tord) ────
// rend le point déformé ; lip est multiplié par l'étirement local
vec3 distort(vec3 p, inout float lip) {
    if (u_dx_twist <= 0.0 && u_dx_bend == 0.0 && u_dx_wave_amp <= 0.0
        && u_dx_warp_amp <= 0.0)
        return p;
    vec3 pl = path_local(p);
    float P = u_sp_period;
    float rv = void_radius();
    float rc = length(pl.xy);
    float fade = smoothstep(rv, rv + P, rc);
    float dfade = (rc > rv && rc < rv + P) ? 1.5 / P : 0.0;   // pente de l'extinction
    // repère des déformations : a.z = axe (le trajet, ou la verticale)
    bool vertical = u_dx_axis > 0.5;
    vec3 a = vertical ? (p - u_path_o).xzy : pl;
    if (u_dx_twist > 0.0) {
        float w = 2.0 * PI * u_dx_twist;
        float r = length(a.xy);
        // autour du trajet, la torsion le laisse en place : pas d'extinction
        float f = vertical ? fade : 1.0;
        a.xy = rot2(w * a.z * f) * a.xy;
        lip *= sqrt(1.0 + w * w * r * r * f * f)
             * (vertical ? 1.0 + w * abs(a.z) * dfade * r : 1.0);
    }
    if (u_dx_bend != 0.0) {
        float x = clamp(a.x, -40.0, 40.0);
        a.y += u_dx_bend * x * x * fade;
        lip *= 1.0 + abs(u_dx_bend) * (2.0 * abs(x) * fade + x * x * dfade);
    }
    if (u_dx_wave_amp > 0.0) {
        float A = u_dx_wave_amp * P * fade;
        float k = 2.0 * PI / u_dx_wave_len;
        a.x += A * sin(k * a.z + u_dx_phase);
        a.y += A * sin(k * 0.73 * a.x + u_dx_phase * 1.3 + 1.7);
    }
    if (u_dx_warp_amp > 0.0) {
        vec3 uvw = a * u_dx_warp_freq / NOISE_CELLS;
        vec3 n = vec3(0.0);
        float amp = 1.0;
        for (int i = u_zero; i < 3; i++) {
            if (float(i) >= u_dx_warp_oct) break;
            n += (texture(u_noise, uvw).rgb - 0.5) * amp;
            uvw = uvw * 2.03 + 0.37;
            amp *= 0.5;
        }
        a += n * 2.0 * u_dx_warp_amp * fade;
    }
    lip *= u_dx_lip;
    return vertical ? u_path_o + a.xzy : u_path_o + transpose(u_path_m) * a;
}

// ── 2. Espace : de quelle instance le point fait partie ──────────────
struct Inst {
    vec3 q;        // point dans l'instance (en périodes : la cellule vaut 1)
    vec3 c;        // centre de l'instance (vague, couleur)
    float h;       // hasard de l'instance
    float k;       // échelle 0–1 (couloir, cellule, instance manquante)
    float unit;    // unités du fond par unité de q
    float bnd;     // borne : distance au bord de la cellule (la voisine diffère)
    float shell;   // borne valable pour toutes les instances (coquille autour
                   // du trajet qui les contient toutes), sinon -FAR
    float ystr;    // étirement vertical en plus (égaliseur)
    float floor_d; // distance au sol (Grille au sol), sinon FAR
};

// réduction d'une instance dont le centre est à `dist` du trajet : sa
// portée n'atteint jamais le couloir ; instance manquante → 0
float presence(float dist, float reach_world, float h) {
    if (h < u_sp_missing) return 0.0;
    float x = clamp((dist - void_radius()) / max(reach_world, 1e-4), 0.0, 1.0);
    return min(x, smoothstep(0.0, 1.0, x));
}

// marge entre le bord d'une cellule et la forme voisine (demi-cellule moins
// sa portée) : ajoutée à la borne de cellule, elle évite qu'un rayon qui
// longe un bord de cellule au loin s'y arrête comme sur une surface
float clearance(float half_cell, float reach_world) {
    return max(half_cell - reach_world, 0.02 * u_sp_period);
}

// les voisines sont identiques à cette instance (ni désordre, ni manquantes,
// ni vague, ni satellites asymétriques) et assez loin du couloir pour ne pas
// être réduites : la répétition simple suffit, sans borne de cellule (les
// rayons ne rampent plus le long des bords : moins de pas épuisés, moins
// de mouchetures)
bool same_neighbours(float dist_c, float reach_world) {
    return u_sp_disorder <= 0.0 && u_sp_missing <= 0.0 && u_sh_wave <= 0.0
        && u_fu_count < 0.5 && u_sp_eq <= 0.0
        && dist_c - void_radius() > reach_world + 1.8 * u_sp_period;
}

// la cellule fait `cell` périodes de large (au plus étroit)
float fit(float cell, float reach) {
    return min(1.0, FIT * cell / max(reach, 1e-4));
}

Inst sp_single(vec3 p) {        // Objet seul : une grande forme de loin en loin
    float U = SINGLE_SIZE * u_sp_period;
    float reach = u_reach * U;
    // l'écart entre deux formes et leur distance au trajet suivent leur
    // taille : elles ne sont jamais réduites, ni par la voisine ni par le couloir
    float L = max(2.0 * u_sp_along, 2.2 * reach);
    float D = void_radius() + reach + u_sp_dist;
    vec3 pl = path_local(p);
    float iz = round(pl.z / L);
    float h = hash12(vec2(iz, 3.0));
    float side = mod(iz, 2.0) < 0.5 ? 0.0 : PI;          // un côté puis l'autre
    float ang = side + (h - 0.5) * 1.4;
    vec3 c = vec3(D * cos(ang), D * sin(ang), iz * L);
    Inst I;
    I.q = (pl - c) / U;
    I.c = c;
    I.h = h;
    I.k = h < u_sp_missing ? 0.0 : 1.0;
    I.unit = U;
    I.bnd = L * 0.5 - abs(pl.z - iz * L) + clearance(L * 0.5, reach);
    I.ystr = 1.0;
    I.floor_d = FAR;
    // toutes les grandes formes sont centrées à la même distance du trajet
    I.shell = abs(length(pl.xy) - D) - reach;
    return I;
}

Inst sp_grid(vec3 p) {          // Grille 3D, alignée sur le décor
    float P = u_sp_period;
    vec3 id = round(p / P);
    vec3 q = p / P - id;
    float kf = fit(1.0, u_reach);
    Inst I;
    I.q = q;
    I.c = id * P;
    I.h = hash13(id + 17.0);
    float dist = path_dist(I.c);
    I.k = kf * presence(dist, u_reach * P * kf, I.h);
    I.unit = P;
    I.bnd = same_neighbours(dist, u_reach * P * kf) ? FAR
          : (0.5 - max(abs(q.x), max(abs(q.y), abs(q.z)))) * P
          + clearance(0.5 * P, u_reach * P * kf);
    I.ystr = 1.0;
    I.floor_d = FAR;
    I.shell = -FAR;
    return I;
}

Inst sp_ground(vec3 p) {        // Grille au sol : formes posées sur un sol fixe
    float P = u_sp_period;
    float floor_y = u_path_o.y - 2.2 / u_scale;        // sous le trajet
    vec2 id = round(p.xz / P);
    vec2 q = p.xz / P - id;
    float h = hash12(id + 5.0);
    // une bande EQ par rangée : l'égaliseur se lit de gauche à droite
    float lvl = u_eq[int(mod(id.x + 700.0, 7.0))];
    float eq = 1.0 + 3.0 * u_sp_eq * lvl;
    float kf = fit(1.0, u_reach_h);
    vec3 c = vec3(id.x * P, u_path_o.y, id.y * P);     // distance au trajet à l'horizontale
    Inst I;
    I.k = kf * presence(path_dist(c), u_reach_h * P * kf, h);
    // la forme repose sur le sol : levée de sa demi-hauteur
    float lift = u_reach_v * eq / (1.0 + 3.0 * u_sp_eq) * I.k;
    I.q = vec3(q.x, (p.y - floor_y) / P - lift, q.y);
    I.c = vec3(c.x, floor_y, c.z);
    I.h = h;
    I.unit = P;
    I.bnd = (0.5 - max(abs(q.x), abs(q.y))) * P + clearance(0.5 * P, u_reach_h * P * kf);
    I.ystr = eq;
    I.shell = -FAR;
    // sol, et plafond : au-dessus de la plus haute forme possible, les
    // rayons du ciel ne rampent pas de cellule en cellule
    I.floor_d = p.y - floor_y;
    I.bnd = max(I.bnd, p.y - floor_y - 2.0 * u_reach_v * P - 0.1);
    return I;
}

Inst sp_rings(vec3 p) {         // Anneaux le long du trajet
    float P = u_sp_period;
    vec3 pl = path_local(p);
    float L = u_sp_along;
    float iz = round(pl.z / L);
    float zq = pl.z - iz * L;
    float N = max(u_sp_count, 3.0);
    float sec = 2.0 * PI / N;
    float kf = fit(L / P, u_reach);
    // l'anneau longe le couloir sans y entrer
    float R = void_radius() + u_reach * P * kf + 0.2;
    kf = min(kf, fit(2.0 * R * sin(sec * 0.5) / P, u_reach));
    float r = length(pl.xy);
    float a = atan(pl.y, pl.x);
    float ia = floor(a / sec + 0.5);
    float al = a - ia * sec;
    Inst I;
    I.q = vec3(r * cos(al) - R, r * sin(al), zq) / P;   // x vers l'extérieur, z le long
    I.c = vec3(R * cos(ia * sec), R * sin(ia * sec), iz * L);
    I.h = hash12(vec2(ia, iz) + 11.0);
    I.k = kf * (I.h < u_sp_missing ? 0.0 : 1.0);
    I.unit = P;
    I.bnd = same_neighbours(FAR, 0.0) ? FAR
          : min(L * 0.5 - abs(zq), r * sin(max(sec * 0.5 - abs(al), 0.0)))
          + clearance(min(L * 0.5, R * sin(sec * 0.5)), u_reach * P * kf);
    I.ystr = 1.0;
    I.floor_d = FAR;
    // les anneaux tiennent dans un tube autour du couloir : au-dehors (le
    // long du couloir, ou au loin), le rayon avance sans buter sur les cellules
    I.shell = abs(r - R) - u_reach * P * kf;
    return I;
}

Inst sp_mirror(vec3 p) {        // Miroirs : grille du repère du trajet, repliée
    float P = u_sp_period;
    vec3 m = path_local(p);
    // les plans de miroir passent par le trajet (ou lui sont perpendiculaires) :
    // la distance au trajet ne change pas, et ce sont des bords de cellule
    if (u_sp_mirror_x > 0.5) m.x = abs(m.x);
    if (u_sp_mirror_y > 0.5) m.y = abs(m.y);
    if (u_sp_mirror_z > 0.5) {
        float Lz = P * max(1.0, round(u_sp_along / P));
        m.z = Lz - abs(mod(m.z, 2.0 * Lz) - Lz);
    }
    vec3 id = floor(m / P);
    vec3 q = m / P - id - 0.5;
    vec3 c = (id + 0.5) * P;
    float kf = fit(1.0, u_reach);
    Inst I;
    I.q = q;
    I.c = c;
    I.h = hash13(id + 23.0);
    I.k = kf * presence(length(c.xy), u_reach * P * kf, I.h);
    I.unit = P;
    I.bnd = same_neighbours(length(c.xy), u_reach * P * kf) ? FAR
          : (0.5 - max(abs(q.x), max(abs(q.y), abs(q.z)))) * P
          + clearance(0.5 * P, u_reach * P * kf);
    I.ystr = 1.0;
    I.floor_d = FAR;
    I.shell = -FAR;
    return I;
}

Inst sp_kifs(vec3 p) {          // Pliage KIFS centré sur le trajet
    float P = u_sp_period;
    vec3 pl = path_local(p) / P;
    float H = u_sp_along / (2.0 * P);
    pl.z = abs(mod(pl.z + H, 4.0 * H) - 2.0 * H) - H;  // répété en miroir le long du trajet
    vec3 q = pl;
    // T = transposée de l'orientation accumulée : le centre réel de la copie
    // se retrouve sans inverser la chaîne de pliages
    mat3 T = mat3(1.0);
    mat3 RT = transpose(u_kifs_rot);
    // distance aux plans de pliage (périodes) : tant que le rayon ne les
    // franchit pas, il reste dans la même copie (borne de la « cellule »)
    float planes = H - abs(pl.z), scale = 1.0;
    for (int i = u_zero; i < 8; i++) {
        if (float(i) >= u_sp_iter) break;
        planes = min(planes, min(abs(q.x), min(abs(q.y), abs(q.z))) * scale);
        scale /= u_sp_fscale;
        vec3 s = step(0.0, q) * 2.0 - 1.0;
        q = abs(q);
        T[0] *= s.x;
        T[1] *= s.y;
        T[2] *= s.z;
        T = T * RT;
        q = u_sp_fscale * (u_kifs_rot * q) - u_kifs_off;
    }
    vec3 c = pl - T * q * u_kifs_inv;                  // centre de la copie (périodes)
    float unit = P * u_kifs_inv * KIFS_SIZE;
    Inst I;
    I.q = q / KIFS_SIZE;
    I.c = c * P;
    I.h = hash13(floor(c * 8.0) + 31.0);
    I.k = presence(length(c.xy) * P, u_reach * unit, I.h);
    I.unit = unit;
    // copie réduite (près du couloir) : ses voisines peuvent être plus
    // proches qu'elle, le pas s'arrête aux plans de pliage
    I.bnd = I.k < 0.99 ? planes * P + 0.01 * unit : FAR;
    I.ystr = 1.0;
    I.floor_d = FAR;
    // par symétrie des pliages, la copie entière est aussi la plus proche
    // des copies entières : sa sphère borne la distance à toutes les copies
    // (réduites ou non), et empêche de ramper le long des plans de pliage
    I.shell = (length(I.q) - u_reach) * unit;
    return I;
}

Inst space(vec3 p) {
    int m = int(u_sp_mode + 0.5);
    if (m == 0) return sp_single(p);
    if (m == 1) return sp_grid(p);
    if (m == 2) return sp_ground(p);
    if (m == 3) return sp_rings(p);
    if (m == 4) return sp_mirror(p);
    return sp_kifs(p);
}

// ── 3. Formes : primitive A ↔ B dans l'instance ──────────────────────
float sd_box_frame(vec3 p, vec3 b, float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(min(
        length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

// primitive de « rayon » r (portée : params.PRIM_REACH), barres d'épaisseur th
float prim(int i, vec3 q, float r, float th) {
    if (i == 0) return length(q) - r;
    if (i == 1) return sd_box(q, vec3(r * 0.8));
    if (i == 2) return sd_box(q, vec3(r * 0.6)) - r * 0.2;
    if (i == 3) return sd_box_frame(q, vec3(r * 0.8), th);
    if (i == 4) return sd_torus(q, r * 0.7, r * 0.28);
    if (i == 5) return (abs(q.x) + abs(q.y) + abs(q.z) - r) * 0.57735;
    if (i == 6) return sd_capsule(q, vec3(0.0, -r * 0.6, 0.0), vec3(0.0, r * 0.6, 0.0), r * 0.4);
    if (i == 7) {
        vec2 d = abs(vec2(length(q.xz), q.y)) - vec2(r * 0.6, r * 0.8);
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    }
    if (i == 8) {
        vec3 a = abs(q);
        return min(min(sd_box(a, vec3(r, th, th)), sd_box(a, vec3(th, r, th))),
                   sd_box(a, vec3(th, th, r)));
    }
    // gyroïde : coque d'environ trois cellules de large, dans une sphère
    float f = 3.0 * PI / r;
    float g = abs(dot(sin(q * f), cos(q.yzx * f))) / (f * 1.8) - th * 0.5;
    return max(g, length(q) - r);
}

float smax_poly(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

// (distance en périodes, structure 0–1, 1 si un satellite est le plus proche)
vec3 shape(Inst I) {
    float h1 = fract(I.h * 13.7), h2 = fract(I.h * 7.31), h3 = fract(I.h * 3.17);
    vec3 q = I.q;
    float ay = u_sh_angle + u_sp_disorder * (h1 - 0.5) * 2.0 * PI;
    q.xz = rot2(ay) * q.xz;
    if (u_sp_disorder > 0.0)
        q.yz = rot2(u_sp_disorder * (h2 - 0.5) * PI) * q.yz;
    float wave = 1.0 + u_sh_wave * 0.5
               * max(0.0, sin(2.0 * PI * (u_beat * 0.25 - dot(I.c, vec3(0.031, 0.017, 0.023)))));
    float r = u_sh_size * (1.0 + 0.4 * u_sp_disorder * (2.0 * h3 - 1.0)) * wave * I.k;
    float th = u_sh_thick * I.k;
    vec3 st = vec3(u_sh_sx, u_sh_sy * I.ystr, u_sh_sz);
    vec3 qs = q / st;
    float d = u_sh_morph < 0.999 ? prim(int(u_sh_a + 0.5), qs, r, th) : 0.0;
    if (u_sh_morph > 0.001)
        d = mix(d, prim(int(u_sh_b + 0.5), qs, r, th), u_sh_morph);
    d *= min(st.x, min(st.y, st.z));
    d -= u_sh_round * r;
    if (u_sh_shell > 0.0)
        d = abs(d) - u_sh_shell * r * 0.5;
    float sat = 0.0;
    // ── 4. Fusion : satellites autour de la forme ──
    if (u_fu_count > 0.5) {
        float kv = max(u_fu_k * r * 0.5, 1e-4);
        // smin exponentiel sur tous les satellites, calculé en ligne
        // (log-somme-exp stable : aucun ordre, aucun débordement)
        float m = FAR, s = 0.0, ds_min = FAR;
        for (int i = u_zero; i < 8; i++) {
            if (float(i) >= u_fu_count) break;
            float ds = length(q - u_sat[i] * u_fu_spread * r) - u_fu_rad * r;
            ds_min = min(ds_min, ds);
            if (ds < m) { s = s * exp((ds - m) / kv) + 1.0; m = ds; }
            else s += exp((m - ds) / kv);
        }
        int combo = int(u_fu_combo + 0.5);
        if (combo == 0) {
            sat = ds_min < d ? 1.0 : 0.0;
            if (d < m) { s = s * exp((d - m) / kv) + 1.0; m = d; }
            else s += exp((m - d) / kv);
            d = m - kv * log(s);
        } else {
            float dsat = m - kv * log(s);
            d = combo == 1 ? smax_poly(d, -dsat, kv) : smax_poly(d, dsat, kv);
        }
    }
    return vec3(d, clamp(length(q) / max(u_reach * I.k, 1e-3), 0.0, 1.0), sat);
}

// (distance, structure, hasard de l'instance, émission)
vec4 chain(vec3 p) {
    float lip = 1.0;
    Inst I = space(distort(p, lip));
    float d = FAR;
    vec3 sh = vec3(0.0);
    if (I.k > 0.01) {
        // hors de la sphère qui contient l'instance : pas besoin de la forme
        float far = (length(I.q) - u_reach * I.k) * I.unit;
        if (far > 0.1 * I.unit) {
            d = far;
        } else {
            sh = shape(I);
            d = sh.x * I.unit;
        }
    }
    d = max(min(d, I.bnd), I.shell);
    if (int(u_sp_mode + 0.5) == 5)
        lip *= 1.25;            // copies de tailles différentes près du couloir
    d /= lip;
    // le couloir est vide : borne qui le fait traverser en un pas
    d = max(d, void_radius() - path_dist(p));
    float em = 0.0;
    if (I.floor_d < d) {        // le sol (Grille au sol) n'est pas dans le couloir
        return vec4(I.floor_d, 0.0, 0.0, 0.0);
    }
    float on = mod(floor(I.h * 4.0) + floor(u_beat), 4.0) < 0.5 ? 2.0 : 0.0;
    em = sh.z * u_fu_glow + u_mat_chase * on;
    return vec4(d, sh.y, I.h, em);
}

float map(vec3 p) {
    return chain(p).x;
}

// (structure, hasard de l'instance, émission)
vec3 material(vec3 p) {
    return chain(p).yzw;
}
