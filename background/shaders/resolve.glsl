// Fond SDF (T16, passe 5) — résolution : accumulation temporelle et
// agrandissement en une passe, à 1,5 × la résolution du fond (au plus la
// résolution interne ; le composite fait le reste en bilinéaire).
//
// Le fond (main.glsl) est calculé à une fraction de la résolution, un rayon
// par pixel, décalé d'une fraction de pixel à chaque frame. Ici, pour chaque
// pixel de l'écran :
// 1. les 3×3 échantillons voisins de la frame, pondérés par leur distance au
//    centre du pixel et par leur ressemblance de profondeur avec le plus
//    proche (un contour ne bave pas d'un plan sur l'autre) ;
// 2. l'historique reprojeté : le point vu ici est ramené là où il était la
//    frame d'avant (le décor est fixe dans l'espace des formes, l'objet y
//    voyage, la caméra bouge), puis rejeté si sa surface n'existe plus dans
//    le voisinage (profondeur), écrêté dans l'étendue des couleurs voisines
//    s'il a changé (formes animées, flux, flash) ;
// 3. le mélange, pondéré par un poids accumulé par pixel (borné par
//    « Lissage temporel ») : un pixel découvert converge en quelques frames.
// Mêmes formules et constantes que background/temporal.py (testé).

in vec2 v_uv;
layout(location = 0) out vec4 o_col;   // rgb = fond, a = part devant le modèle
layout(location = 1) out vec4 o_aux;   // r = poids de l'historique, g = profondeur

uniform sampler2D u_cur;          // fond de la frame (a = profondeur, NO_HIT, MASKED)
uniform sampler2D u_hist;         // historique o_col (filtrage linéaire)
uniform sampler2D u_hist_aux;     // historique o_aux
uniform sampler2D u_model_depth;  // profondeur du modèle (0–1, pleine résolution)
uniform vec2 u_near_far;
uniform vec2 u_cur_size;          // taille du fond (px)
uniform vec2 u_out_size;          // taille de sortie (px)
uniform vec2 u_jitter;            // décalage des rayons de la frame (px du fond)
uniform float u_wmax;             // poids maximal de l'historique (0 = aucun)
uniform float u_reset;            // 1 = historique jeté
uniform float u_same_space;       // 1 = une forme peut passer devant le modèle

// pose de cette frame (c) et de la précédente (p), voir temporal.Pose
uniform mat4 u_c_inv_vp;
uniform vec3 u_c_eye, u_c_fwd;
uniform float u_c_unit, u_c_far;
uniform mat4 u_p_vp;
uniform vec3 u_p_eye, u_p_fwd;
uniform float u_p_unit, u_p_far;
uniform mat3 u_carry_m;           // point de cette frame → même point du décor avant
uniform vec3 u_carry_b;

const float NO_HIT = 60000.0;
const float SIGMA_N = 0.3, SIGMA_W = 0.6;    // pixels du fond
const float DEPTH_LO = 0.03, DEPTH_HI = 0.08;
const float CLIP_K = 2.0;
const float GAMMA = 1.25;                    // largeur de la boîte de couleurs (écarts types)

bool is_hit(float d) { return d >= 0.0 && d < NO_HIT * 0.5; }
bool is_miss(float d) { return d >= NO_HIT * 0.5; }

float depth_conf(float e, float s) {
    if (is_miss(e) && is_miss(s)) return 1.0;
    if (!(is_hit(e) && is_hit(s))) return 0.0;
    float rel = abs(e - s) / max(max(e, s), 1e-6);
    return 1.0 - smoothstep(DEPTH_LO, DEPTH_HI, rel);
}

// ressemblance de profondeur entre deux échantillons de la frame (plus
// tolérante que le rejet : la surface est la même, vue d'un rayon voisin)
float similar(float a, float b) {
    if (is_miss(a) || is_miss(b)) return is_miss(a) == is_miss(b) ? 1.0 : 0.0;
    float rel = abs(a - b) / max(max(a, b), 1e-6);
    return 1.0 - smoothstep(0.05, 0.2, rel);
}

float eye_depth(float d) {
    float n = u_near_far.x, f = u_near_far.y;
    return 2.0 * n * f / (f + n - (d * 2.0 - 1.0) * (f - n));
}

vec3 ycocg(vec3 c) {
    return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
                0.5 * c.r - 0.5 * c.b,
                -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

vec3 rgb(vec3 y) {
    return vec3(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z);
}

// historique lu en Catmull-Rom (5 lectures bilinéaires) : la lecture
// bilinéaire, répétée frame après frame, finirait par tout flouter
vec3 catmull_rom(vec2 uv) {
    vec2 pos = uv * u_out_size;
    vec2 tc = floor(pos - 0.5) + 0.5;
    vec2 f = pos - tc;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = (tc - 1.0) / u_out_size, t3 = (tc + 2.0) / u_out_size;
    vec2 t12 = (tc + w2 / w12) / u_out_size;
    float a = w12.x * w0.y, b = w0.x * w12.y, c = w12.x * w12.y,
          d = w3.x * w12.y, e = w12.x * w3.y;
    vec3 r = texture(u_hist, vec2(t12.x, t0.y)).rgb * a
           + texture(u_hist, vec2(t0.x, t12.y)).rgb * b
           + texture(u_hist, t12).rgb * c
           + texture(u_hist, vec2(t3.x, t12.y)).rgb * d
           + texture(u_hist, vec2(t12.x, t3.y)).rgb * e;
    // le rebond du Catmull-Rom près d'un contour est ensuite écrêté par la
    // boîte de couleurs du voisinage
    return max(r / (a + b + c + d + e), 0.0);
}

// écrêtage vers le centre de la boîte (pas composante par composante :
// la teinte de l'historique est gardée)
vec3 clip_box(vec3 lo, vec3 hi, vec3 h) {
    vec3 c = 0.5 * (hi + lo), e = 0.5 * (hi - lo) + 1e-4;
    vec3 v = h - c, u = abs(v / e);
    float m = max(u.x, max(u.y, u.z));
    return m > 1.0 ? c + v / m : h;
}

// part de l'échantillon devant le modèle (même règle que le composite
// d'avant la passe 5 : 1 % de profondeur de transition)
float in_front(float d, float zm) {
    return is_hit(d) ? clamp((zm - d) / (zm * 0.01), 0.0, 1.0) : 0.0;
}

void main() {
    vec2 x = v_uv * u_cur_size;              // position en pixels du fond
    ivec2 c0 = ivec2(floor(x));
    ivec2 hi = ivec2(u_cur_size) - 1;
    float md = texture(u_model_depth, v_uv).r;
    bool model = u_same_space > 0.5 && md < 1.0;
    float zm = model ? eye_depth(md) : 0.0;

    vec4 S[9];
    float D2[9];
    float best = 1e9, d_ref = NO_HIT, d_mv = NO_HIT;
    bool any_valid = false;
    for (int j = 0; j < 9; j++) {
        ivec2 q = clamp(c0 + ivec2(j % 3 - 1, j / 3 - 1), ivec2(0), hi);
        S[j] = texelFetch(u_cur, q, 0);
        vec2 dd = vec2(q) + 0.5 + u_jitter - x;
        D2[j] = dot(dd, dd);
        if (S[j].a >= 0.0 && D2[j] < best) {      // le plus proche, hors masqués
            best = D2[j];
            d_ref = S[j].a;
            any_valid = true;
        }
        if (is_hit(S[j].a))
            d_mv = min(d_mv, S[j].a);
    }

    vec3 sum_n = vec3(0.0), sum_w = vec3(0.0), m1 = vec3(0.0), m2 = vec3(0.0);
    vec3 sum_m = vec3(0.0);
    float wn = 0.0, ww = 0.0, wm = 0.0, wst = 0.0;
    float fn = 0.0, fnw = 0.0, fw = 0.0, fww = 0.0, fmin = 1.0, fmax = 0.0;
    for (int j = 0; j < 9; j++) {
        float gn = exp(-D2[j] / (2.0 * SIGMA_N * SIGMA_N));
        float gw = exp(-D2[j] / (2.0 * SIGMA_W * SIGMA_W));
        float f = model ? in_front(S[j].a, zm) : 0.0;
        fn += f * gn; fnw += gn;
        fw += f * gw; fww += gw;
        fmin = min(fmin, f); fmax = max(fmax, f);
        if (S[j].a < 0.0) {                       // masqué par le modèle
            sum_m += S[j].rgb * gw; wm += gw;
            continue;
        }
        float s = similar(S[j].a, d_ref);
        sum_n += S[j].rgb * gn * s; wn += gn * s;
        sum_w += S[j].rgb * gw * s; ww += gw * s;
        vec3 y = ycocg(S[j].rgb);
        m1 += y * gw; m2 += y * y * gw; wst += gw;
    }
    float front_n = fn / max(fnw, 1e-6), front_w = fw / max(fww, 1e-6);

    if (!any_valid) {
        // tout est derrière le modèle : la couleur sert seulement sous un
        // modèle transparent, rien n'est gardé pour la suite
        o_col = vec4(sum_m / max(wm, 1e-6), front_w);
        o_aux = vec4(0.0, -1.0, 0.0, 0.0);
        return;
    }
    vec3 c_w = sum_w / max(ww, 1e-6);
    vec3 c_n = wn > 1e-4 ? sum_n / wn : c_w;
    float k = min(wn, 1.0);

    // ── historique reprojeté ──
    float w_h = 0.0;
    vec3 h = vec3(0.0);
    float front_h = 0.0;
    if (u_wmax > 0.0 && u_reset < 0.5) {
        vec4 w = u_c_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
        vec3 rd = normalize(w.xyz / w.w / u_c_unit - u_c_eye / u_c_unit);
        vec4 cl;
        float pd;
        // reprojeté avec la surface la plus proche du voisinage : au bord
        // d'une forme, le pixel suit la forme et pas le ciel (sinon un
        // liseré traîne du côté où elle s'éloigne)
        if (is_miss(d_mv)) {
            cl = u_p_vp * vec4(rd, 0.0);          // à l'infini : la caméra seule
            pd = NO_HIT;
        } else {
            vec3 ro = u_c_eye / u_c_unit * (1.0 - u_c_far);
            float view = max(dot(rd, u_c_fwd), 1e-3);
            vec3 pp = u_carry_m * (ro + rd * d_mv / (view * u_c_unit)) + u_carry_b;
            vec3 rel = pp - u_p_eye / u_p_unit * (1.0 - u_p_far);
            cl = u_p_vp * vec4((u_p_eye / u_p_unit + rel) * u_p_unit, 1.0);
            pd = dot(rel, u_p_fwd) * u_p_unit;
        }
        vec2 puv = cl.xy / cl.w * 0.5 + 0.5;
        if (cl.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0)))
                && all(lessThan(puv, vec2(1.0)))) {
            ivec2 pq = ivec2(puv * u_out_size);
            vec4 aux = texelFetch(u_hist_aux, pq, 0);
            // la surface que l'historique a vue existe-t-elle encore dans le
            // voisinage ? (comparer au seul échantillon le plus proche
            // rejetterait tous les contours : il change de côté à chaque
            // décalage) ; sa profondeur suit celle du point reprojeté
            float dh = aux.g;
            if (is_hit(dh) && is_hit(pd))
                dh += d_mv - pd;
            float conf = 0.0;
            for (int j = 0; j < 9; j++)
                if (S[j].a >= 0.0)
                    conf = max(conf, depth_conf(S[j].a, dh));
            w_h = min(aux.r, u_wmax) * conf;
            if (w_h > 0.0) {
                h = catmull_rom(puv);
                front_h = clamp(texelFetch(u_hist, pq, 0).a, fmin, fmax);
                // couleur : écrêtée dans l'étendue des voisins de la frame
                vec3 mu = m1 / wst;
                vec3 sd = sqrt(max(m2 / wst - mu * mu, 0.0));
                vec3 yc = ycocg(c_n);
                vec3 lo = min(mu - GAMMA * sd, yc), hi3 = max(mu + GAMMA * sd, yc);
                vec3 yh = ycocg(h);
                vec3 yk = clip_box(lo, hi3, yh);
                float dist = length(yh - yk) / (length(sd) + 1e-3);
                w_h *= 1.0 / (1.0 + CLIP_K * dist);
                h = max(rgb(yk), 0.0);
            }
        }
    }

    vec3 col;
    float front, w_new;
    if (w_h <= 0.0) {
        // sans historique : l'agrandissement seul (noyau large, guidé par la
        // profondeur)
        col = c_w;
        front = front_w;
        w_new = min(k, u_wmax);
    } else {
        col = (h * w_h + c_n * k) / (w_h + k);
        front = (front_h * w_h + front_n * k) / (w_h + k);
        w_new = min(w_h + k, u_wmax);
    }
    o_col = vec4(col, front);
    o_aux = vec4(w_new, d_ref, 0.0, 0.0);
}
