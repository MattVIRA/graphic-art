"""
GPU renderer (moderngl).

Pipeline per frame (all HDR, linear light):
  1. Scene: geometry (surface, lines, grid) into a 4× MSAA buffer, resolved;
     soft additive points drawn after the resolve (they don't need MSAA)
  2. Trails: scene merged into a persistent layer (max of decayed past and
     present; opaque surfaces win) → motion trails without ghosting
  3. Bloom: thresholded 13-tap downsample chain + tent upsample (6 levels)
  4. Composite: AgX filmic tone map, background, grading, vignette, dither
  5. (optional) HUD text overlay quad

GLFW owns the window + GL context (main.py); we attach moderngl to it via
moderngl.create_context(). Every size-dependent buffer is allocated at the
framebuffer resolution (logical size × pixel ratio → native Retina).
"""

import colorsys

import numpy as np
import moderngl

from gfx import matcaps


# ── Matrix helpers (row-major; act as M @ column_vector) ──────────────────────

def _perspective(fovy_rad, aspect, near, far):
    f = 1.0 / np.tan(fovy_rad / 2.0)
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def _translate(tx, ty, tz):
    M = np.eye(4, dtype=np.float64)
    M[0, 3], M[1, 3], M[2, 3] = tx, ty, tz
    return M


def _rotate_y(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 2] = c, s
    M[2, 0], M[2, 2] = -s, c
    return M


def _rotate_x(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=np.float64)
    M[1, 1], M[1, 2] = c, -s
    M[2, 1], M[2, 2] = s, c
    return M


def _scale(s):
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = M[1, 1] = M[2, 2] = s
    return M


def _rotate_axis(axis, a):
    """Rodrigues rotation about an arbitrary unit axis."""
    x, y, z = axis
    c, s = np.cos(a), np.sin(a)
    C = 1.0 - c
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = [
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]
    return M


# ── Shaders ───────────────────────────────────────────────────────────────────

EMISSION_GAIN = 3.0   # additive passes (points, lines) — see GLRenderer.__init__

# Scene colors are computed in display (sRGB-like) space, as the sliders and
# matcaps were designed, then converted to LINEAR light at the output of every
# geometry shader: additive blending, bloom and the filmic tone map all work on
# linear values. Alpha carries opacity, not glow: emissive passes (points,
# lines) write 0, the opaque surface writes 1 (it hides the trails and the
# background behind it).
_LINEAR_GLSL = """
vec3 to_linear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
"""

POINT_FS = """
#version 330
uniform float u_emit;   // emission gain of the additive passes
in vec3 v_color;
out vec4 f_color;
""" + _LINEAR_GLSL + """
void main() {
    // soft round falloff (gaussian-ish) → glowing dot
    vec2 c = gl_PointCoord - vec2(0.5);
    float d2 = dot(c, c);
    float a = exp(-d2 * 14.0);
    f_color = vec4(to_linear(v_color) * a * u_emit, 0.0);   // additive emission
}
"""

# ── Stateless GPU point physics (T1) ─────────────────────────────────────────
# Static per-point attributes are uploaded ONCE; every frame only a handful
# of audio-feature uniforms change and the vertex shader recomputes the
# displacement, color and size. No CPU work, no per-frame upload.
#
# Attribute layout (10 floats, see visuals/mesh.build_points):
#   in_pos   vec3 — rest position on the model surface
#   in_color vec3 — base color (texture / vertex colors)
#   in_aux   vec4 — normal.xyz, mel band

# GLSL chunks shared by the point shader and the mesh shaders (T19): pasted
# in place of the //%NOISE% and //%MODEL% markers.
_NOISE_GLSL = """
float hash11(float p) {
    p = fract(p * 0.1031); p *= p + 33.33; return fract(p * (p + p));
}

// 3D value noise (for the organic smoke-like drift)
float vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = dot(i, vec3(1.0, 57.0, 113.0));
    float a = hash11(n),         b = hash11(n + 1.0);
    float c = hash11(n + 57.0),  d = hash11(n + 58.0);
    float e = hash11(n + 113.0), g = hash11(n + 114.0);
    float h = hash11(n + 170.0), k = hash11(n + 171.0);
    return mix(mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
               mix(mix(e, g, f.x), mix(h, k, f.x), f.y), f.z);
}

vec3 flownoise(vec3 p) {   // 3 decorrelated channels → drifting flow field
    return vec3(vnoise(p),
                vnoise(p + vec3(31.7, 0.0, 0.0)),
                vnoise(p + vec3(0.0, 47.3, 0.0))) - 0.5;
}
"""

# One deformation for the three model styles (points / surface / wireframe),
# so they move identically. Neutral by default: both amounts come from the
# Physique routing (u_inflate = Gonflement, u_mturb = Turbulence), so with no
# routing the model stays exactly as modeled. Depends only on the rest
# position → welded vertices stay welded (no cracks in the surface).
_MODEL_GLSL = """
vec3 model_displace(vec3 p, vec3 n, float g) {
    float ph = (p.x + p.y * 0.6 + p.z * 0.3) * 0.010 - u_time * 2.2;
    float wv = sin(ph) * u_mturb * 14.0;
    vec3 drift = flownoise(p * 0.012 + vec3(u_time * 0.30)) * u_mturb * 6.0;
    // the vertex's own band adds spatial texture to the breathing
    return p + n * (u_inflate * 30.0 * (0.6 + 0.8 * g) + wv) + drift;
}
"""


def _glsl(src: str) -> str:
    return (src.replace("//%NOISE%", _NOISE_GLSL)
               .replace("//%MODEL%", _MODEL_GLSL))


GPU_POINT_VS = _glsl("""
#version 330
uniform mat4  u_mvp;
uniform mat4  u_model;       // model rotation (normals → view space)
uniform float u_size_scale;
uniform float u_size_mult;
uniform float u_time;
uniform float u_inflate;     // inflation along the normal (Gonflement)
uniform float u_mturb;       // routed turbulence (wave + drift)
uniform float u_model_gain;  // brightness compensation for point density
uniform float u_rim;         // fresnel rim intensity
uniform vec3  u_rim_color;
uniform float u_spectrum[64];

in vec3  in_pos;
in vec3  in_color;
in vec4  in_aux;
out vec3 v_color;

//%NOISE%
//%MODEL%

void main() {
    // no built-in audio reaction: all animation comes from the routing
    vec3 n = in_aux.xyz;
    float g = u_spectrum[int(in_aux.w + 0.5)];
    vec3 pos = model_displace(in_pos, n, g);
    // light the cloud like a surface: points facing the camera are
    // brighter, grazing ones get the fresnel rim → reads as a 3D volume
    float facing = abs(normalize(mat3(u_model) * n).z);
    vec3 col = in_color * (0.22 + 0.40 * facing);
    col += u_rim_color * pow(1.0 - facing, 3.0) * u_rim * 0.35;
    col *= u_model_gain;

    vec4 clip = u_mvp * vec4(pos, 1.0);
    gl_Position = clip;
    // sub-pixel points flicker as they cross pixel centers: clamp the sprite
    // to a minimum footprint and dim it to keep the same energy
    float ps = 1.6 * u_size_mult * u_size_scale / max(clip.w, 0.001);
    const float MIN_PS = 1.5;
    if (ps < MIN_PS) {
        col *= (ps * ps) / (MIN_PS * MIN_PS);
        ps = MIN_PS;
    }
    gl_PointSize = ps;
    v_color = col;
}
""")

# ── T19: 3D model surface / wireframe ────────────────────────────────────────
# Same deformation as the model points (shared model_displace). Vertex layout:
# pos3, normal3, color3, band1.

MESH_VS = _glsl("""
#version 330
uniform mat4  u_mvp;
uniform mat4  u_model;
uniform float u_view_d;       // camera distance (view = translate(0,0,-d))
uniform float u_time;
uniform float u_inflate;
uniform float u_mturb;
uniform float u_depth_bias;   // pushes the depth pre-pass back (hidden lines)
uniform float u_spectrum[64];
in vec3  in_pos;
in vec3  in_normal;
in vec3  in_color;
in float in_band;
in vec2  in_uv;
in float in_ao;
out vec3  v_normal;
out vec3  v_color;
out vec2  v_uv;
out vec3  v_view;             // surface → camera, view space
out float v_ao;

//%NOISE%
//%MODEL%

void main() {
    float g = u_spectrum[int(in_band + 0.5)];
    vec3 p = model_displace(in_pos, in_normal, g);
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    v_uv = in_uv;
    v_ao = in_ao;
    vec3 vp = (u_model * vec4(p, 1.0)).xyz - vec3(0.0, 0.0, u_view_d);
    v_view = -vp;
    gl_Position = u_mvp * vec4(p, 1.0);
    gl_Position.z += u_depth_bias * gl_Position.w;
}
""")

# Surface shading. Two looks:
#  - Studio (default): physically based — Lambert diffuse + GGX specular
#    (Smith, Schlick Fresnel) from a 3-light photo studio (warm key, cool
#    fill, strong back light) + a sky/ground gradient environment for the
#    ambient and the reflections; roughness / metalness sliders.
#  - Matcap: the original pre-lit sphere look.
# Both get the per-vertex cavity occlusion and the fresnel rim.
SURFACE_FS = """
#version 330
uniform sampler2D u_matcap;
uniform sampler2D u_tex;      // model texture (sampled per pixel via uv)
uniform int   u_shading;      // 0 studio, 1 matcap
uniform float u_rough;
uniform float u_metal;
uniform float u_ao_amt;       // cavity occlusion strength
uniform float u_light;        // light intensity (routable)
uniform float u_rim;
uniform vec3  u_rim_color;
uniform float u_orig;         // 0 = neutral clay / pure matcap, 1 = model colors
uniform float u_alpha;        // 1 = opaque, <1 = transparent (additive)
in vec3  v_normal;
in vec3  v_color;
in vec2  v_uv;
in vec3  v_view;
in float v_ao;
out vec4 f_color;
""" + _LINEAR_GLSL + """
const float PI = 3.14159265;

float D_ggx(float nh, float a) {
    float a2 = a * a, d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float V_smith(float nv, float nl, float a) {
    float k = a * 0.5;
    return 0.25 / ((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k));
}
vec3 F_schlick(vec3 f0, float vh) {
    return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
}
vec3 env(vec3 r, float rough) {   // studio sky / floor gradient (linear)
    float t = r.y * 0.5 + 0.5;
    vec3 c = mix(vec3(0.010, 0.010, 0.014), vec3(0.32, 0.34, 0.40),
                 smoothstep(0.15, 0.95, t));
    // soft box overhead-left, blurred by roughness
    float sb = pow(max(dot(r, normalize(vec3(-0.45, 0.75, 0.5))), 0.0),
                   mix(90.0, 4.0, rough));
    return c + vec3(1.6) * sb * (1.0 - rough * 0.6);
}

vec3 studio(vec3 n, vec3 v, vec3 albedo, float ao) {
    float a = max(u_rough * u_rough, 0.003);
    vec3 f0 = mix(vec3(0.04), albedo, u_metal);
    vec3 diff_c = albedo * (1.0 - u_metal);
    float nv = max(dot(n, v), 1e-4);
    vec3 L[3] = vec3[](normalize(vec3(-0.55, 0.65, 0.55)),    // key
                       normalize(vec3( 0.75, 0.05, 0.45)),    // fill
                       normalize(vec3( 0.25, 0.55, -0.80)));  // back
    vec3 C[3] = vec3[](vec3(3.2, 2.95, 2.7), vec3(0.45, 0.55, 0.75),
                       vec3(2.6, 2.7, 3.0));
    vec3 col = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        vec3 l = L[i], h = normalize(l + v);
        float nl = max(dot(n, l), 0.0);
        if (nl <= 0.0) continue;
        float nh = max(dot(n, h), 0.0), vh = max(dot(v, h), 0.0);
        vec3 F = F_schlick(f0, vh);
        vec3 spec = D_ggx(nh, a) * V_smith(nv, nl, a) * F;
        col += ((1.0 - F) * diff_c / PI + spec) * C[i] * nl;
    }
    // environment: diffuse irradiance ≈ gradient along the normal,
    // specular ≈ gradient + soft box along the reflection
    vec3 Fe = f0 + (max(vec3(1.0 - u_rough), f0) - f0) * pow(1.0 - nv, 5.0);
    col += diff_c * env(n, 1.0) * 0.9 * ao;
    col += env(reflect(-v, n), u_rough) * Fe * mix(1.0, ao, 0.5);
    return col * u_light;
}

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;          // open meshes: light the inside too
    vec3 v = normalize(v_view);
    vec3 base = v_uv.x < -10.0 ? v_color : texture(u_tex, v_uv).rgb;
    float ao = clamp(1.0 - u_ao_amt * (1.0 - v_ao), 0.0, 1.0);
    float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    vec3 lin;
    if (u_shading == 0) {
        vec3 albedo = to_linear(mix(vec3(0.62), base, u_orig));
        lin = studio(n, v, albedo, ao);
        lin += to_linear(u_rim_color) * fres * u_rim * 0.5;
    } else {
        vec3 mc = texture(u_matcap, n.xy * 0.495 + 0.5).rgb;
        float lum = dot(mc, vec3(0.299, 0.587, 0.114));
        vec3 col = mix(mc, base * lum * 1.6, u_orig) * 1.1 * ao * u_light;
        col += u_rim_color * fres * u_rim;
        lin = to_linear(col);
    }
    // opaque: alpha 1 occludes trails + background; transparent: emission
    f_color = vec4(lin * u_alpha, u_alpha >= 0.999 ? 1.0 : 0.0);
}
"""

# Wireframe: each edge is expanded into a screen-aligned quad (geometry
# shader) of a chosen pixel width with an anti-aliased falloff — core-profile
# GL lines are 1 px and aliased. Edges on the far side are dimmed (depth cue).
WIRE_GS = """
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2  u_viewport;     // px
uniform float u_width;        // px
in vec3 v_normal[];
in vec3 v_color[];
out vec3  g_normal;
out vec3  g_color;
out float g_d;                // signed distance across the line, in px
void main() {
    vec4 p0 = gl_in[0].gl_Position, p1 = gl_in[1].gl_Position;
    if (p0.w <= 0.0 || p1.w <= 0.0) return;
    vec2 s0 = p0.xy / p0.w * u_viewport * 0.5;
    vec2 s1 = p1.xy / p1.w * u_viewport * 0.5;
    vec2 dir = s1 - s0;
    float len = length(dir);
    dir = len > 1e-4 ? dir / len : vec2(1.0, 0.0);
    float hw = u_width * 0.5 + 1.0;          // +1 px for the AA ramp
    vec2 off = vec2(-dir.y, dir.x) * hw / (u_viewport * 0.5);
    for (int i = 0; i < 2; i++) {
        vec4 p = i == 0 ? p0 : p1;
        g_normal = v_normal[i];
        g_color = v_color[i];
        g_d = hw;
        gl_Position = p + vec4(off * p.w, 0.0, 0.0);
        EmitVertex();
        g_d = -hw;
        gl_Position = p - vec4(off * p.w, 0.0, 0.0);
        EmitVertex();
    }
    EndPrimitive();
}
"""

WIRE_FS = """
#version 330
uniform float u_rim;
uniform vec3  u_rim_color;
uniform float u_wire_gain;    // per-model: dense meshes are dimmed
uniform float u_emit;
uniform float u_width;
uniform float u_light;
in vec3  g_normal;
in vec3  g_color;
in float g_d;
out vec4 f_color;
""" + _LINEAR_GLSL + """
void main() {
    float cov = clamp(u_width * 0.5 + 0.5 - abs(g_d), 0.0, 1.0);
    if (cov <= 0.0) discard;
    vec3 n = normalize(g_normal);
    float facing = abs(n.z);
    float front = n.z >= 0.0 ? 1.0 : 0.35;   // far side of the model dimmer
    vec3 col = g_color * (0.45 + 0.55 * facing) * front;
    col += u_rim_color * pow(1.0 - facing, 2.0) * u_rim * 0.5;
    f_color = vec4(to_linear(col) * cov * u_wire_gain * u_emit * u_light, 0.0);
}
"""

# Fullscreen triangle (no VBO needed): generated from gl_VertexID
FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Trails: the frame's scene is merged into the persistent trail layer. Light
# keeps the brightest of (decayed past, present) so moving glow leaves a tail;
# where the present frame is opaque (model surface) it wins outright, so a
# rotating surface never ghosts over itself.
TRAIL_FS = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_trail;
uniform float u_decay;   // 1 - fade
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 s = texture(u_scene, v_uv);
    vec4 t = texture(u_trail, v_uv) * u_decay;
    vec3 c = mix(max(t.rgb, s.rgb), s.rgb, s.a);
    f_color = vec4(c, max(t.a, s.a));
}
"""

# Bloom (Jimenez, "Next generation post processing in Call of Duty: AW"):
# 13-tap downsample chain with a soft-knee threshold + Karis average on the
# first level (only highlights glow, no fireflies), then a tent-filter
# upsample chain that blends each level with the next (u_scatter = radius).
BLOOM_DOWN_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2  u_texel;        // source texel size
uniform int   u_prefilter;    // 1 on the first level
uniform float u_threshold;
uniform float u_knee;
in vec2 v_uv;
out vec4 f_color;
vec3 S(float x, float y) { return texture(u_tex, v_uv + vec2(x, y) * u_texel).rgb; }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 karis(vec3 a, vec3 b, vec3 c, vec3 d, out float w) {
    vec3 m = (a + b + c + d) * 0.25;
    w = 1.0 / (1.0 + luma(m));
    return m * w;
}
void main() {
    vec3 a = S(-2.0,  2.0), b = S(0.0,  2.0), c = S(2.0,  2.0);
    vec3 d = S(-2.0,  0.0), e = S(0.0,  0.0), f = S(2.0,  0.0);
    vec3 g = S(-2.0, -2.0), h = S(0.0, -2.0), i = S(2.0, -2.0);
    vec3 j = S(-1.0,  1.0), k = S(1.0,  1.0);
    vec3 l = S(-1.0, -1.0), m = S(1.0, -1.0);
    vec3 col;
    if (u_prefilter == 1) {
        float w0, w1, w2, w3, w4;
        vec3 s = karis(j, k, l, m, w0) * 0.5
               + (karis(a, b, d, e, w1) + karis(b, c, e, f, w2)
                + karis(d, e, g, h, w3) + karis(e, f, h, i, w4)) * 0.125;
        col = s / (w0 * 0.5 + (w1 + w2 + w3 + w4) * 0.125);
        // soft-knee threshold on the brightest channel
        float br = max(max(col.r, col.g), col.b);
        float rq = clamp(br - u_threshold + u_knee, 0.0, 2.0 * u_knee);
        rq = rq * rq / (4.0 * u_knee + 1e-5);
        col *= max(rq, br - u_threshold) / max(br, 1e-5);
    } else {
        col = e * 0.125 + (a + c + g + i) * 0.03125
            + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    f_color = vec4(max(col, vec3(0.0)), 1.0);
}
"""

BLOOM_UP_FS = """
#version 330
uniform sampler2D u_low;      // next (smaller) level, already upsampled
uniform sampler2D u_cur;      // this level's downsample
uniform vec2  u_texel;        // u_low texel size
uniform float u_scatter;      // 0 = tight glow … 1 = wide halos
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec2 t = u_texel;
    vec3 up = texture(u_low, v_uv).rgb * 4.0
        + (texture(u_low, v_uv + vec2(-t.x, 0.0)).rgb
         + texture(u_low, v_uv + vec2( t.x, 0.0)).rgb
         + texture(u_low, v_uv + vec2(0.0, -t.y)).rgb
         + texture(u_low, v_uv + vec2(0.0,  t.y)).rgb) * 2.0
        + texture(u_low, v_uv + vec2(-t.x, -t.y)).rgb
        + texture(u_low, v_uv + vec2( t.x, -t.y)).rgb
        + texture(u_low, v_uv + vec2(-t.x,  t.y)).rgb
        + texture(u_low, v_uv + vec2( t.x,  t.y)).rgb;
    f_color = vec4(mix(texture(u_cur, v_uv).rgb, up / 16.0, u_scatter), 1.0);
}
"""

# AgX (Troy Sobotka; fit by B. Wrensch) with the "punchy" look: filmic
# highlight roll-off that keeps saturated neons from clipping to flat white
# and gives deep blacks, unlike the old Reinhard curve.
COMPOSITE_FS = """
#version 330
uniform sampler2D u_scene;    // trail layer (linear HDR, alpha = opacity)
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
uniform float u_brightness;   // exposure (linear multiplier)
uniform vec3  u_bg;           // background color (display space)
uniform vec3  u_gain;         // per-channel RGB gain
uniform float u_contrast;     // post contrast around 0.5
uniform float u_invert;       // 0 = normal, 1 = negative
uniform float u_grain;        // film grain amount
uniform float u_vignette;     // 0 = off
uniform float u_time;         // seed for the grain noise
uniform float u_ss;           // supersampling factor of u_scene
uniform vec2  u_out_texel;    // 1 / output size
uniform vec2 u_shift_r;       // per-channel uv offsets (chromatic shift)
uniform vec2 u_shift_g;
uniform vec2 u_shift_b;
in vec2 v_uv;
out vec4 f_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec4 scene_at(vec2 uv) {
    if (u_ss < 1.01) return texture(u_scene, uv);
    // supersampled scene: 2×2 bilinear taps = box filter over the footprint
    vec2 o = u_out_texel * 0.25;
    return 0.25 * (texture(u_scene, uv + vec2(-o.x, -o.y)) +
                   texture(u_scene, uv + vec2( o.x, -o.y)) +
                   texture(u_scene, uv + vec2(-o.x,  o.y)) +
                   texture(u_scene, uv + vec2( o.x,  o.y)));
}

vec3 agx_contrast(vec3 x) {
    vec3 x2 = x * x, x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
         - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agx(vec3 v) {
    const mat3 m_in = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                           0.0784335999999992, 0.878468636469772, 0.0784336,
                           0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 m_out = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                            -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                            -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float lo = -12.47393, hi = 4.026069;
    v = m_in * max(v, vec3(1e-10));
    v = (clamp(log2(v), lo, hi) - lo) / (hi - lo);
    v = agx_contrast(v);
    // "punchy" look: a bit more power + saturation
    v = pow(max(v, vec3(0.0)), vec3(1.35));
    float l = dot(v, vec3(0.2126, 0.7152, 0.0722));
    v = l + 1.4 * (v - l);
    return clamp(m_out * v, 0.0, 1.0);   // display-encoded
}

void main() {
    vec4 c = scene_at(v_uv);
    vec3 hdr = c.rgb;
    if (any(notEqual(vec4(u_shift_r, u_shift_g), vec4(0.0)))
            || any(notEqual(u_shift_b, vec2(0.0))))
        hdr = vec3(scene_at(v_uv + u_shift_r).r,
                   scene_at(v_uv + u_shift_g).g,
                   scene_at(v_uv + u_shift_b).b);
    hdr += texture(u_bloom, v_uv).rgb * u_bloom_intensity;
    vec3 col = agx(hdr * u_brightness);
    col += u_bg * (1.0 - clamp(c.a, 0.0, 1.0));
    col *= u_gain;
    col = (col - 0.5) * u_contrast + 0.5;
    col = mix(col, vec3(1.0) - col, u_invert);
    if (u_vignette > 0.001) {
        vec2 q = (v_uv - 0.5) * vec2(u_out_texel.y / u_out_texel.x, 1.0);
        col *= mix(1.0, smoothstep(1.15, 0.25, length(q)), u_vignette);
    }
    if (u_grain > 0.001) {
        float n = hash(v_uv * 971.0 + vec2(u_time, u_time * 1.7));
        col += (n - 0.5) * u_grain;
    }
    // triangular dither ±1 LSB: no banding in the dark gradients / halos
    vec2 px = gl_FragCoord.xy;
    col += (hash(px + fract(u_time) * 61.0) + hash(px * 1.37 + 17.0) - 1.0) / 255.0;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

LINE_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""

LINE_FS = """
#version 330
uniform vec3 u_color;
out vec4 f_color;
void main() { f_color = vec4(pow(u_color, vec3(2.2)) * 3.0, 0.0); }
"""

GRID_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
out float v_dist;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_dist = length(in_pos.xz);
}
"""

GRID_FS = """
#version 330
uniform vec3 u_color;
in float v_dist;
out vec4 f_color;
void main() {
    // fade the grid out toward the horizon
    float fade = 1.0 - smoothstep(700.0, 2100.0, v_dist);
    f_color = vec4(pow(u_color, vec3(2.2)) * fade * 3.0, 0.0);
}
"""

OVERLAY_VS = """
#version 330
in vec2 in_pos;   // NDC
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

OVERLAY_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);   // texture uploaded flipped → samples upright
}
"""


class GLRenderer:
    def __init__(self, ctx: moderngl.Context, size: tuple,
                 bg_color=(4, 4, 10), fade=0.10, bloom_intensity=1.4,
                 output_fbo=None):
        self.ctx = ctx
        self.w, self.h = size
        # Final target for compositing: a custom FBO (e.g. for video export) or
        # the window's default framebuffer.
        self.output_fbo = output_fbo
        self.bg = tuple(c / 255.0 for c in bg_color)
        self.fade = fade
        self.bloom_intensity = bloom_intensity

        self._screen = None
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # ── Programs ──
        self.trail_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=TRAIL_FS)
        self.down_prog = ctx.program(vertex_shader=FULLSCREEN_VS,
                                     fragment_shader=BLOOM_DOWN_FS)
        self.up_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=BLOOM_UP_FS)
        self.comp_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=COMPOSITE_FS)
        self._fs_vaos = {}           # one empty VAO per fullscreen program
        self.overlay_prog = ctx.program(vertex_shader=OVERLAY_VS, fragment_shader=OVERLAY_FS)
        self.grid_prog = ctx.program(vertex_shader=GRID_VS, fragment_shader=GRID_FS)
        self.line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        # 3D model: points (stateless GPU physics, T1), surface, wireframe (T19)
        self.gpu_prog = ctx.program(vertex_shader=GPU_POINT_VS,
                                    fragment_shader=POINT_FS)
        self.mesh_prog = ctx.program(vertex_shader=MESH_VS,
                                     fragment_shader=SURFACE_FS)
        self.wire_prog = ctx.program(vertex_shader=MESH_VS,
                                     geometry_shader=WIRE_GS,
                                     fragment_shader=WIRE_FS)
        # programs that receive the model transform + look uniforms
        self._model_progs = [self.gpu_prog, self.mesh_prog, self.wire_prog]
        self._matcaps = []
        for arr in matcaps.build_all():
            tex = ctx.texture((arr.shape[1], arr.shape[0]), 3,
                              arr.tobytes(), dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self._matcaps.append(tex)
        self._model_tex = None       # texture of the current model
        self._mesh_objs = []         # GL buffers/VAOs of the current model
        self._mesh_vaos = None
        self._mesh_counts = {}
        self._matcap_idx = 0
        self._mesh_transparent = False

        # additive passes used to pile up over ~1/fade frames in the old
        # accumulation buffer; the trail layer keeps the max instead, so the
        # emission gain restores the brightness the presets were tuned for
        for prog in (self.gpu_prog, self.wire_prog):
            if "u_emit" in prog._members:
                prog["u_emit"].value = EMISSION_GAIN
        self.set_post_params()  # initialize composite uniforms to neutral
        self.set_model_look()

        # ── Camera ──
        self.fov = np.radians(45.0)
        # distance so the model height (~h px) fills the vertical field of view
        self.cam_dist = (self.h / 2.0) / np.tan(self.fov / 2.0)
        self._proj = _perspective(self.fov, self.w / self.h, 10.0, 8000.0)
        self._view = _translate(0.0, 0.0, -self.cam_dist)
        self.set_transform(0.0)

        # ── Point VBO / VAO (T1): lazily sized static buffer ──
        self._gpu_vbo = None
        self._gpu_vao = None
        self._gpu_count = 0

        # ── Framebuffers ──
        # The scene is drawn each frame into a multisampled HDR buffer (MSAA
        # 4×: clean triangle edges and lines), resolved, then merged into the
        # persistent trail layer. Sizes = logical size × pixel ratio (Retina)
        # × render_scale (optional extra supersampling).
        self.dpr = 1.0
        self.render_scale = 1.0
        self.bloom_radius = 0.5      # 0 = tight glow, 1 = wide halos
        self.bloom_threshold = 0.8   # linear luminance where the glow starts
        self.vignette = 0.0
        self.samples = min(4, ctx.max_samples)
        self._gl_objs = []           # size-dependent GL objects
        self._trail_idx = 0
        self._resolved = False
        self._build_internal()

        # ── Static floor grid (world-space, does not rotate with the object) ──
        ext, step = 1500.0, 100.0
        gy = -self.h * 0.55
        lines = []
        for v in np.arange(-ext, ext + 1, step):
            lines += [[v, gy, -ext], [v, gy, ext],      # lines along Z
                      [-ext, gy, v], [ext, gy, v]]      # lines along X
        self._grid_vbo = ctx.buffer(np.array(lines, dtype="f4").tobytes())
        self._grid_n = len(lines)
        self._grid_vao = ctx.vertex_array(
            self.grid_prog, [(self._grid_vbo, "3f", "in_pos")])
        self.grid_prog["u_color"].value = (0.16, 0.22, 0.38)

        # ── Rotation-axis indicator (green line, updated in set_transform) ──
        self._axis_dir = (0.0, 1.0, 0.0)
        self._axis_vbo = ctx.buffer(reserve=2 * 3 * 4)
        self._axis_vao = ctx.vertex_array(
            self.line_prog, [(self._axis_vbo, "3f", "in_pos")])
        self.line_prog["u_color"].value = (0.05, 0.55, 0.12)   # green, additive

        # ── HUD overlay texture + quad (lazily sized) ──
        self._overlay_tex = None
        self._overlay_h = 0
        self._overlay_vbo = None
        self._overlay_vao = None

    # ------------------------------------------------------------------
    # Frame

    @property
    def _target(self):
        if self.output_fbo is not None:
            return self.output_fbo
        if self._screen is None:
            # re-detect the window's default framebuffer (glo 0: without it,
            # detect_framebuffer() returns whatever FBO is bound right now,
            # e.g. a bloom level) and force its viewport: it can report a
            # stale size inherited from the previous GL viewport
            self._screen = self.ctx.detect_framebuffer(0)
            self._screen.viewport = (0, 0, int(round(self.w * self.dpr)),
                                     int(round(self.h * self.dpr)))
        return self._screen

    def _build_internal(self):
        """(Re)create the scene MSAA buffer, the trail layer and the bloom chain."""
        for obj in self._gl_objs:
            obj.release()
        objs = self._gl_objs = []
        ctx = self.ctx

        def tex(size, dtype="f2"):
            t = ctx.texture(size, 4, dtype=dtype)
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            objs.append(t)
            return t

        def fbo(color, depth=None):
            f = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
            objs.append(f)
            return f

        # render_scale = total pixel density wanted per logical pixel: a
        # Retina framebuffer (dpr 2) already provides it, MSAA does the rest
        k = max(self.dpr, self.render_scale)
        rw, rh = max(8, int(self.w * k)), max(8, int(self.h * k))
        self._internal_size = (rw, rh)

        # resolved scene (what the trail pass reads)
        self._scene_tex = tex((rw, rh))
        self._scene_depth = ctx.depth_renderbuffer((rw, rh))
        objs.append(self._scene_depth)
        if self.samples > 1:
            rb = ctx.renderbuffer((rw, rh), 4, samples=self.samples, dtype="f2")
            db = ctx.depth_renderbuffer((rw, rh), samples=self.samples)
            objs += [rb, db]
            self._scene_resolve = fbo(self._scene_tex)
            self._accum = fbo(rb, db)            # draw target of the frame
        else:
            self._scene_resolve = None
            self._accum = fbo(self._scene_tex, self._scene_depth)

        # persistent trail layer (ping-pong)
        self._trail_tex = [tex((rw, rh)), tex((rw, rh))]
        self._trail_fbo = [fbo(t) for t in self._trail_tex]
        for f in self._trail_fbo:
            f.use()
            ctx.clear(0.0, 0.0, 0.0, 0.0)

        # bloom chain: 6 levels from 1/2 down to 1/64 of the internal size
        self._bloom_down, self._bloom_up = [], []
        bw, bh = rw, rh
        for _ in range(6):
            bw, bh = max(2, bw // 2), max(2, bh // 2)
            d, u = tex((bw, bh)), tex((bw, bh))
            self._bloom_down.append((d, fbo(d)))
            self._bloom_up.append((u, fbo(u)))

        # point sizes are expressed in internal pixels
        self.set_size_mult(getattr(self, "_size_mult", 1.0))

    def set_render_scale(self, scale: float):
        """Supersampling (1.0–2.0 internal px per logical px; no effect below
        the pixel ratio — a Retina screen is already at 2)."""
        scale = min(max(scale, 1.0), 2.0)
        if abs(scale - self.render_scale) > 1e-3:
            self.render_scale = scale
            self._build_internal()

    def set_pixel_ratio(self, dpr: float):
        """Framebuffer pixels per logical pixel (2 on a Retina screen)."""
        dpr = max(float(dpr), 0.5)
        if abs(dpr - self.dpr) > 1e-3:
            self.dpr = dpr
            self._screen = None
            self._build_internal()

    def resize(self, w: int, h: int):
        """Rebuild size-dependent GL state after a window resize / fullscreen.
        w, h are LOGICAL sizes (the camera framing does not depend on dpr)."""
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = w, h
        self.cam_dist = (h / 2.0) / np.tan(self.fov / 2.0)
        self._proj = _perspective(self.fov, w / h, 10.0, 8000.0)
        self._view = _translate(0.0, 0.0, -self.cam_dist)
        self._screen = None
        self._build_internal()
        # force the HUD overlay quad/texture to be rebuilt at the new size
        self._overlay_h = 0

    def set_transform(self, angle: float, scale: float = 1.0,
                      axis_tilt: float = 0.0, axis_azimuth: float = 0.0,
                      orbit_dist: float = 0.0, obj_tilt: float = 0.0):
        """Update the model-view-projection matrix for the particle pass.

        The object spins by `angle` around a FIXED world axis defined by
        `axis_tilt` (radians from vertical) and `axis_azimuth` (radians),
        offset from that axis by `orbit_dist` (px → the object orbits).
        `obj_tilt` is the object's own inclination. Only `angle` and `scale`
        are music-driven; the axis parameters come from the control panel.
        """
        st, ct = np.sin(axis_tilt), np.cos(axis_tilt)
        axis = (st * np.cos(axis_azimuth), ct, st * np.sin(axis_azimuth))
        self._axis_dir = axis
        model = (_rotate_axis(axis, angle)
                 @ _translate(orbit_dist, 0.0, 0.0)
                 @ _rotate_x(obj_tilt)
                 @ _scale(scale))
        mvp = self._proj @ self._view @ model
        # GL expects column-major → transpose the row-major numpy matrix
        mvp_bytes = mvp.T.astype("f4").tobytes()
        model_bytes = model.T.astype("f4").tobytes()
        for prog in self._model_progs:
            prog["u_mvp"].write(mvp_bytes)
            if "u_model" in prog._members:
                prog["u_model"].write(model_bytes)

    def set_post_params(self, brightness=1.0, gain=(1.0, 1.0, 1.0), contrast=1.0,
                        invert=0.0, grain=0.0, time=0.0,
                        shift_r=(0.0, 0.0), shift_g=(0.0, 0.0), shift_b=(0.0, 0.0)):
        """Update the post-processing (composite) uniforms."""
        p = self.comp_prog
        # the exposure slider was tuned on display values: square it so its
        # perceived range stays similar in linear light
        p["u_brightness"].value = float(brightness) ** 2
        p["u_shift_r"].value = tuple(shift_r)
        p["u_shift_g"].value = tuple(shift_g)
        p["u_shift_b"].value = tuple(shift_b)
        p["u_gain"].value = tuple(float(g) for g in gain)
        p["u_contrast"].value = float(contrast)
        p["u_invert"].value = float(invert)
        p["u_grain"].value = float(grain)
        p["u_time"].value = float(time)

    def set_zoom(self, zoom: float, shake=(0.0, 0.0)):
        """Camera zoom: >1 gets closer, <1 pulls back. shake: small camera
        offset (px) for the routed « Secousse caméra » effect."""
        z = max(zoom, 0.05)
        self._view = _translate(shake[0], shake[1], -self.cam_dist / z)
        for prog in self._model_progs:
            if "u_view_d" in prog._members:
                prog["u_view_d"].value = float(self.cam_dist / z)

    def draw_grid(self):
        """Draw the static world grid into the accumulation buffer."""
        self._accum.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)   # faint additive
        mvp = self._proj @ self._view
        self.grid_prog["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self._grid_vao.render(mode=moderngl.LINES, vertices=self._grid_n)

    def draw_axis(self):
        """Draw the current rotation axis as a green world-space line."""
        L = 1400.0
        a = np.array(self._axis_dir, dtype="f4")
        self._axis_vbo.write(np.concatenate([-a * L, a * L]).astype("f4").tobytes())
        self._accum.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        mvp = self._proj @ self._view
        self.line_prog["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self._axis_vao.render(mode=moderngl.LINES, vertices=2)

    def set_model_look(self, matcap=0, rim=1.0, rim_hue=0.52, orig=0.5,
                       transparent=False, shading=0, roughness=0.4, metal=0.0,
                       ao=1.0, wire_width=1.2, light=1.0):
        """T19: surface look of the 3D model (studio PBR or matcap, cavity
        occlusion, fresnel rim, colors, wireframe line width in logical px)."""
        rim_color = colorsys.hsv_to_rgb(rim_hue % 1.0, 0.65, 1.0)
        self._wire_width = float(wire_width)
        for prog in self._model_progs:
            for name, val in (("u_rim", float(rim)), ("u_rim_color", rim_color),
                              ("u_orig", float(orig)), ("u_shading", int(shading)),
                              ("u_rough", float(min(max(roughness, 0.03), 1.0))),
                              ("u_metal", float(metal)), ("u_ao_amt", float(ao)),
                              ("u_light", float(light))):
                if name in prog._members:
                    prog[name].value = val
        self._matcap_idx = int(matcap) % max(len(self._matcaps), 1)
        self._mesh_transparent = bool(transparent)

    def set_size_mult(self, mult: float):
        """Global particle size multiplier (control-panel 'Taille points').
        Point sizes are expressed in INTERNAL (supersampled) pixels."""
        self._size_mult = float(mult)
        s = self.cam_dist * max(self.dpr, self.render_scale)
        self.gpu_prog["u_size_scale"].value = float(s)
        self.gpu_prog["u_size_mult"].value = float(mult)

    # ------------------------------------------------------------------
    # T1: stateless GPU particle path

    STATIC_FLOATS = 10    # pos3 + color3 + aux4 (visuals/mesh.build_points)

    def upload_particle_static(self, data: np.ndarray, count: int):
        """Upload the per-point rest attributes ONCE (until the next rebuild)."""
        if count <= 0 or data is None:
            self._gpu_count = 0
            return
        needed = count * self.STATIC_FLOATS * 4
        if self._gpu_vbo is None or self._gpu_vbo.size < needed:
            if self._gpu_vbo is not None:
                self._gpu_vbo.release()
                self._gpu_vao.release()
            self._gpu_vbo = self.ctx.buffer(reserve=needed)
            self._gpu_vao = self.ctx.vertex_array(
                self.gpu_prog,
                [(self._gpu_vbo, "3f 3f 4f", "in_pos", "in_color", "in_aux")],
            )
        self._gpu_vbo.write(np.ascontiguousarray(data[:count], dtype="f4").tobytes())
        self._gpu_count = int(count)

    def draw_particles_gpu(self, uniforms: dict):
        """Draw the static buffer; all physics happens in the vertex shader."""
        if self._gpu_vao is None or self._gpu_count <= 0:
            return
        prog = self.gpu_prog
        self._write_frame_uniforms(prog, uniforms, (
            "u_time", "u_inflate", "u_mturb", "u_model_gain"))
        self._points_target().use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)   # additive
        self._gpu_vao.render(mode=moderngl.POINTS, vertices=self._gpu_count)

    @staticmethod
    def _write_frame_uniforms(prog, uniforms: dict, names):
        for name in names:
            # unused uniforms get optimized out of the compiled program
            if name in prog._members:
                prog[name].value = float(uniforms[name])
        if "u_spectrum" in prog._members:
            prog["u_spectrum"].write(
                np.ascontiguousarray(uniforms["u_spectrum"], dtype="f4").tobytes())

    # ------------------------------------------------------------------
    # T19: 3D model surface / wireframe

    @property
    def has_mesh(self) -> bool:
        return self._mesh_vaos is not None

    def upload_mesh(self, m):
        """Upload the model's welded mesh (or free it when m is None)."""
        for obj in self._mesh_objs:
            obj.release()
        self._mesh_objs, self._mesh_vaos, self._mesh_counts = [], None, {}
        if m is None or len(m["tris"]) == 0:
            return
        ctx = self.ctx
        vbo = ctx.buffer(np.ascontiguousarray(m["verts"], dtype="f4").tobytes())
        self._mesh_objs.append(vbo)
        # model texture (1×1 white when the model has none: never sampled then)
        img = m.get("tex")
        if img is None:
            img = np.full((1, 1, 3), 255, dtype=np.uint8)
        h, w = img.shape[:2]
        tex = ctx.texture((w, h), 3, np.ascontiguousarray(img).tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 8.0
        self._mesh_objs.append(tex)
        self._model_tex = tex
        vaos = {}
        for key, prog, idx in (("surf", self.mesh_prog, m["tris"]),
                               ("edges", self.wire_prog, m["edges"]),
                               ("sharp", self.wire_prog, m["sharp"])):
            if len(idx) == 0:
                continue
            # attributes a program never reads are dropped by the compiler:
            # skip them in the layout (uv and ao are surface-only)
            names = ["in_pos", "in_normal", "in_color", "in_band"]
            fmt = "3f 3f 3f 1f"
            for name, n in (("in_uv", 2), ("in_ao", 1)):
                if name in prog._members:
                    fmt += f" {n}f"
                    names.append(name)
                else:
                    fmt += f" {n}x4"
            layout = [(vbo, fmt, *names)]
            ibo = ctx.buffer(np.ascontiguousarray(idx, dtype="u4").tobytes())
            vao = ctx.vertex_array(prog, layout, index_buffer=ibo,
                                   index_element_size=4)
            self._mesh_objs += [ibo, vao]
            vaos[key] = vao
            self._mesh_counts[key] = idx.size
        self._mesh_vaos = vaos
        self.wire_prog["u_wire_gain"].value = m["wire_gain"]

    def draw_model_mesh(self, uniforms: dict, style: int, sharp_only=False,
                        hide_hidden=False):
        """style 1 = surface (matcap + rim), 2 = wireframe."""
        if not self.has_mesh:
            return
        names = ("u_time", "u_inflate", "u_mturb")
        for prog in (self.mesh_prog, self.wire_prog):
            self._write_frame_uniforms(prog, uniforms, names)
        # line width in internal px; the additive brightness of a line grows
        # with its width, so the gain is normalized to a 1.5 px reference
        k = max(self.dpr, self.render_scale)
        w_px = max(getattr(self, "_wire_width", 1.2) * k, 1.0)
        self.wire_prog["u_width"].value = w_px
        self.wire_prog["u_viewport"].value = tuple(float(v) for v in self._internal_size)
        if "u_emit" in self.wire_prog._members:
            self.wire_prog["u_emit"].value = EMISSION_GAIN * 1.5 / w_px
        ctx, fbo, vaos = self.ctx, self._accum, self._mesh_vaos
        fbo.use()
        self._matcaps[self._matcap_idx].use(0)
        self.mesh_prog["u_matcap"].value = 0
        self._model_tex.use(1)
        self.mesh_prog["u_tex"].value = 1
        self.mesh_prog["u_depth_bias"].value = 0.0

        if style == 1:
            if self._mesh_transparent:
                ctx.disable(moderngl.DEPTH_TEST)
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE)
                self.mesh_prog["u_alpha"].value = 0.3
            else:
                ctx.enable(moderngl.DEPTH_TEST)
                ctx.disable(moderngl.BLEND)
                self.mesh_prog["u_alpha"].value = 1.0
            vaos["surf"].render(moderngl.TRIANGLES,
                                vertices=self._mesh_counts["surf"])
        else:
            if hide_hidden:
                # depth-only pre-pass of the surface, pushed slightly back so
                # the visible edges (same depth) pass the test
                fbo.color_mask = (False, False, False, False)
                fbo.use()
                ctx.enable(moderngl.DEPTH_TEST)
                ctx.disable(moderngl.BLEND)
                self.mesh_prog["u_depth_bias"].value = 0.002
                vaos["surf"].render(moderngl.TRIANGLES,
                                    vertices=self._mesh_counts["surf"])
                self.mesh_prog["u_depth_bias"].value = 0.0
                fbo.color_mask = (True, True, True, True)
                fbo.use()
            else:
                ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)   # glowing lines
            # smooth meshes (sphere…) have no sharp edge: fall back to all
            key = "sharp" if sharp_only and "sharp" in vaos else "edges"
            vaos[key].render(moderngl.LINES, vertices=self._mesh_counts[key])
        ctx.disable(moderngl.DEPTH_TEST)

    def begin_frame(self):
        """Start a frame: clear the scene buffer (the trails live apart)."""
        self._accum.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._resolved = False

    def _resolve(self):
        """MSAA scene → _scene_tex (once per frame)."""
        if not self._resolved and self._scene_resolve is not None:
            self.ctx.copy_framebuffer(self._scene_resolve, self._accum)
        self._resolved = True

    def _points_target(self):
        """Soft glowing points gain nothing from MSAA but cost 4× the blending:
        they are drawn into the resolved buffer, after the geometry."""
        if self._scene_resolve is None:
            return self._accum
        self._resolve()
        return self._scene_resolve

    def end_frame(self):
        """Resolve MSAA, merge into the trails, bloom chain, composite."""
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)
        self._resolve()

        # trails: trail' = merge(trail × decay, scene)
        src, dst = self._trail_idx, 1 - self._trail_idx
        self._trail_fbo[dst].use()
        self._scene_tex.use(0)
        self._trail_tex[src].use(1)
        tp = self.trail_prog
        tp["u_scene"].value, tp["u_trail"].value = 0, 1
        tp["u_decay"].value = float(1.0 - min(max(self.fade, 0.0), 1.0))
        self._render_fullscreen(tp)
        self._trail_idx = dst
        trail = self._trail_tex[dst]

        # bloom: thresholded 13-tap downsample chain …
        dp = self.down_prog
        dp["u_tex"].value = 0
        dp["u_threshold"].value = float(self.bloom_threshold)
        dp["u_knee"].value = float(max(self.bloom_threshold * 0.5, 0.05))
        src_tex = trail
        for i, (t, f) in enumerate(self._bloom_down):
            f.use()
            src_tex.use(0)
            dp["u_texel"].value = (1.0 / src_tex.width, 1.0 / src_tex.height)
            dp["u_prefilter"].value = 1 if i == 0 else 0
            self._render_fullscreen(dp)
            src_tex = t
        # … then tent upsample, each level blended with the coarser one
        up = self.up_prog
        up["u_low"].value, up["u_cur"].value = 0, 1
        up["u_scatter"].value = float(0.45 + 0.5 * min(max(self.bloom_radius, 0.0), 1.0))
        low = self._bloom_down[-1][0]
        for i in range(len(self._bloom_down) - 2, -1, -1):
            u_tex, u_fbo = self._bloom_up[i]
            u_fbo.use()
            low.use(0)
            self._bloom_down[i][0].use(1)
            up["u_texel"].value = (1.0 / low.width, 1.0 / low.height)
            self._render_fullscreen(up)
            low = u_tex

        # composite to the output target (resamples the internal resolution)
        target = self._target
        target.use()
        trail.use(0)
        low.use(1)
        cp = self.comp_prog
        cp["u_scene"].value, cp["u_bloom"].value = 0, 1
        cp["u_bloom_intensity"].value = self.bloom_intensity
        cp["u_bg"].value = self.bg
        cp["u_vignette"].value = float(self.vignette)
        ow, oh = target.viewport[2], target.viewport[3]
        cp["u_out_texel"].value = (1.0 / max(ow, 1), 1.0 / max(oh, 1))
        cp["u_ss"].value = float(self._internal_size[0] / max(ow, 1))
        self._render_fullscreen(cp)

    # ------------------------------------------------------------------
    # HUD overlay

    def has_overlay(self, width: int, height: int) -> bool:
        return (self._overlay_tex is not None and self._overlay_h == height
                and self._overlay_tex.size[0] == width)

    def draw_overlay(self, rgba_bytes, width: int, height: int):
        """
        Draw an RGBA buffer as a strip at the top of the screen.
        rgba_bytes must be vertically flipped (GL bottom-up origin);
        None redraws the texture uploaded last time.
        """
        if rgba_bytes is None and not self.has_overlay(width, height):
            return
        if not self.has_overlay(width, height):
            if self._overlay_tex is not None:
                self._overlay_tex.release()
            self._overlay_tex = self.ctx.texture((width, height), 4)
            self._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._overlay_h = height

            # Quad covering the top `height` px of the screen (NDC), uv 0..1.
            top = 1.0
            bot = 1.0 - 2.0 * height / (self.h * self.dpr)   # height in px
            #            x,    y,   u,   v
            quad = np.array([
                -1.0, bot, 0.0, 0.0,
                 1.0, bot, 1.0, 0.0,
                -1.0, top, 0.0, 1.0,
                 1.0, top, 1.0, 1.0,
            ], dtype="f4")
            if self._overlay_vbo is not None:
                self._overlay_vbo.release()
                self._overlay_vao.release()
            self._overlay_vbo = self.ctx.buffer(quad.tobytes())
            self._overlay_vao = self.ctx.vertex_array(
                self.overlay_prog,
                [(self._overlay_vbo, "2f 2f", "in_pos", "in_uv")],
            )

        if rgba_bytes is not None:
            self._overlay_tex.write(rgba_bytes)

        self._target.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._overlay_tex.use(0)
        self.overlay_prog["u_tex"].value = 0
        self._overlay_vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    # ------------------------------------------------------------------
    # Internal

    def _render_fullscreen(self, prog):
        # one cached empty VAO per program (3 verts from gl_VertexID)
        vao = self._fs_vaos.get(id(prog))
        if vao is None:
            vao = self._fs_vaos[id(prog)] = self.ctx.vertex_array(prog, [])
        vao.render(mode=moderngl.TRIANGLES, vertices=3)
