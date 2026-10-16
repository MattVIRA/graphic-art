"""
GPU renderer (moderngl).

Pipeline per frame:
  1. Fade the accumulation buffer toward the background color  → motion trails
  2. Draw particles as additive soft glowing points into accum (HDR float buffer)
  3. Bloom: downsample + separable Gaussian blur of accum
  4. Composite accum + bloom to the screen with tone mapping
  5. (optional) HUD text overlay quad

pygame owns the window + GL context; we attach moderngl to it via
moderngl.create_context().
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

POINT_VS = """
#version 330
uniform mat4 u_mvp;
uniform float u_size_scale;
in vec3 in_pos;      // model-space coords (centered, y up, pixel units)
in vec3 in_color;
in float in_size;    // base point size in px
out vec3 v_color;
void main() {
    vec4 clip = u_mvp * vec4(in_pos, 1.0);
    gl_Position = clip;
    // perspective size scaling: closer particles are bigger
    gl_PointSize = max(1.0, in_size * u_size_scale / max(clip.w, 0.001));
    v_color = in_color;
}
"""

POINT_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    // soft round falloff (gaussian-ish) → glowing dot
    vec2 c = gl_PointCoord - vec2(0.5);
    float d2 = dot(c, c);
    float a = exp(-d2 * 14.0);
    f_color = vec4(v_color * a, a);   // premultiplied; drawn with additive blend
}
"""

# ── T1: stateless GPU particle physics ───────────────────────────────────────
# Static per-particle attributes are uploaded ONCE; every frame only a handful
# of audio-feature uniforms change and the vertex shader recomputes the whole
# physics (displacement, color, size). No CPU work, no per-frame upload.
#
# Attribute layout (11 floats):
#   in_pos   vec3 — rest position (image xy,z | curve xy,z | star direction)
#   in_color vec3 — base color (unused for star: computed from hue in-shader)
#   in_aux   vec4 — curves: normal.xy, t, band | star: t, band, 0, 0
#                   image points: edge, 0, 0, 0 | model: normal.xyz, band
#   in_fg    f    — 1 = cut-out subject, 0 = background

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
uniform int   u_mode;        // 0 image points, 1 curves, 2 star, 3 3D model
uniform float u_time;
uniform float u_angle;
uniform float u_turb;
uniform float u_hide_bg;
uniform float u_bass;
uniform float u_low_mid;
uniform float u_mid;
uniform float u_rms;
uniform float u_centroid;
uniform float u_hz10;        // 10-11 kHz  → turbulence amount
uniform float u_hz11;        // 11-12 kHz  → layout stretch
uniform float u_hz12;        // 12-14 kHz  → shimmer
uniform float u_depth;       // relief/épaisseur : scales the points' z range
uniform float u_edge;        // outline emphasis (0 = off)
uniform float u_inflate;     // model: inflation along the normal (Gonflement)
uniform float u_mturb;       // model: routed turbulence (wave + drift)
uniform float u_model_gain;  // model: brightness compensation for point density
uniform float u_rim;         // model: fresnel rim intensity
uniform vec3  u_rim_color;
// per-scope color grading: subject (fg) vs background (bg). Neutral when the
// corresponding slider targets the whole frame (applied in the composite).
uniform vec3  u_tint_fg;     // rgb gain × brightness
uniform vec3  u_tint_bg;
uniform float u_con_fg;      // contrast
uniform float u_con_bg;
uniform float u_neg_fg;      // invert mix
uniform float u_neg_bg;
// per-scope routed effects (Physique tab: Objet / Décor)
uniform float u_szm_fg;      // point-size multiplier
uniform float u_szm_bg;
uniform float u_brm_fg;      // brightness multiplier
uniform float u_brm_bg;
uniform float u_tbm_fg;      // turbulence multiplier
uniform float u_tbm_bg;
uniform float u_spectrum[64];

in vec3  in_pos;
in vec3  in_color;
in vec4  in_aux;
in float in_fg;
out vec3 v_color;

//%NOISE%
//%MODEL%

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 apply_contrast(vec3 c, float m) {
    return clamp((c - 0.5) * (1.0 + m * 1.3) + 0.5, 0.0, 1.0);
}

void main() {
    if (u_hide_bg > 0.5 && u_mode != 2 && in_fg < 0.5) {
        gl_Position = vec4(4.0, 4.0, 4.0, 1.0);   // outside clip volume → culled
        gl_PointSize = 0.0;
        v_color = vec3(0.0);
        return;
    }

    vec3 pos;
    vec3 col;
    float size;

    if (u_mode == 2) {                       // ── star / sea-urchin
        int band  = int(in_aux.y + 0.5);
        float gain = u_spectrum[band];
        float t    = in_aux.x;
        float radius = 70.0 * (1.0 + u_bass * 0.7) + t * (60.0 + gain * 260.0);
        pos = in_pos * radius;
        float hue = fract(float(band) / 64.0 + u_centroid * 0.3);
        float val = clamp(0.45 + gain * 0.65 + t * 0.1, 0.0, 1.0);
        col  = apply_contrast(hsv2rgb(vec3(hue, 0.9, val)), u_mid);
        size = 2.0 + t * 1.5 + u_bass * 3.0 + u_rms * 1.5;

    } else if (u_mode == 1) {                // ── curves (abstract line art)
        int band   = int(in_aux.w + 0.5);
        float wave = u_spectrum[band];
        float phase = in_aux.z * 34.0 + u_angle * 6.0 + float(band) * 0.7;
        float disp  = sin(phase) * (3.0 + wave * 42.0 * u_turb);
        pos = vec3(in_pos.xy + in_aux.xy * disp,
                   in_pos.z * u_depth);
        col  = apply_contrast(clamp(in_color * (0.40 + wave * 0.55 + u_rms * 0.15),
                                    0.0, 1.0), u_mid * 0.5);
        size = 1.3 + wave * 0.8;

    } else if (u_mode == 3) {                // ── 3D model, points on the surface
        // no built-in audio reaction: all animation comes from the routing
        vec3 n = in_aux.xyz;
        float g = u_spectrum[int(in_aux.w + 0.5)];
        pos = model_displace(in_pos, n, g);
        // light the cloud like a surface: points facing the camera are
        // brighter, grazing ones get the fresnel rim → reads as a 3D volume
        float facing = abs(normalize(mat3(u_model) * n).z);
        col = in_color * (0.22 + 0.40 * facing);
        col += u_rim_color * pow(1.0 - facing, 3.0) * u_rim * 0.35;
        col *= u_model_gain;
        size = 1.6;

    } else {                                 // ── image points
        float spread = 1.0 + u_hz11 * 0.08 * u_turb;
        // traveling wave across the image (replaces most of the random drift)
        float ph = (in_pos.x + in_pos.y * 0.6) * 0.010 - u_time * 2.2;
        float tb = in_fg > 0.5 ? u_tbm_fg : u_tbm_bg;
        float wv = sin(ph) * (4.0 + u_rms * 30.0 + u_hz10 * 24.0) * u_turb * tb;
        float amp = u_hz10 * (6.0 + u_rms * 12.0) * u_turb * tb
                    * (in_fg > 0.5 ? 0.5 : 1.4);
        vec3 drift = flownoise(in_pos * 0.012 + vec3(u_time * 0.30)) * amp * 0.5;
        pos = vec3(in_pos.x * spread,
                   in_pos.y * spread + wv * 0.35,
                   in_pos.z * u_depth + wv) + drift;
        col = apply_contrast(in_color, u_mid * 0.6);
        if (u_hz12 > 0.25) {                 // shimmer, re-seeded ~30×/s
            float sh = hash11(float(gl_VertexID) + floor(u_time * 30.0) * 0.618);
            col *= 1.0 - u_hz12 * 0.4 * sh;
        }
        col *= (in_fg > 0.5) ? 0.50 + u_bass * 0.22 : 0.26 + u_bass * 0.10;
        // in_aux.x = edge flag: outline points are brighter and bigger
        col *= 1.0 + in_aux.x * 0.9 * u_edge;
        size = 1.5 * (in_fg > 0.5 ? 1.25 : 0.85)
               * (1.0 + in_aux.x * 0.6 * u_edge);
    }

    // per-scope color grading (star mode counts as subject)
    {
        bool fg = (u_mode == 2) || (in_fg > 0.5);
        col *= fg ? u_tint_fg : u_tint_bg;
        float cc = fg ? u_con_fg : u_con_bg;
        col = clamp((col - 0.5) * cc + 0.5, 0.0, 1.0);
        float ng = fg ? u_neg_fg : u_neg_bg;
        col = mix(col, vec3(1.0) - col, ng);
        // routed effects targeted at the subject / background only
        col *= fg ? u_brm_fg : u_brm_bg;
        size *= fg ? u_szm_fg : u_szm_bg;
    }

    vec4 clip = u_mvp * vec4(pos, 1.0);
    gl_Position = clip;
    gl_PointSize = max(1.0, size * u_size_mult * u_size_scale / max(clip.w, 0.001));
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
out vec3  v_normal;
out vec3  v_color;
out vec2  v_uv;

//%NOISE%
//%MODEL%

void main() {
    float g = u_spectrum[int(in_band + 0.5)];
    vec3 p = model_displace(in_pos, in_normal, g);
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(p, 1.0);
    gl_Position.z += u_depth_bias * gl_Position.w;
}
""")

# Shared tail: scoped color grading (the model is always "objet").
_MESH_GRADE_GLSL = """
uniform vec3  u_tint_fg;
uniform float u_con_fg;
uniform float u_neg_fg;
uniform float u_brm_fg;
vec3 grade(vec3 col) {
    col *= u_tint_fg;
    col = max((col - 0.5) * u_con_fg + 0.5, 0.0);
    col = mix(col, max(vec3(1.0) - col, 0.0), u_neg_fg);
    return col * u_brm_fg;
}
"""

SURFACE_FS = """
#version 330
uniform sampler2D u_matcap;
uniform sampler2D u_tex;      // model texture (sampled per pixel via uv)
uniform float u_rim;
uniform vec3  u_rim_color;
uniform float u_orig;         // 0 = pure matcap, 1 = model colors lit by it
uniform float u_alpha;        // 1 = opaque, <1 = transparent (additive)
in vec3  v_normal;
in vec3  v_color;
in vec2  v_uv;
out vec4 f_color;
""" + _MESH_GRADE_GLSL + """
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;          // open meshes: light the inside too
    vec3 mc = texture(u_matcap, n.xy * 0.495 + 0.5).rgb;
    float lum = dot(mc, vec3(0.299, 0.587, 0.114));
    vec3 base = v_uv.x < -10.0 ? v_color : texture(u_tex, v_uv).rgb;
    vec3 col = mix(mc, base * lum * 1.6, u_orig) * 1.1;
    float fres = pow(1.0 - max(n.z, 0.0), 3.0);
    col += u_rim_color * fres * u_rim;
    f_color = vec4(grade(col) * u_alpha, 1.0);
}
"""

WIRE_FS = """
#version 330
uniform float u_rim;
uniform vec3  u_rim_color;
uniform float u_wire_gain;    // per-model: dense meshes are dimmed
in vec3  v_normal;
in vec3  v_color;
out vec4 f_color;
""" + _MESH_GRADE_GLSL + """
void main() {
    float facing = abs(normalize(v_normal).z);
    vec3 col = v_color * (0.45 + 0.55 * facing);
    col += u_rim_color * pow(1.0 - facing, 2.0) * u_rim * 0.5;
    f_color = vec4(grade(col) * u_wire_gain, 1.0);
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

FADE_FS = """
#version 330
uniform vec3 u_bg;
uniform float u_fade;   // 0..1, fraction pulled toward bg each frame
out vec4 f_color;
void main() {
    f_color = vec4(u_bg, u_fade);   // normal alpha blend → accum*(1-f) + bg*f
}
"""

BLUR_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_dir;     // (1/w, 0) or (0, 1/h)
in vec2 v_uv;
out vec4 f_color;
void main() {
    float w[5] = float[](0.227027, 0.194595, 0.121622, 0.054054, 0.016216);
    vec3 acc = texture(u_tex, v_uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        acc += texture(u_tex, v_uv + u_dir * float(i)).rgb * w[i];
        acc += texture(u_tex, v_uv - u_dir * float(i)).rgb * w[i];
    }
    f_color = vec4(acc, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom0;   // 1/2 res — tight glow
uniform sampler2D u_bloom1;   // 1/4 res — medium halo
uniform sampler2D u_bloom2;   // 1/8 res — wide soft halo
uniform float u_bloom_intensity;
uniform float u_bloom_radius;   // 0 = tight, 1 = wide halos
uniform float u_brightness;   // pre-tonemap exposure
uniform vec3  u_gain;         // per-channel RGB gain
uniform float u_contrast;     // post contrast around 0.5
uniform float u_invert;       // 0 = normal, 1 = negative
uniform float u_grain;        // film grain amount
uniform float u_time;         // seed for the grain noise
uniform vec2 u_shift_r;       // per-channel uv offsets (chromatic shift)
uniform vec2 u_shift_g;
uniform vec2 u_shift_b;
in vec2 v_uv;
out vec4 f_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec3 scene = vec3(texture(u_scene, v_uv + u_shift_r).r,
                      texture(u_scene, v_uv + u_shift_g).g,
                      texture(u_scene, v_uv + u_shift_b).b);
    // multi-scale bloom: blend the 3 blur levels; the radius shifts the
    // energy toward the wider levels (normalized so intensity stays stable)
    float w0 = 1.0 - u_bloom_radius * 0.5;
    float w1 = u_bloom_radius * 0.9;
    float w2 = u_bloom_radius * u_bloom_radius * 1.4;
    vec3 bloom = texture(u_bloom0, v_uv).rgb * w0 +
                 texture(u_bloom1, v_uv).rgb * w1 +
                 texture(u_bloom2, v_uv).rgb * w2;
    vec3 hdr = (scene + bloom * u_bloom_intensity) * u_brightness;
    vec3 mapped = hdr / (hdr + vec3(1.0));   // Reinhard tone map
    mapped *= u_gain;
    mapped = (mapped - 0.5) * u_contrast + 0.5;
    mapped = mix(mapped, vec3(1.0) - mapped, u_invert);
    if (u_grain > 0.001) {
        float n = hash(v_uv * 971.0 + vec2(u_time, u_time * 1.7));
        mapped += (n - 0.5) * u_grain;
    }
    f_color = vec4(clamp(mapped, 0.0, 1.0), 1.0);
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
void main() { f_color = vec4(u_color, 1.0); }
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
    f_color = vec4(u_color * fade, fade);
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
    FLOATS_PER_PARTICLE = 7   # x, y, z, r, g, b, size

    def __init__(self, ctx: moderngl.Context, size: tuple, max_particles: int,
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
        self.point_prog = ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
        self.fade_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=FADE_FS)
        self.blur_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=BLUR_FS)
        self.comp_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=COMPOSITE_FS)
        self.overlay_prog = ctx.program(vertex_shader=OVERLAY_VS, fragment_shader=OVERLAY_FS)
        self.grid_prog = ctx.program(vertex_shader=GRID_VS, fragment_shader=GRID_FS)
        self.line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        # T1: stateless GPU-physics program (fallback to the CPU path if it fails)
        try:
            self.gpu_prog = ctx.program(vertex_shader=GPU_POINT_VS,
                                        fragment_shader=POINT_FS)
        except Exception as e:
            print(f"[warning] GPU particle shader failed ({e}); CPU fallback")
            self.gpu_prog = None

        # T19: 3D model surface / wireframe (GPU path only)
        self.mesh_prog = self.wire_prog = None
        if self.gpu_prog is not None:
            try:
                self.mesh_prog = ctx.program(vertex_shader=MESH_VS,
                                             fragment_shader=SURFACE_FS)
                self.wire_prog = ctx.program(vertex_shader=MESH_VS,
                                             fragment_shader=WIRE_FS)
            except Exception as e:
                print(f"[warning] mesh shaders failed ({e}); models as points only")
                self.mesh_prog = self.wire_prog = None
        # programs that receive the model transform + scoped grading uniforms
        self._model_progs = [p for p in (self.gpu_prog, self.mesh_prog,
                                         self.wire_prog) if p is not None]
        self._matcaps = []
        if self.mesh_prog is not None:
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

        self.fade_prog["u_bg"].value = self.bg
        self.set_post_params()  # initialize composite uniforms to neutral
        self.set_scoped_colors()   # GL uniforms default to 0 → set neutral
        self.set_scoped_fx()
        self.set_model_look()

        # ── Camera ──
        self.fov = np.radians(45.0)
        # distance so the model height (~h px) fills the vertical field of view
        self.cam_dist = (self.h / 2.0) / np.tan(self.fov / 2.0)
        self._proj = _perspective(self.fov, self.w / self.h, 10.0, 8000.0)
        self._view = _translate(0.0, 0.0, -self.cam_dist)
        # base point-size scale: a particle at the model origin keeps its px size
        self.point_prog["u_size_scale"].value = float(self.cam_dist)
        self.set_transform(0.0)

        # Empty VAO for fullscreen passes (3 verts from gl_VertexID)
        self._fs_vao = ctx.vertex_array(self.fade_prog, [])

        # ── Particle VBO / VAO ──
        self._vbo = ctx.buffer(reserve=max_particles * self.FLOATS_PER_PARTICLE * 4)
        # T1: lazily-sized static buffer for the GPU-physics path
        self._gpu_vbo = None
        self._gpu_vao = None
        self._gpu_mode = 0
        self._gpu_count = 0
        self._point_vao = ctx.vertex_array(
            self.point_prog,
            [(self._vbo, "3f 3f 1f", "in_pos", "in_color", "in_size")],
        )

        # ── Framebuffers ──
        # T6: internal supersampling — the scene is rendered at render_scale ×
        # the window size then downsampled at composite time (kills aliasing on
        # thin curves/points; pygame cannot request a native Retina GL drawable)
        self.render_scale = 1.5
        self.bloom_radius = 0.5      # 0 = tight glow, 1 = wide halos
        self._accum_tex = None
        self._accum_depth = None     # T19: depth for the opaque model surface
        self._accum = None
        self._bloom_tex = []         # 3 levels × ping-pong pair
        self._bloom_fbo = []
        self._bloom_sizes = []
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
            self._screen = self.ctx.screen
        return self._screen

    def _build_internal(self):
        """(Re)create the supersampled HDR accum + 3-level bloom pyramid."""
        for pair_f, pair_t in zip(self._bloom_fbo, self._bloom_tex):
            for f in pair_f:
                f.release()
            for t in pair_t:
                t.release()
        if self._accum is not None:
            self._accum.release(); self._accum_tex.release()
            self._accum_depth.release()

        rw = max(8, int(self.w * self.render_scale))
        rh = max(8, int(self.h * self.render_scale))
        self._internal_size = (rw, rh)

        self._accum_tex = self.ctx.texture((rw, rh), 4, dtype="f2")
        self._accum_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._accum_depth = self.ctx.depth_renderbuffer((rw, rh))
        self._accum = self.ctx.framebuffer(color_attachments=[self._accum_tex],
                                           depth_attachment=self._accum_depth)
        self._accum.use()
        self.ctx.clear(*self.bg, 1.0)

        # bloom pyramid: 3 levels (1/2, 1/4, 1/8 of internal res), 2 tex each
        self._bloom_tex, self._bloom_fbo, self._bloom_sizes = [], [], []
        for div in (2, 4, 8):
            bw, bh = max(4, rw // div), max(4, rh // div)
            pair = [self.ctx.texture((bw, bh), 4, dtype="f2") for _ in range(2)]
            for t in pair:
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._bloom_tex.append(pair)
            self._bloom_fbo.append(
                [self.ctx.framebuffer(color_attachments=[t]) for t in pair])
            self._bloom_sizes.append((bw, bh))

        # point sizes are expressed in internal pixels
        self.set_size_mult(getattr(self, "_size_mult", 1.0))

    def set_render_scale(self, scale: float):
        """Supersampling factor (1.0–2.0). Rebuilds the internal buffers."""
        scale = min(max(scale, 1.0), 2.0)
        if abs(scale - self.render_scale) > 1e-3:
            self.render_scale = scale
            self._build_internal()

    def resize(self, w: int, h: int):
        """Rebuild size-dependent GL state after a window resize / fullscreen."""
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = w, h
        self.cam_dist = (h / 2.0) / np.tan(self.fov / 2.0)
        self._proj = _perspective(self.fov, w / h, 10.0, 8000.0)
        self._view = _translate(0.0, 0.0, -self.cam_dist)

        # re-detect the default framebuffer (its size changed with the window)
        # and force its viewport: detect_framebuffer can report a stale size
        # inherited from the previous GL viewport.
        self._screen = self.ctx.detect_framebuffer()
        self._screen.viewport = (0, 0, w, h)

        # recreate the supersampled accum + bloom pyramid at the new size
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
        self.point_prog["u_mvp"].write(mvp_bytes)
        for prog in self._model_progs:
            prog["u_mvp"].write(mvp_bytes)
            if "u_model" in prog._members:
                prog["u_model"].write(model_bytes)

    def set_post_params(self, brightness=1.0, gain=(1.0, 1.0, 1.0), contrast=1.0,
                        invert=0.0, grain=0.0, time=0.0,
                        shift_r=(0.0, 0.0), shift_g=(0.0, 0.0), shift_b=(0.0, 0.0)):
        """Update the post-processing (composite) uniforms."""
        p = self.comp_prog
        p["u_brightness"].value = float(brightness)
        p["u_shift_r"].value = tuple(shift_r)
        p["u_shift_g"].value = tuple(shift_g)
        p["u_shift_b"].value = tuple(shift_b)
        p["u_gain"].value = tuple(float(g) for g in gain)
        p["u_contrast"].value = float(contrast)
        p["u_invert"].value = float(invert)
        p["u_grain"].value = float(grain)
        p["u_time"].value = float(time)

    def set_zoom(self, zoom: float):
        """Camera zoom: >1 gets closer, <1 pulls back."""
        z = max(zoom, 0.05)
        self._view = _translate(0.0, 0.0, -self.cam_dist / z)

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

    def set_scoped_colors(self, tint_fg=(1.0, 1.0, 1.0), tint_bg=(1.0, 1.0, 1.0),
                          con_fg=1.0, con_bg=1.0, neg_fg=0.0, neg_bg=0.0):
        """Per-scope color grading for the GPU path (subject vs background).
        Pass neutral values for sliders that target the whole frame."""
        vals = (("u_tint_fg", tuple(float(v) for v in tint_fg)),
                ("u_tint_bg", tuple(float(v) for v in tint_bg)),
                ("u_con_fg", float(con_fg)), ("u_con_bg", float(con_bg)),
                ("u_neg_fg", float(neg_fg)), ("u_neg_bg", float(neg_bg)))
        for prog in self._model_progs:
            for name, val in vals:
                if name in prog._members:
                    prog[name].value = val

    def set_scoped_fx(self, size_fg=1.0, size_bg=1.0, bright_fg=1.0,
                      bright_bg=1.0, turb_fg=1.0, turb_bg=1.0):
        """Routed Physique effects targeted at the subject / background only."""
        vals = (("u_szm_fg", size_fg), ("u_szm_bg", size_bg),
                ("u_brm_fg", bright_fg), ("u_brm_bg", bright_bg),
                ("u_tbm_fg", turb_fg), ("u_tbm_bg", turb_bg))
        for prog in self._model_progs:
            for name, val in vals:
                if name in prog._members:
                    prog[name].value = float(val)

    def set_model_look(self, matcap=0, rim=1.0, rim_hue=0.52, orig=0.5,
                       transparent=False):
        """T19: surface look of the 3D model (matcap, fresnel rim, colors)."""
        rim_color = colorsys.hsv_to_rgb(rim_hue % 1.0, 0.65, 1.0)
        for prog in self._model_progs:
            for name, val in (("u_rim", float(rim)), ("u_rim_color", rim_color),
                              ("u_orig", float(orig))):
                if name in prog._members:
                    prog[name].value = val
        self._matcap_idx = int(matcap) % max(len(self._matcaps), 1)
        self._mesh_transparent = bool(transparent)

    def set_size_mult(self, mult: float):
        """Global particle size multiplier (control-panel 'Taille points').
        Point sizes are expressed in INTERNAL (supersampled) pixels."""
        self._size_mult = float(mult)
        s = self.cam_dist * self.render_scale
        self.point_prog["u_size_scale"].value = float(s * mult)
        if self.gpu_prog is not None:
            self.gpu_prog["u_size_scale"].value = float(s)
            self.gpu_prog["u_size_mult"].value = float(mult)

    # ------------------------------------------------------------------
    # T1: stateless GPU particle path

    STATIC_FLOATS = 11    # pos3 + color3 + aux4 + fg1

    def upload_particle_static(self, data: np.ndarray, mode: int, count: int):
        """Upload the per-particle rest attributes ONCE (until the next rebuild)."""
        if self.gpu_prog is None:
            return
        needed = count * self.STATIC_FLOATS * 4
        if self._gpu_vbo is None or self._gpu_vbo.size < needed:
            if self._gpu_vbo is not None:
                self._gpu_vbo.release()
                self._gpu_vao.release()
            self._gpu_vbo = self.ctx.buffer(reserve=needed)
            self._gpu_vao = self.ctx.vertex_array(
                self.gpu_prog,
                [(self._gpu_vbo, "3f 3f 4f 1f",
                  "in_pos", "in_color", "in_aux", "in_fg")],
            )
        self._gpu_vbo.write(np.ascontiguousarray(data[:count], dtype="f4").tobytes())
        self._gpu_mode = int(mode)
        self._gpu_count = int(count)

    def draw_particles_gpu(self, uniforms: dict):
        """Draw the static buffer; all physics happens in the vertex shader."""
        if self.gpu_prog is None or self._gpu_vao is None or self._gpu_count <= 0:
            return
        prog = self.gpu_prog
        prog["u_mode"].value = self._gpu_mode
        self._write_frame_uniforms(prog, uniforms, (
            "u_time", "u_angle", "u_turb", "u_hide_bg", "u_bass",
            "u_low_mid", "u_mid", "u_rms", "u_centroid",
            "u_hz10", "u_hz11", "u_hz12", "u_depth", "u_edge", "u_inflate",
            "u_mturb", "u_model_gain"))
        self._accum.use()
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
        if m is None or self.mesh_prog is None or len(m["tris"]) == 0:
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
            # the wire program never reads uv (the compiler drops the attribute)
            uv = "2f" if "in_uv" in prog._members else "2x4"
            layout = [(vbo, f"3f 3f 3f 1f {uv}", "in_pos", "in_normal", "in_color",
                       "in_band", *(("in_uv",) if uv == "2f" else ()))]
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
        ctx, fbo, vaos = self.ctx, self._accum, self._mesh_vaos
        fbo.use()
        # clear depth only: the color buffer holds the trails
        fbo.color_mask = (False, False, False, False)
        fbo.clear(depth=1.0)
        fbo.color_mask = (True, True, True, True)
        fbo.use()     # re-applies the color mask to the GL state
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
        """Fade the accumulation buffer toward the background (trail decay)."""
        self._accum.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.fade_prog["u_fade"].value = self.fade
        self._render_fullscreen(self.fade_prog)

    def draw_particles(self, data: np.ndarray, count: int):
        """
        data: float32 array shape (count, 6) → [x, y, r, g, b, size].
        Drawn additively into the accumulation buffer.
        """
        if count <= 0:
            return
        self._accum.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)   # additive
        self._vbo.write(data[:count].astype("f4").tobytes())
        self._point_vao.render(mode=moderngl.POINTS, vertices=count)

    def end_frame(self):
        """Bloom pyramid + composite (supersampled accum → window)."""
        self.ctx.disable(moderngl.BLEND)
        self.blur_prog["u_tex"].value = 0

        # build the 3-level pyramid: downsample from the previous level,
        # then one separable Gaussian blur per level
        src_tex = self._accum_tex
        for lvl in range(3):
            pair_t, pair_f = self._bloom_tex[lvl], self._bloom_fbo[lvl]
            bw, bh = self._bloom_sizes[lvl]
            pair_f[0].use()
            src_tex.use(0)
            self.blur_prog["u_dir"].value = (0.0, 0.0)   # copy/downsample
            self._render_fullscreen(self.blur_prog)
            # horizontal then vertical blur
            pair_f[1].use()
            pair_t[0].use(0)
            self.blur_prog["u_dir"].value = (1.0 / bw, 0.0)
            self._render_fullscreen(self.blur_prog)
            pair_f[0].use()
            pair_t[1].use(0)
            self.blur_prog["u_dir"].value = (0.0, 1.0 / bh)
            self._render_fullscreen(self.blur_prog)
            src_tex = pair_t[0]      # next level downsamples this one

        # composite to the output target (downsamples the supersampled scene)
        self._target.use()
        self.ctx.disable(moderngl.BLEND)
        self._accum_tex.use(0)
        for lvl in range(3):
            self._bloom_tex[lvl][0].use(1 + lvl)
            self.comp_prog[f"u_bloom{lvl}"].value = 1 + lvl
        self.comp_prog["u_scene"].value = 0
        self.comp_prog["u_bloom_intensity"].value = self.bloom_intensity
        self.comp_prog["u_bloom_radius"].value = float(self.bloom_radius)
        self._render_fullscreen(self.comp_prog)

    # ------------------------------------------------------------------
    # HUD overlay

    def draw_overlay(self, rgba_bytes: bytes, width: int, height: int):
        """
        Draw an RGBA buffer as a strip at the top of the screen.
        rgba_bytes must be vertically flipped (GL bottom-up origin).
        """
        if self._overlay_tex is None or self._overlay_h != height \
                or self._overlay_tex.size[0] != width:
            if self._overlay_tex is not None:
                self._overlay_tex.release()
            self._overlay_tex = self.ctx.texture((width, height), 4)
            self._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._overlay_h = height

            # Quad covering the top `height` px of the screen (NDC), uv 0..1.
            top = 1.0
            bot = 1.0 - 2.0 * height / self.h
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
        # Rebind the empty VAO to the requested program and draw 3 verts.
        vao = self.ctx.vertex_array(prog, [])
        vao.render(mode=moderngl.TRIANGLES, vertices=3)
        vao.release()
