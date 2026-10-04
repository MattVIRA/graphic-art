"""
Procedural matcaps (T19b) — numpy only, no image files (*.png is gitignored).

A matcap is a lit sphere seen from the front: the surface shader looks it up
with the view-space normal (uv = n.xy * 0.5 + 0.5), which gives a studio look
without any light setup. Values may exceed 1 (HDR highlights feed the bloom).
Row 0 of each array is v = 0, i.e. the BOTTOM of the sphere (GL convention).
"""

import numpy as np

NAMES = ["Argile", "Chrome", "Perle", "Néon", "Or"]
SIZE = 256


def _normals(size=SIZE):
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(t, t)
    r = np.sqrt(x * x + y * y)
    # outside the disc: clamp onto the silhouette (texture filtering at the rim)
    k = np.where(r > 0.999, 0.999 / np.maximum(r, 1e-6), 1.0)
    x, y = x * k, y * k
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    return x, y, z


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _lambert(n, light):
    x, y, z = n
    lx, ly, lz = _unit(light)
    return np.clip(x * lx + y * ly + z * lz, 0.0, None)


def _reflect(n):
    """Reflection of the view vector (0,0,1) about the normal."""
    x, y, z = n
    return 2.0 * z * x, 2.0 * z * y, 2.0 * z * z - 1.0


def _spec(n, light, power):
    rx, ry, rz = _reflect(n)
    lx, ly, lz = _unit(light)
    return np.clip(rx * lx + ry * ly + rz * lz, 0.0, None) ** power


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rgb(*channels):
    return np.stack(channels, axis=-1).astype(np.float32)


def _studio_env(n):
    """Bright sky / dark ground with a crisp horizon, seen in reflection."""
    _, ry, _ = _reflect(n)
    sky = 0.55 + 0.6 * np.clip(ry, 0.0, 1.0)
    ground = 0.06 + 0.05 * np.clip(-ry, 0.0, 1.0)
    h = _smoothstep(-0.03, 0.03, ry)
    return ground * (1.0 - h) + sky * h


def _argile(n):
    key = (-0.4, 0.6, 0.7)
    d = 0.22 + 0.78 * _lambert(n, key)
    s = 0.10 * _spec(n, key, 18.0)
    base = (0.80, 0.73, 0.66)
    return _rgb(*(base[i] * d + s for i in range(3)))


def _chrome(n):
    env = _studio_env(n)
    s = 2.6 * _spec(n, (-0.3, 0.7, 0.65), 90.0)
    return _rgb(env * 0.92 + s, env * 0.96 + s, env * 1.05 + s)


def _perle(n):
    x, y, z = n
    hue = (0.55 + 0.35 * (1.0 - z) + 0.10 * x) % 1.0
    sat, val = 0.35, 0.62 + 0.38 * _lambert(n, (-0.3, 0.5, 0.8))
    h6 = hue * 6.0
    c = val * sat
    xx = c * (1.0 - np.abs(h6 % 2.0 - 1.0))
    m = val - c
    i = np.floor(h6).astype(int) % 6
    zero = np.zeros_like(c)
    r = np.choose(i, [c, xx, zero, zero, xx, c]) + m
    g = np.choose(i, [xx, c, c, xx, zero, zero]) + m
    b = np.choose(i, [zero, zero, xx, c, c, xx]) + m
    s = 0.7 * _spec(n, (-0.3, 0.6, 0.75), 30.0)
    return _rgb(r + s, g + s, b + s)


def _neon(n):
    x, y, z = n
    fres = (1.0 - z) ** 2.5
    t = (y + 1.0) * 0.5                        # bottom magenta → top cyan
    rim = _rgb(1.0 - 0.9 * t, 0.2 + 0.7 * t, 0.8 + 0.2 * t) * 1.9
    body = 0.03 + 0.10 * _lambert(n, (0.0, 0.6, 0.8))
    return rim * fres[..., None] + _rgb(body * 0.4, body * 0.4, body)


def _or(n):
    gold = (1.0, 0.76, 0.33)
    env = _studio_env(n)
    d = 0.12 + 0.30 * _lambert(n, (-0.4, 0.6, 0.7))
    s = 2.2 * _spec(n, (-0.3, 0.7, 0.65), 60.0)
    return _rgb(*(gold[i] * (env * 0.8 + d) + s * (0.6 + 0.4 * gold[i])
                  for i in range(3)))


_BUILDERS = [_argile, _chrome, _perle, _neon, _or]


def build_all(size=SIZE) -> list:
    """[(size, size, 3) float32] in NAMES order."""
    n = _normals(size)
    return [np.ascontiguousarray(b(n), dtype=np.float32) for b in _BUILDERS]
