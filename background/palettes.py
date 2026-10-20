"""Palettes du fond (T16), calculées en numpy puis envoyées au shader comme
une petite texture 1D (`SIZE` couleurs, linéaires).

Le shader lit la palette d'un bout à l'autre, sans miroir ni répétition
(passe 3 : la répétition dessinait des bandes « vieux graphisme »). La
teinte globale (décalage, cycle, musique, harmonie) n'est pas cuite dans
la texture : le shader la fait tourner chaque frame, la texture n'est
renvoyée que si la palette change.
"""

import numpy as np

SIZE = 256

# palettes cosinus d'Inigo Quilez : a + b·cos(2π(c·t + d))
COSINE = [
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.0, 0.33, 0.67)),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.0, 0.10, 0.20)),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.3, 0.20, 0.20)),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 0.5), (0.8, 0.90, 0.30)),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 0.7, 0.4), (0.0, 0.15, 0.20)),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (2.0, 1.0, 0.0), (0.5, 0.20, 0.25)),
    ((0.8, 0.5, 0.4), (0.2, 0.4, 0.2), (2.0, 1.0, 1.0), (0.0, 0.25, 0.25)),
]
# décalage de teinte visé par chaque harmonie (Libre = aucun calage)
HARMONY_OFFSET = [None, 0.0, 0.5, 1.0 / 3.0, 1.0 / 12.0]


def hsv(h, s, v) -> np.ndarray:
    """HSV → RGB vectorisé (h en tours, v peut dépasser 1 : HDR)."""
    h = np.asarray(h, dtype=np.float64) % 1.0
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    v = np.asarray(v, dtype=np.float64)
    k = (np.array([5.0, 3.0, 1.0]) + h[..., None] * 6.0) % 6.0
    return v[..., None] - v[..., None] * s[..., None] * np.clip(
        np.minimum(k, 4.0 - k), 0.0, 1.0)


def _ramp(t, stops) -> np.ndarray:
    """Interpolation linéaire entre des couleurs régulièrement espacées."""
    stops = np.asarray(stops, dtype=np.float64)
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return stops[i] * (1.0 - f) + stops[i + 1] * f


def build(mode: int, preset: int, a, b, c) -> np.ndarray:
    """(SIZE, 3) couleurs en espace écran ; a, b, c = (teinte, sat, lum)."""
    t = np.linspace(0.0, 1.0, SIZE)
    ca, cb, cc = (hsv(*col) for col in (a, b, c))
    if mode == 0:                                   # Cosinus
        pa, pb, pc, pd = (np.array(v) for v in COSINE[preset % len(COSINE)])
        return pa + pb * np.cos(2.0 * np.pi * (pc * t[:, None] + pd))
    if mode == 1:                                   # Dégradé 2 couleurs
        return _ramp(t, [ca, cb])
    if mode == 2:                                   # Dégradé 3 couleurs
        return _ramp(t, [ca, cb, cc])
    if mode == 3:                                   # Monochrome (teinte de A)
        return _ramp(t, [hsv(a[0], a[1], 0.02), ca, hsv(a[0], a[1] * 0.3, 1.2)])
    if mode == 4:                                   # Spectre
        return hsv(a[0] + t, np.full_like(t, max(a[1], 0.6)), np.full_like(t, 1.0))
    if mode == 5:                                   # Néon : noir, pics saturés
        return _ramp(t, [(0.0, 0.0, 0.0), hsv(a[0], 1.0, 1.2),
                         (0.0, 0.0, 0.0), hsv(b[0], 1.0, 1.2)])
    if mode == 6:                                   # Pastel
        return hsv(a[0] + t * 0.5, np.full_like(t, 0.35), np.full_like(t, 0.95))
    return _ramp(t, [(0.0, 0.0, 0.0), (0.6, 0.02, 0.0), (1.0, 0.35, 0.0),  # Thermique
                     (1.0, 0.85, 0.2), (1.2, 1.2, 1.1)])


def to_linear(rgb) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, None), 2.2).astype("f4")


def mean_hue(rgb) -> float:
    """Teinte moyenne d'une palette (moyenne circulaire pondérée par la
    saturation), en tours ; 0 pour une palette grise."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    chroma = mx - mn
    r, g, b = rgb.T
    safe = np.where(chroma > 1e-6, chroma, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6.0,
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) / 6.0
    ang = 2.0 * np.pi * h
    x, y = (chroma * np.cos(ang)).sum(), (chroma * np.sin(ang)).sum()
    if x * x + y * y < 1e-12:
        return 0.0
    return float(np.arctan2(y, x) / (2.0 * np.pi) % 1.0)


def harmony_shift(harmony: int, palette_hue: float, model_hue: float) -> float:
    """Rotation de teinte (tours) qui cale la palette sur la teinte du rim
    du modèle selon l'harmonie choisie ; 0 en « Libre »."""
    offset = HARMONY_OFFSET[harmony] if 0 <= harmony < len(HARMONY_OFFSET) else None
    if offset is None:
        return 0.0
    return (model_hue + offset - palette_hue) % 1.0


def kelvin(temp: float) -> np.ndarray:
    """Couleur d'un corps noir (approximation de Tanner Helland), écran 0–1."""
    t = min(max(temp, 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19.0 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60.0) ** -0.1332047592
        g = 288.1221695283 * (t - 60.0) ** -0.0755148492
        b = 255.0
    return np.clip(np.array([r, g, b]) / 255.0, 0.0, 1.0)


def tint(amount: float) -> np.ndarray:
    """Filtre de température des couleurs : −1 froid … +1 chaud, luminance
    constante (linéaire)."""
    rgb = to_linear(kelvin(6500.0 - 3500.0 * float(amount))).astype(np.float64)
    return rgb / max(float(rgb @ np.array([0.2126, 0.7152, 0.0722])), 1e-6)


LUMA = np.array([0.2126, 0.7152, 0.0722])


def hue_rotate(rgb, angle: float) -> np.ndarray:
    """Rotation de teinte autour de l'axe des gris, comme hue_rotate() du
    shader (main.glsl)."""
    k = np.full(3, 1.0 / np.sqrt(3.0))
    c = np.asarray(rgb, dtype=np.float64)
    ca = np.cos(angle)
    return c * ca + np.cross(k, c) * np.sin(angle) + k * (c @ k) * (1.0 - ca)


def fog_color(horizon, palette_linear, amount: float, hue_angle: float) -> np.ndarray:
    """Couleur de l'horizon et de la brume : celle réglée, mêlée (amount)
    à la couleur moyenne de la palette tournée comme les formes, à la même
    luminance : le lointain se fond dans une brume de la couleur du décor."""
    horizon = np.asarray(horizon, dtype=np.float64)
    mean = np.maximum(hue_rotate(np.asarray(palette_linear).mean(axis=0), hue_angle), 0.0)
    tinted = mean * (horizon @ LUMA) / max(float(mean @ LUMA), 1e-4)
    return horizon + (tinted - horizon) * float(amount)
