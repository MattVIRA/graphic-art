"""Bruit 3D du fond (T16, passe 2), sans GL.

Le warp de la Distorsion lit ce bruit dans une texture 3D au lieu de le
calculer à chaque pas de marche (mesuré : un bruit calculé coûtait ×8 à
×13 le treillis). Bruit blanc filtré dans le domaine de Fourier : il est
lisse, et périodique par construction (la texture se répète sans couture
en mode REPEAT). Trois canaux indépendants : un déplacement par axe.
"""

import numpy as np

SIZE = 64            # texels par côté
CELLS = 8            # bosses par côté de la texture (≈ 8 texels par bosse)
SEED = 1616


def texture(size: int = SIZE, cells: int = CELLS, seed: int = SEED) -> np.ndarray:
    """(size, size, size, 3) float32 dans [0, 1], indexé [z, y, x, canal]
    (l'ordre des octets attendu par une texture 3D GL)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size) * size                 # fréquences entières
    fz, fy, fx = np.meshgrid(f, f, f, indexing="ij")
    # passe-bas gaussien : longueur de corrélation ≈ size / cells texels
    sigma = cells * 0.6
    lowpass = np.exp(-(fx ** 2 + fy ** 2 + fz ** 2) / (2.0 * sigma ** 2))
    out = np.empty((size, size, size, 3), dtype=np.float32)
    for c in range(3):
        white = rng.standard_normal((size, size, size))
        field = np.real(np.fft.ifftn(np.fft.fftn(white) * lowpass))
        lo, hi = field.min(), field.max()
        out[..., c] = (field - lo) / (hi - lo)
    return out


def max_gradient(tex: np.ndarray) -> float:
    """Plus forte variation d'un texel au voisin (tous axes et canaux,
    raccord compris) : sert à borner l'étirement du warp."""
    g = 0.0
    for axis in range(3):
        g = max(g, float(np.abs(np.roll(tex, -1, axis=axis) - tex).max()))
    return g
