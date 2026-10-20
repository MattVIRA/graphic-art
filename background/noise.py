"""Bruit 3D du fond (T16, passe 2), sans GL.

Le warp de la Distorsion lit ce bruit dans une texture 3D au lieu de le
calculer à chaque pas de marche (mesuré : un bruit calculé coûtait ×8 à
×13 le treillis). Bruit blanc filtré dans le domaine de Fourier : il est
lisse, et périodique par construction (la texture se répète sans couture
en mode REPEAT). Trois canaux indépendants : un déplacement par axe pour
le warp, trois couches pour les mondes de bruit (passe 4).
"""

from functools import lru_cache

import numpy as np

SIZE = 64            # texels par côté
CELLS = 8            # bosses par côté de la texture (≈ 8 texels par bosse)
SEED = 1616


@lru_cache(maxsize=2)
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
    out.setflags(write=False)       # partagée (cache) : jamais modifiée
    return out


def max_gradient(tex: np.ndarray) -> float:
    """Plus forte variation d'un texel au voisin (tous axes et canaux,
    raccord compris) : sert à borner l'étirement du warp."""
    g = 0.0
    for axis in range(3):
        g = max(g, float(np.abs(np.roll(tex, -1, axis=axis) - tex).max()))
    return g


@lru_cache(maxsize=1)
def max_gradient_norm() -> float:
    """Plus forte pente (norme du gradient, par texel) d'un canal de la
    texture, raccord compris : un champ de bruit n'est pas une distance, le
    shader divise sa valeur par cette pente pour ne jamais sauter une paroi."""
    tex = texture().astype(np.float64)
    g2 = np.zeros(tex.shape)
    for axis in range(3):
        g2 += ((np.roll(tex, -1, axis=axis) - np.roll(tex, 1, axis=axis)) * 0.5) ** 2
    return float(np.sqrt(g2).max())


@lru_cache(maxsize=1)
def _sorted_centered() -> np.ndarray:
    return np.sort((texture()[..., 0].astype(np.float64) - 0.5) * 2.0, axis=None)


def centered_quantile(q: float) -> float:
    """Quantile de la couche de fond centrée (valeurs 2·(n − 0,5)) : le seuil
    qui laisse la part `1 − q` de l'espace au-dessus (la roche)."""
    values = _sorted_centered()
    i = int(round(min(max(q, 0.0), 1.0) * (values.size - 1)))
    return float(values[i])
