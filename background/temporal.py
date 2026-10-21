"""Accumulation temporelle du fond (T1.16, passe 5), sans GL.

Le fond est calculé à une fraction de la résolution (⅓ par défaut), avec un
rayon par pixel. Chaque frame, ce rayon est décalé d'une fraction de pixel
(suite de Halton) ; la passe de résolution (`shaders/resolve.glsl`) remet
les échantillons à la pleine résolution et les mélange à l'historique
reprojeté : quand l'image bouge peu, elle converge vers la pleine
résolution, sans escaliers ni zébrures.

Ce module contient ce que le shader de résolution applique, écrit ici pour
être testé (le GLSL reprend les mêmes formules) :
- la suite de décalages (`jitter`) ;
- la reprojection d'un point : le décor est fixe dans l'espace des formes
  (`ps = (R·p + offset) / scale`), l'objet y voyage, la caméra tourne : un
  point touché cette frame est ramené là où il était la frame d'avant ;
- les règles de rejet de l'historique (surface disparue du voisinage,
  couleur écrêtée) et le mélange pondéré (`accumulate`) ;
- le poids maximal de l'historique selon « Lissage temporel » (le
  relief est fixe : tout ce qui bouge est reprojeté) ;
- `History` : la pose de la frame précédente et les remises à zéro.
"""

from dataclasses import dataclass

import numpy as np

CYCLE = 16                 # décalages de Halton avant de reboucler
MAX_HISTORY = 32.0         # poids maximal de l'historique (frames)
NO_HIT = 60000.0           # profondeur « rien touché » (main.glsl)
MASKED = -1.0              # pixel arrêté par le modèle : pas du décor
HISTORY_RATIO = 1.5        # résolution de l'historique / résolution du fond
DEPTH_TOL = (0.03, 0.08)   # écart relatif de profondeur : accepté → rejeté
CLIP_K = 2.0               # perte de poids quand la couleur est écrêtée


def halton(index: int, base: int) -> float:
    """Terme `index` (≥ 1) de la suite de Halton en base `base`, dans [0, 1)."""
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


def jitter(frame: int) -> tuple:
    """Décalage du rayon à la frame `frame`, en pixels du fond, dans
    [−0,5 ; 0,5)² ; centré en moyenne sur un cycle."""
    i = frame % CYCLE + 1
    return halton(i, 2) - 0.5, halton(i, 3) - 0.5


def history_size(bg_size, out_size) -> tuple:
    """Taille de l'historique : 1,5 × celle du fond, au plus la taille de
    rendu. À la pleine résolution, la passe de résolution coûtait 7–8 ms en
    Retina (M3) ; à 1,5 × le fond (½ de la résolution interne en ⅓), le
    décalage remplit encore les 1,5 × 1,5 sous-pixels, pour ¼ du coût."""
    return tuple(min(int(o), max(8, int(round(b * HISTORY_RATIO))))
                 for b, o in zip(bg_size, out_size, strict=True))


# ── poids maximal de l'historique ─────────────────────────────────────
def max_weight(smoothing: float) -> float:
    """Poids maximal de l'historique (≈ nombre de frames mélangées) ;
    0 = pas d'historique (ni décalage)."""
    s = min(max(float(smoothing), 0.0), 1.0)
    if s <= 0.0:
        return 0.0
    return 1.0 + (MAX_HISTORY - 1.0) * s * s


# ── règles de rejet et mélange (mêmes formules que resolve.glsl) ──────
def is_hit(d):
    return (np.asarray(d) >= 0.0) & (np.asarray(d) < NO_HIT * 0.5)


def depth_confidence(expected, stored):
    """Confiance dans l'historique selon la profondeur : `expected` =
    profondeur, à la frame d'avant, du point reprojeté ; `stored` = celle
    que l'historique a gardée. Deux ratés = 1 ; touché contre raté, ou
    pixel masqué = 0 ; deux touchés : selon l'écart relatif."""
    e, s = np.asarray(expected, dtype=float), np.asarray(stored, dtype=float)
    he, hs = is_hit(e), is_hit(s)
    miss = (e >= NO_HIT * 0.5) & (s >= NO_HIT * 0.5)
    rel = np.abs(e - s) / np.maximum(np.maximum(e, s), 1e-6)
    lo, hi = DEPTH_TOL
    x = np.clip((rel - lo) / (hi - lo), 0.0, 1.0)
    both = 1.0 - x * x * (3.0 - 2.0 * x)
    return np.where(miss, 1.0, np.where(he & hs, both, 0.0))


def motion_depth(neighbours) -> float:
    """Profondeur utilisée pour reprojeter un pixel : la surface la plus
    proche parmi ses voisins (NO_HIT s'il n'y a que du ciel). Au bord d'une
    forme, le pixel suit la forme, pas le ciel : sinon un liseré traîne du
    côté où elle s'éloigne."""
    hits = [float(d) for d in neighbours if is_hit(d)]
    return min(hits, default=NO_HIT)


def history_confidence(stored, neighbours, shift: float = 0.0) -> float:
    """Confiance dans l'historique d'un pixel : sa surface (profondeur
    `stored`, décalée de `shift` = changement de profondeur du point
    reprojeté entre les deux frames) existe-t-elle encore parmi les
    échantillons voisins de la frame (masqués exclus) ? Comparer au seul
    échantillon le plus proche rejetterait tous les contours : il change de
    côté à chaque décalage."""
    s = float(stored)
    if is_hit(s):
        s += shift
    valid = [d for d in neighbours if d >= 0.0]
    return max((float(depth_confidence(d, s)) for d in valid), default=0.0)


def clip_confidence(clip_dist):
    """Confiance qui reste à un historique écrêté : `clip_dist` = longueur
    de l'écrêtage rapportée à l'écart type du voisinage (0 = dans la boîte)."""
    return 1.0 / (1.0 + CLIP_K * np.maximum(np.asarray(clip_dist, dtype=float), 0.0))


def accumulate(hist, w_hist, cur, k, w_max):
    """Mélange de l'historique (poids `w_hist`) et de l'échantillon courant
    (poids `k`, ~1 quand un rayon est tombé au centre du pixel) ; rend
    (couleur, nouveau poids). Sans historique (w_max = 0 ou rejeté), c'est
    l'échantillon courant."""
    hist, cur = np.asarray(hist, dtype=float), np.asarray(cur, dtype=float)
    w_hist = min(max(float(w_hist), 0.0), w_max)
    if w_hist <= 0.0:
        return cur, min(k, w_max)
    out = (hist * w_hist + cur * k) / (w_hist + k)
    return out, min(w_hist + k, w_max)


# ── reprojection ──────────────────────────────────────────────────────
@dataclass
class Pose:
    """Ce qu'il faut pour savoir où un point du décor se voit : caméra du
    fond (vp = P·V, œil et axe de visée en px du monde), échelle `unit`
    (px par unité du fond), Éloignement `far`, et placement du décor
    (`rot`, `offset`, `scale` : espace des formes = (rot·p + offset) / scale)."""
    vp: np.ndarray
    eye: np.ndarray
    fwd: np.ndarray
    unit: float
    far: float
    rot: np.ndarray
    offset: np.ndarray
    scale: float

    def origin(self) -> np.ndarray:
        """Départ des rayons (unités du fond) : l'Éloignement rapproche
        l'œil du centre."""
        return self.eye / self.unit * (1.0 - self.far)

    def ray(self, uv) -> tuple:
        """Rayon (départ, direction) du pixel `uv` (0–1), comme main.glsl."""
        w = np.linalg.inv(self.vp) @ np.array([uv[0] * 2 - 1, uv[1] * 2 - 1, 1.0, 1.0])
        rd = w[:3] / w[3] / self.unit - self.eye / self.unit
        return self.origin(), rd / np.linalg.norm(rd)

    def project(self, p) -> tuple:
        """Point du monde (unités du fond) → (uv, profondeur de visée en px)."""
        p = np.asarray(p, dtype=float)
        rel = p - self.origin()
        c = self.vp @ np.append((self.eye / self.unit + rel) * self.unit, 1.0)
        uv = (c[:2] / c[3]) * 0.5 + 0.5
        return uv, float(rel @ self.fwd * self.unit)

    def project_dir(self, rd) -> np.ndarray:
        """Direction (point à l'infini : ciel, brume) → uv."""
        c = self.vp @ np.append(np.asarray(rd, dtype=float), 0.0)
        return (c[:2] / c[3]) * 0.5 + 0.5

    def to_shapes(self, p) -> np.ndarray:
        return (self.rot @ np.asarray(p, dtype=float) + self.offset) / self.scale

    def from_shapes(self, ps) -> np.ndarray:
        return self.rot.T @ (np.asarray(ps, dtype=float) * self.scale - self.offset)


def point_at(pose: Pose, uv, depth: float) -> np.ndarray:
    """Point touché par le rayon du pixel `uv` à la profondeur de visée
    `depth` (px) — ce que l'alpha du fond enregistre."""
    ro, rd = pose.ray(uv)
    view = max(float(rd @ pose.fwd), 1e-3)
    return ro + rd * depth / (view * pose.unit)


def carry(cur: Pose, prev: Pose) -> tuple:
    """(M, b) : un point du monde de cette frame → le même point du décor à
    la frame précédente, `p' = M·p + b`. Calculé en float64 : le décalage du
    voyage grandit sans fin, le soustraire en float32 dans le shader
    perdrait la précision."""
    k = prev.scale / cur.scale
    m = prev.rot.T @ (k * cur.rot)
    b = prev.rot.T @ (k * np.asarray(cur.offset, dtype=float) - prev.offset)
    return m, b


def reproject(cur: Pose, prev: Pose, uv, depth: float) -> tuple:
    """Où le point vu en `uv` (profondeur `depth`, NO_HIT = ciel) se
    trouvait à la frame précédente : (uv, profondeur attendue en px)."""
    if depth >= NO_HIT * 0.5:
        _, rd = cur.ray(uv)
        return prev.project_dir(rd), NO_HIT
    m, b = carry(cur, prev)
    return prev.project(m @ point_at(cur, uv, depth) + b)


class History:
    """Pose de la frame précédente, numéro de frame (décalage) et remises
    à zéro : l'historique est jeté quand la clé change (taille, résolution,
    qualité, même espace, structure du décor) ou après une frame sans fond."""

    def __init__(self):
        self.key = None
        self.prev = None
        self.frame = 0

    def step(self, key, pose: Pose) -> tuple:
        """Rend (pose précédente ou None si l'historique est à jeter,
        numéro de frame)."""
        prev = self.prev if key == self.key else None
        self.key, self.prev = key, pose
        self.frame += 1
        return prev, self.frame

    def forget(self):
        """Le fond n'a pas été dessiné : la prochaine frame repart de zéro."""
        self.key = self.prev = None


def resolve_uniforms(cur: Pose, prev: Pose) -> dict:
    """Uniforms de resolve.glsl (matrices en colonne-major pour GL) ; sans
    pose précédente, la reprojection est l'identité (l'historique est
    jeté de toute façon)."""
    prev = cur if prev is None else prev
    m, b = carry(cur, prev)

    def mat(a):
        return tuple(float(v) for v in np.asarray(a, dtype=float).T.flatten())

    def vec(a):
        return tuple(float(v) for v in a)

    return {
        "u_c_inv_vp": mat(np.linalg.inv(cur.vp)), "u_c_eye": vec(cur.eye),
        "u_c_fwd": vec(cur.fwd), "u_c_unit": float(cur.unit), "u_c_far": float(cur.far),
        "u_p_vp": mat(prev.vp), "u_p_eye": vec(prev.eye), "u_p_fwd": vec(prev.fwd),
        "u_p_unit": float(prev.unit), "u_p_far": float(prev.far),
        "u_carry_m": mat(m), "u_carry_b": vec(b),
    }


def gaussian(d2, sigma: float):
    """Poids d'un échantillon à la distance² `d2` (pixels du fond)."""
    return np.exp(-np.asarray(d2, dtype=float) / (2.0 * sigma * sigma))


SIGMA_NARROW = 0.3   # échantillon qui compte pour l'historique (pixels du fond)
SIGMA_WIDE = 0.6     # agrandissement seul (sans historique)
