"""Relief du Paysage (T1.16, passe finale), précalculé en numpy, sans GL.

Le fond n'a plus qu'un générateur, le Paysage. Son relief est fait de
cartes de hauteur **périodiques** (elles se répètent sans couture : l'objet
voyage sans fin) calculées ici une fois par réglage, dans un thread, puis
lues par le shader : quelques lectures de texture par pas de marche, quel
que soit le nombre de couches. Calculer les couches à chaque pas coûtait
cher et laissait des traînées horizontales (pente du bruit déformé mal
bornée).

Deux cartes, en « tailles du monde » (la Taille du monde et la Hauteur du
relief s'appliquent dans le shader : les changer ne recalcule rien) :
- la **carte** couvre TILE tailles du monde : les grandes couches,
  déformées, érodées, en terrasses ; canaux hauteur normalisée (≈ −1…1),
  creux (> 0 dans les creux, < 0 sur les bosses : écart à la moyenne
  locale rapporté au rayon) et poids du détail (l'érosion) ;
- le **détail** couvre une taille du monde : les couches trop fines pour la
  carte (texture de près, effacée au loin par les mipmaps).
Les couches plus fines que le détail (≈ 1/250 de la taille du monde) sont
ignorées : le relief de surface du shader prend le relais.
"""

import math
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

RES = 1024              # texels par côté de la carte (le détail : la moitié)
TILE = 12.0             # la carte couvre TILE × « Taille du monde » de côté
DETAIL_TILE = 1.0       # le détail couvre une taille du monde
MIN_TEXELS = 4.0        # une couche garde au moins 4 texels par cellule de bruit
EVAL_TEXELS = 8.0       # une couche large est calculée à ≥ 8 texels par cellule, puis agrandie
SLOPE_QUANTILE = 99.5   # pente retenue : les plus raides (contremarches) sont ignorées
AO_RADII = (0.08, 0.3)  # rayons des creux mesurés (tailles du monde)
# réglages qui changent les cartes (les autres s'appliquent dans le shader)
SETTINGS = ("bg_ld_seed", "bg_ld_octaves", "bg_ld_rough", "bg_ld_lacun",
            "bg_ld_ridges", "bg_ld_erosion", "bg_ld_warp", "bg_ld_warp_size",
            "bg_ld_terrace", "bg_ld_terrace_n")
DETAIL_SETTINGS = ("bg_ld_seed", "bg_ld_octaves", "bg_ld_rough", "bg_ld_lacun",
                   "bg_ld_ridges")


@dataclass(frozen=True)
class TerrainMap:
    """Cartes calculées : `data` (res, res, 3) [hauteur, creux, poids du
    détail], `detail` (res / 2, res / 2) ; pentes par taille du monde (quantile
    99,5 %) ; `detail_gain` = facteur qui ramène le détail à l'échelle de la
    hauteur normalisée (0 = pas de détail) ; `detail_amp` = plus grand écart
    du détail, à cette échelle (sa pente ne compte que près de la surface)."""
    data: np.ndarray
    detail: np.ndarray
    slope: float
    detail_slope: float
    detail_gain: float
    detail_amp: float
    key: tuple


def settings_key(v: dict, keys=SETTINGS) -> tuple:
    """Clé des cartes : deux réglages qui donnent la même clé donnent les
    mêmes cartes (arrondis : un slider à peine bougé ne recalcule rien)."""
    return tuple((k, round(float(v[k]), 3)) for k in keys)


def layers(v: dict, res: int = RES) -> tuple:
    """Fréquences (cellules par taille du monde) et amplitudes des couches,
    réparties entre la carte (`res` texels) et le détail (`res` / 2) :
    (carte, détail), listes de (fréquence, amplitude, numéro)."""
    octaves = max(1, int(round(float(v["bg_ld_octaves"]))))
    rough, lacun = float(v["bg_ld_rough"]), float(v["bg_ld_lacun"])
    base, detail = [], []
    for i in range(octaves):
        f, amp = lacun ** i, rough ** i
        if f * TILE <= res / MIN_TEXELS:
            base.append((f, amp, i))
        elif f * DETAIL_TILE <= res / 2 / MIN_TEXELS:
            detail.append((f, amp, i))
    return base, detail


# ── bruit ─────────────────────────────────────────────────────────────
def gradients(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n + 1, n + 1) directions aléatoires en complexes (gx + i·gy), la
    dernière ligne et la dernière colonne répètent la première : la
    période est de n cellules, sans modulo sur les voisins."""
    g = np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, (n, n))).astype(np.complex64)
    return np.pad(g, ((0, 1), (0, 1)), mode="wrap")


def gradient_noise(u: np.ndarray, v: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """Bruit de gradient (Perlin) périodique ; u, v en périodes (toute
    valeur : il boucle). ≈ −1…1, float32."""
    n = grads.shape[0] - 1
    f32 = np.float32
    x = np.mod(u * f32(n), f32(n))
    y = np.mod(v * f32(n), f32(n))
    # en float32, le modulo d'un nombre à peine négatif peut rendre n tout
    # juste : la cellule est bornée (fx vaut alors 1, le bruit reste continu)
    top_cell = f32(n - 1)
    xf, yf = np.minimum(np.floor(x), top_cell), np.minimum(np.floor(y), top_cell)
    fx, fy = x - xf, y - yf
    idx = yf.astype(np.int32) * (n + 1) + xf.astype(np.int32)
    flat = grads.ravel()
    a, b = flat.take(idx), flat.take(idx + 1)
    c, d = flat.take(idx + (n + 1)), flat.take(idx + (n + 2))
    gx, gy = fx - f32(1.0), fy - f32(1.0)
    da = a.real * fx + a.imag * fy
    db = b.real * gx + b.imag * fy
    dc = c.real * fx + c.imag * gy
    dd = d.real * gx + d.imag * gy
    sx = fx * fx * fx * (fx * (fx * f32(6.0) - f32(15.0)) + f32(10.0))   # fondu quintique
    sy = fy * fy * fy * (fy * (fy * f32(6.0) - f32(15.0)) + f32(10.0))
    top = da + (db - da) * sx
    bottom = dc + (dd - dc) * sx
    return (top + (bottom - top) * sy) * f32(math.sqrt(2.0))


def upsample2(a: np.ndarray) -> np.ndarray:
    """Agrandissement ×2 périodique, bilinéaire (texels centrés)."""
    for axis in (0, 1):
        prev, nxt = np.roll(a, 1, axis), np.roll(a, -1, axis)
        even = a * 0.75 + prev * 0.25
        odd = a * 0.75 + nxt * 0.25
        a = np.stack([even, odd], axis=axis + 1)
        shape = list(even.shape)
        shape[axis] *= 2
        a = a.reshape(shape)
    return a.astype(np.float32)


def _grid(res: int) -> tuple:
    t = (np.arange(res, dtype=np.float32) + 0.5) / res
    return np.meshgrid(t, t)                       # [z, x], en périodes


def _eval_res(cells: int, res: int) -> int:
    """Résolution de calcul d'une couche : assez pour elle, au plus `res`."""
    r = 64
    while r < res and r < cells * EVAL_TEXELS:
        r *= 2
    return min(r, res)


def _ridge(n: np.ndarray) -> np.ndarray:
    """Crête vive là où le bruit passe par 0 (arêtes de montagne)."""
    r = 1.0 - np.abs(n)
    return 2.0 * r * r - 1.0


def _slope2(h: np.ndarray, tile: float) -> np.ndarray:
    """Pente² de h par taille du monde (différences centrées, carte bouclée)."""
    k = h.shape[0] / (2.0 * tile)
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * k
    gz = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * k
    return gx * gx + gz * gz


def _slope(h: np.ndarray, tile: float) -> float:
    return float(np.percentile(np.sqrt(_slope2(h, tile)[::2, ::2]), SLOPE_QUANTILE))


def _blur(spectrum: np.ndarray, res: int, sigma_texels: float) -> np.ndarray:
    """Flou gaussien périodique d'une carte dont `spectrum` = rfft2."""
    f = np.fft.fftfreq(res)
    fr = np.fft.rfftfreq(res)
    g = np.exp(-2.0 * (math.pi * sigma_texels) ** 2 * (f[:, None] ** 2 + fr[None, :] ** 2))
    return np.fft.irfft2(spectrum * g, s=(res, res)).astype(np.float32)


def _terraces(h: np.ndarray, amount: float, steps: int) -> np.ndarray:
    """Plateaux en gradins : paliers plats séparés par des contremarches
    raides (le quart du palier)."""
    if amount <= 0.0:
        return h
    x = (h + 1.0) * 0.5 * steps
    k = np.floor(x)
    f = np.clip((x - k - 0.5) * 4.0 + 0.5, 0.0, 1.0)
    stepped = (k + f * f * (3.0 - 2.0 * f)) / steps * 2.0 - 1.0
    return (h + (stepped - h) * amount).astype(np.float32)


def _octave(u, w, f: float, tile: float, ridges: float, rng) -> np.ndarray:
    cells = max(1, round(f * tile))
    n = gradient_noise(u, w, gradients(cells, rng))
    if ridges > 0.0:
        n = n + (_ridge(n) - n) * ridges
    return n


def _displacement(v: dict, res: int, rng):
    """Déformation du domaine : le bruit déplace les points où le relief est
    lu (formes liquides, vallées sinueuses), 0,3 taille du monde par unité
    de Déformation. Rend (du, dw) en périodes de la carte, ou None."""
    warp = float(v["bg_ld_warp"])
    if warp <= 0.0:
        return None
    cells = max(1, round(TILE / float(v["bg_ld_warp_size"])))
    u, w = _grid(_eval_res(cells, res))
    amp = warp * 0.3 / TILE
    return (gradient_noise(u, w, gradients(cells, rng)) * amp,
            gradient_noise(u + 0.37, w + 0.61, gradients(cells, rng)) * amp)


def _coords(r: int, disp, cache: dict) -> tuple:
    """Grille de la résolution r, déplacée par la déformation (agrandie)."""
    if r not in cache:
        u, w = _grid(r)
        if disp is not None:
            du, dw = disp
            while du.shape[0] < r:
                du, dw = upsample2(du), upsample2(dw)
            step = du.shape[0] // r
            u, w = u + du[::step, ::step], w + dw[::step, ::step]
        cache[r] = (u, w)
    return cache[r]


def _erosion_weight(h: np.ndarray, erosion: float) -> np.ndarray:
    """Érosion : les pentes et le bas des vallées reçoivent moins de détail
    (lisses), les sommets et les plats en gardent. h : hauteur ≈ −1…1."""
    low = np.clip((h + 0.5) / 1.1, 0.0, 1.0)
    soft = 1.0 / (1.0 + 0.6 * erosion * _slope2(h, TILE))
    return ((1.0 - erosion) + erosion * soft * (0.35 + 0.65 * low)).astype(np.float32)


def _base(v: dict, res: int, base: list) -> tuple:
    """Grandes couches : (hauteur brute, poids du détail). Chaque couche est
    calculée à la résolution qui lui suffit, la somme est agrandie au fur
    et à mesure (les couches larges coûtent 4 à 16 fois moins)."""
    rng = np.random.default_rng([int(v["bg_ld_seed"]), 1])
    disp = _displacement(v, res, rng)
    ridges, erosion = float(v["bg_ld_ridges"]), float(v["bg_ld_erosion"])
    r0 = _eval_res(round(base[0][0] * TILE), res)
    h = np.zeros((r0, r0), np.float32)
    weight = np.ones((r0, r0), np.float32)
    total, cache = 0.0, {}
    for f, amp, _ in base:
        while h.shape[0] < _eval_res(round(f * TILE), res):
            h, weight = upsample2(h), upsample2(weight)
        u, w = _coords(h.shape[0], disp, cache)
        h += _octave(u, w, f, TILE, ridges, rng) * (amp * weight)
        total += amp
        if erosion > 0.0:
            weight = _erosion_weight(h / total, erosion)
    while h.shape[0] < res:
        h, weight = upsample2(h), upsample2(weight)
    return h, weight


@lru_cache(maxsize=2)
def _detail(key: tuple, res: int) -> tuple:
    """Couches fines (carte du détail, `res` / 2 texels), mises en cache : la
    déformation, l'érosion et les terrasses ne les changent pas. Rend
    (détail brut, pente)."""
    v = dict(key)
    _, detail = layers(v, res)
    res //= 2
    if not detail:
        return np.zeros((4, 4), np.float32), 0.0
    rng = np.random.default_rng([int(v["bg_ld_seed"]), 2])
    ridges = float(v["bg_ld_ridges"])
    h = None
    for f, amp, _ in detail:
        r = _eval_res(round(f * DETAIL_TILE), res)
        if h is None:
            h = np.zeros((r, r), np.float32)
        while h.shape[0] < r:
            h = upsample2(h)
        u, w = _grid(h.shape[0])
        h += _octave(u, w, f, DETAIL_TILE, ridges, rng) * amp
    while h.shape[0] < res:
        h = upsample2(h)
    h -= h.mean()
    h.setflags(write=False)        # partagé (cache) : jamais modifié
    return h, _slope(h, DETAIL_TILE)


def build(v: dict, res: int = RES) -> TerrainMap:
    """Calcule les cartes des réglages `v` (clés de SETTINGS)."""
    key = settings_key(v)
    base, detail_layers = layers(v, res)
    h, weight = _base(v, res, base)
    # hauteur ramenée à ≈ −1…1 (1 % et 99 % des valeurs) : la Hauteur du
    # relief garde le même sens quel que soit le nombre de couches
    lo, mid, hi = np.percentile(h[::4, ::4], (1.0, 50.0, 99.0))
    gain = 2.0 / max(float(hi - lo), 1e-6)
    h = ((h - mid) * gain).astype(np.float32)
    h = _terraces(h, float(v["bg_ld_terrace"]),
                  max(2, int(round(float(v["bg_ld_terrace_n"])))))
    spec = np.fft.rfft2(h)
    texel = TILE / res                             # tailles du monde par texel
    cavity = np.zeros_like(h)
    for r in AO_RADII:
        cavity += (_blur(spec, res, r / texel) - h) / (r * len(AO_RADII))
    detail, detail_slope = _detail(settings_key(v, DETAIL_SETTINGS), res)
    data = np.ascontiguousarray(np.stack([h, cavity, weight], axis=-1), dtype=np.float32)
    detail_gain = gain if detail_layers else 0.0
    return TerrainMap(data=data, detail=detail, slope=_slope(h, TILE),
                      detail_slope=detail_slope, detail_gain=detail_gain,
                      detail_amp=float(np.abs(detail).max()) * detail_gain, key=key)


def flat(res: int = 4) -> TerrainMap:
    """Plaine plate : affichée tant que les premières cartes ne sont pas prêtes."""
    return TerrainMap(data=np.zeros((res, res, 3), np.float32),
                      detail=np.zeros((res, res), np.float32), slope=0.0,
                      detail_slope=0.0, detail_gain=0.0, detail_amp=0.0, key=())


# ── calcul en arrière-plan ────────────────────────────────────────────
class TerrainBuilder:
    """Calcule les cartes dans un thread. Les cartes affichées restent en
    place pendant le calcul ; une demande faite pendant un calcul attend la
    fin de celui-ci (seule la dernière est gardée). Une erreur du calcul est
    signalée par `report` depuis `take()`, dans le thread de l'appelant.
    `sync` : calcul immédiat dans `request()` (banc de rendu, tests)."""

    def __init__(self, report, sync: bool = False, res: int = RES):
        self.report = report
        self.sync = sync
        self.res = res
        self._lock = threading.Condition()
        self._wanted = None            # réglages à calculer
        self._asked = None             # dernière clé demandée
        self._done = None              # cartes prêtes, pas encore prises
        self._error = None             # erreur du dernier calcul, pas encore signalée
        self._thread = None

    def request(self, v: dict):
        """Demande les cartes des réglages `v` (rien si elles sont déjà
        demandées ou affichées)."""
        key = settings_key(v)
        if key == self._asked:
            return
        self._asked = key
        settings = {k: v[k] for k in SETTINGS}
        if self.sync:
            self._finish(settings)
            return
        with self._lock:
            self._wanted = settings
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="terrain",
                                                daemon=True)
                self._thread.start()
            self._lock.notify()

    def take(self):
        """Les cartes calculées depuis le dernier appel, ou None."""
        with self._lock:
            done, self._done = self._done, None
            error, self._error = self._error, None
        if error is not None:
            self.report("Fond : le relief n'a pas pu être calculé, l'ancien reste affiché",
                        error)
        return done

    def _finish(self, settings: dict):
        try:
            done, error = build(settings, self.res), None
        except Exception as e:  # noqa: BLE001 — frontière du thread de calcul
            done, error = None, e
        with self._lock:
            if done is not None:
                self._done = done
            if error is not None:
                self._error = error

    def _run(self):
        while True:
            with self._lock:
                while self._wanted is None:
                    self._lock.wait()
                settings, self._wanted = self._wanted, None
            self._finish(settings)
