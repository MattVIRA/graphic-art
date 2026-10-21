"""Relief du Paysage (T1.16, passe finale) : cartes de hauteur précalculées et
calcul en arrière-plan (background/terrain.py), sans GL. Cartes réduites
(128 ou 256 texels : à 128, la carte ne garde que 2 couches) : le calcul
reste rapide."""

import math
import threading

import numpy as np
import pytest

from background import macros, schema, terrain

RES = 128


def settings(**kw) -> dict:
    v = macros.apply(schema.defaults())
    v.update(kw)
    return {k: v[k] for k in terrain.SETTINGS}


def maps(res=RES, **kw):
    return terrain.build(settings(**kw), res)


def seam_ratio(a: np.ndarray) -> float:
    """Écart au raccord (dernière → première colonne / ligne) rapporté à
    l'écart moyen entre voisins : ≈ 1 pour une carte périodique."""
    inner = (np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2
    seam = (np.abs(a[0] - a[-1]).mean() + np.abs(a[:, 0] - a[:, -1]).mean()) / 2
    return float(seam / inner)


# ── bruit ─────────────────────────────────────────────────────────────
def test_gradient_noise_is_periodic_bounded_and_survives_float_edges():
    rng = np.random.default_rng(0)
    g = terrain.gradients(8, rng)
    u, w = terrain._grid(64)
    n = terrain.gradient_noise(u, w, g)
    assert n.dtype == np.float32 and np.abs(n).max() <= 1.0 + 1e-5
    assert np.allclose(terrain.gradient_noise(u + 1.0, w - 2.0, g), n, atol=1e-4)
    # une coordonnée à peine négative (déformation) : le modulo float32 peut
    # rendre n tout juste, la cellule reste dans la table
    edge = np.full((4, 4), -1e-9, dtype=np.float32)
    assert np.isfinite(terrain.gradient_noise(edge, edge, g)).all()


def test_upsample_keeps_the_mean_and_the_period():
    a = terrain.gradient_noise(*terrain._grid(32), terrain.gradients(4, np.random.default_rng(1)))
    up = terrain.upsample2(a)
    assert up.shape == (64, 64)
    assert up.mean() == pytest.approx(a.mean(), abs=1e-6)
    assert seam_ratio(up) < 1.5


# ── cartes ────────────────────────────────────────────────────────────
def test_maps_are_deterministic_periodic_and_normalised():
    a, b = maps(), maps()
    assert np.array_equal(a.data, b.data) and a.key == b.key
    assert a.data.shape == (RES, RES, 3) and a.data.dtype == np.float32
    assert a.detail.shape == (RES // 2, RES // 2)
    h = a.data[..., 0]
    lo, hi = np.percentile(h, (1.0, 99.0))
    assert hi - lo == pytest.approx(2.0, rel=0.05)          # ≈ −1…1
    for ch in range(3):
        assert seam_ratio(a.data[..., ch]) < 1.6, ch         # la carte boucle
    assert a.slope > 0.0 and math.isfinite(a.slope)


def test_seed_changes_the_relief_but_not_its_kind():
    a, b = maps(bg_ld_seed=1), maps(bg_ld_seed=2)
    assert not np.allclose(a.data[..., 0], b.data[..., 0])
    assert a.slope == pytest.approx(b.slope, rel=0.35)


def test_each_setting_changes_the_map():
    base = maps(256, bg_ld_ridges=0.0, bg_ld_erosion=0.0, bg_ld_warp=0.0)
    for key, value in (("bg_ld_octaves", 1.0), ("bg_ld_rough", 0.65), ("bg_ld_lacun", 2.8),
                       ("bg_ld_ridges", 1.0), ("bg_ld_erosion", 1.0), ("bg_ld_warp", 1.5),
                       ("bg_ld_terrace", 0.9)):
        other = maps(256, **{**dict(bg_ld_ridges=0.0, bg_ld_erosion=0.0, bg_ld_warp=0.0),
                        key: value})
        assert not np.allclose(other.data[..., 0], base.data[..., 0], atol=1e-3), key


def test_more_layers_means_more_fine_detail():
    """Une couche de plus ajoute du relief fin (la pente grandit) ; les couches
    trop fines pour la carte vont dans le détail."""
    few, many = maps(256, bg_ld_octaves=1.0), maps(256, bg_ld_octaves=3.0)
    assert many.slope > few.slope
    base, detail = terrain.layers({"bg_ld_octaves": 10, "bg_ld_rough": 0.5,
                                   "bg_ld_lacun": 2.0}, terrain.RES)
    assert [f for f, _, _ in base] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert [f for f, _, _ in detail] == [32.0, 64.0, 128.0]   # 4 texels par cellule
    assert terrain.build(settings(bg_ld_octaves=1.0), RES).detail_gain == 0.0


def test_terraces_flatten_the_steps():
    smooth, steps = maps(), maps(bg_ld_terrace=1.0, bg_ld_terrace_n=4.0)
    s_smooth = np.sqrt(terrain._slope2(smooth.data[..., 0], terrain.TILE))
    s_steps = np.sqrt(terrain._slope2(steps.data[..., 0], terrain.TILE))
    # des paliers plats : bien plus de texels presque horizontaux
    assert (s_steps < 0.3).mean() > 2.0 * (s_smooth < 0.3).mean()


def test_cavity_is_positive_in_hollows():
    m = maps()
    h, cav = m.data[..., 0], m.data[..., 1]
    assert cav[h < np.percentile(h, 5)].mean() > 0.0 > cav[h > np.percentile(h, 95)].mean()


def test_detail_amplitude_bounds_the_detail():
    m = maps(256, bg_ld_octaves=8.0)
    assert m.detail_gain > 0.0
    assert m.detail_amp == pytest.approx(np.abs(m.detail).max() * m.detail_gain)
    # petit devant la carte (≈ −1…1) ; à 256 texels, le détail reçoit déjà
    # la 4e couche (à 1024, il commence à la 6e)
    assert m.detail_amp < 0.5


def test_erosion_weight_stays_in_range():
    w = maps(bg_ld_erosion=1.0).data[..., 2]
    assert 0.0 < w.min() and w.max() <= 1.0 + 1e-6
    assert np.allclose(maps(bg_ld_erosion=0.0).data[..., 2], 1.0)


def test_settings_key_ignores_tiny_moves():
    v = settings()
    nudged = dict(v, bg_ld_warp=v["bg_ld_warp"] + 1e-5)
    assert terrain.settings_key(v) == terrain.settings_key(nudged)
    assert terrain.settings_key(v) != terrain.settings_key(dict(v, bg_ld_warp=1.2))


def test_flat_map_is_a_plain():
    f = terrain.flat()
    assert not f.data.any() and f.slope == 0.0 and f.detail_gain == 0.0


# ── calcul en arrière-plan ────────────────────────────────────────────
def test_sync_builder_builds_once_per_key():
    reports = []
    b = terrain.TerrainBuilder(lambda m, e=None: reports.append(m), sync=True, res=RES)
    b.request(settings())
    first = b.take()
    assert first is not None and b.take() is None
    b.request(settings())                    # même clé : rien à refaire
    assert b.take() is None
    b.request(settings(bg_ld_seed=5))
    assert b.take().key != first.key
    assert reports == []


def test_threaded_builder_delivers_the_last_request():
    b = terrain.TerrainBuilder(lambda m, e=None: None, res=RES)
    for seed in (1, 2, 3):
        b.request(settings(bg_ld_seed=float(seed)))
    last = terrain.settings_key(settings(bg_ld_seed=3.0))
    got = None
    for _ in range(400):                     # au plus ~4 s
        done = b.take()
        if done is not None:
            got = done
            if done.key == last:
                break
        threading.Event().wait(0.01)
    assert got is not None and got.key == last


def test_builder_reports_errors_from_the_caller_thread(monkeypatch):
    reports = []
    caller = threading.get_ident()

    def report(message, exc=None):
        reports.append((message, type(exc), threading.get_ident()))

    def broken(v, res):
        raise ValueError("carte cassée")

    monkeypatch.setattr(terrain, "build", broken)
    b = terrain.TerrainBuilder(report, res=RES)
    b.request(settings())
    for _ in range(200):
        if b.take() is None and reports:
            break
        threading.Event().wait(0.01)
    assert reports and reports[0][1] is ValueError and reports[0][2] == caller
