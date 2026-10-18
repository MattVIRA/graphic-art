"""Routing Physique (T22) : enveloppes, ressort, couche percussive, sources
→ effets, et paramètres dérivés passés au renderer (app/frame.py)."""

import numpy as np
import pytest

from app import frame
from app.routing import Envelope, Reactor, Spring, route
from app.sources import SILENT_FEATURES
from gfx.controls import DEFAULTS, EFFECTS, EQ_BANDS

DT = 1 / 60


def params(**over):
    p = dict(DEFAULTS)
    p.update(over)
    return p


def run(reactor, features, p, frames):
    out = None
    for _ in range(frames):
        out = reactor.step(features, p, DT)
    return out


# ── Envelope ──────────────────────────────────────────────────────────

def test_envelope_first_value_passes_through():
    assert Envelope().smooth("k", 0.7, DT) == 0.7


def test_envelope_attack_is_faster_than_release():
    env = Envelope()
    env.smooth("up", 0.0, DT)
    env.smooth("down", 1.0, DT)
    up = env.smooth("up", 1.0, DT)
    down = env.smooth("down", 0.0, DT)
    assert up > 1.0 - down        # monte plus vite qu'il ne redescend


def test_envelope_reactivity_shortens_the_release():
    slow, fast = Envelope(), Envelope()
    for env in (slow, fast):
        env.smooth("k", 1.0, DT)
    a = slow.smooth("k", 0.0, DT, reactivity=0.5)
    b = fast.smooth("k", 0.0, DT, reactivity=3.0)
    assert b < a


def test_envelope_smooths_arrays_per_bin():
    env = Envelope()
    env.smooth("s", np.zeros(4, "f4"), DT)
    out = env.smooth("s", np.array([1, 0, 1, 0], "f4"), DT)
    assert out.shape == (4,)
    assert out[0] > 0 and out[1] == 0


# ── Spring ────────────────────────────────────────────────────────────

def test_spring_peaks_near_the_kick_then_settles():
    s = Spring()
    s.kick(1.0)
    xs = [s.step(DT) for _ in range(240)]
    assert 0.6 < max(xs) < 1.4
    assert abs(xs[-1]) < 0.02


# ── route ─────────────────────────────────────────────────────────────

def test_route_sends_each_band_to_its_effect():
    rot = EFFECTS.index("Vitesse rotation")
    p = params(band0_fx=float(rot), band0_amt=2.0)
    bands = [0.5] + [0.0] * (len(EQ_BANDS) - 1)
    fx = route(bands, p, section=1.0)
    assert set(fx) == set(EFFECTS)
    assert fx["Vitesse rotation"] == pytest.approx(1.0)


def test_route_dynamics_damps_calm_sections():
    zoom = EFFECTS.index("Zoom (pulse)")
    p = params(band0_fx=float(zoom), dynamics=1.0)
    bands = [1.0] + [0.0] * (len(EQ_BANDS) - 1)
    assert route(bands, p, section=0.0)["Zoom (pulse)"] == pytest.approx(0.25)
    assert route(bands, p, section=1.0)["Zoom (pulse)"] == pytest.approx(1.0)


# ── Reactor ───────────────────────────────────────────────────────────

def test_reactor_silence_gives_no_effect():
    out = run(Reactor(), SILENT_FEATURES, params(), 30)
    assert len(out.bands) == len(EQ_BANDS)
    assert all(v == 0.0 for v in out.fx.values())
    assert out.beat_env == 0.0


def test_reactor_does_not_modify_the_input_features():
    f = dict(SILENT_FEATURES, bass=1.0)
    Reactor().step(f, params(), DT)
    assert f["bass"] == 1.0


def test_reactor_kick_id_launches_the_spring():
    kick = EQ_BANDS.index("KICK")
    r = Reactor()
    run(r, SILENT_FEATURES, params(), 5)
    hit = dict(SILENT_FEATURES, kick_id=1, kick_amp=1.0)
    peak = max(r.step(hit, params(), DT).bands[kick] for _ in range(20))
    assert peak > 0.5


def test_reactor_kick_id_going_back_is_not_a_hit():
    kick = EQ_BANDS.index("KICK")
    r = Reactor()
    run(r, dict(SILENT_FEATURES, kick_id=9, kick_amp=1.0), params(), 240)
    back = dict(SILENT_FEATURES, kick_id=2, kick_amp=1.0)  # nouvelle source
    peak = max(abs(r.step(back, params(), DT).bands[kick]) for _ in range(20))
    assert peak < 0.05


def test_reactor_beat_flash_and_beat_pulse():
    beat = EQ_BANDS.index("BEAT")
    r = Reactor()
    on = r.step(dict(SILENT_FEATURES, is_beat=True, bpm=120.0), params(), DT)
    assert on.beat_env == 1.0
    assert on.bands[beat] == pytest.approx(1.0)       # beat_phase = 0
    off = r.step(dict(SILENT_FEATURES, bpm=120.0, beat_phase=0.5), params(), DT)
    assert off.beat_env < 1.0
    assert off.bands[beat] == pytest.approx(0.25)
    # sans tempo connu, pas d'impulsion BEAT
    assert r.step(SILENT_FEATURES, params(), DT).bands[beat] == 0.0


# ── app/frame.py ──────────────────────────────────────────────────────

def no_fx(**over):
    fx = {name: 0.0 for name in EFFECTS}
    fx.update(over)
    return fx


def test_model_params_freeze_and_routed_amounts():
    p = params(music_spin=0.0, inflate=0.5, turbulence=2.0)
    p2 = frame.model_params(p, no_fx(**{"Gonflement": 0.25, "Turbulence": 0.5,
                                         "Vitesse rotation": 0.3}))
    assert p2["rot_gain"] == 0.0
    assert p2["inflate"] == pytest.approx(0.75)
    assert p2["model_turb"] == pytest.approx(1.0)
    assert p2["fx_rot"] == pytest.approx(0.3)
    assert p["inflate"] == 0.5                        # p n'est pas modifié


def test_no_turbulence_unless_routed():
    assert frame.model_params(params(turbulence=2.0), no_fx())["model_turb"] == 0.0


def test_shake_and_zoom_pulse_are_neutral_without_fx():
    assert frame.shake(no_fx(), 12.3) == (0.0, 0.0)
    assert frame.zoom_pulse(no_fx()) == 1.0
    assert frame.shake(no_fx(**{"Secousse caméra": 1.0}), 12.3) != (0.0, 0.0)
