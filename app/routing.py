"""Features audio → effets (T4, T21, T22).

Chaque frame : lissage attack/release des features, couche percussive
(ressort du KICK, flash SNARE, énergie de section), puis routing Physique —
chaque source (7 bandes EQ + BEAT, KICK, SNARE, ÉNERGIE) est multipliée par
le montant choisi dans le panneau et accumulée sur son effet.

Sans GL ni fenêtre : testé dans tests/test_routing.py.
"""

import math
from dataclasses import dataclass

import numpy as np

from gfx.controls import EFFECTS, EQ_BANDS


class Envelope:
    """Asymmetric smoother for audio features: fast attack (a kick still hits
    instantly), slow release (the effect breathes out instead of flickering).
    Works on scalars and numpy arrays, one state per key."""

    def __init__(self, attack_ms: float = 20.0, release_ms: float = 180.0):
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self._state = {}

    @staticmethod
    def _coef(ms: float, dt: float) -> float:
        return 1.0 - math.exp(-dt / max(ms / 1000.0, 1e-4))

    def smooth(self, key: str, value, dt: float, reactivity: float = 1.0):
        """reactivity scales the release: >1 = percussive, <1 = floaty."""
        prev = self._state.get(key)
        if prev is None or np.shape(prev) != np.shape(value):
            self._state[key] = np.array(value, dtype=np.float32) \
                if not np.isscalar(value) else float(value)
            return value
        ka = self._coef(self.attack_ms, dt)
        kr = self._coef(self.release_ms / max(reactivity, 0.1), dt)
        if np.isscalar(prev):
            k = ka if value > prev else kr
            new = prev + (value - prev) * k
        else:
            v = np.asarray(value, dtype=np.float32)
            k = np.where(v > prev, ka, kr).astype(np.float32)
            new = prev + (v - prev) * k
        self._state[key] = new
        return new


class Spring:
    """Damped oscillator kicked by impulses: snaps out, overshoots a little
    and settles — an elastic response instead of a linear follow."""

    def __init__(self, freq_hz: float = 3.0, damping: float = 0.35):
        self.w = 2.0 * math.pi * freq_hz
        self.z = damping
        self.x = self.v = 0.0

    def kick(self, amount: float):
        # velocity such that the first peak ≈ amount
        self.v += amount * self.w * 1.5

    def step(self, dt: float) -> float:
        n = max(1, int(dt / 0.004))          # stable sub-steps
        h = dt / n
        for _ in range(n):
            a = -self.w * self.w * self.x - 2.0 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


# features scalaires lissées ; beat_phase / is_beat / transients restent
# bruts exprès (un beat doit tomber pile)
SMOOTHED_KEYS = ("bass", "high", "rms")


def route(bands, p: dict, section: float) -> dict:
    """{effet: montant} : Σ source × montant du panneau, atténué dans les
    sections calmes par le réglage « Dynamique »."""
    fx = {name: 0.0 for name in EFFECTS}
    dyn_gain = 1.0 - p.get("dynamics", 0.0) * 0.75 * (1.0 - section)
    for i in range(len(EQ_BANDS)):
        eff = EFFECTS[int(p[f"band{i}_fx"])]
        fx[eff] += float(bands[i]) * p[f"band{i}_amt"] * dyn_gain
    return fx


@dataclass
class Routed:
    """Résultat d'une frame de `Reactor.step()`."""
    features: dict    # features lissées
    bands: list       # sources du routing, dans l'ordre de EQ_BANDS
    fx: dict          # {effet: montant}
    beat_env: float   # flash de beat, 1 sur le beat puis décroissant


class Reactor:
    """État qui traverse les frames : enveloppes, ressort du kick, flash
    snare et flash de beat. `step()` transforme les features brutes d'une
    frame en sources de routing et en effets."""

    def __init__(self):
        self.envelope = Envelope()   # T4: attack/release smoothing
        self.kick_spring = Spring()  # KICK band: elastic response to each kick
        self.last_kick = self.last_snare = 0
        self.snare_env = 0.0
        self.beat_env = 0.0          # beat flash (spikes on beat, fast decay)

    def step(self, features: dict, p: dict, dt: float) -> Routed:
        """dt en secondes (borné à 1 ms) ; p = paramètres du panneau."""
        dt = max(dt, 1e-3)
        env = self.envelope
        react = p["reactivity"]
        f = dict(features)
        for k in SMOOTHED_KEYS:
            f[k] = float(env.smooth(k, f[k], dt, react))
        f["spectrum"] = env.smooth("spectrum", f["spectrum"], dt, react)
        f["eq_bands"] = env.smooth("eq_bands", f["eq_bands"], dt, react)

        # beat flash: spikes on each beat then decays quickly
        if f.get("is_beat"):
            self.beat_env = 1.0
        else:
            self.beat_env *= 0.86

        # ── percussive layer: kick spring, snare flash, section energy ──
        kid, sid = f.get("kick_id", 0), f.get("snare_id", 0)
        if kid != self.last_kick:
            # un id qui recule = nouvelle source ou retour en arrière : pas de coup
            if kid > self.last_kick:
                self.kick_spring.kick(f.get("kick_amp", 0.0))
            self.last_kick = kid
        if sid != self.last_snare:
            if sid > self.last_snare:
                self.snare_env = max(self.snare_env, f.get("snare_amp", 0.0))
            self.last_snare = sid
        kick_x = self.kick_spring.step(dt)
        self.snare_env *= math.exp(-dt / 0.12)
        section = float(env.smooth("section", f.get("section", 0.0), dt, 0.15))

        # "BEAT": 1.0 on the beat, decaying to 0 until the next one
        beat_pulse = ((1.0 - f.get("beat_phase", 0.0)) ** 2
                      if f.get("bpm", 0.0) > 0 else 0.0)
        bands = list(f["eq_bands"]) + [beat_pulse, kick_x, self.snare_env,
                                       section]
        return Routed(f, bands, route(bands, p, section), self.beat_env)
