"""Contrat des features (T25) : AudioAnalyzer renvoie toutes les clés du
dictionnaire d'architecture.md, avec les bons types. Les visuels et main.py
s'appuient sur ces clés : un renommage doit casser ce test, pas l'appli.

Lent (analyse librosa ≈ 2 s) : marqué `slow`, exclu du hook de fin de tour,
lancé au commit et par /implement-ticket."""

import numpy as np
import pytest
import soundfile as sf

from audio.analyzer import AudioAnalyzer

pytestmark = pytest.mark.slow

SCALARS = ("bass", "high", "rms", "transients", "beat_phase",
           "kick_amp", "snare_amp", "section")
NORMALIZED = ("bass", "high", "rms", "transients", "section")
KEYS = set(SCALARS) | {"bpm", "eq_bands", "spectrum", "is_beat",
                       "kick_id", "snare_id", "frame_idx"}


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory):
    """4 s de son : un sinus 220 Hz + un clic toutes les 0,5 s (120 BPM)."""
    sr = 44100
    t = np.arange(4 * sr) / sr
    y = 0.3 * np.sin(2 * np.pi * 220 * t)
    for k in range(8):
        i = int(k * 0.5 * sr)
        y[i:i + 400] += np.hanning(400) * 0.9
    path = tmp_path_factory.mktemp("audio") / "contrat.wav"
    sf.write(path, y.astype("f4"), sr)
    return AudioAnalyzer(str(path))


def test_all_documented_keys(analyzer):
    assert set(analyzer.get_features(10)) == KEYS


def test_types_and_shapes(analyzer):
    f = analyzer.get_features(10)
    for k in SCALARS:
        assert isinstance(f[k], float), k
    assert isinstance(f["is_beat"], bool)
    for k in ("kick_id", "snare_id", "frame_idx"):
        assert isinstance(f[k], int), k
    assert f["spectrum"].shape == (AudioAnalyzer.N_MELS,)
    assert f["eq_bands"].shape == (7,)
    assert float(f["bpm"]) > 0.0


def test_normalized_ranges_over_the_file(analyzer):
    for i in range(0, analyzer.n_frames, 7):
        f = analyzer.get_features(i)
        for k in NORMALIZED:
            assert 0.0 <= f[k] <= 1.0 + 1e-6, (k, i, f[k])
        assert 0.0 <= f["beat_phase"] <= 1.0
        assert np.all(np.isfinite(f["spectrum"]))
        assert np.all(np.isfinite(f["eq_bands"]))


def test_frame_index_is_clamped(analyzer):
    assert analyzer.get_features(-5)["frame_idx"] == 0
    last = analyzer.n_frames - 1
    assert analyzer.get_features(10**9)["frame_idx"] == last
