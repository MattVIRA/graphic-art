"""Contrat des features (architecture.md) partagé par les tests des deux
analyseurs et du silence : un renommage doit casser les tests, pas l'appli."""

SCALARS = ("bass", "high", "rms", "transients", "beat_phase",
           "kick_amp", "snare_amp", "section")
NORMALIZED = ("bass", "high", "rms", "transients", "section")
INTS = ("kick_id", "snare_id", "frame_idx")
KEYS = set(SCALARS) | set(INTS) | {"bpm", "eq_bands", "spectrum", "is_beat"}
N_MELS = 64


def check_types(f: dict):
    """Types et formes attendus pour chaque clé."""
    import numpy as np
    for k in SCALARS:
        assert isinstance(f[k], float), k
    assert isinstance(f["is_beat"], bool)
    for k in INTS:
        assert isinstance(f[k], int), k
    assert np.shape(f["spectrum"]) == (N_MELS,)
    assert np.shape(f["eq_bands"]) == (7,)
