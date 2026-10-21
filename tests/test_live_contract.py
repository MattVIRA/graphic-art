"""LiveAnalyzer (T1.15) : même contrat de features que l'analyzer, perte du
périphérique et décrochages — sans carte son (flux sounddevice simulé)."""

import numpy as np
import pytest

import audio.live as live
from tests.contract import KEYS, NORMALIZED, check_types

SR = 44100


class FakeStream:
    def __init__(self, **kw):
        self.callback = kw["callback"]
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.setattr(live.sd, "query_devices",
                        lambda device, kind: {"max_input_channels": 1})
    monkeypatch.setattr(live.sd, "InputStream", FakeStream)
    return live.LiveAnalyzer(device=0, sr=SR)


def feed(a, seconds=0.5, level=0.3):
    """Blocs de 512 échantillons : un 110 Hz + un clic toutes les 0,25 s."""
    t = np.arange(int(seconds * SR)) / SR
    y = level * np.sin(2 * np.pi * 110 * t)
    y[(t % 0.25) < 0.005] += level
    for i in range(0, len(y) - 511, 512):
        a._callback(y[i:i + 512, None].astype("f4"), 512, None, None)


def test_same_keys_and_types_as_the_analyzer(analyzer):
    feed(analyzer)
    f = analyzer.get_current_features()
    assert set(f) == KEYS
    check_types(f)
    for k in NORMALIZED:
        assert 0.0 <= f[k] <= 1.0 + 1e-6, (k, f[k])


def test_silence_gives_zeros(analyzer):
    feed(analyzer, level=0.0)
    f = analyzer.get_current_features()
    check_types(f)
    assert f["rms"] == 0.0 and not f["spectrum"].any()


def test_lost_when_blocks_stop(analyzer):
    feed(analyzer, seconds=0.05)
    assert not analyzer.is_lost
    analyzer._last_block -= analyzer.LOST_AFTER_S + 0.1   # plus de bloc depuis 2 s
    assert analyzer.is_lost
    feed(analyzer, seconds=0.05)
    assert not analyzer.is_lost


def test_status_flags_are_counted(analyzer):
    block = np.zeros((512, 1), "f4")
    analyzer._callback(block, 512, None, "input overflow")
    analyzer._callback(block, 512, None, None)
    analyzer._callback(block, 512, None, "input overflow")
    assert analyzer.take_xruns() == 2
    assert analyzer.take_xruns() == 0


def test_stop_closes_the_stream(analyzer):
    analyzer.stop()
    assert analyzer._stream.closed
