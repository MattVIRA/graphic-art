"""Source audio (T1.15, app/sources.py + audio/player.py) : un fichier
introuvable, vide ou illisible lève UserError et la source courante
continue ; fin de fichier, entrée live perdue ; pause."""

import numpy as np
import pytest
import soundfile as sf

import app.sources as sources
import audio.player as player
from app.errors import UserError
from app.sources import SILENT_FEATURES, SourceManager
from tests.contract import KEYS, check_types


class FakeSource:
    is_done = False
    is_paused = False

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def get_current_features(self):
        return SILENT_FEATURES


def manager_with_source():
    reports = []
    m = SourceManager(report=reports.append)
    m.source = current = FakeSource()
    return m, current, reports


def test_silence_follows_the_contract():
    assert set(SILENT_FEATURES) == KEYS
    check_types(SILENT_FEATURES)
    assert SourceManager().features() is SILENT_FEATURES
    assert SourceManager().label == "AUCUNE SOURCE"


@pytest.mark.parametrize("make, expected", [
    (lambda d: str(d / "absent.wav"), "introuvable"),
    (lambda d: (d / "vide.wav").write_bytes(b"") or str(d / "vide.wav"), "vide"),
])
def test_missing_or_empty_file_keeps_the_current_source(tmp_path, make, expected):
    m, current, _ = manager_with_source()
    with pytest.raises(UserError, match=expected):
        m.start_file(make(tmp_path))
    assert m.source is current and not current.stopped


def test_corrupt_file_keeps_the_current_source(tmp_path):
    path = tmp_path / "abime.wav"
    path.write_bytes(b"RIFF pas du tout un wav " * 40)
    m, current, _ = manager_with_source()
    with pytest.raises(UserError, match="illisible"):
        m.start_file(str(path))
    assert m.source is current and not current.stopped


def test_valid_file_replaces_the_source(tmp_path, monkeypatch):
    class FakePlayer(FakeSource):
        def __init__(self, path, analyzer):
            super().__init__()
            self.playing = False

        def play(self):
            self.playing = True

    monkeypatch.setattr(sources, "AudioPlayer", FakePlayer)   # pas de carte son
    path = tmp_path / "ok.wav"
    t = np.arange(2 * 44100) / 44100
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("f4"), 44100)
    titles = []
    m = SourceManager(on_title=titles.append)
    m.source = old = FakeSource()
    m.start_file(str(path))
    assert old.stopped and isinstance(m.source, FakePlayer) and m.source.playing
    assert titles == ["Graphic Art  ·  ok.wav"] and m.label == "FICHIER"


def test_failed_live_start_keeps_the_current_source(monkeypatch):
    import audio.live

    def refuse(device):
        raise audio.live.sd.PortAudioError("périphérique invalide")

    monkeypatch.setattr(audio.live, "LiveAnalyzer", refuse)
    m, current, _ = manager_with_source()
    with pytest.raises(audio.live.sd.PortAudioError):
        m.start_live(3)
    assert m.source is current and not current.stopped


def test_end_of_file_goes_silent():
    m, current, reports = manager_with_source()
    current.is_done = True
    m.update()
    assert m.source is None and current.stopped and not reports


def test_lost_live_input_is_reported_and_goes_silent():
    m, current, reports = manager_with_source()
    current.is_lost = True
    m.update()
    assert m.source is None and current.stopped
    assert len(reports) == 1 and "perdue" in reports[0]


# ── AudioPlayer : la pause n'est pas une fin de morceau ───────────────

class FakeMusic:
    """pygame.mixer.music : get_busy() vaut False en pause (pygame ≥ 2.0.1)."""

    def __init__(self):
        self.state = "stopped"

    def load(self, path):
        pass

    def play(self):
        self.state = "playing"

    def pause(self):
        self.state = "paused"

    def unpause(self):
        self.state = "playing"

    def stop(self):
        self.state = "stopped"

    def get_busy(self):
        return self.state == "playing"


def test_pause_is_not_the_end_of_the_track(monkeypatch):
    music = FakeMusic()
    monkeypatch.setattr(player.pygame.mixer, "music", music)
    p = player.AudioPlayer("x.wav", analyzer=None)
    p.play()
    p.toggle_pause()
    assert p.is_paused and not p.is_done
    p.toggle_pause()
    assert not p.is_paused and not p.is_done
    music.state = "stopped"          # le morceau se termine
    assert p.is_done
