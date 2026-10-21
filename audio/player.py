import pygame

from audio.analyzer import AudioAnalyzer


class AudioPlayer:
    """Wraps pygame.mixer.music and exposes get_current_features() synced to playback position."""

    def __init__(self, filepath: str, analyzer: AudioAnalyzer):
        self.analyzer = analyzer
        self._paused = False
        self._playing = False

        pygame.mixer.music.load(filepath)

    # ------------------------------------------------------------------

    def play(self):
        pygame.mixer.music.play()
        self._playing = True
        self._paused = False

    def toggle_pause(self):
        if not self._playing:
            return
        if self._paused:
            pygame.mixer.music.unpause()
            self._paused = False
        else:
            pygame.mixer.music.pause()
            self._paused = True

    def stop(self):
        pygame.mixer.music.stop()
        self._playing = False

    @property
    def is_done(self) -> bool:
        # depuis pygame 2.0.1, get_busy() vaut False en pause : sans le test
        # de pause, la boucle croyait le morceau fini et coupait la source (T1.15)
        return self._playing and not self._paused and not pygame.mixer.music.get_busy()

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------

    def get_current_features(self) -> dict:
        pos_ms = max(0, pygame.mixer.music.get_pos())
        frame_idx = self.analyzer.ms_to_frame(pos_ms)
        return self.analyzer.get_features(frame_idx)
