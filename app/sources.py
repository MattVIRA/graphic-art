"""Source audio courante : fichier, capture live ou silence (T0, T22).

Une seule source à la fois ; changer de source n'arrête l'ancienne qu'une
fois la nouvelle prête (un fichier illisible ne coupe pas la musique).
"""

import os

import numpy as np

from audio.analyzer import AudioAnalyzer
from audio.player import AudioPlayer

# features renvoyées quand aucune source ne joue (même contrat que les
# analyseurs : voir le dictionnaire des features dans architecture.md)
SILENT_FEATURES = {
    "bass": 0.0, "high": 0.0, "rms": 0.0, "transients": 0.0,
    "is_beat": False, "frame_idx": 0, "bpm": 0.0, "beat_phase": 0.0,
    "spectrum": np.zeros(AudioAnalyzer.N_MELS, dtype="f4"),
    "eq_bands": np.zeros(7, dtype="f4"),
    "kick_id": 0, "kick_amp": 0.0, "snare_id": 0, "snare_amp": 0.0,
    "section": 0.0,
}


class SourceManager:
    """Possède la source audio. `on_title(texte)` reçoit le titre de fenêtre
    à afficher quand la source change."""

    def __init__(self, on_title=None):
        self.source = None      # objet avec get_current_features / toggle_pause / stop
        self.tempo = 0.0        # tempo de l'analyse hors ligne (fichier), pour le HUD
        self._on_title = on_title or (lambda _text: None)

    # ── changer de source ─────────────────────────────────────────────
    def stop(self):
        """Coupe la source courante → silence."""
        if self.source is not None:
            self.source.stop()
        self.source = None
        self.tempo = 0.0

    def start_file(self, path: str):
        """Analyze then play an audio file. On any failure (missing, empty or
        undecodable file) print why and keep the current source playing."""
        print(f"\nLoading: {path}")
        if not os.path.isfile(path):
            print(f"[error] audio file not found: {path}")
            return
        if os.path.getsize(path) == 0:
            print(f"[error] audio file is empty (0 bytes): {path}")
            return
        try:
            analyzer = AudioAnalyzer(path)
            player = AudioPlayer(path, analyzer)
        except Exception as e:  # noqa: BLE001 — frontière : fichier choisi par l'utilisateur
            print(f"[error] cannot read audio file {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e or 'format not recognised'}")
            return
        self.stop()    # only once the new file is known to be readable
        player.play()
        self.source = player
        self.tempo = analyzer.tempo
        self._on_title(f"Graphic Art  ·  {os.path.basename(path)}")

    def start_live(self, device: int):
        """Capture live sur le périphérique d'entrée `device`."""
        self.stop()
        from audio.live import LiveAnalyzer  # sounddevice : chargé à la demande
        print(f"\nLive capture on input device {device}")
        self.source = LiveAnalyzer(device=device)
        self._on_title("Graphic Art  ·  LIVE")

    # ── par frame ─────────────────────────────────────────────────────
    def update(self):
        """Fin du fichier : on garde la fenêtre ouverte, en silence."""
        if self.source is not None and self.source.is_done:
            self.stop()

    def features(self) -> dict:
        """Features de l'instant (silence sans source)."""
        if self.source is None:
            return SILENT_FEATURES
        return self.source.get_current_features()

    def toggle_pause(self):
        if self.source is not None:
            self.source.toggle_pause()

    @property
    def is_paused(self) -> bool:
        return self.source.is_paused if self.source is not None else False

    @property
    def label(self) -> str:
        """Libellé de la barre de statut du panneau."""
        if self.source is None:
            return "AUCUNE SOURCE"
        if type(self.source).__name__ == "LiveAnalyzer":
            return "LIVE"
        return "FICHIER"
