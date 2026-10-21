"""Source audio courante : fichier, capture live ou silence (T1.1, T1.15).

Une seule source à la fois ; changer de source n'arrête l'ancienne qu'une
fois la nouvelle prête (un fichier illisible ne coupe pas la musique).
Une entrée invalide lève `UserError` ; la perte d'une entrée live en cours
de route est signalée par `report(message)` et ramène au silence.
"""

import logging
import os
import time

import numpy as np

from app.errors import UserError
from audio.analyzer import AudioAnalyzer
from audio.player import AudioPlayer

log = logging.getLogger(__name__)

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

XRUN_LOG_PERIOD_S = 10.0   # décrochages live : au plus une ligne de journal par période


class SourceManager:
    """Possède la source audio. `on_title(texte)` reçoit le titre de fenêtre
    à afficher quand la source change ; `report(message)` les incidents
    survenus hors d'une action de l'utilisateur."""

    def __init__(self, on_title=None, report=None):
        self.source = None      # objet avec get_current_features / toggle_pause / stop
        self.tempo = 0.0        # tempo de l'analyse hors ligne (fichier), pour le HUD
        self._on_title = on_title or (lambda _text: None)
        self._report = report or (lambda message: log.warning("%s", message))
        self._xruns = 0
        self._xrun_log_t = 0.0

    # ── changer de source ─────────────────────────────────────────────
    def stop(self):
        """Coupe la source courante → silence."""
        if self.source is not None:
            self.source.stop()
        self.source = None
        self.tempo = 0.0

    def start_file(self, path: str):
        """Analyse puis joue un fichier audio. Fichier introuvable, vide ou
        illisible → `UserError`, et la source courante continue."""
        name = os.path.basename(path)
        log.info("Chargement audio : %s", path)
        if not os.path.isfile(path):
            raise UserError(f"Fichier audio introuvable : {path}")
        if os.path.getsize(path) == 0:
            raise UserError(f"Fichier audio vide (0 octet) : {name}")
        try:
            analyzer = AudioAnalyzer(path)
            player = AudioPlayer(path, analyzer)
        except Exception as e:  # noqa: BLE001 — frontière : fichier choisi par l'utilisateur
            # librosa / audioread / pygame lèvent des types variés selon le
            # format ; la trace va au journal, le message à l'utilisateur
            log.info("lecture de %s impossible", path, exc_info=e)
            raise UserError(f"Fichier audio illisible : {name} "
                            f"({type(e).__name__}: {e or 'format non reconnu'})") from e
        self.stop()    # only once the new file is known to be readable
        player.play()
        self.source = player
        self.tempo = analyzer.tempo
        self._on_title(f"Graphic Art  ·  {name}")

    def start_live(self, device: int):
        """Capture live sur le périphérique d'entrée `device` ; la source
        courante n'est arrêtée qu'une fois le flux ouvert."""
        from audio.live import LiveAnalyzer  # sounddevice : chargé à la demande
        log.info("Capture live sur le périphérique %s", device)
        live = LiveAnalyzer(device=device)
        self.stop()
        self.source = live
        self._xruns = 0
        self._on_title("Graphic Art  ·  LIVE")

    # ── par frame ─────────────────────────────────────────────────────
    def update(self):
        """Fin du fichier ou entrée live perdue → silence."""
        src = self.source
        if src is None:
            return
        if src.is_done:
            self.stop()    # file ended: keep the window open, go silent
        elif getattr(src, "is_lost", False):
            self.stop()
            self._report("Entrée live perdue (périphérique débranché ?) : "
                         "retour au silence")
        else:
            self._log_xruns(src)

    def _log_xruns(self, src):
        """Décrochages du flux live (blocs perdus) : journal seulement, un
        décrochage isolé ne mérite pas d'alerter l'utilisateur."""
        self._xruns += getattr(src, "take_xruns", lambda: 0)()
        now = time.monotonic()
        if self._xruns and now - self._xrun_log_t > XRUN_LOG_PERIOD_S:
            log.warning("capture live : %d décrochage(s) du flux audio", self._xruns)
            self._xruns, self._xrun_log_t = 0, now

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
