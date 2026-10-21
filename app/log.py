"""Journal de l'appli (T1.15) : console + fichier `logs/graphic-art.log`.

Le fichier tourne (1 Mo × 3) et garde l'horodatage et les traces complètes :
c'est ce qu'il faut regarder quand l'appli a été lancée sans terminal
visible. Chaque module écrit avec `logging.getLogger(__name__)`.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "graphic-art.log")


class _ConsoleFormatter(logging.Formatter):
    """Console : la trace seulement pour les erreurs. Un fichier abîmé
    (avertissement) garde sa trace dans le fichier journal, sans noyer le
    terminal."""

    def format(self, record):
        if record.exc_info and record.levelno < logging.ERROR:
            # copie : l'enregistrement est partagé avec le fichier journal
            record = logging.makeLogRecord(record.__dict__)
            record.exc_info = record.exc_text = None
        return super().format(record)


def setup(level: int = logging.INFO, log_path: str = LOG_PATH):
    """Branche la console et le fichier sur le logger racine (une seule fois)."""
    root = logging.getLogger()
    if getattr(root, "_graphic_art", False):
        return
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(_ConsoleFormatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3,
                                   encoding="utf-8")
    except OSError as e:   # disque plein, dossier en lecture seule : la console suffit
        root.warning("journal %s impossible à ouvrir : %s", log_path, e)
    else:
        file.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s : %(message)s"))
        root.addHandler(file)
    root._graphic_art = True
