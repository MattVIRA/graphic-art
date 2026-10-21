"""Erreurs (T1.15) : erreurs dues à l'utilisateur et garde de la boucle.

- `UserError` : un fichier introuvable, vide ou illisible n'est pas un bug ;
  son message suffit (pas de trace) et la source / le modèle courant
  continue.
- `FrameGuard` : compte les frames en erreur. Une erreur nouvelle est
  signalée avec sa trace, une erreur répétée seulement comptée ; au-delà de
  `limit` frames en erreur d'affilée, l'appli doit s'arrêter proprement.

Sans GL ni fenêtre : testé dans tests/test_errors.py.
"""

import traceback


class UserError(Exception):
    """Entrée invalide fournie par l'utilisateur ; le message lui est destiné."""


def signature(exc: BaseException) -> tuple:
    """Identité d'une erreur : type, message et ligne où elle est levée —
    la même erreur à chaque frame a la même signature."""
    tb = traceback.extract_tb(exc.__traceback__)
    where = (tb[-1].filename, tb[-1].lineno) if tb else None
    return type(exc).__name__, str(exc)[:200], where


class FrameGuard:
    """Erreurs de frame consécutives : `failed()` à chaque frame en erreur,
    `ok()` à chaque frame réussie."""

    def __init__(self, limit: int = 30):
        self.limit = limit
        self.consecutive = 0
        self.counts = {}      # signature → nombre d'occurrences

    def failed(self, exc: BaseException) -> tuple:
        """(nouvelle erreur ?, faut-il s'arrêter ?)"""
        self.consecutive += 1
        key = signature(exc)
        new = key not in self.counts
        self.counts[key] = self.counts.get(key, 0) + 1
        return new, self.consecutive >= self.limit

    def ok(self):
        self.consecutive = 0
