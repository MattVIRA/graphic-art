"""Chargement des modèles 3D dans un thread (T19, sorti de main.py par T22).

Le fichier est lu et préparé (`visuals.mesh.prepare`) hors de la boucle de
rendu : l'animation continue et le modèle courant reste affiché jusqu'à ce
que le nouveau soit prêt. Un chemin invalide lève `UserError` tout de
suite ; un fichier illisible n'est connu qu'à la fin du thread et passe par
`report(message, exc)`.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from app.errors import UserError
from visuals import mesh as meshlib

log = logging.getLogger(__name__)

DEFAULT_VISUAL = "primitive:torus"   # shown at launch and on U / « Retour au tore »


class ModelLoader:
    """Prépare les modèles demandés pour `model_source` (un `ModelSource`),
    un à la fois ; seul le dernier demandé est installé."""

    def __init__(self, model_source, report=None):
        self.model = model_source
        self._pool = ThreadPoolExecutor(max_workers=1)
        # (path, up, Future) dans l'ordre des demandes : chaque job est suivi
        # jusqu'au bout, pour qu'une erreur ne se perde pas quand un autre
        # modèle est demandé entre-temps
        self._jobs = []
        self._report = report or (lambda message, exc=None:
                                  log.error("%s", message, exc_info=exc))

    @property
    def busy(self) -> bool:
        return bool(self._jobs)

    def load(self, path: str, up: int):
        """Lance la préparation de `path` avec l'axe haut `up` (index dans
        `mesh.UP_AXES`). Introuvable ou format non géré → `UserError`."""
        if not path.startswith("primitive:") and not os.path.isfile(path):
            raise UserError(f"Modèle introuvable : {path}")
        if not meshlib.is_model_path(path):
            ext = os.path.splitext(path)[1].lower()
            raise UserError(f"Format non géré : {ext or path} — modèles 3D "
                            f"{' '.join(sorted(meshlib.MODEL_EXT))}")
        if not path.startswith("primitive:") and os.path.getsize(path) == 0:
            raise UserError(f"Modèle vide (0 octet) : {os.path.basename(path)}")
        log.info("Chargement du modèle : %s", path)
        # the up axis travels with the job: if the panel changes it
        # meanwhile, gpu_uniforms() sees the mismatch and rebuilds
        self._jobs.append((path, up, self._pool.submit(
            meshlib.prepare, path, up,
            self.model.model_radius, self.model.max_points)))

    def update(self):
        """À appeler chaque frame : signale les échecs, installe le dernier
        modèle demandé quand il est prêt (les précédents sont abandonnés)."""
        while self._jobs and self._jobs[0][2].done():
            path, up, fut = self._jobs.pop(0)
            try:
                prepared = fut.result()
            except Exception as e:  # noqa: BLE001 — frontière : résultat du thread de chargement
                self._report(f"Modèle illisible : {os.path.basename(path)} "
                             f"({type(e).__name__}: {e})", e)
                continue
            if not self._jobs:          # le plus récent : on l'affiche
                self.model.set_model(prepared, up)

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
