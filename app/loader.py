"""Chargement des modèles 3D dans un thread (T19, sorti de main.py par T22).

Le fichier est lu et préparé (`visuals.mesh.prepare`) hors de la boucle de
rendu : l'animation continue et le modèle courant reste affiché jusqu'à ce
que le nouveau soit prêt.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from visuals import mesh as meshlib

DEFAULT_VISUAL = "primitive:torus"   # shown at launch and on U / « Retour au tore »


class ModelLoader:
    """Prépare un modèle à la fois pour `model_source` (un `ModelSource`)."""

    def __init__(self, model_source):
        self.model = model_source
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None     # (path, up, Future) of the model being prepared

    def load(self, path: str, up: int):
        """Lance la préparation de `path` avec l'axe haut `up` (index dans
        `mesh.UP_AXES`) ; un chemin introuvable ou d'un format non géré est
        signalé et ignoré."""
        if not path.startswith("primitive:") and not os.path.isfile(path):
            print(f"[warning] File not found: {path}")
            return
        if not meshlib.is_model_path(path):
            ext = os.path.splitext(path)[1].lower()
            print(f"[warning] Unsupported format: {ext or path} — 3D models "
                  f"{sorted(meshlib.MODEL_EXT)}")
            return
        print(f"  Loading model: {path} …")
        # the up axis travels with the job: if the panel changes it
        # meanwhile, gpu_uniforms() sees the mismatch and rebuilds
        self.pending = (path, up, self._pool.submit(
            meshlib.prepare, path, up,
            self.model.model_radius, self.model.max_points))

    def update(self):
        """À appeler chaque frame : installe le modèle quand il est prêt."""
        if self.pending is None or not self.pending[2].done():
            return
        path, up, fut = self.pending
        self.pending = None
        try:
            self.model.set_model(fut.result(), up)
        except Exception as e:  # noqa: BLE001 — frontière : résultat du thread de chargement
            print(f"[error] model load failed ({path}): {e}")

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
