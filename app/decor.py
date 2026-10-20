"""Brique « fond » de l'application (T16) : l'état du fond (sans GL), la
passe GL, le calcul du relief et la résolution automatique, reliés au
renderer.

Les shaders sont compilés au démarrage ; s'ils sont illisibles, le fond est
désactivé (message) et l'appli continue. Un .glsl modifié pendant que
l'appli tourne est recompilé ; s'il ne compile pas, l'ancien reste affiché.
Les cartes du relief sont calculées dans un thread : le paysage affiché
reste en place jusqu'à ce que le nouveau soit prêt.
"""

import logging

import moderngl

from background import quality
from background.params import BackgroundState
from background.pass_ import BackgroundPass
from background.schema import RESOLUTIONS
from background.terrain import TerrainBuilder

log = logging.getLogger(__name__)

RELOAD_PERIOD_S = 1.0   # vérification des .glsl modifiés


class Decor:
    def __init__(self, ctx: moderngl.Context, report, sync_terrain: bool = False):
        """sync_terrain : relief calculé tout de suite (banc de rendu)."""
        self.report = report
        self.state = BackgroundState()
        self.auto = quality.AutoResolution()
        self.terrain = TerrainBuilder(report, sync=sync_terrain)
        self.res_index = None          # résolution effective (Auto peut la baisser)
        self._next_reload = 0.0
        try:
            self.gl = BackgroundPass(ctx)
        except moderngl.Error as e:
            self.gl = None
            report("Fond : les shaders ne compilent pas, fond désactivé", e)

    @property
    def gpu_ms(self):
        """Temps GPU du fond à la dernière frame dessinée (None = masqué)."""
        return self.gl.gpu_ms if self.gl is not None and self.res_index is not None else None

    def _reload(self, now: float):
        if self.gl is None or now < self._next_reload:
            return
        self._next_reload = now + RELOAD_PERIOD_S
        try:
            if self.gl.reload_if_changed():
                log.info("fond : shaders rechargés")
        except moderngl.Error as e:
            self.report("Fond : erreur de compilation, l'ancien shader reste affiché", e)

    def frame(self, renderer, p: dict, features: dict, bands, dt: float,
              view, unit: float, now: float):
        """Une frame : état du fond, couleur unie, et passe du fond posée sur
        le renderer (avant `end_frame`). `view` = vue de la caméra du fond."""
        if p["bg_visible"] < 0.5 or self.gl is None:
            # fond masqué : rien à calculer, l'état repartira de zéro
            renderer.bg = (p["bg_r"] / 255.0, p["bg_g"] / 255.0, p["bg_b"] / 255.0)
            renderer.set_background()
            self.state.values = None
            self.res_index = None
            if self.gl is not None:
                self.gl.forget()
            return None
        f = self.state.step(p, features, bands, dt, model_hue=p.get("rim_hue", 0.0))
        renderer.bg = f.solid
        self._reload(now)
        self.terrain.request(f.terrain)
        maps = self.terrain.take()
        if maps is not None:
            self.gl.set_terrain(maps)
        self.res_index = self.auto.step(f.res_index, f.quality == quality.AUTO,
                                        self.gl.gpu_ms, dt)
        renderer.set_background(self.gl, f, view, RESOLUTIONS[self.res_index][1], unit)
        return f
