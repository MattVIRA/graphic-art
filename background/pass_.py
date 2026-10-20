"""Passe GL du fond (T16) : un programme par niveau de qualité, tous
compilés au démarrage (pas de binaires de programme fiables sur macOS :
compiler en cours de route ferait une saccade). La chaîne d'étages choisit
ses opérations par uniforms : changer de mode ne recompile rien. Plus une
texture HDR à la résolution du fond, une palette 1D, la texture 3D de
bruit du warp et la mesure du temps GPU.

Les sources GLSL peuvent être modifiées pendant que l'appli tourne :
`reload_if_changed()` recompile, et garde les anciens programmes si la
compilation échoue (l'erreur remonte à l'appelant, qui la signale).
"""

import logging
import os
import time

import moderngl
import numpy as np

from background import noise, palettes, quality, schema

log = logging.getLogger(__name__)

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
SHADER_FILES = ["common.glsl", "chain.glsl", "main.glsl"]   # dans cet ordre
PALETTE_UNIT, NOISE_UNIT = 3, 6          # 4 et 5 : scène et profondeur (renderer)

FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


def _read(name: str) -> str:
    with open(os.path.join(SHADER_DIR, name), encoding="utf-8") as f:
        return f.read()


def fragment_source(level: int) -> str:
    """Shader complet à un niveau de qualité."""
    defines = "".join(f"#define {k} {v}\n" for k, v in quality.LEVELS[level].items())
    return ("#version 330\n" + defines + schema.glsl_declarations()
            + "".join(_read(name) for name in SHADER_FILES))


class BackgroundPass:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._stamp = self._mtimes()
        self.programs = self._compile_all()
        self._vaos = {}
        self.palette_tex = ctx.texture((palettes.SIZE, 1), 3, dtype="f4")
        self.palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.palette_tex.repeat_x = self.palette_tex.repeat_y = False
        self._palette_rev = -1
        self.noise_tex = ctx.texture3d((noise.SIZE,) * 3, 3, noise.texture().tobytes(),
                                       dtype="f4")
        self.noise_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.noise_tex.repeat_x = self.noise_tex.repeat_y = self.noise_tex.repeat_z = True
        self.tex = self.fbo = None
        self._queries = [ctx.query(time=True), ctx.query(time=True)]
        self._pending = [False, False]
        self._qi = 0
        self.gpu_ms = None            # dernière mesure (frame précédente)
        self._warm_up()

    # ── programmes ────────────────────────────────────────────────────
    @staticmethod
    def _mtimes() -> tuple:
        return tuple(os.path.getmtime(os.path.join(SHADER_DIR, n)) for n in SHADER_FILES)

    def _compile_all(self) -> dict:
        t0 = time.perf_counter()
        progs = {}
        for lv in range(len(quality.LEVELS)):
            progs[lv] = self.ctx.program(vertex_shader=FULLSCREEN_VS,
                                         fragment_shader=fragment_source(lv))
        log.info("fond : %d programmes compilés en %.0f ms", len(progs),
                 (time.perf_counter() - t0) * 1000.0)
        return progs

    def _warm_up(self):
        """Un dessin minuscule par programme : le pilote finit sa compilation
        au premier dessin, pas pendant la musique."""
        tex = self.ctx.texture((4, 4), 4, dtype="f2")
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        self._bind_textures()
        for prog in self.programs.values():
            self._set_samplers(prog)
            self._vao(prog).render(moderngl.TRIANGLES, vertices=3)
        fbo.release()
        tex.release()

    def _bind_textures(self):
        self.palette_tex.use(PALETTE_UNIT)
        self.noise_tex.use(NOISE_UNIT)

    @staticmethod
    def _set_samplers(prog):
        for name, unit in (("u_palette", PALETTE_UNIT), ("u_noise", NOISE_UNIT)):
            if name in prog._members:
                prog[name].value = unit

    def reload_if_changed(self) -> bool:
        """Recompile si un fichier .glsl a changé. Lève moderngl.Error si la
        nouvelle version ne compile pas (les anciens programmes restent)."""
        stamp = self._mtimes()
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        progs = self._compile_all()
        for prog in self.programs.values():
            prog.release()
        self.programs, self._vaos = progs, {}
        return True

    def _vao(self, prog):
        vao = self._vaos.get(id(prog))
        if vao is None:
            vao = self._vaos[id(prog)] = self.ctx.vertex_array(prog, [])
        return vao

    # ── dessin ────────────────────────────────────────────────────────
    def _target(self, size):
        if self.tex is None or self.tex.size != size:
            if self.tex is not None:
                self.fbo.release()
                self.tex.release()
            self.tex = self.ctx.texture(size, 4, dtype="f2")
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex.repeat_x = self.tex.repeat_y = False
            self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])
        return self.fbo

    def render(self, frame, size, camera: dict):
        """Dessine le fond dans sa texture (rendue) ; camera : u_inv_vp,
        u_eye, u_unit, u_pix, u_center_uv, u_aspect."""
        prog = self.programs[quality.level(frame.quality)]
        if frame.palette_rev != self._palette_rev and frame.palette is not None:
            self.palette_tex.write(np.ascontiguousarray(frame.palette, dtype="f4").tobytes())
            self._palette_rev = frame.palette_rev
        members = prog._members
        for name, value in list(frame.uniforms.items()) + list(camera.items()):
            if name not in members:
                continue   # éliminé par le compilateur
            if isinstance(value, np.ndarray):   # tableaux d'uniforms
                prog[name].write(np.ascontiguousarray(value, dtype="f4").tobytes())
            else:
                prog[name].value = value
        if "u_zero" in members:
            prog["u_zero"].value = 0
        self._set_samplers(prog)
        self._bind_textures()
        self._target(size).use()
        q = self._queries[self._qi]
        with q:
            self._vao(prog).render(moderngl.TRIANGLES, vertices=3)
        self._pending[self._qi] = True
        # lire la mesure de la frame précédente : celle-ci n'est pas finie
        other = 1 - self._qi
        if self._pending[other]:
            self.gpu_ms = self._queries[other].elapsed / 1e6
            self._pending[other] = False
        self._qi = other
        return self.tex
