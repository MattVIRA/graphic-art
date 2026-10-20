"""Passe GL du fond (T16) : un programme par niveau de qualité, tous
compilés au démarrage (pas de binaires de programme fiables sur macOS :
compiler en cours de route ferait une saccade). Plus une texture HDR à la
résolution du fond, une palette 1D, la texture 3D de bruit de la matière,
les deux cartes du relief (background/terrain.py, remplacées quand un
nouveau calcul est prêt) et la mesure du temps GPU. Passe 5 : le rayon est décalé
d'une fraction de pixel à chaque frame, puis `resolve.glsl` mélange la
frame à l'historique reprojeté (ping-pong à la résolution interne) et
l'agrandit en suivant les contours (`background/temporal.py`).

Les sources GLSL peuvent être modifiées pendant que l'appli tourne :
`reload_if_changed()` recompile, et garde les anciens programmes si la
compilation échoue (l'erreur remonte à l'appelant, qui la signale).
"""

import logging
import os
import time

import moderngl
import numpy as np

from background import noise, palettes, quality, schema, temporal, terrain

log = logging.getLogger(__name__)

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
SHADER_FILES = ["common.glsl", "landscape.glsl", "main.glsl"]   # dans cet ordre
RESOLVE_FILE = "resolve.glsl"            # passe 5 : accumulation + agrandissement
PALETTE_UNIT, NOISE_UNIT = 3, 6          # 4 et 5 : scène et profondeur (renderer)
CUR_UNIT, HIST_UNIT, HIST_AUX_UNIT = 7, 8, 9
TERRAIN_UNIT, FINE_UNIT = 10, 11         # cartes du relief

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


def resolve_source() -> str:
    return "#version 330\n" + _read(RESOLVE_FILE)


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
        self.terrain_tex = self.fine_tex = None
        self.terrain_rev = 0
        self._terrain_uniforms = {}
        self.set_terrain(terrain.flat())  # plaine, le temps du premier calcul
        self.tex = self.fbo = None
        # passe 5 : historique ping-pong à la résolution interne (couleur +
        # part devant le modèle ; poids + profondeur)
        self.history = temporal.History()
        self._hist = []
        self._hist_i = 0
        self._queries = [ctx.query(time=True), ctx.query(time=True)]
        self._pending = [False, False]
        self._qi = 0
        self.gpu_ms = None            # dernière mesure (frame précédente)
        self._warm_up()

    # ── programmes ────────────────────────────────────────────────────
    @staticmethod
    def _mtimes() -> tuple:
        return tuple(os.path.getmtime(os.path.join(SHADER_DIR, n))
                     for n in SHADER_FILES + [RESOLVE_FILE])

    def _compile_all(self) -> dict:
        """Programmes de marche par niveau, plus celui de la résolution
        (clé "resolve")."""
        t0 = time.perf_counter()
        progs = {}
        for lv in range(len(quality.LEVELS)):
            progs[lv] = self.ctx.program(vertex_shader=FULLSCREEN_VS,
                                         fragment_shader=fragment_source(lv))
        progs["resolve"] = self.ctx.program(vertex_shader=FULLSCREEN_VS,
                                            fragment_shader=resolve_source())
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
        self.history.forget()

    def _bind_textures(self):
        self.palette_tex.use(PALETTE_UNIT)
        self.noise_tex.use(NOISE_UNIT)
        self.terrain_tex.use(TERRAIN_UNIT)
        self.fine_tex.use(FINE_UNIT)

    def _map_texture(self, data: np.ndarray):
        """Carte du relief : float32 bouclée, mipmaps (lissée au loin)."""
        h, w = data.shape[:2]
        comps = 1 if data.ndim == 2 else data.shape[2]
        tex = self.ctx.texture((w, h), comps, np.ascontiguousarray(data).tobytes(),
                               dtype="f4")
        tex.repeat_x = tex.repeat_y = True
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return tex

    def set_terrain(self, maps):
        """Installe les cartes calculées (background.terrain.TerrainMap) ;
        l'historique du lissage repart de zéro (le décor a changé)."""
        for tex in (self.terrain_tex, self.fine_tex):
            if tex is not None:
                tex.release()
        self.terrain_tex = self._map_texture(maps.data)
        self.fine_tex = self._map_texture(maps.detail)
        self._terrain_uniforms = {
            "u_ld_res": (float(maps.data.shape[0]), float(maps.detail.shape[0])),
            "u_ld_slope": maps.slope, "u_ld_dslope": maps.detail_slope,
            "u_ld_dgain": maps.detail_gain, "u_ld_damp": maps.detail_amp,
        }
        self.terrain_rev += 1

    @staticmethod
    def _set_samplers(prog):
        for name, unit in (("u_palette", PALETTE_UNIT), ("u_noise", NOISE_UNIT),
                           ("u_terrain", TERRAIN_UNIT), ("u_terrain_fine", FINE_UNIT),
                           ("u_cur", CUR_UNIT), ("u_hist", HIST_UNIT),
                           ("u_hist_aux", HIST_AUX_UNIT), ("u_model_depth", 5)):
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
        self.history.forget()
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

    def _history_targets(self, size):
        """Deux jeux (couleur RGBA, poids + profondeur RG) à la taille de
        sortie ; recréés (et vidés : poids 0) quand la taille change."""
        if self._hist and self._hist[0][0].size == size:
            return
        for col, aux, fbo in self._hist:
            fbo.release()
            col.release()
            aux.release()
        self._hist = []
        for _ in range(2):
            col = self.ctx.texture(size, 4, dtype="f2")
            col.filter = (moderngl.LINEAR, moderngl.LINEAR)
            col.repeat_x = col.repeat_y = False
            aux = self.ctx.texture(size, 2, dtype="f2")
            aux.filter = (moderngl.NEAREST, moderngl.NEAREST)
            fbo = self.ctx.framebuffer(color_attachments=[col, aux])
            fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self._hist.append((col, aux, fbo))
        self.history.forget()

    @staticmethod
    def _write(prog, uniforms: dict):
        members = prog._members
        for name, value in uniforms.items():
            if name not in members:
                continue   # éliminé par le compilateur
            if isinstance(value, np.ndarray):   # tableaux d'uniforms
                prog[name].write(np.ascontiguousarray(value, dtype="f4").tobytes())
            else:
                prog[name].value = value

    def render(self, frame, size, camera: dict, proj, view, unit: float, out_size):
        """Dessine le fond (taille `size`) puis le résout vers `out_size`
        (la taille de rendu ; l'historique en prend au plus 1,5 × `size`) ;
        rend la texture résolue (rgb = fond, a = part devant le modèle).
        camera : u_inv_vp, u_eye, u_unit, u_pix, u_center_uv, u_aspect… ;
        proj, view : la même caméra (matrices 4×4), pour la reprojection.
        Profondeur du modèle sur l'unité 5 (renderer)."""
        u = frame.uniforms
        pose = temporal.Pose(
            vp=proj @ view, eye=np.linalg.inv(view)[:3, 3], fwd=-view[2, :3],
            unit=float(unit), far=float(u["u_comp_far"]),
            rot=np.asarray(u["u_rot"], dtype=np.float64).reshape(3, 3).T,
            offset=np.asarray(u["u_offset"], dtype=np.float64),
            scale=float(u["u_scale"]))
        out_size = temporal.history_size(size, out_size)
        self._history_targets(out_size)
        prev, n = self.history.step((size, frame.quality, frame.same_space,
                                     frame.structure, self.terrain_rev), pose)
        wmax = float(frame.history)
        jit = temporal.jitter(n) if wmax > 0.0 else (0.0, 0.0)

        prog = self.programs[quality.level(frame.quality)]
        if frame.palette_rev != self._palette_rev and frame.palette is not None:
            self.palette_tex.write(np.ascontiguousarray(frame.palette, dtype="f4").tobytes())
            self._palette_rev = frame.palette_rev
        self._write(prog, dict(frame.uniforms, **camera, **self._terrain_uniforms,
                               u_jitter=jit))
        if "u_zero" in prog._members:
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

        # résolution : accumulation temporelle + agrandissement
        src_col, src_aux, _ = self._hist[self._hist_i]
        dst_col, _, dst_fbo = self._hist[1 - self._hist_i]
        rp = self.programs["resolve"]
        self._write(rp, dict(temporal.resolve_uniforms(pose, prev),
                             u_near_far=camera["u_near_far"],
                             u_cur_size=tuple(float(v) for v in size),
                             u_out_size=tuple(float(v) for v in out_size),
                             u_jitter=jit, u_wmax=wmax,
                             u_reset=1.0 if prev is None else 0.0,
                             u_same_space=camera["u_same_space"]))
        self._set_samplers(rp)
        self.tex.use(CUR_UNIT)
        src_col.use(HIST_UNIT)
        src_aux.use(HIST_AUX_UNIT)
        dst_fbo.use()
        self._vao(rp).render(moderngl.TRIANGLES, vertices=3)
        self._hist_i = 1 - self._hist_i
        return dst_col

    def forget(self):
        """Fond masqué : l'historique repartira de zéro."""
        self.history.forget()
