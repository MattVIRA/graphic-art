"""
Visual source: a 3D model (T1.9) — the only one since T1.11 (the image and star
modes were removed). The default visual is the built-in torus.

Model space: centered on the origin, Y up, pixel units. The renderer applies
the perspective + rotation (MVP); all the motion is stateless in the vertex
shader (gfx/renderer.py), so this class only holds the buffers prepared by
visuals/mesh.py, advances the music-driven rotation and gathers the per-frame
uniforms.

  model_points  (N,10) points sampled on the surface      → style Points
  model_mesh    welded mesh (or None for a point cloud)   → Surface / Wireframe
"""

from visuals import mesh


class ModelSource:
    """Le modèle 3D affiché : buffers GPU, rotation musicale, uniforms par frame."""

    def __init__(self, width: int, height: int, max_points: int = 1_000_000):
        self.w = width
        self.h = height
        self.max_points = max_points  # point budget of the Points style

        self.mode = None              # None until a model is installed → "model"
        self.n = 0
        self.rev = 0                  # bumped on every rebuild → re-upload GPU buffers
        self.angle = 0.0              # accumulated rotation (rad)
        self.rot_speed = 0.004        # smoothed rotation speed (rad per 1/60 s)

        self._model = None            # mesh.RawModel (cached parsed file)
        self.model_points = None      # (N,10) static point buffer
        self.model_mesh = None        # surface/wire buffers, None = point cloud
        self.model_up = 0             # index in mesh.UP_AXES
        self.model_gain = 1.0         # brightness compensation of the points
        # bounding-sphere radius on screen (room left for the HUD + inflation)
        self.model_radius = 0.36 * height

    # ------------------------------------------------------------------

    def set_model(self, prepared: tuple, up: int):
        """Install a model prepared by mesh.prepare() in the worker thread."""
        self._model, self.model_points, self.model_mesh = prepared
        self.model_up = up
        self.n = len(self.model_points)
        self.model_gain = mesh.point_gain(self._model, self.model_radius, self.n)
        self.mode = "model"
        self.rev += 1

    def set_budget(self, n: int):
        """Change the point budget; resamples the cached model (no file read)."""
        n = max(10_000, int(n))
        if n == self.max_points:
            return
        self.max_points = n
        if self._model is not None:
            self._rebuild(points_only=True)

    def _rebuild(self, points_only: bool = False):
        self.model_points = mesh.build_points(self._model, self.model_up,
                                              self.model_radius, self.max_points)
        if not points_only:
            self.model_mesh = mesh.build_mesh(self._model, self.model_up,
                                              self.model_radius)
        self.n = len(self.model_points)
        self.model_gain = mesh.point_gain(self._model, self.model_radius, self.n)
        self.rev += 1

    # ------------------------------------------------------------------
    # GPU path: static attributes + per-frame uniforms

    def build_static(self):
        """(array (N,10) f4, count) — rest attributes for the point shader:
        pos3, color3, aux4 = [normal3, band]."""
        return self.model_points, self.n

    def gpu_uniforms(self, f, params=None, time_s: float = 0.0,
                     dt: float = 1.0 / 60.0) -> dict:
        """Advance the rotation and return the per-frame uniform dict.
        dt en secondes : les constantes sont réglées pour 60 fps, `k`
        ramène le pas réel à ce repère (T1.15 : même vitesse à 30 ou 120 fps)."""
        params = params or {}
        want_up = int(params.get("model_up", 0.0))
        if self._model is not None and want_up != self.model_up:
            self.model_up = want_up
            self._rebuild()

        # rotation speed driven by the HF transient density: slow during
        # breaks, fast during drops / busy drum sections. Smoothed so it ramps.
        target_speed = (0.0015 + f.get("transients", 0.0) * 0.035 + f["high"] * 0.008
                        + params.get("fx_rot", 0.0) * 0.05)
        k = dt * 60.0
        self.rot_speed += (target_speed - self.rot_speed) * (1.0 - 0.96 ** k)
        self.angle += self.rot_speed * params.get("rot_gain", 1.0) * k

        return {
            "u_inflate": params.get("inflate", 0.0),
            "u_mturb": params.get("model_turb", 0.0),
            "u_model_gain": self.model_gain,
            "u_time": time_s,
            "u_spectrum": f["spectrum"],
            "angle": self.angle,
            # a model keeps its exact shape: no built-in bass pulse (the
            # routed "Zoom (pulse)" effect still applies)
            "scale": 1.0,
        }
