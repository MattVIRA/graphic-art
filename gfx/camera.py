"""
Caméra orbitale à la souris (T26) — calculs seuls, sans GL ni GLFW.

La caméra tourne autour d'une cible (lacet / tangage borné), à une distance
`base_dist / zoom`. Vue par défaut (lacet = tangage = 0, cible à l'origine,
zoom 1) = l'ancienne vue fixe `translate(0, 0, -base_dist)`.

Mouvements amortis : chaque entrée déplace un *but*, l'état courant le suit
avec une constante de temps courte ; au relâchement d'une orbite, la vitesse
du geste continue sur le but puis s'éteint (la caméra glisse puis s'arrête).
"""

import math

import numpy as np

PITCH_LIMIT = math.radians(85.0)   # jamais à la verticale : pas de retournement
ORBIT_RAD_PER_PX = 0.006           # ≈ 0,34° par pixel logique
ZOOM_MIN, ZOOM_MAX = 0.2, 4.0
FOLLOW_TAU = 0.07                  # s : suivi du but pendant le geste
RESET_TAU = 0.22                   # s : retour à la vue par défaut, plus doux
GLIDE_TAU = 0.25                   # s : extinction de l'élan après relâchement


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Matrice de vue 4×4 (ligne-major, agit comme M @ vecteur colonne)."""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float64)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = (-s @ eye, -u @ eye, f @ eye)
    return M


def _ease(cur, goal, dt, tau):
    return cur + (goal - cur) * (1.0 - math.exp(-dt / tau))


class OrbitCamera:
    def __init__(self):
        self.yaw = self.pitch = 0.0          # état affiché (rad)
        self.target = np.zeros(3)
        self.zoom = 1.0
        self._goal_yaw = self._goal_pitch = 0.0
        self._goal_target = np.zeros(3)
        self._goal_zoom = 1.0
        self._vel = np.zeros(2)              # élan (lacet, tangage) en rad/s
        self._gesture = np.zeros(2)          # rotation reçue depuis la frame précédente
        self._orbiting = False
        self._tau = FOLLOW_TAU
        self._dist_scale = 1.0               # unités monde par pixel logique

    # ── entrées ───────────────────────────────────────────────────────
    def begin_orbit(self):
        self._orbiting = True
        self._vel[:] = 0.0
        self._tau = FOLLOW_TAU

    def end_orbit(self):
        self._orbiting = False   # l'élan mesuré pendant le geste continue

    def orbit(self, dx_px: float, dy_px: float):
        """Glisser vers la droite → la caméra passe à gauche (le modèle
        semble tourner vers la droite) ; vers le bas → vue plus plongeante."""
        d = np.array([-dx_px, dy_px]) * ORBIT_RAD_PER_PX
        self._gesture += d
        self._goal_yaw += d[0]
        self._goal_pitch = min(max(self._goal_pitch + d[1], -PITCH_LIMIT),
                               PITCH_LIMIT)
        self._tau = FOLLOW_TAU

    def pan(self, dx_px: float, dy_px: float):
        """Déplace la cible dans le plan de l'écran : le décor suit le
        curseur, quelle que soit la distance."""
        right, up, _ = self._basis(self._goal_yaw, self._goal_pitch)
        self._goal_target += (-right * dx_px + up * dy_px) * self._dist_scale
        self._vel[:] = 0.0
        self._tau = FOLLOW_TAU

    def zoom_by(self, steps: float):
        """Molette / trackpad : >0 rapproche."""
        self._goal_zoom = min(max(self._goal_zoom * 1.06 ** steps, ZOOM_MIN),
                              ZOOM_MAX)
        self._tau = FOLLOW_TAU

    def reset(self):
        """Retour en douceur à la vue par défaut, par le plus court chemin."""
        turn = 2.0 * math.pi
        k = round(self.yaw / turn)
        self.yaw -= k * turn
        self._goal_yaw = 0.0
        self._goal_pitch = 0.0
        self._goal_target = np.zeros(3)
        self._goal_zoom = 1.0
        self._vel[:] = 0.0
        self._orbiting = False
        self._tau = RESET_TAU

    # ── animation ─────────────────────────────────────────────────────
    def update(self, dt: float):
        dt = max(dt, 1e-4)
        if self._orbiting:
            # vitesse du geste, lissée : c'est l'élan donné au relâchement
            self._vel += (self._gesture / dt - self._vel) * min(1.0, dt / 0.05)
        elif self._vel.any():
            step = self._vel * dt
            self._goal_yaw += step[0]
            self._goal_pitch = min(max(self._goal_pitch + step[1],
                                       -PITCH_LIMIT), PITCH_LIMIT)
            self._vel *= math.exp(-dt / GLIDE_TAU)
            if np.abs(self._vel).max() < 1e-3:
                self._vel[:] = 0.0
        self._gesture[:] = 0.0
        tau = self._tau
        self.yaw = _ease(self.yaw, self._goal_yaw, dt, tau)
        self.pitch = _ease(self.pitch, self._goal_pitch, dt, tau)
        self.target = _ease(self.target, self._goal_target, dt, tau)
        # zoom interpolé en log : même douceur en avant et en arrière
        self.zoom = math.exp(_ease(math.log(self.zoom),
                                   math.log(self._goal_zoom), dt, tau))

    # ── sortie ────────────────────────────────────────────────────────
    @staticmethod
    def _basis(yaw, pitch):
        """(droite, haut, direction caméra → cible)."""
        cp = math.cos(pitch)
        back = np.array([cp * math.sin(yaw), math.sin(pitch), cp * math.cos(yaw)])
        fwd = -back
        right = np.cross(fwd, (0.0, 1.0, 0.0))
        right /= np.linalg.norm(right)
        return right, np.cross(right, fwd), fwd

    def view_matrix(self, base_dist: float, extra_zoom: float = 1.0,
                    shake=(0.0, 0.0)) -> np.ndarray:
        """base_dist : distance de la vue par défaut (cadrage du modèle) ;
        extra_zoom : slider Zoom du panneau ; shake : décalage écran (px) de
        la « Secousse caméra »."""
        z = max(self.zoom * extra_zoom, 0.05)
        dist = base_dist / z
        self._dist_scale = 1.0 / z
        _, _, fwd = self._basis(self.yaw, self.pitch)
        eye = self.target - fwd * dist
        V = look_at(eye, self.target)
        V[0, 3] += shake[0]
        V[1, 3] += shake[1]
        return V

    @property
    def is_default(self) -> bool:
        return (abs(self._goal_yaw) < 1e-6 and abs(self._goal_pitch) < 1e-6
                and not self._goal_target.any() and self._goal_zoom == 1.0)

    # ── autosave (pas dans les presets) ───────────────────────────────
    def state(self) -> dict:
        return {"yaw": float(self._goal_yaw), "pitch": float(self._goal_pitch),
                "target": [float(v) for v in self._goal_target],
                "zoom": float(self._goal_zoom)}

    def load_state(self, s):
        """Restaure une position sauvée ; ignore une valeur invalide."""
        try:
            yaw, pitch = float(s["yaw"]), float(s["pitch"])
            target = np.array([float(v) for v in s["target"]][:3])
            zoom = float(s["zoom"])
        except (TypeError, KeyError, ValueError, IndexError):
            return
        if target.shape != (3,) or not np.isfinite([yaw, pitch, zoom, *target]).all():
            return
        self.yaw = self._goal_yaw = math.remainder(yaw, 2.0 * math.pi)
        self.pitch = self._goal_pitch = min(max(pitch, -PITCH_LIMIT), PITCH_LIMIT)
        self.target, self._goal_target = target.copy(), target.copy()
        self.zoom = self._goal_zoom = min(max(zoom, ZOOM_MIN), ZOOM_MAX)
