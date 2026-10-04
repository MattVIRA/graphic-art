"""
GPU particle system — 3D (physics in numpy, rendering via gfx.renderer).

Model space: centered on the origin, Y up, pixel units. The renderer applies
the perspective + rotation (MVP), so here we only produce model-space positions.

Two modes
---------
IMAGE  : the picture stays recognizable. Pixels become particles kept at their
         original (x, y); rembg cuts out the main subject, which gets pushed
         toward the camera (foreground z+) while the background sits behind
         (z-). The whole thing rotates slowly in 3D → parallax depth.

STAR   : a rotating 3D "sea-urchin". n_mels spikes spread over a sphere
         (Fibonacci distribution); each spike's length is driven by its mel band.

MODEL  : a 3D model file (T19, visuals/mesh.py). Points sampled on the real
         surface (GPU only); the surface / wireframe styles are drawn by the
         renderer's mesh programs from `model_mesh`.

Frequency mapping (the chosen preset)
-------------------------------------
  bass     20–250 Hz   → element SIZE (+ global zoom pulse)
  low_mid  250–800 Hz  → 3D DEPTH separation (subject vs background)
  mid      800–4k Hz   → CONTRAST of the colors
  high     4k–20k Hz   → ROTATION speed
  10–11kHz             → particle SPEED (turbulence)
  11–12kHz             → DISTANCE (radial spread)
  12–14kHz             → SHIMMER (per-particle brightness)
"""

import numpy as np
from PIL import Image


def _hsv_to_rgb_array(h, s, v):
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def _apply_contrast(colors, mid):
    """Boost contrast around 0.5 by an amount driven by mid energy."""
    factor = 1.0 + mid * 1.3
    return np.clip((colors - 0.5) * factor + 0.5, 0.0, 1.0)


class ParticleSystem:
    def __init__(self, width: int, height: int, n_generative: int = 40000,
                 max_image: int = 90000, n_mels: int = 64):
        self.w = width
        self.h = height
        self.n_mels = n_mels
        self.max_image = max_image

        self.angle = 0.0          # accumulated Y rotation
        self.image_style = "points"   # "points" | "curves" (abstract line art)
        self._src = None              # cached source image arrays
        self.rev = 0                  # bumped on every rebuild → re-upload GPU buffer
        self.rot_speed = 0.004    # smoothed rotation speed (rad/frame)
        self._rembg_session = None
        self._jitter = None       # persistent smoothed jitter state (image mode)
        # T19: 3D model source
        self._model = None            # mesh.RawModel (cached parsed file)
        self.model_points = None      # (N,11) static point buffer
        self.model_mesh = None        # surface/wire buffers, None = point cloud
        self.model_up = 0             # index in mesh.UP_AXES
        # bounding-sphere radius on screen (room left for the HUD + inflation)
        self.model_radius = 0.36 * height

        self._init_star(n_generative)
        self._out = np.zeros((max(self.n, max_image), 7), dtype=np.float32)

    # ==================================================================
    # STAR (sea-urchin) mode
    # ==================================================================

    def _init_star(self, n_target: int):
        self.mode = "star"
        n_spikes = self.n_mels
        K = max(2, n_target // n_spikes)        # particles per spike

        # Fibonacci sphere → even spike directions
        idx = np.arange(n_spikes)
        phi = np.arccos(1.0 - 2.0 * (idx + 0.5) / n_spikes)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
        dirs = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ], axis=1).astype(np.float32)            # (n_spikes, 3)

        self.star_dir = np.repeat(dirs, K, axis=0)              # (n, 3)
        self.star_t = np.tile(np.linspace(0, 1, K, dtype=np.float32), n_spikes)
        self.star_band = np.repeat(idx, K).astype(int)
        self.n = n_spikes * K
        self.rev += 1

    def _update_star(self, f):
        spectrum = f["spectrum"]
        gain = spectrum[self.star_band]                         # (n,)

        r_inner = 70.0 * (1.0 + f["bass"] * 0.7)
        spike_len = 60.0 + gain * 260.0
        radius = r_inner + self.star_t * spike_len
        pos = self.star_dir * radius[:, None]                  # (n, 3)

        # colors: hue by band, brightness by gain, contrast by mid
        hue = (self.star_band / self.n_mels + f["centroid"] * 0.3) % 1.0
        val = np.clip(0.45 + gain * 0.65 + self.star_t * 0.1, 0, 1)
        colors = _hsv_to_rgb_array(hue, np.full(self.n, 0.9, np.float32), val)
        colors = _apply_contrast(colors, f["mid"])

        size = 2.0 + self.star_t * 1.5 + f["bass"] * 3.0 + f["rms"] * 1.5
        return pos, colors, size

    # ==================================================================
    # IMAGE mode
    # ==================================================================

    def _cutout_mask(self, img: Image.Image) -> np.ndarray:
        """Return a boolean foreground mask via rembg; fall back to all-foreground."""
        try:
            from rembg import remove, new_session
            if self._rembg_session is None:
                # BiRefNet: state-of-the-art neural matting, far cleaner edges
                # than u2net/isnet. Fallback chain if download/load fails.
                for model in ("birefnet-general", "isnet-general-use", "u2net"):
                    try:
                        print(f"  rembg: loading {model} (first run downloads it)…")
                        self._rembg_session = new_session(model)
                        break
                    except Exception as e:
                        print(f"  [warning] {model} unavailable ({e})")
            cut = remove(img, session=self._rembg_session)
            alpha = np.array(cut.convert("RGBA"))[:, :, 3]
            mask = alpha > 96
            return self._clean_mask(mask)
        except Exception as e:
            print(f"  [warning] rembg cutout failed ({e}); using flat depth")
            return np.ones((img.height, img.width), dtype=bool)

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup: drop speckles, close holes, keep big components."""
        try:
            import cv2
            m = mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_lbl > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                keep = np.where(areas >= max(areas.max() * 0.05,
                                             0.002 * mask.size))[0] + 1
                m = np.isin(labels, keep).astype(np.uint8)
            return m.astype(bool)
        except Exception:
            return mask

    def load_image(self, filepath: str, depth: float = 140.0, relief: float = 40.0):
        print(f"  Loading image: {filepath}")
        img = Image.open(filepath).convert("RGB")
        img.thumbnail((self.w, self.h), Image.LANCZOS)
        iw, ih = img.width, img.height

        fg_mask = self._cutout_mask(img)
        rgb = np.array(img).astype(np.float32) / 255.0           # (ih, iw, 3)
        lum = rgb.mean(axis=2)                                    # (ih, iw)

        # keep the source arrays: the points/curves representations are
        # (re)built from them without re-running rembg (e.g. budget change)
        self._src = {"rgb": rgb, "fg": fg_mask, "lum": lum,
                     "iw": iw, "ih": ih, "depth": depth, "relief": relief}
        if self.image_style == "curves":
            self._build_curves()
        else:
            self._build_points()

    def set_budget(self, n: int):
        """Change the particle budget; rebuilds the image representation
        (rembg mask is cached, only the point grid is recomputed)."""
        n = max(10_000, int(n))
        if n == self.max_image:
            return
        self.max_image = n
        if self._out.shape[0] < n:
            self._out = np.zeros((n, 7), dtype=np.float32)
        if self.mode == "image" and self._src is not None:
            (self._build_curves if self.image_style == "curves"
             else self._build_points)()
        elif self.mode == "model":
            self._rebuild_model(points_only=True)

    # ==================================================================
    # MODEL mode (T19)
    # ==================================================================

    def set_model(self, prepared: tuple, up: int):
        """Install a model prepared by mesh.prepare() in the worker thread."""
        from visuals import mesh
        self._model, self.model_points, self.model_mesh = prepared
        self.model_up = up
        self.n = len(self.model_points)
        self.model_gain = mesh.point_gain(self._model, self.model_radius, self.n)
        self.mode = "model"
        self.rev += 1

    def _rebuild_model(self, points_only: bool = False):
        from visuals import mesh
        self.model_points = mesh.build_points(self._model, self.model_up,
                                              self.model_radius, self.max_image)
        if not points_only:
            self.model_mesh = mesh.build_mesh(self._model, self.model_up,
                                              self.model_radius)
        self.n = len(self.model_points)
        self.model_gain = mesh.point_gain(self._model, self.model_radius, self.n)
        self.rev += 1

    def _build_points(self):
        s = self._src
        rgb, fg_mask = s["rgb"], s["fg"]
        iw, ih = s["iw"], s["ih"]
        depth, relief = s["depth"], s["relief"]
        lum = s["lum"]

        # densify: upscale so the pixel grid uses (nearly) the whole particle
        # budget — sub-pixel points sharpen the subject's definition. Model
        # coords are divided back by `up` so the on-screen size is unchanged.
        up = min(2.0, float(np.sqrt(self.max_image / (iw * ih))))
        if up > 1.05:
            try:
                import cv2
                iw, ih = int(iw * up), int(ih * up)
                rgb = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
                fg_mask = cv2.resize(fg_mask.astype(np.uint8), (iw, ih),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                lum = rgb.mean(axis=2)
            except Exception:
                up = 1.0
        else:
            up = 1.0

        # edge map: points on strong edges get a brightness/size boost so the
        # subject's outlines stay crisply drawn
        try:
            import cv2
            gray = (lum * 255).astype(np.uint8)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
            # the subject's silhouette is always an edge
            sil = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                   np.ones((3, 3), np.uint8))
            edge = ((edges > 0) | (sil > 0)).ravel().astype(np.float32)
        except Exception:
            edge = np.zeros(ih * iw, dtype=np.float32)

        # clean relief: a smoothed luminance heightmap gives the subject a
        # coherent embossed surface instead of per-pixel z noise
        try:
            import cv2
            relief_map = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 8)
        except Exception:
            relief_map = lum
        relief_z = (relief_map.ravel() - 0.5).astype(np.float32)

        ys, xs = np.mgrid[0:ih, 0:iw]
        xs = xs.ravel(); ys = ys.ravel()
        fg = fg_mask.ravel()
        colors = rgb.reshape(-1, 3)
        lum = lum.ravel()

        total = len(xs)
        if total > self.max_image:
            # Weighted sampling: the subject gets the lion's share of the particle
            # budget so it stays dense and recognizable; the background is sparser.
            fg_idx = np.flatnonzero(fg)
            bg_idx = np.flatnonzero(~fg)
            n_fg = min(len(fg_idx), int(self.max_image * 0.65))
            n_bg = min(len(bg_idx), self.max_image - n_fg)
            sel = np.concatenate([
                np.random.choice(fg_idx, n_fg, replace=False) if n_fg else fg_idx[:0],
                np.random.choice(bg_idx, n_bg, replace=False) if n_bg else bg_idx[:0],
            ])
            xs, ys, fg, colors, lum, edge, relief_z = (
                xs[sel], ys[sel], fg[sel], colors[sel], lum[sel], edge[sel],
                relief_z[sel])

        n = len(xs)
        # centered model coords, Y up (÷ up: the densified grid keeps its size)
        mx = (xs.astype(np.float32) - iw / 2.0) / up
        my = (ih / 2.0 - ys.astype(np.float32)) / up
        # depth: foreground toward camera (+), background away (−). The subject
        # gets the smooth luminance relief; the background stays a flat plane
        # so its points don't scatter in z.
        base_z = np.where(fg, depth, -depth).astype(np.float32)
        base_z = base_z + np.where(fg, relief_z * relief * 1.5, 0.0)

        # recenter on the SUBJECT's centroid (not the image center): the cut-out
        # object must sit exactly on the rotation origin, otherwise a subject
        # placed off-center in the photo looks like orbit_dist > 0.
        if fg.any():
            mx -= mx[fg].mean()
            my -= my[fg].mean()
            base_z -= base_z[fg].mean()

        self.is_fg = fg.astype(bool)
        self.edge = edge.astype(np.float32)
        self.home_xy = np.column_stack([mx, my]).astype(np.float32)
        self.home_z = base_z.astype(np.float32)
        self.base_color = colors.astype(np.float32)
        # angle of each particle around center (for turbulence sampling)
        self.theta = np.arctan2(my, mx).astype(np.float32)
        self.radial_dir = np.column_stack([
            np.cos(self.theta), np.sin(self.theta)]).astype(np.float32)

        self.n = n
        self.mode = "image"
        self._jitter = np.zeros((n, 3), dtype=np.float32)
        if n > self._out.shape[0]:
            self._out = np.zeros((n, 7), dtype=np.float32)
        self.rev += 1
        print(f"  Particles ready: {n}  (foreground {int(fg.sum())} / {n})")

    # ==================================================================
    # CURVES (abstract line-art) style
    # ==================================================================

    def _build_curves(self):
        """Edge-detect the image and turn the contours into dense polylines:
        the subject becomes abstract flowing curves instead of a point grid."""
        import cv2
        s = self._src
        rgb, fg_mask, lum = s["rgb"], s["fg"], s["lum"]
        iw, ih = s["iw"], s["ih"]
        depth, relief = s["depth"], s["relief"]

        # ── edge detection on a denoised image, thresholds auto-tuned ──
        img8 = (rgb * 255).astype(np.uint8)
        smooth_img = cv2.bilateralFilter(img8, 9, 75, 75)   # kill texture noise,
        gray = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY)  # keep real edges
        ref_px = gray[fg_mask] if fg_mask.any() and not fg_mask.all() else gray
        med = float(np.median(ref_px))
        edges = cv2.Canny(gray, max(20, 0.66 * med), max(60, 1.33 * med))

        raw = []
        # the subject's SILHOUETTE first: the strongest, cleanest curve
        sil, _ = cv2.findContours(fg_mask.astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        raw += [c[:, 0, :].astype(np.float32) for c in sil if len(c) >= 40]
        # then the inner detail edges, longest first
        inner, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        inner = [c[:, 0, :].astype(np.float32) for c in inner if len(c) >= 24]
        inner.sort(key=len, reverse=True)
        raw += inner[:400]

        def polish(c, step=1.6, k=9):
            """Smooth a pixel-chain polyline and resample it at uniform arc length."""
            pad = k // 2
            cp = np.pad(c, ((pad, pad), (0, 0)), mode="edge")
            kern = np.ones(k, dtype=np.float32) / k
            sm = np.column_stack([np.convolve(cp[:, j], kern, mode="valid")
                                  for j in range(2)])
            seg = np.hypot(*np.diff(sm, axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            length = s[-1]
            if length < 12.0:
                return None
            n = max(12, int(length / step))
            t = np.linspace(0.0, length, n)
            return np.column_stack([np.interp(t, s, sm[:, 0]),
                                    np.interp(t, s, sm[:, 1])]).astype(np.float32)

        pts, normals, tpar, band = [], [], [], []
        total = 0
        for ci, c in enumerate(raw):
            c = polish(c)
            if c is None:
                continue
            if total + len(c) > self.max_image:
                break
            d = np.gradient(c, axis=0)                      # tangents
            nrm = np.column_stack([-d[:, 1], d[:, 0]])      # perpendicular
            ln = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
            pts.append(c)
            normals.append(nrm / ln)
            tpar.append(np.linspace(0.0, 1.0, len(c), dtype=np.float32))
            band.append(np.full(len(c), ci % self.n_mels, dtype=int))
            total += len(c)

        if not pts:
            print("  [warning] no contours found; keeping point style")
            self._build_points()
            return

        P = np.concatenate(pts)                             # (n, 2) image coords
        N = np.concatenate(normals).astype(np.float32)
        xs = np.clip(P[:, 0].astype(int), 0, iw - 1)
        ys = np.clip(P[:, 1].astype(int), 0, ih - 1)

        fg = fg_mask[ys, xs]
        colors = rgb[ys, xs]
        z = np.where(fg, depth, -depth).astype(np.float32) \
            + (lum[ys, xs] - 0.5) * relief

        mx = P[:, 0] - iw / 2.0
        my = ih / 2.0 - P[:, 1]                             # flip to Y-up
        N[:, 1] = -N[:, 1]                                  # flip normals too
        ref = fg if fg.any() else np.ones(len(mx), dtype=bool)
        mx -= mx[ref].mean()
        my -= my[ref].mean()
        z -= z[ref].mean()

        self.cv_xy = np.column_stack([mx, my]).astype(np.float32)
        self.cv_z = z.astype(np.float32)
        self.cv_norm = N
        self.cv_t = np.concatenate(tpar)
        self.cv_band = np.concatenate(band)
        self.cv_color = colors.astype(np.float32)
        self.is_fg = fg.astype(bool)

        self.n = len(mx)
        self.mode = "image"
        if self.n > self._out.shape[0]:
            self._out = np.zeros((self.n, 7), dtype=np.float32)
        self.rev += 1
        print(f"  Curves ready: {len(pts)} contours, {self.n} points "
              f"(subject {int(fg.sum())} / {self.n})")

    def _update_curves(self, f, turb_gain: float = 1.0, depth: float = 0.5):
        # each contour is bound to a mel band: its energy makes the curve
        # undulate along its normals → flowing abstract line art
        wave = f["spectrum"][self.cv_band].astype(np.float32)
        phase = self.cv_t * 34.0 + self.angle * 6.0 + self.cv_band * 0.7
        disp = np.sin(phase) * (3.0 + wave * 42.0 * turb_gain)

        pos = np.empty((self.n, 3), dtype=np.float32)
        pos[:, 0] = self.cv_xy[:, 0] + self.cv_norm[:, 0] * disp
        pos[:, 1] = self.cv_xy[:, 1] + self.cv_norm[:, 1] * disp
        pos[:, 2] = self.cv_z * depth

        colors = self.cv_color * (0.40 + wave[:, None] * 0.55 + f["rms"] * 0.15)
        colors = _apply_contrast(np.clip(colors, 0, 1), f["mid"] * 0.5)

        size = 1.3 + wave * 0.8
        return pos, colors, size.astype(np.float32)

    def _update_image(self, f, turb_gain: float = 1.0, depth: float = 0.5,
                      edge_boost: float = 1.0):
        n = self.n
        # z range fully user-controlled (Épaisseur slider) — flat by default
        z = self.home_z * depth

        # distance: multiplicative stretch on 11–12 kHz — scales the layout
        # around the origin instead of pushing every particle outward by a
        # fixed offset (the old additive spread emptied the subject's middle)
        spread = 1.0 + f["hz_11_12"] * 0.08 * turb_gain
        # turbulence: smoothed per-particle jitter (EMA) instead of raw noise —
        # per-frame white noise destroyed the image legibility; kept small,
        # the traveling wave below is the main motion now
        turb = f["hz_10_11"] * (3.0 + f["rms"] * 6.0) * turb_gain * 0.25
        target = (np.random.rand(n, 3).astype(np.float32) - 0.5) * turb
        # the subject stays crisp; the background is allowed to swirl more
        target *= np.where(self.is_fg, 0.5, 1.4).astype(np.float32)[:, None]
        self._jitter += (target - self._jitter) * 0.15
        jitter = self._jitter

        # traveling wave across the image (replaces most of the random drift)
        self._wave_t = getattr(self, "_wave_t", 0.0) + 0.037
        ph = (self.home_xy[:, 0] + self.home_xy[:, 1] * 0.6) * 0.010 \
            - self._wave_t
        wv = np.sin(ph) * (4.0 + f["rms"] * 30.0
                           + f["hz_10_11"] * 24.0) * turb_gain

        pos = np.empty((n, 3), dtype=np.float32)
        pos[:, 0] = self.home_xy[:, 0] * spread + jitter[:, 0]
        pos[:, 1] = self.home_xy[:, 1] * spread + jitter[:, 1] + wv * 0.35
        pos[:, 2] = z + jitter[:, 2] + wv

        # contrast on mid, shimmer on 12–14 kHz
        colors = _apply_contrast(self.base_color, f["mid"] * 0.6)
        if f["hz_12_14"] > 0.25:
            shim = np.random.uniform(1.0 - f["hz_12_14"] * 0.4, 1.0,
                                     (n, 1)).astype(np.float32)
            colors = np.clip(colors * shim, 0, 1)
        # tame additive blowout: with ~90k overlapping additive points, full
        # brightness saturates to white — scale down, let bass push it up a bit.
        # The subject stays bright while the background is dimmed → it pops out.
        level = np.where(self.is_fg, 0.50 + f["bass"] * 0.22,
                                     0.26 + f["bass"] * 0.10).astype(np.float32)
        colors = colors * level[:, None]
        # edge points pop: brighter and bigger → crisply drawn outlines
        edge = getattr(self, "edge", None)
        if edge is not None and edge_boost > 0.0:
            colors = np.clip(colors * (1.0 + edge[:, None] * 0.9 * edge_boost),
                             0, 1)
        size = np.where(self.is_fg, 1.5 * 1.25, 1.5 * 0.85).astype(np.float32)
        if edge is not None and edge_boost > 0.0:
            size = size * (1.0 + edge * 0.6 * edge_boost)
        return pos, colors, size

    # ==================================================================
    # Common
    # ==================================================================

    def unload(self, n_generative: int = 40000):
        self._init_star(n_generative)

    # ==================================================================
    # T1 — stateless GPU path: static attributes + per-frame uniforms
    # ==================================================================

    def build_static(self):
        """(array (N,11) f4, mode int, count) — rest attributes for the GPU
        vertex shader. Layout: pos3, color3, aux4, fg1."""
        if self.mode == "image" and self.image_style == "curves":
            n = self.n
            out = np.zeros((n, 11), dtype=np.float32)
            out[:, 0:2] = self.cv_xy
            out[:, 2] = self.cv_z
            out[:, 3:6] = self.cv_color
            out[:, 6:8] = self.cv_norm
            out[:, 8] = self.cv_t
            out[:, 9] = self.cv_band
            out[:, 10] = self.is_fg
            return out, 1, n
        if self.mode == "image":
            n = self.n
            out = np.zeros((n, 11), dtype=np.float32)
            out[:, 0:2] = self.home_xy
            out[:, 2] = self.home_z
            out[:, 3:6] = self.base_color
            out[:, 6] = getattr(self, "edge", np.zeros(n, dtype=np.float32))
            out[:, 10] = self.is_fg
            return out, 0, n
        if self.mode == "model":
            return self.model_points, 3, self.n
        # star
        n = self.n
        out = np.zeros((n, 11), dtype=np.float32)
        out[:, 0:3] = self.star_dir
        out[:, 6] = self.star_t
        out[:, 7] = self.star_band
        out[:, 10] = 1.0
        return out, 2, n

    def gpu_uniforms(self, f, params=None, time_s: float = 0.0) -> dict:
        """Advance the rotation and return the per-frame uniform dict."""
        turb_gain = params.get("turbulence", 1.0) if params else 1.0
        rot_gain = params.get("rot_gain", 1.0) if params else 1.0

        # hot-switch styles here too (the GPU path bypasses update())
        want_curves = bool(params and params.get("curve_mode", 0.0) >= 0.5)
        want_style = "curves" if want_curves else "points"
        if want_style != self.image_style:
            self.image_style = want_style
            if self.mode == "image" and self._src is not None:
                (self._build_curves if want_curves else self._build_points)()

        want_up = int(params.get("model_up", 0.0)) if params else 0
        if self.mode == "model" and want_up != self.model_up:
            self.model_up = want_up
            self._rebuild_model()

        target_speed = (0.0015 + f.get("transients", 0.0) * 0.035 + f["high"] * 0.008
                        + (params.get("fx_rot", 0.0) if params else 0.0) * 0.05)
        self.rot_speed += (target_speed - self.rot_speed) * 0.04
        self.angle += self.rot_speed * rot_gain

        return {
            "u_inflate": params.get("inflate", 0.0) if params else 0.0,
            "u_mturb": params.get("model_turb", 0.0) if params else 0.0,
            "u_model_gain": getattr(self, "model_gain", 1.0),
            "u_time": time_s,
            "u_angle": self.angle,
            "u_turb": turb_gain,
            "u_hide_bg": params.get("hide_bg", 0.0) if params else 0.0,
            "u_bass": f["bass"], "u_low_mid": f["low_mid"], "u_mid": f["mid"],
            "u_rms": f["rms"], "u_centroid": f["centroid"],
            "u_hz10": f["hz_10_11"], "u_hz11": f["hz_11_12"],
            "u_hz12": f["hz_12_14"],
            "u_depth": params.get("depth", 0.5) if params else 0.5,
            "u_edge": params.get("edge_boost", 1.0) if params else 1.0,
            "u_spectrum": f["spectrum"],
            "angle": self.angle,
            # a model keeps its exact shape: no built-in bass pulse (the
            # routed "Zoom (pulse)" effect still applies)
            "scale": 1.0 if self.mode == "model" else 1.0 + f["bass"] * 0.12,
        }

    def update(self, f, params=None) -> tuple:
        """Return (out_buffer (N,7), count, render_params)."""
        turb_gain = params.get("turbulence", 1.0) if params else 1.0
        rot_gain = params.get("rot_gain", 1.0) if params else 1.0

        # hot-switch between point grid and curve art when the panel toggles
        want_curves = bool(params and params.get("curve_mode", 0.0) >= 0.5)
        want_style = "curves" if want_curves else "points"
        if want_style != self.image_style:
            self.image_style = want_style
            if self.mode == "image" and self._src is not None:
                (self._build_curves if want_curves else self._build_points)()

        if self.mode == "image":
            depth = params.get("depth", 0.5) if params else 0.5
            if self.image_style == "curves":
                pos, colors, size = self._update_curves(f, turb_gain, depth)
            else:
                pos, colors, size = self._update_image(
                    f, turb_gain, depth,
                    params.get("edge_boost", 1.0) if params else 1.0)
            # per-scope color grading (CPU fallback of the GPU path)
            if params and "_scope" in params:
                t_fg, t_bg, c_fg, c_bg, n_fg, n_bg = params["_scope"]
                fgc = self.is_fg[:, None]
                colors = colors * np.where(fgc, t_fg, t_bg)
                con = np.where(fgc, c_fg, c_bg)
                colors = np.clip((colors - 0.5) * con + 0.5, 0.0, 1.0)
                neg = np.where(fgc, n_fg, n_bg)
                colors = colors * (1.0 - neg) + (1.0 - colors) * neg
            if params and params.get("hide_bg", 0.0) >= 0.5:
                # keep only the cut-out subject
                keep = self.is_fg
                pos, colors = pos[keep], colors[keep]
                if isinstance(size, np.ndarray):
                    size = size[keep]
        else:
            pos, colors, size = self._update_star(f)

        # rotation speed driven by the HF transient density: slow during breaks,
        # fast during drops / busy drum sections. Smoothed so it ramps.
        target_speed = (0.0015 + f.get("transients", 0.0) * 0.035 + f["high"] * 0.008
                        + (params.get("fx_rot", 0.0) if params else 0.0) * 0.05)
        self.rot_speed += (target_speed - self.rot_speed) * 0.04
        self.angle += self.rot_speed * rot_gain

        out = self._out
        n = len(pos)
        out[:n, 0:3] = pos
        out[:n, 3:6] = colors
        out[:n, 6] = size

        render_params = {
            "angle": self.angle,
            "scale": 1.0 + f["bass"] * 0.12,     # bass zoom pulse → bigger elements
        }
        return out, n, render_params
