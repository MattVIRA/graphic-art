"""
3D model source (T19) — loading, normalization and GPU buffers.

Knows neither pygame nor moderngl. Two stages:
  load_model(path)  → RawModel   heavy (file parsing, texture → vertex colors,
                                 welding, decimation, edges): run it in a
                                 worker thread.
  build_points / build_mesh      light (orientation, scale, surface sampling):
                                 run on the main thread, re-run when the up
                                 axis or the point budget changes.

Model space matches the rest of the app: centered on the origin, Y up, pixel
units (the bounding sphere gets a fixed radius, whatever the file's units).
"""

import os

import numpy as np

MODEL_EXT = {".glb", ".gltf", ".obj", ".stl", ".ply"}
PRIMITIVES = ("icosphere", "torus", "box")
UP_AXES = ["Y (glTF)", "Z (STL / scan)", "−Y", "−Z"]

MAX_RENDER_FACES = 2_000_000    # above: decimated for the surface/wire styles
DECIMATE_TO = 1_000_000
SHARP_ANGLE = np.radians(30.0)  # "arêtes vives" dihedral threshold
GREY = 0.78
NO_UV = -1000.0                 # uv.x below -10 → the shader uses vertex color
N_BANDS = 64


def is_model_path(path: str) -> bool:
    return (path.startswith("primitive:")
            or os.path.splitext(path)[1].lower() in MODEL_EXT)


class _Part:
    """One geometry of the file, in the file's own coordinates."""

    def __init__(self, tm=None, points=None, vcolor=None, uv=None, tex=None):
        self.tm = tm              # trimesh.Trimesh, or None for a point cloud
        self.points = points      # (V,3) for a point cloud
        self.vcolor = vcolor      # (V,3) float 0..1
        self.uv = uv              # (V,2) or None
        self.tex = tex            # (H,W,3) float 0..1 or None

    @property
    def area(self):
        return float(self.tm.area) if self.tm is not None else 0.0


class RawModel:
    def __init__(self, name, parts, render):
        self.name = name
        self.parts = parts
        # welded render mesh (None for a pure point cloud):
        # dict(v, n, c, tris, edges, sharp) in file coordinates
        self.render = render
        allv = np.concatenate([p.tm.vertices if p.tm is not None else p.points
                               for p in parts]).astype(np.float64)
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        self.center = (lo + hi) * 0.5
        self.radius = float(np.linalg.norm(allv - self.center, axis=1).max()) or 1.0


# ── loading (worker thread) ──────────────────────────────────────────────────

def _primitive(kind: str):
    import trimesh
    if kind == "icosphere":
        return trimesh.creation.icosphere(subdivisions=5)
    if kind == "torus":
        return trimesh.creation.torus(1.0, 0.38, major_sections=256,
                                      minor_sections=96)
    if kind == "box":
        # subdivided so the vertex-shader deformations have vertices to move
        return trimesh.creation.box((1.2, 1.2, 1.2)).subdivide_to_size(0.04)
    raise ValueError(f"unknown primitive '{kind}' (choices: {', '.join(PRIMITIVES)})")


def _vertex_colors(g) -> np.ndarray:
    n = len(g.vertices)
    grey = np.full((n, 3), GREY, dtype=np.float32)
    try:
        vis = g.visual
        if vis.kind == "texture":
            c = vis.to_color().vertex_colors
        elif vis.kind in ("vertex", "face"):
            c = vis.vertex_colors
        else:
            return grey
        c = np.asarray(c, dtype=np.float32)[:, :3] / 255.0
        return c if len(c) == n else grey
    except Exception:
        return grey


def _texture(g):
    """(uv (V,2), image (H,W,3) float) of a textured mesh, else (None, None)."""
    try:
        vis = g.visual
        if vis.kind != "texture" or vis.uv is None or len(vis.uv) != len(g.vertices):
            return None, None
        mat = vis.material
        img = getattr(mat, "baseColorTexture", None)
        if img is None:
            img = getattr(mat, "image", None)
        if img is None:
            return None, None
        tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        fac = getattr(mat, "baseColorFactor", None)
        if fac is not None:
            tex = tex * (np.asarray(fac, dtype=np.float32)[:3] / 255.0)
        return np.asarray(vis.uv, dtype=np.float32), tex
    except Exception:
        return None, None


def _to_part(g):
    import trimesh
    if isinstance(g, trimesh.PointCloud):
        pts = np.asarray(g.vertices, dtype=np.float32)
        if len(pts) == 0:
            return None
        try:
            c = np.asarray(g.colors, dtype=np.float32)[:, :3] / 255.0
            if len(c) != len(pts):
                raise ValueError
        except Exception:
            c = np.full((len(pts), 3), GREY, dtype=np.float32)
        return _Part(points=pts, vcolor=c)
    if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
        return None
    uv, tex = _texture(g)
    return _Part(tm=g, vcolor=_vertex_colors(g), uv=uv, tex=tex)


def _build_render(parts):
    """Weld all mesh parts into one indexed mesh (smooth normals, no cracks
    when the vertex shader pushes vertices along their normals)."""
    import trimesh
    meshes = [p for p in parts if p.tm is not None]
    if not meshes:
        return None
    # one texture per render mesh: the largest textured part wins, the other
    # parts keep their (baked) vertex colors
    tex_part = max((p for p in meshes if p.tex is not None),
                   key=lambda p: p.area, default=None)
    vs, fs, cs, uvs, off = [], [], [], [], 0
    for p in meshes:
        vs.append(p.tm.vertices)
        fs.append(p.tm.faces + off)
        cs.append(p.vcolor)
        if p is tex_part:
            uvs.append(np.column_stack([p.uv[:, 0], 1.0 - p.uv[:, 1]]))
        else:
            uvs.append(np.full((len(p.tm.vertices), 2), NO_UV, dtype=np.float32))
        off += len(p.tm.vertices)
    V, F = np.concatenate(vs), np.concatenate(fs)
    colors0 = np.concatenate(cs)

    # Textured: keep the vertices split along UV seams (a texture needs them)
    # but compute normals / edges on the position-welded mesh, so the vertex
    # shader still deforms both sides of a seam identically (no cracks).
    textured = tex_part is not None and len(F) <= MAX_RENDER_FACES
    if textured:
        from trimesh.grouping import unique_rows
        uniq, inv = unique_rows(V)
        m = trimesh.Trimesh(V[uniq], inv[F], process=False)
        colors = colors0[uniq]
    else:
        m = trimesh.Trimesh(V, F,
                            vertex_colors=(np.clip(colors0, 0, 1) * 255).astype(np.uint8),
                            process=False)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        colors = np.asarray(m.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0

    if len(m.faces) > MAX_RENDER_FACES:
        print(f"  [warning] {len(m.faces):,} triangles — decimating to "
              f"{DECIMATE_TO:,} for the surface/wire styles")
        src_v = np.asarray(m.vertices)
        m = m.simplify_quadric_decimation(face_count=DECIMATE_TO)
        from scipy.spatial import cKDTree
        _, idx = cKDTree(src_v).query(m.vertices)
        colors = colors[idx]

    edges = m.edges_unique
    ao = _cavity(np.asarray(m.vertices), np.asarray(m.vertex_normals), edges)
    # sharp edges: dihedral angle above the threshold + open boundaries
    sharp = m.face_adjacency_edges[m.face_adjacency_angles > SHARP_ANGLE]
    from trimesh.grouping import group_rows
    boundary = m.edges_sorted[group_rows(m.edges_sorted, require_count=1)]
    sharp = np.concatenate([sharp, boundary]) if len(boundary) else sharp

    if textured:
        # back to the seam-split vertices; edges point at one representative
        # of each welded position (same position → same deformation)
        return {
            "v": np.asarray(V, dtype=np.float64),
            "n": np.asarray(m.vertex_normals, dtype=np.float32)[inv],
            "ao": ao[inv],
            "c": colors0.astype(np.float32),
            "uv": np.concatenate(uvs).astype(np.float32),
            "tex": (np.clip(tex_part.tex, 0.0, 1.0) * 255).astype(np.uint8),
            "tris": np.asarray(F, dtype=np.uint32),
            "edges": uniq[np.asarray(edges)].astype(np.uint32),
            "sharp": uniq[np.asarray(sharp)].astype(np.uint32),
        }
    return {
        "v": np.asarray(m.vertices, dtype=np.float64),
        "n": np.asarray(m.vertex_normals, dtype=np.float32),
        "ao": ao,
        "c": colors.astype(np.float32),
        "uv": None,
        "tex": None,
        "tris": np.asarray(m.faces, dtype=np.uint32),
        "edges": np.asarray(edges, dtype=np.uint32),
        "sharp": np.asarray(sharp, dtype=np.uint32),
    }


def _cavity(v: np.ndarray, n: np.ndarray, edges: np.ndarray,
            smooth_iters: int = 4) -> np.ndarray:
    """Per-vertex cavity occlusion in [0, 1] (1 = open, lower = crease).

    For each vertex, the mean of dot(normal, direction to each neighbour):
    neighbours rising above the tangent plane mean a concave spot (folds,
    eye sockets, between fingers). Smoothed over the 1-ring a few times so
    the darkening spreads like ambient occlusion; normalized per model."""
    a, b = edges[:, 0], edges[:, 1]
    d = v[b] - v[a]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    acc = np.zeros(len(v))
    cnt = np.zeros(len(v))
    np.add.at(acc, a, np.einsum("ij,ij->i", n[a], d))
    np.add.at(acc, b, -np.einsum("ij,ij->i", n[b], d))
    np.add.at(cnt, a, 1.0)
    np.add.at(cnt, b, 1.0)
    cav = acc / np.maximum(cnt, 1.0)
    for _ in range(smooth_iters):
        s = np.zeros(len(v))
        np.add.at(s, a, cav[b])
        np.add.at(s, b, cav[a])
        cav = 0.5 * cav + 0.5 * s / np.maximum(cnt, 1.0)
    concave = np.maximum(cav, 0.0)
    hi = float(np.percentile(concave, 98)) if len(concave) else 0.0
    if hi <= 1e-6:
        return np.ones(len(v), dtype=np.float32)
    return (1.0 - np.clip(concave / hi, 0.0, 1.0) * 0.85).astype(np.float32)


def load_model(path: str) -> RawModel:
    """Parse a model file (or 'primitive:<kind>'). Raises on failure."""
    import trimesh
    if path.startswith("primitive:"):
        geoms = [_primitive(path.split(":", 1)[1])]
        name = path
    else:
        scene = trimesh.load(path, force="scene")
        geoms = scene.dump()          # copies with the node transforms applied
        name = os.path.basename(path)
    parts = [p for p in (_to_part(g) for g in geoms) if p is not None]
    if not parts:
        raise ValueError("no mesh or point geometry in the file")
    render = _build_render(parts)
    model = RawModel(name, parts, render)
    n_tri = len(render["tris"]) if render else 0
    print(f"  Model loaded: {name} — {len(parts)} part(s), {n_tri:,} triangles"
          + (f", {len(render['edges']):,} edges" if render else " (point cloud)"))
    return model


# ── per-configuration buffers (main thread) ──────────────────────────────────

def _orient(v: np.ndarray, up: int) -> np.ndarray:
    """Rotate so the chosen file axis points up (+Y). Proper rotations only."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    if up == 1:                     # Z-up (STL, scans, CAD)
        return np.column_stack([x, z, -y])
    if up == 2:                     # upside down
        return np.column_stack([x, -y, -z])
    if up == 3:                     # −Z up
        return np.column_stack([x, -z, y])
    return np.column_stack([x, y, z])


def _place(raw: RawModel, v, up: int, radius: float) -> np.ndarray:
    return _orient((v - raw.center) * (radius / raw.radius), up).astype(np.float32)


def _bands(pos: np.ndarray) -> np.ndarray:
    """Mel band per point from its height: bottom → bass, top → treble."""
    y = pos[:, 1]
    t = (y - y.min()) / max(float(np.ptp(y)), 1e-6)
    return np.clip((t * N_BANDS).astype(int), 0, N_BANDS - 1).astype(np.float32)


def _sample_texture(tex, uv):
    h, w = tex.shape[:2]
    px = ((uv[:, 0] % 1.0) * (w - 1)).astype(int)
    py = ((1.0 - uv[:, 1] % 1.0) * (h - 1)).astype(int)
    return tex[py, px]


def build_points(raw: RawModel, up: int, radius: float, n: int) -> np.ndarray:
    """(N,11) static buffer for the GPU point shader (u_mode 3):
    pos3, color3, aux4 = [normal3, band], fg = 1."""
    import trimesh
    meshes = [p for p in raw.parts if p.tm is not None]
    pos, nrm, col = [], [], []
    if meshes:
        total = sum(p.area for p in meshes) or 1.0
        for p in meshes:
            k = max(1, int(round(n * p.area / total)))
            pts, fidx, bary = trimesh.sample.sample_surface(
                p.tm, k, return_barycentric=True)
            tri = p.tm.faces[fidx]
            w = bary[:, :, None]
            vn = np.asarray(p.tm.vertex_normals)
            nrm.append((vn[tri] * w).sum(axis=1))
            if p.tex is not None:
                col.append(_sample_texture(p.tex, (p.uv[tri] * w).sum(axis=1)))
            else:
                col.append((p.vcolor[tri] * w).sum(axis=1))
            pos.append(pts)
        pos = np.concatenate(pos)
        nrm = np.concatenate(nrm)
    else:
        # point cloud: subsample the vertices, radial "normals"
        pts = np.concatenate([p.points for p in raw.parts])
        c = np.concatenate([p.vcolor for p in raw.parts])
        if len(pts) > n:
            sel = np.random.choice(len(pts), n, replace=False)
            pts, c = pts[sel], c[sel]
        pos, col = pts, [c]
        nrm = pts - raw.center

    pos = _place(raw, pos, up, radius)
    nrm = _orient(np.asarray(nrm, dtype=np.float32), up)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6

    out = np.zeros((len(pos), 11), dtype=np.float32)
    out[:, 0:3] = pos
    out[:, 3:6] = np.clip(np.concatenate(col), 0.0, 1.0)
    out[:, 6:9] = nrm
    out[:, 9] = _bands(pos)
    out[:, 10] = 1.0
    return out


def point_gain(raw: RawModel, radius: float, n: int) -> float:
    """Brightness compensation for the additive point cloud: the same budget
    on a small surface piles up into white. ~1 at 1 point per px² of the
    visible half of the surface (image mode's density)."""
    area = sum(p.area for p in raw.parts) * (radius / raw.radius) ** 2
    if area <= 0.0:
        return 1.0
    density = n / (0.5 * area)
    return float(np.clip(1.0 / density, 0.15, 1.5))


def build_mesh(raw: RawModel, up: int, radius: float):
    """Buffers for the surface / wireframe programs, or None (point cloud):
    dict(verts (V,13) f4 [pos3, normal3, color3, band, uv2, ao], tris, edges,
    sharp)."""
    r = raw.render
    if r is None:
        return None
    pos = _place(raw, r["v"], up, radius)
    verts = np.empty((len(pos), 13), dtype=np.float32)
    verts[:, 0:3] = pos
    verts[:, 3:6] = _orient(r["n"], up)
    verts[:, 6:9] = r["c"]
    verts[:, 9] = _bands(pos)
    verts[:, 10:12] = r["uv"] if r["uv"] is not None else NO_UV
    verts[:, 12] = r["ao"]
    # additive lines: dense meshes (short edges) would burn to white
    e = r["edges"]
    mean_len = float(np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1).mean())
    return {"verts": verts, "tex": r["tex"], "tris": r["tris"], "edges": e,
            "sharp": r["sharp"], "wire_gain": float(np.clip(mean_len / 25.0, 0.05, 1.0))}


def prepare(path: str, up: int, radius: float, n: int) -> tuple:
    """Everything a model load needs, for the worker thread:
    (raw, points buffer, mesh buffers or None)."""
    raw = load_model(path)
    return raw, build_points(raw, up, radius, n), build_mesh(raw, up, radius)

