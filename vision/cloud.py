#!/usr/bin/env python
"""Generic 3-D perception layer over the RealSense depth: point clouds, multi-plane
extraction, and geometric-prior fitters. Object localizers (ball today, future objects
tomorrow) plug a 2-D detector into one of the fitters here — nothing in this module is
ball-specific.

Design notes (see vision/PERCEPTION3D.md):
- depth bleeding at object silhouettes biases naive box-median depth toward the
  background; fitters here are robust to it (background points don't fit the prior),
- known dimensions are constraints, not estimates: fit_sphere with fixed radius has
  3 unknowns instead of 4 and rejects outliers far better,
- every fitter returns a quality record (inliers, rms) so callers can refuse bad frames.
"""
import numpy as np

# ── point clouds ──────────────────────────────────────────────────────────────────────

_GRID = {}


def deproject(depth_m, K, box=None):
    """Depth image (m) -> Nx3 points in the camera frame. `box`=(x1,y1,x2,y2) crops.
    Invalid (<=0) depths are dropped. Pixel grids are cached per resolution."""
    H, W = depth_m.shape
    key = (H, W)
    if key not in _GRID:
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        _GRID[key] = (u, v)
    u, v = _GRID[key]
    if box is not None:
        x1, y1, x2, y2 = (max(box[0], 0), max(box[1], 0), box[2], box[3])
        depth_m = depth_m[y1:y2, x1:x2]
        u, v = u[y1:y2, x1:x2], v[y1:y2, x1:x2]
    z = depth_m.ravel()
    ok = z > 0
    z = z[ok]
    x = (u.ravel()[ok] - K["ppx"]) * z / K["fx"]
    y = (v.ravel()[ok] - K["ppy"]) * z / K["fy"]
    return np.stack([x, y, z], axis=1)


def crop_z(points, z_range):
    return points[(points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])]


# ── planes ────────────────────────────────────────────────────────────────────────────

def fit_plane(points, iters=400, thresh=0.006, seed=0):
    """RANSAC plane n·p + d = 0 (n unit). Returns (n, d, inlier_mask)."""
    rng = np.random.default_rng(seed)
    N = len(points)
    best = None
    for _ in range(iters):
        p = points[rng.choice(N, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p[0]
        inl = np.abs(points @ n + d) < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    P = points[inl]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    return n, -float(n @ c), inl


def extract_planes(points, max_planes=3, min_frac=0.08, thresh=0.006, seed=0):
    """Sequential RANSAC: fit a plane, remove its inliers, repeat. Returns a list of
    dicts (n, d, n_inliers, centroid, extent) sorted by support — separates e.g. the
    white plate from the wooden desk instead of fitting one mongrel plane."""
    pts = points.copy()
    out = []
    for k in range(max_planes):
        if len(pts) < 500:
            break
        n, d, inl = fit_plane(pts, thresh=thresh, seed=seed + k)
        if inl.sum() < min_frac * len(points):
            break
        P = pts[inl]
        out.append(dict(n=n, d=d, n_inliers=int(inl.sum()),
                        centroid=P.mean(0),
                        extent=(P.min(0), P.max(0))))
        pts = pts[~inl]
    return sorted(out, key=lambda p: -p["n_inliers"])


# ── geometric-prior fitters ───────────────────────────────────────────────────────────

def fit_sphere_known_r(points, r, iters=150, thresh=0.004, min_inliers=60, seed=0):
    """RANSAC sphere with KNOWN radius (3 unknowns: the centre). Hypotheses come from
    pushing single surface points one radius away from the camera along their ray —
    exact at the cap top, good enough elsewhere to seed inlier counting. Refined by
    Gauss-Newton on the inliers. Returns dict(center, inliers, rms) or None.

    Background/bleed pixels don't lie on an r-sphere, so they are rejected instead of
    biasing the result (the failure mode of box-median depth)."""
    if len(points) < min_inliers:
        return None
    rng = np.random.default_rng(seed)
    rays = points / np.linalg.norm(points, axis=1, keepdims=True)
    # prefer near (top-of-object) points as hypothesis seeds: immune to background
    order = np.argsort(points[:, 2])
    seeds = order[:max(len(points) // 4, 30)]
    best = None
    for _ in range(iters):
        i = seeds[rng.integers(len(seeds))]
        c = points[i] + r * rays[i]
        err = np.abs(np.linalg.norm(points - c, axis=1) - r)
        inl = err < thresh
        if best is None or inl.sum() > best[1].sum():
            best = (c, inl)
    c, inl = best
    if inl.sum() < min_inliers:
        return None
    P = points[inl]
    for _ in range(15):                                   # Gauss-Newton refine on inliers
        diff = P - c
        dist = np.linalg.norm(diff, axis=1)
        res = dist - r
        J = diff / dist[:, None]                          # d(dist)/dc = -(P-c)/dist; sign in step
        step, *_ = np.linalg.lstsq(J, res, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-5:
            break
    err = np.abs(np.linalg.norm(points - c, axis=1) - r)
    inl = err < thresh
    rms = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.sum() else float("inf")
    return dict(center=c, inliers=int(inl.sum()), rms=rms)


def voxel_downsample(points, voxel=0.008):
    """One representative (centroid) point per occupied voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    sums = np.zeros((len(idx), 3))
    np.add.at(sums, inv, points)
    counts = np.bincount(inv, minlength=len(idx)).astype(float)
    return sums / counts[:, None]


def euclidean_clusters(points, radius=0.02, min_pts=15):
    """Greedy BFS clustering (classic tabletop segmentation). Returns a list of index
    arrays, largest first. Run on VOXEL-DOWNSAMPLED points — it's O(N·neighbors)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    unvisited = np.ones(len(points), bool)
    clusters = []
    for s in range(len(points)):
        if not unvisited[s]:
            continue
        queue = [s]
        unvisited[s] = False
        members = [s]
        while queue:
            for j in tree.query_ball_point(points[queue.pop()], radius):
                if unvisited[j]:
                    unvisited[j] = False
                    queue.append(j)
                    members.append(j)
        if len(members) >= min_pts:
            clusters.append(np.array(members))
    return sorted(clusters, key=len, reverse=True)


def nearest_depth_center(points, r, pct=5):
    """Cheap fallback: the object's top is the NEAREST depth percentile (background
    bleed is always farther), centre = top + r along its viewing ray."""
    if len(points) == 0:
        return None
    z = points[:, 2]
    i = np.argsort(z)[max(int(len(z) * pct / 100) - 1, 0)]
    p = points[i]
    return p + r * p / np.linalg.norm(p)
