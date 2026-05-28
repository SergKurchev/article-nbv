"""Runtime 3D geometry memory for Wang-style visibility/coverage NBV.

Accumulates object-level coverage from ODIN segmentation masks, depth maps,
and camera poses. Purely PyBullet-compatible — no Isaac Sim dependencies.

Score formula (from Wang et al., adapted for CAD-free setting):
    S(v) = sum_k [ (1 - max_{r in R_k} dir(v,k) . r) * w_k ] + lambda_nov * nov(v)

where:
    R_k         = set of already-observed viewing directions for object k
    dir(v, k)   = unit vector from centroid_k toward candidate pose v
    w_k         = 0.70*(1 - coverage_k) + 0.25*small_mask_k + 0.05*unc_k
    nov(v)      = distance to nearest previously-visited camera pose (normalized)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (x,y,z,w) quaternion to 3x3 rotation matrix."""
    x, y, z, w = q / (np.linalg.norm(q) + 1e-8)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),       2*(y*z - x*w)],
        [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


@dataclass
class ObjectCoverage:
    class_id: int
    centroid: np.ndarray           # (3,) world coords
    bbox_min: np.ndarray           # (3,) world coords
    bbox_max: np.ndarray           # (3,) world coords
    view_dirs: list[np.ndarray] = field(default_factory=list)
    visible_pixels: int = 0
    mean_depth: float = 0.0
    mean_uncertainty: float = 0.0


class GeometryMemory:
    """Accumulates object-level 3D visibility from observed RGB-D frames.

    Usage:
        mem = GeometryMemory(fx=128, fy=128, cx=128, cy=128)
        mem.reset()
        for each step:
            mem.update(seg_mask, depth, cam_pos_xyz, cam_quat_xyzw, uncertainty_map)
        scores = mem.score_candidates(candidate_positions_Nx3)
    """

    def __init__(
        self,
        *,
        stride: int = 8,
        max_view_dirs: int = 24,
        min_pixels: int = 20,
        fx: float = 128.0,
        fy: float = 128.0,
        cx: float = 128.0,
        cy: float = 128.0,
        img_h: int = 256,
        img_w: int = 256,
    ):
        self.stride = int(stride)
        self.max_view_dirs = int(max_view_dirs)
        self.min_pixels = int(min_pixels)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.objects: dict[int, ObjectCoverage] = {}
        self.pose_history: list[np.ndarray] = []  # list of (3,) cam_pos

    def reset(self) -> None:
        self.objects.clear()
        self.pose_history.clear()

    def update(
        self,
        seg_mask: np.ndarray,    # (H, W) int, class id per pixel, -1=background
        depth: np.ndarray,       # (H, W) float32, metres
        cam_pos: np.ndarray,     # (3,) world xyz
        cam_quat_xyzw: np.ndarray,  # (4,) xyzw quaternion (PyBullet convention)
        uncertainty: np.ndarray | None = None,  # (H, W) float32 [0,1], optional
    ) -> None:
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        self.pose_history.append(cam_pos.copy())

        R = _quat_xyzw_to_matrix(cam_quat_xyzw)

        # Subsample spatial grid
        ys = np.arange(0, self.img_h, self.stride)
        xs = np.arange(0, self.img_w, self.stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yf = yy.ravel()
        xf = xx.ravel()

        d = depth[yf, xf].astype(np.float32)
        valid = (d > 1e-4) & np.isfinite(d)
        if not valid.any():
            return

        yf, xf, d = yf[valid], xf[valid], d[valid]

        # Back-project to camera coords (OpenGL: forward = -Z)
        x_cam = (xf - self.cx) * d / self.fx
        y_cam = -(yf - self.cy) * d / self.fy
        z_cam = -d
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

        # Transform to world
        pts_world = pts_cam @ R.T + cam_pos.reshape(1, 3)
        classes = seg_mask[yf, xf]

        unc_flat = uncertainty[yf, xf] if uncertainty is not None else None

        for cls_id in np.unique(classes):
            if int(cls_id) < 0:
                continue
            mask = classes == cls_id
            pix = int(mask.sum())
            if pix < self.min_pixels:
                continue

            pts = pts_world[mask]
            centroid = pts.mean(axis=0)
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
            view_dir = cam_pos - centroid
            norm = np.linalg.norm(view_dir)
            view_dir = view_dir / (norm + 1e-8)
            mean_depth = float(d[mask].mean())
            mean_unc = float(unc_flat[mask].mean()) if unc_flat is not None else 0.0

            old = self.objects.get(int(cls_id))
            if old is None:
                self.objects[int(cls_id)] = ObjectCoverage(
                    class_id=int(cls_id),
                    centroid=centroid,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    view_dirs=[view_dir],
                    visible_pixels=pix,
                    mean_depth=mean_depth,
                    mean_uncertainty=mean_unc,
                )
            else:
                total = old.visible_pixels + pix
                old.centroid = (old.centroid * old.visible_pixels + centroid * pix) / max(1, total)
                old.bbox_min = np.minimum(old.bbox_min, bbox_min)
                old.bbox_max = np.maximum(old.bbox_max, bbox_max)
                old.visible_pixels = total
                old.mean_depth = 0.8 * old.mean_depth + 0.2 * mean_depth
                old.mean_uncertainty = 0.8 * old.mean_uncertainty + 0.2 * mean_unc
                if not old.view_dirs:
                    old.view_dirs.append(view_dir)
                else:
                    seen = np.stack(old.view_dirs)  # (M, 3)
                    if float((seen @ view_dir).max()) < 0.96:
                        old.view_dirs.append(view_dir)
                        if len(old.view_dirs) > self.max_view_dirs:
                            old.view_dirs = old.view_dirs[-self.max_view_dirs:]

    def score_candidates(self, candidate_positions: np.ndarray) -> np.ndarray:
        """Wang-style expected visibility gain for candidate camera positions.

        Args:
            candidate_positions: (N, 3) array of xyz camera positions

        Returns:
            scores: (N,) normalized [0, 1] — higher is better
        """
        N = len(candidate_positions)
        if not self.objects:
            return self._history_novelty(candidate_positions)

        surface_scores = np.zeros(N, dtype=np.float32)
        total_weight = 0.0
        for obj in self.objects.values():
            cand_dir = candidate_positions - obj.centroid.reshape(1, 3)
            norms = np.linalg.norm(cand_dir, axis=1, keepdims=True)
            cand_dir = cand_dir / (norms + 1e-8)  # (N, 3)

            seen = np.stack(obj.view_dirs)  # (M, 3)
            cosines = cand_dir @ seen.T     # (N, M)
            max_seen = cosines.max(axis=1)  # (N,)
            new_surface = np.clip(1.0 - max_seen, 0.0, 2.0)  # (N,)

            coverage = min(1.0, len(obj.view_dirs) / max(1, self.max_view_dirs))
            under_seen = 1.0 - coverage
            small_mask_bonus = 1.0 / (1.0 + obj.visible_pixels / 1200.0)
            weight = 0.70 * under_seen + 0.25 * small_mask_bonus + 0.05 * obj.mean_uncertainty

            surface_scores += float(weight) * new_surface
            total_weight += float(weight)

        if total_weight > 1e-8:
            surface_scores /= total_weight

        novelty = self._history_novelty(candidate_positions)
        scores = 0.75 * surface_scores + 0.25 * novelty

        # Normalize
        smin, smax = scores.min(), scores.max()
        if smax - smin > 1e-8:
            scores = (scores - smin) / (smax - smin)

        return scores

    def _history_novelty(self, candidate_positions: np.ndarray) -> np.ndarray:
        """Novelty: distance to nearest previously-visited camera pose."""
        N = len(candidate_positions)
        if not self.pose_history:
            return np.ones(N, dtype=np.float32)

        hist = np.stack(self.pose_history)  # (K, 3)
        diff = candidate_positions[:, None, :] - hist[None, :, :]  # (N, K, 3)
        dists = np.linalg.norm(diff, axis=-1).min(axis=1)          # (N,)
        dmin, dmax = dists.min(), dists.max()
        if dmax - dmin < 1e-8:
            return np.ones(N, dtype=np.float32)
        return ((dists - dmin) / (dmax - dmin)).astype(np.float32)

    @property
    def visible_classes(self) -> int:
        return len(self.objects)

    @property
    def mean_coverage(self) -> float:
        if not self.objects:
            return 0.0
        coverages = [
            min(1.0, len(obj.view_dirs) / max(1, self.max_view_dirs))
            for obj in self.objects.values()
        ]
        return float(sum(coverages) / len(coverages))
