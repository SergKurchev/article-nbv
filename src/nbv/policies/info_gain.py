"""Geometric NBV policy — Wang-style active 3D visibility gain.

Adapted from strawberry/odin_connection/nbv/policies/info_gain.py for our
PyBullet UR3 environment. Uses GeometryMemory to score hemisphere candidates
and returns the best xyz camera position deterministically.
"""
from __future__ import annotations

import numpy as np

from ..geometry_memory import GeometryMemory
from ..sampler import HemisphereSampler
import config


class GeometricNBVPolicy:
    """Deterministic geometric baseline: picks candidate with highest coverage gain."""

    def __init__(
        self,
        geometry_weight: float = 0.90,
        height_weight: float = 0.07,
        uncertainty_tiebreak: float = 0.03,
    ):
        self.geometry_weight = geometry_weight
        self.height_weight = height_weight
        self.uncertainty_tiebreak = uncertainty_tiebreak
        self.geometry_memory = GeometryMemory(
            fx=config.IMAGE_SIZE / 2.0,
            fy=config.IMAGE_SIZE / 2.0,
            cx=config.IMAGE_SIZE / 2.0,
            cy=config.IMAGE_SIZE / 2.0,
        )
        self.sampler = HemisphereSampler()

    def reset(self) -> None:
        self.geometry_memory.reset()

    def act(
        self,
        seg_mask: np.ndarray,
        depth: np.ndarray,
        cam_pos: np.ndarray,
        cam_quat_xyzw: np.ndarray,
        uncertainty: np.ndarray | None = None,
        mean_uncertainty: float = 0.0,
    ) -> np.ndarray:
        """
        Update geometry memory with current observation, return best xyz (3,).

        seg_mask:       (H, W) int  — class id per pixel, -1=background
        depth:          (H, W) float32  — metres
        cam_pos:        (3,) world xyz
        cam_quat_xyzw:  (4,) PyBullet quaternion
        uncertainty:    (H, W) float32 optional per-pixel epistemic uncertainty
        mean_uncertainty: scalar mean uncertainty (for tiebreak scoring)
        """
        self.geometry_memory.update(seg_mask, depth, cam_pos, cam_quat_xyzw, uncertainty)
        candidates = self.sampler.positions_np()  # (N, 3)
        geom_scores = self.geometry_memory.score_candidates(candidates)

        # Height bonus: prefer higher viewpoints
        cam_z = candidates[:, 2]
        z_min, z_max = cam_z.min(), cam_z.max()
        height = (cam_z - z_min) / (z_max - z_min + 1e-8)

        scores = (
            self.geometry_weight * geom_scores
            + self.height_weight * height
            + self.uncertainty_tiebreak * mean_uncertainty * geom_scores
        )
        best_idx = int(np.argmax(scores))
        return candidates[best_idx].copy()
