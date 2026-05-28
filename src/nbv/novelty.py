"""Voxel-based novelty tracker for exploration reward."""
from __future__ import annotations

import numpy as np


class VoxelNoveltyTracker:
    """Tracks seen world-space voxels and returns fraction of new voxels per step."""

    def __init__(self, voxel_size: float = 0.02):
        self.voxel_size = float(voxel_size)
        self._seen: set[tuple[int, int, int]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def update_and_score(self, pts_world: np.ndarray) -> float:
        """
        pts_world: (N, 3) world-space points (e.g. from ODIN original_xyz)
        Returns fraction of new voxels ∈ [0, 1].
        """
        if pts_world is None or len(pts_world) == 0:
            return 0.0
        keys = set(
            map(tuple, (pts_world / self.voxel_size).astype(np.int32).tolist())
        )
        new_count = len(keys - self._seen)
        self._seen.update(keys)
        return float(new_count) / max(1, len(keys))
