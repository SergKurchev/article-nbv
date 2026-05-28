"""Uncertainty policy — follows ODIN's NBV head suggestion directly."""
from __future__ import annotations

import numpy as np
import config


class UncertaintyPolicy:
    """Greedy uncertainty-driven policy: clips and returns ODIN's NBV hint as action."""

    def reset(self) -> None:
        pass

    def act(self, nbv_hint: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
        """
        nbv_hint: (3,) xyz from ODIN NBV head
        Returns clipped (3,) xyz target position within ACTION bounds.
        """
        pos = np.array(nbv_hint, dtype=np.float32)[:3]
        lo = np.array(config.ACTION_MIN[:3], dtype=np.float32)
        hi = np.array(config.ACTION_MAX[:3], dtype=np.float32)
        return np.clip(pos, lo, hi)
