"""Hemisphere candidate sampler (deterministic Fibonacci hemisphere, N poses).

Adapted from strawberry/odin_connection/nbv/sampler.py for PyBullet UR3 workspace.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import config


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = torch.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / (q.norm() + 1e-8)


def _look_at_quat(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Camera-to-world quaternion (wxyz) in OpenGL convention (-Z forward)."""
    fwd = target - eye
    fwd = fwd / (fwd.norm() + 1e-8)
    up_world = torch.tensor([0.0, 0.0, 1.0])
    if abs(float(fwd @ up_world)) > 0.999:
        up_world = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(fwd, up_world)
    right = right / (right.norm() + 1e-8)
    up = torch.linalg.cross(right, fwd)
    up = up / (up.norm() + 1e-8)
    R = torch.stack([right, up, -fwd], dim=1)
    return _mat_to_quat(R)


class HemisphereSampler:
    """Fixed set of N camera poses on the upper hemisphere over the UR3 workspace."""

    def __init__(
        self,
        n: int = config.N_CANDIDATES,
        radius: float = config.HEMI_RADIUS,
        center: tuple[float, float, float] = config.HEMI_CENTER,
        min_elevation: float = 0.20,
    ):
        self.n = n
        self.radius = radius
        self.center = torch.tensor(center, dtype=torch.float32)
        self.min_elevation = float(min_elevation)
        self._poses = self._build()

    def _build(self) -> torch.Tensor:
        ga = math.pi * (3.0 - math.sqrt(5.0))
        z_min = max(0.0, min(0.95, self.min_elevation))
        z_top = 1.0
        out: list[torch.Tensor] = []
        for i in range(self.n):
            t = i / max(1, self.n - 1)
            z = z_top - t * (z_top - z_min)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            theta = ga * i
            x = math.cos(theta) * r
            y = math.sin(theta) * r
            pos = self.center + self.radius * torch.tensor([x, y, z], dtype=torch.float32)
            quat = _look_at_quat(pos, self.center)
            out.append(torch.cat([pos, quat]))
        return torch.stack(out, dim=0)  # (N, 7): xyz + wxyz

    def __call__(self) -> torch.Tensor:
        return self._poses.clone()

    def positions_np(self) -> np.ndarray:
        return self._poses[:, :3].numpy().copy()
