"""
test_odin_nbv_agent.py — Мок-тест для проверки REINFORCE обучения
                         с NBV Head в качестве RL-политики.

Проверяет:
  1. Gradient flow: градиенты текут через NBV Head при backward().
  2. Freeze backbone: при --freeze_backbone веса backbone НЕ меняются,
     а веса NBV Head МЕНЯЮТСЯ.
  3. End-to-end: без --freeze_backbone все веса меняются.
  4. Policy output: NBV Head выдает pos (3D) + quat (4D) → действие (6D).

Запуск:
  python -m pytest tests/test_odin_nbv_agent.py -v
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Мок-модели (вместо реального ODIN + Detectron2)
# ---------------------------------------------------------------------------

class MockCoverageHead(nn.Module):
    """Мок Coverage Head: features → p_hidden (скаляр 0..1)."""
    def __init__(self, in_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)


class MockNBVHead(nn.Module):
    """Мок NBV Head: features + current_pose → next_pos (3) + next_quat (4)."""
    def __init__(self, in_dim=256):
        super().__init__()
        self.pos_head = nn.Sequential(
            nn.Linear(in_dim + 7, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.quat_head = nn.Sequential(
            nn.Linear(in_dim + 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, features, current_pos, current_quat):
        x = torch.cat([features, current_pos, current_quat], dim=-1)
        pos = self.pos_head(x)
        quat = self.quat_head(x)
        # Нормализуем кватернион
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        return pos, quat


class MockBackbone(nn.Module):
    """Мок ODIN backbone: image → features (256-dim)."""
    def __init__(self, in_channels=4, feat_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, feat_dim)

    def forward(self, x):
        # x: (B, 4, H, W)
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class MockNBVActiveODIN(nn.Module):
    """
    Мок полной модели NBVActiveODIN.
    Имитирует структуру: backbone → (coverage_head, nbv_head).
    """
    def __init__(self, feat_dim=256):
        super().__init__()
        self.backbone = MockBackbone(feat_dim=feat_dim)
        self.coverage_head = MockCoverageHead(in_dim=feat_dim)
        self.nbv_head = MockNBVHead(in_dim=feat_dim)

    def forward(self, batched_inputs):
        # Берём первое изображение из батча
        bi = batched_inputs[0]
        img = bi["image"].unsqueeze(0)  # (1, C, H, W)
        if img.shape[1] == 3:
            # Добавляем канал depth (заглушка)
            depth_ch = torch.zeros(1, 1, img.shape[2], img.shape[3], device=img.device)
            img = torch.cat([img, depth_ch], dim=1)

        features = self.backbone(img)  # (1, feat_dim)

        p_hidden = self.coverage_head(features)  # (1, 1)

        pos = bi["current_camera_position"].unsqueeze(0) if "current_camera_position" in bi else torch.zeros(1, 3)
        quat = bi["current_camera_quaternion"].unsqueeze(0) if "current_camera_quaternion" in bi else torch.tensor([[0., 0., 0., 1.]])
        nbv_pos, nbv_quat = self.nbv_head(features, pos.to(features.device), quat.to(features.device))

        return {
            "p_hidden": p_hidden,
            "nbv_pos": nbv_pos,
            "nbv_quat": nbv_quat,
        }


# ---------------------------------------------------------------------------
# Мок-адаптер (без зависимостей от detectron2/pybullet)
# ---------------------------------------------------------------------------

class MockODINAdapter:
    """Мок ODINAdapter для тестов без PyBullet и Detectron2."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.last_p_hidden = 1.0
        self.last_nbv_pos = None
        self.last_nbv_quat = None
        self._frame_buffer = []

    def reset(self):
        self._frame_buffer.clear()
        self.last_p_hidden = 1.0

    def add_frame(self, rgb, depth, pose_c2w, fx, fy, cx, cy):
        img_t = torch.as_tensor(rgb.transpose(2, 0, 1).copy(), dtype=torch.float32)
        self._frame_buffer.append({"image": img_t})
        if len(self._frame_buffer) > 5:
            self._frame_buffer.pop(0)

    def infer_for_rl(self, current_pos, current_quat):
        if not self._frame_buffer:
            return {
                "nbv_pos_t": torch.zeros(3, device=self.device, requires_grad=True),
                "nbv_quat_t": torch.tensor([0., 0., 0., 1.], device=self.device, requires_grad=True),
                "p_hidden": self.last_p_hidden,
                "delta_p_hidden": 0.0,
            }

        frame = self._frame_buffer[-1]
        pos_t = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
        quat_t = torch.tensor(current_quat, dtype=torch.float32, device=self.device)

        batched_inputs = [{"image": frame["image"].to(self.device), "current_camera_position": pos_t, "current_camera_quaternion": quat_t}]
        outputs = self.model(batched_inputs)

        prev_p = self.last_p_hidden
        p_hidden = torch.sigmoid(outputs["p_hidden"]).detach().item()
        self.last_p_hidden = p_hidden

        return {
            "nbv_pos_t": outputs["nbv_pos"].squeeze(),
            "nbv_quat_t": outputs["nbv_quat"].squeeze(),
            "p_hidden": p_hidden,
            "delta_p_hidden": prev_p - p_hidden,
        }


# ---------------------------------------------------------------------------
# Вспомогательная функция: дифференцируемый quat → euler
# ---------------------------------------------------------------------------

def quat_to_euler_torch(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


# ---------------------------------------------------------------------------
# ТЕСТЫ
# ---------------------------------------------------------------------------

class TestNBVHeadGradientFlow:
    """Проверка, что градиенты текут через NBV Head при REINFORCE."""

    @pytest.fixture
    def setup(self):
        device = torch.device("cpu")
        model = MockNBVActiveODIN(feat_dim=256).to(device)
        adapter = MockODINAdapter(model, device)
        return model, adapter, device

    def test_gradient_flows_through_nbv_head(self, setup):
        """Градиенты от REINFORCE loss текут через NBV Head."""
        model, adapter, device = setup
        model.train()

        # Записываем начальные веса NBV Head
        nbv_params_before = {
            name: p.clone() for name, p in model.nbv_head.named_parameters()
        }

        optimizer = torch.optim.Adam(model.nbv_head.parameters(), lr=1e-3)

        # Симулируем один эпизод
        adapter.reset()
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = np.random.rand(64, 64).astype(np.float32) * 5.0
        pose = np.eye(4, dtype=np.float32)
        adapter.add_frame(rgb, depth, pose, 32, 32, 32, 32)

        result = adapter.infer_for_rl(
            current_pos=np.array([0.5, 0.0, 0.3], dtype=np.float32),
            current_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

        nbv_pos = result["nbv_pos_t"]
        nbv_quat = result["nbv_quat_t"]
        euler = quat_to_euler_torch(nbv_quat)
        mean_action = torch.cat([nbv_pos, euler])

        # Стохастическая политика
        dist = torch.distributions.Normal(mean_action, 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # Псевдо-награда
        reward = 1.0
        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Проверяем, что веса NBV Head изменились
        changed = False
        for name, p in model.nbv_head.named_parameters():
            if not torch.allclose(p, nbv_params_before[name], atol=1e-7):
                changed = True
                break
        assert changed, "NBV Head weights should change after REINFORCE update"

    def test_backbone_frozen_when_freeze_backbone(self, setup):
        """При --freeze_backbone backbone НЕ обновляется."""
        model, adapter, device = setup

        # Замораживаем всё кроме nbv_head
        for name, p in model.named_parameters():
            if "nbv_head" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        backbone_before = {
            name: p.clone() for name, p in model.backbone.named_parameters()
        }
        coverage_before = {
            name: p.clone() for name, p in model.coverage_head.named_parameters()
        }

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        # Один REINFORCE шаг
        adapter.reset()
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = np.random.rand(64, 64).astype(np.float32)
        adapter.add_frame(rgb, depth, np.eye(4, dtype=np.float32), 32, 32, 32, 32)

        result = adapter.infer_for_rl(
            current_pos=np.array([0.5, 0.0, 0.3]),
            current_quat=np.array([0., 0., 0., 1.]),
        )

        mean_action = torch.cat([
            result["nbv_pos_t"],
            quat_to_euler_torch(result["nbv_quat_t"])
        ])
        dist = torch.distributions.Normal(mean_action, 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        loss = -log_prob * 1.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Backbone — НЕ изменился
        for name, p in model.backbone.named_parameters():
            assert torch.allclose(p, backbone_before[name], atol=1e-9), \
                f"Backbone param {name} should NOT change with --freeze_backbone"

        # Coverage Head — НЕ изменился
        for name, p in model.coverage_head.named_parameters():
            assert torch.allclose(p, coverage_before[name], atol=1e-9), \
                f"Coverage param {name} should NOT change with --freeze_backbone"

    def test_end_to_end_all_weights_change(self, setup):
        """Без --freeze_backbone все веса обновляются."""
        model, adapter, device = setup
        model.train()

        for p in model.parameters():
            p.requires_grad_(True)

        all_before = {name: p.clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 3 шага для надёжного изменения
        for _ in range(3):
            adapter.reset()
            rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            depth = np.random.rand(64, 64).astype(np.float32)
            adapter.add_frame(rgb, depth, np.eye(4, dtype=np.float32), 32, 32, 32, 32)

            result = adapter.infer_for_rl(
                current_pos=np.array([0.5, 0.0, 0.3]),
                current_quat=np.array([0., 0., 0., 1.]),
            )
            mean_action = torch.cat([
                result["nbv_pos_t"],
                quat_to_euler_torch(result["nbv_quat_t"])
            ])
            dist = torch.distributions.Normal(mean_action, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            loss = -log_prob * 5.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Проверяем: NBV Head точно изменился
        nbv_changed = any(
            not torch.allclose(p, all_before[n], atol=1e-6)
            for n, p in model.named_parameters() if "nbv_head" in n
        )
        assert nbv_changed, "NBV Head should change in end-to-end mode"

        # Backbone тоже должен измениться (end-to-end)
        backbone_changed = any(
            not torch.allclose(p, all_before[n], atol=1e-6)
            for n, p in model.named_parameters() if "backbone" in n
        )
        assert backbone_changed, "Backbone should also change in end-to-end mode"

    def test_policy_output_shape(self, setup):
        """NBV Head выдает pos (3D) и quat (4D), конвертируемые в 6D action."""
        model, adapter, device = setup
        adapter.reset()

        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = np.random.rand(64, 64).astype(np.float32)
        adapter.add_frame(rgb, depth, np.eye(4, dtype=np.float32), 32, 32, 32, 32)

        result = adapter.infer_for_rl(
            current_pos=np.array([0.5, 0.0, 0.3]),
            current_quat=np.array([0., 0., 0., 1.]),
        )

        assert result["nbv_pos_t"].shape == (3,), f"Expected (3,), got {result['nbv_pos_t'].shape}"
        assert result["nbv_quat_t"].shape == (4,), f"Expected (4,), got {result['nbv_quat_t'].shape}"

        # Конвертация в 6D action
        euler = quat_to_euler_torch(result["nbv_quat_t"])
        action = torch.cat([result["nbv_pos_t"], euler])
        assert action.shape == (6,), f"Expected 6D action, got {action.shape}"

    def test_multi_step_episode(self, setup):
        """Симуляция эпизода из нескольких шагов с REINFORCE обновлением."""
        model, adapter, device = setup
        model.train()

        for name, p in model.named_parameters():
            if "nbv_head" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        adapter.reset()
        log_probs = []
        rewards = []

        # 5 шагов эпизода
        for step in range(5):
            rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            depth = np.random.rand(64, 64).astype(np.float32) * 3.0
            adapter.add_frame(rgb, depth, np.eye(4, dtype=np.float32), 32, 32, 32, 32)

            result = adapter.infer_for_rl(
                current_pos=np.array([0.5, 0.0, 0.3 + step * 0.05]),
                current_quat=np.array([0., 0., 0., 1.]),
            )

            mean_action = torch.cat([
                result["nbv_pos_t"],
                quat_to_euler_torch(result["nbv_quat_t"])
            ])
            dist = torch.distributions.Normal(mean_action, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            log_probs.append(log_prob)
            rewards.append(float(result["delta_p_hidden"]) * 20.0 + 0.1)  # Добавляем бонус

        # REINFORCE update
        gamma = 0.99
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.tensor(0.0)
        for lp, R in zip(log_probs, returns_t):
            loss = loss - lp * R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Проверяем, что потери конечны
        assert torch.isfinite(torch.tensor(loss.item())), "Loss should be finite"
        print(f"Multi-step episode: loss={loss.item():.4f}, "
              f"total_reward={sum(rewards):.4f}")


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
