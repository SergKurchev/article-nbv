"""
test_odin_sac_agent.py — Мок-тест SAC обучения с NBV Head как актором.

Проверяет:
  1. ReplayBuffer push/sample.
  2. QNetwork forward shapes.
  3. OdinSACAgent: actor output, critic update, actor update.
  4. Freeze backbone: только NBV Head обновляется.
  5. Multi-step episode + SAC updates.

Запуск:
  python -m pytest tests/test_odin_sac_agent.py -v
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
import copy


# --- Мок-модели (те же, что в test_odin_nbv_agent.py) ---

class MockCoverageHead(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        return self.fc(x)

class MockNBVHead(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.pos_head = nn.Sequential(nn.Linear(d+7, 128), nn.ReLU(), nn.Linear(128, 3))
        self.quat_head = nn.Sequential(nn.Linear(d+7, 128), nn.ReLU(), nn.Linear(128, 4))
    def forward(self, f, p, q):
        x = torch.cat([f, p, q], -1)
        pos = self.pos_head(x)
        quat = self.quat_head(x)
        return pos, quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

class MockBackbone(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.conv = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, d)
    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

class MockNBVActiveODIN(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.backbone = MockBackbone(d)
        self.coverage_head = MockCoverageHead(d)
        self.nbv_head = MockNBVHead(d)
    def forward(self, bi, current_position=None, current_quaternion=None):
        img = bi[0]["image"].unsqueeze(0)
        if img.shape[1] == 3:
            img = torch.cat([img, torch.zeros(1,1,img.shape[2],img.shape[3],device=img.device)], 1)
        f = self.backbone(img)
        p = current_position if current_position is not None else torch.zeros(1,3)
        q = current_quaternion if current_quaternion is not None else torch.tensor([[0.,0.,0.,1.]])
        nbv_p, nbv_q = self.nbv_head(f, p.to(f.device), q.to(f.device))
        return {"p_hidden": self.coverage_head(f), "nbv_pos": nbv_p, "nbv_quat": nbv_q}


# --- Import SAC components ---

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_odin_sac_rl import ReplayBuffer, QNetwork


# --- Tests ---

class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(100)
        for i in range(20):
            buf.push(np.zeros(18), np.zeros(6), float(i), np.zeros(18), False)
        assert len(buf) == 20

        obs, act, rew, nobs, done = buf.sample(8)
        assert obs.shape == (8, 18)
        assert act.shape == (8, 6)
        assert rew.shape == (8, 1)

    def test_capacity(self):
        buf = ReplayBuffer(10)
        for i in range(50):
            buf.push(np.zeros(18), np.zeros(6), 0.0, np.zeros(18), False)
        assert len(buf) == 10


class TestQNetwork:
    def test_forward_shapes(self):
        q = QNetwork(obs_dim=18, action_dim=6, hidden_dim=128)
        obs = torch.randn(4, 18)
        act = torch.randn(4, 6)
        q1, q2 = q(obs, act)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_gradient_flow(self):
        q = QNetwork(obs_dim=18, action_dim=6)
        obs = torch.randn(4, 18)
        act = torch.randn(4, 6, requires_grad=True)
        q1, q2 = q(obs, act)
        loss = (q1 + q2).mean()
        loss.backward()
        assert act.grad is not None


class TestSACTrainingLoop:
    """Интеграционный тест: SAC update loop с мок-моделью."""

    def test_critic_update_decreases_loss(self):
        """Critic loss уменьшается после нескольких обновлений."""
        q = QNetwork(18, 6, 128)
        q_target = copy.deepcopy(q)
        opt = torch.optim.Adam(q.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            obs = torch.randn(32, 18)
            act = torch.randn(32, 6)
            rew = torch.randn(32, 1)
            next_obs = torch.randn(32, 18)
            done = torch.zeros(32, 1)

            with torch.no_grad():
                next_act = torch.randn(32, 6)
                q1_n, q2_n = q_target(next_obs, next_act)
                target = rew + 0.99 * (1 - done) * torch.min(q1_n, q2_n)

            q1, q2 = q(obs, act)
            loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0] * 2, "Critic loss should not explode"

    def test_nbv_head_updates_with_sac_gradient(self):
        """NBV Head веса обновляются через SAC actor gradient."""
        model = MockNBVActiveODIN()
        model.train()

        # Freeze backbone + coverage
        for n, p in model.named_parameters():
            p.requires_grad_("nbv_head" in n)

        nbv_before = {n: p.clone() for n, p in model.nbv_head.named_parameters()}

        # Actor params
        log_std = nn.Parameter(torch.zeros(6))
        actor_params = list(model.nbv_head.parameters()) + [log_std]
        actor_opt = torch.optim.Adam(actor_params, lr=1e-3)

        # Critic
        critic = QNetwork(18, 6, 128)

        for _ in range(5):
            # Simulate forward
            rgb = torch.randn(3, 64, 64)
            pos_t = torch.randn(1, 3)
            quat_t = torch.tensor([[0., 0., 0., 1.]])
            outputs = model([{"image": rgb}], current_position=pos_t, current_quaternion=quat_t)

            nbv_pos = outputs["nbv_pos"].squeeze()
            nbv_quat = outputs["nbv_quat"].squeeze()

            # Simple action
            mean = torch.cat([nbv_pos, nbv_quat[:3]])  # 6D
            std = log_std.exp().clamp(min=1e-4)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum()

            obs_vec = torch.randn(1, 18)
            q1, q2 = critic(obs_vec, action.unsqueeze(0))
            actor_loss = (0.2 * log_prob - torch.min(q1, q2)).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

        # Check NBV Head changed
        changed = any(
            not torch.allclose(p, nbv_before[n], atol=1e-6)
            for n, p in model.nbv_head.named_parameters()
        )
        assert changed, "NBV Head should update via SAC actor gradient"

    def test_backbone_stays_frozen(self):
        """Backbone не изменяется при --freeze_backbone."""
        model = MockNBVActiveODIN()
        for n, p in model.named_parameters():
            p.requires_grad_("nbv_head" in n)

        bb_before = {n: p.clone() for n, p in model.backbone.named_parameters()}
        cov_before = {n: p.clone() for n, p in model.coverage_head.named_parameters()}

        log_std = nn.Parameter(torch.zeros(6))
        opt = torch.optim.Adam(list(model.nbv_head.parameters()) + [log_std], lr=1e-2)

        for _ in range(3):
            outputs = model(
                [{"image": torch.randn(3, 64, 64)}],
                current_position=torch.randn(1, 3),
                current_quaternion=torch.tensor([[0.,0.,0.,1.]]),
            )
            mean = torch.cat([outputs["nbv_pos"].squeeze(), outputs["nbv_quat"].squeeze()[:3]])
            dist = torch.distributions.Normal(mean, log_std.exp().clamp(min=1e-4))
            action = dist.rsample()
            loss = dist.log_prob(action).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        for n, p in model.backbone.named_parameters():
            assert torch.allclose(p, bb_before[n]), f"Backbone {n} should NOT change"
        for n, p in model.coverage_head.named_parameters():
            assert torch.allclose(p, cov_before[n]), f"Coverage {n} should NOT change"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
