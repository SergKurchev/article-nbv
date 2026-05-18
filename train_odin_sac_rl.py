"""
train_odin_sac_rl.py — SAC обучение, где NBV Head модели ODIN
                       ЯВЛЯЕТСЯ RL-агентом (политикой).

Отличия от REINFORCE (train_odin_nbv_rl.py):
    - Off-policy: Replay Buffer для переиспользования опыта.
    - Twin Q-critics: Два Q-network для снижения переоценки.
    - Entropy tuning: Автоматический коэффициент энтропии α.
    - Reparameterization trick: Градиенты текут через семплирование.

Архитектура:
    PyBullet → RGB-D → ODIN backbone → scene features
                                           ↓
                    Actor (NBV Head) → next_pose (ACTION)
                    Critic (Twin Q)  → Q(s, a)
                    Coverage Head    → p_hidden (REWARD signal)

Флаги заморозки:
    --freeze_backbone : Замораживает backbone + Coverage Head.
                        Обучается NBV Head (actor) + Q-networks.
    (без флага)       : End-to-end. backbone тоже адаптируется.

Запуск:
    python train_odin_sac_rl.py \\
        --odin_weights ./model_final.pth \\
        --odin_cfg my_odin/configs/scannet_context/3d.yaml \\
        --freeze_backbone \\
        --total_steps 100000

Тест:
    python -m pytest tests/test_odin_sac_agent.py -v
"""

import os
import sys
import csv
import copy
import signal
import argparse
import random
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Differentiable quaternion → euler
# ---------------------------------------------------------------------------

def quat_to_euler_torch(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


# ---------------------------------------------------------------------------
# SAC компоненты
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Простой replay buffer для SAC."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_vec, action, reward, next_obs_vec, done):
        self.buffer.append((obs_vec, action, reward, next_obs_vec, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(act)),
            torch.FloatTensor(np.array(rew)).unsqueeze(1),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Twin Q-network для SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class OdinSACAgent:
    """
    SAC-агент, где Actor = NBV Head модели ODIN.

    NBV Head предсказывает mean действия (pos + quat → euler).
    Learnable log_std добавляет стохастичность.
    Twin Q-critics + auto entropy tuning.
    """

    def __init__(
        self,
        odin_model,
        adapter,
        obs_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        auto_entropy: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.odin_model = odin_model
        self.adapter = adapter
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.action_dim = action_dim

        # Learnable log_std для стохастической политики NBV Head
        self.log_std = nn.Parameter(
            torch.zeros(action_dim, device=self.device)
        )

        # Собираем trainable параметры актора:
        # NBV Head params + log_std
        self.actor_params = [
            p for n, p in odin_model.named_parameters()
            if p.requires_grad and "nbv_head" in n
        ] + [self.log_std]

        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=lr_actor)

        # Twin Q-critic
        self.critic = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Entropy tuning
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(
                np.log(init_alpha), requires_grad=True, device=self.device
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)
        else:
            self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(
        self,
        obs: dict,
        adapter,
        deterministic: bool = False,
    ) -> tuple:
        """
        Прогоняет ODIN → NBV Head → action.

        Returns: (action_np, log_prob_scalar, obs_vector_np)
        """
        rgb = obs.get("_raw_rgb")
        depth = obs["image"][3] * 10.0  # обратная нормализация
        cam_pos = obs["vector"][:3]
        cam_orn = obs["vector"][3:7]

        # Добавляем кадр
        import pybullet as pb
        R = np.array(pb.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = cam_pos
        fx = fy = config.IMAGE_SIZE / 2.0
        cx = cy = config.IMAGE_SIZE / 2.0

        if rgb is not None:
            adapter.add_frame(rgb, depth.astype(np.float32), pose, fx, fy, cx, cy)

        # ODIN forward с градиентами → NBV Head output
        result = adapter.infer_for_rl(
            current_pos=cam_pos.astype(np.float32),
            current_quat=cam_orn.astype(np.float32),
        )

        nbv_pos = result["nbv_pos_t"]      # (3,) tensor
        nbv_quat = result["nbv_quat_t"]    # (4,) tensor
        euler = quat_to_euler_torch(nbv_quat)  # (3,)
        mean_action = torch.cat([nbv_pos, euler])  # (6,)

        # Stochastic policy
        std = self.log_std.exp().clamp(min=1e-4, max=2.0)
        dist = torch.distributions.Normal(mean_action, std)

        if deterministic:
            action_t = mean_action
            log_prob = torch.tensor(0.0)
        else:
            # Reparameterization trick
            action_t = dist.rsample()
            log_prob = dist.log_prob(action_t).sum()

        # Clip to action bounds
        action_np = action_t.detach().cpu().numpy().astype(np.float32)
        action_np = np.clip(action_np, config.ACTION_MIN, config.ACTION_MAX)

        # Observation vector для replay buffer
        obs_vec = obs["vector"].copy()

        return action_np, log_prob, obs_vec, result["delta_p_hidden"]

    def get_action_from_obs_vec(self, obs_vec_batch):
        """
        Для обновления critic/actor из replay buffer.
        Возвращает sampled actions + log_probs для batch.

        Так как NBV Head зависит от ODIN backbone (нужны фреймы),
        для off-policy используем surrogate actor: MLP с теми же весами.
        Здесь используем mean_action из obs_vec (nbv_hint уже в векторе).
        """
        # obs_vec содержит: pos(3) + orn(4) + delta_p(1) + joints(7) + nbv_hint(3)
        # Используем nbv_hint (позиция 15:18) как "подсказку" актора
        # и генерируем действие из стохастической политики
        nbv_hint = obs_vec_batch[:, 15:18]  # (B, 3)
        euler_hint = obs_vec_batch[:, 3:6] * 0.1  # грубая оценка из orn

        mean = torch.cat([nbv_hint, euler_hint], dim=-1).to(self.device)  # (B, 6)
        std = self.log_std.exp().clamp(min=1e-4, max=2.0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob

    def update_critic(self, batch):
        obs, action, reward, next_obs, done = [x.to(self.device) for x in batch]

        with torch.no_grad():
            next_action, next_log_prob = self.get_action_from_obs_vec(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + self.gamma * (1 - done) * q_next

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor_and_alpha(self, batch):
        obs = batch[0].to(self.device)

        action, log_prob = self.get_action_from_obs_vec(obs)
        q1, q2 = self.critic(obs, action)
        q_min = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
        self.actor_optimizer.step()

        # Entropy tuning
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss_t = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_t.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss_t.item()

        return actor_loss.item(), alpha_loss

    def soft_update_target(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAC: ODIN NBV Head as RL policy")
    p.add_argument("--odin_weights", type=str, required=True)
    p.add_argument("--odin_cfg", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=24)

    p.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone + Coverage Head. Train NBV Head + critics.")

    p.add_argument("--total_steps", type=int, default=100000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer_size", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=1000)
    p.add_argument("--update_freq", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--scene_stage", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--no_arm", action="store_true")
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--record_episodes", nargs="+", type=int, default=[],
                    help="Список эпизодов для записи видео (например: 10 50 100)")

    p.add_argument("--output_dir", type=str, default="./output_odin_sac")
    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--log_freq", type=int, default=100)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ODIN ---
    print(f"[INFO] Loading ODIN from {args.odin_weights} ...")
    from src.vision.odin_adapter import load_nbv_active_odin
    model, adapter = load_nbv_active_odin(
        args.odin_weights, args.odin_cfg, args.num_classes,
        torch.device(device),
    )

    # --- Freeze ---
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad_("nbv_head" in name)
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] --freeze_backbone: trainable {tp:,} params (NBV Head only)")
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        print(f"[INFO] End-to-end: all params trainable")

    model.train()

    # --- Environment ---
    config.SCENE_STAGE = args.scene_stage
    config.MAX_STEPS_PER_EPISODE = args.max_steps
    from src.simulation.environment_odin import NBVODINEnv

    env = NBVODINEnv(
        render_mode="rgb_array" if args.record_episodes else None,
        headless=True, no_arm=args.no_arm,
        odin_adapter=adapter, use_nbv_hint=False,
        log_dir=str(output_dir / "logs"),
        external_infer=True,
    )

    obs_dim = env.observation_space["vector"].shape[0]  # 18

    # --- SAC Agent ---
    agent = OdinSACAgent(
        odin_model=model, adapter=adapter,
        obs_dim=obs_dim, action_dim=6,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        gamma=args.gamma, tau=args.tau,
        grad_clip=args.grad_clip, device=device,
    )
    replay_buffer = ReplayBuffer(args.buffer_size)

    # --- CSV ---
    csv_path = output_dir / "sac_metrics.csv"
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if csv_path.stat().st_size == 0:
        csv_w.writerow(["step", "episode", "ep_reward", "ep_steps",
                        "p_hidden", "critic_loss", "actor_loss", "alpha"])

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"  SAC Training: NBV Head as RL Policy (Actor)")
    print(f"  Steps: {args.total_steps} | γ={args.gamma} | freeze={args.freeze_backbone}")
    print(f"{'='*60}\n")

    total_step = 0
    episode = 0
    best_reward = -float("inf")
    reward_history = deque(maxlen=50)

    while total_step < args.total_steps:
        obs, _ = env.reset()
        adapter.reset()
        obs["_raw_rgb"] = env.last_rgb

        ep_reward = 0.0
        ep_steps = 0
        c_loss = a_loss = 0.0

        record_video = (episode + 1) in args.record_episodes
        video_frames = []
        if record_video and env.render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                video_frames.append(frame)

        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Action from NBV Head
            action_np, log_prob, obs_vec, delta_p = agent.get_action(
                obs, adapter, deterministic=False,
            )

            next_obs, _env_rew, terminated, truncated, info = env.step(action_np)
            next_obs["_raw_rgb"] = env.last_rgb

            if record_video and env.render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

            reward = float(delta_p) * config.REWARD_SCALE
            done = terminated or truncated
            next_obs_vec = next_obs["vector"].copy()

            replay_buffer.push(obs_vec, action_np, reward, next_obs_vec, float(done))

            ep_reward += reward
            ep_steps += 1
            total_step += 1

            # SAC updates
            if len(replay_buffer) >= args.learning_starts and total_step % args.update_freq == 0:
                batch = replay_buffer.sample(args.batch_size)
                c_loss = agent.update_critic(batch)
                a_loss, _ = agent.update_actor_and_alpha(batch)
                agent.soft_update_target()

            obs = next_obs

            if done:
                break

        episode += 1
        reward_history.append(ep_reward)

        if record_video and video_frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{episode}.gif"
            import imageio
            imageio.mimsave(str(video_path), video_frames, fps=15)
            print(f"[INFO] Video saved: {video_path}")

        csv_w.writerow([total_step, episode, f"{ep_reward:.4f}", ep_steps,
                        f"{adapter.last_p_hidden:.4f}", f"{c_loss:.4f}",
                        f"{a_loss:.4f}", f"{agent.alpha.item():.4f}"])
        csv_f.flush()

        if episode % 10 == 0:
            avg = np.mean(list(reward_history))
            print(f"[Step {total_step:6d}|Ep {episode:4d}] "
                  f"R={ep_reward:+7.2f}  avg={avg:+7.2f}  "
                  f"p={adapter.last_p_hidden:.3f}  "
                  f"α={agent.alpha.item():.3f}  "
                  f"cL={c_loss:.3f} aL={a_loss:.3f}")

        if total_step % args.save_freq == 0:
            _save(model, agent, output_dir / f"ckpt_step{total_step}.pth")

        if ep_reward > best_reward:
            best_reward = ep_reward
            _save(model, agent, output_dir / "best.pth")

    _save(model, agent, output_dir / "last.pth")
    csv_f.close()
    env.close()
    print(f"\n  Done! Best reward: {best_reward:.2f}. Output: {output_dir}")


def _save(model, agent, path):
    state = {
        "nbv_head": {k: v for k, v in model.state_dict().items()
                     if "nbv_head" in k or "coverage_head" in k},
        "critic": agent.critic.state_dict(),
        "log_std": agent.log_std.data,
        "log_alpha": agent.log_alpha.data,
    }
    torch.save(state, str(path))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
