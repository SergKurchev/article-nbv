"""
train_odin_ppo_rl.py — PPO (Proximal Policy Optimization) — DEFAULT ALGORITHM.

Why PPO over SAC for this task:
  - On-policy: no actor/replay-buffer mismatch by design.
  - Actor and policy are always the same function — no surrogate needed.
  - Simpler to debug: rollout buffer, GAE, single optimizer.
  - Works naturally with visual features (scene_emb from ODIN backbone).

Architecture:
    PyBullet → RGB-D → ODIN (infer_for_rl, backbone frozen by default)
                               ↓
              obs_vec(18) + scene_emb(256) = obs_vec_full(274)
                               ↓
         PPOActorCritic (MLP) → action (6D), value (1D)
         Coverage Head        → p_hidden (BCE supervised, decoupled)

ODIN role in PPO:
  - Provides scene_emb for PPOActorCritic input (visual context).
  - Provides p_hidden for coverage reward.
  - Provides instances_3d for env classification tracking.
  - NBV Head output is NOT used as policy — PPOActorCritic IS the policy.

Per-step flow:
  1. ODIN(obs)  → scene_emb, p_hidden, p_hidden_logit, instances_3d
  2. obs_vec_full = obs["vector"] + scene_emb
  3. PPOActorCritic(obs_vec_full) → action, log_prob, value
  4. env.step(action, instances_3d) → next_obs, _env_rew
  5. ODIN(next_obs) → next_scene_emb, next_p_hidden
  6. reward = env_rew + coverage(delta_p_hidden) + exploration_bonus
  7. push to rollout_buffer; after n_steps: PPO update

Run:
    python train_odin_ppo_rl.py \\
        --odin_weights ./model_final.pth \\
        --odin_cfg my_odin/configs/scannet_context/3d.yaml \\
        --freeze_backbone \\
        --total_steps 200000
"""

import os
import sys
import csv
import signal
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config


# ---------------------------------------------------------------------------
# PPO Actor-Critic
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """
    Shared-trunk MLP Actor-Critic for PPO.

    obs_dim: 18 (obs_vec) + 256 (scene_emb) = 274
    action_dim: 6 (x, y, z, roll, pitch, yaw)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(hidden_dim, 1)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> tuple:
        h = self._features(obs)
        mean = self.actor_head(h)
        std = self.actor_log_std.exp().clamp(min=1e-4, max=2.0)
        dist = Normal(mean, std)
        if action is None:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(h).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._features(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout Buffer (on-policy)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size on-policy rollout buffer with GAE computation."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: float,
    ):
        idx = self.ptr % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Generalized Advantage Estimation in-place."""
        last_gae = 0.0
        n = self.size
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - self.dones[t]
            next_val = last_value if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get(self):
        n = self.size
        return (
            torch.FloatTensor(self.obs[:n]),
            torch.FloatTensor(self.actions[:n]),
            torch.FloatTensor(self.advantages[:n]),
            torch.FloatTensor(self.returns[:n]),
            torch.FloatTensor(self.log_probs[:n]),
        )

    def clear(self):
        self.ptr = 0
        self.size = 0


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.ac = PPOActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

    @torch.no_grad()
    def get_action(self, obs_vec: np.ndarray) -> tuple:
        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.ac.get_action_and_value(obs_t)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, config.ACTION_MIN, config.ACTION_MAX)
        return action_np, log_prob.item(), value.item()

    @torch.no_grad()
    def get_value(self, obs_vec: np.ndarray) -> float:
        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)
        return self.ac.get_value(obs_t).item()

    def update(
        self,
        buffer: RolloutBuffer,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> dict:
        obs, actions, advantages, returns, old_log_probs = [
            x.to(self.device) for x in buffer.get()
        ]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_losses, v_losses, ent_losses, approx_kls = [], [], [], []

        for _ in range(n_epochs):
            perm = torch.randperm(len(obs))
            for start in range(0, len(obs), batch_size):
                idx = perm[start: start + batch_size]
                _, new_lp, entropy, value = self.ac.get_action_and_value(obs[idx], actions[idx])

                ratio = (new_lp - old_log_probs[idx]).exp()
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(value, returns[idx])
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = ((ratio - 1) - (new_lp - old_log_probs[idx])).mean().item()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                approx_kls.append(approx_kl)

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "ent_loss": float(np.mean(ent_losses)),
            "approx_kl": float(np.mean(approx_kls)),
        }


# ---------------------------------------------------------------------------
# ODIN scene embedding — mirrors SAC's per-step ODIN call
# ---------------------------------------------------------------------------

def run_odin_step(obs: dict, adapter, device: str) -> tuple:
    """
    Run ODIN on current obs.
    Returns:
        scene_emb      : np.ndarray (256,)
        p_hidden       : float
        p_hidden_logit : Tensor | None  (for coverage head BCE)
        instances_3d   : dict | None    (for env classification tracking)
        original_xyz   : np.ndarray | None
    """
    rgb = obs.get("_raw_rgb")
    if rgb is None:
        return (
            np.zeros(256, dtype=np.float32),
            adapter.last_p_hidden,
            None, None, None,
        )

    import pybullet as pb

    depth = obs["image"][3] * 10.0
    cam_pos = obs["vector"][:3]
    cam_orn = obs["vector"][3:7]

    R = np.array(pb.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = cam_pos
    fx = fy = config.IMAGE_SIZE / 2.0
    cx = cy = config.IMAGE_SIZE / 2.0

    adapter.add_frame(rgb, depth.astype(np.float32), pose, fx, fy, cx, cy)

    result = adapter.infer_for_rl(
        current_pos=cam_pos.astype(np.float32),
        current_quat=cam_orn.astype(np.float32),
    )

    scene_emb = result.get("scene_embedding", np.zeros(256, dtype=np.float32))
    p_hidden = float(result.get("p_hidden", adapter.last_p_hidden))
    p_hidden_logit = result.get("p_hidden_logit_t", None)
    instances_3d = result.get("instances_3d", None)
    original_xyz = result.get("original_xyz", None)

    return scene_emb, p_hidden, p_hidden_logit, instances_3d, original_xyz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PPO (default): ODIN scene features + MLP actor-critic")
    p.add_argument("--odin_weights", type=str, required=True)
    p.add_argument("--odin_cfg", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=24)

    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze ODIN backbone. Only PPOActorCritic + Coverage Head train.")
    p.add_argument("--train_last_transformer_block", action="store_true",
                   help="Also fine-tune last transformer decoder block of ODIN.")

    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--n_steps", type=int, default=512,
                   help="Rollout length before each PPO update.")
    p.add_argument("--n_epochs", type=int, default=10,
                   help="PPO update epochs per rollout.")
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    p.add_argument("--scene_stage", type=int, default=config.SCENE_STAGE, choices=[1, 2, 3])
    p.add_argument("--no_arm", action="store_true")
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS_PER_EPISODE)
    p.add_argument("--record_episodes", nargs="+", type=int, default=[])

    p.add_argument("--output_dir", type=str, default="./output_odin_ppo")
    p.add_argument("--save_freq", type=int, default=10000)
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ODIN from {args.odin_weights} ...")
    from src.vision.odin_adapter import load_nbv_active_odin
    model, adapter = load_nbv_active_odin(
        args.odin_weights, args.odin_cfg, args.num_classes,
        torch.device(device),
    )

    # Freeze strategy
    if args.freeze_backbone:
        max_layer_idx = -1
        for name in model.state_dict().keys():
            for layer_name in ["transformer_self_attention_layers",
                                "transformer_cross_attention_layers",
                                "transformer_ffn_layers"]:
                if layer_name in name:
                    parts = name.split(layer_name + ".")
                    if len(parts) > 1:
                        idx_str = parts[1].split(".")[0]
                        if idx_str.isdigit():
                            max_layer_idx = max(max_layer_idx, int(idx_str))

        for name, param in model.named_parameters():
            is_trainable = "coverage_head" in name
            if args.train_last_transformer_block and max_layer_idx >= 0:
                is_last_block = any(
                    f"{ln}.{max_layer_idx}." in name
                    for ln in [
                        "transformer_self_attention_layers",
                        "transformer_cross_attention_layers",
                        "transformer_text_cross_attention_layers",
                        "transformer_ffn_layers",
                    ]
                )
                if is_last_block:
                    is_trainable = True
            param.requires_grad_(is_trainable)

        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        suffix = f" + Last Transformer Block {max_layer_idx}" if args.train_last_transformer_block else ""
        print(f"[INFO] --freeze_backbone: ODIN {tp:,} trainable params (Coverage Head{suffix})")
    else:
        for param in model.parameters():
            param.requires_grad_(True)
        print("[INFO] ODIN end-to-end: all params trainable")

    model.eval()

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

    obs_dim = env.observation_space["vector"].shape[0] + 256  # 18 + 256 = 274

    agent = PPOAgent(
        obs_dim=obs_dim, action_dim=6,
        lr=args.lr, clip_eps=args.clip_eps,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, device=device,
    )

    rollout_buffer = RolloutBuffer(args.n_steps, obs_dim, 6)

    # Coverage head optimizer — supervised BCE, independent of PPO
    coverage_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "coverage_head" in n
    ]
    coverage_optimizer = torch.optim.Adam(coverage_params, lr=1e-4) if coverage_params else None

    # CSV logging
    csv_path = output_dir / "ppo_metrics.csv"
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if csv_path.stat().st_size == 0:
        csv_w.writerow([
            "step", "episode", "ep_reward", "ep_steps", "success",
            "p_hidden_mean", "collision_rate", "oob_rate",
            "pg_loss", "v_loss", "ent_loss", "approx_kl", "coverage_loss",
            "rew_env", "rew_coverage", "rew_exploration",
        ])

    print(f"\n{'='*60}")
    print(f"  PPO (default): ODIN scene_emb + MLP ActorCritic")
    print(f"  obs_dim={obs_dim}  total_steps={args.total_steps}")
    print(f"  n_steps={args.n_steps}  n_epochs={args.n_epochs}  batch={args.batch_size}")
    print(f"  lr={args.lr}  γ={args.gamma}  λ={args.gae_lambda}  clip={args.clip_eps}")
    print(f"  freeze={args.freeze_backbone}  stage={args.scene_stage}  max_steps={args.max_steps}")
    print(f"{'='*60}\n")

    total_step = 0
    episode = 0
    best_reward = -float("inf")
    reward_history = deque(maxlen=50)

    _coll_thresh = config.PENALTY_COLLISION / 2.0
    _oob_thresh = config.PENALTY_OOB / 2.0

    ppo_metrics = {"pg_loss": 0.0, "v_loss": 0.0, "ent_loss": 0.0, "approx_kl": 0.0}

    # Episode-level accumulators
    ep_reward = ep_env_rew = ep_coverage_rew = ep_exploration_rew = 0.0
    ep_steps = ep_collisions = ep_oob = 0
    ep_p_hidden_sum = 0.0
    ep_cov_losses = []
    success_flag = 0

    obs, _ = env.reset()
    adapter.reset()
    obs["_raw_rgb"] = env.last_rgb

    # First ODIN call: get scene_emb + instances_3d for the initial obs
    scene_emb, p_hidden, p_hidden_logit, instances_3d, original_xyz = run_odin_step(
        obs, adapter, device
    )
    obs_vec_full = np.concatenate([obs["vector"], scene_emb])

    while total_step < args.total_steps:
        rollout_buffer.clear()

        for _ in range(args.n_steps):
            # ---- Train Coverage Head on current obs (before env.step) ----
            current_gt_p_hidden = 1.0 if env.num_hidden_objects > 0 else 0.0
            if p_hidden_logit is not None and coverage_optimizer is not None:
                gt_t = torch.tensor(
                    [current_gt_p_hidden], dtype=torch.float32, device=device
                )
                loss_cov = F.binary_cross_entropy_with_logits(
                    p_hidden_logit.view(-1), gt_t.view(-1)
                )
                coverage_optimizer.zero_grad()
                loss_cov.backward()
                coverage_optimizer.step()
                ep_cov_losses.append(loss_cov.item())

            # ---- PPO action from learned actor ----
            action_np, log_prob, value = agent.get_action(obs_vec_full)

            # ---- Environment step — pass instances_3d from current ODIN call ----
            next_obs, _env_rew, terminated, truncated, info = env.step(
                action_np,
                info_from_adapter={
                    "instances_3d": instances_3d,
                    "original_xyz": original_xyz,
                },
            )
            next_obs["_raw_rgb"] = env.last_rgb

            # ---- ODIN on next_obs — get next coverage state ----
            with torch.no_grad():
                next_scene_emb, next_p_hidden, next_p_hidden_logit, next_instances_3d, next_original_xyz = (
                    run_odin_step(next_obs, adapter, device)
                )

            # ---- Reward ----
            delta_p_hidden = p_hidden - next_p_hidden
            rew_coverage = delta_p_hidden * config.REWARD_SCALE

            is_collision = _env_rew <= _coll_thresh
            is_oob = (not is_collision) and (_env_rew <= _oob_thresh)
            if is_collision:
                ep_collisions += 1
            if is_oob:
                ep_oob += 1

            reward = float(_env_rew)
            rew_expl = 0.0
            if not is_collision and not is_oob:
                reward += rew_coverage
                if p_hidden < config.REWARD_EXPLORATION_THRESHOLD:
                    rew_expl = config.REWARD_EXPLORATION_COMPLETE_BONUS
                else:
                    rew_expl = (1.0 - p_hidden) * config.REWARD_EXPLORATION_PARTIAL_FACTOR
                reward += rew_expl

            done = terminated or truncated
            rollout_buffer.push(obs_vec_full, action_np, reward, value, log_prob, float(done))

            ep_reward += reward
            ep_env_rew += float(_env_rew)
            ep_coverage_rew += rew_coverage
            ep_exploration_rew += rew_expl
            ep_steps += 1
            ep_p_hidden_sum += p_hidden
            total_step += 1

            if info.get("success", False):
                success_flag = 1

            # ---- Advance state ----
            obs = next_obs
            scene_emb = next_scene_emb
            p_hidden = next_p_hidden
            p_hidden_logit = next_p_hidden_logit
            instances_3d = next_instances_3d
            original_xyz = next_original_xyz
            obs_vec_full = np.concatenate([obs["vector"], scene_emb])

            if done:
                # ---- Episode end: log ----
                episode += 1
                reward_history.append(ep_reward)
                avg_cov_loss = float(np.mean(ep_cov_losses)) if ep_cov_losses else 0.0
                p_hidden_mean = ep_p_hidden_sum / max(ep_steps, 1)
                collision_rate = ep_collisions / max(ep_steps, 1)
                oob_rate = ep_oob / max(ep_steps, 1)

                csv_w.writerow([
                    total_step, episode, f"{ep_reward:.4f}", ep_steps, success_flag,
                    f"{p_hidden_mean:.4f}", f"{collision_rate:.3f}", f"{oob_rate:.3f}",
                    f"{ppo_metrics['pg_loss']:.4f}", f"{ppo_metrics['v_loss']:.4f}",
                    f"{ppo_metrics['ent_loss']:.4f}", f"{ppo_metrics['approx_kl']:.4f}",
                    f"{avg_cov_loss:.4f}",
                    f"{ep_env_rew:.4f}", f"{ep_coverage_rew:.4f}", f"{ep_exploration_rew:.4f}",
                ])
                csv_f.flush()

                if episode % args.log_freq == 0:
                    avg = float(np.mean(list(reward_history))) if reward_history else 0.0
                    print(
                        f"[Step {total_step:6d}|Ep {episode:4d}] "
                        f"R={ep_reward:+7.2f}  avg={avg:+7.2f}  "
                        f"p={p_hidden_mean:.3f}  col={collision_rate:.2f}  oob={oob_rate:.2f}  "
                        f"pgL={ppo_metrics['pg_loss']:.3f}  vL={ppo_metrics['v_loss']:.3f}  "
                        f"kl={ppo_metrics['approx_kl']:.4f}  covL={avg_cov_loss:.3f}"
                    )

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    _save(model, agent, output_dir / "best.pth")

                # Reset episode counters
                ep_reward = ep_env_rew = ep_coverage_rew = ep_exploration_rew = 0.0
                ep_steps = ep_collisions = ep_oob = 0
                ep_p_hidden_sum = 0.0
                ep_cov_losses = []
                success_flag = 0

                obs, _ = env.reset()
                adapter.reset()
                obs["_raw_rgb"] = env.last_rgb
                scene_emb, p_hidden, p_hidden_logit, instances_3d, original_xyz = (
                    run_odin_step(obs, adapter, device)
                )
                obs_vec_full = np.concatenate([obs["vector"], scene_emb])

        # ---- PPO update after full rollout ----
        last_value = agent.get_value(obs_vec_full)
        rollout_buffer.compute_gae(
            last_value, gamma=args.gamma, gae_lambda=args.gae_lambda
        )
        ppo_metrics = agent.update(
            rollout_buffer, n_epochs=args.n_epochs, batch_size=args.batch_size
        )

        if total_step % args.save_freq == 0:
            _save(model, agent, output_dir / f"ckpt_step{total_step}.pth")

    _save(model, agent, output_dir / "last.pth")
    csv_f.close()
    env.close()
    print(f"\n  Done! Best reward: {best_reward:.2f}. Output: {output_dir}")


def _save(model, agent, path):
    state = {
        "coverage_head": {k: v for k, v in model.state_dict().items()
                          if "coverage_head" in k},
        "actor_critic": agent.ac.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
    }
    torch.save(state, str(path))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
