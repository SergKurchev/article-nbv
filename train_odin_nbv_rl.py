"""
train_odin_nbv_rl.py — REINFORCE обучение, где NBV Head модели ODIN
                       ЯВЛЯЕТСЯ RL-агентом (политикой).

Архитектура:
    PyBullet → RGB-D → ODIN backbone → scene features
                                           ↓
                                     NBV Head → next_pose (ACTION)
                                     Coverage Head → p_hidden (REWARD signal)

    Reward = prev_p_hidden − cur_p_hidden  (снижение неопределённости)
    Loss  = −E[log π(a|s) · R]             (REINFORCE with baseline)

    НЕТ ОТДЕЛЬНОГО SAC АГЕНТА. NBV Head — это и есть политика.

Флаги заморозки:
    --freeze_backbone : Замораживает ODIN backbone (ResNet + pixel decoder +
                        Coverage Head). Обучается ТОЛЬКО NBV Head.
                        Быстрее, стабильнее, меньше GPU памяти.
    (без флага)       : End-to-end. Градиенты текут через весь backbone.
                        Медленнее, больше памяти, но backbone тоже адаптируется.

Запуск (Kaggle):
    python train_odin_nbv_rl.py \\
        --odin_weights /path/to/model_final.pth \\
        --odin_cfg my_odin/configs/scannet_context/3d.yaml \\
        --freeze_backbone \\
        --total_episodes 500 \\
        --output_dir ./output_nbv_rl

Запуск (Local, без ODIN, mock-тест):
    python -m pytest tests/test_odin_nbv_agent.py -v
"""

import os
import sys
import csv
import signal
import argparse
import math
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Differentiable quaternion → euler conversion
# ---------------------------------------------------------------------------

def quat_to_euler_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Кватернион (x, y, z, w) → эйлер (roll, pitch, yaw).
    Полностью дифференцируемая реализация на PyTorch.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (X)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr, cosr)

    # Pitch (Y) — clamp to avoid NaN в asin
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (Z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny, cosy)

    return torch.stack([roll, pitch, yaw], dim=-1)


# ---------------------------------------------------------------------------
# Утилиты REINFORCE
# ---------------------------------------------------------------------------

def compute_discounted_returns(rewards: list, gamma: float) -> list:
    """Вычисление дисконтированных returns G_t = Σ γ^k · r_{t+k}."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# Аргументы CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="REINFORCE training: ODIN NBV Head as RL policy"
    )
    # Модель ODIN
    p.add_argument("--odin_weights", type=str, required=True,
                    help="Path to NBVActiveODIN weights (.pth)")
    p.add_argument("--odin_cfg", type=str, required=True,
                    help="Path to ODIN YAML config")
    p.add_argument("--num_classes", type=int, default=24)

    # Заморозка
    p.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze ODIN backbone + Coverage Head. Train ONLY NBV Head.")

    # RL гиперпараметры
    p.add_argument("--total_episodes", type=int, default=500,
                    help="Число эпизодов обучения")
    p.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate для NBV Head")
    p.add_argument("--exploration_std", type=float, default=0.1,
                    help="Стандартное отклонение шума исследования (Gaussian)")
    p.add_argument("--std_decay", type=float, default=0.999,
                    help="Множитель decay для exploration_std каждый эпизод")
    p.add_argument("--std_min", type=float, default=0.01,
                    help="Минимальный exploration_std")
    p.add_argument("--grad_clip", type=float, default=1.0,
                    help="Max gradient norm для стабильности")
    p.add_argument("--baseline_window", type=int, default=50,
                    help="Скользящее окно для baseline REINFORCE")

    # Среда
    p.add_argument("--scene_stage", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--no_arm", action="store_true")
    p.add_argument("--max_steps", type=int, default=10,
                    help="Шагов в эпизоде (default: config.MAX_STEPS_PER_EPISODE)")
    p.add_argument("--record_episodes", nargs="+", type=int, default=[],
                    help="Список эпизодов для записи видео (например: 10 50 100)")

    # Вывод
    p.add_argument("--output_dir", type=str, default="./output_nbv_rl")
    p.add_argument("--save_freq", type=int, default=50,
                    help="Сохранять checkpoint каждые N эпизодов")
    p.add_argument("--log_freq", type=int, default=10,
                    help="Печатать метрики каждые N эпизодов")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Основной тренировочный цикл
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Устройство ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --- Папки ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Загрузка NBVActiveODIN ---
    print(f"[INFO] Loading NBVActiveODIN from {args.odin_weights} ...")
    from src.vision.odin_adapter import load_nbv_active_odin, ODINAdapter
    model, adapter = load_nbv_active_odin(
        weights_path=args.odin_weights,
        cfg_path=args.odin_cfg,
        num_classes=args.num_classes,
        device=device,
    )
    print("[INFO] NBVActiveODIN loaded.")

    # --- Заморозка ---
    if args.freeze_backbone:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if "nbv_head" in name or "coverage_head" in name:
                param.requires_grad_(True)
                trainable_count += 1
            else:
                # Замораживаем backbone, pixel decoder
                param.requires_grad_(False)
                frozen_count += 1
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] --freeze_backbone: frozen {frozen_count} param groups, "
              f"trainable {trainable_count} (NBV Head + Coverage Head).")
        print(f"[INFO] Trainable params: {trainable_params:,} / {total_params:,}")
    else:
        # End-to-end: всё размораживается
        for param in model.parameters():
            param.requires_grad_(True)
        trainable_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] End-to-end mode: ALL {trainable_params:,} params trainable.")

    # Переводим модель в train() для NBV Head (BatchNorm/Dropout если есть)
    model.train()

    # --- Оптимизатор (только для NBV Head / Policy) ---
    trainable = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "nbv_head" in n
    ]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    print(f"[INFO] Optimizer (Policy): Adam, lr={args.lr}, params={len(trainable)}")

    # --- Оптимизатор для Coverage Head ---
    coverage_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "coverage_head" in n
    ]
    if len(coverage_params) > 0:
        coverage_optimizer = torch.optim.Adam(coverage_params, lr=1e-4)
    else:
        coverage_optimizer = None
    print(f"[INFO] Optimizer (Coverage): Adam, lr=1e-4, params={len(coverage_params)}")

    # --- Среда ---
    config.SCENE_STAGE = args.scene_stage
    if args.max_steps:
        config.MAX_STEPS_PER_EPISODE = args.max_steps
    from src.simulation.environment_odin import NBVODINEnv

    # Среда создаётся БЕЗ адаптера — ODIN вызывается снаружи
    env = NBVODINEnv(
        render_mode="rgb_array" if args.record_episodes else None,
        headless=True,
        no_arm=args.no_arm,
        odin_adapter=adapter,    # ODIN вызывается снаружи в цикле
        use_nbv_hint=False,
        log_dir=str(output_dir / "logs"),
        external_infer=True,
    )
    print(f"[INFO] Environment ready (stage={args.scene_stage})")

    # --- CSV-логирование ---
    metrics_path = output_dir / "nbv_rl_metrics.csv"
    csv_file = open(metrics_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if metrics_path.stat().st_size == 0:
        csv_writer.writerow([
            "episode", "total_reward", "mean_delta_p", "final_p_hidden",
            "policy_loss", "coverage_loss", "exploration_std", "steps", "success"
        ])

    # --- REINFORCE baseline ---
    reward_history = deque(maxlen=args.baseline_window)
    exploration_std = args.exploration_std
    best_reward = -float("inf")

    # --- Цикл обучения ---
    print(f"\n{'='*60}")
    print(f"  REINFORCE Training: NBV Head as RL Policy")
    print(f"  Episodes: {args.total_episodes}  |  γ={args.gamma}  |  σ₀={exploration_std}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"{'='*60}\n")

    for episode in range(1, args.total_episodes + 1):
        obs, _ = env.reset()
        adapter.reset()

        log_probs = []
        rewards = []
        delta_ps = []
        ep_cov_losses = []
        success_flag = 0

        record_video = episode in args.record_episodes
        video_frames = []
        if record_video and env.render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                video_frames.append(frame)

        for step in range(config.MAX_STEPS_PER_EPISODE):
            # 1. Извлекаем сырые данные из среды
            rgb = env.last_rgb
            if rgb is None:
                rgb = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)

            # Depth из нормализованного наблюдения
            depth = obs["image"][3] * 10.0  # обратная нормализация

            # Позиция и ориентация камеры
            cam_pos = obs["vector"][:3]
            cam_orn = obs["vector"][3:7]

            # Camera-to-world поза
            import pybullet as pb
            R_mat = np.array(pb.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
            pose_c2w = np.eye(4, dtype=np.float32)
            pose_c2w[:3, :3] = R_mat
            pose_c2w[:3, 3] = cam_pos

            # 2. Добавляем кадр в адаптер
            fx = fy = config.IMAGE_SIZE / 2.0
            cx = cy = config.IMAGE_SIZE / 2.0
            adapter.add_frame(
                rgb, depth.astype(np.float32), pose_c2w, fx, fy, cx, cy
            )

            # 3. Прямой проход ODIN с градиентами → NBV Head = ПОЛИТИКА
            result = adapter.infer_for_rl(
                current_pos=cam_pos.astype(np.float32),
                current_quat=cam_orn.astype(np.float32),
            )

            nbv_pos_t = result["nbv_pos_t"]    # (3,) tensor с grad
            nbv_quat_t = result["nbv_quat_t"]  # (4,) tensor с grad
            p_hidden_logit_t = result.get("p_hidden_logit_t")
            instances_3d = result.get("instances_3d")
            original_xyz = result.get("original_xyz")
            delta_p = result["delta_p_hidden"]
            p_hidden = result["p_hidden"]

            # Обучение Coverage Head на GT
            if p_hidden_logit_t is not None and coverage_optimizer is not None:
                gt_p_hidden = float(env.num_hidden_objects) / float(env.total_target_objects)
                gt_tensor = torch.tensor([gt_p_hidden], dtype=torch.float32, device=device)
                loss_cov = F.binary_cross_entropy_with_logits(p_hidden_logit_t.view(-1), gt_tensor.view(-1))
                
                coverage_optimizer.zero_grad()
                loss_cov.backward()
                coverage_optimizer.step()
                ep_cov_losses.append(loss_cov.item())

            # 4. Конвертируем кватернион → эйлер (дифференцируемо)
            euler_t = quat_to_euler_torch(nbv_quat_t)  # (3,)

            # 5. Формируем mean действия: [pos(3), euler(3)]
            mean_action = torch.cat([nbv_pos_t, euler_t])  # (6,) с grad

            # 6. Стохастическая политика: гауссов шум для исследования
            std = torch.tensor(exploration_std, device=device)
            dist = torch.distributions.Normal(mean_action, std)
            action_t = dist.sample()  # (6,) — сэмпл действия
            log_prob = dist.log_prob(action_t).sum()  # скалярный log π(a|s)

            log_probs.append(log_prob)

            # 7. Применяем действие в среде
            action_np = action_t.detach().cpu().numpy().astype(np.float32)
            action_np = np.clip(action_np, config.ACTION_MIN, config.ACTION_MAX)

            obs, _env_reward, terminated, truncated, info = env.step(
                action_np,
                info_from_adapter={"instances_3d": instances_3d, "original_xyz": original_xyz}
            )

            done = terminated or truncated
            if record_video and env.render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

            # 8. Reward = env reward (includes collision and OOB penalties)
            reward = float(_env_reward)

            # Дополнительная награда за то, что Coverage Head считает, что спрятанных предметов больше нет
            if _env_reward > -5.0:  # Нет штрафов (поскольку штрафы -10 и -15)
                if p_hidden < 0.05:
                    reward += 10.0  # Существенный бонус за полное исследование сцены
                else:
                    reward += (1.0 - p_hidden) * 2.0  # Постоянный стимул стремиться к меньшему p_hidden

            rewards.append(reward)
            delta_ps.append(delta_p)
            if info.get("success", False):
                success_flag = 1

            if terminated or truncated:
                break

        # --- REINFORCE update ---
        if len(rewards) > 0 and len(log_probs) > 0:
            returns = compute_discounted_returns(rewards, args.gamma)

            # Baseline: среднее по истории эпизодов
            baseline = np.mean(reward_history) if len(reward_history) > 0 else 0.0

            # Нормализация returns
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            returns_t = returns_t - baseline

            # Policy gradient loss: −Σ log π(a|s) · (G − baseline)
            policy_loss = torch.tensor(0.0, device=device)
            for lp, R in zip(log_probs, returns_t):
                policy_loss = policy_loss - lp * R

            optimizer.zero_grad()
            policy_loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

            optimizer.step()

            ep_reward = sum(rewards)
            reward_history.append(ep_reward)
            loss_val = policy_loss.item()
        else:
            ep_reward = 0.0
            loss_val = 0.0

        # --- Decay exploration ---
        exploration_std = max(
            exploration_std * args.std_decay,
            args.std_min
        )

        # --- Логирование ---
        mean_dp = np.mean(delta_ps) if delta_ps else 0.0
        final_p = adapter.last_p_hidden
        avg_cov_loss = np.mean(ep_cov_losses) if ep_cov_losses else 0.0

        if record_video and video_frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{episode}.gif"
            import imageio
            imageio.mimsave(str(video_path), video_frames, fps=15)
            print(f"[INFO] Video saved: {video_path}")

        csv_writer.writerow([
            episode, f"{ep_reward:.4f}", f"{mean_dp:.4f}", f"{final_p:.4f}",
            f"{loss_val:.4f}", f"{avg_cov_loss:.4f}", f"{exploration_std:.4f}", len(rewards), success_flag
        ])
        csv_file.flush()

        if episode % args.log_freq == 0:
            avg_rew = np.mean(list(reward_history)) if reward_history else 0
            print(f"[Ep {episode:4d}/{args.total_episodes}]  "
                  f"R={ep_reward:+7.2f}  avg_R={avg_rew:+7.2f}  "
                  f"p_hidden={final_p:.3f}  loss={loss_val:.3f}  covL={avg_cov_loss:.3f}  "
                  f"σ={exploration_std:.4f}")

        # --- Checkpoint ---
        if episode % args.save_freq == 0:
            ckpt_path = output_dir / f"nbv_head_ep{episode}.pth"
            _save_nbv_head(model, ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}")

        # --- Best model ---
        if ep_reward > best_reward:
            best_reward = ep_reward
            _save_nbv_head(model, output_dir / "nbv_head_best.pth")

    # --- Финальное сохранение ---
    _save_nbv_head(model, output_dir / "nbv_head_last.pth")
    csv_file.close()
    env.close()

    print(f"\n{'='*60}")
    print(f"  Training complete! Best reward: {best_reward:.2f}")
    print(f"  Outputs: {output_dir}")
    print(f"{'='*60}")


def _save_nbv_head(model, path):
    """Сохраняет только веса NBV Head (и Coverage Head)."""
    state = {
        k: v for k, v in model.state_dict().items()
        if "nbv_head" in k or "coverage_head" in k
    }
    torch.save(state, str(path))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")
