"""
train_odin_rl.py — Скрипт обучения RL агента (SAC) с NBVActiveODIN.

Предназначен для запуска на Kaggle из ноутбука как:
    !venv/bin/python my_odin/train_odin_rl.py \\
        --odin_weights ./output_nbv_stage2/model_final.pth \\
        --odin_cfg my_odin/configs/scannet_context/3d.yaml \\
        --dataset_dir /kaggle/input/datasets/sergeykurchev/nbv-stage2-dataset \\
        --output_dir ./output_rl_odin \\
        --total_timesteps 300000

Архитектура:
    NBVActiveODIN (frozen/partially frozen) → ODINAdapter → NBVODINEnv → SAC
"""

import os
import sys
import argparse
import signal
from pathlib import Path

import numpy as np
import torch
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback


class VideoRecorderCallback(BaseCallback):
    """
    Callback для записи видео конкретных эпизодов.
    """
    def __init__(self, record_episodes, video_save_path, verbose=1):
        super().__init__(verbose)
        self.record_episodes = record_episodes
        self.video_save_path = Path(video_save_path)
        self.video_save_path.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.is_recording = False
        self.frames = []

    def _on_step(self) -> bool:
        # Проверяем, закончился ли эпизод
        dones = self.locals.get("dones")
        
        # Если мы сейчас записываем, добавляем кадр
        if self.is_recording:
            frame = self.training_env.render()
            self.frames.append(frame)

        if dones[0]:
            if self.is_recording:
                # Сохраняем накопленные кадры в видео
                video_name = self.video_save_path / f"episode_{self.episode_count}.mp4"
                imageio.mimsave(str(video_name), self.frames, fps=20)
                print(f"[INFO] Video saved: {video_name}")
                self.is_recording = False
                self.frames = []

            self.episode_count += 1
            # Проверяем, нужно ли записывать СЛЕДУЮЩИЙ эпизод
            if self.episode_count in self.record_episodes:
                print(f"[INFO] Starting video recording for episode {self.episode_count}...")
                self.is_recording = True
                # Захватываем начальный кадр нового эпизода
                self.frames.append(self.training_env.render())

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC + NBVActiveODIN RL Agent")
    parser.add_argument("--odin_weights", type=str, required=True,
                        help="Path to NBVActiveODIN weights (.pth)")
    parser.add_argument("--odin_cfg", type=str, required=True,
                        help="Path to ODIN config YAML (configs/scannet_context/3d.yaml)")
    parser.add_argument("--dataset_dir", type=str, default="",
                        help="Dataset dir (NBV Stage 2). Used only for logging.")
    parser.add_argument("--output_dir", type=str, default="./output_rl_odin",
                        help="Output directory for checkpoints and logs.")
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=24,
                        help="Number of ODIN classes (24 for NBV Stage 2)")
    parser.add_argument("--freeze_odin", action="store_true",
                        help="Freeze ODIN backbone, train only heads + SAC policy.")
    parser.add_argument("--no_arm", action="store_true",
                        help="Run without robot arm (camera flies freely).")
    parser.add_argument("--scene_stage", type=int, default=2, choices=[1, 2, 3],
                        help="Scene stage (1=single obj, 2=multi obj, 3=multi+obstacles)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to existing SAC checkpoint to resume training.")
    parser.add_argument("--record_episodes", type=int, nargs="+", default=[],
                        help="List of episode numbers to record video for (e.g. 0 10 50)")
    return parser.parse_args()


def handle_interrupt(sig, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, handle_interrupt)


def make_odin_env(odin_adapter, no_arm: bool, headless: bool):
    """Фабрика для DummyVecEnv."""
    def _init():
        import config
        config.SCENE_STAGE = args.scene_stage  # Задаём стадию сцены
        from src.simulation.environment_odin import NBVODINEnv
        return NBVODINEnv(
            render_mode="rgb_array",
            headless=headless,
            no_arm=no_arm,
            odin_adapter=odin_adapter,
            use_nbv_hint=True,
        )
    return _init


def main():
    global args
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # --------------- Устройство ---------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --------------- Загрузка NBVActiveODIN ---------------
    print(f"[INFO] Loading NBVActiveODIN from {args.odin_weights}")
    from src.vision.odin_adapter import load_nbv_active_odin, ODINAdapter
    model, adapter = load_nbv_active_odin(
        weights_path=args.odin_weights,
        cfg_path=args.odin_cfg,
        num_classes=args.num_classes,
        device=device,
    )
    print("[INFO] NBVActiveODIN loaded successfully.")

    # Заморозка backbone (опционально)
    if args.freeze_odin:
        frozen_count = 0
        for name, param in model.named_parameters():
            # Размораживаем только Coverage Head и NBV Head
            if "coverage_head" not in name and "nbv_head" not in name:
                param.requires_grad_(False)
                frozen_count += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Frozen {frozen_count} ODIN params. Trainable: {trainable:,}/{total:,}")

    # --------------- Создание сред ---------------
    print("[INFO] Creating environments...")
    env = DummyVecEnv([make_odin_env(adapter, args.no_arm, headless=True)])
    eval_env = DummyVecEnv([make_odin_env(adapter, args.no_arm, headless=True)])
    print("[INFO] Environments ready.")

    # --------------- SAC агент ---------------
    print("[INFO] Building SAC agent...")
    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming SAC from {args.resume}")
        sac_model = SAC.load(args.resume, env=env, device=str(device))
        sac_model.learning_starts = 0  # Не ждём при возобновлении
    else:
        sac_model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=1,
            device=str(device),
        )

    # --------------- Callbacks ---------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix="sac_odin",
    )
    
    callbacks_list = [eval_callback, checkpoint_callback]
    
    if args.record_episodes:
        video_callback = VideoRecorderCallback(
            record_episodes=args.record_episodes,
            video_save_path=output_dir / "videos"
        )
        callbacks_list.append(video_callback)
        
    callbacks = CallbackList(callbacks_list)

    # --------------- Обучение ---------------
    print(f"[INFO] Starting SAC training for {args.total_timesteps} timesteps...")
    try:
        sac_model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
    finally:
        last_path = str(output_dir / "sac_odin_last")
        sac_model.save(last_path)
        print(f"[INFO] Last policy saved to {last_path}.zip")

        # Сохраняем также веса ODIN Coverage+NBV голов (если дообучали)
        if not args.freeze_odin:
            heads_path = str(output_dir / "odin_heads_finetuned.pth")
            heads_state = {
                k: v for k, v in model.state_dict().items()
                if "coverage_head" in k or "nbv_head" in k
            }
            torch.save(heads_state, heads_path)
            print(f"[INFO] ODIN heads saved to {heads_path}")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
