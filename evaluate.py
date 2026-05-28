"""
evaluate.py — Unified policy comparison: SAC vs geometric vs uncertainty baseline.

Loads a trained SAC checkpoint (ODIN + ActorMLP), runs each specified policy for
--num_episodes episodes in identical scene conditions, collects per-step and per-episode
metrics (Wang, Korbach, ODIN uncertainty, novelty), writes CSVs and prints aggregate table.

Usage:
    python evaluate.py \\
        --checkpoint ./output_odin_sac/best.pth \\
        --odin_weights ./model_final.pth \\
        --odin_cfg my_odin/configs/scannet_context/3d.yaml \\
        --policies sac geometric uncertainty \\
        --num_episodes 50 \\
        --scene_stage 2 \\
        --output_dir ./eval_results

Outputs (in --output_dir):
    eval_episode_metrics.csv  — one row per episode with all EpisodeMetrics fields
    eval_step_metrics.csv     — one row per env step
    eval_summary.txt          — aggregate table per policy
    vis/<policy>_ep<N>.html   — HTML point-cloud viewer (if --vis_episodes)
"""

import csv
import io
import signal
import argparse
import builtins
import random
from pathlib import Path

import numpy as np
import torch

import config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_obs_frame(obs: dict, adapter) -> None:
    import pybullet as pb
    rgb = obs.get("_raw_rgb")
    if rgb is None:
        return
    depth = obs["image"][3] * 10.0
    cam_pos = obs["vector"][:3]
    cam_orn = obs["vector"][3:7]
    R = np.array(pb.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = cam_pos
    fx = fy = cx = cy = config.IMAGE_SIZE / 2.0
    adapter.add_frame(rgb, depth.astype(np.float32), pose, fx, fy, cx, cy)


def _gpu_stats() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return f"  GPU={util}% {mem.used/1024**3:.1f}/{mem.total/1024**3:.0f}GB"
    except Exception:
        return f"  gpuMem={torch.cuda.memory_allocated()/1024**3:.1f}GB"


# ---------------------------------------------------------------------------
# Single-policy evaluation loop
# ---------------------------------------------------------------------------

def run_policy_episodes(
    *,
    policy_name: str,
    agent,
    adapter,
    env,
    geometric_policy,
    uncertainty_policy,
    VoxelNoveltyTracker,
    EpisodeMetricsTracker,
    EpisodeVisualizer,
    num_episodes: int,
    num_classes: int,
    max_steps: int,
    output_dir: Path,
    vis_episodes: set,
    step_csv_w,
    ep_csv_w,
    global_step: list,      # [int] mutable counter shared across all policies
) -> list:
    """Run num_episodes for policy_name. Returns list[EpisodeMetrics]."""
    from src.nbv.geometry_memory import GeometryMemory

    all_metrics = []
    _coll_thresh = config.PENALTY_COLLISION / 2.0
    _amin3 = np.array(config.ACTION_MIN[:3], dtype=np.float32)
    _amax3 = np.array(config.ACTION_MAX[:3], dtype=np.float32)

    for ep_idx in range(num_episodes):
        obs, _ = env.reset()
        adapter.reset()
        obs["_raw_rgb"] = env.last_rgb
        _add_obs_frame(obs, adapter)

        ep_reward = ep_env_rew = ep_cov_rew = ep_expl_rew = ep_nov_rew = 0.0
        ep_steps = ep_collisions = ep_oob = 0
        ep_p_hidden_sum = 0.0
        success_flag = 0

        novelty_tracker = VoxelNoveltyTracker(config.REWARD_NOVELTY_VOXEL_SIZE)
        geom_memory = GeometryMemory(
            fx=config.IMAGE_SIZE / 2.0, fy=config.IMAGE_SIZE / 2.0,
            cx=config.IMAGE_SIZE / 2.0, cy=config.IMAGE_SIZE / 2.0,
        )
        gt_classes = (list(env.asset_loader.target_objects_classes)
                      if hasattr(env, "asset_loader") else [])
        tracker = EpisodeMetricsTracker(policy_name, ep_idx, num_classes, max_steps)
        tracker.set_scene(gt_classes)
        if policy_name == "geometric":
            geometric_policy.reset()

        viz = None
        if ep_idx in vis_episodes:
            vis_path = output_dir / "vis" / f"{policy_name}_ep{ep_idx}.html"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            viz = EpisodeVisualizer(vis_path)

        for step in range(max_steps):
            with torch.no_grad():
                (action_np, _log_prob, _obs_vec, _delta_p, p_hidden,
                 _logit_t, instances_3d, original_xyz, nbv_hint_np) = agent.get_action(
                     obs, adapter, deterministic=True,
                 )
            ep_p_hidden_sum += p_hidden

            # Policy override
            if policy_name == "geometric":
                seg_mask = (env.last_seg if hasattr(env, "last_seg")
                            else np.full((config.IMAGE_SIZE, config.IMAGE_SIZE), -1, np.int32))
                best_xyz = geometric_policy.act(
                    seg_mask, obs["image"][3] * 10.0,
                    obs["vector"][:3], obs["vector"][3:7],
                    None, float(obs["vector"][7]),
                )
                action_np[:3] = best_xyz
            elif policy_name == "uncertainty":
                action_np[:3] = uncertainty_policy.act(nbv_hint_np, obs["vector"][:3])

            is_manual_oob = bool(np.any(action_np[:3] < _amin3) or np.any(action_np[:3] > _amax3))
            action_np = np.clip(action_np, config.ACTION_MIN, config.ACTION_MAX)

            next_obs, _env_rew, terminated, truncated, info = env.step(
                action_np,
                info_from_adapter={"instances_3d": instances_3d, "original_xyz": original_xyz},
            )
            next_obs["_raw_rgb"] = env.last_rgb
            _add_obs_frame(next_obs, adapter)

            with torch.no_grad():
                next_out = adapter.infer_for_rl(
                    current_pos=next_obs["vector"][:3].astype(np.float32),
                    current_quat=next_obs["vector"][3:7].astype(np.float32),
                )
                next_p_hidden = float(next_out["p_hidden"])

            delta_p_hidden = p_hidden - next_p_hidden
            rew_coverage = delta_p_hidden * config.REWARD_SCALE

            rew_novelty = 0.0
            if original_xyz is not None and len(original_xyz) > 0:
                rew_novelty = (novelty_tracker.update_and_score(original_xyz)
                               * config.REWARD_NOVELTY_FACTOR)

            geom_memory.update(
                env.last_seg if hasattr(env, "last_seg")
                else np.full((config.IMAGE_SIZE, config.IMAGE_SIZE), -1, np.int32),
                obs["image"][3] * 10.0,
                obs["vector"][:3], obs["vector"][3:7],
            )

            is_collision = _env_rew <= _coll_thresh
            is_oob = is_manual_oob and not is_collision
            if is_oob:
                _env_rew = config.PENALTY_OOB
            ep_collisions += int(is_collision)
            ep_oob += int(is_oob)

            reward = float(_env_rew)
            rew_expl = 0.0
            if not is_collision and not is_oob:
                reward += rew_coverage
                if p_hidden < config.REWARD_EXPLORATION_THRESHOLD:
                    rew_expl = config.REWARD_EXPLORATION_COMPLETE_BONUS
                else:
                    rew_expl = (1.0 - p_hidden) * config.REWARD_EXPLORATION_PARTIAL_FACTOR
                reward += rew_expl + rew_novelty

            done = terminated or truncated
            cur_unc = float(next_out.get("mean_uncertainty", next_p_hidden))

            pred_classes, pred_scores, detected_ids = [], [], []
            if instances_3d is not None:
                try:
                    pred_classes = [int(c) for c in instances_3d.pred_classes.tolist()]
                    pred_scores = [float(s) for s in instances_3d.scores.tolist()]
                    detected_ids = list(range(len(pred_classes)))
                except Exception:
                    pass

            tracker.update(
                reward=reward,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                detected_ids=detected_ids,
                p_hidden=p_hidden,
                uncertainty=cur_unc,
                is_collision=is_collision,
                is_oob=is_oob,
            )

            step_csv_w.writerow([
                global_step[0], policy_name, ep_idx, ep_steps + 1,
                f"{reward:.4f}", f"{float(_env_rew):.4f}",
                f"{rew_coverage:.4f}", f"{rew_expl:.4f}", f"{rew_novelty:.4f}",
                f"{p_hidden:.4f}", f"{delta_p_hidden:.4f}", f"{cur_unc:.4f}",
                int(is_collision), int(is_oob),
            ])
            global_step[0] += 1

            if viz is not None:
                rgb_vis = obs.get("_raw_rgb")
                if rgb_vis is not None:
                    # Build per-pixel ODIN prediction masks from instances_3d
                    pred_cat_mask_vis = None
                    pred_inst_mask_vis = None
                    if instances_3d is not None:
                        try:
                            H = W = config.IMAGE_SIZE
                            pred_cat_mask_vis  = np.full((H, W), -1, dtype=np.int32)
                            pred_inst_mask_vis = np.full((H, W), -1, dtype=np.int32)
                            if hasattr(instances_3d, "pred_masks") and instances_3d.pred_masks is not None:
                                masks = instances_3d.pred_masks  # (N_inst, H, W) bool tensor
                                cats  = instances_3d.pred_classes  # (N_inst,) int tensor
                                masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.array(masks)
                                cats_np  = cats.cpu().numpy()  if hasattr(cats,  "cpu") else np.array(cats)
                                for inst_id, (m, c) in enumerate(zip(masks_np, cats_np)):
                                    pred_cat_mask_vis[m]  = int(c)
                                    pred_inst_mask_vis[m] = inst_id
                        except Exception:
                            pass
                    viz.add_frame(
                        rgb=rgb_vis,
                        depth=obs["image"][3] * 10.0,
                        seg_mask=env.last_seg if hasattr(env, "last_seg") else None,
                        cam_pos=obs["vector"][:3],
                        cam_quat_xyzw=obs["vector"][3:7],
                        pred_cat_mask=pred_cat_mask_vis,
                        pred_inst_mask=pred_inst_mask_vis,
                    )

            ep_reward += reward
            ep_env_rew += float(_env_rew)
            ep_cov_rew += rew_coverage
            ep_expl_rew += rew_expl
            ep_nov_rew += rew_novelty
            ep_steps += 1

            if info.get("success", False):
                success_flag = 1

            obs = next_obs
            if done:
                break

        if viz is not None:
            viz.save(ep_idx)

        ep_m = tracker.finalize(success=bool(success_flag))
        ep_m.geometry_visible_classes = geom_memory.visible_classes
        ep_m.geometry_mean_coverage = geom_memory.mean_coverage
        all_metrics.append(ep_m)

        p_hidden_mean = ep_p_hidden_sum / max(ep_steps, 1)
        ep_csv_w.writerow([
            policy_name, ep_idx, f"{ep_reward:.4f}", ep_steps, success_flag,
            f"{p_hidden_mean:.4f}",
            f"{ep_env_rew:.4f}", f"{ep_cov_rew:.4f}", f"{ep_expl_rew:.4f}", f"{ep_nov_rew:.4f}",
            f"{ep_m.wang_macro_accuracy:.4f}", f"{ep_m.wang_macro_precision:.4f}",
            f"{ep_m.wang_macro_recall:.4f}", f"{ep_m.wang_macro_f1:.4f}",
            f"{ep_m.final_confidence_diff:.4f}", f"{ep_m.best_confidence_diff:.4f}",
            f"{ep_m.uncertainty_reduction:.4f}", f"{ep_m.uncertainty_auc:.4f}",
            f"{ep_m.objects_found_fraction:.4f}", ep_m.objects_correct, ep_m.objects_in_scene,
            ep_m.steps_to_all_objects, ep_m.steps_to_all_correct,
            ep_m.geometry_visible_classes, f"{ep_m.geometry_mean_coverage:.4f}",
            f"{ep_m.p_hidden_initial:.4f}", f"{ep_m.p_hidden_final:.4f}",
            ep_m.collision_steps, ep_m.out_of_reach_steps,
        ])

        print(
            f"  [{policy_name}] ep={ep_idx:3d}  "
            f"R={ep_reward:+7.2f}  p={p_hidden_mean:.3f}  "
            f"suc={success_flag}  col={ep_collisions}  oob={ep_oob}  "
            f"wang_f1={ep_m.wang_macro_f1:.3f}"
            + _gpu_stats()
        )

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SAC / geometric / uncertainty policies")
    p.add_argument("--odin_weights", type=str, required=True)
    p.add_argument("--odin_cfg", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="SAC .pth checkpoint from train_odin_sac_rl.py "
                        "(required when 'sac' is in --policies)")
    p.add_argument("--policies", nargs="+",
                   choices=["sac", "geometric", "uncertainty"],
                   default=["sac", "geometric", "uncertainty"])
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--scene_stage", type=int, default=config.SCENE_STAGE, choices=[1, 2, 3])
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS_PER_EPISODE)
    p.add_argument("--no_arm", action="store_true")
    p.add_argument("--swag", action="store_true",
                   help="Enable SWAG Bayesian inference (10 MC samples)")
    p.add_argument("--vis_episodes", nargs="*", type=int, default=[],
                   help="Episode indices (per policy) to save HTML point-cloud visualization")
    p.add_argument("--output_dir", type=str, default="./eval_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from src.nbv.metrics import EpisodeMetricsTracker, aggregate_metrics, print_metrics_table
    from src.nbv.novelty import VoxelNoveltyTracker
    from src.nbv.policies.info_gain import GeometricNBVPolicy
    from src.nbv.policies.uncertainty import UncertaintyPolicy
    from src.nbv.visualization import EpisodeVisualizer

    print(f"[INFO] Loading ODIN from {args.odin_weights} ...")
    from src.vision.odin_adapter import load_nbv_active_odin
    swag_opts = (["MODEL.BAYESIAN_TYPE", "swag", "MODEL.BAYESIAN_SAMPLES", "10",
                  "MODEL.SWAG.SCALE", "1.0"] if args.swag else None)
    model, adapter = load_nbv_active_odin(
        args.odin_weights, args.odin_cfg, args.num_classes,
        torch.device(device), extra_opts=swag_opts,
    )
    model.eval()
    if args.swag:
        print("[INFO] SWAG Bayesian inference enabled (10 MC samples)")

    config.SCENE_STAGE = args.scene_stage
    config.MAX_STEPS_PER_EPISODE = args.max_steps

    from src.simulation.environment_odin import NBVODINEnv
    env = NBVODINEnv(
        headless=True, no_arm=args.no_arm,
        odin_adapter=adapter, use_nbv_hint=False,
        log_dir=str(output_dir / "logs"),
        external_infer=True,
    )

    obs_dim = env.observation_space["vector"].shape[0] + 256

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_odin_sac_rl import OdinSACAgent

    agent = OdinSACAgent(
        odin_model=model, adapter=adapter,
        obs_dim=obs_dim, action_dim=6,
        device=device,
    )

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "actor_mlp" in ckpt:
            agent.actor_mlp.load_state_dict(ckpt["actor_mlp"])
        if "critic" in ckpt:
            agent.critic.load_state_dict(ckpt["critic"])
        if "log_std" in ckpt:
            agent.log_std.data.copy_(ckpt["log_std"])
        if "log_alpha" in ckpt:
            agent.log_alpha.data.copy_(ckpt["log_alpha"])
        if "nbv_head" in ckpt:
            model.load_state_dict(ckpt["nbv_head"], strict=False)
        print("[INFO] Checkpoint loaded.")
    elif "sac" in args.policies:
        print("[WARN] --checkpoint not given; SAC actor uses random weights.")

    agent.actor_mlp.eval()
    agent.critic.eval()

    geometric_policy = GeometricNBVPolicy()
    uncertainty_policy = UncertaintyPolicy()
    vis_set = set(args.vis_episodes or [])

    # CSV setup
    step_path = output_dir / "eval_step_metrics.csv"
    ep_path = output_dir / "eval_episode_metrics.csv"
    step_f = open(step_path, "w", newline="")
    ep_f = open(ep_path, "w", newline="")
    step_w = csv.writer(step_f)
    ep_w = csv.writer(ep_f)

    step_w.writerow([
        "total_step", "policy", "episode", "ep_step", "reward",
        "rew_env", "rew_coverage", "rew_exploration", "rew_novelty",
        "p_hidden", "delta_p_hidden", "uncertainty",
        "collision", "oob",
    ])
    ep_w.writerow([
        "policy", "episode", "ep_reward", "ep_steps", "success",
        "p_hidden_mean",
        "rew_env", "rew_coverage", "rew_exploration", "rew_novelty",
        "wang_macro_accuracy", "wang_macro_precision", "wang_macro_recall", "wang_macro_f1",
        "korbach_final_confidence_diff", "korbach_best_confidence_diff",
        "odin_uncertainty_reduction", "odin_uncertainty_auc",
        "objects_found_fraction", "objects_correct", "objects_in_scene",
        "steps_to_all_objects", "steps_to_all_correct",
        "geometry_visible_classes", "geometry_mean_coverage",
        "p_hidden_initial", "p_hidden_final",
        "collision_steps", "out_of_reach_steps",
    ])

    global_step = [0]
    all_logs = []

    for policy_name in args.policies:
        print(f"\n{'='*60}")
        print(f"  Policy: {policy_name}  |  {args.num_episodes} episodes  |  stage={args.scene_stage}")
        print(f"{'='*60}")

        logs = run_policy_episodes(
            policy_name=policy_name,
            agent=agent,
            adapter=adapter,
            env=env,
            geometric_policy=geometric_policy,
            uncertainty_policy=uncertainty_policy,
            VoxelNoveltyTracker=VoxelNoveltyTracker,
            EpisodeMetricsTracker=EpisodeMetricsTracker,
            EpisodeVisualizer=EpisodeVisualizer,
            num_episodes=args.num_episodes,
            num_classes=args.num_classes,
            max_steps=args.max_steps,
            output_dir=output_dir,
            vis_episodes=vis_set,
            step_csv_w=step_w,
            ep_csv_w=ep_w,
            global_step=global_step,
        )
        all_logs.extend(logs)
        step_f.flush()
        ep_f.flush()

    step_f.close()
    ep_f.close()
    env.close()

    # Aggregate and print
    agg = aggregate_metrics(all_logs)

    print(f"\n{'='*60}")
    print(f"  Aggregate  ({args.num_episodes} eps × {len(args.policies)} policies)")
    print(f"{'='*60}")
    print_metrics_table(agg)

    # Capture table for file
    buf = io.StringIO()
    _orig = builtins.print
    builtins.print = lambda *a, **kw: _orig(*a, **{**kw, "file": buf})
    print_metrics_table(agg)
    builtins.print = _orig

    summary_path = output_dir / "eval_summary.txt"
    summary_path.write_text(buf.getvalue())

    print(f"\nResults → {output_dir}/")
    print(f"  {step_path.name}")
    print(f"  {ep_path.name}")
    print(f"  {summary_path.name}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
