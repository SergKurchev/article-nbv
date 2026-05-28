# Implementation Plan: NBV Evaluation & UR3 Migration

Branch: `metrics/evaluation`

---

## Overview

Nine changes in dependency order. Each section lists exact files, what changes, and which existing code from strawberry/ODIN is reused vs. written fresh.

---

## 1. Shapes — Restrict to 5 (Cube, Pyramid, Sphere, Cylinder, Hourglass)

**Files changed:**

### `config.py`
```python
# Add after NUM_CLASSES:
ACTIVE_SHAPE_IDS = [0, 1, 3, 5, 6]   # Object_01=Sphere, 02=Cube, 04=Pyramid, 06=Cylinder, 07=Hourglass
NUM_CLASSES = 5
SHAPE_NAMES = ["sphere", "cube", "pyramid", "cylinder", "hourglass"]
```

### `src/simulation/asset_loader.py`
- In `generate_scene()`, change `random.randint(0, NUM_CLASSES-1)` → `random.choice(ACTIVE_SHAPE_IDS)`
- Existing objects for these 5 shapes are already defined (Object_01,02,04,06,07)
- No mesh files change

### `src/data/generate_primitives.py`
- No change needed (shapes already exist)

---

## 2. UR3 Robot

### 2a. Create `src/data/robot/ur3.urdf`

URDF from scratch using UR3 DH parameters (d1=0.1519, a2=0.24365, a3=0.21325, d4=0.11235, d5=0.08535, d6=0.0819, all metres). Minimal inertia, no mesh needed — use collision cylinders/boxes. 6 revolute joints: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3. Base at origin.

### 2b. `config.py`
```python
ROBOT_TYPE = "ur3"                                      # was "kuka"
ROBOT_URDF = "src/data/robot/ur3.urdf"                  # relative path
UR3_BASE_POS = [0.0, 0.0, 0.0]
UR3_BASE_ORN = [0, 0, 0, 1]

# UR3 reachable workspace (radius ~0.50m from base):
WORKSPACE_X = (0.10, 0.50)    # was (0.2, 0.8)
WORKSPACE_Y = (-0.25, 0.25)   # was (-0.3, 0.3)
WORKSPACE_Z = (0.10, 0.45)    # was (0.15, 0.4)

# Action space adjusted to UR3 reach:
ACTION_MIN = [0.10, -0.25, 0.10, -pi, -pi, -pi]   # was [0.2,-0.5,0.0,...]
ACTION_MAX = [0.50,  0.25, 0.45,  pi,  pi,  pi]   # was [0.8, 0.5,0.8,...]

# Object placement zone (inside UR3 reach):
OBJ_ZONE_X = (0.20, 0.40)
OBJ_ZONE_Y = (-0.15, 0.15)
```

### 2c. `src/simulation/robot.py`
- Change `home_pos` from 7-joint KUKA to 6-joint UR3: `[0, -1.57, 1.57, -1.57, -1.57, 0]`
- Change `range(min(7, ...))` → `range(min(6, ...))`
- `get_joint_states()` returns 6 values instead of 7
- IK end-effector link index changes (KUKA=6 → UR3=5, last revolute joint + flange)
- Load URDF from config path instead of `pybullet_data`
- Keep `apply_action()` logic identical (IK + resetJointState + setJointMotorControl2)

### 2d. `src/simulation/environment_odin.py`
- `VECTOR_DIM = 17` (was 18): pos(3) + orn(4) + delta_p(1) + joints(6, was 7) + nbv_hint(3)
- Update `obs["vector"]` construction at lines ~654-661
- `obs_vec_full` in training loop becomes 273D (17 + 256) — update any hardcoded 274/18 references

### 2e. `train_odin_sac_rl.py`
- `OBS_DIM = 17` (update constant)
- `ActorMLP` input 274 → 273
- `QNetwork` input unchanged (derives from tensor slicing, but update `obs[18:274]` → `obs[17:273]` and `obs[7:8]`, `obs[8:15]` → `obs[8:14]` for joints, `obs[15:18]` → `obs[14:17]` for nbv_hint)
- Update ACTION_MIN/MAX loading from config

---

## 3. Metrics (already implemented)

`src/nbv/metrics.py` — **DONE** (created in previous session). Contains:
- `EpisodeMetrics` dataclass — all Wang/Korbach/ODIN fields
- `EpisodeMetricsTracker` — stateful accumulator with `update()` / `finalize()`
- `macro_classification_metrics()` — support-set macro one-vs-all
- `aggregate_metrics()`, `print_metrics_table()`

### Integration into `train_odin_sac_rl.py`

**Per-step CSV** (new file: `output_dir/logs/step_metrics.csv`):
```
step, episode, reward, env_reward, coverage_reward, exploration_reward,
p_hidden, delta_p_hidden, uncertainty, critic_loss, actor_loss,
coverage_loss, alpha, collision, oob
```

**Per-episode CSV** (existing `sac_metrics.csv` + add all EpisodeMetrics fields):
```
episode, policy, seed, steps, total_reward, correct, objects_correct,
objects_found, objects_found_fraction, objects_in_scene, target_correct,
wang_macro_accuracy, wang_macro_precision, wang_macro_recall, wang_macro_f1,
korbach_final_confidence_diff, korbach_best_confidence_diff, korbach_steps_to_best,
odin_uncertainty_reduction, odin_uncertainty_auc, steps_to_all_objects,
steps_to_all_correct, p_hidden_initial, p_hidden_final,
geometry_visible_classes, geometry_mean_coverage,
collision_steps, out_of_reach_steps
```

**How**: instantiate `EpisodeMetricsTracker` at episode start, call `.update()` each step, `.finalize()` on done, write to CSV.

---

## 4. GeometryMemory (already implemented)

`src/nbv/geometry_memory.py` — **DONE** (created in previous session). Contains:
- `GeometryMemory.update(seg_mask, depth, cam_pos, cam_quat_xyzw, uncertainty)`
- `score_candidates(candidate_positions_Nx3)` — Wang-style scores
- `_history_novelty()`, `visible_classes`, `mean_coverage` properties

---

## 5. Policies: Geometric NBV and Uncertainty Baselines

### 5a. `src/nbv/sampler.py`

**Copy directly from** `C:/temp/strawberry_nbv/odin_connection/nbv/sampler.py` with two changes:
- Replace strawberry's `WORKSPACE_CENTER / HEMI_RADIUS / N_CANDIDATES` imports with our config values: center `(0.30, 0.0, 0.25)`, radius `0.35`, N=64
- Keep all math identical (Fibonacci golden-angle, `_look_at_quat`, `snap_to_nearest`)

### 5b. `src/nbv/policies/info_gain.py`

**Copy from** `C:/temp/strawberry_nbv/odin_connection/nbv/policies/info_gain.py` and adapt:
- `InfoGainPolicy` → rename to `GeometricNBVPolicy` (or keep name, wrap in `GeometricNBVBaseline`)
- Replace strawberry's `SharedState` with our env's observation format
- `act(obs_vec, cam_pos, seg_mask, depth, cam_quat, uncertainty)` → calls `geom_mem.update()` then `geom_mem.score_candidates(sampler_positions)`
- Returns best candidate position (3D xyz)

### 5c. `src/nbv/policies/uncertainty.py` (new, ~30 lines)

Pure uncertainty baseline: at each step, pick candidate from hemisphere with highest expected uncertainty reduction. Uses only `p_hidden` and `mean_uncertainty` from ODIN — no geometry tracking. Greedy: move toward NBV hint from ODIN.

```python
class UncertaintyPolicy:
    def act(self, nbv_hint, current_pos):
        # Simply follow ODIN's NBV head suggestion
        return np.clip(nbv_hint, ACTION_MIN[:3], ACTION_MAX[:3])
```

### 5d. `train_odin_sac_rl.py` — policy dispatch

Add CLI argument:
```python
parser.add_argument("--policy", choices=["sac", "geometric", "uncertainty"], default="sac")
```

At the top of the episode loop:
```python
if args.policy == "sac":
    action = get_action(obs_vec_full, agent, ...)
elif args.policy == "geometric":
    action = geometric_policy.act(obs_vec, cam_pos, seg_mask, depth, cam_quat, unc)
elif args.policy == "uncertainty":
    action = uncertainty_policy.act(result["nbv_pos_t"], current_pos)
```

SAC training (critic/actor updates) only runs when `args.policy == "sac"`.

---

## 6. SWAG Inference

### `train_odin_sac_rl.py`

Add CLI argument:
```python
parser.add_argument("--swag", action="store_true", help="Enable SWAG Bayesian inference")
```

After loading ODIN config (before building model), inject overrides:
```python
if args.swag:
    cfg.merge_from_list([
        "MODEL.BAYESIAN_TYPE", "swag",
        "MODEL.BAYESIAN_SAMPLES", "10",
        "MODEL.SWAG.SCALE", "1.0",
    ])
```

**No changes to ODINAdapter** — ODIN model itself handles SWAG internally when config is set. The per-sample uncertainty output in `instances_3d.pred_scores` will reflect MC uncertainty.

---

## 7. Novelty Reward (Voxel-Based)

### `config.py`
```python
REWARD_NOVELTY_FACTOR = 1.0        # weight for novelty reward component
REWARD_NOVELTY_VOXEL_SIZE = 0.02   # 2cm voxels
```

### `src/nbv/novelty.py` (new, ~60 lines)

```python
class VoxelNoveltyTracker:
    def __init__(self, voxel_size=0.02):
        self._seen: set[tuple] = set()
        self.voxel_size = voxel_size

    def reset(self): self._seen.clear()

    def update_and_score(self, pts_world: np.ndarray) -> float:
        """
        pts_world: (N,3) foreground 3D points from backprojection
        Returns fraction of new voxels (novelty reward ∈ [0,1])
        """
        if len(pts_world) == 0:
            return 0.0
        keys = set(map(tuple, (pts_world / self.voxel_size).astype(int)))
        new = keys - self._seen
        self._seen.update(keys)
        return len(new) / max(1, len(keys))
```

### Integration in `train_odin_sac_rl.py`

After `env.step()`, extract foreground points from `result["original_xyz"]` (already returned by `infer_for_rl`), call `novelty_tracker.update_and_score(foreground_pts)`, multiply by `REWARD_NOVELTY_FACTOR`:

```python
novelty_pts = result["original_xyz"]   # np.ndarray (N,3), all 3D points
rew_novelty = novelty_tracker.update_and_score(novelty_pts) * REWARD_NOVELTY_FACTOR
reward = _env_rew + rew_coverage + rew_expl + rew_novelty
```

Reset `novelty_tracker` at each `env.reset()`.

---

## 8. Point Cloud HTML Visualization

### Strategy: maximum reuse of ODIN's `generate_isaac_viewer.py`

The key functions to reuse:
- `unproject_frame_isaac(depth, rgb, mask, fx, fy, cx, cy, R, t, ...)` → `(N,8)` points
- `build_html(pts, cam_list, color_map, sample_name)` → HTML string

Both are pure Python/NumPy — no Isaac/ODIN dependencies. Import them directly.

### `src/nbv/visualization.py` (new, ~120 lines)

```python
import sys
sys.path.insert(0, str(ODIN_LOCAL_PATH))  # C:/.../my_ODIN_for_strawberry/odin
from generate_isaac_viewer import unproject_frame_isaac, build_html, quat_to_rotmat
```

Class `EpisodeVisualizer`:
```python
class EpisodeVisualizer:
    def __init__(self, output_path: Path, max_points=400_000, stride=4):
        self._frames: list[dict] = []   # cam data for frustums
        self._chunks: list[np.ndarray] = []  # (N,8) point arrays
        self.output_path = output_path
        self.max_points = max_points
        self.stride = stride

    def add_frame(self, rgb, depth, seg_mask, cam_pos, cam_quat_xyzw,
                  fx=128., fy=128., cx=128., cy=128.):
        """
        seg_mask: (H,W) int — class id per pixel, -1=background
        cam_quat_xyzw: PyBullet (x,y,z,w) convention
        """
        # Convert xyzw → rotation matrix, fix OpenGL -Z convention
        qx, qy, qz, qw = cam_quat_xyzw
        R = quat_to_rotmat(qx, qy, qz, qw)
        R[:, 2] = -R[:, 2]   # OpenGL → OpenCV (same fix as generate_isaac_viewer.py:186)
        t = np.array(cam_pos, dtype=np.float64)

        # Build colour_to_info from seg_mask (class ids → RGB hack)
        color_to_info = _build_color_to_info(seg_mask)

        # Encode seg_mask as RGB mask image for unproject_frame_isaac
        mask_rgb = _seg_to_rgb_mask(seg_mask)

        chunk = unproject_frame_isaac(depth, rgb.astype(np.float32), mask_rgb,
                                       fx, fy, cx, cy, R, t, color_to_info,
                                       stride=self.stride, max_depth=5.0, negate_y=True)
        self._chunks.append(chunk)

        # Store camera for frustum visualization
        self._frames.append({
            "frame_index": len(self._frames),
            "position": cam_pos.tolist(),
            "rotation": [qx, qy, qz, qw],
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        })

    def save(self, episode_id: int):
        pts = np.concatenate(self._chunks) if self._chunks else np.zeros((0,8))
        if len(pts) > self.max_points:
            idx = np.random.choice(len(pts), self.max_points, replace=False)
            pts = pts[idx]
        color_map = _build_color_map()   # sphere/cube/pyramid/cylinder/hourglass
        html = build_html(pts, self._frames, color_map, f"Episode {episode_id}")
        self.output_path.write_text(html, encoding="utf-8")
```

Helper `_build_color_map()` returns the 5-shape list in the format `build_html` expects:
```python
[{"instance_id": i, "category_id": i, "category_name": name, "color": [r,g,b]}
 for i, name in enumerate(SHAPE_NAMES)]
```

Helper `_seg_to_rgb_mask(seg)`: encode class id into (H,W,3) where R=inst_id=cat_id, G=0, B=cat_id.

### Integration in `train_odin_sac_rl.py`

Add CLI argument:
```python
parser.add_argument("--vis_episodes", type=int, nargs="*", default=[],
                    help="Episode indices to save HTML visualization")
```

At step inside episode loop (if `ep_idx in args.vis_episodes`):
```python
viz.add_frame(rgb=obs_rgb, depth=obs_depth, seg_mask=seg_mask,
              cam_pos=current_pos, cam_quat_xyzw=current_quat)
```

On episode end:
```python
if ep_idx in args.vis_episodes:
    out = Path(args.output_dir) / "vis" / f"episode_{ep_idx}.html"
    viz.save(ep_idx)
```

---

## 9. Wiring Everything Together in `train_odin_sac_rl.py`

Summary of all changes to the training script:

### New imports
```python
from src.nbv.metrics import EpisodeMetricsTracker, aggregate_metrics
from src.nbv.geometry_memory import GeometryMemory
from src.nbv.novelty import VoxelNoveltyTracker
from src.nbv.sampler import HemisphereSampler
from src.nbv.policies.info_gain import GeometricNBVPolicy
from src.nbv.policies.uncertainty import UncertaintyPolicy
from src.nbv.visualization import EpisodeVisualizer
```

### New CLI args (summary)
```
--policy {sac,geometric,uncertainty}   default: sac
--swag                                  enable SWAG
--vis_episodes [N ...]                  which episodes to visualize
```

### Per-episode initialization
```python
metrics_tracker = EpisodeMetricsTracker(args.policy, seed, NUM_CLASSES, MAX_STEPS)
metrics_tracker.set_scene(gt_classes)
novelty_tracker = VoxelNoveltyTracker(REWARD_NOVELTY_VOXEL_SIZE)
geom_memory = GeometryMemory(fx=128, fy=128, cx=128, cy=128)
if ep_idx in args.vis_episodes:
    viz = EpisodeVisualizer(vis_path)
```

### Per-step updates
```python
# After ODIN inference:
geom_memory.update(seg_mask, depth, cam_pos, cam_quat_xyzw, uncertainty_map)

# After env.step():
metrics_tracker.update(
    reward=reward, pred_classes=pred_classes, pred_scores=pred_scores,
    detected_ids=detected_ids, p_hidden=p_hidden, uncertainty=unc,
    is_collision=is_collision, is_oob=is_oob,
)

# Novelty reward:
rew_novelty = novelty_tracker.update_and_score(original_xyz) * REWARD_NOVELTY_FACTOR
reward += rew_novelty

# Step CSV:
write_step_csv_row(step, episode, reward, ...)
```

### Per-episode finalization
```python
ep_metrics = metrics_tracker.finalize(success=success)
write_episode_csv_row(ep_metrics)
if args.policy != "sac":  # baselines: skip SAC updates
    continue
```

---

## File-by-File Change Summary

| File | Status | Source / Notes |
|---|---|---|
| `config.py` | Modify | Add UR3 workspace, shapes, novelty params |
| `src/data/robot/ur3.urdf` | Create | Fresh from UR3 DH params |
| `src/simulation/robot.py` | Modify | 6-joint UR3 support |
| `src/simulation/asset_loader.py` | Modify | Only 5 active shapes |
| `src/simulation/environment_odin.py` | Modify | VECTOR_DIM=17, UR3 workspace |
| `train_odin_sac_rl.py` | Modify | Policy dispatch, metrics, novelty, SWAG, CSV |
| `src/nbv/__init__.py` | Created (empty) | Already done |
| `src/nbv/metrics.py` | Created | Already done (strawberry port) |
| `src/nbv/geometry_memory.py` | Created | Already done (strawberry port) |
| `src/nbv/sampler.py` | Create | Copy + adapt from strawberry |
| `src/nbv/policies/__init__.py` | Created (empty) | Already done |
| `src/nbv/policies/info_gain.py` | Create | Copy + adapt from strawberry |
| `src/nbv/policies/uncertainty.py` | Create | ~30 lines fresh |
| `src/nbv/novelty.py` | Create | ~60 lines fresh |
| `src/nbv/visualization.py` | Create | Thin wrapper around ODIN's `generate_isaac_viewer.py` |

---

## Dependency Order (Implementation Sequence)

1. `config.py` — all new constants (everything else depends on this)
2. `src/data/robot/ur3.urdf` — needed before robot.py changes
3. `src/simulation/robot.py` — 6-DoF UR3
4. `src/simulation/asset_loader.py` — 5 shapes
5. `src/simulation/environment_odin.py` — VECTOR_DIM=17
6. `src/nbv/sampler.py` — copied from strawberry
7. `src/nbv/policies/info_gain.py` — depends on sampler + geometry_memory
8. `src/nbv/policies/uncertainty.py` — standalone
9. `src/nbv/novelty.py` — standalone
10. `src/nbv/visualization.py` — wraps ODIN code
11. `train_odin_sac_rl.py` — wires everything together (update OBS_DIM, action bounds, policy dispatch, metrics integration, SWAG, novelty, visualization)

---

## Key Design Decisions

**UR3 obs_dim**: dropping from 274→273 because joints 7→6. All hardcoded `18`, `274`, `obs[8:15]` slices in train script must be updated.

**Visualization reuse**: `generate_isaac_viewer.py` is Python with no Isaac dependency — we `sys.path.insert` and import its functions directly. Zero reimplementation of the Three.js HTML generation.

**Baselines vs SAC**: single training script, `--policy` flag. When policy is `geometric` or `uncertainty`: ODIN inference still runs (for metrics and state), but SAC actor/critic updates are skipped. Replay buffer is still populated for potential analysis but `agent.update_*` are not called.

**SWAG**: zero code change to ODINAdapter. Injecting ODIN config overrides before model build is sufficient — ODIN handles MC sampling internally.

**Novelty reward**: `original_xyz` is already returned by `infer_for_rl()` — no extra backprojection needed.
