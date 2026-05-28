"""
tests/test_odin_adapter.py

Тесты ODINAdapter и NBVODINEnv с моками тяжёлых зависимостей.
Запуск: python -m pytest tests/test_odin_adapter.py -v
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────
#  МОКАЕМ ВСЕ ТЯЖЁЛЫЕ ЗАВИСИМОСТИ ДО ЛЮБЫХ ИМПОРТОВ
# ─────────────────────────────────────────────────────────────

def _make_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# pybullet
pb = _make_module("pybullet")
pb.DIRECT = 0
pb.GUI = 1
pb.KEY_IS_DOWN = 1
pb.KEY_WAS_TRIGGERED = 2
pb.B3G_UP_ARROW = 65297
pb.B3G_DOWN_ARROW = 65298
pb.B3G_LEFT_ARROW = 65295
pb.B3G_RIGHT_ARROW = 65296
pb.connect = MagicMock(return_value=0)
pb.disconnect = MagicMock()
pb.isConnected = MagicMock(return_value=True)
pb.resetSimulation = MagicMock()
pb.setGravity = MagicMock()
pb.setAdditionalSearchPath = MagicMock()
pb.loadURDF = MagicMock(return_value=1)
pb.changeVisualShape = MagicMock()
pb.changeDynamics = MagicMock()
pb.stepSimulation = MagicMock()
pb.resetBasePositionAndOrientation = MagicMock()
pb.getBasePositionAndOrientation = MagicMock(
    return_value=([0.5, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0])
)
pb.getClosestPoints = MagicMock(return_value=[])
pb.getMatrixFromQuaternion = MagicMock(
    return_value=[1,0,0, 0,1,0, 0,0,1]
)
pb.getQuaternionFromEuler = MagicMock(return_value=[0,0,0,1])
pb.error = Exception

_make_module("pybullet_data").getDataPath = MagicMock(return_value="/tmp")

# detectron2 stub
d2 = _make_module("detectron2")
d2_struct = _make_module("detectron2.structures")
class _Instances:
    def __init__(self, shape): self.image_size = shape
    def __setattr__(self, k, v): object.__setattr__(self, k, v)
    def __getattr__(self, k): return None
d2_struct.Instances = _Instances
_make_module("detectron2.modeling")
_make_module("detectron2.config")
_make_module("detectron2.projects")
_make_module("detectron2.projects.deeplab")

# odin stubs
_make_module("odin")
_make_module("odin.modeling")
_make_module("odin.modeling.backproject")
odin_bp = _make_module("odin.modeling.backproject.backproject")
odin_bp.backprojector_dataloader = MagicMock(return_value=([None]*4, None, [None]))
odin_bp.multiscsale_voxelize = MagicMock(return_value=[None]*4)

# opencv stub
cv2 = _make_module("cv2")
cv2.resize = MagicMock(side_effect=lambda x, size, **kw: np.zeros((*size[::-1], 3), dtype=np.uint8))
cv2.cvtColor = MagicMock(side_effect=lambda x, *a: x)
cv2.imshow = MagicMock()
cv2.waitKey = MagicMock()
cv2.FONT_HERSHEY_SIMPLEX = 0
cv2.putText = MagicMock()
cv2.COLOR_RGB2BGR = 4
cv2.COLOR_RGB2BGRA = 3

# matplotlib stub
mpl = _make_module("matplotlib")
mpl_plt = _make_module("matplotlib.pyplot")
mpl_plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))
mpl_plt.close = MagicMock()
mpl.use = MagicMock()

# src stubs
_make_module("src")
_make_module("src.simulation")
_make_module("src.vision")

# config stub
cfg_mod = _make_module("config")
cfg_mod.ACTION_MIN = [0.2, -0.5, 0.0, -3.14, -3.14, -3.14]
cfg_mod.ACTION_MAX = [0.8, 0.5, 0.8, 3.14, 3.14, 3.14]
cfg_mod.IMAGE_SIZE = 64
cfg_mod.IMAGE_CHANNELS = 4
cfg_mod.NUM_CLASSES = 8
cfg_mod.REWARD_SCALE = 10.0
cfg_mod.PENALTY_COLLISION = -15.0
cfg_mod.PENALTY_OOB = -10.0
cfg_mod.MAX_STEPS_PER_EPISODE = 10
cfg_mod.REWARD_CLASSIFIER_SCALE = 30.0
cfg_mod.REWARD_ALL_FOUND_BONUS = 50.0
cfg_mod.REWARD_SUCCESS_CLASSIFIED_BONUS = 100.0
cfg_mod.REWARD_SURVIVAL = 0.1
cfg_mod.PENALTY_STEP = -0.05
cfg_mod.PLOT_MOVING_AVERAGE_WINDOW = 5
cfg_mod.SCENE_STAGE = 2
cfg_mod.get_short_path = lambda x: str(x)

# AssetLoader / Robot / Camera stubs
class _FakeCamera:
    def __init__(self, *a, **kw): pass
    def get_image(self, **kw):
        H = W = cfg_mod.IMAGE_SIZE
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        seg = np.zeros((H, W), dtype=np.uint8)
        return rgb, depth, seg

class _FakeRobot:
    def __init__(self, *a, **kw): self.robot_id = 99
    def reset(self): pass
    def get_ee_pose(self):
        return np.array([0.5, 0.0, 0.4], dtype=np.float32), np.array([0,0,0,1], dtype=np.float32)
    def get_joint_states(self): return np.zeros(7, dtype=np.float32)
    def apply_action(self, *a, **kw): pass

class _FakeAssetLoader:
    def __init__(self, *a, **kw):
        self.obstacles = []
        self.target_objects = [1]
        self.target_objects_classes = [0]
    def load_robot(self): return 0
    def generate_scene(self): return [1]

sim_mod = sys.modules["src.simulation"]
sim_mod.asset_loader = types.ModuleType("src.simulation.asset_loader")
sim_mod.asset_loader.AssetLoader = _FakeAssetLoader
sys.modules["src.simulation.asset_loader"] = sim_mod.asset_loader

sim_mod.camera = types.ModuleType("src.simulation.camera")
sim_mod.camera.Camera = _FakeCamera
sys.modules["src.simulation.camera"] = sim_mod.camera

sim_mod.robot = types.ModuleType("src.simulation.robot")
sim_mod.robot.Robot = _FakeRobot
sys.modules["src.simulation.robot"] = sim_mod.robot

sim_mod.environment_odin = types.ModuleType("src.simulation.environment_odin")
sys.modules["src.simulation.environment_odin"] = sim_mod.environment_odin

vision_mod = sys.modules["src.vision"]
vision_mod.odin_adapter = types.ModuleType("src.vision.odin_adapter")
sys.modules["src.vision.odin_adapter"] = vision_mod.odin_adapter

# utils stub
utils_mod = _make_module("src.utils")
math_mod = _make_module("src.utils.math_utils")
math_mod.normalize_quaternion = lambda q: q
sys.modules["src.utils"] = utils_mod
sys.modules["src.utils.math_utils"] = math_mod

# ─────────────────────────────────────────────────────────────
#  ТЕПЕРЬ ИМПОРТИРУЕМ НАСТОЯЩИЙ КОД
# ─────────────────────────────────────────────────────────────
import importlib.util, pathlib

ROOT = pathlib.Path(__file__).parent.parent

def _load(rel_path):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(rel_path, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

adapter_mod = _load("src/vision/odin_adapter.py")
ODINAdapter = adapter_mod.ODINAdapter
_build_intrinsics = adapter_mod._build_intrinsics
_depth_to_xyz = adapter_mod._depth_to_xyz
_apply_pose = adapter_mod._apply_pose

env_mod = _load("src/simulation/environment_odin.py")
NBVODINEnv = env_mod.NBVODINEnv


# ─────────────────────────────────────────────────────────────
#  ФЕЙКОВАЯ МОДЕЛЬ (заменяет NBVActiveODIN)
# ─────────────────────────────────────────────────────────────
class FakeODINModel(torch.nn.Module):
    """Минималистичный заменитель NBVActiveODIN."""
    def __init__(self, p_hidden_val=0.6):
        super().__init__()
        self.p_val = p_hidden_val

    def forward(self, batched_inputs):
        return {
            "p_hidden": torch.tensor([[self.p_val]]),
            "nbv_pos":  torch.tensor([[0.3, 0.1, 0.5]]),
            "nbv_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        }

    def eval(self): return self


# ─────────────────────────────────────────────────────────────
#  UNIT ТЕСТЫ
# ─────────────────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_build_intrinsics_shape(self):
        K = _build_intrinsics(200, 200, 100, 100)
        self.assertEqual(K.shape, (3, 3))
        self.assertAlmostEqual(K[0, 0], 200.0)
        self.assertAlmostEqual(K[1, 2], 100.0)
        self.assertAlmostEqual(K[2, 2], 1.0)

    def test_depth_to_xyz_shape(self):
        depth = np.ones((32, 32), dtype=np.float32)
        K = _build_intrinsics(100, 100, 16, 16)
        xyz = _depth_to_xyz(depth, K)
        self.assertEqual(xyz.shape, (32, 32, 3))

    def test_depth_to_xyz_center_zero_xy(self):
        """Центральный пиксель должен иметь X=0, Y=0."""
        depth = np.ones((32, 32), dtype=np.float32)
        K = _build_intrinsics(100, 100, 16, 16)
        xyz = _depth_to_xyz(depth, K)
        self.assertAlmostEqual(xyz[16, 16, 0], 0.0, places=4)
        self.assertAlmostEqual(xyz[16, 16, 1], 0.0, places=4)

    def test_apply_pose_identity(self):
        xyz = np.random.randn(8, 8, 3).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        result = _apply_pose(xyz, pose)
        np.testing.assert_allclose(result, xyz, atol=1e-5)


class TestODINAdapter(unittest.TestCase):

    def _make_adapter(self, p_val=0.6):
        device = torch.device("cpu")
        model = FakeODINModel(p_hidden_val=p_val)
        return ODINAdapter(model=model, device=device, image_size=64, num_frames=3)

    def _dummy_frame(self):
        H = W = 64
        rgb   = np.zeros((H, W, 3), dtype=np.uint8)
        depth = np.ones((H, W), dtype=np.float32)
        pose  = np.eye(4, dtype=np.float32)
        return rgb, depth, pose

    # --- reset ---
    def test_reset_clears_buffer(self):
        a = self._make_adapter()
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        self.assertEqual(len(a._frame_buffer), 1)
        a.reset()
        self.assertEqual(len(a._frame_buffer), 0)
        self.assertEqual(a.last_p_hidden, 1.0)
        self.assertIsNone(a.last_nbv_pos)

    # --- add_frame ---
    def test_add_frame_appends(self):
        a = self._make_adapter()
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        self.assertEqual(len(a._frame_buffer), 1)

    def test_add_frame_max_window(self):
        a = self._make_adapter()
        rgb, depth, pose = self._dummy_frame()
        for _ in range(6):
            a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        self.assertLessEqual(len(a._frame_buffer), a.num_frames)

    def test_frame_image_tensor_shape(self):
        a = self._make_adapter()
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        img_t = a._frame_buffer[0]["image"]
        self.assertEqual(img_t.shape, (3, 64, 64))

    def test_pose_z_flipped(self):
        """OpenGL→ODIN: третий столбец матрицы должен быть перевёрнут."""
        a = self._make_adapter()
        pose = np.eye(4, dtype=np.float32)
        rgb, depth, _ = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        pose_t = a._frame_buffer[0]["pose"].numpy()
        np.testing.assert_allclose(pose_t[:3, 2], -np.eye(4)[:3, 2], atol=1e-5)

    # --- infer: empty buffer ---
    def test_infer_empty_buffer_returns_defaults(self):
        a = self._make_adapter()
        result = a.infer(np.zeros(3), np.array([0, 0, 0, 1.0]))
        self.assertEqual(result["delta_p_hidden"], 0.0)
        self.assertEqual(result["p_hidden"], 1.0)
        self.assertIsNone(result["nbv_pos"])

    # --- infer: full pass ---
    def test_infer_returns_p_hidden(self):
        a = self._make_adapter(p_val=0.6)
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        result = a.infer(np.zeros(3), np.array([0, 0, 0, 1.0]))
        self.assertIn("p_hidden", result)
        ph = result["p_hidden"]
        self.assertGreater(ph, 0.0)
        self.assertLessEqual(ph, 1.0)

    def test_infer_delta_p_hidden(self):
        """delta = prev - cur. После первого infer delta должна быть 1.0 - p_val."""
        a = self._make_adapter(p_val=0.4)
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        result = a.infer(np.zeros(3), np.array([0, 0, 0, 1.0]))
        expected_p = torch.sigmoid(torch.tensor(0.4)).item()
        expected_delta = 1.0 - expected_p
        self.assertAlmostEqual(result["delta_p_hidden"], expected_delta, places=4)

    def test_infer_nbv_pos_shape(self):
        a = self._make_adapter()
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        result = a.infer(np.zeros(3), np.array([0, 0, 0, 1.0]))
        self.assertIsNotNone(result["nbv_pos"])
        self.assertEqual(result["nbv_pos"].shape, (3,))

    def test_infer_updates_last_p_hidden(self):
        a = self._make_adapter(p_val=0.0)   # sigmoid(0) = 0.5
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        a.infer(np.zeros(3), np.array([0, 0, 0, 1.0]))
        self.assertNotEqual(a.last_p_hidden, 1.0)

    def test_infer_model_exception_returns_empty(self):
        """Если модель бросает исключение — возвращаем last values."""
        class BadModel(torch.nn.Module):
            def forward(self, *a, **kw): raise RuntimeError("boom")
            def eval(self): return self
        a = ODINAdapter(BadModel(), torch.device("cpu"), image_size=64)
        rgb, depth, pose = self._dummy_frame()
        a.add_frame(rgb, depth, pose, 100, 100, 32, 32)
        result = a.infer(np.zeros(3), np.array([0, 0, 0, 1.0]))
        self.assertEqual(result["p_hidden"], 1.0)


class TestNBVODINEnvNoAdapter(unittest.TestCase):
    """Тесты среды БЕЗ ODIN адаптера (delta_p_hidden = 0)."""

    def setUp(self):
        self.env = NBVODINEnv(headless=True, no_arm=True, odin_adapter=None)

    def tearDown(self):
        self.env.close()

    def test_observation_space_dims(self):
        self.assertEqual(self.env.VECTOR_DIM, 14)
        img_shape = self.env.observation_space["image"].shape
        self.assertEqual(img_shape, (4, cfg_mod.IMAGE_SIZE, cfg_mod.IMAGE_SIZE))
        vec_shape = self.env.observation_space["vector"].shape
        self.assertEqual(vec_shape, (14,))

    def test_action_space_shape(self):
        self.assertEqual(self.env.action_space.shape, (6,))

    def test_reset_returns_obs(self):
        obs, info = self.env.reset()
        self.assertIn("image", obs)
        self.assertIn("vector", obs)
        self.assertEqual(obs["image"].shape, (4, cfg_mod.IMAGE_SIZE, cfg_mod.IMAGE_SIZE))
        self.assertEqual(obs["vector"].shape, (14,))
        self.assertIsInstance(info, dict)

    def test_reset_step_count_zero(self):
        self.env.reset()
        self.assertEqual(self.env.step_count, 0)

    def test_step_increments_count(self):
        self.env.reset()
        action = self.env.action_space.sample()
        self.env.step(action)
        self.assertEqual(self.env.step_count, 1)

    def test_step_returns_correct_structure(self):
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIn("image", obs)
        self.assertIn("vector", obs)
        # reward may be np.float32 or float — both are numeric
        self.assertTrue(isinstance(reward, (float, np.floating)),
                        f"Expected float-like, got {type(reward)}")
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIn("delta_p_hidden", info)
        self.assertIn("p_hidden", info)

    def test_no_adapter_delta_zero(self):
        """Без адаптера delta_p_hidden всегда 0."""
        self.env.reset()
        action = self.env.action_space.sample()
        _, _, _, _, info = self.env.step(action)
        self.assertEqual(info["delta_p_hidden"], 0.0)

    def test_truncated_after_max_steps(self):
        self.env.reset()
        action = self.env.action_space.sample()
        truncated = False
        for _ in range(cfg_mod.MAX_STEPS_PER_EPISODE):
            _, _, _, truncated, _ = self.env.step(action)
        self.assertTrue(truncated)

    def test_oob_penalty(self):
        """Выход за границы (z < 0.05) → штраф PENALTY_OOB."""
        self.env.reset()
        # Мокаем getBasePositionAndOrientation возвращающий z=0.0 (ниже порога 0.05)
        import pybullet as _pb
        _pb.getBasePositionAndOrientation = MagicMock(
            return_value=([0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        )
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, _, _, _ = self.env.step(action)
        self.assertEqual(float(reward), cfg_mod.PENALTY_OOB)
        # Восстанавливаем мок
        _pb.getBasePositionAndOrientation = MagicMock(
            return_value=([0.5, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0])
        )

    def test_vector_dtype(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs["vector"].dtype, np.float32)

    def test_image_range(self):
        obs, _ = self.env.reset()
        self.assertGreaterEqual(obs["image"].min(), 0.0)
        self.assertLessEqual(obs["image"].max(), 1.01)


class TestNBVODINEnvWithAdapter(unittest.TestCase):
    """Тесты среды С мок-адаптером."""

    def _make_mock_adapter(self, p_val=0.6):
        device = torch.device("cpu")
        model = FakeODINModel(p_hidden_val=p_val)
        return ODINAdapter(model=model, device=device, image_size=64, num_frames=3)

    def setUp(self):
        self.adapter = self._make_mock_adapter(p_val=0.7)
        self.env = NBVODINEnv(headless=True, no_arm=True, odin_adapter=self.adapter)

    def tearDown(self):
        self.env.close()

    def test_reset_calls_adapter_reset(self):
        self.env.reset()
        self.assertEqual(len(self.adapter._frame_buffer), 1)  # 1 кадр после _get_obs в reset
        self.env.reset()
        # После второго reset буфер снова сбрасывается и пополняется 1 кадром
        self.assertLessEqual(len(self.adapter._frame_buffer), 2)



    def test_reward_positive_when_p_hidden_decreases(self):
        """Если p_hidden снижается → delta > 0 → reward > 0.
        
        1. reset() вызывает первый инференс (p=0.7).
        2. Меняем p модели на 0.4.
        3. step() вызывает второй инференс -> delta = sigmoid(0.7) - sigmoid(0.4) > 0.
        """
        self.env.reset()
        
        # Уменьшаем значение в модели
        self.env.odin_adapter.model.p_val = 0.4
        
        action = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, _, _, info = self.env.step(action)
        
        delta = float(obs["vector"][7])
        self.assertGreater(delta, 0.0, f"delta_p_hidden ({delta}) должна быть > 0")
        self.assertGreater(float(reward), 0.0, f"reward ({reward}) должен быть > 0")

    def test_adapter_buffer_grows_per_step(self):
        self.env.reset()
        n_before = len(self.adapter._frame_buffer)
        action = self.env.action_space.sample()
        self.env.step(action)
        n_after = len(self.adapter._frame_buffer)
        self.assertGreaterEqual(n_after, n_before)

    def test_multiple_steps_p_hidden_tracked(self):
        self.env.reset()
        action = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(3):
            self.env.step(action)
        self.assertLess(self.env.last_p_hidden, 1.0)

    def test_collision_overrides_odin_reward(self):
        """Столкновение → PENALTY_COLLISION вне зависимости от p_hidden."""
        self.env.reset()
        fake_pts = [("dummy",)]  # непустой список = коллизия
        self.env.asset_loader.obstacles = [42]
        import pybullet as _pb
        _pb.getClosestPoints = MagicMock(return_value=fake_pts)
        action = self.env.action_space.sample()
        _, reward, _, _, _ = self.env.step(action)
        self.assertEqual(reward, cfg_mod.PENALTY_COLLISION)
        _pb.getClosestPoints = MagicMock(return_value=[])


if __name__ == "__main__":
    unittest.main(verbosity=2)
