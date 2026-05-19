"""
demo_video.py — Демонстрация записи видео и работы среды NBVODINEnv.
Использует моки для ODIN и имитацию RL-агента.
ПОДДЕРЖИВАЕТ КУМУЛЯТИВНОЕ ЛОГИРОВАНИЕ.
"""
import sys
import os
import types
import numpy as np
import ctypes
import csv
import cv2
import imageio
from pathlib import Path

# 1. МОКИ
def _mock_heavy():
    m = types.ModuleType("pybullet")
    m.DIRECT = 0; m.GUI = 1; m.connect = lambda x: 0; m.disconnect = lambda: None
    m.isConnected = lambda **kw: True; m.resetSimulation = lambda **kw: None
    m.ER_TINY_RENDERER = 1
    m.setGravity = lambda *a, **kw: None; m.setAdditionalSearchPath = lambda *a, **kw: None
    m.loadURDF = lambda *a, **kw: 1; m.changeVisualShape = lambda *a, **kw: None
    m.changeDynamics = lambda *a, **kw: None; m.stepSimulation = lambda **kw: None
    m.resetBasePositionAndOrientation = lambda *a, **kw: None
    m.getBasePositionAndOrientation = lambda *a, **kw: ([0.5, 0, 0.2], [0,0,0,1])
    m.getClosestPoints = lambda **kw: []
    m.getMatrixFromQuaternion = lambda *a: [1,0,0, 0,1,0, 0,0,1]
    m.getQuaternionFromEuler = lambda euler, *a: [euler[0], euler[1], euler[2], 1.0]
    m.getEulerFromQuaternion = lambda quat, *a: [quat[0], quat[1], quat[2]]
    m.computeViewMatrix = lambda *a, **kw: np.eye(4).flatten()
    m.computeProjectionMatrixFOV = lambda *a, **kw: np.eye(4).flatten()
    def mock_get_camera_image(*a, **kw):
        img = np.zeros((224, 224, 4), dtype=np.uint8)
        img[:, :] = [30, 45, 55, 255] # темно-серый фон сцены
        cv2.circle(img, (112, 112), 35, (0, 255, 100, 255), -1) # зеленый объект
        cv2.rectangle(img, (40, 40), (80, 80), (120, 100, 80, 255), -1) # коричневое препятствие 1
        cv2.rectangle(img, (140, 140), (180, 180), (120, 100, 80, 255), -1) # коричневое препятствие 2
        return 224, 224, img, np.ones((224, 224)), np.zeros((224, 224))
    m.getCameraImage = mock_get_camera_image
    m.error = Exception
    sys.modules["pybullet"] = m
    sys.modules["pybullet_data"] = types.ModuleType("pybullet_data")
    sys.modules["pybullet_data"].getDataPath = lambda: ""
    sys.modules["detectron2"] = types.ModuleType("detectron2")
    sys.modules["detectron2.structures"] = types.ModuleType("detectron2.structures")
    sys.modules["detectron2.structures"].Instances = lambda x: None
    sys.modules["odin"] = types.ModuleType("odin")

_mock_heavy()

# --- Fake Components ---
class FakeCamera:
    def __init__(self, *a, **kw): pass
    def get_image(self, **kw):
        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb[:, :] = [45, 45, 45] # темно-серый фон
        cv2.circle(rgb, (112, 112), 50, (0, 0, 255), -1) # красный круг
        return rgb, np.ones((224, 224)), np.zeros((224, 224))

class FakeAssetLoader:
    def __init__(self, *a, **kw):
        self.obstacles = []
        self.target_objects = [1]
        self.target_objects_classes = [0]
    def load_robot(self): return 1
    def generate_scene(self): return [1]

class FakeRobot:
    def __init__(self, *a, **kw): self.robot_id = 99
    def reset(self): pass
    def get_ee_pose(self): return np.array([0.5, 0, 0.4]), np.array([0,0,0,1])
    def get_joint_states(self): return np.zeros(7)
    def apply_action(self, *a, **kw): pass

import src.simulation.asset_loader
import src.simulation.camera
import src.simulation.robot
src.simulation.asset_loader.AssetLoader = FakeAssetLoader
src.simulation.camera.Camera = FakeCamera
src.simulation.robot.Robot = FakeRobot

import config
from src.simulation.environment_odin import NBVODINEnv
import src.simulation.environment_odin
print(f"[DEBUG] NBVODINEnv imported from: {src.simulation.environment_odin.__file__}")

class FakeAdapter:
    def __init__(self, episode_idx=0):
        # Начальное p_hidden уменьшается с ростом номера эпизода (имитация обучения)
        self.initial_p_hidden = max(0.4, 0.95 - 0.05 * episode_idx)
        self.last_p_hidden = self.initial_p_hidden
        self.last_nbv_pos = np.array([0.5, 0.1, 0.5])
        self.episode_idx = episode_idx
    def reset(self):
        self.last_p_hidden = self.initial_p_hidden
    def add_frame(self, *a, **kw): pass
    def infer(self, **kw):
        prev = self.last_p_hidden
        # Имитируем уменьшение неопределенности с небольшим случайным шумом
        decay = np.random.uniform(0.90, 0.96)
        self.last_p_hidden *= decay
        # Имитируем уверенность классификатора (растет с эпизодами)
        conf = min(0.99, 0.3 + 0.06 * self.episode_idx + np.random.uniform(0.0, 0.05))
        return {"delta_p_hidden": prev - self.last_p_hidden, 
                "p_hidden": self.last_p_hidden, 
                "nbv_pos": self.last_nbv_pos,
                "instances_3d": {
                    "pred_scores": [conf],
                    "pred_masks": [np.ones((224, 224), dtype=bool)],
                    "pred_classes": [0]
                },
                "original_xyz": np.zeros((100, 3))}

def run_demo(episode_idx):
    print(f"\n[DEMO] Initializing environment...")
    adapter = FakeAdapter(episode_idx)
    env = NBVODINEnv(render_mode="rgb_array", headless=True, no_arm=True, odin_adapter=adapter, log_dir="demo_output")
    
    env.no_arm_pos = [1.0, 0.0, 0.5]
    env.no_arm_orn = [0, 0, 0, 1]
    
    video_path = Path("demo_output/videos")
    video_path.mkdir(parents=True, exist_ok=True)
    
    cur_ep = env.episode_count
    print(f"[DEMO] Starting Simulation for Episode {cur_ep} (max {config.MAX_STEPS_PER_EPISODE} steps)...")
    
    frames = []
    obs, _ = env.reset()
    
    # Обновляем кэш графика на случай, если есть ранее загруженная история
    env._update_plot_cache()
    
    for i in range(5):
        angle = i * 0.3
        # Имитируем динамические повороты камеры (Roll, Pitch, Yaw в радианах)
        roll = np.sin(i * 0.5) * 0.5
        pitch = np.cos(i * 0.5) * 0.4
        yaw = i * 0.25
        action = np.array([0.5 + 0.2*np.cos(angle), 0.2*np.sin(angle), 0.3, roll, pitch, yaw])
        obs, reward, done, trunc, info = env.step(action)
        
        # Записываем каждый шаг (от 1 до 5)
        frame = env.render()
        frames.append(frame)
        
        if done or trunc:
            break

    # Сохраняем гифку на диск только для 1-го (index 0), 5-го (index 4) и 10-го (index 9) эпизодов
    if cur_ep in [0, 4, 9]:
        print(f"[DEMO] Saving video: episode_{cur_ep}.gif")
        imageio.mimsave(f"demo_output/videos/episode_{cur_ep}.gif", frames, fps=2)
    
    env._save_episode_metrics()
    print(f"[DEMO] Run complete. CSV size: {os.path.getsize(env.metrics_path)} bytes")
    try:
        env.close()
    except:
        pass

if __name__ == "__main__":
    # Удалим старый лог для чистоты демо если он есть
    p = Path("demo_output/training_metrics.csv")
    if p.exists(): p.unlink()
    
    # Запустим 10 эпизодов, чтобы увидеть динамику и работу скользящего среднего
    for i in range(10):
        run_demo(i)
