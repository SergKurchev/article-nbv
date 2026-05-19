"""
environment_odin.py — NBVEnv, использующий NBVActiveODIN как vision model.

Ключевые отличия от environment.py:
    1. vision_model заменен на odin_adapter (ODINAdapter).
    2. Награда = delta_p_hidden (уменьшение вероятности скрытых объектов).
       Чем сильнее снизилась неопределённость — тем больше награда.
    3. Наблюдение расширено: в вектор добавлены nbv_hint (3) — подсказка
       от NBV Head ODIN о следующей позиции камеры.
    4. Все остальные механики (collision, OOB, robot IK) без изменений.

Использование на Kaggle:
    from src.simulation.environment_odin import NBVODINEnv
    env = NBVODINEnv(headless=True, odin_adapter=adapter)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import pybullet as p
import pybullet_data
import cv2
import config
from src.simulation.asset_loader import AssetLoader
from src.simulation.robot import Robot
from src.simulation.camera import Camera
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Импорт адаптера (опциональный — может быть None при старом режиме)
try:
    from src.vision.odin_adapter import ODINAdapter
    ODIN_AVAILABLE = True
except ImportError:
    ODIN_AVAILABLE = False
    ODINAdapter = None


class NBVODINEnv(gym.Env):
    """
    NBV-среда с NBVActiveODIN в качестве vision backbone.

    Parameters
    ----------
    render_mode : str, optional
    headless : bool
        True = p.DIRECT (для Kaggle), False = p.GUI.
    no_arm : bool
        Если True — камера летает свободно без манипулятора.
    odin_adapter : ODINAdapter, optional
        Предзагруженный адаптер ODIN. Если None — среда работает в режиме
        «случайной» награды (только для отладки).
    use_nbv_hint : bool
        Добавлять ли NBV-подсказку в вектор наблюдения.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Размер вектора наблюдения:
    # pos(3) + orn(4) + delta_p_hidden(1) + joints(7) + nbv_hint(3) = 18
    VECTOR_DIM = 18

    def __init__(
        self,
        render_mode=None,
        headless=True,
        no_arm=False,
        odin_adapter: "ODINAdapter | None" = None,
        use_nbv_hint: bool = True,
        log_dir: str = "eval_logs",
        external_infer: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.headless = headless
        self.no_arm = no_arm
        self.odin_adapter = odin_adapter
        self.use_nbv_hint = use_nbv_hint
        self.log_dir = Path(log_dir)
        self.external_infer = external_infer

        # Подключаем PyBullet
        connection_mode = p.DIRECT if headless else p.GUI
        self.client_id = p.connect(connection_mode)

        # ---------- Пространство действий ----------
        # [x, y, z, roll, pitch, yaw] — абсолютные координаты (как в оригинале)
        self.action_space = spaces.Box(
            low=np.array(config.ACTION_MIN, dtype=np.float32),
            high=np.array(config.ACTION_MAX, dtype=np.float32),
            dtype=np.float32,
        )

        # ---------- Пространство наблюдений ----------
        # image: RGB-D (4, 224, 224)
        # vector: pos(3) + orn(4) + delta_p_hidden(1) + joints(7) + nbv_hint(3)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4, config.IMAGE_SIZE, config.IMAGE_SIZE),
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.VECTOR_DIM,),
                    dtype=np.float32,
                ),
            }
        )

        # ---------- Компоненты симуляции ----------
        self.asset_loader = AssetLoader(self.client_id)
        self.camera = Camera(self.client_id)
        self.robot = None
        self.target_obj_id = None

        # ---------- Статус ----------
        self.step_count = 0
        self.last_p_hidden = 1.0    # В начале эпизода всё «скрыто»
        self.last_rgb = None
        self.fps_eye = np.array([1.5, 0.0, 0.5])

        # История для графиков и логов
        self.episode_history = []
        self.p_hidden_history = []
        self.reward_history = []
        self.total_reward = 0
        self.episode_count = 0

        # Статистика по объектам
        self.num_found_objects = 0
        self.num_hidden_objects = 0
        self.total_target_objects = 0
        self.current_episode_reward = 0.0
        self.current_episode_min_p = 1.0
        self.metrics_path = self.log_dir / "training_metrics.csv"
        self.summary_plot_path = self.log_dir / "training_summary.png"
        
        # История для графиков (загружаем если есть)
        self.episode_rewards_history = []
        self.episode_p_hidden_history = []
        self._load_history_from_csv()

        # Инициализируем пустой кэш графика
        self.cached_plot_bgr = np.zeros((300, 700, 3), dtype=np.uint8)

        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Сохраняем историю прошлого эпизода
        if self.step_count > 0:
            self.episode_rewards_history.append(self.current_episode_reward)
            self.episode_p_hidden_history.append(self.current_episode_min_p)

        self.step_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_min_p = 1.0
        self.last_p_hidden = 1.0
        self.episode_history = []
        self.cumulative_found_objects = set()

        # Сброс адаптера ODIN
        if self.odin_adapter is not None:
            self.odin_adapter.reset()

        # Переподключение при необходимости
        if not p.isConnected(physicsClientId=self.client_id):
            connection_mode = p.DIRECT if self.headless else p.GUI
            self.client_id = p.connect(connection_mode)
            self.asset_loader = AssetLoader(self.client_id)
            self.camera = Camera(self.client_id)

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(
            config.get_short_path(pybullet_data.getDataPath()),
            physicsClientId=self.client_id,
        )

        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0, 0, 0, 1], physicsClientId=self.client_id)

        if not self.no_arm:
            robot_urdf = self.asset_loader.load_robot()
            self.robot = Robot(self.client_id, robot_urdf)
            self.robot.reset()

        target_objects = self.asset_loader.generate_scene()
        self.target_obj_id = target_objects[0]
        self.total_target_objects = len(self.asset_loader.target_objects)

        if self.no_arm:
            p.changeDynamics(self.target_obj_id, -1, mass=0)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_obs()

        if not self.headless:
            self._update_plot_cache()

        return obs, {}

    def step(self, action):
        # Применяем действие
        if not self.no_arm:
            target_pos = action[:3]
            target_euler = action[3:]
            target_orn = p.getQuaternionFromEuler(target_euler)
            self.robot.apply_action(target_pos, target_orn)
        else:
            # В no_arm режиме action это позиция и ориентация камеры
            target_pos = action[:3]
            target_euler = action[3:]
            target_orn = p.getQuaternionFromEuler(target_euler)
            # Если нет робота, мы просто двигаем камеру (в данной реализации это имитация перемещения)
            # Но для корректного наблюдения в _get_obs, сохраним состояние
            self.no_arm_pos = target_pos
            self.no_arm_orn = target_orn

        if not p.isConnected(physicsClientId=self.client_id):
            dummy_obs = self._get_dummy_obs()
            return dummy_obs, 0.0, True, True, {}

        p.stepSimulation(physicsClientId=self.client_id)
        self.step_count += 1

        # Проверка столкновений
        moving_body_id = self.target_obj_id if self.no_arm else self.robot.robot_id
        collision = False

        if not self.no_arm:
            for obs_id in self.asset_loader.obstacles:
                pts = p.getClosestPoints(
                    bodyA=moving_body_id, bodyB=obs_id, distance=0.01,
                    physicsClientId=self.client_id
                )
                if pts:
                    collision = True
                    break

            for obj_id in self.asset_loader.target_objects:
                pts = p.getClosestPoints(
                    bodyA=self.robot.robot_id, bodyB=obj_id, distance=0.01,
                    physicsClientId=self.client_id
                )
                if pts:
                    collision = True
                    break

        obs = self._get_obs()
        delta_p_hidden = obs["vector"][7]  # Индекс 7 в векторе

        # Вычисляем награду
        terminated = False
        if collision:
            reward = config.PENALTY_COLLISION
        elif self._is_out_of_bounds(obs["vector"][:3]):
            reward = config.PENALTY_OOB
        else:
            # Основная награда: снижение неопределённости (delta_p_hidden > 0 — хорошо)
            reward = delta_p_hidden * config.REWARD_SCALE
            
            # Проверяем, найдены ли все объекты за этот эпизод
            if hasattr(self, "cumulative_found_objects") and len(self.cumulative_found_objects) == self.total_target_objects:
                reward += 50.0  # Крупный бонус за нахождение всех объектов!
                terminated = True  # Успешно завершаем эпизод досрочно

        self.current_episode_reward += reward
        self.current_episode_min_p = min(self.current_episode_min_p, self.last_p_hidden)

        truncated = self.step_count >= config.MAX_STEPS_PER_EPISODE

        if not self.headless:
            self._handle_keyboard_and_camera()
            self._render_dashboard(reward, delta_p_hidden)

        # Сохраняем метрики шага
        if self.robot is not None:
            pos, orn = self.robot.get_ee_pose()
        else:
            pos = getattr(self, "no_arm_pos", [1.0, 0.0, 0.5])
            orn = getattr(self, "no_arm_orn", [0, 0, 0, 1])

        step_metrics = {
            "episode": self.episode_count,
            "step": self.step_count,
            "p_hidden": float(self.last_p_hidden),
            "delta_p_hidden": float(delta_p_hidden),
            "reward": float(reward),
            "cum_reward": float(self.current_episode_reward),
            "found_objects": int(self.num_found_objects),
            "hidden_objects": int(self.num_hidden_objects),
            "total_objects": int(self.total_target_objects),
            "cam_x": float(pos[0]), "cam_y": float(pos[1]), "cam_z": float(pos[2]),
            "cam_roll": float(orn[0]), "cam_pitch": float(orn[1]), "cam_yaw": float(orn[2]), "cam_w": float(orn[3])
        }
        self._save_step_metrics(step_metrics)
        self.episode_history.append(step_metrics)

        info = {
            "delta_p_hidden": delta_p_hidden,
            "p_hidden": self.last_p_hidden,
            "nbv_pos": self.odin_adapter.last_nbv_pos if self.odin_adapter else None,
        }

        if terminated or truncated:
            self.episode_count += 1
            self._save_episode_metrics()

        return obs, float(reward), terminated, truncated, info

    def _save_step_metrics(self, step_metrics):
        """Дозаписывает метрики одного шага в общий CSV."""
        import csv
        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_exists = self.metrics_path.exists()
        
        keys = step_metrics.keys()
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists or f.tell() == 0:
                writer.writeheader()
            writer.writerow(step_metrics)

    def _save_episode_metrics(self):
        """Вызывается в конце эпизода: обновляет историю и итоговый график."""
        self.episode_rewards_history.append(self.current_episode_reward)
        self.episode_p_hidden_history.append(self.current_episode_min_p)
        
        # Сохраняем PNG график всей истории
        self._save_summary_plot()

    def _save_summary_plot(self):
        """Сохраняет PNG файл с премиальным дизайном и двумя осями Y."""
        if not self.episode_rewards_history: return
        
        plt.ioff()
        fig = plt.figure(figsize=(14, 8), dpi=120)
        ax1 = fig.add_subplot(111)
        
        eps = range(1, len(self.episode_rewards_history) + 1)
        window = config.PLOT_MOVING_AVERAGE_WINDOW
        
        # --- Левая ось: Reward ---
        color_rew = '#1F77B4' # Стандартный синий
        ax1.set_xlabel('Training Episodes', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Reward', color=color_rew, fontsize=12, fontweight='bold')
        
        # Сырые данные (тонкие линии)
        ax1.plot(eps, self.episode_rewards_history, color=color_rew, alpha=0.15, linewidth=1, label='Raw Episode Reward')
        
        if len(self.episode_rewards_history) >= window:
            rew_ma = np.convolve(self.episode_rewards_history, np.ones(window)/window, mode='valid')
            ma_eps = range(window, len(self.episode_rewards_history) + 1)
            ax1.plot(ma_eps, rew_ma, color=color_rew, linewidth=3, 
                    label=f'Avg Reward (Window: {window} eps)')
            ax1.fill_between(ma_eps, rew_ma, min(self.episode_rewards_history), color=color_rew, alpha=0.1)

        ax1.tick_params(axis='y', labelcolor=color_rew)
        ax1.spines['left'].set_color(color_rew)
        ax1.spines['left'].set_linewidth(2)
        
        # --- Правая ось: p_hidden ---
        ax2 = ax1.twinx()
        color_ph = '#D62728' # Стандартный красный
        ax2.set_ylabel('Min p_hidden (Uncertainty)', color=color_ph, fontsize=12, fontweight='bold')
        
        # Сырые данные (тонкие линии)
        ax2.plot(eps, self.episode_p_hidden_history, color=color_ph, alpha=0.15, linewidth=1, label='Raw p_hidden')
        
        if len(self.episode_p_hidden_history) >= window:
            ph_ma = np.convolve(self.episode_p_hidden_history, np.ones(window)/window, mode='valid')
            ma_eps = range(window, len(self.episode_p_hidden_history) + 1)
            ax2.plot(ma_eps, ph_ma, color=color_ph, linewidth=3, 
                    label=f'Avg p_hidden (Window: {window} eps)')
            ax2.fill_between(ma_eps, ph_ma, 0, color=color_ph, alpha=0.1)

        ax2.tick_params(axis='y', labelcolor=color_ph)
        ax2.spines['right'].set_color(color_ph)
        ax2.spines['right'].set_linewidth(2)
        ax2.set_ylim(0, 1.05)
        
        # Заголовок
        plt.title(f"ODIN-RL Progress: {len(eps)} Episodes\nLast Reward: {self.episode_rewards_history[-1]:.1f} | Last p_hidden: {self.episode_p_hidden_history[-1]:.3f}", 
                  fontsize=14, fontweight='bold', pad=15)
        
        ax1.grid(True, linestyle='--', alpha=0.4)
        
        # Сбор легенды (объединяем все)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True, fontsize=10)
        
        fig.savefig(self.summary_plot_path, bbox_inches='tight')
        plt.close(fig)

    def _load_history_from_csv(self):
        """Загружает историю из CSV для продолжения графиков."""
        if not self.metrics_path.exists():
            self.episode_count = 0
            return

        import csv
        try:
            with open(self.metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                if not data:
                    self.episode_count = 0
                    return
                
                # Собираем статистику по эпизодам
                ep_data = {}
                for row in data:
                    ep = int(row['episode'])
                    rew = float(row['reward'])
                    p_hid = float(row['p_hidden'])
                    if ep not in ep_data:
                        ep_data[ep] = {'total_rew': 0, 'min_p': 1.0}
                    ep_data[ep]['total_rew'] += rew
                    ep_data[ep]['min_p'] = min(ep_data[ep]['min_p'], p_hid)
                
                # Заполняем историю в правильном порядке
                sorted_eps = sorted(ep_data.keys())
                for ep in sorted_eps:
                    self.episode_rewards_history.append(ep_data[ep]['total_rew'])
                    self.episode_p_hidden_history.append(ep_data[ep]['min_p'])
                
                self.episode_count = sorted_eps[-1] + 1
                print(f"[INFO] Loaded history: {len(sorted_eps)} episodes. Next episode: {self.episode_count}")
        except Exception as e:
            print(f"[WARNING] Could not load history from CSV: {e}")
            self.episode_count = 0

    def close(self):
        p.disconnect(physicsClientId=self.client_id)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _update_object_counts(self, seg):
        """Считает сколько целевых объектов видно в данный момент и накапливает найденные за эпизод."""
        visible_ids = np.unique(seg)
        found = 0
        for obj_id in self.asset_loader.target_objects:
            if obj_id in visible_ids:
                pixels = np.sum(seg == obj_id)
                if pixels > 50: # Порог видимости
                    found += 1
                    if not hasattr(self, "cumulative_found_objects"):
                        self.cumulative_found_objects = set()
                    self.cumulative_found_objects.add(obj_id)
        self.num_found_objects = found
        self.num_hidden_objects = self.total_target_objects - len(self.cumulative_found_objects)

    def _get_dummy_obs(self):
        return {
            "image": np.zeros((4, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32),
            "vector": np.zeros(self.VECTOR_DIM, dtype=np.float32),
        }

    def _get_obs(self) -> dict:
        if not self.no_arm:
            pos, orn = self.robot.get_ee_pose()
            joints = self.robot.get_joint_states()
        else:
            pos, orn = p.getBasePositionAndOrientation(
                self.target_obj_id, physicsClientId=self.client_id
            )
            pos = np.array(pos, dtype=np.float32)
            orn = np.array(orn, dtype=np.float32)
            joints = np.zeros(7, dtype=np.float32)

        if not p.isConnected(physicsClientId=self.client_id):
            return self._get_dummy_obs()

        rgb, depth, seg = self.camera.get_image(cam_pos=pos, cam_orn=orn)
        self.last_rgb = rgb

        # Нормализация depth и RGB → 4-канальный тензор
        depth_norm = np.clip(depth / 10.0, 0, 1)
        rgb_norm = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        rgbd = np.concatenate([rgb_norm, depth_norm[np.newaxis, :, :]], axis=0)

        # Подача кадра в ODIN адаптер
        delta_p_hidden = 0.0
        nbv_hint = np.zeros(3, dtype=np.float32)

        if self.odin_adapter is not None:
            if not getattr(self, "external_infer", False):
                # Интринзика камеры из PyBullet
                fx = fy = config.IMAGE_SIZE / 2  # Приближённые значения
                cx = cy = config.IMAGE_SIZE / 2
                # Поза камеры (camera-to-world)
                pose_c2w = self._get_camera_pose(pos, orn)

                self.odin_adapter.add_frame(rgb, depth.astype(np.float32), pose_c2w, fx, fy, cx, cy)

                result = self.odin_adapter.infer(
                    current_pos=np.array(pos, dtype=np.float32),
                    current_quat=np.array(orn, dtype=np.float32),
                )
                delta_p_hidden = float(result["delta_p_hidden"])
                self.last_p_hidden = float(result["p_hidden"])

                if self.use_nbv_hint and result["nbv_pos"] is not None:
                    nbv_hint = np.array(result["nbv_pos"], dtype=np.float32)[:3]
            else:
                # Внешний инференс (SAC / REINFORCE):
                # Адаптер управляется внешним циклом обучения.
                # Мы просто считываем актуальный p_hidden из адаптера и вычисляем delta_p_hidden!
                new_p = float(self.odin_adapter.last_p_hidden)
                delta_p_hidden = float(self.last_p_hidden - new_p)
                self.last_p_hidden = new_p
                if self.use_nbv_hint and getattr(self.odin_adapter, "last_nbv_pos", None) is not None:
                    nbv_hint = np.array(self.odin_adapter.last_nbv_pos, dtype=np.float32)[:3]

        # Дополнительно считаем найденные объекты (на основе маски)
        self._update_object_counts(seg)

        # Вектор: pos(3) + orn(4) + delta_p_hidden(1) + joints(7) + nbv_hint(3)
        vector = np.concatenate([
            np.array(pos, dtype=np.float32),    # 3
            np.array(orn, dtype=np.float32),    # 4
            [delta_p_hidden],                   # 1
            joints,                             # 7
            nbv_hint,                           # 3
        ]).astype(np.float32)                   # = 18

        return {"image": rgbd, "vector": vector}

    def _get_camera_pose(self, cam_pos, cam_orn) -> np.ndarray:
        """Строим camera-to-world матрицу 4×4 из pos и orn (кватернион)."""
        R_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_mat
        pose[:3, 3] = np.array(cam_pos, dtype=np.float32)
        return pose

    def _is_out_of_bounds(self, pos):
        if pos[2] < 0.05:
            return True
        if np.linalg.norm(pos[:2]) > 1.0:
            return True
        return False

    # ------------------------------------------------------------------
    # Визуализация (только non-headless)
    # ------------------------------------------------------------------

    def _update_plot_cache(self):
        """Обновляет кэш графика обучения (история эпизодов)."""
        if not hasattr(self, "episode_rewards_history") or not self.episode_rewards_history:
            return
        
        fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
        eps = range(1, len(self.episode_rewards_history) + 1)
        ax.plot(eps, self.episode_rewards_history, alpha=0.3, color='blue', label='Reward')
        ax.plot(eps, self.episode_p_hidden_history, alpha=0.3, color='red', label='Min p_hidden')
        window = config.PLOT_MOVING_AVERAGE_WINDOW
        if len(self.episode_rewards_history) >= window:
            rew_ma = np.convolve(self.episode_rewards_history, np.ones(window)/window, mode='valid')
            ph_ma = np.convolve(self.episode_p_hidden_history, np.ones(window)/window, mode='valid')
            ma_eps = range(window, len(self.episode_rewards_history) + 1)
            ax.plot(ma_eps, rew_ma, color='blue', linewidth=2, label=f'Reward MA({window})')
            ax.plot(ma_eps, ph_ma, color='red', linewidth=2, label=f'p_hidden MA({window})')
        ax.set_xlabel('Episodes')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        self.cached_plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    def _update_plot(self):
        """Рисует историю p_hidden и Reward текущего эпизода."""
        if len(self.p_hidden_history) < 2:
            return

        fig, ax1 = plt.subplots(figsize=(7, 3), dpi=100)
        
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('p_hidden', color='tab:blue')
        ax1.plot(self.p_hidden_history, color='tab:blue', label='p_hidden')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward', color='tab:green')
        ax2.plot(self.reward_history, color='tab:green', label='Reward')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        fig.tight_layout()
        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        self.cached_plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    def render(self):
        """Возвращает текущий кадр дашборда для записи видео."""
        if self.render_mode == "rgb_array":
            # Создаем дашборд (аналогично _render_dashboard, но без cv2.imshow)
            if self.last_rgb is not None:
                cam_rgb = cv2.resize(self.last_rgb, (300, 300))
            else:
                cam_rgb = np.zeros((300, 300, 3), dtype=np.uint8)

            text_panel = np.ones((300, 400, 3), dtype=np.uint8) * 40
            # Рисуем текст (используем белый цвет для RGB)
            cv2.putText(text_panel, f"Step: {self.step_count}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(text_panel, f"p_hidden: {self.last_p_hidden:.3f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            # Рисуем количество найденных объектов (кумулятивно за эпизод)
            cv2.putText(text_panel, f"Objects: {len(self.cumulative_found_objects)}/{self.total_target_objects}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            cv2.putText(text_panel, f"Ep.Rew: {self.current_episode_reward:.1f}",
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)

            # Добавляем инфо о позе
            if self.robot is not None:
                pos, _ = self.robot.get_ee_pose()
            else:
                pos = getattr(self, "no_arm_pos", [1.0, 0.0, 0.5])
            cv2.putText(text_panel, f"Pos: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}",
                        (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            top_row = np.hstack((cam_rgb, text_panel))
            if self.cached_plot_bgr is not None:
                plot_rgb = cv2.cvtColor(self.cached_plot_bgr, cv2.COLOR_BGR2RGB)
                plot_resized = cv2.resize(plot_rgb, (top_row.shape[1], 300))
                dashboard = np.vstack((top_row, plot_resized))
            else:
                dashboard = top_row
            return dashboard
        return None

    def _render_dashboard(self, current_reward, delta_p_hidden):
        if self.last_rgb is not None:
            cam_bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
            cam_bgr = cv2.resize(cam_bgr, (300, 300))
        else:
            cam_bgr = np.zeros((300, 300, 3), dtype=np.uint8)

        text_panel = np.ones((300, 400, 3), dtype=np.uint8) * 40
        cv2.putText(text_panel, f"Step: {self.step_count}/{config.MAX_STEPS_PER_EPISODE}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(text_panel, f"Reward: {current_reward:.3f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
        cv2.putText(text_panel, f"delta_p_hidden: {delta_p_hidden:.3f}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(text_panel, f"p_hidden: {self.last_p_hidden:.3f}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.putText(text_panel, f"Objects: {len(self.cumulative_found_objects)}/{self.total_target_objects}",
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(text_panel, f"Ep.Rew: {self.current_episode_reward:.2f}",
                    (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        top_row = np.hstack((cam_bgr, text_panel))
        if self.cached_plot_bgr is not None:
            plot_resized = cv2.resize(self.cached_plot_bgr, (top_row.shape[1], 300))
            dashboard = np.vstack((top_row, plot_resized))
        else:
            dashboard = top_row

        cv2.imshow("ODIN NBV Agent Dashboard", dashboard)
        cv2.waitKey(1)

    def _handle_keyboard_and_camera(self):
        try:
            keys = p.getKeyboardEvents(physicsClientId=self.client_id)
            cam_info = p.getDebugVisualizerCamera(physicsClientId=self.client_id)
        except p.error:
            return
        cam_yaw = cam_info[8]
        cam_pitch = cam_info[9]
        cam_forward = np.array(cam_info[5])
        right = np.cross(cam_forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-5:
            right = np.array([0, 1, 0])
        else:
            right /= np.linalg.norm(right)
        step = 0.05
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            self.fps_eye += cam_forward * step
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            self.fps_eye -= cam_forward * step
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            self.fps_eye += right * step
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            self.fps_eye -= right * step
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            self.fps_eye[2] += step
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            self.fps_eye[2] -= step
        new_target = self.fps_eye + cam_forward * 1.0
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=cam_yaw, cameraPitch=cam_pitch,
            cameraTargetPosition=new_target.tolist(), physicsClientId=self.client_id
        )
