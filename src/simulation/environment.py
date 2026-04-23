import gymnasium as gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import config
from src.simulation.asset_loader import AssetLoader
from src.simulation.robot import Robot
from src.simulation.camera import Camera
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NBVEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, headless=True, no_arm=False, vision_model=None):
        super().__init__()
        self.render_mode = render_mode
        self.headless = headless
        self.no_arm = no_arm
        
        # Connect to PyBullet
        connection_mode = p.DIRECT if headless else p.GUI
        self.client_id = p.connect(connection_mode)
        
        # Spaces
        # Action: [x, y, z, roll, pitch, yaw] (Absolute)
        self.action_space = spaces.Box(
            low=np.array(config.ACTION_MIN, dtype=np.float32),
            high=np.array(config.ACTION_MAX, dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation: Dict with image and vector
        # image: [4, 224, 224] (RGB-D)
        # vector: [15] (pos, orn, acc_diff, joints)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(4, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        })
        
        # Components
        self.asset_loader = AssetLoader(self.client_id)
        self.camera = Camera(self.client_id)
        
        self.robot = None
        self.target_obj_id = None
        self.current_class_id = 0
        self.last_probs = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        
        self.vision_model = vision_model
        if self.vision_model is not None:
            self.vision_model.eval()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.step_count = 0
        

        self.fps_eye = np.array([1.5, 0.0, 0.5])
        self.text_ids = None
        self.last_rgb = None

        # --- Переменные для графиков и истории ---
        self.episode_rewards_history = []
        self.episode_acc_diffs_history = []
        self.current_episode_reward = 0.0
        self.current_episode_max_acc = 0.0
        self.cached_plot_bgr = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.step_count > 0:
            self.episode_rewards_history.append(self.current_episode_reward)
            self.episode_acc_diffs_history.append(self.current_episode_max_acc)
            
        self.step_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_max_acc = 0.0
        
        # Генерируем новый кадр графика (1 раз за эпизод, чтобы не тормозить RL!)
        if not self.headless:
            self._update_plot_cache()
                    
        # Reconnect if disconnected
        if not p.isConnected(physicsClientId=self.client_id):
            print("Reconnecting to physics server...")
            connection_mode = p.DIRECT if self.headless else p.GUI
            self.client_id = p.connect(connection_mode)
            # Re-init asset loader and camera with new client_id
            self.asset_loader = AssetLoader(self.client_id)
            self.camera = Camera(self.client_id)

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            
        # Add pybullet_data to search path for basic shapes like plane.urdf
        p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=self.client_id)

        # Load plane and make it black
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0, 0, 0, 1], physicsClientId=self.client_id)
        
        # If robot is used
        if not self.no_arm:
            robot_urdf = self.asset_loader.load_robot()
            self.robot = Robot(self.client_id, robot_urdf)
            self.robot.reset()

        # Generate scene based on SCENE_STAGE
        target_objects = self.asset_loader.generate_scene()

        # For Stage 1, use single object as target
        # For Stage 2/3, select first object as primary target (for now)
        self.target_obj_id = target_objects[0]
        self.current_class_id = self.np_random.integers(0, config.NUM_CLASSES)

        if self.no_arm:
            # Let it float
            p.changeDynamics(self.target_obj_id, -1, mass=0)

        # Wait for settling
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            
        if not self.no_arm:
            # We explicitly DO NOT wrap the target object in a constraint to the EE!
            # The object must rest on the ground while the arm flys around it.
            pass

        obs = self._get_obs()
                               
        # if not self.headless:
            # self.text_ids = self._update_debug_text(self.step_count, 0.0, 0.0, text_ids=None)

        
            
        return obs, {}
        
    def step(self, action):
        if not self.no_arm:
            target_pos = action[:3]
            target_euler = action[3:]
            target_orn = p.getQuaternionFromEuler(target_euler)
            
            # Action is absolute now
            self.robot.apply_action(target_pos, target_orn)
        else:
            # Move the object directly
            target_pos = action[:3]
            target_euler = action[3:]
            target_orn = p.getQuaternionFromEuler(target_euler)
            p.resetBasePositionAndOrientation(self.target_obj_id, target_pos, target_orn, physicsClientId=self.client_id)
            
        if not p.isConnected(physicsClientId=self.client_id):
            print("Warning: Physics server disconnected. Skipping step.")
            return obs, 0, True, True, {}

        p.stepSimulation(physicsClientId=self.client_id)
        self.step_count += 1
        
        # check collisions with obstacles (bad)
        moving_body_id = self.target_obj_id if self.no_arm else self.robot.robot_id

        collision = False

        # 1. Check collisions with all obstacles
        for obs_id in self.asset_loader.obstacles:
            pts = p.getClosestPoints(bodyA=moving_body_id, bodyB=obs_id, distance=0.01, physicsClientId=self.client_id)
            if pts:
                collision = True
                break

        # 2. If robot is moving, check collision with all target objects
        if not self.no_arm and not collision:
            for obj_id in self.asset_loader.target_objects:
                pts = p.getClosestPoints(bodyA=self.robot.robot_id, bodyB=obj_id, distance=0.01, physicsClientId=self.client_id)
                if pts:
                    collision = True
                    break

                
        obs = self._get_obs()
        acc_diff = obs["vector"][7]
        
        # Reward function
        if collision:
            reward = config.PENALTY_COLLISION
            # terminated = True  
            terminated = False  
            if not self.headless:
                print("Collision detected! Penalty applied.")
                
        elif self._is_out_of_bounds(obs["vector"][:3]):
            reward = config.PENALTY_OOB
            # terminated = True 
            terminated = False 
            if not self.headless:
                print("Out of bounds! Penalty applied.")
                
        else:
            reward = acc_diff * config.REWARD_SCALE
            terminated = False
            
        
        # if not self.headless:
            # self.text_ids = self._update_debug_text(self.step_count, reward, acc_diff, text_ids=self.text_ids)
            # self._handle_keyboard_and_camera()
            
            # if self.last_rgb is not None:
            #     bgr_img = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
                
            #     bgr_img_large = cv2.resize(bgr_img, (448, 448), interpolation=cv2.INTER_NEAREST)
            #     cv2.imshow("Agent Camera View", bgr_img_large)
            #     cv2.waitKey(1)
        

        self.current_episode_reward += reward
        self.current_episode_max_acc = max(self.current_episode_max_acc, acc_diff)
            
        truncated = self.step_count >= config.MAX_STEPS_PER_EPISODE
        
        if not self.headless:
            self._handle_keyboard_and_camera()
            self._render_dashboard(reward, acc_diff)

            # self.text_ids = self._update_debug_text(self.step_count, reward, acc_diff, text_ids=self.text_ids)
            # self._handle_keyboard_and_camera()
            
            # if self.last_rgb is not None:
            #     bgr_img = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
                
            #     bgr_img_large = cv2.resize(bgr_img, (448, 448), interpolation=cv2.INTER_NEAREST)
            #     cv2.imshow("Agent Camera View", bgr_img_large)
            #     cv2.waitKey(1)
        
        return obs, reward, terminated, truncated, {"acc_diff": acc_diff, "cnn_probs": self.last_probs}
        
    def _is_out_of_bounds(self, pos):
        # Prevent going under floor or too far
        if pos[2] < 0.05: return True
        if np.linalg.norm(pos[:2]) > 1.0: return True
        return False

    def _get_obs(self):
        if not self.no_arm:
            pos, orn = self.robot.get_ee_pose()
            joints = self.robot.get_joint_states()
        else:
            # When floating, attach camera closely in front of target obj
            pos, orn = p.getBasePositionAndOrientation(self.target_obj_id, physicsClientId=self.client_id)
            joints = np.zeros(7, dtype=np.float32) # No joints if no arm
            
        # Render image from end effector perspective
        if not p.isConnected(physicsClientId=self.client_id):
            return {"image": np.zeros((4, 224, 224), dtype=np.float32), "vector": np.zeros(15, dtype=np.float32)}

        rgb, depth, seg = self.camera.get_image(cam_pos=pos, cam_orn=orn)
        
        self.last_rgb = rgb
            
        # ---------------------------------------------------------
        # 1. СОБИРАЕМ 4-КАНАЛЬНОЕ ИЗОБРАЖЕНИЕ (RGB-D) ЗАРАНЕЕ
        # ---------------------------------------------------------
        # Normalize depth to [0, 1] relative to [near, far]
        depth_normalized = np.clip(depth / 10.0, 0, 1)
        # Transpose RGB to [3, H, W] and normalize
        rgb_normalized = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Stack into 4 channels: [4, 224, 224]
        rgbd = np.concatenate([rgb_normalized, depth_normalized[np.newaxis, :, :]], axis=0)
            
        # ---------------------------------------------------------
        # 2. ПОЛУЧАЕМ ПРЕДСКАЗАНИЕ СЕТИ
        # ---------------------------------------------------------
        acc_diff = 0.0
        
        # Собираем предварительный вектор (acc_diff = 0.0) для подачи в нейросеть
        pre_vector = np.concatenate([pos, orn, [acc_diff], joints]).astype(np.float32)

        if self.vision_model is not None:
            from src.vision.metrics import get_accuracy_difference
            device = next(self.vision_model.parameters()).device
            
            with torch.no_grad():
                # ИСПОЛЬЗУЕМ rgbd (4 канала) ВМЕСТО self.transform(rgb) (3 канала)
                img_t = torch.FloatTensor(rgbd).unsqueeze(0).to(device)
                vec_t = torch.FloatTensor(pre_vector).unsqueeze(0).to(device)
                
                outputs = self.vision_model(img_t, vec_t)
                
                # Распаковываем вывод (logits)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                acc_diff = get_accuracy_difference(logits).item()
                import torch.nn.functional as F
                self.last_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
        # ---------------------------------------------------------
        # 3. СОБИРАЕМ ФИНАЛЬНЫЙ ВЕКТОР ДЛЯ RL
        # ---------------------------------------------------------
        # Vector part: pos(3), orn(4), acc_diff(1), joints(7) = 15 total
        vector = np.concatenate([
            pos, 
            orn, 
            [acc_diff], 
            joints
        ]).astype(np.float32)
        
        return {
            "image": rgbd,
            "vector": vector
        }

    def _update_plot_cache(self):
        """Рендерит график через Matplotlib и кэширует его в виде картинки"""
        fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
        
        eps = range(1, len(self.episode_rewards_history) + 1)
        
        # Сырые данные (полупрозрачные)
        ax.plot(eps, self.episode_rewards_history, alpha=0.3, color='blue', label='Reward')
        ax.plot(eps, self.episode_acc_diffs_history, alpha=0.3, color='orange', label='Max Acc Diff')
        
        # Скользящее среднее
        window = config.PLOT_MOVING_AVERAGE_WINDOW
        if len(self.episode_rewards_history) >= window:
            rew_ma = np.convolve(self.episode_rewards_history, np.ones(window)/window, mode='valid')
            acc_ma = np.convolve(self.episode_acc_diffs_history, np.ones(window)/window, mode='valid')
            ma_eps = range(window, len(self.episode_rewards_history) + 1)
            
            ax.plot(ma_eps, rew_ma, color='blue', linewidth=2, label=f'Reward MA({window})')
            ax.plot(ma_eps, acc_ma, color='orange', linewidth=2, label=f'Acc Diff MA({window})')
            
        ax.set_xlabel('Episodes')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout(pad=0.5)
        
        # Конвертируем график в массив numpy
        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())
        plot_img_rgb = plot_img[:, :, :3]  # Берем только RGB каналы, отбрасывая Alpha-канал
        plt.close(fig)
        
        # Сохраняем в BGR для OpenCV
        self.cached_plot_bgr = cv2.cvtColor(plot_img_rgb, cv2.COLOR_RGB2BGR)

    def _render_dashboard(self, current_reward, current_acc_diff):
        """Собирает картинку с камеры, текст и график в единый UI"""
        # 1. Подготовка камеры робота
        if self.last_rgb is not None:
            cam_bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
            cam_bgr = cv2.resize(cam_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)
        else:
            cam_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
            
        # 2. Текстовая панель
        text_panel = np.ones((300, 400, 3), dtype=np.uint8) * 40 # Темно-серый фон
        
        # Текущие метрики шага
        cv2.putText(text_panel, f"Step: {self.step_count} / {config.MAX_STEPS_PER_EPISODE}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(text_panel, f"Step Reward: {current_reward:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
        cv2.putText(text_panel, f"Step Acc Diff: {current_acc_diff:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 255), 2)
        
        # Метрики текущего эпизода
        cv2.putText(text_panel, "--- Current Episode ---", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(text_panel, f"Cumulative Rew: {self.current_episode_reward:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        cv2.putText(text_panel, f"Max Acc Diff: {self.current_episode_max_acc:.2f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)
        
        # 3. Собираем верхний ряд (Камера + Текст)
        top_row = np.hstack((cam_bgr, text_panel))
        
        # 4. Собираем всё вместе с графиком
        if self.cached_plot_bgr is not None:
            # Растягиваем график под ширину верхнего ряда
            plot_resized = cv2.resize(self.cached_plot_bgr, (top_row.shape[1], 300))
            dashboard = np.vstack((top_row, plot_resized))
        else:
            dashboard = top_row
            
        cv2.imshow("Agent Dashboard (CV + RL)", dashboard)
        cv2.waitKey(1)

    def close(self):
        p.disconnect(physicsClientId=self.client_id)


    # def _update_debug_text(self, env_steps, reward, acc_diff, text_ids=None):
    #     pos_steps = [-0.5, 0.5, 1.1]
    #     pos_rew   = [-0.5, 0.5, 1.0]
    #     pos_acc   = [-0.5, 0.5, 0.9]
        
    #     try:
    #         if text_ids is None:
    #             id1 = p.addUserDebugText(f"Steps: {env_steps}", pos_steps, textColorRGB=[1,1,1], textSize=1.5, physicsClientId=self.client_id)
    #             id2 = p.addUserDebugText(f"Reward: {reward:.2f}", pos_rew, textColorRGB=[1,0.2,0.2], textSize=1.5, physicsClientId=self.client_id)
    #             id3 = p.addUserDebugText(f"Acc Diff: {acc_diff:.2f}", pos_acc, textColorRGB=[1,1,0], textSize=1.5, physicsClientId=self.client_id)
    #             return id1, id2, id3
    #         else:
    #             id1 = p.addUserDebugText(f"Steps: {env_steps}", pos_steps, textColorRGB=[1,1,1], textSize=1.5, replaceItemUniqueId=text_ids[0], physicsClientId=self.client_id)
    #             id2 = p.addUserDebugText(f"Reward: {reward:.2f}", pos_rew, textColorRGB=[1,0.2,0.2], textSize=1.5, replaceItemUniqueId=text_ids[1], physicsClientId=self.client_id)
    #             id3 = p.addUserDebugText(f"Acc Diff: {acc_diff:.2f}", pos_acc, textColorRGB=[1,1,0], textSize=1.5, replaceItemUniqueId=text_ids[2], physicsClientId=self.client_id)
    #             return id1, id2, id3
    #     except p.error:
    #         return None

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
            right = right / np.linalg.norm(right)

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
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=cam_yaw, 
                                     cameraPitch=cam_pitch, cameraTargetPosition=new_target.tolist(), 
                                     physicsClientId=self.client_id)