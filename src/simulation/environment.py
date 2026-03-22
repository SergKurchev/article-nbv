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
        
        self.vision_model = vision_model
        if self.vision_model is not None:
            self.vision_model.eval()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            
        # Add pybullet_data to search path for basic shapes like plane.urdf
        p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=self.client_id)
        
        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # If robot is used
        if not self.no_arm:
            robot_urdf = self.asset_loader.load_robot()
            self.robot = Robot(self.client_id, robot_urdf)
            self.robot.reset()
            
        # Select target object
        self.current_class_id = self.np_random.integers(0, config.NUM_CLASSES)
        self.target_obj_id = self.asset_loader.load_target_object(self.current_class_id)
        if self.no_arm:
            # Let it float
            p.changeDynamics(self.target_obj_id, -1, mass=0)
            
        # Generate obstacles
        self.asset_loader.generate_obstacles()
        
        # Wait for settling
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            
        if not self.no_arm:
            # We explicitly DO NOT wrap the target object in a constraint to the EE!
            # The object must rest on the ground while the arm flys around it.
            pass
                               
        obs = self._get_obs()
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
            
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_count += 1
        
        # check collisions with obstacles (bad)
        collision = False
        for obs_id in self.asset_loader.obstacles:
            pts = p.getContactPoints(self.target_obj_id, obs_id, physicsClientId=self.client_id)
            if pts:
                collision = True
                break
                
        obs = self._get_obs()
        acc_diff = obs[7]
        
        # Reward function
        if collision or self._is_out_of_bounds(obs[:3]):
            reward = config.PENALTY_OOB
            terminated = True
        else:
            # Reward for increasing accuracy difference
            reward = acc_diff * config.REWARD_SCALE
            terminated = False
            
        truncated = self.step_count >= config.MAX_STEPS_PER_EPISODE
        
        return obs, reward, terminated, truncated, {"acc_diff": acc_diff}
        
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
        rgb, depth, seg = self.camera.get_image(cam_pos=pos, cam_orn=orn)
            
        # Get vision inference using ONLY RGB for now (or RGB-D if model is updated)
        acc_diff = 0.0
        if self.vision_model is not None:
            # Prepare image
            from src.vision.metrics import get_accuracy_difference
            with torch.no_grad():
                img_t = self.transform(rgb).unsqueeze(0) # [1, 3, 224, 224]
                logits = self.vision_model(img_t)
                acc_diff = get_accuracy_difference(logits, self.current_class_id)
                
        # Vector part: pos(3), orn(4), acc_diff(1), joints(7) = 15 total
        vector = np.concatenate([
            pos, 
            orn, 
            [acc_diff], 
            joints
        ]).astype(np.float32)
        
        # Image part: RGB-D [4, 224, 224]
        # Normalize depth to [0, 1] relative to [near, far]
        depth_normalized = np.clip(depth / 10.0, 0, 1)
        # Transpose RGB to [3, H, W] and normalize
        rgb_normalized = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Stack into 4 channels
        rgbd = np.concatenate([rgb_normalized, depth_normalized[np.newaxis, :, :]], axis=0)
        
        return {
            "image": rgbd,
            "vector": vector
        }

    def close(self):
        p.disconnect(physicsClientId=self.client_id)
