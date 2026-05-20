import pybullet as p
import pybullet_data
import random
import math
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from src.simulation.robot import Robot
from src.simulation.asset_loader import AssetLoader

client_id = p.connect(p.DIRECT)
p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=client_id)
plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)

robot_id = p.loadURDF(config.ROBOT_URDF, [0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=client_id)
robot = Robot(client_id, robot_id)

fail_reach = 0
fail_col_plane = 0
success = 0

for _ in range(200):
    cam_x = random.uniform(-0.8, 0.8)
    cam_y = random.uniform(-0.8, 0.8)
    cam_z = random.uniform(0.05, 0.8)
    candidate_pos = [cam_x, cam_y, cam_z]
    if np.linalg.norm(candidate_pos) > 0.8:
        continue
        
    target_x = random.uniform(config.SCENE_BOUNDS_X_MIN, config.SCENE_BOUNDS_X_MAX)
    target_y = random.uniform(config.SCENE_BOUNDS_Y_MIN, config.SCENE_BOUNDS_Y_MAX)
    target_z = random.uniform(config.SCENE_BOUNDS_Z_MIN, config.SCENE_BOUNDS_Z_MAX)
    cam_target = [target_x, target_y, target_z]
    
    diff = np.array(cam_target) - np.array(candidate_pos)
    norm = np.linalg.norm(diff)
    if norm < 0.1: continue
    
    yaw = math.atan2(diff[1], diff[0])
    pitch = math.asin(np.clip(diff[2] / norm, -1.0, 1.0))
    cam_orn = p.getQuaternionFromEuler([0, pitch - math.pi/2, yaw])
    
    robot.apply_action(candidate_pos, cam_orn)
    p.stepSimulation(physicsClientId=client_id)
    
    collision = False
    pts = p.getClosestPoints(bodyA=robot.robot_id, bodyB=plane_id, distance=0.01, physicsClientId=client_id)
    for pt in pts:
        if pt[3] > 2: # Ignore base links
            collision = True
            break
            
    if collision:
        fail_col_plane += 1
        continue
        
    actual_pos, actual_orn = robot.get_ee_pose()
    if np.linalg.norm(np.array(actual_pos) - np.array(candidate_pos)) > 0.1:
        fail_reach += 1
        continue
        
    success += 1

print(f"Success: {success}, Reach fail: {fail_reach}, Plane col fail: {fail_col_plane}")
