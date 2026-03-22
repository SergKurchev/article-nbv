import pybullet as p
import random
import numpy as np
import config
import os

class AssetLoader:
    def __init__(self, client_id):
        self.client_id = client_id
        self.obstacles = []
        
    def load_robot(self):
        # Load kinova arm using absolute short path to avoid PyBullet Cyrillic/absolute path issues
        robot_path_short = config.get_short_path(config.ROBOT_URDF)
        robot_id = p.loadURDF(str(robot_path_short), [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
        return robot_id

    def load_target_object(self, class_id):
        # class_id is 0 to 17
        obj_name = f"Object_{class_id + 1:02d}"
        obj_path = config.OBJECTS_DIR / obj_name / f"{obj_name}.STL"
        
        # We need to create visual and collision shape for STL using short paths
        obj_path_short = config.get_short_path(obj_path)
        
        # 1. Create temporary body to measure AABB and compute proper mesh scale (normalize to ~15cm max)
        temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), physicsClientId=self.client_id)
        temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=self.client_id)
        
        aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
        p.removeBody(temp_body, physicsClientId=self.client_id)
        
        size = np.array(aabb_max) - np.array(aabb_min)
        max_dim = np.max(size)
        
        target_size = 0.15 # 15cm max dimension
        scale = target_size / max_dim if max_dim > 0 else 1.0
        
        visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), meshScale=[scale, scale, scale], physicsClientId=self.client_id)
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), meshScale=[scale, scale, scale], physicsClientId=self.client_id)
        
        # spawn object
        body_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_shape, 
                                    baseVisualShapeIndex=visual_shape, basePosition=[0.5, 0.0, 0.2], # Slightly forward so it doesn't clip robot base
                                    physicsClientId=self.client_id)
                                    
        # Apply the required texture
        tex_path_short = config.get_short_path(config.TEXTURE_PATH)
        tex_id = p.loadTexture(str(tex_path_short), physicsClientId=self.client_id)
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id, physicsClientId=self.client_id)
        
        return body_id

    def generate_obstacles(self):
        self.clear_obstacles()
        num_obs = random.randint(config.MIN_OBSTACLES, config.MAX_OBSTACLES)
        
        for _ in range(num_obs):
            # random dimension for thin panel
            half_extents = [
                random.uniform(0.01, 0.05), # thickness
                random.uniform(0.1, 0.2), # width
                random.uniform(0.1, 0.3)   # height
            ]
            
            # We want obstacles scattered around the target object [0.5, 0.0, 0.2] 
            # to occlude the robot's view dynamically.
            angle = random.uniform(0, 2*np.pi)
            distance = random.uniform(0.2, 0.4) # 20cm to 40cm away from the object
            
            pos = [
                0.5 + distance * np.cos(angle),
                0.0 + distance * np.sin(angle),
                random.uniform(0.0, 0.3)
            ]
            
            # check distance to robot base [0,0,0] to ensure it doesn't block the base linkage
            if np.linalg.norm([pos[0], pos[1]]) < 0.2:
                # push further away from base
                pos[0] += 0.2
            
            # random yaw
            yaw = random.uniform(0, 2*np.pi)
            orn = p.getQuaternionFromEuler([0, 0, yaw])
            
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client_id)
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
            
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, 
                                        baseVisualShapeIndex=visual_shape, basePosition=pos, 
                                        baseOrientation=orn, physicsClientId=self.client_id)
            self.obstacles.append(body_id)
            
    def clear_obstacles(self):
        for obs in self.obstacles:
            p.removeBody(obs, physicsClientId=self.client_id)
        self.obstacles.clear()
