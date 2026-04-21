import pybullet as p
import random
import numpy as np
import config
import os

class AssetLoader:
    def __init__(self, client_id):
        self.client_id = client_id
        self.obstacles = []
        self.texture_cache = {}
        
    def load_robot(self):
        # Load kinova arm using absolute short path to avoid PyBullet Cyrillic/absolute path issues
        robot_path_short = config.get_short_path(config.ROBOT_URDF)
        robot_id = p.loadURDF(str(robot_path_short), [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
        return robot_id

    def load_target_object(self, class_id, texture_type=None):
        """Load target object with specified texture.

        Args:
            class_id: Object class ID (0 to NUM_CLASSES-1)
            texture_type: Texture type ('red', 'mixed', 'green').
                         If None, uses class_id % 3 mapping.

        Returns:
            int: PyBullet body ID
        """
        # class_id is 0 to 17
        obj_name = f"Object_{class_id + 1:02d}"

        # Determine texture type if not specified
        if texture_type is None:
            texture_type = ['red', 'mixed', 'green'][class_id % 3]
        obj_path = config.OBJECTS_DIR / obj_name / f"{obj_name}.STL"
        
        # Check if JSON optimized mesh data exists (bypasses potential STL parser crashes)
        json_path = config.OBJECTS_DIR / obj_name / "mesh.json"
        
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                mesh_data = json.load(f)
            v = mesh_data["vertices"]
            ind = mesh_data["indices"]
            
            # Temporary creation to measure scale
            temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, physicsClientId=self.client_id)
            temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=self.client_id)
            aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
            p.removeBody(temp_body, physicsClientId=self.client_id)
            
            size = np.array(aabb_max) - np.array(aabb_min)
            max_dim = np.max(size)
            target_size = 0.15
            scale = target_size / max_dim if max_dim > 0 else 1.0
            
            visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, meshScale=[scale, scale, scale], physicsClientId=self.client_id)
            collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, meshScale=[scale, scale, scale], physicsClientId=self.client_id)
        else:
            # Traditional STL load (as a fallback or for complex objects)
            obj_path_short = config.get_short_path(obj_path)
            
            temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), physicsClientId=self.client_id)
            temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=self.client_id)
            aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
            p.removeBody(temp_body, physicsClientId=self.client_id)
            
            size = np.array(aabb_max) - np.array(aabb_min)
            max_dim = np.max(size)
            target_size = 0.15
            scale = target_size / max_dim if max_dim > 0 else 1.0
            
            visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), meshScale=[scale, scale, scale], physicsClientId=self.client_id)
            collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), meshScale=[scale, scale, scale], physicsClientId=self.client_id)
        
        # spawn object
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape, basePosition=[0.5, 0.0, 0.2],
                                    physicsClientId=self.client_id)

        # Apply the required texture based on texture_type
        tex_id = self._load_texture(texture_type)

        # Apply texture WITHOUT rgbaColor to preserve texture patterns
        # Only set specularColor to reduce lighting artifacts
        p.changeVisualShape(
            body_id,
            -1,
            textureUniqueId=tex_id,
            specularColor=[0, 0, 0],  # No specular reflection
            physicsClientId=self.client_id
        )

        return body_id

    def _load_texture(self, texture_type):
        """Load texture with caching and random variant selection.

        Args:
            texture_type: One of 'red', 'mixed', 'green'

        Returns:
            int: PyBullet texture ID

        Raises:
            FileNotFoundError: If texture file doesn't exist
        """
        # For mixed textures, randomly select a variant
        if texture_type == 'mixed':
            import random
            # Check how many mixed variants exist
            texture_dir = config.DATA_DIR / "objects" / "textures"
            variants = list(texture_dir.glob("mixed*.png"))
            if variants:
                # Randomly select one variant
                selected_texture = random.choice(variants)
                cache_key = f"mixed_{selected_texture.stem}"
            else:
                cache_key = texture_type
                selected_texture = texture_dir / f"{texture_type}.png"
        else:
            cache_key = texture_type
            texture_dir = config.DATA_DIR / "objects" / "textures"
            selected_texture = texture_dir / f"{texture_type}.png"

        # Check cache
        if cache_key in self.texture_cache:
            return self.texture_cache[cache_key]

        if not selected_texture.exists():
            raise FileNotFoundError(
                f"Texture not found: {selected_texture}. "
                f"Run 'uv run python src/vision/texture_generator.py' to generate textures."
            )

        tex_path_short = config.get_short_path(selected_texture)
        tex_id = p.loadTexture(str(tex_path_short), physicsClientId=self.client_id)
        self.texture_cache[cache_key] = tex_id

        return tex_id

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
