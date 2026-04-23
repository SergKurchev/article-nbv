import pybullet as p
import random
import numpy as np
import config
import os

class AssetLoader:
    def __init__(self, client_id):
        self.client_id = client_id
        self.obstacles = []
        self.target_objects = []
        self.target_objects_classes = []  # Track class IDs for target objects
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
            target_size = 0.15 * config.OBJECT_SCALE_FACTOR
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
            target_size = 0.15 * config.OBJECT_SCALE_FACTOR
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
        texture_dir = config.DATA_DIR / "objects" / "textures"

        # For mixed textures, randomly select from pre-generated variants
        if texture_type == 'mixed':
            # Randomly select one of the 20 pre-generated mixed textures
            variant_idx = random.randint(0, config.TEXTURE_NUM_MIXED_VARIANTS - 1)
            selected_texture = texture_dir / f"mixed_{variant_idx}.png"
            # Don't cache mixed textures - we want random selection each time
            cache_key = None
        else:
            cache_key = texture_type
            selected_texture = texture_dir / f"{texture_type}.png"

        # Check cache (only for non-mixed textures)
        if cache_key and cache_key in self.texture_cache:
            return self.texture_cache[cache_key]

        if not selected_texture.exists():
            raise FileNotFoundError(
                f"Texture not found: {selected_texture}. "
                f"Run 'uv run python src/vision/texture_generator.py' to generate textures."
            )

        tex_path_short = config.get_short_path(selected_texture)
        tex_id = p.loadTexture(str(tex_path_short), physicsClientId=self.client_id)

        # Cache only non-mixed textures
        if cache_key:
            self.texture_cache[cache_key] = tex_id

        return tex_id

    def generate_scene(self):
        """Generate scene based on SCENE_STAGE configuration.

        Stage 1: Single object at fixed position, no obstacles
        Stage 2: Multiple objects (2-10) with uniform spatial distribution, no obstacles
        Stage 3: Multiple objects (2-10) + obstacles with uniform spatial distribution

        Returns:
            list: List of (object_id, class_id) tuples
        """
        self.clear_scene()

        if config.SCENE_STAGE == 1:
            # Stage 1: Single object at fixed position
            class_id = random.randint(0, config.NUM_CLASSES - 1)
            texture_type = ['red', 'mixed', 'green'][class_id % 3]
            # Calculate proper Z position: center of object must be at least half_height above ground
            _, half_height = self._get_object_bounding_box(class_id)
            z_position = config.SCENE_BOUNDS_Z_MIN + half_height + 0.05  # Add 5cm safety margin
            obj_id = self._create_object_at_position(class_id, texture_type, [0.5, 0.0, z_position])
            self.target_objects.append(obj_id)
            self.target_objects_classes.append(class_id)
            num_obstacles = 0

        elif config.SCENE_STAGE == 2:
            # Stage 2: Multiple objects, no obstacles
            num_objects = random.randint(config.MIN_OBJECTS, config.MAX_OBJECTS)
            positions = self._generate_collision_free_positions(num_objects)

            for pos, class_id, radius in positions:
                texture_type = ['red', 'mixed', 'green'][class_id % 3]
                obj_id = self._create_object_at_position(class_id, texture_type, pos)
                self.target_objects.append(obj_id)
                self.target_objects_classes.append(class_id)

            num_obstacles = 0

        elif config.SCENE_STAGE == 3:
            # Stage 3: Multiple objects + obstacles with interleaved placement
            num_objects = random.randint(config.MIN_OBJECTS, config.MAX_OBJECTS)
            num_obstacles = random.randint(config.MIN_OBSTACLES, config.MAX_OBSTACLES)

            # Create interleaved placement queue
            placement_queue = []
            for i in range(max(num_objects, num_obstacles)):
                if i < num_objects:
                    placement_queue.append(('object', i))
                if i < num_obstacles:
                    placement_queue.append(('obstacle', i))

            # Shuffle to randomize order
            random.shuffle(placement_queue)

            # Track all placed items for collision detection
            placed_items = []  # List of (position, radius) tuples

            # Place items sequentially
            for item_type, item_idx in placement_queue:
                if item_type == 'object':
                    # Randomly select object class and texture
                    class_id = random.randint(0, config.NUM_CLASSES - 1)
                    texture_type = ['red', 'mixed', 'green'][class_id % 3]
                    object_radius, object_half_height = self._get_object_bounding_box(class_id)

                    # Find collision-free position
                    placed = False
                    for attempt in range(10):  # 10 attempts per object
                        x = random.uniform(config.SCENE_BOUNDS_X_MIN, config.SCENE_BOUNDS_X_MAX)
                        y = random.uniform(config.SCENE_BOUNDS_Y_MIN, config.SCENE_BOUNDS_Y_MAX)
                        z_min = config.SCENE_BOUNDS_Z_MIN + object_half_height + 0.05
                        z = random.uniform(z_min, config.SCENE_BOUNDS_Z_MAX)
                        pos = [x, y, z]

                        # Check collision with all placed items
                        collision = False
                        for existing_pos, existing_radius in placed_items:
                            distance = np.linalg.norm(np.array(pos) - np.array(existing_pos))
                            min_safe_distance = object_radius + existing_radius + 0.05
                            if distance < min_safe_distance:
                                collision = True
                                break

                        if not collision:
                            obj_id = self._create_object_at_position(class_id, texture_type, pos)
                            self.target_objects.append(obj_id)
                            self.target_objects_classes.append(class_id)
                            placed_items.append((pos, object_radius))
                            placed = True
                            break

                    if not placed:
                        print(f"Warning: Could not place object {len(self.target_objects)+1} after 10 attempts")

                elif item_type == 'obstacle':
                    # Randomly select obstacle dimensions
                    half_extents = [
                        random.uniform(0.01, 0.05) * config.OBSTACLE_SCALE_FACTOR,
                        random.uniform(0.1, 0.2) * config.OBSTACLE_SCALE_FACTOR,
                        random.uniform(0.1, 0.3) * config.OBSTACLE_SCALE_FACTOR
                    ]
                    obstacle_radius = max(half_extents)

                    # Find collision-free position
                    placed = False
                    for attempt in range(10):  # 10 attempts per obstacle
                        x = random.uniform(config.SCENE_BOUNDS_X_MIN, config.SCENE_BOUNDS_X_MAX)
                        y = random.uniform(config.SCENE_BOUNDS_Y_MIN, config.SCENE_BOUNDS_Y_MAX)
                        z_min = config.SCENE_BOUNDS_Z_MIN + half_extents[2] + 0.05
                        z = random.uniform(z_min, config.SCENE_BOUNDS_Z_MAX)
                        pos = [x, y, z]

                        # Check collision with all placed items
                        collision = False
                        for existing_pos, existing_radius in placed_items:
                            distance = np.linalg.norm(np.array(pos) - np.array(existing_pos))
                            min_safe_distance = obstacle_radius + existing_radius + 0.05
                            if distance < min_safe_distance:
                                collision = True
                                break

                        # Check collision with robot base
                        if not collision:
                            base_distance = np.linalg.norm([pos[0], pos[1]])
                            if base_distance < 0.25:
                                collision = True

                        if not collision:
                            yaw = random.uniform(0, 2 * np.pi)
                            orn = p.getQuaternionFromEuler([0, 0, yaw])

                            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                                              rgbaColor=[0.5, 0.5, 0.5, 1],
                                                              physicsClientId=self.client_id)
                            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                                                     physicsClientId=self.client_id)

                            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                                       baseVisualShapeIndex=visual_shape, basePosition=pos,
                                                       baseOrientation=orn, physicsClientId=self.client_id)
                            self.obstacles.append(body_id)
                            placed_items.append((pos, obstacle_radius))
                            placed = True
                            break

                    if not placed:
                        print(f"Warning: Could not place obstacle {len(self.obstacles)+1} after 10 attempts")

        else:
            raise ValueError(f"Invalid SCENE_STAGE: {config.SCENE_STAGE}. Must be 1, 2, or 3.")

        return list(zip(self.target_objects, self.target_objects_classes))

    def _get_object_bounding_box(self, class_id):
        """Get bounding box size for object class.

        Args:
            class_id: Object class ID

        Returns:
            tuple: (radius, half_height) - radius for XY collision, half_height for Z placement
        """
        obj_name = f"Object_{class_id + 1:02d}"
        json_path = config.OBJECTS_DIR / obj_name / "mesh.json"

        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                mesh_data = json.load(f)
            v = np.array(mesh_data["vertices"])

            # Calculate bounding box
            min_coords = v.min(axis=0)
            max_coords = v.max(axis=0)
            size = max_coords - min_coords
            max_dim = np.max(size)

            # Apply same scaling as in object creation
            target_size = 0.15
            scale = target_size / max_dim if max_dim > 0 else 1.0

            # Return (radius for XY collision, half_height for Z placement)
            scaled_size = size * scale
            radius = max(scaled_size[0], scaled_size[1]) / 2  # Max of X, Y dimensions
            half_height = scaled_size[2] / 2  # Z dimension
            return (radius, half_height)
        else:
            # Fallback: conservative estimate
            return (0.075, 0.075)  # (radius, half_height)

    def _generate_collision_free_positions(self, num_objects):
        """Generate collision-free positions for objects using uniform distribution.

        Args:
            num_objects: Number of objects to place

        Returns:
            list: List of [x, y, z, class_id, radius] tuples
        """
        positions = []

        for _ in range(num_objects):
            # Randomly select object class
            class_id = random.randint(0, config.NUM_CLASSES - 1)
            object_radius, object_half_height = self._get_object_bounding_box(class_id)

            for attempt in range(config.SCENE_MAX_PLACEMENT_ATTEMPTS):
                # Uniform distribution within bounds
                x = random.uniform(config.SCENE_BOUNDS_X_MIN, config.SCENE_BOUNDS_X_MAX)
                y = random.uniform(config.SCENE_BOUNDS_Y_MIN, config.SCENE_BOUNDS_Y_MAX)
                # Z coordinate: center must be at least half_height above ground + safety margin
                z_min = config.SCENE_BOUNDS_Z_MIN + object_half_height + 0.05  # 5cm safety margin
                z = random.uniform(z_min, config.SCENE_BOUNDS_Z_MAX)

                pos = [x, y, z]

                # Check collision with existing positions
                # Use actual bounding box radii + safety margin
                collision = False
                for existing_pos, existing_class, existing_radius in positions:
                    distance = np.linalg.norm(np.array(pos) - np.array(existing_pos[:3]))
                    min_safe_distance = object_radius + existing_radius + 0.05  # 5cm safety margin
                    if distance < min_safe_distance:
                        collision = True
                        break

                if not collision:
                    positions.append((pos, class_id, object_radius))
                    break
            else:
                # Failed to find collision-free position after max attempts
                print(f"Warning: Could not place object {len(positions)+1} after {config.SCENE_MAX_PLACEMENT_ATTEMPTS} attempts")
                continue

        return positions

    def _create_object_at_position(self, class_id, texture_type, position):
        """Create object at specified position.

        Args:
            class_id: Object class ID
            texture_type: Texture type ('red', 'mixed', 'green')
            position: [x, y, z] position

        Returns:
            int: PyBullet body ID
        """
        obj_name = f"Object_{class_id + 1:02d}"
        obj_path = config.OBJECTS_DIR / obj_name / f"{obj_name}.STL"
        json_path = config.OBJECTS_DIR / obj_name / "mesh.json"

        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                mesh_data = json.load(f)
            v = mesh_data["vertices"]
            ind = mesh_data["indices"]

            temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, physicsClientId=self.client_id)
            temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=self.client_id)
            aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
            p.removeBody(temp_body, physicsClientId=self.client_id)

            size = np.array(aabb_max) - np.array(aabb_min)
            max_dim = np.max(size)
            target_size = 0.15 * config.OBJECT_SCALE_FACTOR
            scale = target_size / max_dim if max_dim > 0 else 1.0

            visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, meshScale=[scale, scale, scale], physicsClientId=self.client_id)
            collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, meshScale=[scale, scale, scale], physicsClientId=self.client_id)
        else:
            obj_path_short = config.get_short_path(obj_path)

            temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), physicsClientId=self.client_id)
            temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=self.client_id)
            aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
            p.removeBody(temp_body, physicsClientId=self.client_id)

            size = np.array(aabb_max) - np.array(aabb_min)
            max_dim = np.max(size)
            target_size = 0.15 * config.OBJECT_SCALE_FACTOR
            scale = target_size / max_dim if max_dim > 0 else 1.0

            visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), meshScale=[scale, scale, scale], physicsClientId=self.client_id)
            collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), meshScale=[scale, scale, scale], physicsClientId=self.client_id)

        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape, basePosition=position,
                                    physicsClientId=self.client_id)

        tex_id = self._load_texture(texture_type)
        p.changeVisualShape(
            body_id,
            -1,
            textureUniqueId=tex_id,
            specularColor=[0, 0, 0],
            physicsClientId=self.client_id
        )

        return body_id

    def _generate_obstacles_uniform(self, num_obstacles, object_positions):
        """Generate obstacles with uniform spatial distribution.

        Args:
            num_obstacles: Number of obstacles to generate
            object_positions: List of (position, class_id, radius) tuples
        """
        obstacle_positions = []

        for _ in range(num_obstacles):
            # Random obstacle dimensions (scaled by OBSTACLE_SCALE_FACTOR)
            half_extents = [
                random.uniform(0.01, 0.05) * config.OBSTACLE_SCALE_FACTOR,
                random.uniform(0.1, 0.2) * config.OBSTACLE_SCALE_FACTOR,
                random.uniform(0.1, 0.3) * config.OBSTACLE_SCALE_FACTOR
            ]
            # Obstacle radius (max half-extent)
            obstacle_radius = max(half_extents)

            for attempt in range(config.SCENE_MAX_PLACEMENT_ATTEMPTS):
                # Uniform distribution within same bounds as objects
                x = random.uniform(config.SCENE_BOUNDS_X_MIN, config.SCENE_BOUNDS_X_MAX)
                y = random.uniform(config.SCENE_BOUNDS_Y_MIN, config.SCENE_BOUNDS_Y_MAX)
                # Z coordinate: center must be at least half-height above ground + safety margin
                z_min = config.SCENE_BOUNDS_Z_MIN + half_extents[2] + 0.05  # 5cm safety margin
                z = random.uniform(z_min, config.SCENE_BOUNDS_Z_MAX)

                pos = [x, y, z]

                collision = False

                # Check collision with objects using their actual radii
                for obj_pos, obj_class, obj_radius in object_positions:
                    distance = np.linalg.norm(np.array(pos) - np.array(obj_pos))
                    min_safe_distance = obj_radius + obstacle_radius + 0.05  # 5cm safety margin
                    if distance < min_safe_distance:
                        collision = True
                        break

                # Check collision with other obstacles
                if not collision:
                    for obs_pos, obs_radius in obstacle_positions:
                        distance = np.linalg.norm(np.array(pos) - np.array(obs_pos))
                        min_safe_distance = obstacle_radius + obs_radius + 0.05
                        if distance < min_safe_distance:
                            collision = True
                            break

                # Check collision with robot base [0, 0, 0]
                if not collision:
                    base_distance = np.linalg.norm([pos[0], pos[1]])
                    if base_distance < 0.25:  # Safety margin around robot base
                        collision = True

                if not collision:
                    obstacle_positions.append((pos, obstacle_radius))
                    break
            else:
                # Failed to place obstacle
                print(f"Warning: Could not place obstacle {len(obstacle_positions)+1} after {config.SCENE_MAX_PLACEMENT_ATTEMPTS} attempts")
                continue

        # Create obstacles at validated positions
        for pos, radius in obstacle_positions:
            # Use pre-calculated dimensions (scaled by OBSTACLE_SCALE_FACTOR)
            half_extents = [
                random.uniform(0.01, 0.05) * config.OBSTACLE_SCALE_FACTOR,
                random.uniform(0.1, 0.2) * config.OBSTACLE_SCALE_FACTOR,
                random.uniform(0.1, 0.3) * config.OBSTACLE_SCALE_FACTOR
            ]

            yaw = random.uniform(0, 2 * np.pi)
            orn = p.getQuaternionFromEuler([0, 0, yaw])

            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client_id)
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)

            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                        baseVisualShapeIndex=visual_shape, basePosition=pos,
                                        baseOrientation=orn, physicsClientId=self.client_id)
            self.obstacles.append(body_id)

    def clear_scene(self):
        """Clear all objects and obstacles from scene."""
        for obj_id in self.target_objects:
            p.removeBody(obj_id, physicsClientId=self.client_id)
        self.target_objects.clear()
        self.target_objects_classes.clear()

        for obs_id in self.obstacles:
            p.removeBody(obs_id, physicsClientId=self.client_id)
        self.obstacles.clear()

    def generate_obstacles(self):
        """Legacy method for backward compatibility. Use generate_scene() instead."""
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
