"""Interactive texture testing tool for NBV project.

Visualizes textures on primitive shapes in PyBullet.

Controls:
- 1/2/3: Switch texture (red/mixed/green)
- Space: Cycle through shapes
- Arrow keys: Rotate camera
- R: Reset view
- Q: Quit
"""

import pybullet as p
import pybullet_data
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.vision.texture_generator import generate_all_textures


class TextureTester:
    def __init__(self, show_all=False):
        self.client_id = p.connect(p.GUI)
        self.show_all = show_all

        # Use short path to avoid Cyrillic issues
        pybullet_data_path = config.get_short_path(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=self.client_id)

        # Disable gravity so objects stay in place
        p.setGravity(0, 0, 0, physicsClientId=self.client_id)

        # Disable lighting to preserve texture colors
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client_id)

        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Camera settings
        if show_all:
            # Wide view for grid
            self.camera_distance = 2.5
            self.camera_yaw = 45
            self.camera_pitch = -30
            self.camera_target = [0, 0, 0.3]
        else:
            # Close view for single object
            self.camera_distance = 0.8
            self.camera_yaw = 45
            self.camera_pitch = -30
            self.camera_target = [0, 0, 0.2]

        # Generate textures if they don't exist
        self.texture_dir = config.DATA_DIR / "objects" / "textures"
        if not (self.texture_dir / "red.png").exists():
            print("Generating textures...")
            generate_all_textures()

        # Load textures
        self.textures = {}
        for texture_type in ["red", "mixed", "green"]:
            texture_path = self.texture_dir / f"{texture_type}.png"
            texture_path_short = config.get_short_path(texture_path)
            self.textures[texture_type] = p.loadTexture(str(texture_path_short))

        self.texture_types = ["red", "mixed", "green"]
        self.current_texture_idx = 0

        # Object list (8 primitives)
        self.objects = []
        self.object_names = [
            "cube", "sphere", "cylinder", "cone",
            "capsule", "torus", "hourglass", "prism"
        ]
        self.current_object_idx = 0

        # Load all objects
        if show_all:
            self.load_all_objects_grid()
        else:
            self.load_all_objects()
            # Show only first object
            self.update_visibility()
            # Apply initial texture
            self.apply_texture()

        # Update camera
        self.update_camera()

        if show_all:
            print("\n=== Texture Tester (Grid Mode) ===")
            print("All 8 objects displayed in 4x2 grid")
            print("Each row shows same shape with 3 textures (red, mixed, green)")
            print("\nControls:")
            print("  Arrow keys: Rotate camera")
            print("  +/-: Zoom in/out")
            print("  R: Reset view")
            print("  Q: Quit")
        else:
            print("\n=== Texture Tester (Interactive Mode) ===")
            print("Controls:")
            print("  1/2/3: Switch texture (red/mixed/green)")
            print("  Space: Cycle through shapes")
            print("  Arrow keys: Rotate camera")
            print("  R: Reset view")
            print("  Q: Quit")
            print(f"\nCurrent shape: {self.object_names[self.current_object_idx]}")
            print(f"Current texture: {self.texture_types[self.current_texture_idx]}")

    def load_all_objects(self):
        """Load all 8 primitive objects."""
        for i in range(8):
            obj_name = f"Object_{i + 1:02d}"
            obj_path = config.OBJECTS_DIR / obj_name / f"{obj_name}.STL"

            # Check for JSON mesh data
            json_path = config.OBJECTS_DIR / obj_name / "mesh.json"

            if json_path.exists():
                import json
                import numpy as np

                with open(json_path, 'r') as f:
                    mesh_data = json.load(f)

                v = mesh_data["vertices"]
                ind = mesh_data["indices"]

                # Measure scale
                temp_col = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    vertices=v,
                    indices=ind,
                    physicsClientId=self.client_id
                )
                temp_body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=temp_col,
                    physicsClientId=self.client_id
                )
                aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
                p.removeBody(temp_body, physicsClientId=self.client_id)

                size = np.array(aabb_max) - np.array(aabb_min)
                max_dim = np.max(size)
                target_size = 0.15
                scale = target_size / max_dim if max_dim > 0 else 1.0

                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    vertices=v,
                    indices=ind,
                    meshScale=[scale, scale, scale],
                    physicsClientId=self.client_id
                )
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    vertices=v,
                    indices=ind,
                    meshScale=[scale, scale, scale],
                    physicsClientId=self.client_id
                )
            else:
                # Load from STL
                obj_path_short = config.get_short_path(obj_path)

                temp_col = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(obj_path_short),
                    physicsClientId=self.client_id
                )
                temp_body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=temp_col,
                    physicsClientId=self.client_id
                )
                aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
                p.removeBody(temp_body, physicsClientId=self.client_id)

                size = np.array(aabb_max) - np.array(aabb_min)
                max_dim = np.max(size)
                target_size = 0.15
                scale = target_size / max_dim if max_dim > 0 else 1.0

                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(obj_path_short),
                    meshScale=[scale, scale, scale],
                    physicsClientId=self.client_id
                )
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(obj_path_short),
                    meshScale=[scale, scale, scale],
                    physicsClientId=self.client_id
                )

            # Create body
            body_id = p.createMultiBody(
                baseMass=0,  # Static object (no gravity)
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0.2],
                physicsClientId=self.client_id
            )

            self.objects.append(body_id)

    def load_all_objects_grid(self):
        """Load all 8 objects in a 4x2 grid with all 3 textures."""
        spacing = 0.5  # Distance between objects
        start_x = -0.75  # Center the grid
        start_y = -0.25
        z_height = 0.3

        grid_objects = []
        object_count = 0

        for i in range(8):
            obj_name = f"Object_{i + 1:02d}"
            obj_path = config.OBJECTS_DIR / obj_name / f"{obj_name}.STL"
            json_path = config.OBJECTS_DIR / obj_name / "mesh.json"

            # Calculate grid position (4 columns x 2 rows)
            col = i % 4
            row = i // 4
            base_x = start_x + col * spacing
            base_y = start_y + row * spacing

            # Create 3 copies with different textures (red, mixed, green)
            for tex_idx, texture_type in enumerate(["red", "mixed", "green"]):
                # Load mesh data for each object separately
                if json_path.exists():
                    import json
                    import numpy as np

                    with open(json_path, 'r') as f:
                        mesh_data = json.load(f)

                    v = mesh_data["vertices"]
                    ind = mesh_data["indices"]

                    # Measure scale
                    temp_col = p.createCollisionShape(
                        shapeType=p.GEOM_MESH,
                        vertices=v,
                        indices=ind,
                        physicsClientId=self.client_id
                    )
                    temp_body = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=temp_col,
                        physicsClientId=self.client_id
                    )
                    aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
                    p.removeBody(temp_body, physicsClientId=self.client_id)

                    size = np.array(aabb_max) - np.array(aabb_min)
                    max_dim = np.max(size)
                    target_size = 0.12  # Smaller for grid view
                    scale = target_size / max_dim if max_dim > 0 else 1.0

                    # Create unique shapes for this object
                    visual_shape = p.createVisualShape(
                        shapeType=p.GEOM_MESH,
                        vertices=v,
                        indices=ind,
                        meshScale=[scale, scale, scale],
                        physicsClientId=self.client_id
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_MESH,
                        vertices=v,
                        indices=ind,
                        meshScale=[scale, scale, scale],
                        physicsClientId=self.client_id
                    )
                else:
                    # Load from STL
                    obj_path_short = config.get_short_path(obj_path)

                    temp_col = p.createCollisionShape(
                        shapeType=p.GEOM_MESH,
                        fileName=str(obj_path_short),
                        physicsClientId=self.client_id
                    )
                    temp_body = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=temp_col,
                        physicsClientId=self.client_id
                    )
                    aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=self.client_id)
                    p.removeBody(temp_body, physicsClientId=self.client_id)

                    size = np.array(aabb_max) - np.array(aabb_min)
                    max_dim = np.max(size)
                    target_size = 0.12
                    scale = target_size / max_dim if max_dim > 0 else 1.0

                    # Create unique shapes for this object
                    visual_shape = p.createVisualShape(
                        shapeType=p.GEOM_MESH,
                        fileName=str(obj_path_short),
                        meshScale=[scale, scale, scale],
                        physicsClientId=self.client_id
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_MESH,
                        fileName=str(obj_path_short),
                        meshScale=[scale, scale, scale],
                        physicsClientId=self.client_id
                    )

                # Position: spread textures horizontally within each cell
                x_offset = (tex_idx - 1) * 0.15  # -0.15, 0, +0.15
                position = [base_x + x_offset, base_y, z_height]

                # Create body
                body_id = p.createMultiBody(
                    baseMass=0,  # Static object
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    physicsClientId=self.client_id
                )

                print(f"Created body {body_id}: {self.object_names[i]} with {texture_type} texture at {position}")

                # Apply texture
                texture_id = self.textures[texture_type]
                p.changeVisualShape(
                    body_id,
                    -1,
                    textureUniqueId=texture_id,
                    rgbaColor=[1, 1, 1, 1],  # White tint to preserve texture colors
                    specularColor=[0, 0, 0],  # No specular to preserve true colors
                    physicsClientId=self.client_id
                )

                grid_objects.append({
                    "body_id": body_id,
                    "object_name": self.object_names[i],
                    "texture_type": texture_type,
                    "position": position
                })

                object_count += 1

        self.grid_objects = grid_objects
        print(f"\n[OK] Loaded {object_count} objects in grid (8 shapes x 3 textures)")
        print(f"[OK] Grid objects list has {len(self.grid_objects)} entries")

        # Verify all objects still exist
        print("\nVerifying objects exist:")
        for i, obj_info in enumerate(self.grid_objects[:3]):  # Check first 3
            body_id = obj_info["body_id"]
            try:
                pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
                print(f"  [OK] Body {body_id} exists at {pos}")
            except:
                print(f"  [ERROR] Body {body_id} NOT FOUND!")

    def update_visibility(self):
        """Show only current object, hide others."""
        for i, obj_id in enumerate(self.objects):
            if i == self.current_object_idx:
                # Show object
                p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])
            else:
                # Hide object
                p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 0])

    def apply_texture(self):
        """Apply current texture to current object."""
        texture_type = self.texture_types[self.current_texture_idx]
        texture_id = self.textures[texture_type]
        obj_id = self.objects[self.current_object_idx]

        p.changeVisualShape(
            obj_id,
            -1,
            textureUniqueId=texture_id,
            rgbaColor=[1, 1, 1, 1],  # White tint to preserve texture colors
            physicsClientId=self.client_id
        )

    def update_camera(self):
        """Update camera view."""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target,
            physicsClientId=self.client_id
        )

    def handle_keys(self):
        """Handle keyboard input."""
        keys = p.getKeyboardEvents()

        # Common controls for both modes
        # Camera rotation (Arrow keys)
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            self.camera_yaw -= 2
            self.update_camera()

        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            self.camera_yaw += 2
            self.update_camera()

        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            self.camera_pitch = min(self.camera_pitch + 2, 89)
            self.update_camera()

        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            self.camera_pitch = max(self.camera_pitch - 2, -89)
            self.update_camera()

        # Zoom (+ and -)
        if ord('+') in keys and keys[ord('+')] & p.KEY_IS_DOWN:
            self.camera_distance = max(0.3, self.camera_distance - 0.1)
            self.update_camera()

        if ord('-') in keys and keys[ord('-')] & p.KEY_IS_DOWN:
            self.camera_distance = min(5.0, self.camera_distance + 0.1)
            self.update_camera()

        # Reset view (R)
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            if self.show_all:
                self.camera_yaw = 45
                self.camera_pitch = -30
                self.camera_distance = 2.5
            else:
                self.camera_yaw = 45
                self.camera_pitch = -30
                self.camera_distance = 0.8
            self.update_camera()
            print("View reset")

        # Quit (Q)
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            return False

        # Interactive mode controls (only if not show_all)
        if not self.show_all:
            # Texture switching (1/2/3)
            if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
                self.current_texture_idx = 0
                self.apply_texture()
                print(f"Texture: {self.texture_types[self.current_texture_idx]}")

            if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
                self.current_texture_idx = 1
                self.apply_texture()
                print(f"Texture: {self.texture_types[self.current_texture_idx]}")

            if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
                self.current_texture_idx = 2
                self.apply_texture()
                print(f"Texture: {self.texture_types[self.current_texture_idx]}")

            # Shape cycling (Space)
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                self.current_object_idx = (self.current_object_idx + 1) % len(self.objects)
                self.update_visibility()
                self.apply_texture()
                print(f"Shape: {self.object_names[self.current_object_idx]}")

        return True

    def run(self):
        """Main loop."""
        try:
            print("\nViewer is running. Press Q to quit.")
            while True:
                # No physics simulation needed (objects are static)
                time.sleep(1.0 / 60.0)

                if not self.handle_keys():
                    break

        finally:
            p.disconnect()
            print("\nTexture tester closed.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test textures on primitive shapes")
    parser.add_argument("--show-all", action="store_true", help="Show all shapes in grid layout")
    args = parser.parse_args()

    tester = TextureTester(show_all=args.show_all)
    tester.run()


if __name__ == "__main__":
    main()
