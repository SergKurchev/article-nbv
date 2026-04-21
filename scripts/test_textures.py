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
    def __init__(self):
        self.client_id = p.connect(p.GUI)

        # Use short path to avoid Cyrillic issues
        pybullet_data_path = config.get_short_path(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Camera settings
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
        self.load_all_objects()

        # Show only first object
        self.update_visibility()

        # Apply initial texture
        self.apply_texture()

        # Update camera
        self.update_camera()

        print("\n=== Texture Tester ===")
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
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0.2],
                physicsClientId=self.client_id
            )

            self.objects.append(body_id)

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

        # Reset view (R)
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.camera_yaw = 45
            self.camera_pitch = -30
            self.camera_distance = 0.8
            self.update_camera()
            print("View reset")

        # Quit (Q)
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            return False

        return True

    def run(self):
        """Main loop."""
        try:
            while True:
                p.stepSimulation()

                if not self.handle_keys():
                    break

                time.sleep(1.0 / 60.0)

        finally:
            p.disconnect()
            print("\nTexture tester closed.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test textures on primitive shapes")
    parser.add_argument("--show-all", action="store_true", help="Show all shapes at once")
    args = parser.parse_args()

    if args.show_all:
        print("--show-all mode not yet implemented. Use interactive mode.")
        return

    tester = TextureTester()
    tester.run()


if __name__ == "__main__":
    main()
