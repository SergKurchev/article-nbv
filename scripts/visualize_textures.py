"""Quick texture visualization script - saves screenshots of all textures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as p
import pybullet_data
import time
import config


def visualize_all_textures():
    """Generate screenshots of all texture variants on a sphere."""
    # Connect to PyBullet with GUI for better rendering
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Setup scene
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Camera settings
    camera_distance = 1.5
    camera_yaw = 45
    camera_pitch = -30
    camera_target = [0, 0, 0.5]

    # Use primitive sphere instead of mesh
    textures = ["red", "mixed", "green"]
    output_dir = config.BASE_DIR / "texture_previews"
    output_dir.mkdir(exist_ok=True)

    print("\n=== Generating Texture Previews ===")

    for texture_name in textures:
        print(f"\n[{texture_name}]")

        # Load texture
        texture_path = config.DATA_DIR / "objects" / "textures" / f"{texture_name}.png"

        if not texture_path.exists():
            print(f"  Texture not found: {texture_path}")
            continue

        # Create sphere with texture
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[1, 1, 1, 1]
        )

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.3
        )

        obj_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0.5]
        )

        # Apply texture
        texture_id = p.loadTexture(str(texture_path))
        p.changeVisualShape(obj_id, -1, textureUniqueId=texture_id)

        # Reset camera
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )

        # Wait for rendering
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)

        # Take screenshot
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100
        )

        width, height = 512, 512
        img = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Save screenshot
        import numpy as np
        from PIL import Image
        rgb_array = np.array(img[2]).reshape(height, width, 4)[:, :, :3]
        image = Image.fromarray(rgb_array, 'RGB')
        output_path = output_dir / f"{texture_name}_preview.png"
        image.save(output_path)
        print(f"  Saved: {output_path}")

        # Remove object for next iteration
        p.removeBody(obj_id)

    print("\n=== Complete ===")
    print(f"Previews saved to: {output_dir}")
    print("\nClose the PyBullet window to exit.")

    # Keep window open
    while True:
        p.stepSimulation()
        time.sleep(0.01)


if __name__ == "__main__":
    visualize_all_textures()
