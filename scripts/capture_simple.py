"""Capture screenshots using GUI mode with built-in primitives."""

import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def capture_with_gui():
    """Capture screenshots using GUI mode."""

    # Connect to PyBullet in GUI mode
    client_id = p.connect(p.GUI)

    # Use short path
    pybullet_data_path = config.get_short_path(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=client_id)

    # Disable gravity
    p.setGravity(0, 0, 0, physicsClientId=client_id)

    # Load plane
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    # Load textures
    texture_dir = config.DATA_DIR / "objects" / "textures"
    textures = {}
    for texture_type in ["red", "mixed", "green"]:
        texture_path = texture_dir / f"{texture_type}.png"
        texture_path_short = config.get_short_path(texture_path)
        textures[texture_type] = p.loadTexture(str(texture_path_short))
        print(f"Loaded texture: {texture_type}")

    # Create 3 spheres with different textures
    positions = [[-0.3, 0, 0.2], [0, 0, 0.2], [0.3, 0, 0.2]]
    texture_types = ["red", "mixed", "green"]

    for pos, tex_type in zip(positions, texture_types):
        # Create sphere using built-in shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 1, 1, 1],
            physicsClientId=client_id
        )

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            physicsClientId=client_id
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos,
            physicsClientId=client_id
        )

        # Apply texture
        p.changeVisualShape(
            body_id,
            -1,
            textureUniqueId=textures[tex_type],
            physicsClientId=client_id
        )

        print(f"Created sphere with {tex_type} texture at {pos}")

    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.2],
        physicsClientId=client_id
    )

    # Wait for rendering
    time.sleep(1)

    # Output directory
    output_dir = Path("texture_analysis")
    output_dir.mkdir(exist_ok=True)

    # Capture image
    width, height = 1280, 720
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.2],
        distance=0.8,
        yaw=45,
        pitch=-20,
        roll=0,
        upAxisIndex=2
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.01,
        farVal=10.0
    )

    img_data = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id
    )

    # Extract RGB
    rgb_array = np.array(img_data[2], dtype=np.uint8).reshape((height, width, 4))
    rgb_image = Image.fromarray(rgb_array[:, :, :3])

    # Save
    output_path = output_dir / "texture_rendering.png"
    rgb_image.save(output_path)
    print(f"\nScreenshot saved to: {output_path.absolute()}")

    # Keep window open for 3 seconds
    print("\nWindow will close in 3 seconds...")
    time.sleep(3)

    # Disconnect
    p.disconnect(physicsClientId=client_id)

    print("\nDone! Check the screenshot to verify:")
    print("1. Red sphere should be bright red")
    print("2. Green sphere should be bright green")
    print("3. Mixed sphere should show red-green gradient")
    print("4. No brown/muddy colors")


if __name__ == "__main__":
    capture_with_gui()
