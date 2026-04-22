"""Capture multiple views of textured objects for analysis."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pybullet as p
import pybullet_data
import config
from src.vision.texture_generator import generate_all_textures

def capture_views():
    """Capture 3 views of textured objects."""
    client_id = p.connect(p.DIRECT)

    # Setup
    pybullet_data_path = config.get_short_path(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=client_id)
    p.setGravity(0, 0, 0, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    # Generate textures
    print("Generating textures...")
    generate_all_textures()

    # Load textures
    texture_dir = config.DATA_DIR / "objects" / "textures"
    textures = {}
    for texture_type in ["red", "mixed", "green"]:
        texture_path = texture_dir / f"{texture_type}.png"
        texture_path_short = config.get_short_path(texture_path)
        textures[texture_type] = p.loadTexture(str(texture_path_short), physicsClientId=client_id)

    # Create 3 spheres with different textures
    spacing = 0.5
    for i, (texture_type, texture_id) in enumerate(textures.items()):
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.15,
            physicsClientId=client_id
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.15,
            physicsClientId=client_id
        )

        position = [(i - 1) * spacing, 0, 0.3]
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=client_id
        )

        # Apply texture with no specular
        p.changeVisualShape(
            body_id,
            -1,
            textureUniqueId=texture_id,
            rgbaColor=[1, 1, 1, 1],  # White tint to preserve texture colors
            specularColor=[0, 0, 0],
            physicsClientId=client_id
        )

    # Capture from 3 different angles
    width, height = 1280, 720
    yaw_angles = [0, 45, 90]

    screenshots = []
    for idx, yaw in enumerate(yaw_angles):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.3],
            distance=1.5,
            yaw=yaw,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )

        img_arr = p.getCameraImage(
            width,
            height,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            lightDirection=[0, 0, -1],
            lightColor=[1, 1, 1],
            lightDistance=100,
            shadow=0,
            lightAmbientCoeff=0.8,   # High ambient = less directional darkening
            lightDiffuseCoeff=0.2,   # Low diffuse = preserve texture colors
            lightSpecularCoeff=0.0,
            physicsClientId=client_id
        )

        # Extract RGB
        rgb_array = np.array(img_arr[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

        # Save screenshot
        screenshot_path = Path(f"texture_view_{idx+1}_yaw{yaw}.png")
        Image.fromarray(rgb_array).save(screenshot_path)
        screenshots.append(screenshot_path)
        print(f"Saved: {screenshot_path}")

    p.disconnect(client_id)

    return screenshots

if __name__ == "__main__":
    print("Capturing texture views...")
    screenshots = capture_views()
    print(f"\nCaptured {len(screenshots)} views:")
    for path in screenshots:
        print(f"  - {path}")
