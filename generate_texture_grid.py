"""Generate grid view of all textures with proper lighting."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pybullet as p
import pybullet_data
import config
from src.vision.texture_generator import generate_all_textures

def capture_all_textures():
    """Capture all 8 shapes × 3 textures in one image."""
    client_id = p.connect(p.DIRECT)

    pybullet_data_path = config.get_short_path(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=client_id)
    p.setGravity(0, 0, 0, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    print("Generating textures...")
    generate_all_textures()

    # Load textures
    texture_dir = config.DATA_DIR / "objects" / "textures"
    textures = {}
    for texture_type in ["red", "mixed", "green"]:
        texture_path = texture_dir / f"{texture_type}.png"
        texture_path_short = config.get_short_path(texture_path)
        textures[texture_type] = p.loadTexture(str(texture_path_short), physicsClientId=client_id)

    # Load all 8 primitive objects
    object_names = ["cube", "sphere", "cylinder", "cone", "capsule", "torus", "hourglass", "prism"]

    # Create grid: 4 columns × 2 rows
    spacing = 0.5
    start_x = -0.75
    start_y = -0.25
    z_height = 0.3

    print("Creating 24 objects (8 shapes x 3 textures)...")
    for i in range(8):
        obj_name = f"Object_{i + 1:02d}"
        obj_path = config.OBJECTS_DIR / obj_name / f"{obj_name}.STL"
        json_path = config.OBJECTS_DIR / obj_name / "mesh.json"

        col = i % 4
        row = i // 4
        base_x = start_x + col * spacing
        base_y = start_y + row * spacing

        # Create 3 copies with different textures
        for tex_idx, texture_type in enumerate(["red", "mixed", "green"]):
            if json_path.exists():
                import json
                with open(json_path, 'r') as f:
                    mesh_data = json.load(f)
                v = mesh_data["vertices"]
                ind = mesh_data["indices"]

                temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind, physicsClientId=client_id)
                temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=client_id)
                aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=client_id)
                p.removeBody(temp_body, physicsClientId=client_id)

                size = np.array(aabb_max) - np.array(aabb_min)
                max_dim = np.max(size)
                target_size = 0.12
                scale = target_size / max_dim if max_dim > 0 else 1.0

                visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind,
                                                  meshScale=[scale, scale, scale], physicsClientId=client_id)
                collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=v, indices=ind,
                                                        meshScale=[scale, scale, scale], physicsClientId=client_id)
            else:
                obj_path_short = config.get_short_path(obj_path)
                temp_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short), physicsClientId=client_id)
                temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=temp_col, physicsClientId=client_id)
                aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=client_id)
                p.removeBody(temp_body, physicsClientId=client_id)

                size = np.array(aabb_max) - np.array(aabb_min)
                max_dim = np.max(size)
                target_size = 0.12
                scale = target_size / max_dim if max_dim > 0 else 1.0

                visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short),
                                                  meshScale=[scale, scale, scale], physicsClientId=client_id)
                collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(obj_path_short),
                                                        meshScale=[scale, scale, scale], physicsClientId=client_id)

            x_offset = (tex_idx - 1) * 0.15
            position = [base_x + x_offset, base_y, z_height]

            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                       baseVisualShapeIndex=visual_shape, basePosition=position,
                                       physicsClientId=client_id)

            # Apply texture (Version 2 settings)
            texture_id = textures[texture_type]
            p.changeVisualShape(body_id, -1, textureUniqueId=texture_id,
                              specularColor=[0, 0, 0], physicsClientId=client_id)

    print("Capturing screenshot...")
    width, height = 1920, 1080

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.3],
        distance=2.5,
        yaw=45,
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

    # Version 2 lighting (best results)
    img_arr = p.getCameraImage(
        width, height, view_matrix, projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        lightDirection=[0, 0, -1],
        lightColor=[1, 1, 1],
        lightDistance=100,
        shadow=0,
        lightAmbientCoeff=0.8,
        lightDiffuseCoeff=0.2,
        lightSpecularCoeff=0.0,
        physicsClientId=client_id
    )

    rgb_array = np.array(img_arr[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    screenshot_path = Path("all_textures_grid.png")
    Image.fromarray(rgb_array).save(screenshot_path)
    print(f"\nSaved: {screenshot_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Objects: 24 (8 shapes x 3 textures)")
    print(f"  Layout: 4 columns x 2 rows")
    print(f"  Each shape shows: red (left), mixed (center), green (right)")

    p.disconnect(client_id)
    return screenshot_path

if __name__ == "__main__":
    print("=== Generating All Textures Grid ===\n")
    capture_all_textures()
