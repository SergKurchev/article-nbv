"""Generate grid view of all textures using PyBullet primitives (with UV coordinates)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as p
import pybullet_data
import config
from src.vision.texture_generator import generate_all_textures


def capture_all_textures(gui_mode=False):
    """Capture all 8 primitive shapes x 3 textures in one image.

    Args:
        gui_mode: If True, opens interactive GUI viewer instead of saving screenshot
    """
    client_id = p.connect(p.GUI if gui_mode else p.DIRECT)

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

    def generate_spherical_uvs(vertices):
        """Generate spherical UV coordinates for mesh vertices."""
        uvs = []
        for v in vertices:
            x, y, z = v
            # Normalize to unit sphere
            length = np.sqrt(x*x + y*y + z*z)
            if length > 0:
                x, y, z = x/length, y/length, z/length

            # Spherical coordinates
            u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
            v = 0.5 - np.arcsin(y) / np.pi
            uvs.append([u, v])
        return uvs

    # Create grid: 4 columns x 2 rows
    spacing = 0.6  # Increased spacing to prevent overlap
    start_x = -0.9
    start_y = -0.3
    z_height = 0.3

    print("Loading 8 primitive objects from Object_01 to Object_08...")

    for i in range(8):
        obj_name = f"Object_{i + 1:02d}"
        json_path = config.OBJECTS_DIR / obj_name / "mesh.json"

        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping...")
            continue

        # Load mesh data
        import json
        with open(json_path, 'r') as f:
            mesh_data = json.load(f)

        vertices = mesh_data["vertices"]
        indices = mesh_data["indices"]

        # Generate UV coordinates
        uvs = generate_spherical_uvs(vertices)

        # Calculate scale to normalize size
        temp_col = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            physicsClientId=client_id
        )
        temp_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=temp_col,
            physicsClientId=client_id
        )
        aabb_min, aabb_max = p.getAABB(temp_body, physicsClientId=client_id)
        p.removeBody(temp_body, physicsClientId=client_id)

        size = np.array(aabb_max) - np.array(aabb_min)
        max_dim = np.max(size)
        target_size = 0.12
        scale = target_size / max_dim if max_dim > 0 else 1.0

        col = i % 4
        row = i // 4
        base_x = start_x + col * spacing
        base_y = start_y + row * spacing

        # Create 3 copies with different textures
        for tex_idx, texture_type in enumerate(["red", "mixed", "green"]):
            x_offset = (tex_idx - 1) * 0.18  # Increased offset
            position = [base_x + x_offset, base_y, z_height]

            # Create visual shape with UV coordinates
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                uvs=uvs,
                meshScale=[scale, scale, scale],
                physicsClientId=client_id
            )

            # Create collision shape
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                meshScale=[scale, scale, scale],
                physicsClientId=client_id
            )

            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                physicsClientId=client_id
            )

            # Apply texture and disable backface culling
            texture_id = textures[texture_type]
            p.changeVisualShape(
                body_id, -1,
                textureUniqueId=texture_id,
                specularColor=[0, 0, 0],
                flags=p.VISUAL_SHAPE_DOUBLE_SIDED,  # Render both sides of faces
                physicsClientId=client_id
            )

    print("Capturing screenshot...")
    width, height = 1920, 1080

    if gui_mode:
        # Interactive GUI mode
        print("\n=== Interactive GUI Mode ===")
        print("Controls:")
        print("  Mouse drag: Rotate camera")
        print("  Mouse wheel: Zoom in/out")
        print("  Arrow keys: Move camera target (Left/Right/Forward/Backward)")
        print("  [ / ]: Move camera target up/down")
        print("  Space: Reset camera view")
        print("  Q: Quit and save screenshot")
        print("\nPress Q to quit...")

        # Set initial camera view
        camera_distance = 2.5
        camera_yaw = 45
        camera_pitch = -30
        camera_target = [0, 0, 0.3]

        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target,
            physicsClientId=client_id
        )

        # Disable GUI panels for cleaner view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)

        # Wait for user to press Q with camera controls
        import time
        try:
            while True:
                # Check if connection is still alive
                if not p.isConnected(physicsClientId=client_id):
                    print("\nWindow closed by user")
                    return None

                keys = p.getKeyboardEvents(physicsClientId=client_id)

                # Quit
                if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
                    print("\nQuitting...")
                    break

                # Move target with arrow keys (XY plane)
                if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                    camera_target[0] -= 0.05
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)

                if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                    camera_target[0] += 0.05
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)

                if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                    camera_target[1] += 0.05
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)

                if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                    camera_target[1] -= 0.05
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)

                # Move target up/down with [ and ]
                if ord('[') in keys and keys[ord('[')] & p.KEY_IS_DOWN:
                    camera_target[2] -= 0.05
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)

                if ord(']') in keys and keys[ord(']')] & p.KEY_IS_DOWN:
                    camera_target[2] += 0.05
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)

                # Reset view with Space
                if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                    camera_distance = 2.5
                    camera_yaw = 45
                    camera_pitch = -30
                    camera_target = [0, 0, 0.3]
                    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target, physicsClientId=client_id)
                    print("Camera reset")

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return None

        # Get current camera view for screenshot
        debug_info = p.getDebugVisualizerCamera(physicsClientId=client_id)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=debug_info[11],
            distance=debug_info[10],
            yaw=debug_info[8],
            pitch=debug_info[9],
            roll=0,
            upAxisIndex=2
        )
    else:
        # Automatic screenshot mode
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

    # Capture with proper lighting
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
    print(f"\nNOTE: Using real primitive objects (Object_01 - Object_08) with procedural UV coordinates")

    p.disconnect(client_id)
    return screenshot_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate texture grid visualization")
    parser.add_argument("--gui", action="store_true", help="Open interactive GUI viewer")
    args = parser.parse_args()

    if args.gui:
        print("=== Interactive Texture Viewer ===\n")
    else:
        print("=== Generating All Textures Grid ===\n")

    capture_all_textures(gui_mode=args.gui)
