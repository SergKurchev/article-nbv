import pybullet as p
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.simulation.camera import Camera

def calibrate():
    print("=== Coordinate System Calibration ===")
    client_id = p.connect(p.DIRECT)
    
    # 1. Place a tiny sphere at a known world position
    target_world_pos = np.array([0.5, 0.2, 0.3])
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])
    test_obj = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=target_world_pos)
    
    # 2. Setup camera at a known position looking at the target
    cam_pos = np.array([0.0, 0.0, 0.5])
    cam_target = target_world_pos
    cam_up = [0, 0, 1]
    
    camera = Camera(client_id)
    view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
    camera.view_matrix = view_matrix
    
    # 3. Capture image
    w, h, rgb, depth, seg = p.getCameraImage(
        camera.width, camera.height, camera.view_matrix, camera.projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )
    
    # Find the target object in the segmentation buffer
    depth_arr = np.reshape(depth, (h, w))
    seg_arr = np.reshape(seg, (h, w))
    target_pixels = np.where(seg_arr == test_obj)
    
    if len(target_pixels[0]) == 0:
        print("Error: Target object not visible in camera!")
        return
    
    # Get the center pixel of the target
    py, px = int(np.mean(target_pixels[0])), int(np.mean(target_pixels[1]))
    target_depth = depth_arr[py, px]
    
    # 4. Linearize depth
    near = camera.near
    far = camera.far
    z_linear = far * near / (far - (far - near) * target_depth)
    
    # 5. Run unproject logic for this pixel
    fx, fy, cx, cy = camera.get_intrinsics()["fx"], camera.get_intrinsics()["fy"], camera.get_intrinsics()["cx"], camera.get_intrinsics()["cy"]
    
    # Camera space coordinates (OpenGL convention)
    X_cam = (px - cx) * z_linear / fx
    Y_cam = -(py - cy) * z_linear / fy
    Z_cam = -z_linear
    pts_cam = np.array([X_cam, Y_cam, Z_cam])
    
    # Get rotation from view matrix
    vm = np.array(view_matrix).reshape(4, 4)
    rot_mat = vm[:3, :3] # Cam to World (no .T because reshape(4,4) on column-major list is already transposed)
    
    # Project to world
    projected_world_pos = rot_mat @ pts_cam + cam_pos
    
    print(f"Target World Pos:    {target_world_pos}")
    print(f"Projected World Pos: {projected_world_pos}")
    print(f"Difference:          {projected_world_pos - target_world_pos}")
    
    # Check if we need to flip anything
    if np.allclose(projected_world_pos, target_world_pos, atol=1e-3):
        print("\n[OK] Projection logic is CORRECT!")
    else:
        print("\n[FAIL] Discrepancy detected!")
        
        # Test alternative: what if Z_cam is positive?
        pts_cam_alt = np.array([X_cam, Y_cam, z_linear])
        proj_alt = rot_mat @ pts_cam_alt + cam_pos
        print(f"Alt (Z_cam positive): {proj_alt} (Diff: {proj_alt - target_world_pos})")
        
        # Test alternative: what if Y_cam is not flipped?
        Y_cam_alt = (py - cy) * z_linear / fy
        pts_cam_alt2 = np.array([X_cam, Y_cam_alt, -z_linear])
        proj_alt2 = rot_mat @ pts_cam_alt2 + cam_pos
        print(f"Alt (Y_cam not flipped): {proj_alt2} (Diff: {proj_alt2 - target_world_pos})")

    p.disconnect()

if __name__ == "__main__":
    calibrate()
