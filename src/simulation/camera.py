import pybullet as p
import numpy as np
import config

class Camera:
    def __init__(self, client_id):
        self.client_id = client_id
        
        self.width = config.IMAGE_SIZE
        self.height = config.IMAGE_SIZE
        
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=config.CAMERA_POS,
            cameraTargetPosition=config.CAMERA_TARGET,
            cameraUpVector=config.CAMERA_UP,
            physicsClientId=self.client_id
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0,
            physicsClientId=self.client_id
        )
        
    def get_image(self, cam_pos=None, cam_orn=None):
        if cam_pos is not None and cam_orn is not None:
            # orn is quaternion [x,y,z,w]
            rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
            
            # Kuka iiwa end effector: +Z usually points forward along the flange
            forward = rot_matrix[:, 2]
            up = rot_matrix[:, 0] # Local X as UP (if it looks sideways we can flip to Y later, standard depends on specific joint)
            
            # Small forward offset to prevent seeing the inside of the gripper/flange
            eye_pos = np.array(cam_pos) + forward * 0.05
            target = eye_pos + forward * 1.0 # arbitrary point in front to look at
            
            self.view_matrix = p.computeViewMatrix(
                cameraEyePosition=eye_pos.tolist(),
                cameraTargetPosition=target.tolist(),
                cameraUpVector=up.tolist(),
                physicsClientId=self.client_id
            )

        w, h, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id
        )
        
        # RGB
        rgba = np.reshape(rgb, (h, w, 4)).astype(np.uint8)
        rgb_img = rgba[:, :, :3]
        
        # Depth linearization
        # depth buffer is [0, 1] non-linear. Linearize to [near, far]
        near = 0.1
        far = 10.0
        depth_img = far * near / (far - (far - near) * np.reshape(depth, (h, w)))
        
        # Segmentation mask
        seg_img = np.reshape(seg, (h, w)).astype(np.int32)
        
        return rgb_img, depth_img, seg_img
