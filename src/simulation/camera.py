import pybullet as p
import numpy as np
import config

class Camera:
    def __init__(self, client_id):
        self.client_id = client_id

        self.width = config.IMAGE_SIZE
        self.height = config.IMAGE_SIZE
        self.fov = 60.0
        self.near = 0.1
        self.far = 10.0

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=config.CAMERA_POS,
            cameraTargetPosition=config.CAMERA_TARGET,
            cameraUpVector=config.CAMERA_UP,
            physicsClientId=self.client_id
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=1.0,
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=self.client_id
        )

        self.cam_position = None
        self.cam_rotation = None
        
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
            lightDirection=[0, 0, -1],
            lightColor=[1, 1, 1],
            lightDistance=100,
            shadow=0,
            lightAmbientCoeff=0.8,   # High ambient = preserve texture colors
            lightDiffuseCoeff=0.2,   # Low diffuse = reduce directional darkening
            lightSpecularCoeff=0.0,  # No specular
            physicsClientId=self.client_id
        )

        # RGB
        rgba = np.reshape(rgb, (h, w, 4)).astype(np.uint8)
        rgb_img = rgba[:, :, :3]

        # Depth linearization
        # depth buffer is [0, 1] non-linear. Linearize to [near, far]
        depth_img = self.far * self.near / (self.far - (self.far - self.near) * np.reshape(depth, (h, w)))

        # Segmentation mask
        seg_img = np.reshape(seg, (h, w)).astype(np.int32)

        return rgb_img, depth_img, seg_img

    def get_intrinsics(self):
        """Return camera intrinsics."""
        fov_rad = np.deg2rad(self.fov)
        fx = fy = self.width / (2.0 * np.tan(fov_rad / 2.0))
        cx = self.width / 2.0
        cy = self.height / 2.0

        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": self.width,
            "height": self.height,
            "near": self.near,
            "far": self.far,
            "fov_deg": self.fov
        }

    def get_rotation_quaternion(self):
        """Extract rotation quaternion from view matrix."""
        if self.cam_rotation is not None:
            return self.cam_rotation

        # Extract rotation from view matrix (inverse of camera transform)
        vm = np.array(self.view_matrix).reshape(4, 4)
        rot_mat = vm[:3, :3].T  # Transpose to get world-to-local

        # Convert rotation matrix to quaternion
        trace = np.trace(rot_mat)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
            y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
            z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
        else:
            if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
                w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
                x = 0.25 * s
                y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            elif rot_mat[1, 1] > rot_mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
                w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
                x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                y = 0.25 * s
                z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
                w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
                x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                z = 0.25 * s

        return [float(x), float(y), float(z), float(w)]
