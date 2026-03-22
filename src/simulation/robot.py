import pybullet as p
import numpy as np
import config
from src.utils.math_utils import normalize_quaternion

class Robot:
    def __init__(self, client_id, robot_id):
        self.client_id = client_id
        self.robot_id = robot_id
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        
        # Identify end effector (often the last link before fingers, assume index 7 for 7DOF Kinova j2s7s300)
        # We need to find the correct end-effector index dynamically
        self.ee_index = self.num_joints - 1
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if b"end_effector" in info[1] or b"ee_link" in info[1] or b"j2s7s300_end_effector" in info[1] or b"lbr_iiwa_link_7" in info[1]:
                self.ee_index = i
                break
                
    def reset(self):
        # Reset joints to home position
        home_pos = [0, 0, 0, -1.57, 0, 1.57, 0] # Safe 7DOF home (e.g. for Kuka)
        for i in range(min(7, self.num_joints)):
            p.resetJointState(self.robot_id, i, home_pos[i], physicsClientId=self.client_id)
            
    def get_joint_states(self):
        # We only care about the arm joints (0 to 6)
        positions = []
        for i in range(min(7, self.num_joints)):
            state = p.getJointState(self.robot_id, i, physicsClientId=self.client_id)
            positions.append(state[0])
        return np.array(positions, dtype=np.float32)

    def get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_index, physicsClientId=self.client_id)
        pos = state[4]
        orn = state[5]
        return np.array(pos), np.array(orn)
        
    def apply_action(self, target_pos, target_orn=None):
        """
        Move EE to target pos/orn via IK.
        target_orn should be a quaternion.
        """
        if target_orn is not None:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.ee_index, target_pos, target_orn,
                physicsClientId=self.client_id
            )
        else:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.ee_index, target_pos,
                physicsClientId=self.client_id
            )
            
        # apply motor control
        # apply motor control and instant teleportation to fix momentum lag
        for i in range(len(joint_poses)):
            p.resetJointState(self.robot_id, i, joint_poses[i], physicsClientId=self.client_id)
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=1000.0, # High force
                physicsClientId=self.client_id
            )
