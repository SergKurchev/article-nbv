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
        # UR3: revolute joints are 1-6, end-effector is tool0 (joint index 8)
        self.ee_index = self.num_joints - 1
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            jname = info[1]
            if b"tool0" in jname or b"flange-tool0" in jname:
                self.ee_index = i
                break
        # UR3 revolute joints are at indices 1..6 (index 0 is the fixed base joint)
        self._revolute = [
            i for i in range(self.num_joints)
            if p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)[2] == p.JOINT_REVOLUTE
        ]

    def reset(self):
        # UR3 home: elbow-up configuration
        home_pos = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        for idx, joint_idx in enumerate(self._revolute[:len(home_pos)]):
            p.resetJointState(self.robot_id, joint_idx, home_pos[idx], physicsClientId=self.client_id)

    def get_joint_states(self):
        positions = []
        for joint_idx in self._revolute:
            state = p.getJointState(self.robot_id, joint_idx, physicsClientId=self.client_id)
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
            
        # Apply IK result to revolute joints only (teleport + motor control)
        for idx, joint_idx in enumerate(self._revolute[:len(joint_poses)]):
            p.resetJointState(self.robot_id, joint_idx, joint_poses[idx], physicsClientId=self.client_id)
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[idx],
                force=1000.0,
                physicsClientId=self.client_id
            )
