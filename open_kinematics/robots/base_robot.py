import numpy as np
from open_kinematics.solvers.dh_solver import forward_kinematics
from open_kinematics.exceptions import JointLimitViolation, InvalidDHParameters


class BaseRobot:
    def __init__(self, dh_parameters: list[tuple], joint_types: list[str], joint_limits: list[tuple]):
        """
        Initialize a serial robot model using DH parameters.
        :param dh_parameters: List of DH parameter tuples (theta_offset, d, r, alpha)
        """

        if not isinstance(dh_parameters, (list, tuple)):
            raise TypeError("dh_parameters must be a list or tuple.")

        for parameters in dh_parameters:
            if len(parameters) != 4:
                raise ValueError("Each DH parameter set must contain 4 values.")

        self.dh_parameters = dh_parameters
        self.num_joints = len(dh_parameters)
        self.joint_types = joint_types
        self.joint_limits = joint_limits

    def forward_kinematics(self, joint_values: list[float], return_all=False):
        """
        Compute forward kinematics using the robot's stored DH parameters.
        :param return_all:
        :param joint_values: List of joint angles (radians)
        :return: 4x4 homogeneous transformation matrix
        """

        if not isinstance(joint_values, (list, tuple, np.ndarray)):
            raise TypeError("joint_values must be a list, tuple or numpy.ndarray.")

        if len(joint_values) != self.num_joints:
            raise ValueError("Number of joint values must match number of joints.")

        self.check_joint_limits(joint_values)

        updated_dh_table = []
        for (theta_offset, d, r, alpha), joint, joint_type in zip(self.dh_parameters, joint_values, self.joint_types):
            if joint_type == 'revolute':
                updated_dh_table.append((theta_offset+joint, d, r, alpha))
            elif joint_type == 'prismatic':
                updated_dh_table.append((theta_offset, d+joint, r, alpha))
            else:
                raise InvalidDHParameters
        return forward_kinematics(updated_dh_table, return_all=return_all)

    def check_joint_limits(self, joint_values):
        if self.joint_limits is not None:
            for i, (value, (low, high)) in enumerate(zip(joint_values, self.joint_limits)):
                if value > high or value < low:
                    raise JointLimitViolation(f"Joint {i} value {value} exceeds limits ({low}, {high})")