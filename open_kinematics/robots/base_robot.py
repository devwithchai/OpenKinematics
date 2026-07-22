import numpy as np
from open_kinematics.solvers.dh_solver import forward_kinematics
from open_kinematics.exceptions import JointLimitViolation, InvalidDHParameters


class BaseRobot:
    """
    Base class representing a generic serial robot manipulator.

    Stores the robot's Denavit-Hartenberg (DH) parameters, joint configuration, and joint limits.
    Provides shared functionality for forward kinematics and joint-limit validation
    that is reused by all concrete robot models.
    """
    def __init__(self, dh_parameters: list[tuple], joint_types: list[str], joint_limits: list[tuple]):
        """
        Initialize a serial robot model using Denavit-Hartenberg parameters.

        :param dh_parameters: List of DH parameter tuples, each of the form (theta_offset, d, r, alpha).
        :param joint_types: List describing the type of each joint ("revolute" or "prismatic").
        :param joint_limits: List of (minimum, maximum) joint limits corresponding to each joint.
        :return: None.
        :raises TypeError: If ``dh_parameters`` is not a list or tuple.
        :raises ValueError: If any DH parameter tuple does not contain exactly four values.
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
        Compute forward kinematics for the robot using its stored Denavit-Hartenberg parameters.

        :param joint_values: Joint values supplied in the same order as the robot's joint definitions.
        :param return_all: If True, return the cumulative homogeneous transformation matrix after every joint.
            Otherwise, return only the final end-effector transformation.
        :return: The end-effector homogeneous transformation matrix when ``return_all`` is False,
            otherwise a list containing every intermediate transformation matrix.
        :raises TypeError: If ``joint_values`` is not a list, tuple, or NumPy array.
        :raises ValueError: If the number of supplied joint values does not match the number of robot joints.
        :raises JointLimitViolation: If any joint value exceeds its configured limits.
        :raises InvalidDHParameters: If an unsupported joint type is encountered while updating the DH parameter table.
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
        """
        Validate that every joint value lies within its configured limits.

        :param joint_values: Sequence of joint values to validate.
        :return: None.
        :raises JointLimitViolation: If any joint value exceeds its configured minimum or maximum limit.
        """

        if self.joint_limits is not None:
            for i, (value, (low, high)) in enumerate(zip(joint_values, self.joint_limits)):
                if value > high or value < low:
                    raise JointLimitViolation(f"Joint {i} value {value} exceeds limits ({low}, {high})")
