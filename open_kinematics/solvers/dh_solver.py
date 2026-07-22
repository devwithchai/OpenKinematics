import numpy as np
from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.solvers.base_solver import BaseSolver

def dh_transform(theta: float, d: float, r: float, alpha: float) -> np.ndarray:
    """
    Compute the standard Denavit-Hartenberg (DH) homogeneous transformation matrix.

    :param theta: Joint angle (rotation about the Z-axis) in radians.
    :param d: Link offset (translation along the Z-axis).
    :param r: Link length (translation along the X-axis).
    :param alpha: Link twist (rotation about the X-axis), in radians.
    :return: A 4x4 homogeneous transformation matrix representing the transformation between two consecutive coordinate frames.
    :raises TypeError: If any DH parameter is not numeric.
    """

    for value in (theta, d, r, alpha):
        if not isinstance(value, (int, float)):
            raise TypeError("All DH parameters must be numeric.")

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca, st * sa, r * ct],
        [st, ct * ca, -ct * sa, r * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return T

def forward_kinematics(dh_table: list[tuple], return_all: bool=False):
    """
    Compute the forward kinematics for a serial manipulator using (DH) convention.

    :param dh_table: Sequence of DH parameter dictionaries describing the robot.
    :param return_all: If True, return every intermediate transformation matrix.
        Otherwise, return only the end-effector transformation.
    :return: The end-effector homogeneous transformation matrix when ``return_all`` is False.
        Otherwise, a list containing the cumulative homogeneous transformation matrix after each joint.
    :raises TypeError: If any DH parameter is not numeric.
    :raises ValueError: If DH parameters are not 4.
    """

    if not isinstance(dh_table, (list, tuple)):
        raise TypeError("dh_table must be a list or tuple of DH parameter tuples.")

    T_total = identity_matrix(4)
    transforms = []

    for parameters in dh_table:
        if len(parameters) != 4:
            raise ValueError("Each DH entry must contain 4 parameters.")

        T = dh_transform(*parameters)
        T_total = T_total @ T
        transforms.append(T_total.copy())

    if return_all:
        return transforms

    return T_total


class DHSolver(BaseSolver):

    def forward(self, dh_table, joint_values) -> np.ndarray:
        """
        Compute forward kinematics using the Denavit-Hartenberg convention.

        :param dh_table: DH parameter table describing the robot geometry.
        :param joint_values: Joint values added to the corresponding DH joint-angle offsets.
        :return: End-effector pose as a 4x4 homogeneous transformation matrix.
        """
        updated_dh_table = [(theta_offset+joint, d, r, alpha) for (theta_offset, d, r, alpha), joint in zip(dh_table, joint_values)]
        return  forward_kinematics(updated_dh_table)

    def inverse(self, target_pose, initial_guess) -> list:
        """
        Inverse kinematics is not implemented by the generic DH solver.

        :param target_pose: Desired end-effector pose.
        :param initial_guess: Initial joint configuration.
        :return: This method does not return a value.
        :raises NotImplementedError: Always raised because inverse kinematics is robot-specific and must
            be implemented by specialized solver classes.
        """
        raise NotImplementedError("DHSolver does not support inverse kinematics. Use GeometricIKSolver or JacobianSolver instead.")