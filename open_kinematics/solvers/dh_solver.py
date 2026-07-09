import numpy as np
from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.solvers.base_solver import BaseSolver

def dh_transform(theta: float, d: float, r: float, alpha: float) -> np.ndarray:
    """
    Compute the standard Denavit-Hartenberg (DH) homogeneous transformation matrix.

    The DH convention is widely used in robotics to represent the relationship
    between successive coordinate frames in a kinematic chain.

    :param theta: Joint angle (rotation about the Z-axis), in radians.
    :param d: Link offset (translation along the Z-axis).
    :param r: Link length (translation along the X-axis).
    :param alpha: Link twist (rotation about the X-axis), in radians.
    :return: A 4x4 homogeneous transformation matrix representing the pose of one
        link relative to the previous link.
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
    Compute the forward kinematics for a serial manipulator using the
    Denavit-Hartenberg (DH) convention.
    :param return_all:
    :param dh_table: A list of DH parameter tuples, each of the form (theta, d, r, alpha).
        - theta : float
            Joint angle (rotation about Z-axis), in radians.
        - d : float
            Link offset (translation along Z-axis).
        - r : float
            Link length (translation along X-axis).
        - alpha : float
            Link twist (rotation about X-axis), in radians.
    :return: A 4x4 homogeneous transformation matrix representing the pose of the
        end-effector relative to the base frame.
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
        updated_dh_table = [(theta_offset+joint, d, r, alpha) for (theta_offset, d, r, alpha), joint in zip(dh_table, joint_values)]
        return  forward_kinematics(updated_dh_table)
    def inverse(self, target_pose, initial_guess) -> list:
        raise NotImplementedError("DHSolver does not support inverse kinematics. Use GeometricIKSolver or JacobianSolver instead.")