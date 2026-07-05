from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """
    Abstract base class for all kinematic solvers.
    Every solver in OpenKinematics must implement the forward method.
    This enforces a consistent interface across FK, IK, and Jacobian solvers.
    """

    @abstractmethod
    def forward(self, dh_table, joint_values) -> np.ndarray:
        """
        Compute forward kinematics.
        Takes a DH table and joint values, returns a 4x4 homogeneous transformation matrix of the end-effector pose.
        """
        pass

    @abstractmethod
    def inverse(self, target_pose, initial_guess) -> list:
        """
        Compute inverse kinematics.
        Takes a target end-effector pose as a 4x4 matrix and an initial joint configuration, returns a list of joint angles.
        """
        pass
