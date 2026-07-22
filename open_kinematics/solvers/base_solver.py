from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """
    Abstract base class for all kinematic solvers.
    Every solver in OpenKinematics must implement the forward and inverse kinematics interfaces,
    ensuring a consistent API across concrete solver implementations.
    """

    @abstractmethod
    def forward(self, dh_table, joint_values) -> np.ndarray:
        """
        Compute forward kinematics.

        :param dh_table: DH parameters describing the robot.
        :param joint_values: Joint configuration.
        :return: End-effector pose.
        """
        pass

    @abstractmethod
    def inverse(self, target_pose, initial_guess) -> list:
        """
        Compute inverse kinematics.

        :param target_pose: Position of end-effector as 4x4 matrix
        :param initial_guess: Initial joint configuration
        :return: List of joint angles.
        """
        pass
