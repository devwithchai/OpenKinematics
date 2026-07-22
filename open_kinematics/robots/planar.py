import numpy as np
from open_kinematics.robots.base_robot import BaseRobot
from open_kinematics.exceptions import InvalidDHParameters

class PlanarRobot(BaseRobot):
    """
    Represents a planar serial robot consisting entirely of revolute joints.

    Link lengths supplied by the user are internally converted into DH parameters.
    Forward kinematics and joint-limit validation are inherited from ``BaseRobot``.
    """

    def __init__(self, link_lengths: list[float]):
        """
        Construct a planar robot from its link lengths.

        :param link_lengths: Sequence of positive link lengths defining the robot.
        :return: None.
        :raises InvalidDHParameters: If ``link_lengths`` is not a list or tuple, is empty,
            contains non-numeric values, or contains non-positive link lengths.
        """

        if not isinstance(link_lengths, (list, tuple)):
            raise InvalidDHParameters

        if len(link_lengths) == 0:
            raise InvalidDHParameters

        for element in link_lengths:
            if not isinstance(element, (int, float)) or not element>0:
                raise InvalidDHParameters

        self.link_lengths = link_lengths

        dh_parameters = [(0.0, 0.0, L, 0.0) for L in link_lengths]

        self.joint_types = ["revolute"] * len(link_lengths)
        self.joint_limits = [(-np.pi, np.pi)] * len(link_lengths)

        super().__init__(dh_parameters=dh_parameters, joint_types=self.joint_types, joint_limits=self.joint_limits)

    def get_joint_positions(self, joint_values):
        """
        Compute the Cartesian position of every joint in the robot.

        :param joint_values: Joint values defining the robot configuration.
        :return: A list of ``[x, y]`` coordinates representing the base followed by every joint position.
            Intended primarily for visualization.
        :raises TypeError: Propagated from ``forward_kinematics()`` if ``joint_values`` is not a list, tuple, or NumPy array.
        :raises ValueError: Propagated from ``forward_kinematics()`` if the number of supplied joint values does not match the robot's joint count.
        :raises JointLimitViolation: Propagated from ``forward_kinematics()`` if any joint value exceeds its configured limits.
        """
        transform = self.forward_kinematics(joint_values, return_all=True)
        result_list = [[0.0, 0.0]]
        for T in transform:
            x = T[0, 3]
            y = T[1, 3]
            result_list.append([x, y])

        return result_list
