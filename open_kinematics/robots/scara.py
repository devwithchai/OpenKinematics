import numpy as np
from open_kinematics.robots.base_robot import BaseRobot
from open_kinematics.exceptions import InvalidDHParameters

class ScaraRobot(BaseRobot):
    """
    Represents a four-degree-of-freedom SCARA robot.

    The robot consists of two revolute joints, one prismatic joint, and a final revolute joint.
    Link lengths and prismatic joint limits are internally converted into DH parameters.
    Forward kinematics and joint-limit validation are inherited from ``BaseRobot``.
    """

    def __init__(self, link_lengths: list[float], d_range: list[float]):
        """
        Construct a SCARA robot model.

        :param link_lengths: Two positive link lengths defining the planar arm.
        :param d_range: Minimum and maximum allowable displacement for the prismatic joint.
        :return: None.
        :raises InvalidDHParameters: If ``link_lengths`` or ``d_range`` are not valid sequences,
            if the number of supplied link lengths or displacement limits is incorrect,
            if any supplied value is non-numeric,
            if any link length is non-positive,
            or if the lower displacement limit is greater than or equal to the upper limit.
        """

        if (
            not isinstance(link_lengths, (list, tuple))
            or len(link_lengths) != 2
            or any(not isinstance(x, (int, float)) or x <= 0 for x in link_lengths)
        ):
            raise InvalidDHParameters

        if (
            not isinstance(d_range, (list, tuple))
            or len(d_range) != 2
            or any(not isinstance(x, (int, float)) for x in d_range)
            or d_range[0] >= d_range[1]
        ):
            raise InvalidDHParameters

        self.link_lengths = link_lengths
        self.d_range = d_range

        dh_parameters = [(0.0, 0.0, link_lengths[0], 0.0), (0.0, 0.0, link_lengths[1], np.pi), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, np.pi)]

        self.joint_types = ["revolute", "revolute", "prismatic", "revolute"]
        self.joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi), d_range, (-np.pi, np.pi)]

        super().__init__(dh_parameters=dh_parameters, joint_types=self.joint_types, joint_limits=self.joint_limits)

    def get_joint_positions(self, joint_values):
        """
        Compute the Cartesian position of every joint in the robot.

        :param joint_values: Joint values defining the robot configuration.
        :return: A list of ``[x, y, z]`` coordinates representing the base followed by every joint position.
            Intended primarily for visualization.
        :raises TypeError: Propagated from ``forward_kinematics()`` if ``joint_values`` is not a list, tuple, or NumPy array.
        :raises ValueError: Propagated from ``forward_kinematics()`` if the number of supplied joint values does not match the robot's joint count.
        :raises JointLimitViolation: Propagated from ``forward_kinematics()`` if any joint value exceeds its configured limits.
        """

        transform = self.forward_kinematics(joint_values, return_all=True)
        result_list = [[0.0, 0.0, 0.0]]
        for T in transform:
            x = T[0, 3]
            y = T[1, 3]
            z = T[2, 3]
            result_list.append([x, y, z])

        return result_list
