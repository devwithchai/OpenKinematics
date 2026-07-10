import numpy as np
from open_kinematics.robots.base_robot import BaseRobot
from open_kinematics.exceptions import InvalidDHParameters

class PlanarRobot(BaseRobot):
    def __init__(self, link_lengths: list[float]):

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
        transform = self.forward_kinematics(joint_values, return_all=True)
        result_list = [[0.0, 0.0]]
        for T in transform:
            x = T[0, 3]
            y = T[1, 3]
            result_list.append([x, y])

        return result_list
