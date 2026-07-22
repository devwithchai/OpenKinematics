import numpy as np
from open_kinematics.robots.base_robot import BaseRobot
from open_kinematics.exceptions import InvalidDHParameters

class ArticulatedRobot(BaseRobot):
    """
    Represents a generic articulated serial robot manipulator.

    The robot is constructed directly from a user-supplied DH parameter table.
    The constructor validates the table and converts each parameter dictionary into the tuple representation used internally.
    Forward kinematics and joint-limit validation are inherited from ``BaseRobot``.
    """

    def __init__(self, dh_table: list[dict]):
        """
        Construct an articulated robot from a Denavit-Hartenberg table.

        :param dh_table: Sequence of dictionaries containing the keys ``theta``, ``d``, ``r``, and ``alpha``.
        :return: None.
        :raises InvalidDHParameters: If ``dh_table`` is not a list or tuple, is empty, contains non-dictionary rows,
            contains missing or unexpected keys, or contains non-numeric parameter values.
        """

        if not isinstance(dh_table, (list, tuple)):
            raise InvalidDHParameters("dh_table must be a list of dicts.")

        if len(dh_table) == 0:
            raise InvalidDHParameters

        for i, row in enumerate(dh_table):
            if not isinstance(row, dict):
                raise InvalidDHParameters(f"Row {i} must be a dict. got {type(row).__name__}.")

            if not set(row.keys()) == {"theta", "d", "r", "alpha"}:
                raise InvalidDHParameters(f"Row {i} has invalid keys. Expected {{'theta', 'd', 'r', 'alpha'}}, got {set(row.keys())}.")

            for key, value in row.items():
                if not isinstance(value, (int, float)):
                    raise InvalidDHParameters(f"Row {i}, key '{key}' must be numeric, got {type(value).__name__}.")

        dh_parameters = [(row['theta'], row['d'], row['r'], row['alpha']) for row in dh_table]

        self.joint_types = ['revolute'] * len(dh_table)
        self.joint_limits = [(-np.pi, np.pi)] * len(dh_table)

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
