"""
Public API facade for OpenKinematics.

This module provides a unified user-facing interface while delegating
all robotics computations to the underlying robot, solver, and
visualization layers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from open_kinematics.robots.base_robot import BaseRobot
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot
from open_kinematics.solvers.geometric_ik import planar_2r_ik, scara_ik

from open_kinematics.solvers.jacobian_solver import geometric_jacobian, pseudoinverse_ik

from open_kinematics.visualization.plot_2d import plot_2d
from open_kinematics.visualization.plot_3d import plot_3d

class Robot:
    """
    Public API facade for OpenKinematics.

    This class wraps an existing robot model and exposes a clean,
    unified interface for forward kinematics, inverse kinematics,
    Jacobians, and visualization.

    :param robot_instance : PlanarRobot | ScaraRobot | ArticulatedRobot (Robot model to wrap)
    """
    _PLOT_DISPATCH: dict[type[BaseRobot], Callable] = {PlanarRobot: plot_2d,
                                                       ScaraRobot: plot_2d,
                                                       ArticulatedRobot: plot_3d}
    _IK_DISPATCH = {
        (PlanarRobot, "geometric"): "_ik_planar_geometric",
        (ScaraRobot, "geometric"): "_ik_scara_geometric",
        (ArticulatedRobot, "pseudoinverse"): "_ik_articulated_pseudoinverse",
    }

    def __init__(self, robot_instance):
        """
        Construct a Robot facade around an existing robot model.
        """
        self._robot: BaseRobot = robot_instance

    # Factory Methods

    @classmethod
    def planar(cls, link_lengths):
        """
        Create a PlanarRobot facade.
        :param link_lengths : list[float]
        :return Robot
        """
        return cls(PlanarRobot(link_lengths))

    @classmethod
    def scara(cls, link_lengths, d_range):
        """
        Create a ScaraRobot facade.
        :param link_lengths : list[float]
        :param d_range : tuple[float, float]
        :return Robot
        """
        return cls(ScaraRobot(link_lengths, d_range))

    @classmethod
    def articulated(cls, dh_table):
        """
        Create an ArticulatedRobot facade.
        :param dh_table : list[dict]
        :return Robot
        """
        return cls(ArticulatedRobot(dh_table))

    # Forward Kinematics

    def fk(self, joint_values, return_all=False):
        """
        Compute forward kinematics.
        :param joint_values : list | tuple | numpy.ndarray
        :param return_all : bool, optional (If True, return every intermediate transform)
        :return numpy.ndarray: End-effector transform
        OR
        :return list[numpy.ndarray]: All transforms if return_all=True.
        """
        return self._robot.forward_kinematics(joint_values, return_all=return_all)

    # Jacobian

    def jacobian(self, joint_values):
        """
        Compute the geometric Jacobian.
        :param joint_values: list | tuple | numpy.ndarray
        :return: numpy.ndarray
        """
        transforms = self._robot.forward_kinematics(joint_values, return_all=True)
        return  geometric_jacobian(transforms, self._robot.joint_types)

    # Plotting

    def plot(self, joint_values, ax=None):
        """
        Plot the robot configuration.
        :param joint_values: list | tuple | numpy.ndarray
        :param ax: matplotlib.axes.Axes, optional
        :return: tuple (fig, ax)
        """
        renderer = self._PLOT_DISPATCH.get(type(self._robot))

        if renderer is None:
            raise TypeError(f"Unsupported robot type: {type(self._robot).__name__}")

        return renderer(self._robot, joint_values, ax=ax)

    # Validation Helper

    def _validate_target_pose(self, target_pose):
        target_pose = np.asarray(target_pose, dtype=float)
        if not target_pose.shape == (4, 4):
            raise ValueError(f"target_pose must be a 4x4 homogeneous transform, got shape {target_pose.shape}.")
        return target_pose

    # Extraction Helpers

    def _extract_xy(self, T):
        x = T[0, 3]
        y = T[1, 3]
        return (x, y)

    def _extract_scara_pose(self, T):
        x, y = self._extract_xy(T)
        z = T[2, 3]
        phi = np.arctan2(T[1, 0], T[0, 0])
        return (x, y, z, phi)

    def _ik_planar_geometric(self, target_pose, **kwargs):
        x, y = self._extract_xy(target_pose)
        l1, l2 = self._robot.link_lengths
        return planar_2r_ik(l1, l2, x, y)

    def _ik_scara_geometric(self, target_pose, **kwargs):
        x, y, z, phi = self._extract_scara_pose(target_pose)
        return scara_ik(self._robot, x, y, z, phi)

    def _ik_articulated_pseudoinverse(self, target_pose, **kwargs):
        q0 = kwargs.get("q0")
        if q0 is None:
            raise ValueError("pseudoinverse IK requires an initial guess 'q0'.")

        tol = kwargs.get("tol", 1e-6)
        max_iter = kwargs.get("max_iter", 1000)
        step_size = kwargs.get("step_size", 0.1)

        return pseudoinverse_ik(self._robot, target_pose, q0, tol, max_iter, step_size)

    # Inverse Kinematics

    def ik(self, target_pose, method='geometric', **kwargs):
        target_pose = self._validate_target_pose(target_pose)
        key = (type(self._robot), method)
        method_name = self._IK_DISPATCH.get(key)
        if method_name is None:
            raise ValueError(f"Unsupported combination: {type(self._robot)} with method '{method}'.")

        handler = getattr(self, method_name)
        return handler(target_pose, **kwargs)