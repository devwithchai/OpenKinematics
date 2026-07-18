import unittest
import matplotlib
# Use a non-interactive backend for testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy

from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.robots.articulated import ArticulatedRobot
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.visualization.plot_3d import plot_3d


class TestPlot3D(unittest.TestCase):

    def setUp(self):
        self.dh_table = [
            {"theta": 0.0, "d": 0.3, "r": 0.2, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.5, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 0.4, "alpha": 0.0},
            {"theta": 0.0, "d": 0.4, "r": 0.0, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.0, "alpha": -numpy.pi / 2},
            {"theta": 0.0, "d": 0.2, "r": 0.0, "alpha": 0.0},
        ]

        self.robot = ArticulatedRobot(self.dh_table)
        self.joint_values = [0.4, -0.7, 0.8, -0.5, 0.6, -0.3]

    def tearDown(self):
        plt.close("all")

    def test_returns_figure_and_axes(self):
        fig, ax = plot_3d(self.robot, self.joint_values)

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes3D)

    def test_reuses_existing_axes(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        returned_fig, returned_ax = plot_3d(self.robot, self.joint_values, ax=ax)

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)

    def test_planar_robot_raises_type_error(self):
        robot = PlanarRobot([1.0, 1.0])

        with self.assertRaises(TypeError):
            plot_3d(robot, [0.0, 0.0])

    def test_scara_robot_raises_type_error(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))

        with self.assertRaises(TypeError):
            plot_3d(robot, [0.0, 0.0, 0.0, 0.0])

    def test_link_data_matches_forward_kinematics(self):
        fig, ax = plot_3d(self.robot, self.joint_values)

        transforms = self.robot.forward_kinematics(self.joint_values, return_all=True)

        augmented = [identity_matrix(4)] + transforms

        expected_x = [T[0, 3] for T in augmented]
        expected_y = [T[1, 3] for T in augmented]
        expected_z = [T[2, 3] for T in augmented]

        link_line = ax.lines[0]

        numpy.testing.assert_allclose(link_line.get_xdata(), expected_x)
        numpy.testing.assert_allclose(link_line.get_ydata(), expected_y)
        numpy.testing.assert_allclose(link_line.get_data_3d()[2], expected_z)

    def test_coordinate_frames_drawn(self):
        fig, ax = plot_3d(self.robot, self.joint_values)

        transforms = self.robot.forward_kinematics(self.joint_values, return_all=True)

        n_frames = len(transforms) + 1 # +1 for base frame
        expected_quivers = 3 * n_frames # X, Y, Z axes

        self.assertEqual(len(ax.collections), expected_quivers)

if __name__ == "__main__":
    unittest.main()