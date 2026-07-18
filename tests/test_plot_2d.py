import unittest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from open_kinematics.visualization.plot_2d import plot_2d
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.robots.articulated import ArticulatedRobot

class TestPlot2D(unittest.TestCase):

    def setUp(self):
        self.planar_robot = PlanarRobot([1.0, 1.0])
        self.scara_robot = ScaraRobot(link_lengths=[1.0, 1.0], d_range=[0.0, 0.5])
        self.articulated_robot = ArticulatedRobot([
            {"theta": 0.0, "d": 0.3, "r": 0.2, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.5, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 0.4, "alpha": 0.0},
            {"theta": 0.0, "d": 0.4, "r": 0.0, "alpha": numpy.pi / 2},
            {"theta": 0.0, "d": 0.0, "r": 0.0, "alpha": -numpy.pi / 2},
            {"theta": 0.0, "d": 0.2, "r": 0.0, "alpha": 0.0},
        ])

    def test_returns_figure_and_axes(self):
        fig, ax = plot_2d(self.planar_robot,[numpy.pi / 4, numpy.pi / 6])

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        plt.close(fig)

    def test_reuses_existing_axes(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_2d(self.planar_robot,[numpy.pi / 4, numpy.pi / 6], ax=ax)

        self.assertIs(returned_ax, ax)
        self.assertIs(returned_fig, fig)

        plt.close(fig)

    def test_plotted_data_matches_planar_joint_positions(self):
        joint_values = [numpy.pi / 4, numpy.pi / 6]

        expected_points = self.planar_robot.get_joint_positions(joint_values)

        expected_x = [point[0] for point in expected_points]
        expected_y = [point[1] for point in expected_points]

        fig, ax = plot_2d(self.planar_robot, joint_values)

        plotted_line = ax.lines[0]

        self.assertTrue(numpy.allclose(plotted_line.get_xdata(), expected_x))

        self.assertTrue(numpy.allclose(plotted_line.get_ydata(), expected_y))

        plt.close(fig)

    def test_scara_projection_to_xy(self):
        joint_values = [numpy.pi / 6, numpy.pi / 4, 0.2, numpy.pi / 3,]

        expected_points = self.scara_robot.get_joint_positions(joint_values)

        expected_x = [point[0] for point in expected_points]
        expected_y = [point[1] for point in expected_points]

        fig, ax = plot_2d(self.scara_robot, joint_values)

        plotted_line = ax.lines[0]

        self.assertTrue(numpy.allclose(plotted_line.get_xdata(), expected_x))

        self.assertTrue(numpy.allclose(plotted_line.get_ydata(), expected_y))

        plt.close(fig)

    def test_articulated_robot_raises_type_error(self):
        with self.assertRaises(TypeError):
            plot_2d(self.articulated_robot, [0.0] * 6)

if __name__ == "__main__":
    unittest.main()