import math, numpy
import unittest

from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.solvers.geometric_ik import planar_2r_ik
from open_kinematics.exceptions import IKNoSolution


class TestPlanarIK(unittest.TestCase):

    def setUp(self):
        self.l1 = 1.0
        self.l2 = 1.0
        self.robot = PlanarRobot([self.l1, self.l2])

    def test_round_trip_validation(self):
        x = 1.0
        y = 1.0
        solutions = planar_2r_ik(self.l1, self.l2, x, y)
        self.assertEqual(len(solutions), 2)
        for theta1, theta2 in solutions:
            T = self.robot.forward_kinematics([theta1, theta2])
            self.assertAlmostEqual(T[0, 3], x, places=6)
            self.assertAlmostEqual(T[1, 3], y, places=6)
            self.assertAlmostEqual(T[2, 3], 0.0, places=6)

    def test_out_of_workspace_beyond_max_reach(self):
        with self.assertRaises(IKNoSolution):
            planar_2r_ik(1.0, 1.0, 3.0, 0.0)

    def test_out_of_workspace_inside_inner_radius(self):
        with self.assertRaises(IKNoSolution):
            planar_2r_ik(2.0, 1.0, 0.0, 0.0)

    def test_full_extension_singularity(self):
        solutions = planar_2r_ik(1.0, 1.0, 2.0, 0.0)
        self.assertEqual(len(solutions), 2)
        theta1_up, theta2_up = solutions[0]
        theta1_down, theta2_down = solutions[1]
        self.assertAlmostEqual(theta1_up, theta1_down, places=6)
        self.assertAlmostEqual(theta2_up, theta2_down, places=6)
        self.assertAlmostEqual(theta2_up, 0.0, places=6)

    def test_folded_singularity(self):
        solutions = planar_2r_ik(1.0, 1.0, 0.0, 0.0)
        self.assertEqual(len(solutions), 2)
        for theta1, theta2 in solutions:
            self.assertAlmostEqual(abs(theta2), math.pi, places=6)
            T = self.robot.forward_kinematics([theta1, theta2])
            self.assertAlmostEqual(T[0, 3], 0.0, places=6)
            self.assertAlmostEqual(T[1, 3], 0.0, places=6)

    def test_invalid_input_types(self):
        with self.assertRaises(TypeError):
            planar_2r_ik("1", 1.0, 0.0, 0.0)

        with self.assertRaises(TypeError):
            planar_2r_ik(1.0, None, 0.0, 0.0)

        with self.assertRaises(TypeError):
            planar_2r_ik(1.0, 1.0, "x", 0.0)

    def test_invalid_link_lengths(self):
        with self.assertRaises(ValueError):
            planar_2r_ik(0.0, 1.0, 0.0, 0.0)

        with self.assertRaises(ValueError):
            planar_2r_ik(-1.0, 1.0, 0.0, 0.0)

        with self.assertRaises(ValueError):
            planar_2r_ik(1.0, 0.0, 0.0, 0.0)

        with self.assertRaises(ValueError):
            planar_2r_ik(1.0, -1.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()