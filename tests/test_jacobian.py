import unittest
import numpy
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.solvers.jacobian_solver import geometric_jacobian
from open_kinematics.exceptions import InvalidDHParameters
from open_kinematics.solvers.jacobian_solver import manipulability
from open_kinematics.robots.articulated import ArticulatedRobot

class TestJacobian(unittest.TestCase):

    def setUp(self):
        self.robot = PlanarRobot([1.0, 1.0])

    def test_shape_is_6xn(self):
        # 2R Planar Robot
        transforms = self.robot.forward_kinematics([0.0, 0.0], return_all=True)
        J = geometric_jacobian(transforms,self.robot.joint_types)
        self.assertEqual(J.shape, (6, 2))

        # 3R Planar Robot
        robot = PlanarRobot([1.0, 1.0, 1.0])
        transforms = robot.forward_kinematics([0.0, 0.0, 0.0], return_all=True)
        J = geometric_jacobian(transforms, robot.joint_types)
        self.assertEqual(J.shape, (6, 3))

    def test_zero_config_known_values(self):
        transforms = self.robot.forward_kinematics([0.0, 0.0], return_all=True)
        J = geometric_jacobian(transforms, self.robot.joint_types)
        expected_J = numpy.array([
            [0.0, 0.0],
            [2.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        numpy.testing.assert_array_almost_equal(J, expected_J)

    def test_invalid_joint_type_raises(self):
        transforms = self.robot.forward_kinematics([0.0, 0.0], return_all=True)
        with self.assertRaises(InvalidDHParameters):
            geometric_jacobian(transforms,["revolute", "invalid"])

    def test_mismatched_lengths_raises(self):
        transforms = self.robot.forward_kinematics([0.0, 0.0], return_all=True)
        with self.assertRaises(InvalidDHParameters):
            geometric_jacobian(transforms,["revolute"])

    def test_empty_inputs_raises(self):
        with self.assertRaises(InvalidDHParameters):
            geometric_jacobian([], [])

    def test_manipulability_singular_2r_robot(self):
        transforms = self.robot.forward_kinematics([0.0, 0.0], return_all=True)
        J = geometric_jacobian(transforms, self.robot.joint_types)
        w = manipulability(J)
        self.assertAlmostEqual(w, 0.0, places=6)

    def test_manipulability_nonsingular_case(self):
        dh_table = [
            {"theta": 0.0, "d": 0.3, "r": 0.2, "alpha": 1.57079632679},
            {"theta": 0.0, "d": 0.0, "r": 0.4, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 0.3, "alpha": -1.57079632679},
            {"theta": 0.0, "d": 0.5, "r": 0.0, "alpha": 1.57079632679},
            {"theta": 0.0, "d": 0.0, "r": 0.0, "alpha": -1.57079632679},
            {"theta": 0.0, "d": 0.2, "r": 0.0, "alpha": 0.0},
        ]
        robot = ArticulatedRobot(dh_table)
        transforms = robot.forward_kinematics([0.2, -0.5, 0.7, -0.3, 0.6, -0.4], return_all=True)
        J = geometric_jacobian(transforms, robot.joint_types)
        w = manipulability(J)
        self.assertGreater(w, 0.0)

    def test_manipulability_invalid_input_raises(self):
        with self.assertRaises(InvalidDHParameters):
            manipulability([1, 2, 3])
        with self.assertRaises(InvalidDHParameters):
            manipulability(numpy.array([1, 2, 3]))

if __name__ == "__main__":
    unittest.main()