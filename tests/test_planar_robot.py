import unittest
import math, numpy
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.exceptions import JointLimitViolation, InvalidDHParameters

class TestPlanarRobot(unittest.TestCase):

    def test_empty_link_lengths_raises(self):
        with self.assertRaises(InvalidDHParameters):
                PlanarRobot([])

    def test_non_list_input_raises(self):
        with self.assertRaises(InvalidDHParameters):
            PlanarRobot(5)

    def test_negative_link_length_raises(self):
        with self.assertRaises(InvalidDHParameters):
            PlanarRobot([1.0, -2.0])

    def test_valid_construction(self):
        robot = PlanarRobot([1.0, 1.0])
        self.assertEqual(robot.num_joints, 2)
        self.assertEqual(robot.joint_types, ["revolute", "revolute"])

    def test_fk_2r_known_config(self):
        robot = PlanarRobot([1.0, 1.0])

        T = robot.forward_kinematics([0, 0])
        self.assertAlmostEqual(T[0, 3], 2.0, places=6)
        self.assertAlmostEqual(T[1, 3], 0.0, places=6)

        T = robot.forward_kinematics([math.pi/2, 0])
        self.assertAlmostEqual(T[0, 3], 0.0, places=6)
        self.assertAlmostEqual(T[1, 3], 2.0, places=6)

    def test_get_joint_positions(self):
        robot = PlanarRobot([1.0, 1.0])
        positions = robot.get_joint_positions([0, 0])
        self.assertEqual(len(positions), 3)
        numpy.testing.assert_array_almost_equal(positions[0], [0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[1], [1.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[2], [2.0, 0.0])

    def test_joint_limit_violation(self):
        robot = PlanarRobot([1.0, 1.0])
        with self.assertRaises(JointLimitViolation):
            robot.forward_kinematics([4.0, 0])

if __name__ == '__main__':
    unittest.main()