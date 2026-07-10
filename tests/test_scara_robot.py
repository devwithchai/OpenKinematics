import unittest
import numpy
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.exceptions import JointLimitViolation, InvalidDHParameters

class TestScaraRobot(unittest.TestCase):

    def test_wrong_link_count_raises(self):
        with self.assertRaises(InvalidDHParameters):
                ScaraRobot([1.0, 1.0, 1.0], (-0.5, 0.5))

    def test_negative_link_length_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ScaraRobot([1.0, -1.0], (-0.5, 0.5))

    def test_invalid_d_range_raises(self):
        with self.assertRaises(InvalidDHParameters):
            robot = ScaraRobot([1.0, 1.0], (0.5, -0.5))

    def test_valid_construction(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        self.assertEqual(robot.num_joints, 4)
        self.assertEqual(robot.joint_types, ["revolute", "revolute", "prismatic", "revolute"])

    def test_fk_zero_config(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        T = robot.forward_kinematics([0, 0, 0, 0])
        self.assertAlmostEqual(T[0, 3], 2.0, places=6)
        self.assertAlmostEqual(T[1, 3], 0.0, places=6)
        self.assertAlmostEqual(T[2, 3], 0.0, places=6)

    def test_fk_prismatic_motion(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        T = robot.forward_kinematics([0, 0, 0.3, 0])
        self.assertAlmostEqual(T[0, 3], 2.0, places=6)
        self.assertAlmostEqual(T[1, 3], 0.0, places=6)
        self.assertAlmostEqual(T[2, 3], -0.3, places=6)

    def test_get_joint_positions(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        positions = robot.get_joint_positions([0, 0, 0, 0])
        self.assertEqual(len(positions), 5)
        numpy.testing.assert_array_almost_equal(positions[0], [0.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[1], [1.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[2], [2.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[3], [2.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[4], [2.0, 0.0, 0.0])

    def test_angular_joint_limit_violation(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        with self.assertRaises(JointLimitViolation):
            robot.forward_kinematics([4.0, 0, 0, 0])

    def test_prismatic_joint_limit_violation(self):
        robot = ScaraRobot([1.0, 1.0], (-0.5, 0.5))
        with self.assertRaises(JointLimitViolation):
            robot.forward_kinematics([0, 0, 0.9, 0])

if __name__ == '__main__':
    unittest.main()