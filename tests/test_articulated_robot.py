import unittest
import numpy
from open_kinematics.robots.articulated import ArticulatedRobot
from open_kinematics.exceptions import JointLimitViolation,InvalidDHParameters

class TestArticulatedRobot(unittest.TestCase):

    def setUp(self):
        self.dh_table = [
            {"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0},
            {"theta": 0.0, "d": 0.0, "r": 1.0, "alpha": 0.0},
        ]

    def test_non_list_input_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ArticulatedRobot(5)

    def test_empty_table_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ArticulatedRobot([])

    def test_row_not_dict_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ArticulatedRobot([(0.0, 0.0, 0.0, 0.0)])

    def test_missing_key_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ArticulatedRobot([{"theta": 0.0, "d": 0.0, "r": 1.0}])

    def test_extra_key_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ArticulatedRobot([{"theta": 0.0,"d": 0.0,"r": 1.0,"alpha": 0.0,"extra": 1.0}])

    def test_non_numeric_value_raises(self):
        with self.assertRaises(InvalidDHParameters):
            ArticulatedRobot([{"theta": "bad","d": 0.0,"r": 1.0,"alpha": 0.0}])

    def test_valid_construction(self):
        robot = ArticulatedRobot(self.dh_table)
        self.assertEqual(robot.num_joints, 3)
        self.assertEqual(robot.joint_types,["revolute", "revolute", "revolute"])

    def test_fk_3r_zero_config(self):
        robot = ArticulatedRobot(self.dh_table)
        T = robot.forward_kinematics([0.0, 0.0, 0.0])
        self.assertAlmostEqual(T[0, 3], 3.0, places=6)
        self.assertAlmostEqual(T[1, 3], 0.0, places=6)
        self.assertAlmostEqual(T[2, 3], 0.0, places=6)

    def test_get_joint_positions(self):
        robot = ArticulatedRobot(self.dh_table)
        positions = robot.get_joint_positions([0.0, 0.0, 0.0])
        self.assertEqual(len(positions), 4)
        numpy.testing.assert_array_almost_equal(positions[0],[0.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[1],[1.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[2],[2.0, 0.0, 0.0])
        numpy.testing.assert_array_almost_equal(positions[3],[3.0, 0.0, 0.0])

    def test_joint_limit_violation(self):
        robot = ArticulatedRobot(self.dh_table)
        with self.assertRaises(JointLimitViolation):
            robot.forward_kinematics([4.0, 0.0, 0.0])

if __name__ == "__main__":
    unittest.main()