import math
import unittest
from open_kinematics.robots.scara import ScaraRobot
from open_kinematics.solvers.geometric_ik import scara_ik
from open_kinematics.exceptions import IKNoSolution

class TestScaraIK(unittest.TestCase):

    def setUp(self):
        self.robot = ScaraRobot([1.0, 1.0], [-0.5, 0.5])

    def test_round_trip_validation(self):
        x, y, z = 1.0, 1.0, -0.2
        phi = math.pi / 4
        solutions = scara_ik(self.robot, x, y, z, phi)
        self.assertEqual(len(solutions), 2)
        for joint_values in solutions:
            T = self.robot.forward_kinematics(joint_values)
            # Position
            self.assertAlmostEqual(T[0, 3], x, places=6)
            self.assertAlmostEqual(T[1, 3], y, places=6)
            self.assertAlmostEqual(T[2, 3], z, places=6)
            # Orientation about Z
            computed_phi = math.atan2(T[1, 0], T[0, 0])
            self.assertAlmostEqual(computed_phi, phi, places=6)

    def test_z_out_of_range_raises(self):
        with self.assertRaises(IKNoSolution):
            scara_ik(self.robot, 1.0, 1.0, -1.0, 0.0)

    def test_xy_out_of_workspace_raises(self):
        with self.assertRaises(IKNoSolution):
            scara_ik(self.robot, 3.0, 0.0, 0.0, 0.0)

    def test_invalid_input_types(self):
        with self.assertRaises(TypeError):
            scara_ik("robot", 1, 1,  0, 0)
        with self.assertRaises(TypeError):
            scara_ik(self.robot, "1", 1,  0, 0)
        with self.assertRaises(TypeError):
            scara_ik(self.robot, 1, None,  0, 0)

    def test_theta4_normalization(self):
        solutions = scara_ik(self.robot, 1, 1, -0.2, 10)
        for solution in solutions:
            theta4 = solution[3]
            self.assertGreater(theta4, -math.pi)
            self.assertLessEqual(theta4, math.pi)

if __name__ == "__main__":
    unittest.main()