import unittest
import numpy
from open_kinematics.robots.articulated import ArticulatedRobot
from open_kinematics.solvers.jacobian_solver import pseudoinverse_ik
from open_kinematics.exceptions import IKNoSolution, SingularityWarning

class TestPseudoinverseIK(unittest.TestCase):

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

    def test_converges_to_reachable_target(self):
        q_known = [0.4, -0.7, 0.8, -0.5, 0.6, -0.3]
        target = self.robot.forward_kinematics(q_known)
        q0 = [0.3, -0.5, 0.6, -0.3, 0.4, -0.1]
        q_solution = pseudoinverse_ik(self.robot, target, q0)
        T_solution = self.robot.forward_kinematics(q_solution)
        self.assertTrue(numpy.allclose(T_solution, target, atol=1e-4))

    def test_converges_from_close_initial_guess(self):
        q_known = [0.4, -0.7, 0.8, -0.5, 0.6, -0.3]
        target = self.robot.forward_kinematics(q_known)
        q0 = [0.45, -0.65, 0.85, -0.45, 0.55, -0.25]
        q_solution = pseudoinverse_ik(self.robot, target, q0)
        T_solution = self.robot.forward_kinematics(q_solution)
        self.assertTrue(numpy.allclose(T_solution, target, atol=1e-4))

    def test_unreachable_target_raises(self):
        q_known = [0.4, -0.7, 0.8, -0.5, 0.6, -0.3]
        target = self.robot.forward_kinematics(q_known)
        target[:3, 3] *= 100.0
        q0 = [0.3, -0.5, 0.6, -0.3, 0.4, -0.1]
        with self.assertRaises(IKNoSolution):
            pseudoinverse_ik(self.robot, target, q0)

    def test_warns_near_singular_configuration(self):
        target = self.robot.forward_kinematics([0.4, -0.7, 0.8, -0.5, 0.6, -0.3])
        q0 = [0.0] * 6
        with self.assertWarns(SingularityWarning):
            with self.assertRaises(IKNoSolution):
                pseudoinverse_ik(self.robot, target, q0)

if __name__ == "__main__":
    unittest.main()