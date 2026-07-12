import unittest
import numpy
from open_kinematics.solvers.jacobian_solver import pose_error
from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.math.transformations import rot_z,homogeneous_transform

class TestPoseError(unittest.TestCase):

    def setUp(self):
        self.identity = identity_matrix(4)

    def test_identity_pose_error(self):
        error = pose_error(self.identity, self.identity)
        expected = numpy.zeros(6)
        numpy.testing.assert_array_almost_equal(error, expected)

    def test_pure_translation(self):
        T_current = identity_matrix(4)
        T_target = identity_matrix(4)
        T_target[:3, 3] = [1.0, 2.0, 3.0]

        error = pose_error(T_current, T_target)
        expected = numpy.array([
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
        ])
        numpy.testing.assert_array_almost_equal(error, expected)

    def test_pure_rotation_about_z(self):
        R = rot_z(numpy.pi / 2)

        T_current = identity_matrix(4)
        T_target = homogeneous_transform(R,numpy.array([0.0, 0.0, 0.0]))

        error = pose_error(T_current, T_target)
        expected = numpy.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            numpy.pi / 2,
        ])
        numpy.testing.assert_allclose(error, expected, atol=1e-6)

    def test_small_angle_returns_zero(self):
        R = rot_z(1e-9)

        T_current = identity_matrix(4)
        T_target = homogeneous_transform(R,numpy.array([0.0, 0.0, 0.0]))

        error = pose_error(T_current, T_target)
        expected = numpy.zeros(6)

        numpy.testing.assert_array_equal(error, expected)

if __name__ == "__main__":
    unittest.main()