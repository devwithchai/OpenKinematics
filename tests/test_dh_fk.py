import unittest
import numpy as np
import math
from open_kinematics.solvers.dh_solver import dh_transform, forward_kinematics

class TestDHTransform(unittest.TestCase):

    def test_zero_config_single_joint(self):
        """A single joint with all DH parameters set to zero should return identity."""
        T = dh_transform(0, 0, 0,0)
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_pure_rotation(self):
        """A joint with theta = π/2 should rotate about Z by 90°."""
        T = dh_transform(math.pi / 2, 0, 0, 0)

        self.assertAlmostEqual(T[0, 0], 0.0, places=6)
        self.assertAlmostEqual(T[0, 1], -1.0, places=6)
        self.assertAlmostEqual(T[1, 0], 1.0, places=6)
        self.assertAlmostEqual(T[1, 1], 0.0, places=6)

    def test_pure_translation(self):
        """A joint with only r = 1.0 should translate along the X-axis."""
        T = dh_transform(0, 0, 1.0, 0)

        self.assertAlmostEqual(T[0, 3], 1.0, places=6)
        self.assertAlmostEqual(T[1, 3], 0.0, places=6)
        self.assertAlmostEqual(T[2, 3], 0.0, places=6)

    def test_fk_zero_config_returns_identity(self):
        """Two zero DH transforms multiplied together should still be identity."""
        dh_table = [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ]

        T = forward_kinematics(dh_table)
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_fk_2r_planar_known_config(self):
        """Validate FK for a simple 2R planar robot."""

        # Configuration 1: θ1 = 0, θ2 = 0
        dh_table = [
            (0, 0, 1.0, 0),
            (0, 0, 1.0, 0),
        ]

        T = forward_kinematics(dh_table)

        self.assertAlmostEqual(T[0, 3], 2.0, places=6)
        self.assertAlmostEqual(T[1, 3], 0.0, places=6)

        # Configuration 2: θ1 = π/2, θ2 = 0
        dh_table = [
            (math.pi / 2, 0, 1.0, 0),
            (0, 0, 1.0, 0),
        ]

        T = forward_kinematics(dh_table)

        self.assertAlmostEqual(T[0, 3], 0.0, places=6)
        self.assertAlmostEqual(T[1, 3], 2.0, places=6)

if __name__ == '__main__':
    unittest.main()