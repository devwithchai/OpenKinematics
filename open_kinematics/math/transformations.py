"""
Transformation matrix utilities for OpenKinematics.

This module provides rotation matrices, translation matrices,
and homogeneous transformation matrix construction.

All angles are expressed in radians.
"""

import numpy as np

def rot_x(theta: float) -> np.ndarray:
    """
    Rotation matrix about the X-axis.
    :param theta: Rotation angle in radians
    :return: 3x3 rotation matrix
    :raises TypeError: If theta is not numeric.
    """
    if not isinstance(theta, (int, float)):
        raise TypeError('theta must be a numeric value in radians.')

    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rot_y(theta: float) -> np.ndarray:
    """
    Rotation matrix about the Y-axis.
    :param theta: Rotation angle in radians
    :return: 3x3 rotation matrix
    :raises TypeError: If theta is not numeric.
    """
    if not isinstance(theta, (int, float)):
        raise TypeError('theta must be a numeric value in radians.')

    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rot_z(theta: float) -> np.ndarray:
    """
    Rotation matrix about the Z-axis.
    :param theta: Rotation angle in radians
    :return: 3x3 rotation matrix
    :raises TypeError: If theta is not numeric.
    """
    if not isinstance(theta, (int, float)):
        raise TypeError('theta must be a numeric value in radians.')

    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# Translation Matrix -
def translation(x: float, y: float, z: float) -> np.ndarray:
    """
    Generate a homogeneous translation matrix.
    :param x: Translation along the X-axis.
    :param y: Translation along the Y-axis.
    :param z: Translation along the Z-axis.
    :return: 4x4 homogeneous translation matrix
    :raises TypeError: If x, y, or z is not numeric.
    """
    for value in (x, y, z):
        if not isinstance(value, (int, float)):
            raise TypeError('Translation components must be numeric.')

    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

# Homogeneous Transformation Matrix
def homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Construct a homogeneous transformation matrix from rotation and translation.
    :param R: A 3x3 rotation matrix.
    :param p: A translation vector of shape (3,) or (3,1).
    :return: 4x4 homogeneous transformation matrix
    :raises TypeError: If R or p is not a NumPy ndarray.
    :raises ValueError: If R is not a 3x3 matrix or p does not have shape (3,) or (3,1).
    """
    if not isinstance(R, np.ndarray):
        raise TypeError('R must be a numpy.ndarray.')
    if R.shape != (3,3):
        raise ValueError('R must be a 3x3 rotation matrix.')

    if not isinstance(p, np.ndarray):
        raise TypeError('p must be a numpy.ndarray.')

    if p.shape == (3,):
        p = p.reshape(3,1)
    elif p.shape != (3,1):
        raise ValueError('p must have shape (3,) or (3,1).')

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p.flatten()

    return T