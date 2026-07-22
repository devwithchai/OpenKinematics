"""
Matrix utility functions for OpenKinematics.

This module provides common matrix operations used throughout
the library, including identity matrix creation, matrix shape
validation, and safe matrix multiplication.
"""

import numpy as np

def identity_matrix(size: int=4) -> np.ndarray:
    """
    Create a square identity matrix.
    :param size: Dimension of the identity matrix.
    :return: A square identity matrix of shape (size, size).
    :raises TypeError: If size is not an integer.
    :raises ValueError: If size is less than 1.
    """
    # 1. Check if size is actually an integer
    if not isinstance(size, int):
        raise TypeError("Size must be an integer.")

    # 2. Check if size is a valid positive number
    if size < 1:
        raise ValueError("Size of the matrix must be at least 1.")

    return np.eye(size)

def validate_matrix_shape(matrix: np.ndarray, shape: tuple, name: str = "matrix") -> None:
    """
    Validate the shape of a NumPy matrix.
    :param matrix: Matrix whose shape will be validated.
    :param shape: Expected matrix shape.
    :param name: Name used in generated error messages.
    :return: None. Raises an exception if validation fails.
    :raises TypeError: If matrix is not a NumPy ndarray.
    :raises ValueError: If matrix does not have the expected shape.
    """
    # 1. Type Check: Ensure it is a NumPy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Error in {name}: Expected a numpy.ndarray, but got {type(matrix).__name__}")

    # 2. Shape Check: Ensure dimensions match
    if matrix.shape != shape:
        raise ValueError(f"Error in {name}: expected shape {shape}, but got {matrix.shape}")

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices.
    :param a: Left-hand matrix.
    :param b: Right-hand matrix.
    :return: Matrix product of a and b.
    :raises TypeError: If either input is not a NumPy ndarray.
    :raises ValueError: If either input is not two-dimensional, or if the matrix dimensions are incompatible for multiplication.
    """
    # 1. Type Check
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")

    # 2. Rank Check (Ensure both are 2D)
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(f"matmul only supports 2D matrices. Got {len(a.shape)}D and {len(b.shape)}D.")

    # 3. Inner Dimension Check
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for multiplication: {a.shape} and {b.shape}. "
            f"Columns in A ({a.shape[1]}) must match Rows in B ({b.shape[0]})."
        )
    return a @ b