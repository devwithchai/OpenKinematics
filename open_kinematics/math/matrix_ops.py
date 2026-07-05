import numpy as np

"""
This module provides matrix utility functions to support 
robotics and kinematics calculations.
"""

def identity_matrix(size: int=4) -> np.ndarray:
    """
    Creates an identity matrix of a given size.
    Default size is 4x4.
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
    Validates that the input is a NumPy array and matches the expected shape.
    """
    # 1. Type Check: Ensure it is a NumPy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Error in {name}: Expected a numpy.ndarray, but got {type(matrix).__name__}")

    # 2. Shape Check: Ensure dimensions match
    if matrix.shape != shape:
        raise ValueError(f"Error in {name}: expected shape {shape}, but got {matrix.shape}")

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiplies two 2D NumPy arrays after validating types and dimensions.
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