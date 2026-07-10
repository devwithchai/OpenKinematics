import numpy as np
from open_kinematics.exceptions import InvalidDHParameters
from open_kinematics.math.matrix_ops import validate_matrix_shape, identity_matrix

def geometric_jacobian(transforms: list[np.ndarray], joint_types: list[str]):
    if (
        not isinstance(transforms, list)
        or not isinstance(joint_types, list)
        or len(transforms) == 0
        or len(joint_types) == 0
        or len(transforms) != len(joint_types)
    ):
        raise InvalidDHParameters

    for element in transforms:
        validate_matrix_shape(matrix=element, shape=(4,4))

    augmented_transforms = [identity_matrix(4)] + transforms

    n = len(joint_types)
    J = np.zeros((6, n))
    p_e = augmented_transforms[-1][:3, 3]

    for i in range(n):
        z = augmented_transforms[i][:3, 2]
        p = augmented_transforms[i][:3, 3]
        if joint_types[i] == "revolute":
            linear, angular = np.cross(z, p_e - p), z
        elif joint_types[i] == "prismatic":
            linear, angular = z, np.zeros(3)
        else:
            raise InvalidDHParameters(f"Invalid joint type '{joint_types[i]}' at index {i}.")

        column = np.concatenate([linear, angular])
        J[:, i] = column

    return J

def manipulability(J: np.ndarray) -> float:
    if (
        not isinstance(J, np.ndarray)
        or not len(J.shape) == 2
    ):
        raise InvalidDHParameters

    w = np.sqrt(np.maximum(np.linalg.det(J @ J.T), 0.0))
    return w