import warnings
import numpy as np
from open_kinematics.exceptions import InvalidDHParameters, SingularityWarning, IKNoSolution, JointLimitViolation
from open_kinematics.math.matrix_ops import validate_matrix_shape, identity_matrix

SINGULARITY_THRESHOLD = 1e-4

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

def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    inputs: two 4x4 homogeneous tranforms
    output: a (6,) NumPy array, rows 0-2 = linear position error, rows 3-5 = angular (axis-angle) error.

    Orientation error is represented as an axis-angle vector
    (theta * axis).

    Assumes the relative rotation is not exactly π radians.
    The axis-angle extraction becomes singular at θ = π and is
    intentionally left unhandled because this helper is intended
    for iterative Jacobian IK with small corrective steps.
    """

    validate_matrix_shape(T_current, (4, 4))
    validate_matrix_shape(T_target, (4, 4))

    p_current = T_current[:3, 3]
    p_target = T_target[:3, 3]

    delta_p = p_target - p_current

    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]

    R_err = R_target @ R_current.T

    trace = np.trace(R_err)

    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0,1.0)
    theta = np.arccos(cos_theta)

    epsilon = 1e-9

    if theta < epsilon:
        delta_omega = np.zeros(3)
    else:
        axis = ((1.0 / (2.0 * np.sin(theta)))
                * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1],
        ]))
        delta_omega = theta * axis

    return np.concatenate([delta_p, delta_omega])

def pseudoinverse_ik(robot, target, q0, tol=1e-6, max_iter=1000, step_size=0.1):
    """
    Iterative Jacobian pseudoinverse inverse kinematics.

    Solves inverse kinematics using the Jacobian pseudoinverse method.
    Valid for robots with n ≥ 6 joints (full row-rank Jacobian required by the right pseudoinverse formula).
    Lower-DOF robots should use geometric_ik.py instead.

    Raises:
        IKNoSolution:
             - if convergence is not achieved within max_iter iterations,
             - if a joint limit is exceeded during the iterative search,
             - if the Jacobian becomes singular during pseudoinverse computation.
    """
    q = np.array(q0, dtype=float, copy=True)

    for iteration in range(max_iter):
        try:
            transforms = robot.forward_kinematics(q, return_all=True)
        except JointLimitViolation as e:
            raise IKNoSolution("JointLimitViolation path") from e
        T_current = transforms[-1]

        delta_x = pose_error(T_current, target)

        if np.linalg.norm(delta_x) < tol:
            return list(q)

        J = geometric_jacobian(transforms, robot.joint_types)

        w = manipulability(J)

        if w < SINGULARITY_THRESHOLD:
            warnings.warn(
                (
                f"Near singular configuration at "
                f"iteration {iteration}. "
                f"Manipulability = {w:.6e}"
                ),
                SingularityWarning,
            )

        try:
            J_pinv = (J.T @ np.linalg.inv(J @ J.T))
        except np.linalg.LinAlgError as e:
            raise IKNoSolution("LinAlgError path") from e

        delta_q = J_pinv @ delta_x

        q = q + step_size * delta_q

    else:
        raise IKNoSolution(
            (
                "Pseudoinverse IK did not converge "
                f"within {max_iter} iterations. "
                f"Final pose error norm = "
                f"{np.linalg.norm(delta_x):.6e}"
            )
        )