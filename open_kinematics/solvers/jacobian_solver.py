import warnings
import numpy as np
from open_kinematics.exceptions import InvalidDHParameters, SingularityWarning, IKNoSolution, JointLimitViolation
from open_kinematics.math.matrix_ops import validate_matrix_shape, identity_matrix

# Manipulability threshold below which a SingularityWarning is emitted.
SINGULARITY_THRESHOLD = 1e-4

def geometric_jacobian(transforms: list[np.ndarray], joint_types: list[str]):
    """
    Compute the geometric Jacobian matrix of a serial manipulator.

    The Jacobian relates joint velocities to the linear and angular velocities of the end-effector.
    Revolute and prismatic joints are handled according to their respective geometric definitions.

    :param transforms: List of cumulative homogeneous transformation matrices, one for each robot joint.
    :param joint_types: List describing the type of each joint ("revolute" or "prismatic").
    :return: A 6xn geometric Jacobian matrix, where n is the number of robot joints.
    :raises InvalidDHParameters: If the transform list, joint type list, or their contents are invalid.
    """
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
    """
    Compute Yoshikawa's manipulability measure for a robot Jacobian.

    The manipulability value provides an indication of how easily the robot can generate motion in arbitrary directions.
    Values approaching zero indicate robot configurations that are approaching kinematic singularities.

    :param J: Geometric Jacobian matrix of the robot.
    :return: Yoshikawa's manipulability measure.
    :raises InvalidDHParameters: If the supplied Jacobian is not a valid two-dimensional NumPy array.
    """
    if (
        not isinstance(J, np.ndarray)
        or not len(J.shape) == 2
    ):
        raise InvalidDHParameters

    w = np.sqrt(np.maximum(np.linalg.det(J @ J.T), 0.0))
    return w

def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    Compute the pose error between two homogeneous transformation matrices.

    This helper computes the translational and rotational error required to move the current end-effector pose
    toward the target pose. The returned error vector is represented as a six-element NumPy array
    containing the linear position error followed by the angular orientation error expressed using the axis-angle representation.

    The orientation error assumes the relative rotation is not exactly π radians.
    Axis-angle extraction becomes singular at θ = π and this special case is intentionally left unhandled
    because the function is designed for iterative Jacobian-based inverse kinematics,
    where only small corrective pose updates are expected.

    :param T_current: Current end-effector pose as a 4x4 homogeneous transformation matrix.
    :param T_target: Desired end-effector pose as a 4x4 homogeneous transformation matrix.
    :return: A six-element pose error vector consisting of linear position error
        followed by angular orientation error expressed in axis-angle form.
    :raises TypeError: If either transformation is not a NumPy array.
    :raises ValueError: If either transformation does not have shape (4,4).
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

    The solver emits ``SingularityWarning`` when the robot approaches a kinematic singularity during the iterative optimization process.
    This warning indicates reduced manipulability and potential numerical instability, but does not terminate the optimization.

    :param robot: Robot model providing forward kinematics and joint information.
    :param target: Desired end-effector homogeneous transformation matrix.
    :param q0: Initial joint configuration used as the starting point of the iterative optimization.
    :param tol: Pose-error convergence tolerance (default: ``1e-6``).
    :param max_iter: Maximum number of solver iterations (default: ``1000``).
    :param step_size: Jacobian update step size (default: ``0.1``).
    :return: Joint configuration that satisfies the requested pose within the specified tolerance.
    :raises IKNoSolution: If a valid solution cannot be obtained because the solver fails to converge,
        a joint limit is violated, or the Jacobian cannot be inverted during the iterative solution process.
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