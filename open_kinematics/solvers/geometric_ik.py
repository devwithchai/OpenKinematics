import math
import numpy as np
from open_kinematics.exceptions import IKNoSolution
from open_kinematics.robots.scara import ScaraRobot

def planar_2r_ik(l1: float, l2: float, x: float, y: float):
    """
    Compute the closed-form inverse kinematics of a 2R planar manipulator.

    This solver computes the two possible joint configurations (elbow-up and elbow-down)
    that place the end-effector at the requested Cartesian position.
    Since only the end-effector position is specified, both configurations are valid solutions.

    :param l1: Length of the first link.
    :param l2: Length of the second link.
    :param x: Desired X-coordinate of the end-effector.
    :param y: Desired Y-coordinate of the end-effector.
    :return: A list containing two inverse-kinematics solutions represented as``(theta1, theta2)`` tuples.
        The first solution corresponds to the elbow-up configuration
        and the second corresponds to the elbow-down configuration.
    :raises TypeError: If any input parameter is not numeric.
    :raises ValueError: If either link length is less than or equal to zero.
    :raises IKNoSolution: If the requested Cartesian position lies outside the robot's reachable workspace.
    """
    for value in (l1, l2, x, y):
        if not isinstance(value, (int, float)):
            raise TypeError

    if not (l1 > 0 and l2 > 0):
        raise ValueError

    r_sq = x**2 + y**2
    c2 = (r_sq - l1**2 - l2**2) / (2 * l1 * l2)

    epsilon = 1e-9

    if (c2 < -1 - epsilon) or (c2 > 1 + epsilon):
        raise IKNoSolution(f"Target {(x, y)} is outside reachable workspace for given l1, & l2")

    c2 = np.clip(c2, -1.0, 1.0)

    s2_pos = math.sqrt(1 - c2**2)
    theta2_up = math.atan2(s2_pos, c2) # elbow-up
    theta2_down = math.atan2(-s2_pos, c2) # elbow-down

    theta1_up  = math.atan2(y, x) - math.atan2(l2*np.sin(theta2_up), l1+l2*np.cos(theta2_up))
    theta1_down = math.atan2(y, x) - math.atan2(l2*np.sin(theta2_down), l1+l2*np.cos(theta2_down))

    return [(theta1_up, theta2_up), (theta1_down, theta2_down)]

def scara_ik(robot: ScaraRobot, x: float, y: float, z: float, phi: float):
    """
    Compute the closed-form inverse kinematics of a SCARA robot.

    This solver computes all valid joint configurations that satisfy the requested end-effector position and orientation.
    Unlike the planar 2R solver, this solver simultaneously satisfies the requested Cartesian position and end-effector orientation.

    :param robot: The ScaraRobot instance whose geometric parameters are used for the inverse kinematics calculation.
    :param x: Desired X-coordinate of the end-effector.
    :param y: Desired Y-coordinate of the end-effector.
    :param z: Desired Z-coordinate of the end-effector.
    :param phi: Desired end-effector orientation about the Z-axis, in radians.
    :return: A list containing two inverse-kinematics solutions represented as ``(theta1, theta2, d3, theta4)`` tuples.
    :raises TypeError: If the robot is not a ScaraRobot instance or if any Cartesian pose parameter is not numeric.
    :raises IKNoSolution: If the requested pose cannot be reached within the robot's prismatic joint limits or lies outside the planar workspace.
    """
    if not isinstance(robot, ScaraRobot):
        raise TypeError("robot must be a ScaraRobot instance.")

    for values in (x, y, z, phi):
        if not isinstance(values, (int, float)):
            raise TypeError("x, y, z, and phi must be numeric.")

    l1, l2 = robot.link_lengths
    d_min, d_max = robot.d_range

    d = -z

    epsilon = 1e-9

    if d < d_min - epsilon or d > d_max + epsilon:
        raise IKNoSolution(f"Target z={z} requires prismatic joint value "
                           f"{d}, which is outside range ({d_min}, {d_max}).")

    d = max(d_min, min(d, d_max))

    planar_solutions = planar_2r_ik(l1=l1, l2=l2, x=x, y=y)
    solutions = []
    for theta1, theta2 in planar_solutions:
        theta4 = theta1 + theta2 - phi
        theta4 = math.atan2(math.sin(theta4), math.cos(theta4)) # Normalize to (-π, π)
        solutions.append((theta1, theta2, d, theta4))

    return solutions