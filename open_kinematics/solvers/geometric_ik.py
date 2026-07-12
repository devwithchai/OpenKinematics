import math
import numpy as np
from open_kinematics.exceptions import IKNoSolution
from open_kinematics.robots.scara import ScaraRobot

def planar_2r_ik(l1: float, l2: float, x: float, y: float):
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