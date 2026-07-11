import math
from open_kinematics.exceptions import IKNoSolution
import numpy as np

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