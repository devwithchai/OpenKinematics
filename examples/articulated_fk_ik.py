import matplotlib.pyplot as plt
import numpy as np

from open_kinematics.api import Robot

dh_table = [
    {"theta": 0.0, "d": 0.35, "r": 0.10, "alpha": np.pi / 2},
    {"theta": 0.0, "d": 0.00, "r": 0.45, "alpha": 0.0},
    {"theta": 0.0, "d": 0.00, "r": 0.35, "alpha": -np.pi / 2},
    {"theta": 0.0, "d": 0.30, "r": 0.00, "alpha": np.pi / 2},
    {"theta": 0.0, "d": 0.00, "r": 0.00, "alpha": -np.pi / 2},
    {"theta": 0.0, "d": 0.15, "r": 0.00, "alpha": 0.0},
]

robot = Robot.articulated(dh_table=dh_table)

joint_values = [0.40, -0.60, 0.50, 0.80, -0.30, 0.20]

transform = robot.fk(joint_values)

print("Joint Values (radians)")
print(np.round(joint_values, 3))

print("\nEnd-Effector Transformation Matrix")
print(np.round(transform, 3))

q0 = [0.10, -0.20, 0.10, 0.20, -0.10, 0.10]

print("\nInitial Guess")
print(np.round(q0, 3))

solution = robot.ik(transform, method="pseudoinverse", q0=q0)

print("\nRecovered Joint Values")
print(np.round(solution, 3))

recovered_transform = robot.fk(solution)

print("\nVerification")

print("Full Pose Match :",np.allclose(recovered_transform, transform, atol=1e-5))

print("\nRecovered Transformation Matrix")
print(np.round(recovered_transform, 3))

fig, ax = robot.plot(joint_values)
ax.set_title("6-DOF Articulated Robot - Forward & Pseudoinverse IK")
plt.show()