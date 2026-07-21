import matplotlib.pyplot as plt
import numpy as np

from open_kinematics.api import Robot

robot = Robot.scara(link_lengths=[1.0, 0.8], d_range=(0.0, 0.5))

joint_values = [0.5, 0.4, 0.2, 0.1]

transform = robot.fk(joint_values)

print("Joint Values")
print(np.round(joint_values, 3))

print("\nEnd-Effector Transformation Matrix")
print(np.round(transform, 3))

solutions = robot.ik(transform, method="geometric")

print("\nInverse Kinematics Solutions")

for index, solution in enumerate(solutions, start=1):
    print(f"\nSolution {index}")
    print(np.round(solution, 3))

print("\nVerification")

target_position = transform[:3, 3]
print("\nTarget Position")
print(np.round(target_position, 3))

for index, solution in enumerate(solutions, start=1):

    recovered_transform = robot.fk(solution)
    recovered_position = recovered_transform[:3, 3]

    print(f"\nSolution {index}")
    print("Joint Values :", np.round(solution, 3))
    print("Recovered Position :", np.round(recovered_position, 3))
    print("Position Match :", np.allclose(recovered_position, target_position, atol=1e-6))
    print("Full Pose Match :", np.allclose(recovered_transform, transform, atol=1e-6))

fig, ax = robot.plot(joint_values)
ax.set_title("SCARA Robot - Forward & Inverse Kinematics")
plt.show()