import matplotlib.pyplot as plt
import numpy as np

from open_kinematics.api import Robot

robot = Robot.planar([1.0, 1.0])

joint_values = [0.5, 0.3]

transform = robot.fk(joint_values)

print("Joint values (radians):")
print(joint_values)

print("\nEnd-effector transformation matrix:")
print(np.round(transform, 3))

fig, ax = robot.plot(joint_values)
plt.show()