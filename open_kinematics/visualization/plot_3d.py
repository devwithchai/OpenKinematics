import numpy as np
import matplotlib.pyplot as plt
from open_kinematics.math.matrix_ops import identity_matrix
from open_kinematics.robots.articulated import ArticulatedRobot

def plot_3d(robot, joint_values, ax=None):
    """
    :param robot: ArticulatedRobot instance
    :param joint_values: Joint values
    :param ax: Optional matplotlib 3D Axes
    :return: (Figure, Axes)
    """

    if not isinstance(robot, ArticulatedRobot):
        raise TypeError("plot_3d() only supports Articulated Robot.")

    transforms = robot.forward_kinematics(joint_values, return_all=True)

    augmented_transforms = [identity_matrix(4)] + list(transforms)

    xs = []
    ys = []
    zs = []

    for transform in augmented_transforms:
        origin = transform[:3, 3]

        xs.append(origin[0])
        ys.append(origin[1])
        zs.append(origin[2])

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    else:
        fig = ax.figure

    ax.plot(xs, ys, zs, color="steelblue", linewidth=3, zorder=1, label="Links") # Draw robot links
    ax.plot(xs, ys, zs, linestyle="None", marker="o", markersize=8, color="black", zorder=2, label="Joints") # Draw joint markers
    ax.plot(xs[-1], ys[-1], zs[-1], linestyle="None", marker="*", markersize=15, color="red", zorder=3, label="End Effector") # Highlight end-effector

    # Compute coordinate-frame arrow scale
    link_lengths = []

    for i in range(len(augmented_transforms) - 1):
        p1 = augmented_transforms[i][:3, 3]
        p2 = augmented_transforms[i + 1][:3, 3]

        distance = np.linalg.norm(p2 - p1)

        if distance > 1e-9:
            link_lengths.append(distance)

    if link_lengths:
        scale = 0.15 * (sum(link_lengths) / len(link_lengths))
    else:
        scale = 0.1

    # Draw coordinate frames
    for transform in augmented_transforms:
        origin = transform[:3, 3]

        x_axis = transform[:3, 0]
        y_axis = transform[:3, 1]
        z_axis = transform[:3, 2]

        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=scale, normalize=True, color="red") # X-axis (Red)
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=scale, normalize=True, color="green") # Y-axis (Green)
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=scale, normalize=True, color="blue") # Z-axis (Blue)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax