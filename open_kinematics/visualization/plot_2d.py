import matplotlib.pyplot as plt
from open_kinematics.robots.planar import PlanarRobot
from open_kinematics.robots.scara import ScaraRobot

def plot_2d(robot, joint_values, ax=None):
    """
    Visualize a planar or SCARA robot configuration in two dimensions.

    :param robot: PlanarRobot or ScaraRobot instance to visualize.
    :param joint_values: Joint values describing the robot configuration.
    :param ax: Existing Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    :return: Tuple containing the Matplotlib figure and axes.
    :raises TypeError: If ``robot`` is not a supported robot type, or propagated from ``get_joint_positions()``
        if ``joint_values`` has an invalid type.
    :raises ValueError: Propagated from ``get_joint_positions()`` if the number of supplied joint values is incorrect.
    :raises JointLimitViolation: Propagated from ``get_joint_positions()`` if a joint exceeds its configured limits.
    """

    if not isinstance(robot, (PlanarRobot, ScaraRobot)):
        raise TypeError("plot_2d() supports only PlanarRobot and ScaraRobot.")

    # Get joint positions from the robot
    points = robot.get_joint_positions(joint_values)

    # Project to XY plane
    xs = []
    ys = []

    for point in points:
        xs.append(point[0])
        ys.append(point[1])

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(xs, ys, linewidth=2, color="black", label="Links") # Draw links
    ax.plot(xs, ys, marker="o", linestyle="None", markersize=6, color="blue", label="Joints") # Draw joints
    ax.plot(xs[-1], ys[-1], marker="*", markersize=12, color="red") # Highlight end-effector

    # Axis formatting
    ax.set_aspect("equal", adjustable='box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax