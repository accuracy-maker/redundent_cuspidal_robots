import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb


def plot_workspace(robot, grid_size=300):
    theta2_range = np.linspace(-np.pi, np.pi, grid_size)
    theta3_range = np.linspace(-np.pi, np.pi, grid_size)

    Q2, Q3 = np.meshgrid(theta2_range, theta3_range)

    theta1 = 0.0

    rho_values = np.zeros_like(Q2)
    z_values = np.zeros_like(Q2)
    det_values = np.zeros_like(Q2)

    for i in range(grid_size):
        for j in range(grid_size):
            q = np.array([
                theta1,
                Q2[i, j],
                Q3[i, j],
            ])

            T = robot.fkine(q).A

            x = T[0, 3]
            y = T[1, 3]
            z = T[2, 3]

            rho_values[i, j] = np.hypot(x, y)
            z_values[i, j] = z

            Jv = robot.jacob0(q)[:3, :]
            det_values[i, j] = np.linalg.det(Jv)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the mapped configuration-space grid
    for i in range(grid_size):
        ax.plot(
            rho_values[i, :],
            z_values[i, :],
            color="lightgray",
            linewidth=0.25,
            zorder=1,
        )

    for j in range(grid_size):
        ax.plot(
            rho_values[:, j],
            z_values[:, j],
            color="lightgray",
            linewidth=0.25,
            zorder=1,
        )

    # Extract singularity contours in joint space
    contour_fig, contour_ax = plt.subplots()

    contour_set = contour_ax.contour(
        Q2,
        Q3,
        det_values,
        levels=[0.0],
    )

    plt.close(contour_fig)

    if len(contour_set.allsegs[0]) == 0:
        raise RuntimeError("No singularity curve was detected.")

    # Map every singularity contour into workspace coordinates
    for path_index, segment in enumerate(contour_set.allsegs[0]):
        theta2_segment = segment[:, 0]
        theta3_segment = segment[:, 1]

        rho_segment = []
        z_segment = []

        for theta2, theta3 in zip(theta2_segment, theta3_segment):
            q = np.array([
                theta1,
                theta2,
                theta3,
            ])

            T = robot.fkine(q).A

            x = T[0, 3]
            y = T[1, 3]
            z = T[2, 3]

            rho_segment.append(np.hypot(x, y))
            z_segment.append(z)

        rho_segment = np.asarray(rho_segment)
        z_segment = np.asarray(z_segment)

        ax.plot(
            rho_segment,
            z_segment,
            color="red",
            linewidth=2.5,
            label="Mapped singularity curve"
            if path_index == 0
            else None,
            zorder=3,
        )


    ax.set_xlabel(r"$\rho=\sqrt{x^2+y^2}$")
    ax.set_ylabel(r"$z$")
    ax.set_title(r"Workspace and mapped singularity curves in $(\rho,z)$")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


link1 = rtb.RevoluteDH(a=1.0, alpha=np.pi / 2)
link2 = rtb.RevoluteDH(a=1.0, alpha=np.pi / 2)
link3 = rtb.RevoluteDH(a=1.0)

robot = rtb.DHRobot([link1, link2, link3], name="3R Orthogonal-Axis Robot")
print(robot)

plot_workspace(robot, grid_size=80)