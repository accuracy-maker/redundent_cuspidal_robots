"""
This file plots the configuration and workspace of 3R cuspidal robot
"""

import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb


robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(
            d=0.0,
            a=1.0,
            alpha=-np.pi / 2,
        ),
        rtb.RevoluteDH(
            d=1.0,
            a=2.0,
            alpha=np.pi / 2,
        ),
        rtb.RevoluteDH(
            d=0.0,
            a=1.5,
            alpha=0.0,
        ),
    ],
    name="3R_cuspidal_robot",
)

print(robot)

q_sol1 = np.array([-1.8, -2.8, 1.9])
q_sol2 = np.array([-0.9, -0.7, 2.5])
q_sol3 = np.array([-2.9, -3.0, -0.2])
q_sol4 = np.array([0.2, -0.3, -1.9])

def plot_configSpace(robot):
    theta2_range = np.linspace(-np.pi, np.pi, 40)
    theta3_range = np.linspace(-np.pi, np.pi, 40)

    Q2, Q3 = np.meshgrid(theta2_range, theta3_range)

    # Singularity curves are independent of theta1
    theta1_slice = 0.0

    det_values = np.zeros_like(Q2)

    for i in range(Q2.shape[0]):
        for j in range(Q2.shape[1]):

            q = np.array([
                theta1_slice,
                Q2[i, j],
                Q3[i, j],
            ])

            Jv = robot.jacob0(q)[:3, :]
            det_values[i, j] = np.linalg.det(Jv)

    print("determinant range:")
    print(det_values.min(), det_values.max())

    plt.figure(figsize=(8, 7))

    # Singularity set
    plt.contour(
        Q2,
        Q3,
        det_values,
        levels=[0.0],
        colors="black",
        linewidths=2.0,
    )

    # Plot the four IK solutions in the (theta2, theta3) plane
    solutions = [
        q_sol1,
        q_sol2,
        q_sol3,
        q_sol4,
    ]

    for index, q_sol in enumerate(solutions, start=1):
        plt.scatter(
            q_sol[1],
            q_sol[2],
            marker="x",
            s=80,
            linewidths=2,
            label=fr"$q_{{{index}}}$",
        )

    # plot a dashed line between q2 and q3
    plt.plot(
        [q_sol2[1], q_sol3[1]],  # theta2 coordinates
        [q_sol2[2], q_sol3[2]],  # theta3 coordinates
        linestyle="--",
        color="red",
        linewidth=1.5,
        label=r"$q_2 \rightarrow q_3$",
    )

    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(0.0, color="black", linewidth=1)

    plt.xlabel(r"$\theta_2$")
    plt.ylabel(r"$\theta_3$")
    plt.title(
        r"Singularity curve and four IK solutions for "
        r"$p=[2.5,\,0,\,0.5]$"
    )

    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_configSpace(robot)