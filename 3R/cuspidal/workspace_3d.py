"""
3D workspace plot for 3R cuspidal robot

reference:
A Review of Cuspidal Serial and Parallel Manipulators (Fig 8)
"""

import roboticstoolbox as rtb

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label


def plot_workspace_3d(robot, q_solutions, grid_size=150, singularity_tol=1e-15):
    theta2_range = np.linspace(-np.pi, np.pi, grid_size, endpoint=True)
    theta3_range = np.linspace(-np.pi, np.pi, grid_size, endpoint=True)
    TH2, TH3 = np.meshgrid(theta2_range, theta3_range)

    rho = np.empty_like(TH2)
    z = np.empty_like(TH2)
    det_jv = np.empty_like(TH2)

    theta1 = 0.0

    # Sample and map the joint-space grid to (rho, z, cos(theta2)).
    for i in range(grid_size):
        for j in range(grid_size):
            q = [theta1, TH2[i, j], TH3[i, j]]
            T = robot.fkine(q).A
            Jv = robot.jacob0(q)[:3, :]

            rho[i, j] = np.hypot(T[0, 3], T[1, 3])
            z[i, j] = T[2, 3]
            det_jv[i, j] = np.linalg.det(Jv)

    fig = plt.figure(figsize=(15, 6))
    positive_aspect_ax = None

    for index, sign in enumerate((-1, 1), start=1):
        mask = (np.sign(det_jv) == sign) & (np.abs(det_jv) > singularity_tol)
        ax = fig.add_subplot(1, 2, index, projection="3d")

        # Honor explicit zorder instead of matplotlib's automatic depth sorting
        ax.computed_zorder = False

        ax.plot_surface(
            np.where(mask, rho, np.nan),
            np.where(mask, z, np.nan),
            np.where(mask, np.cos(TH2), np.nan),
            cmap="gray",
            linewidth=0,
            antialiased=True,
            alpha=0.6,
            zorder=1,
        )

        x0, x1 = ax.get_xlim3d()
        y0, y1 = ax.get_ylim3d()
        z0, z1 = ax.get_zlim3d()
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))

        ax.set_title(f"Aspect {index} (sign(det(Jv)) = {sign:+d})")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$z$")
        ax.set_zlabel(r"$\cos(\theta_2)$")
        ax.view_init(elev=25, azim=-55)

        if sign == 1:
            positive_aspect_ax = ax

    # Mark f(q2) on the positive determinant surface.
    q2 = np.asarray(q_solutions[1], dtype=float)
    T2 = robot.fkine(q2).A
    rho_q2 = np.hypot(T2[0, 3], T2[1, 3])
    z_q2 = T2[2, 3]
    cos_theta2_q2 = np.cos(q2[1])

    positive_aspect_ax.scatter(
        rho_q2,
        z_q2,
        cos_theta2_q2,
        color="red",
        edgecolors="black",
        s=80,
        marker="o",
        label=r"f($q_2$)",
        depthshade=False,
        zorder=11,
    )

    # mark f(q3)
    q3 = np.asarray(q_solutions[2], dtype=float)
    T3 = robot.fkine(q3).A
    rho_q3 = np.hypot(T2[0, 3], T2[1, 3])
    z_q3 = T3[2, 3]
    cos_theta2_q3 = np.cos(q3[1])

    positive_aspect_ax.scatter(
        rho_q3,
        z_q3,
        cos_theta2_q3,
        color="blue",
        edgecolors="black",
        s=80,
        marker="o",
        label=r"f($q_3$)",
        depthshade=False,
        zorder=11,
    )
     

    # Interpolate joint configurations, then map each one with forward kinematics.
    q3 = np.asarray(q_solutions[2], dtype=float)
    joint_path = np.linspace(q2, q3, 10000)

    path_rho = np.empty(len(joint_path))
    path_z = np.empty(len(joint_path))
    path_cos = np.empty(len(joint_path))
    path_det = np.empty(len(joint_path))

    for k, q in enumerate(joint_path):
        T = robot.fkine(q).A
        Jv = robot.jacob0(q)[:3, :]

        path_rho[k] = np.hypot(T[0, 3], T[1, 3])
        path_z[k] = T[2, 3]
        path_cos[k] = np.cos(q[1])
        path_det[k] = np.linalg.det(Jv)

    if np.all(path_det > singularity_tol):
        positive_aspect_ax.plot3D(
            path_rho,
            path_z,
            path_cos,
            color="red",
            linewidth=3,
            label=r"$q_2 \rightarrow q_3$",
            zorder=10,
        )
    else:
        print(
            "The interpolated joint path does not stay entirely in the "
            "det(Jv) > 0 aspect; it is omitted from that subplot."
        )

    positive_aspect_ax.legend()
    plt.tight_layout()
    plt.show()



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
q_solutions = [q_sol1, q_sol2, q_sol3, q_sol4]

plot_workspace_3d(robot,q_solutions,grid_size=200)

    
    
    
            
    