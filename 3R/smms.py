"""
This script is a self-motion manifolds study of cuspidal 3R
In this case, we study the planar cuspidal 3R
"""

import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb


def self_motion_manifold_xy(
    robot,
    x_target,
    y_target,
    n_samples=1000,
    q3_range=(-np.pi, np.pi),
):
    """
    Analytic self-motion manifold for fixed planar position (x, y).

    Returns:
        branches: list of arrays with shape (N, 3)
                  Each array contains [q1, q2, q3].
    """

    rho = np.hypot(x_target, y_target)
    phi = np.arctan2(y_target, x_target)

    q3_values = np.linspace(
        q3_range[0],
        q3_range[1],
        n_samples,
    )

    branches = []

    # Sign associated with A = +/- sqrt(...)
    for sign_A in [1.0, -1.0]:

        # Sign associated with q2 = +/- arccos(...)
        for sign_q2 in [1.0, -1.0]:

            branch = []

            for q3 in q3_values:

                B = 1.0 + 1.5 * np.sin(q3)

                radicand_A = rho**2 - B**2

                # No real solution for this q3
                if radicand_A < 0.0:
                    branch.append([np.nan, np.nan, np.nan])
                    continue

                A_abs = np.sqrt(max(radicand_A, 0.0))
                A = sign_A * A_abs

                denominator = 2.0 + 1.5 * np.cos(q3)

                cos_q2 = (A - 1.0) / denominator

                # Numerical tolerance
                if abs(cos_q2) > 1.0 + 1e-9:
                    branch.append([np.nan, np.nan, np.nan])
                    continue

                cos_q2 = np.clip(cos_q2, -1.0, 1.0)

                q2 = sign_q2 * np.arccos(cos_q2)

                q1 = phi - np.arctan2(B, A)

                # Wrap q1 to [-pi, pi]
                q1 = np.arctan2(np.sin(q1), np.cos(q1))

                branch.append([q1, q2, q3])

            branches.append(np.asarray(branch))

    return branches

def plot_self_motion_branches(robot, branches):
    """
    Plot self-motion manifolds in the (q2, q3) plane together
    with singularity points of the planar (x, y) task.

    Each branch must have shape (N, 3), with columns [q1, q2, q3].
    """

    # ---------------------------------------------------------
    # Compute planar-task singularity measure
    # ---------------------------------------------------------
    theta2_range = np.linspace(-np.pi, np.pi, 40)
    theta3_range = np.linspace(-np.pi, np.pi, 40)

    Q2, Q3 = np.meshgrid(theta2_range, theta3_range)

    singularity_measure = np.zeros_like(Q2)

    for i in range(Q2.shape[0]):
        for j in range(Q2.shape[1]):

            q = np.array([
                0.0,
                Q2[i, j],
                Q3[i, j],
            ])

            # Planar position Jacobian:
            # J_xy = d(x, y) / d(q1, q2, q3)
            J_xy = robot.jacob0(q)[:2, :]

            row_1 = J_xy[0, :]
            row_2 = J_xy[1, :]

            # Norm of all 2x2 minors
            singularity_measure[i, j] = np.linalg.norm(
                np.cross(row_1, row_2)
            )

    print(
        "Planar singularity measure range:",
        singularity_measure.min(),
        singularity_measure.max(),
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 8))

    # The planar singularities may be isolated points
    singularity_tolerance = 1e-2
    singular_mask = singularity_measure < singularity_tolerance

    plt.scatter(
        Q2[singular_mask],
        Q3[singular_mask],
        color="black",
        s=15,
        label=r"Planar singularity points: "
              r"$\operatorname{rank}(J_{xy})<2$",
        zorder=5,
    )

    # ---------------------------------------------------------
    # Plot self-motion manifold branches
    # ---------------------------------------------------------
    for branch_index, branch in enumerate(branches):

        branch = np.asarray(branch)

        if branch.ndim != 2 or branch.shape[1] != 3:
            raise ValueError(
                "Each branch must have shape (N, 3), "
                "with columns [q1, q2, q3]."
            )

        valid = np.isfinite(branch).all(axis=1)

        plt.plot(
            branch[valid, 1],
            branch[valid, 2],
            linewidth=2.0,
            label=f"SMM branch {branch_index + 1}",
        )

    plt.axhline(0.0, color="gray", linewidth=1.0)
    plt.axvline(0.0, color="gray", linewidth=1.0)

    plt.xlabel(r"$q_2$")
    plt.ylabel(r"$q_3$")
    plt.title(
        r"Self-motion manifolds and planar singularity points "
        r"in $(q_2,q_3)$"
    )

    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    plt.xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )

    plt.yticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

	# define robot
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

	# define plane target pose
	x_target = 2.5
	y_target = 0

	branches = self_motion_manifold_xy(
	    robot,
	    x_target,
	    y_target,
	    n_samples=2000,
	)

	print(f"number of branches: {len(branches)}")

	plot_self_motion_branches(robot, branches)
