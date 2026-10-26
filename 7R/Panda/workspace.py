import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import roboticstoolbox as rtb
from spatialmath import SE3


def plot_workspace_projections(robot, n_samples=2**18, seed=0, bins=2000):
    qlim = np.asarray(robot.qlim, dtype=float)

    # Sample all joints across their limits.
    sampler = qmc.Sobol(d=robot.n, scramble=True, seed=seed)
    m = math.ceil(math.log2(n_samples))
    u = sampler.random_base2(m=m)[:n_samples]
    q = qmc.scale(u, qlim[0], qlim[1])

    # End-effector positions from forward kinematics, converted to mm.
    xyz = np.asarray(robot.fkine(q).t) * 1000

    # Each histogram marks the 2D bins reached by at least one sample.
    projections = [
        (0, 1, "Top view: X–Y", "X [mm]", "Y [mm]"),
        (0, 2, "Side view: X–Z", "X [mm]", "Z [mm]"),
        (1, 2, "Side view: Y–Z", "Y [mm]", "Z [mm]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, (i, j, title, xlabel, ylabel) in zip(axes, projections):
        H, xedges, yedges = np.histogram2d(
            xyz[:, i], xyz[:, j], bins=bins
        )
        ax.imshow(
            H.T > 0,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            interpolation="nearest",
            cmap="Blues",
            aspect="equal",
        )
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()
    return xyz

robot = rtb.models.DH.Panda()

# print("Tool transform before:", robot.tool)
robot.tool = SE3()  # identity: no tool offset
print("Tool transform after:", robot.tool)


xyz = plot_workspace_projections(robot)
print("XYZ min [mm]:", xyz.min(axis=0))
print("XYZ max [mm]:", xyz.max(axis=0))