"""
This script is plotting the characteristic surfaces.
Reference: A Review of Cuspidal Serial and Parallel Manipulators (Fig 13)
"""

import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt


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
