"""
DH parameters definition of a 3 DoF Cuspidal Robot
This reference is Wenger and Chablat's A Review of Cuspidal
Serial and Parallel Manipulators
"""

import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3


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

# exmaple: forward kinematics
q1 = np.array([-1.8, -2.8, 1.9])
q2 = np.array([-0.9, -0.7, 2.5])
q3 = np.array([-2.9, -3, -0.2])
q4 = np.array([0.2, -0.3, -1.9])

T = robot.fkine(q1)

print(f"end-effector pose:\n {T}")
print("Position:", T.t)
print("Rotation:\n", T.R)


# Geometric Jacobian
# J = robot.jacob0(q)
# print(f"Geometric Jacobian Matrix:\n {J}")

robot.plot(
    q1,
    block=True,
    backend="pyplot",
)