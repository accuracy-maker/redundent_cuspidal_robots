import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3

link1 = rtb.RevoluteDH(a=1.0, alpha=np.pi / 2)
link2 = rtb.RevoluteDH(a=1.0, alpha=np.pi / 2)
link3 = rtb.RevoluteDH(a=1.0)

robot = rtb.DHRobot([link1, link2, link3], name="3R Orthogonal-Axis Robot")
print(robot)

q0 = [0,0,0]

robot.plot(
    q0,
    block=True,
    backend="pyplot",
)