# Franka Research 3 Robot DH parameters

import roboticstoolbox as rtb

robot = rtb.models.DH.Panda()
print(f"robot DH parameters:\n {robot}")

print(f"robot configurations dimension: {robot.n}")

print(f"joint limits:\n {robot.qlim}")

# this DH parameters are same with FR3 DH parameters shown in 
# https://frankarobotics.github.io/docs/robot_specifications.html#kinematic-configuration