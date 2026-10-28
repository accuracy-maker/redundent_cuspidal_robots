# Cuspidal 3R Robot
A cuspidal robot means its IK solutions can be transfered without crossing a singularity surface, which provides the challenges for path planning as there might not be a repeatable continuous path in configuration space existing.

This DH parameters used here is:
| θⱼ | dⱼ | aⱼ | αⱼ |
|---|---:|---:|---:|
| q₁ | 0 | 1 | −90.0° |
| q₂ | 1 | 2 | 90.0° |
| q₃ | 0 | 1.5 | 0.0° |

![config-space](/figures/cuspidal_3R_robot.png)


![config-space](/figures/config_space.png)

This figure shows that there are four IK solutions for the same target poses where $q_2$ and $q_3$ can transfer to each other without crossing the singularity surface.

![config-space](/figures/workspace.png)

This figure shows that workspace of this cuspidal 3R robot where real line is the boundaries. The outter boundary is the maximum workspace boundaries. The inner boundary is the due to the inner singularity. The red dashed line is the workspace path when $q_2$ transfers to $q_3$ as a straight line.

## Path Tracing
Cuspidal robot has path planning issue because of the non-singular transition between different IK solutions. Some paths the robot might just trace once. One example is the this path as shown below:

![3d_param1](/figures/3d_params2.png)

When the robot trace the loop and goes back to the origin, the posture are not recovered and goes to another different IK solution in the same aspect as shown here:

![3d_param1](/figures/3d_params1.png)

Therefore, it is no way the robot can repeatly follow this path.