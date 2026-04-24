import numpy as np

def quat2euler(q):
    qw, qx, qy, qz = q

    # roll (x-axis)
    sinr = 2*(qw*qx + qy*qz)
    cosr = 1 - 2*(qx*qx + qy*qy)
    roll = np.arctan2(sinr, cosr)

    # pitch (y-axis)
    sinp = 2*(qw*qy - qz*qx)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    # yaw (z-axis)
    siny = 2*(qw*qz + qx*qy)
    cosy = 1 - 2*(qy*qy + qz*qz)
    yaw = np.arctan2(siny, cosy)

    Euler = [roll, pitch, yaw]

    return Euler