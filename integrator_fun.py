# integrator_fun.py
# I/O
# I: state derivative x_dot, time step dt
# O: state x

import numpy as np

def integrator(state, x_dot, dt):

    # Unpack the state derivative
    x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, qw_dot, qx_dot, qy_dot, qz_dot, p_dot, q_dot, r_dot = x_dot

    # Unpack the state
    x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r = state

    # Integrate the state derivative
    x = x + x_dot * dt
    y = y + y_dot * dt
    z = z + z_dot * dt
    vx = vx + vx_dot * dt
    vy = vy + vy_dot * dt
    vz = vz + vz_dot * dt
    qw = qw + qw_dot * dt
    qx = qx + qx_dot * dt
    qy = qy + qy_dot * dt
    qz = qz + qz_dot * dt
    p = p + p_dot * dt
    q = q + q_dot * dt
    r = r + r_dot * dt

    # Normalize the quaternion
    quat = np.array([qw, qx, qy, qz])
    quat = quat / np.linalg.norm(quat)

    qw, qx, qy, qz = quat

    # Pack the state
    new_state = [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]

    # Return the updated state
    return new_state