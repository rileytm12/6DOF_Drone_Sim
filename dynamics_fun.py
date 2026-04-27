# dynamics_fun.py

# I/O
# I: state vector x, control vector u
#       state = [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
#       u = [T_1, T_2, T_3, T_4]
# O: state derivative dx/dt
#       x_dot = [vx, vy, vz, ax, ay, az, p, q, r, alpha, beta, gamma]

import numpy as np

from transform.body2global_fun import body2global
from transform.angularAccel_fun import angularAccel
from disturbances_fun import disturbances

def dynamics(state, u, wind):

    # Geometry / inertia
    L = 0.25                              # m, arm length
    k_m = 0.01                            # yaw-torque per rotor thrust
    mass = 2.0                            # kg
    g = 9.81                              # m/s^2
    I = np.diag([0.02, 0.02, 0.04])
    I_inv = np.linalg.inv(I)
    C_d = 1 # Drag coefficient
    A = 0.15 # m^2, drone area
    rho = 1.225 # kg/m^3, air density

    # Unpack
    x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r = state
    T_1, T_2, T_3, T_4 = u

    # Translational dynamics: thrust is +Z in body frame, rotate to global, add gravity.
    T_total = T_1 + T_2 + T_3 + T_4
    thrust_body = np.array([0.0, 0.0, T_total])
    thrust_global = body2global([qw, qx, qy, qz], thrust_body)

    # Disturbances
    disturbance = disturbances(wind)

    # Drone relative velocity
    v_rel = np.array([vx-disturbance[0], vy-disturbance[1]])

    # Drag force: must oppose relative velocity in BOTH directions, so use |v_rel|*v_rel
    drag_force = -0.5 * C_d * A * rho * np.abs(v_rel) * v_rel
    drag_force = np.array([drag_force[0], drag_force[1], 0])

    force_x = thrust_global[0] + drag_force[0] # x-axis force
    force_y = thrust_global[1] + drag_force[1] # y-axis force
    force_z = thrust_global[2] - mass * g + drag_force[2] # z-axis force

    ax = force_x / mass
    ay = force_y / mass
    az = force_z / mass

    # Body-frame torques from rotor thrusts.
    # NOTE: this convention MUST match the A matrix inside Tcalc_fun.py.
    tau_x = L   * (T_2 - T_4)                   # roll
    tau_y = L   * (T_3 - T_1)                   # pitch
    tau_z = k_m * (T_1 - T_2 + T_3 - T_4)       # yaw
    tau = np.array([tau_x, tau_y, tau_z])

    # Rotational dynamics
    p_dot, q_dot, r_dot = angularAccel([p, q, r], tau, I, I_inv)

    # Quaternion kinematics: q_dot = 0.5 * Omega(omega) * q
    Omega = np.array([[0, -p, -q, -r],
                      [p,  0, -r,  q],
                      [q,  r,  0, -p],
                      [r, -q,  p,  0]])
    q_vec = np.array([qw, qx, qy, qz])
    quat_dot = 0.5 * Omega @ q_vec

    x_dot = [vx, vy, vz,
             ax, ay, az,
             quat_dot[0], quat_dot[1], quat_dot[2], quat_dot[3],
             p_dot, q_dot, r_dot]

    return x_dot, disturbance
