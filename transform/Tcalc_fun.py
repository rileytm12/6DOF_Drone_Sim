# Tcalc_fun.py
#
# Control allocator: map a commanded total thrust + body torques to the four
# individual rotor thrusts.
# I: T_total (N), tau_vec = [tau_x, tau_y, tau_z] (Nm)
# O: [T_1, T_2, T_3, T_4] (N)

import numpy as np


def Tcalc(T_total, tau_vec):

    L = 0.25
    k_m = 0.01

    # Rows: [total thrust, roll tau_x, pitch tau_y, yaw tau_z].
    # MUST match the torque convention inside dynamics_fun.py.
    A = np.array([[1,    1,    1,    1  ],
                  [0,    L,    0,   -L  ],
                  [-L,   0,    L,    0  ],
                  [k_m, -k_m,  k_m, -k_m]])

    command_vec = np.array([T_total, tau_vec[0], tau_vec[1], tau_vec[2]])

    T_vec = np.linalg.solve(A, command_vec)

    return T_vec
