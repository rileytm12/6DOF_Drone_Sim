import numpy as np

# Angular acceleration function
# I/O:
# I: omega, tau, I, I_inv
# O: omega_dot

def angularAccel(omega, tau, I, I_inv):

    omega = np.array(omega)
    tau = np.array(tau)

    gyro = np.cross(omega, I @ omega)

    omega_dot = I_inv @ (tau - gyro)

    return omega_dot