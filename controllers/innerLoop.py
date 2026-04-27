# innerLoop.py

# I/O:  
# I: state, ref, params
# O: control_vec

# Description:
# This function implements the inner loop control for the drone.
# It takes the current state, reference, parameters, controllers, and time step as input.
# It returns the control vector.
# The control vector is a 4-element array representing the thrust and torque commands for the drone.
# The thrust command is the total thrust required to achieve the desired altitude.
# The torque commands are the roll, pitch, and yaw torques required to achieve the desired attitude.
# The control vector is used to update the state of the drone.

import numpy as np

from transform.quat2euler_fun import quat2euler
from controllers.PID import PID
from transform.Tcalc_fun import Tcalc

def innerLoop(state, ref, params, controllers, dt):

    # Yaw reference
    yaw_ref = 0.0 

    # Unpack controllers
    x_pid = controllers["x"]
    y_pid = controllers["y"]
    z_pid = controllers["z"]
    roll_pid  = controllers["roll"]
    pitch_pid = controllers["pitch"]
    yaw_pid   = controllers["yaw"]

    # Unpack state
    x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r = state

    # Unpack ref
    x_ref, y_ref, z_ref = ref

    # Unpack params
    mass, g = params

    # Control: use current attitude to compute PID torque commands
    roll, pitch, yaw = quat2euler([qw, qx, qy, qz])

    # Desired attitude (x and y)
    ax_des = x_pid.update(x_ref - x, -vx, dt)
    ay_des = y_pid.update(y_ref - y, -vy, dt)

    # Desired pitch and roll
    pitch_des = ax_des / g
    roll_des  = -ay_des / g

    # Max roll and pitch angles
    max_angle = np.radians(20)

    # Clamp pitch and roll to max angle
    pitch_des = np.clip(pitch_des, -max_angle, max_angle)
    roll_des  = np.clip(roll_des, -max_angle, max_angle)

    # Desired thrust
    T = mass * (g + z_pid.update((z_ref - z), -vz, dt))

    tau_x = roll_pid.update((roll_des - roll), -p, dt)
    tau_y = pitch_pid.update((pitch_des - pitch), -q, dt)
    tau_z = yaw_pid.update((yaw_ref - yaw), -r, dt)

    # Mixer: (T_total, torques) -> individual rotor thrusts
    control_vec = Tcalc(T, [tau_x, tau_y, tau_z])

    return control_vec