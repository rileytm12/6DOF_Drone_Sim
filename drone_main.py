# drone_main.py

# Description:
# This file is the main file for the drone simulation.
# It sets up the simulation and runs the simulation.
# It also plots the results.

import numpy as np
import matplotlib.pyplot as plt

from dynamics_fun import dynamics
from integrator_fun import integrator
from controllers.PID import PID
from transform.quat2euler_fun import quat2euler
from transform.euler2quat_fun import euler2quat
from controllers.innerLoop import innerLoop
from controllers.outerLoop import outerLoop

# Set the time step
dt = 0.01 # s
n_steps = 5000

# Vehicle params (must match those inside dynamics_fun.py)
mass = 2.0
g = 9.81
params = [mass, g]

# Set initial attitude
roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 0.0

quat = euler2quat(
    np.deg2rad(roll_deg),
    np.deg2rad(pitch_deg),
    np.deg2rad(yaw_deg),
)

qw, qx, qy, qz = quat    

state = [0, 0, 0,                # x, y, z
         0, 0, 0,                # vx, vy, vz   
         qw, qx, qy, qz,         # qw, qx, qy, qz
         0, 0, 0]                # p, q, r

# Attitude controller gains
x_pid = PID(1.5, 0.0, 2.0)
y_pid = PID(1.5, 0.0, 2.0)
z_pid = PID(3.0, 0.0, 3.0)
roll_pid  = PID(6.0, 0.0, 2.0)
pitch_pid = PID(6.0, 0.0, 2.0)
yaw_pid   = PID(0.8, 0.1, 0.3)

# Controller bundle
controllers = {
    "x": x_pid,
    "y": y_pid,
    "z": z_pid,
    "roll": roll_pid,
    "pitch": pitch_pid,
    "yaw": yaw_pid
}

# Logging
t_hist = np.zeros(n_steps + 1)
state_hist = np.zeros((n_steps + 1, 13))
state_hist[0, :] = state

# Per-step logs (one entry per control iteration)
x_target_hist  = np.zeros(n_steps)
y_target_hist  = np.zeros(n_steps)
z_target_hist  = np.zeros(n_steps)

print("Initial state: ", state)

j=0
target = [0.0, 0.0, 0.0]

for i in range(n_steps):

    # Outer loop control
    target = outerLoop(state, i, j, n_steps, dt, target)

    # Inner loop control
    control_vec = innerLoop(state, target, params, controllers, dt)

    # Plant
    x_dot = dynamics(state, control_vec)

    # Integrate
    state = integrator(state, x_dot, dt)

    # Log target history
    x_target_hist[i]  = target[0]
    y_target_hist[i]  = target[1]
    z_target_hist[i]  = target[2]

    # Log state history
    t_hist[i + 1] = (i + 1) * dt
    state_hist[i + 1, :] = state

clean_state = [f"{float(x):.2f}" for x in state]
print("Final state: ", clean_state)

# Create history time plotting vector
import numpy as np
t_history_plot = np.linspace(0, n_steps * dt, len(x_target_hist))

# Unpack position
x, y, z, vx, vy, vz = state_hist[:, 0:6].T

# Recover roll/pitch/yaw from the quaternion history (in degrees for readability)
rpy_hist = np.zeros((n_steps + 1, 3))
for k in range(n_steps + 1):
    rpy_hist[k, :] = quat2euler(state_hist[k, 6:10])
rpy_hist_deg = np.rad2deg(rpy_hist)


############################################ Plotting ############################################

########################################################
# Figure 1: 3D trajectory + x/y/z vs time
########################################################

fig1 = plt.figure(figsize=(12, 5))

ax3d = fig1.add_subplot(1, 2, 1, projection='3d')
ax3d.plot(x, y, z, color='tab:blue')
ax3d.plot(x_target_hist, y_target_hist, z_target_hist, color='tab:red', linestyle='--')
ax3d.scatter(x[0], y[0], z[0], color='green', label='start')
ax3d.scatter(x[-1], y[-1], z[-1], color='red', label='end')
ax3d.set_xlabel('x [m]')
ax3d.set_ylabel('y [m]')
ax3d.set_zlabel('z [m]')
ax3d.set_title('Drone trajectory')
ax3d.legend()

ax_pos = fig1.add_subplot(1, 2, 2)
ax_pos.plot(t_hist, x, label='x')
ax_pos.plot(t_hist, y, label='y')
ax_pos.plot(t_hist, z, label='z')
ax_pos.plot(t_history_plot, x_target_hist, label='x_target', linestyle='--')
ax_pos.plot(t_history_plot, y_target_hist, label='y_target', linestyle='--')
ax_pos.plot(t_history_plot, z_target_hist, label='z_target', linestyle='--')
ax_pos.set_xlabel('time [s]')
ax_pos.set_ylabel('position [m]')
ax_pos.set_title('Position vs time')
ax_pos.grid(True)
ax_pos.legend()

fig1.tight_layout()

########################################################
# Figure 2: roll/pitch/yaw vs time on its own
########################################################

fig2, ax_rpy = plt.subplots(figsize=(8, 4))
ax_rpy.plot(t_hist, rpy_hist_deg[:, 0], label='roll')
ax_rpy.plot(t_hist, rpy_hist_deg[:, 1], label='pitch')
ax_rpy.plot(t_hist, rpy_hist_deg[:, 2], label='yaw')
ax_rpy.set_xlabel('time [s]')
ax_rpy.set_ylabel('angle [deg]')
ax_rpy.set_title('Attitude vs time')
ax_rpy.grid(True)
ax_rpy.legend()

fig2.tight_layout()

########################################################
# Figure 3: vx/vy/vz vs time on its own
########################################################

fig3, ax_vel = plt.subplots(figsize=(8, 4))
ax_vel.plot(t_hist, vx, label='vx')
ax_vel.plot(t_hist, vy, label='vy')
ax_vel.plot(t_hist, vz, label='vz')
ax_vel.set_xlabel('time [s]')
ax_vel.set_ylabel('velocity [m/s]')
ax_vel.set_title('Velocity vs time')
ax_vel.grid(True)
ax_vel.legend()

fig3.tight_layout()

########################################################
# Figure 5: top-down 2D view (x vs y) with reference overlay
########################################################

fig5, ax_xy = plt.subplots(figsize=(6, 6))

ax_xy.plot(x_target_hist, y_target_hist, label='reference', linestyle='--', color='tab:gray')
ax_xy.plot(x, y, label='actual', color='tab:blue')
ax_xy.scatter(x[0], y[0], color='green', label='start', zorder=3)
ax_xy.scatter(x[-1], y[-1], color='red', label='end', zorder=3)

ax_xy.set_xlabel('x [m]')
ax_xy.set_ylabel('y [m]')
ax_xy.set_title('Top-down trajectory (x-y plane)')
ax_xy.set_aspect('equal', adjustable='datalim')
ax_xy.grid(True)
ax_xy.legend()

fig5.tight_layout()

plt.show()
