# PID.py
# I/O:
# I: error, error_dot, dt
# O: control_signal

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0

    def update(self, error, error_dot, dt):
        self.integral += error * dt
        return self.kp*error + self.ki*self.integral + self.kd*error_dot