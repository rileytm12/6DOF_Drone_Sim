import numpy as np

def outerLoop(state, i, j, n_steps, dt, target):

    # Unpack state
    x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r = state

    if(i % 500 == 0):
        j += 1
        if(target[0] == 5):
            target[0] = 0
        else:
            target[0] = 5
            
        target[1] = 0.0
        target[2] = 0.0

    return target

