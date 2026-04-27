import numpy as np
import random

def disturbances(wind):

    alpha = 0.98
    slow_var_const = 0.25
    gust = 0.0

    # --- steady wind ---
    base_wind = np.array([3.0, 3.0])

    # --- slow variation ---
    slow_var = alpha * slow_var_const + (1 - alpha) * np.random.normal(0, 1, 2)

    # --- gust events ---
    if np.random.rand() < 0.01:
        gust = np.random.uniform(-5, 5, size=2)

    # --- total wind ---
    wind_disturbance = base_wind + slow_var + gust

    return wind_disturbance