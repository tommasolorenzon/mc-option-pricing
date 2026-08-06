import numpy as np


def terminal_price(Z, r, T, S0, sigma, **kwargs):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)