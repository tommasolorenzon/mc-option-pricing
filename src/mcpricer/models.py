import numpy as np


def terminal_price(Z, r, T, S0, sigma, y=0.0, **kwargs):
    """Terminal price under geometric Brownian motion.

    Exact solution at T, so no time stepping is needed for a European payoff.
    The -sigma**2/2 in the drift is what keeps E[S_T] = S0*exp((r-y)*T).
    """
    return S0 * np.exp((r - y - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
