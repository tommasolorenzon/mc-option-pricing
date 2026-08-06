import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
