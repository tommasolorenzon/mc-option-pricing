import numpy as np


def call_payoff(S_T, K, **kwargs):
    return np.maximum(S_T - K, 0.0)


def put_payoff(S_T, K, **kwargs):
    return np.maximum(K - S_T, 0.0)