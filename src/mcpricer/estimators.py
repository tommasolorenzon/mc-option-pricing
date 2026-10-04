import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    """Price, error bar and cost of one Monte Carlo run."""

    price: float
    std_err: float
    var: float
    n_samples: int
    n_evals: int
    elapsed: float | None = None

    @property
    def rel_err(self) -> float:
        return self.std_err / self.price

    @property
    def var_per_eval(self) -> float:
        return self.var * self.n_evals / self.n_samples

    @property
    def fom(self) -> float | None:
        if self.elapsed is None:
            return None
        return 1.0 / (self.rel_err**2 * self.elapsed)


def summarize(samples, n_evals, elapsed=None) -> MCResult:
    """Price and error bar from a sample of discounted payoffs.

    n_evals is the number of payoff evaluations spent. It is not len(samples)
    for every estimator: one antithetic sample is a +Z/-Z pair and costs two.
    """
    samples = np.asarray(samples)
    n = samples.size

    price = samples.mean()
    var = samples.var(ddof=1)
    std_err = np.sqrt(var / n)

    return MCResult(price, std_err, var, n, n_evals, elapsed)


def mc_price_european(payoff_fn, model_fn, N, r, T, rng, **params) -> MCResult:
    """Plain Monte Carlo: average the discounted payoff over N draws of Z."""
    N = int(N)

    t_start = time.perf_counter()

    Z = rng.normal(size=N)
    S_T = model_fn(Z, r=r, T=T, **params)
    discounted = np.exp(-r * T) * payoff_fn(S_T, **params)

    elapsed = time.perf_counter() - t_start

    return summarize(discounted, n_evals=N, elapsed=elapsed)


def mc_antithetic():
    raise NotImplementedError


def mc_control_variate():
    raise NotImplementedError


def mc_importance_sampling():
    raise NotImplementedError
