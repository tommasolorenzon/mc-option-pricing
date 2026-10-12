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


def mc_antithetic(payoff_fn, model_fn, N, r, T, rng, **params) -> MCResult:
    N = int(N)

    t_start = time.perf_counter()

    Z = rng.normal(size=N // 2)
    S_T_dir = model_fn(Z, r=r, T=T, **params)
    S_T_ant = model_fn(-Z, r=r, T=T, **params)

    avg_payoff = 0.5 * (payoff_fn(S_T_dir, **params) + payoff_fn(S_T_ant, **params))

    discounted = np.exp(-r * T) * avg_payoff

    elapsed = time.perf_counter() - t_start

    return summarize(discounted, n_evals=2 * len(Z), elapsed=elapsed)


def mc_control_variate(payoff_fn, model_fn, N, r, T, rng, c=None, **params) -> MCResult:
    """Subtract the discounted terminal price, whose mean is known exactly.

    Under the risk-neutral measure E[exp(-r*T) * S_T] = S0 * exp(-y*T), so the
    control costs nothing: S_T is already simulated. Leaving c=None estimates
    the optimal coefficient from the same paths, which makes the estimator
    biased at order 1/N; pass a c from a pilot run to remove it.
    """
    N = int(N)

    t_start = time.perf_counter()

    Z = rng.normal(size=N)
    S_T = model_fn(Z, r=r, T=T, **params)

    X = np.exp(-r * T) * payoff_fn(S_T, **params)
    Y = np.exp(-r * T) * S_T
    mean_Y = params["S0"] * np.exp(-params.get("y", 0.0) * T)

    if c is None:
        cov = np.cov(X, Y, ddof=1)
        c = cov[0, 1] / cov[1, 1]

    corrected = X - c * (Y - mean_Y)

    elapsed = time.perf_counter() - t_start

    return summarize(corrected, n_evals=N, elapsed=elapsed)


def mc_importance_sampling(payoff_fn, model_fn, N, r, T, rng, mu, **params) -> MCResult:
    """Sample Z from N(mu, 1) and weight each draw by the likelihood ratio.

    The weight exp(-mu*Z + mu**2/2) is f/g for f = N(0,1) and g = N(mu, 1), so
    the estimator is unbiased for any mu. mu is required rather than defaulted:
    choosing it is the whole technique, and a good choice depends on the model,
    which this function does not know.
    """
    N = int(N)

    t_start = time.perf_counter()

    Z = rng.normal(loc=mu, size=N)
    S_T = model_fn(Z, r=r, T=T, **params)

    weight = np.exp(-mu * Z + 0.5 * mu**2)
    weighted = np.exp(-r * T) * payoff_fn(S_T, **params) * weight

    elapsed = time.perf_counter() - t_start

    return summarize(weighted, n_evals=N, elapsed=elapsed)
