# Variance Reduction in Monte Carlo Option Pricing

Study project on variance reduction for Monte Carlo option pricing, written to map the
techniques onto the ones I use in Monte Carlo for radiation transport. A small package
holds the code, the notebooks develop the concepts.

## At a glance

European call, S0 = 100, K = 100, sigma = 20%, tau = 1, r = 5%, checked against
Black-Scholes. Gain is the variance per payoff evaluation against plain Monte Carlo.

| technique | variance per evaluation | gain |
|---|---|---|
| plain | 217 | 1.00x |
| antithetic | 108 | 2.02x |
| control variate | 31 | 6.91x |
| importance sampling | depends on the strike | 0.7x at the money, about 3000x at K = 200 |

## Summary

**Pricing by Monte Carlo.** The model makes the log-return normal, so a path is one draw
of a standard normal Z and everything after it is deterministic. Draw Z, map it to the
terminal price S_T, compute the payoff `max(S_T - K, 0)`, average the payoffs over the
paths, discount with `exp(-r*tau)`. The discounting is what brings the payoff, received
at maturity, to today's money, which is the price being computed.

**Where plain Monte Carlo fails.** For a strike far from the underlying, the payoff is
rarely non-zero. In our example, with S0 = 100, at K = 200 the stock has to double in a
year. This is roughly 3.3 standard deviations of the log return, so about 100 draws out of
200,000 pay anything. Precision is then set by the number of hits rather than by N, and
the variance is estimated from the same handful of samples, so the error bar is unreliable
as well.

Conceptually this is the same rare-event problem as a deep penetration shielding study in
Monte Carlo for radiation transport, where the tally is carried by the few histories that
get through.

**The three techniques.** Each one bets on a property of the payoff and works only when
the bet holds.

- *Antithetic* pairs every Z with -Z. The bet is monotonicity: a monotone chain makes the
  two legs negatively correlated, measured at -0.50, which halves the variance per
  evaluation. On a payoff symmetric in Z the covariance turns positive and the pair costs
  two evaluations to be noisier than two independent draws.
- *Control variates* subtract a correlated quantity whose mean is known, here the
  discounted terminal price, a martingale with `E[exp(-r*tau)*S_T] = S0*exp(-y*tau)`. The
  reduction is `1/(1-rho^2)`, governed by the correlation alone: rho = 0.9245 predicts
  6.88x against 6.91x measured.
- *Importance sampling* draws from a shifted normal and reweights each draw by the ratio
  of the two densities, which is the particle weight of transport. Several proposal
  densities are tested here. They are heuristic, as are the importance biasing schemes
  commonly used in Monte Carlo for radiation transport, and centring where no rare event
  exists costs more than it saves, which is the 0.7x at the money.

**How the techniques are compared.** Two criteria.

- The payoff evaluations spent, which are deterministic and the same on any machine.
- The figure of merit `1/(R^2 t)`, the usual indicator in Monte Carlo for radiation
  transport. It needs the elapsed time, so it is the cost actually paid, and it is
  specific to the machine.

## Layout

    src/mcpricer/
      analytical.py      Black-Scholes, the reference price
      models.py          GBM, Z -> S_T
      payoffs.py         call, put
      estimators.py      plain MC, antithetic, control variate, importance sampling
    notebooks/
      00_minimal_theory.ipynb      definitions, no-arbitrage, the model, the price
      01_variance_reduction.ipynb  the techniques, measured

## Running it

    uv sync
    uv run jupyter lab notebooks/

The notebooks are committed with outputs, so they read without executing, at the cost of
big diffs in the history.
