# Variance Reduction in Monte Carlo Option Pricing

Study project to understand the usage of variance reduction techniques in
Monte Carlo option pricing, trying to translate concepts from my expertise
in radiation transport.

A simple package is built and used in jupyter notebooks.

## Layout

    src/mcpricer/
      analytical.py
      models.py
      payoffs.py
    notebooks/
      00_minimal_theory.ipynb


## Running it

    uv sync
    uv run jupyter lab notebooks/00_minimal_theory.ipynb

The notebook is committed with outputs, so it reads without executing at the cost
of big diffs in the history tracking.
