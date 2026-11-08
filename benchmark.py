#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""High-accuracy arithmetic-Asian benchmark using a geometric control variate."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pricing import exact_geometric_asian_call


def _asian_payoffs_batch(Z, S0, K, r, sigma, T, q=0.0):
    """Compute arithmetic and geometric Asian payoffs for one batch.

    This avoids storing a full N x d stock-price matrix for the whole benchmark.
    """
    n, d = Z.shape
    dt = T / d
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    log_S = np.full(n, np.log(S0), dtype=float)
    arithmetic_sum = np.zeros(n, dtype=float)
    log_sum = np.zeros(n, dtype=float)

    for j in range(d):
        log_S += drift + diffusion * Z[:, j]
        S = np.exp(log_S)
        arithmetic_sum += S
        log_sum += log_S

    arithmetic_average = arithmetic_sum / d
    geometric_average = np.exp(log_sum / d)
    discount = np.exp(-r * T)

    arithmetic_payoff = discount * np.maximum(arithmetic_average - K, 0.0)
    geometric_payoff = discount * np.maximum(geometric_average - K, 0.0)
    return arithmetic_payoff, geometric_payoff


def price_arithmetic_asian_control_variate(
        N, d, S0, K, r, sigma, T,
        q=0.0, seed=None, batch_size=32768):
    """Estimate an arithmetic Asian price using a geometric control variate.

    The calculation is batched so N=2^20 and d=512 do not require several GB
    of memory at once.
    """
    if N <= 1:
        raise ValueError("N must be greater than 1.")
    if d <= 0:
        raise ValueError("d must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rng = np.random.default_rng(seed)
    start = perf_counter()

    sum_a = 0.0
    sum_g = 0.0
    sum_a2 = 0.0
    sum_g2 = 0.0
    sum_ag = 0.0
    processed = 0

    while processed < N:
        n_batch = min(batch_size, N - processed)
        Z = rng.standard_normal((n_batch, d))
        A, G = _asian_payoffs_batch(Z, S0, K, r, sigma, T, q)

        sum_a += float(np.sum(A))
        sum_g += float(np.sum(G))
        sum_a2 += float(np.dot(A, A))
        sum_g2 += float(np.dot(G, G))
        sum_ag += float(np.dot(A, G))
        processed += n_batch

    mean_a = sum_a / N
    mean_g = sum_g / N

    # Unbiased sample moments.
    var_a = (sum_a2 - N * mean_a**2) / (N - 1)
    var_g = (sum_g2 - N * mean_g**2) / (N - 1)
    cov_ag = (sum_ag - N * mean_a * mean_g) / (N - 1)

    if var_g <= 0:
        raise RuntimeError("Geometric-payoff variance is non-positive.")

    beta = cov_ag / var_g
    exact_g = exact_geometric_asian_call(
        S0=S0, K=K, r=r, sigma=sigma, T=T, d=d, q=q
    )

    price = mean_a - beta * (mean_g - exact_g)
    adjusted_variance = var_a + beta**2 * var_g - 2.0 * beta * cov_ag
    adjusted_variance = max(adjusted_variance, 0.0)
    standard_error = np.sqrt(adjusted_variance / N)
    correlation = cov_ag / np.sqrt(var_a * var_g)

    return {
        "price": float(price),
        "standard_error": float(standard_error),
        "beta": float(beta),
        "correlation": float(correlation),
        "elapsed_time": perf_counter() - start,
        "N": int(N),
        "d": int(d)
    }


def generate_benchmark_prices(
        d_values, N_benchmark, S0, K, r, sigma, T,
        q=0.0, base_seed=12345, batch_size=32768,
        save_path=None):
    """Generate benchmark prices for all requested dimensions."""
    rows = []
    prices = {}

    for i, d in enumerate(d_values):
        result = price_arithmetic_asian_control_variate(
            N=N_benchmark,
            d=d,
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            q=q,
            seed=base_seed + i,
            batch_size=batch_size
        )
        prices[d] = result["price"]
        rows.append(result)

        print(
            f"d={d}: benchmark={result['price']:.8f}, "
            f"SE={result['standard_error']:.8f}, "
            f"time={result['elapsed_time']:.2f}s"
        )

    if save_path is not None:
        pd.DataFrame(rows).to_csv(save_path, index=False)

    return prices


def load_benchmark_prices(file_path):
    """Load a benchmark CSV created by generate_benchmark_prices."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)
    required = {"d", "price"}
    if not required.issubset(df.columns):
        raise ValueError("Benchmark CSV must contain columns 'd' and 'price'.")

    return {int(row.d): float(row.price) for row in df.itertuples(index=False)}
