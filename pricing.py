#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pricing utilities for the QF4204 Asian-option experiment."""

from time import perf_counter

import numpy as np
from scipy.stats import norm


def stock_paths_from_normals(Z, S0, r, sigma, T, q=0.0):
    """Construct discretely monitored Black-Scholes paths from N(0,1) inputs."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array with shape (N, d).")

    _, d = Z.shape
    if d <= 0:
        raise ValueError("d must be positive.")

    dt = T / d
    times = dt * np.arange(1, d + 1)
    W = np.sqrt(dt) * np.cumsum(Z, axis=1)

    return S0 * np.exp(
        (r - q - 0.5 * sigma**2) * times
        + sigma * W
    )


def asian_call_payoffs_from_normals(
        Z, S0, K, r, sigma, T,
        average_type="arithmetic", q=0.0):
    """Return discounted arithmetic- or geometric-Asian call payoffs."""
    S = stock_paths_from_normals(Z, S0, r, sigma, T, q)

    if average_type == "arithmetic":
        path_average = np.mean(S, axis=1)
    elif average_type == "geometric":
        path_average = np.exp(np.mean(np.log(S), axis=1))
    else:
        raise ValueError("average_type must be 'arithmetic' or 'geometric'.")

    return np.exp(-r * T) * np.maximum(path_average - K, 0.0)


def price_asian_call_mc(
        N, d, S0, K, r, sigma, T,
        average_type="arithmetic", q=0.0, seed=None):
    """Price an Asian call with ordinary Monte Carlo."""
    if N <= 1:
        raise ValueError("N must be greater than 1.")
    if d <= 0:
        raise ValueError("d must be positive.")

    rng = np.random.default_rng(seed)
    start = perf_counter()

    Z = rng.standard_normal((N, d))
    payoffs = asian_call_payoffs_from_normals(
        Z, S0, K, r, sigma, T,
        average_type=average_type, q=q
    )

    price = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(N))
    elapsed = perf_counter() - start

    return price, se, elapsed


def black_scholes_call(T, S0, K, r, q, sigma):
    """Black-Scholes price of a European call with continuous dividend yield q."""
    d1 = (
        np.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return float(
        S0 * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def exact_geometric_asian_call(S0, K, r, sigma, T, d, q=0.0):
    """Closed-form price for a discretely monitored geometric Asian call."""
    if d <= 0:
        raise ValueError("d must be positive.")

    mean_log_G = (
        np.log(S0)
        + (r - q - 0.5 * sigma**2)
        * T * (d + 1) / (2 * d)
    )

    var_log_G = (
        sigma**2 * T
        * (d + 1) * (2 * d + 1)
        / (6 * d**2)
    )

    std_log_G = np.sqrt(var_log_G)
    d1 = (mean_log_G - np.log(K) + var_log_G) / std_log_G
    d2 = d1 - std_log_G

    return float(
        np.exp(-r * T)
        * (
            np.exp(mean_log_G + 0.5 * var_log_G) * norm.cdf(d1)
            - K * norm.cdf(d2)
        )
    )
