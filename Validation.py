#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validation checks for the QF4204 implementation."""

import numpy as np

from pricing import (
    black_scholes_call,
    exact_geometric_asian_call,
    price_asian_call_mc,
)
from sobol import (
    generate_randomized_sobol_points,
    price_asian_call_rqmc,
)


def validate_european_call(
        S0=100.0, K=100.0, r=0.10, sigma=0.20, T=1.0, q=0.0,
        N=2**16, seed=12345):
    """d=1 arithmetic Asian equals a European call."""
    mc_price, se, elapsed = price_asian_call_mc(
        N=N, d=1, S0=S0, K=K, r=r, sigma=sigma, T=T,
        average_type="arithmetic", q=q, seed=seed
    )
    exact = black_scholes_call(T=T, S0=S0, K=K, r=r, q=q, sigma=sigma)
    lower, upper = mc_price - 1.96 * se, mc_price + 1.96 * se
    return {
        "exact_price": exact,
        "mc_price": mc_price,
        "absolute_error": abs(mc_price - exact),
        "standard_error": se,
        "lower_bound": lower,
        "upper_bound": upper,
        "exact_inside_interval": lower <= exact <= upper,
        "elapsed_time": elapsed,
    }


def validate_geometric_asian_mc(
        d=32, N=2**16,
        S0=100.0, K=100.0, r=0.10, sigma=0.20, T=1.0, q=0.0,
        seed=12345):
    """Validate multidimensional MC against the geometric-Asian closed form."""
    mc_price, se, elapsed = price_asian_call_mc(
        N=N, d=d, S0=S0, K=K, r=r, sigma=sigma, T=T,
        average_type="geometric", q=q, seed=seed
    )
    exact = exact_geometric_asian_call(S0, K, r, sigma, T, d, q)
    lower, upper = mc_price - 1.96 * se, mc_price + 1.96 * se
    return {
        "exact_price": exact,
        "mc_price": mc_price,
        "absolute_error": abs(mc_price - exact),
        "standard_error": se,
        "lower_bound": lower,
        "upper_bound": upper,
        "exact_inside_interval": lower <= exact <= upper,
        "elapsed_time": elapsed,
    }


def validate_geometric_asian_rqmc(
        d=32, N=2**10,
        direction_set="2008", randomization="DS",
        S0=100.0, K=100.0, r=0.10, sigma=0.20, T=1.0, q=0.0,
        seed=12345):
    """Check one RQMC geometric-Asian estimate against the closed form."""
    rqmc_price, elapsed = price_asian_call_rqmc(
        N=N, d=d, S0=S0, K=K, r=r, sigma=sigma, T=T,
        average_type="geometric",
        direction_set=direction_set,
        randomization=randomization,
        q=q,
        seed=seed
    )
    exact = exact_geometric_asian_call(S0, K, r, sigma, T, d, q)
    return {
        "direction_set": direction_set,
        "randomization": randomization,
        "exact_price": exact,
        "rqmc_price": rqmc_price,
        "absolute_error": abs(rqmc_price - exact),
        "elapsed_time": elapsed,
    }


def validate_joe_kuo_2003_points():
    """Check known initial deterministic Joe-Kuo 2003 points."""
    expected = np.array([
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.500],
        [0.750, 0.250, 0.750],
        [0.250, 0.750, 0.250],
        [0.375, 0.375, 0.625],
        [0.875, 0.875, 0.125],
        [0.625, 0.125, 0.375],
        [0.125, 0.625, 0.875],
    ])
    generated = generate_randomized_sobol_points(
        N=8, d=3, direction_set="2003", randomization=False
    )
    if not np.allclose(generated, expected):
        raise AssertionError("Joe-Kuo 2003 deterministic point check failed.")
    return True


def validate_randomization(N=32, d=4):
    """Check same-seed reproducibility and different-seed variation."""
    rows = []
    for direction_set in ("2003", "2008"):
        for randomization in ("DS", "NUS"):
            U1 = generate_randomized_sobol_points(
                N, d, direction_set, randomization, seed=12345
            )
            U2 = generate_randomized_sobol_points(
                N, d, direction_set, randomization, seed=12345
            )
            U3 = generate_randomized_sobol_points(
                N, d, direction_set, randomization, seed=54321
            )
            same = np.allclose(U1, U2)
            different = not np.allclose(U1, U3)
            if not same or not different:
                raise AssertionError(
                    f"Randomization check failed for {direction_set} {randomization}."
                )
            rows.append({
                "direction_set": direction_set,
                "randomization": randomization,
                "same_seed_same": same,
                "different_seed_different": different,
            })
    return rows


def run_all_validations():
    print("\n--- European Call Validation ---")
    european = validate_european_call()
    print(european)

    print("\n--- Geometric Asian MC Validation ---")
    geometric_mc = validate_geometric_asian_mc()
    print(geometric_mc)

    print("\n--- Geometric Asian RQMC Validation ---")
    for direction_set in ("2003", "2008"):
        for randomization in ("DS", "NUS"):
            result = validate_geometric_asian_rqmc(
                direction_set=direction_set,
                randomization=randomization
            )
            print(result)

    print("\n--- Joe-Kuo 2003 Deterministic-Point Check ---")
    validate_joe_kuo_2003_points()
    print("passed")

    print("\n--- Randomization Reproducibility Checks ---")
    for result in validate_randomization():
        print(result)

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    run_all_validations()
