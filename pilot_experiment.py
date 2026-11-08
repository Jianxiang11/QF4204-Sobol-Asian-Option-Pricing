#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pilot experiments used to choose N, d, and R before the final experiment.

It answers three design questions:

1. Which dimensions should be studied?
2. Is R=10 stable enough, or should we increase to R=20?
3. How expensive is Owen/NUS at d=512, and what N_max is practical?
"""

import numpy as np
import pandas as pd

from benchmark import generate_benchmark_prices
from experiments import (
    ALL_METHODS,
    estimate_convergence_rates,
    repeated_experiment,
    summarize_experiment,
)

# ============================================================
# OPTION PARAMETERS
# ============================================================

S0 = 100.0
K = 100.0
r = 0.10
sigma = 0.20
T = 1.0
q = 0.0

# We deliberately test a moderate, high, and very-high dimension.
D_VALUES = [32, 128, 512]

# A preliminary benchmark is enough for pilot design decisions.
# The final experiment can later regenerate a more accurate 2^20 benchmark.
PILOT_BENCHMARK_N = 2**18


def build_pilot_benchmarks():
    print("\n=== Preliminary benchmark prices for the pilot ===")
    return generate_benchmark_prices(
        d_values=D_VALUES,
        N_benchmark=PILOT_BENCHMARK_N,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        q=q,
        base_seed=12345,
        batch_size=16384,
        save_path="pilot_benchmark_prices.csv"
    )


# ============================================================
# PILOT 1: SAMPLE SIZE AND REPETITION STABILITY
# ============================================================

def run_accuracy_and_R_pilot(benchmark_prices):
    """Use a sparse N-grid for d=32 and d=128.

    We run 20 repetitions once. The first 10 repetitions represent the
    initial R=10 pilot; all 20 represent the extended pilot. This avoids
    paying for two separate simulations and lets us compare R=10 vs R=20
    using exactly the same first ten randomizations.
    """
    d_values = [32, 128]
    N_values = [256, 1024, 4096, 16384]
    R_max = 20

    rows_R10 = []
    rows_R20 = []

    print("\n=== Pilot 1: sparse N-grid and R stability ===")
    print(f"d values: {d_values}")
    print(f"N values: {N_values}")
    print("Compare first R=10 repetitions with all R=20 repetitions.\n")

    for d_index, d in enumerate(d_values):
        for N_index, N in enumerate(N_values):
            print(f"Running pilot d={d}, N={N}, R={R_max}")

            base_seed = 1000 + d_index * 1_000_000 + N_index * 10_000
            prices, times = repeated_experiment(
                N=N,
                d=d,
                R=R_max,
                S0=S0,
                K=K,
                r=r,
                sigma=sigma,
                T=T,
                q=q,
                base_seed=base_seed,
                progress=True
            )

            prices_10 = {method: values[:10] for method, values in prices.items()}
            times_10 = {method: values[:10] for method, values in times.items()}

            summary_10 = summarize_experiment(
                prices_10, times_10, benchmark_prices[d], N=N, d=d
            )
            summary_10["R"] = 10
            rows_R10.append(summary_10)

            summary_20 = summarize_experiment(
                prices, times, benchmark_prices[d], N=N, d=d
            )
            summary_20["R"] = 20
            rows_R20.append(summary_20)

    results_R10 = pd.concat(rows_R10, ignore_index=True)
    results_R20 = pd.concat(rows_R20, ignore_index=True)

    results_R10.to_csv("pilot_accuracy_R10.csv", index=False)
    results_R20.to_csv("pilot_accuracy_R20.csv", index=False)

    conv_R10 = estimate_convergence_rates(results_R10)
    conv_R20 = estimate_convergence_rates(results_R20)
    conv_R10["R"] = 10
    conv_R20["R"] = 20

    conv_R10.to_csv("pilot_convergence_R10.csv", index=False)
    conv_R20.to_csv("pilot_convergence_R20.csv", index=False)

    comparison = conv_R10[["d", "method", "convergence_rate"]].rename(
        columns={"convergence_rate": "rate_R10"}
    ).merge(
        conv_R20[["d", "method", "convergence_rate"]].rename(
            columns={"convergence_rate": "rate_R20"}
        ),
        on=["d", "method"],
        how="inner"
    )
    comparison["absolute_change"] = np.abs(
        comparison["rate_R20"] - comparison["rate_R10"]
    )
    comparison.to_csv("pilot_R10_vs_R20_convergence.csv", index=False)

    print("\n--- Empirical convergence-rate comparison ---")
    print(comparison.to_string(index=False))
    print(
        "\nInterpretation: if R=20 gives noticeably more stable/plausible "
        "rates than the first R=10 runs, keep R=20 for the final experiment."
    )

    return results_R10, results_R20, comparison


# ============================================================
# PILOT 2: HIGH-DIMENSIONAL TIMING
# ============================================================

def run_d512_timing_pilot(benchmark_prices):
    """Measure d=512 runtime with only R=3 before choosing final N_max."""
    d = 512
    N_values = [256, 1024, 4096]
    R = 3
    summaries = []

    print("\n=== Pilot 2: d=512 timing ===")
    print(f"N values: {N_values}, R={R}")

    for N_index, N in enumerate(N_values):
        print(f"Running timing pilot d={d}, N={N}, R={R}")
        prices, times = repeated_experiment(
            N=N,
            d=d,
            R=R,
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            q=q,
            base_seed=3_000_000 + N_index * 10_000,
            progress=True
        )
        summary = summarize_experiment(
            prices, times, benchmark_prices[d], N=N, d=d
        )
        summary["R"] = R
        summaries.append(summary)

    results = pd.concat(summaries, ignore_index=True)
    results.to_csv("pilot_timing_d512_R3.csv", index=False)

    print("\n--- Measured d=512 timing ---")
    print(
        results[["N", "method", "mean_time"]]
        .to_string(index=False)
    )

    # Use the measured Owen/NUS cost at N=4096 to estimate larger powers of two.
    owen_4096 = results[
        (results["N"] == 4096)
        & (results["method"].isin(["2003_Owen", "2008_Owen"]))
    ]["mean_time"]

    mean_4096 = float(owen_4096.mean())
    estimated_8192 = 2.0 * mean_4096
    estimated_16384 = 4.0 * mean_4096

    print("\n--- Rough Owen/NUS extrapolation assuming near-linear scaling in N ---")
    print(f"Measured mean at N=4096: {mean_4096:.2f} s/method/repetition")
    print(f"Estimated N=8192:        {estimated_8192:.2f} s/method/repetition")
    print(f"Estimated N=16384:       {estimated_16384:.2f} s/method/repetition")

    # Candidate final cost if later choosing R=20.
    candidate_R = 20
    minutes_8192 = estimated_8192 * 2 * candidate_R / 60.0
    minutes_16384 = estimated_16384 * 2 * candidate_R / 60.0

    print("\nIf the final experiment later uses R=20 and both Owen methods:")
    print(f"N=8192 Owen-only cost:  ~{minutes_8192:.1f} min")
    print(f"N=16384 Owen-only cost: ~{minutes_16384:.1f} min")
    print(
        "Use these timings to decide whether N=16384 at d=512 is worth the "
        "additional computation."
    )

    return results


def main():
    benchmark_prices = build_pilot_benchmarks()
    run_accuracy_and_R_pilot(benchmark_prices)
    run_d512_timing_pilot(benchmark_prices)

    print("\nPilot complete. Review the CSV files before fixing the final grid.")


if __name__ == "__main__":
    main()
