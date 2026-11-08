#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final QF4204 experiment, run only after the pilot design has been fixed."""

from pathlib import Path

from benchmark import generate_benchmark_prices, load_benchmark_prices
from experiments import estimate_convergence_rates, run_full_experiment
from plots import plot_improvement_factor, plot_rmse_convergence, plot_runtime

# ============================================================
# OPTION PARAMETERS
# ============================================================

S0 = 100.0
K = 100.0
r = 0.10
sigma = 0.20
T = 1.0
q = 0.0

# ============================================================
# FINAL DESIGN CHOSEN AFTER THE PILOT
# ============================================================

D_VALUES = [32, 128, 512]
N_VALUES_BY_D = {
    32: [2**k for k in range(8, 15)],
    128: [2**k for k in range(8, 15)],
    512: [2**k for k in range(8, 14)],
}
R = 20

BENCHMARK_FILE = Path("benchmark_prices_final.csv")
CHECKPOINT_DIR = Path("checkpoints")
RESULT_FILE = Path("final_experiment_results.csv")
CONVERGENCE_FILE = Path("convergence_rates.csv")


def get_final_benchmarks():
    """Generate the expensive final benchmark once, then reuse it."""
    if BENCHMARK_FILE.exists():
        print(f"Loading benchmark prices from {BENCHMARK_FILE}")
        return load_benchmark_prices(BENCHMARK_FILE)

    print("Generating final N=2^20 control-variate benchmarks...")
    return generate_benchmark_prices(
        d_values=D_VALUES,
        N_benchmark=2**20,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        q=q,
        base_seed=12345,
        batch_size=32768,
        save_path=BENCHMARK_FILE
    )


def main():
    benchmark_prices = get_final_benchmarks()

    print("\n--- Final Experimental Grid ---")
    for d in D_VALUES:
        print(f"d={d}: N={N_VALUES_BY_D[d]}, R={R}")

    results = run_full_experiment(
        N_values_by_d=N_VALUES_BY_D,
        d_values=D_VALUES,
        R=R,
        benchmark_prices=benchmark_prices,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        q=q,
        base_seed=1000,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=True,
        progress_repetitions=True
    )

    results.to_csv(RESULT_FILE, index=False)
    print(f"\nSaved final results to {RESULT_FILE}")

    print("\n--- Final Experimental Results ---")
    print(
        results[[
            "N", "d", "method", "mean_price", "RMSE",
            "relative_RMSE", "mean_time", "improvement_factor"
        ]].to_string(index=False)
    )

    convergence = estimate_convergence_rates(results)
    convergence.to_csv(CONVERGENCE_FILE, index=False)

    print("\n--- Empirical Convergence Rates ---")
    print(convergence.to_string(index=False))

    for d in D_VALUES:
        plot_rmse_convergence(results, d)
        plot_improvement_factor(results, d)
        plot_runtime(results, d)


if __name__ == "__main__":
    main()
