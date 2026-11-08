#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Experimental framework for MC vs randomized Sobol QMC."""

from pathlib import Path

import numpy as np
import pandas as pd

from pricing import price_asian_call_mc
from sobol import price_asian_call_rqmc

RQMC_METHODS = {
    "2003_DS": {"direction_set": "2003", "randomization": "DS"},
    "2003_Owen": {"direction_set": "2003", "randomization": "NUS"},
    "2008_DS": {"direction_set": "2008", "randomization": "DS"},
    "2008_Owen": {"direction_set": "2008", "randomization": "NUS"},
}

ALL_METHODS = ["MC", "2003_DS", "2003_Owen", "2008_DS", "2008_Owen"]


def run_all_methods_once(N, d, S0, K, r, sigma, T, q=0.0, seed=None):
    """Run MC and all four RQMC methods once."""
    results = {}

    mc_price, mc_se, mc_time = price_asian_call_mc(
        N=N, d=d, S0=S0, K=K, r=r, sigma=sigma, T=T,
        average_type="arithmetic", q=q, seed=seed
    )
    results["MC"] = {
        "price": mc_price,
        "time": mc_time,
        "within_run_se": mc_se
    }

    for method_name, settings in RQMC_METHODS.items():
        price, elapsed = price_asian_call_rqmc(
            N=N,
            d=d,
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            average_type="arithmetic",
            direction_set=settings["direction_set"],
            randomization=settings["randomization"],
            q=q,
            seed=seed
        )
        results[method_name] = {"price": price, "time": elapsed}

    return results


def repeated_experiment(
        N, d, R, S0, K, r, sigma, T,
        q=0.0, base_seed=1000, progress=False):
    """Repeat all five methods R times with independent seeds across repetitions."""
    if R < 2:
        raise ValueError("R should be at least 2 for repeated-run error estimates.")

    prices = {method: np.zeros(R) for method in ALL_METHODS}
    times = {method: np.zeros(R) for method in ALL_METHODS}

    for rep in range(R):
        seed = base_seed + rep
        results = run_all_methods_once(
            N=N, d=d, S0=S0, K=K, r=r, sigma=sigma, T=T, q=q, seed=seed
        )

        for method in ALL_METHODS:
            prices[method][rep] = results[method]["price"]
            times[method][rep] = results[method]["time"]

        if progress and ((rep + 1) % 5 == 0 or rep == 0 or rep + 1 == R):
            print(f"    repetition {rep + 1}/{R}")

    return prices, times


def summarize_experiment(prices, times, benchmark_price, N=None, d=None):
    """Calculate RMSE, relative RMSE, bias, variability, runtime, and MC improvement."""
    rows = []

    for method in ALL_METHODS:
        estimates = np.asarray(prices[method], dtype=float)
        method_times = np.asarray(times[method], dtype=float)
        errors = estimates - benchmark_price

        rmse = np.sqrt(np.mean(errors**2))
        mean_price = np.mean(estimates)
        estimate_sd = np.std(estimates, ddof=1)

        rows.append({
            "N": N,
            "d": d,
            "method": method,
            "mean_price": mean_price,
            "bias": mean_price - benchmark_price,
            "RMSE": rmse,
            "relative_RMSE": rmse / abs(benchmark_price),
            "estimate_SD": estimate_sd,
            "mean_standard_error": estimate_sd / np.sqrt(len(estimates)),
            "mean_time": np.mean(method_times)
        })

    summary = pd.DataFrame(rows)
    mc_rmse = summary.loc[summary["method"] == "MC", "RMSE"].iloc[0]
    summary["improvement_factor"] = mc_rmse / summary["RMSE"]
    return summary


def run_sample_size_experiment(
        N_values, d, R, benchmark_price,
        S0, K, r, sigma, T, q=0.0,
        base_seed=1000, verbose=True,
        checkpoint_file=None, resume=False,
        progress_repetitions=False):
    """Run all methods over a list of N for one dimension.

    If checkpoint_file is supplied, cumulative results are saved after every N.
    With resume=True, completed N values are skipped.
    """
    checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
    existing = pd.DataFrame()

    if resume and checkpoint_file and checkpoint_file.exists():
        existing = pd.read_csv(checkpoint_file)

    completed_N = set()
    if not existing.empty:
        counts = existing.groupby("N")["method"].nunique()
        completed_N = set(counts[counts >= len(ALL_METHODS)].index.astype(int))

    summaries = []
    if not existing.empty:
        summaries.append(existing)

    for N_index, N in enumerate(N_values):
        if N in completed_N:
            if verbose:
                print(f"Skipping completed d={d}, N={N}")
            continue

        if verbose:
            print(f"Running d={d}, N={N}, R={R}")

        seed_for_N = base_seed + N_index * 10000
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
            base_seed=seed_for_N,
            progress=progress_repetitions
        )

        summary = summarize_experiment(
            prices=prices,
            times=times,
            benchmark_price=benchmark_price,
            N=N,
            d=d
        )
        summaries.append(summary)

        if checkpoint_file is not None:
            current = pd.concat(summaries, ignore_index=True)
            current = current.drop_duplicates(subset=["N", "d", "method"], keep="last")
            current = current.sort_values(["N", "method"]).reset_index(drop=True)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            current.to_csv(checkpoint_file, index=False)

    if not summaries:
        return pd.DataFrame()

    result = pd.concat(summaries, ignore_index=True)
    result = result.drop_duplicates(subset=["N", "d", "method"], keep="last")
    return result.sort_values(["N", "method"]).reset_index(drop=True)


def run_full_experiment(
        N_values_by_d, d_values, R, benchmark_prices,
        S0, K, r, sigma, T, q=0.0,
        base_seed=1000, verbose=True,
        checkpoint_dir=None, resume=False,
        progress_repetitions=False):
    """Run the complete experiment over several dimensions."""
    all_results = []
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    for d_index, d in enumerate(d_values):
        if d not in benchmark_prices:
            raise ValueError(f"No benchmark price provided for d={d}.")
        if d not in N_values_by_d:
            raise ValueError(f"No N-values provided for d={d}.")

        checkpoint_file = None
        if checkpoint_dir is not None:
            checkpoint_file = checkpoint_dir / f"checkpoint_d{d}.csv"

        results_d = run_sample_size_experiment(
            N_values=N_values_by_d[d],
            d=d,
            R=R,
            benchmark_price=benchmark_prices[d],
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            q=q,
            base_seed=base_seed + d_index * 1_000_000,
            verbose=verbose,
            checkpoint_file=checkpoint_file,
            resume=resume,
            progress_repetitions=progress_repetitions
        )
        all_results.append(results_d)

    return pd.concat(all_results, ignore_index=True)


def estimate_convergence_rates(results_df):
    """Estimate empirical alpha in RMSE ~= c N^(-alpha) using log-log OLS."""
    required = {"N", "d", "method", "RMSE"}
    if not required.issubset(results_df.columns):
        raise ValueError("results_df must contain N, d, method and RMSE.")

    rows = []
    for d in sorted(results_df["d"].unique()):
        for method in ALL_METHODS:
            subset = results_df[
                (results_df["d"] == d) & (results_df["method"] == method)
            ].sort_values("N")

            if len(subset) < 2:
                continue
            if np.any(subset["RMSE"].to_numpy() <= 0):
                raise ValueError("RMSE values must be positive for log-log regression.")

            log_N = np.log(subset["N"].to_numpy(dtype=float))
            log_RMSE = np.log(subset["RMSE"].to_numpy(dtype=float))
            slope, intercept = np.polyfit(log_N, log_RMSE, 1)

            rows.append({
                "d": d,
                "method": method,
                "slope": slope,
                "convergence_rate": -slope,
                "intercept": intercept,
                "n_points": len(subset)
            })

    return pd.DataFrame(rows)


def save_results(results_df, file_path):
    """Save experimental results to CSV."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    print("experiments.py loaded successfully.")
