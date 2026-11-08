#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:39:09 2026

@author: tangjianxiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting utilities for the QF4204 Sobol Asian option project.

Includes:
    - Property A illustration
    - 2D Sobol projection comparison
    - RMSE convergence plots
    - Relative RMSE plots
    - Improvement-factor plots
    - Runtime plots
"""

import numpy as np
import matplotlib.pyplot as plt

from sobol import (
    generate_randomized_sobol_points
)


# ============================================================
# 1. PROPERTY A VISUALISATION
# ============================================================

def plot_property_A(
        direction_set="2003"):
    """
    Illustrate Property A in two dimensions.

    For d = 2, the first 2^2 = 4 points occupy each of the
    four binary quadrants exactly once.
    """

    U = generate_randomized_sobol_points(
        N=4,
        d=2,
        direction_set=direction_set,
        randomization=False
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        U[:, 0],
        U[:, 1],
        s=80
    )

    # Binary quadrant boundaries
    plt.axvline(
        0.5,
        linestyle="--"
    )

    plt.axhline(
        0.5,
        linestyle="--"
    )

    # Label points
    for i, (x, y) in enumerate(U):

        plt.annotate(
            str(i + 1),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points"
        )

    plt.xlim(
        -0.02,
        1.02
    )

    plt.ylim(
        -0.02,
        1.02
    )

    plt.xticks([
        0,
        0.5,
        1
    ])

    plt.yticks([
        0,
        0.5,
        1
    ])

    plt.xlabel(
        "Dimension 1"
    )

    plt.ylabel(
        "Dimension 2"
    )

    plt.title(
        "Property A in 2D: "
        "One Point in Each Binary Quadrant"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.show()


# ============================================================
# 2. COMPARE ONE 2D PROJECTION
# ============================================================

def plot_projection_comparison(
        N=2**12,
        d=32,
        coord_1=0,
        coord_2=1):
    """
    Compare the same 2D coordinate projection under the
    Joe-Kuo 2003 and Joe-Kuo 2008 D(6) constructions.

    coord_1 and coord_2 use Python's zero-based indexing.
    """

    if coord_1 == coord_2:
        raise ValueError(
            "Choose two different coordinates."
        )

    if (
        coord_1 < 0
        or coord_2 < 0
        or coord_1 >= d
        or coord_2 >= d
    ):

        raise ValueError(
            "Projection coordinates "
            "must lie between 0 and d-1."
        )

    U_2003 = (
        generate_randomized_sobol_points(
            N=N,
            d=d,
            direction_set="2003",
            randomization=False
        )
    )

    U_2008 = (
        generate_randomized_sobol_points(
            N=N,
            d=d,
            direction_set="2008",
            randomization=False
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5)
    )

    axes[0].scatter(
        U_2003[:, coord_1],
        U_2003[:, coord_2],
        s=2
    )

    axes[0].set_title(
        f"Joe-Kuo 2003: "
        f"Dimensions "
        f"{coord_1 + 1}, "
        f"{coord_2 + 1}"
    )

    axes[0].set_xlabel(
        f"Dimension {coord_1 + 1}"
    )

    axes[0].set_ylabel(
        f"Dimension {coord_2 + 1}"
    )

    axes[0].set_xlim(
        0,
        1
    )

    axes[0].set_ylim(
        0,
        1
    )

    axes[0].set_aspect(
        "equal",
        adjustable="box"
    )


    axes[1].scatter(
        U_2008[:, coord_1],
        U_2008[:, coord_2],
        s=2
    )

    axes[1].set_title(
        f"Joe-Kuo 2008 D6: "
        f"Dimensions "
        f"{coord_1 + 1}, "
        f"{coord_2 + 1}"
    )

    axes[1].set_xlabel(
        f"Dimension {coord_1 + 1}"
    )

    axes[1].set_ylabel(
        f"Dimension {coord_2 + 1}"
    )

    axes[1].set_xlim(
        0,
        1
    )

    axes[1].set_ylim(
        0,
        1
    )

    axes[1].set_aspect(
        "equal",
        adjustable="box"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 3. RMSE CONVERGENCE PLOT
# ============================================================

def plot_rmse_convergence(
        results_df,
        d):
    """
    Plot RMSE against sample size on log-log axes.
    """

    subset = results_df[
        results_df["d"] == d
    ]

    plt.figure(
        figsize=(8, 6)
    )

    for method in (
        subset["method"].unique()
    ):

        method_data = subset[
            subset["method"] == method
        ].sort_values("N")

        plt.loglog(
            method_data["N"],
            method_data["RMSE"],
            marker="o",
            label=method
        )

    plt.xlabel(
        "Number of simulated paths N"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        f"RMSE Convergence, d = {d}"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 4. RELATIVE RMSE
# ============================================================

def plot_relative_rmse(
        results_df,
        d):
    """
    Plot relative RMSE against sample size.
    """

    subset = results_df[
        results_df["d"] == d
    ]

    plt.figure(
        figsize=(8, 6)
    )

    for method in (
        subset["method"].unique()
    ):

        method_data = subset[
            subset["method"] == method
        ].sort_values("N")

        plt.loglog(
            method_data["N"],
            method_data[
                "relative_RMSE"
            ],
            marker="o",
            label=method
        )

    plt.xlabel(
        "Number of simulated paths N"
    )

    plt.ylabel(
        "Relative RMSE"
    )

    plt.title(
        f"Relative RMSE, d = {d}"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 5. IMPROVEMENT FACTOR
# ============================================================

def plot_improvement_factor(
        results_df,
        d):
    """
    Plot

        RMSE_MC / RMSE_method

    against sample size.

    MC itself has improvement factor 1.
    """

    subset = results_df[
        results_df["d"] == d
    ]

    plt.figure(
        figsize=(8, 6)
    )

    for method in (
        subset["method"].unique()
    ):

        if method == "MC":
            continue

        method_data = subset[
            subset["method"] == method
        ].sort_values("N")

        plt.plot(
            method_data["N"],
            method_data[
                "improvement_factor"
            ],
            marker="o",
            label=method
        )

    # MC reference level
    plt.axhline(
        y=1.0,
        linestyle="--"
    )

    plt.xscale(
        "log",
        base=2
    )

    plt.xlabel(
        "Number of simulated paths N"
    )

    plt.ylabel(
        "RMSE improvement factor over MC"
    )

    plt.title(
        f"Improvement Relative to MC, "
        f"d = {d}"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 6. COMPUTATION TIME
# ============================================================

def plot_runtime(
        results_df,
        d):
    """
    Compare mean computation time across sample sizes.
    """

    subset = results_df[
        results_df["d"] == d
    ]

    plt.figure(
        figsize=(8, 6)
    )

    for method in (
        subset["method"].unique()
    ):

        method_data = subset[
            subset["method"] == method
        ].sort_values("N")

        plt.plot(
            method_data["N"],
            method_data[
                "mean_time"
            ],
            marker="o",
            label=method
        )

    plt.xscale(
        "log",
        base=2
    )

    plt.xlabel(
        "Number of simulated paths N"
    )

    plt.ylabel(
        "Mean computation time (seconds)"
    )

    plt.title(
        f"Computation Time, d = {d}"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. CONVERGENCE RATE BAR PLOT
# ============================================================

def plot_convergence_rates(
        convergence_df,
        d):
    """
    Plot estimated alpha in

        RMSE ~ N^(-alpha)

    for each method.
    """

    subset = convergence_df[
        convergence_df["d"] == d
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        subset["method"],
        subset["convergence_rate"]
    )

    # MC theoretical benchmark
    plt.axhline(
        y=0.5,
        linestyle="--",
        label="MC theoretical rate = 0.5"
    )

    plt.ylabel(
        "Estimated convergence rate"
    )

    plt.title(
        f"Empirical Convergence Rates, "
        f"d = {d}"
    )

    plt.xticks(
        rotation=30
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 8. PROPERTY A: SATISFIES VS DOES NOT SATISFY
# ============================================================

def plot_property_A_comparison():
    """Simple visual explanation of the 2D Property A condition."""
    satisfies = np.array([
        [0.125, 0.125],
        [0.625, 0.125],
        [0.125, 0.625],
        [0.625, 0.625],
    ])

    does_not = np.array([
        [0.125, 0.125],
        [0.375, 0.375],
        [0.625, 0.625],
        [0.875, 0.875],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, points, title in [
        (axes[0], satisfies, "Satisfies Property A"),
        (axes[1], does_not, "Does Not Satisfy Property A"),
    ]:
        ax.scatter(points[:, 0], points[:, 1], s=80)
        ax.axvline(0.5, linestyle="--")
        ax.axhline(0.5, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("plots.py loaded successfully.")
