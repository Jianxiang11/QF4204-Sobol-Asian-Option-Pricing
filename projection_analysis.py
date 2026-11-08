# ============================================================
# projection_actual_points.py
#
# Week 12 supplementary analysis:
# Plot the ACTUAL Sobol points used in one experiment setting
# (e.g. d=512, N=8192, 2003_DS vs 2008_DS),
# and also compute 2D t-values for the same coordinate pair.
#
# This does NOT modify the pricing experiment.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sobol import generate_randomized_sobol_points


# ============================================================
# OUTPUT FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "projection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

# Exact experiment configuration you want to analyse
D = 512
N = 8192
M = int(np.log2(N))   # here M = 13

# Compare these two exact methods from the experiment
METHOD_LEFT = "2003_DS"
METHOD_RIGHT = "2008_DS"

# Use a fixed seed / repetition
# If your experiment used base_seed=1000 and repetition-specific seeds,
# choose one of those seeds here. For example, repetition 1 might be 1001.
SEED = 2051000

# ------------------------------------------------------------
# Choose ONE of the following:
#
# 1) Manually specify a coordinate pair:
#    USE_MANUAL_PAIR = True
#    MANUAL_PAIR = (2, 15)
#
# 2) Automatically search for an interesting pair among the
#    first SCAN_MAX_DIM dimensions, then plot that pair.
# ------------------------------------------------------------

USE_MANUAL_PAIR = False
MANUAL_PAIR = (2, 15)

SCAN_MAX_DIM = 512   


# ============================================================
# METHOD PARSER
# ============================================================

def parse_method(method_name):
    """
    Convert strings like:
        '2003_DS', '2008_DS', '2003_Owen', '2008_Owen'
    into:
        direction_set, randomization
    """

    if method_name == "2003_DS":
        return "2003", "DS"

    if method_name == "2008_DS":
        return "2008", "DS"

    if method_name == "2003_Owen":
        return "2003", "NUS"

    if method_name == "2008_Owen":
        return "2008", "NUS"

    raise ValueError(
        f"Unknown method: {method_name}"
    )


# ============================================================
# GENERATE EXACT POINTS USED IN ONE EXPERIMENT RUN
# ============================================================

def get_points_for_method(
        N,
        d,
        method_name,
        seed):
    """
    Generate the ACTUAL randomized Sobol points used by one
    experiment method and one seed.

    IMPORTANT:
    This assumes your sobol.py function accepts:
        direction_set = "2003" or "2008"
        randomization = "DS" or "NUS"
        seed = integer

    If your sobol.py uses slightly different argument names,
    only edit this wrapper.
    """

    direction_set, randomization = parse_method(
        method_name
    )

    U = generate_randomized_sobol_points(
        N=N,
        d=d,
        direction_set=direction_set,
        randomization=randomization,
        seed=seed,
    )

    return U


# ============================================================
# GENERATE DETERMINISTIC UNDERLYING POINTS
# ============================================================

def get_deterministic_points(
        N,
        d,
        direction_set):
    """
    Generate the deterministic underlying Sobol construction
    (no randomization).
    """

    U = generate_randomized_sobol_points(
        N=N,
        d=d,
        direction_set=direction_set,
        randomization=False,
        seed=None,
    )

    return U


# ============================================================
# CHECK (t,m,2)-NET CONDITION FROM THE POINTS DIRECTLY
# ============================================================

def satisfies_t_value_2d(
        U_2d,
        m,
        t):
    """
    Check if the first 2^m points in a 2D projection satisfy
    the (t, m, 2)-net condition.

    In 2D, for every split r1 + r2 = m - t,
    every elementary interval of size
        2^{-r1} x 2^{-r2}
    must contain exactly 2^t points.
    """

    N = 2 ** m

    if U_2d.shape != (N, 2):
        raise ValueError(
            f"U_2d must have shape ({N}, 2), got {U_2d.shape}"
        )

    total_resolution = m - t
    expected_count = 2 ** t

    for r1 in range(total_resolution + 1):
        r2 = total_resolution - r1

        n_bins_1 = 2 ** r1
        n_bins_2 = 2 ** r2

        idx1 = np.floor(
            U_2d[:, 0] * n_bins_1
        ).astype(int)

        idx2 = np.floor(
            U_2d[:, 1] * n_bins_2
        ).astype(int)

        idx1 = np.clip(
            idx1,
            0,
            n_bins_1 - 1
        )

        idx2 = np.clip(
            idx2,
            0,
            n_bins_2 - 1
        )

        cell_id = idx1 * n_bins_2 + idx2
        counts = np.bincount(
            cell_id,
            minlength=n_bins_1 * n_bins_2
        )

        if not np.all(counts == expected_count):
            return False

    return True


def calculate_t_value_2d(
        U,
        dim1,
        dim2,
        m):
    """
    Calculate t(dim1, dim2; m) from the first 2^m points.
    Dimensions use mathematical indexing:
        dim1=1 means first coordinate.
    """

    N = 2 ** m

    if U.shape[0] < N:
        raise ValueError(
            f"Need at least {N} points."
        )

    projection = U[:N, [dim1 - 1, dim2 - 1]]

    for t in range(m + 1):
        if satisfies_t_value_2d(
                projection,
                m=m,
                t=t):
            return t

    raise RuntimeError(
        "Unable to determine t-value."
    )


# ============================================================
# SCAN FOR AN INTERESTING PAIR
# ============================================================

def scan_pairs_for_best_difference(
        U_det_2003,
        U_det_2008,
        m,
        max_dim):
    """
    Search pairs among dimensions 1,...,max_dim
    and rank them by (t_2003 - t_2008).
    Larger positive difference means 2008 has a smaller t-value.
    """

    results = []

    total_pairs = max_dim * (max_dim - 1) // 2
    count = 0

    for dim1 in range(1, max_dim):
        for dim2 in range(dim1 + 1, max_dim + 1):

            count += 1

            t_2003 = calculate_t_value_2d(
                U_det_2003,
                dim1,
                dim2,
                m,
            )

            t_2008 = calculate_t_value_2d(
                U_det_2008,
                dim1,
                dim2,
                m,
            )

            results.append({
                "dim1": dim1,
                "dim2": dim2,
                "t_2003": t_2003,
                "t_2008": t_2008,
                "difference_2003_minus_2008":
                    t_2003 - t_2008,
            })

            if count % 100 == 0:
                print(
                    f"Scanned {count}/{total_pairs} pairs..."
                )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "difference_2003_minus_2008",
            "t_2003",
            "t_2008"
        ],
        ascending=[False, False, True]
    )

    results_df.to_csv(
        OUTPUT_DIR / f"pair_scan_d{max_dim}_m{m}.csv",
        index=False
    )

    return results_df

# ============================================================
# PLOT DISTRIBUTION OF t-VALUE DIFFERENCES
# ============================================================

def plot_t_value_difference_distribution(
        pair_scan,
        save=True):
    """
    Plot the distribution of

        delta_t = t_2003 - t_2008

    across all scanned coordinate pairs.

    Interpretation:
        delta_t > 0  -> 2008 has smaller t-value
        delta_t = 0  ->  same t-value
        delta_t < 0  -> 2003 has smaller t-value
    """

    differences = (
        pair_scan[
            "difference_2003_minus_2008"
        ]
    )

    # Integer-valued t differences, so choose bins
    # centred on integers.
    min_diff = int(differences.min())
    max_diff = int(differences.max())

    bins = np.arange(
        min_diff - 0.5,
        max_diff + 1.5,
        1
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        differences,
        bins=bins,
        edgecolor="black"
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    ax.set_xlabel(
        r"$\Delta t = t_{2003} - t_{2008}$"
    )

    ax.set_ylabel(
        "Number of coordinate pairs"
    )

    ax.set_title(
        "Distribution of 2D t-value improvement\n"
        f"across all {SCAN_MAX_DIM} dimensions"
    )

    fig.tight_layout()

    if save:
        output_path = (
            OUTPUT_DIR
            / "t_value_difference_distribution.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"\nSaved distribution plot to:\n"
            f"{output_path}"
        )

    plt.show()

# ============================================================
# PLOT ACTUAL RANDOMIZED POINTS
# ============================================================

def plot_actual_projection_comparison(
        U_left,
        U_right,
        dim1,
        dim2,
        t_left,
        t_right,
        method_left,
        method_right,
        d,
        N,
        seed):
    """
    Plot the ACTUAL randomized Sobol points used by the methods.
    """

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    x1 = U_left[:, dim1 - 1]
    y1 = U_left[:, dim2 - 1]

    x2 = U_right[:, dim1 - 1]
    y2 = U_right[:, dim2 - 1]

    axes[0].scatter(
        x1, y1,
        s=3,
        alpha=0.6
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel(f"Dimension {dim1}")
    axes[0].set_ylabel(f"Dimension {dim2}")
    axes[0].set_title(
        f"{method_left}\n"
        f"d={d}, N={N}, seed={seed}\n"
        f"underlying t({dim1},{dim2};{M}) = {t_left}"
    )

    axes[1].scatter(
        x2, y2,
        s=3,
        alpha=0.6
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(f"Dimension {dim1}")
    axes[1].set_ylabel(f"Dimension {dim2}")
    axes[1].set_title(
        f"{method_right}\n"
        f"d={d}, N={N}, seed={seed}\n"
        f"underlying t({dim1},{dim2};{M}) = {t_right}"
    )

    fig.suptitle(
        "Actual Sobol points used in one experiment run"
    )
    fig.tight_layout()

    out_path = (
        OUTPUT_DIR
        / f"actual_projection_{method_left}_vs_{method_right}"
          f"_d{d}_N{N}_seed{seed}_pair_{dim1}_{dim2}.png"
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nSaved plot to:\n{out_path}")


# ============================================================
# SUMMARISE HOW MANY PAIRS FAVOUR EACH CONSTRUCTION
# ============================================================

def plot_pairwise_comparison_summary(
        pair_scan,
        save=True):
    """
    Count coordinate pairs where:

        2008 has lower t
        same t
        2003 has lower t
    """

    differences = (
        pair_scan[
            "difference_2003_minus_2008"
        ]
    )

    better_2008 = np.sum(
        differences > 0
    )

    same = np.sum(
        differences == 0
    )

    better_2003 = np.sum(
        differences < 0
    )

    total = len(differences)

    print("\n======================================")
    print("PAIRWISE t-VALUE SUMMARY")
    print("======================================")

    print(
        f"2008 lower t: {better_2008} "
        f"({100 * better_2008 / total:.1f}%)"
    )

    print(
        f"Same t-value: {same} "
        f"({100 * same / total:.1f}%)"
    )

    print(
        f"2003 lower t: {better_2003} "
        f"({100 * better_2003 / total:.1f}%)"
    )

    print(
        f"Total pairs: {total}"
    )

    categories = [
        "2008 lower t",
        "Same t",
        "2003 lower t"
    ]

    counts = [
        better_2008,
        same,
        better_2003
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        categories,
        counts
    )

    ax.set_ylabel(
        "Number of coordinate pairs"
    )

    ax.set_title(
        "Which construction has the lower 2D t-value?"
    )

    # Add count labels
    for bar, count in zip(
            bars,
            counts):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    if save:
        output_path = (
            OUTPUT_DIR
            / "pairwise_t_value_summary.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"\nSaved summary plot to:\n"
            f"{output_path}"
        )

    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print("Generating exact randomized points for plotting")
    print("==============================================")
    print(
        f"d = {D}, N = {N}, M = {M}, seed = {SEED}"
    )
    print(
        f"Methods: {METHOD_LEFT} vs {METHOD_RIGHT}"
    )

    # --------------------------------------------------------
    # Actual points used by the two chosen experiment methods
    # --------------------------------------------------------

    U_left_actual = get_points_for_method(
        N=N,
        d=D,
        method_name=METHOD_LEFT,
        seed=SEED,
    )

    U_right_actual = get_points_for_method(
        N=N,
        d=D,
        method_name=METHOD_RIGHT,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Deterministic underlying constructions
    # used to compute t-values
    # --------------------------------------------------------

    U_2003_det = get_deterministic_points(
        N=N,
        d=D,
        direction_set="2003",
    )

    U_2008_det = get_deterministic_points(
        N=N,
        d=D,
        direction_set="2008",
    )

    # --------------------------------------------------------
    # Either use a manual pair or search automatically
    # --------------------------------------------------------

    if USE_MANUAL_PAIR:
        dim1, dim2 = MANUAL_PAIR

    else:
        print("\nScanning for an interesting pair...")
        pair_scan = scan_pairs_for_best_difference(
            U_det_2003=U_2003_det,
            U_det_2008=U_2008_det,
            m=M,
            max_dim=SCAN_MAX_DIM,
        )
                # --------------------------------------------------------
        # Distribution of t-value differences across ALL pairs
        # --------------------------------------------------------

        plot_t_value_difference_distribution(
            pair_scan
        )

        plot_pairwise_comparison_summary(
            pair_scan
        )

        print("\nTop 10 pairs by t-value improvement:")
        print(pair_scan.head(10).to_string(index=False))

        dim1 = int(pair_scan.iloc[0]["dim1"])
        dim2 = int(pair_scan.iloc[0]["dim2"])

        print(
            f"\nChosen pair from scan: ({dim1}, {dim2})"
        )



    # --------------------------------------------------------
    # Compute t-values for the chosen pair
    # --------------------------------------------------------

    t_2003 = calculate_t_value_2d(
        U_2003_det,
        dim1,
        dim2,
        M,
    )

    t_2008 = calculate_t_value_2d(
        U_2008_det,
        dim1,
        dim2,
        M,
    )

    print("\n==============================================")
    print(f"Chosen pair: ({dim1}, {dim2})")
    print(f"t_2003 = {t_2003}")
    print(f"t_2008 = {t_2008}")
    print("Smaller t-value means finer guaranteed balance.")
    print("==============================================")

    summary_df = pd.DataFrame([{
        "d": D,
        "N": N,
        "m": M,
        "seed": SEED,
        "method_left": METHOD_LEFT,
        "method_right": METHOD_RIGHT,
        "dim1": dim1,
        "dim2": dim2,
        "t_2003": t_2003,
        "t_2008": t_2008,
        "difference_2003_minus_2008": t_2003 - t_2008,
    }])

    summary_path = OUTPUT_DIR / "chosen_pair_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary to:\n{summary_path}")

    # --------------------------------------------------------
    # Plot ACTUAL randomized points from the experiment methods
    # --------------------------------------------------------

    plot_actual_projection_comparison(
        U_left=U_left_actual,
        U_right=U_right_actual,
        dim1=dim1,
        dim2=dim2,
        t_left=t_2003 if "2003" in METHOD_LEFT else t_2008,
        t_right=t_2003 if "2003" in METHOD_RIGHT else t_2008,
        method_left=METHOD_LEFT,
        method_right=METHOD_RIGHT,
        d=D,
        N=N,
        seed=SEED,
    )


if __name__ == "__main__":
    main()