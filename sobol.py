#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sobol sequence construction and randomized-QMC pricing utilities."""

from pathlib import Path
from time import perf_counter

import numpy as np
from qmcpy import DigitalNetB2
from scipy.special import ndtri

from pricing import asian_call_payoffs_from_normals

BASE_DIR = Path(__file__).resolve().parent
JOE_KUO_2003_FILE = BASE_DIR / "Direction Numbers" / "joe-kuo-old.1111"


def load_joe_kuo_2003_parameters(file_path, d):
    """Load Joe-Kuo 2003 primitive-polynomial parameters up to dimension d."""
    file_path = Path(file_path)

    if not 1 <= d <= 1111:
        raise ValueError("Joe-Kuo 2003 supports dimensions 1 to 1111.")
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find Joe-Kuo 2003 file: {file_path}")

    parameters = {}
    with file_path.open("r") as file:
        next(file)  # header
        for line in file:
            parts = line.split()
            if not parts:
                continue

            dimension = int(parts[0])
            degree = int(parts[1])
            coefficient = int(parts[2])
            initial_m = [int(value) for value in parts[3:]]

            if len(initial_m) != degree:
                raise ValueError(
                    f"Dimension {dimension}: expected {degree} initial m-values "
                    f"but found {len(initial_m)}."
                )

            if dimension <= d:
                parameters[dimension] = (degree, coefficient, initial_m)
            if dimension >= d:
                break

    missing = [j for j in range(2, d + 1) if j not in parameters]
    if missing:
        raise ValueError(f"Missing direction-number information for dimensions: {missing}")

    return parameters


def build_joe_kuo_2003_matrices(file_path, d, t_max=63):
    """Construct Joe-Kuo 2003 Sobol generating matrices in uint64 form."""
    if not 1 <= d <= 1111:
        raise ValueError("Joe-Kuo 2003 supports dimensions 1 to 1111.")
    if not 1 <= t_max <= 63:
        raise ValueError("t_max must be between 1 and 63.")

    parameters = load_joe_kuo_2003_parameters(file_path, d)
    matrices = np.zeros((d, t_max), dtype=np.uint64)

    # Dimension 1: v_k = 1 / 2^k
    for k in range(1, t_max + 1):
        matrices[0, k - 1] = np.uint64(1) << np.uint64(t_max - k)

    # Dimensions 2, ..., d
    for dimension in range(2, d + 1):
        degree, coefficient, initial_m = parameters[dimension]
        row = dimension - 1

        for k in range(1, degree + 1):
            matrices[row, k - 1] = (
                np.uint64(initial_m[k - 1]) << np.uint64(t_max - k)
            )

        for k in range(degree + 1, t_max + 1):
            value = (
                matrices[row, k - degree - 1]
                ^ (matrices[row, k - degree - 1] >> np.uint64(degree))
            )

            for ell in range(1, degree):
                coefficient_bit = (coefficient >> (degree - 1 - ell)) & 1
                if coefficient_bit == 1:
                    value ^= matrices[row, k - ell - 1]

            matrices[row, k - 1] = value

    return matrices


# Precompute once on import. This is small (~0.6 MB) and avoids rebuilding every call.
JOE_KUO_2003_MATRICES = build_joe_kuo_2003_matrices(
    JOE_KUO_2003_FILE,
    d=1111,
    t_max=63
)


def _validate_power_of_two(N):
    if not isinstance(N, (int, np.integer)):
        raise TypeError("N must be an integer.")
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a positive power of two.")


def generate_randomized_sobol_points(
        N, d, direction_set="2008", randomization="DS", seed=None):
    """Generate deterministic or randomized Sobol points."""
    _validate_power_of_two(N)

    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError("d must be a positive integer.")
    if randomization not in ("DS", "NUS", False):
        raise ValueError("randomization must be 'DS', 'NUS', or False.")

    if direction_set == "2003":
        if d > 1111:
            raise ValueError("Joe-Kuo 2003 supports dimensions up to 1111.")

        sobol = DigitalNetB2(
            dimension=d,
            generating_matrices=JOE_KUO_2003_MATRICES[:d, :],
            randomize=randomization,
            seed=seed,
            order="GRAY",
            t=63,
            msb=True
        )

    elif direction_set == "2008":
        if d > 21201:
            raise ValueError("Joe-Kuo 2008 D6 file supports dimensions up to 21201.")

        sobol = DigitalNetB2(
            dimension=d,
            generating_matrices="joe_kuo.6.21201.txt",
            randomize=randomization,
            seed=seed,
            order="GRAY",
            t=63
        )
    else:
        raise ValueError("direction_set must be '2003' or '2008'.")

    return np.asarray(sobol.gen_samples(n=N), dtype=float)


def sobol_points_to_normals(U):
    """Transform Sobol U(0,1) points to standard-normal inputs."""
    U = np.asarray(U, dtype=float)
    epsilon = np.finfo(float).eps
    return ndtri(np.clip(U, epsilon, 1.0 - epsilon))


def price_asian_call_rqmc(
        N, d, S0, K, r, sigma, T,
        average_type="arithmetic",
        direction_set="2008",
        randomization="DS",
        q=0.0,
        seed=None):
    """Price an Asian call with deterministic or randomized Sobol QMC."""
    start = perf_counter()

    U = generate_randomized_sobol_points(
        N=N,
        d=d,
        direction_set=direction_set,
        randomization=randomization,
        seed=seed
    )
    Z = sobol_points_to_normals(U)
    payoffs = asian_call_payoffs_from_normals(
        Z=Z,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        average_type=average_type,
        q=q
    )

    return float(np.mean(payoffs)), perf_counter() - start


if __name__ == "__main__":
    expected = np.array([
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.500],
        [0.750, 0.250, 0.750],
        [0.250, 0.750, 0.250],
        [0.375, 0.375, 0.625],
        [0.875, 0.875, 0.125],
        [0.625, 0.125, 0.375],
        [0.125, 0.625, 0.875]
    ])
    generated = generate_randomized_sobol_points(
        N=8, d=3, direction_set="2003", randomization=False
    )
    assert np.allclose(generated, expected)
    print("Sobol module check passed.")
