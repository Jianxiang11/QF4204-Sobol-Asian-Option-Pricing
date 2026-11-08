# QF4204 Project: Sobol Construction and Randomization in Asian Option Pricing

**“How Much Do Sobol Construction and Randomization Matter? An Empirical Comparison of MC and Randomized QMC in Asian Option Pricing.”**

## Methods

The project compares five estimators for a discretely monitored arithmetic Asian call under Black–Scholes:

- Ordinary Monte Carlo (MC)
- Joe–Kuo 2003 + digital shift
- Joe–Kuo 2003 + Owen-style nested uniform scrambling (QMCPy NUS)
- Joe–Kuo 2008 D6 + digital shift
- Joe–Kuo 2008 D6 + Owen-style nested uniform scrambling (QMCPy NUS)

Main parameters: `S0=100`, `K=100`, `r=0.10`, `sigma=0.20`, `T=1`, `q=0`. Monitoring dimensions are `d=32,128,512`; the final experiment uses 20 independent repetitions per configuration.

## Files

- `pricing.py` — Black–Scholes path simulation, Asian payoffs, analytical European and geometric Asian prices.
- `sobol.py` — Joe–Kuo 2003 reconstruction, Joe–Kuo 2008 D6 through QMCPy `DigitalNetB2`, randomization, and RQMC pricing.
- `benchmark.py` — high-accuracy arithmetic Asian benchmark using a geometric Asian control variate.
- `experiments.py` — repeated experiments, RMSE summaries, improvement factors, and convergence-rate estimation.
- `pilot_experiment.py` — pilot experiments used to choose repetitions and the high-dimensional sample-size cap.
- `main_experiment.py` — final experiment driver.
- `plots.py` — plots used for the main experiment.
- `projection_analysis.py` — supplementary scan of all two-dimensional projections at `d=512, N=8192` and direct calculation of projection-specific t-values.
- `Validation.py` — implementation and reproducibility checks.
- `Direction Numbers/joe-kuo-old.1111` — Joe–Kuo 2003 parameter file used to reconstruct the earlier direction-number set.
- `results/projection/` — saved outputs from the supplementary projection analysis.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Recommended run order

1. Run validation checks:

```bash
python Validation.py
```

2. Run the pilot stage if you want to reproduce the experimental-design decisions:

```bash
python pilot_experiment.py
```

3. Run the final experiment:

```bash
python main_experiment.py
```

This generates the final control-variate benchmark, checkpoint files, experiment results, convergence-rate table, and main plots. The final grid encoded in `main_experiment.py` is:

- `d=32,128`: `N=256,512,1024,2048,4096,8192,16384`
- `d=512`: `N=256,512,1024,2048,4096,8192`
- `R=20`

4. Run the supplementary projection analysis:

```bash
python projection_analysis.py
```

The analysis uses `d=512`, `N=8192`, and seed `2051000`, corresponding to the first repetition of the final `d=512, N=8192` experiment block. It scans all `C(512,2)=130,816` coordinate pairs using the deterministic underlying Sobol constructions and plots the randomized digital-shift points for the selected pair.

## Reproducibility notes

The final benchmark uses `N=2^20` Monte Carlo paths with a geometric Asian control variate. Randomization seeds are assigned deterministically in `experiments.py`. The four Sobol variants are generated through the same QMCPy `DigitalNetB2` framework so that the main construction comparison changes the direction-number set while keeping the surrounding generator framework consistent.

The Owen-style randomization used in the report corresponds to QMCPy's Nested Uniform Scrambling (`NUS`).
