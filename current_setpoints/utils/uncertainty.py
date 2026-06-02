"""
Uncertainty / sensitivity utilities for the neural torque model (NTM).

Two complementary analyses, kept deliberately lightweight for reporting:

- ``measurement_noise_floor`` : the aleatoric (data) uncertainty. Reads the raw
  per-operating-point measurement files, takes the torque time-series in each,
  and reports the within-point scatter (std) and the precision of the averaged
  target (std / sqrt(N)). This is the floor below which a model's test RMSE is
  not physically meaningful.

- ``input_noise_sensitivity`` : robustness to sensor noise. Injects Gaussian
  noise into the test inputs and pushes them through an ALREADY-TRAINED network
  (no retraining), reporting how the RMSE degrades.

A small helper, ``build_residual_test_set``, reconstructs the residual target
(measured - analytical) and the train/test split the NTM was trained against,
so a notebook can produce the exact test set without duplicating that logic.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def measurement_noise_floor(
    mat_dir: str,
    struct_name: str = "meas",
    torq_field: str = "torq",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Computes the measurement-noise floor of the torque data.

    Loops over every ``*.mat`` file in ``mat_dir`` (one per operating point),
    extracts the measured torque time-series, and characterises its scatter.

    Args:
        mat_dir: Directory holding the per-operating-point ``.mat`` files.
        struct_name: Name of the struct inside each file (default ``"meas"``).
        torq_field: Field of the struct holding the torque time-series
            (default ``"torq"``).
        verbose: If True, prints a formatted summary.

    Returns:
        Dict with raw arrays (``stds``, ``sems``, ``n_samples``) and summary
        statistics:

        - ``std_median`` / ``std_mean`` / ``std_iqr`` / ``std_range`` :
          within-point standard deviation (instantaneous scatter: cycle-to-cycle
          torque ripple + transducer noise), in Nm.
        - ``sem_median`` / ``sem_iqr`` : standard error of the mean
          (``std / sqrt(N)``), the precision of the averaged target. This is an
          i.i.d. lower bound; the true value is larger because ripple is
          autocorrelated.
        - ``n_files`` / ``n_processed`` / ``n_skipped``.
    """
    files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))

    stds: list[float] = []
    sems: list[float] = []
    n_samples: list[int] = []
    skipped = 0
    for f in files:
        try:
            m = loadmat(f)
            torq_ts = np.asarray(m[struct_name][torq_field][0, 0]).ravel().astype(float)
            torq_ts = torq_ts[np.isfinite(torq_ts)]
            if torq_ts.size < 2:
                skipped += 1
                continue
            s = float(np.std(torq_ts, ddof=1))
            stds.append(s)
            sems.append(s / np.sqrt(torq_ts.size))
            n_samples.append(int(torq_ts.size))
        except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
            print(f"  skipped {os.path.basename(f)}: {e}")
            skipped += 1

    stds_a = np.asarray(stds)
    sems_a = np.asarray(sems)
    n_a = np.asarray(n_samples)

    if stds_a.size == 0:
        raise RuntimeError(f"No usable torque time-series found in {mat_dir}")

    result: dict[str, Any] = {
        "stds": stds_a,
        "sems": sems_a,
        "n_samples": n_a,
        "n_files": len(files),
        "n_processed": int(stds_a.size),
        "n_skipped": skipped,
        "std_median": float(np.median(stds_a)),
        "std_mean": float(np.mean(stds_a)),
        "std_iqr": (float(np.percentile(stds_a, 25)), float(np.percentile(stds_a, 75))),
        "std_range": (float(np.min(stds_a)), float(np.max(stds_a))),
        "sem_median": float(np.median(sems_a)),
        "sem_iqr": (float(np.percentile(sems_a, 25)), float(np.percentile(sems_a, 75))),
    }

    if verbose:
        print(f"Found {result['n_files']} measurement files in {mat_dir}")
        print(f"Processed {result['n_processed']} points  (skipped {result['n_skipped']})")
        print(
            f"Samples per point: median={int(np.median(n_a))}, "
            f"min={int(np.min(n_a))}, max={int(np.max(n_a))}"
        )
        print("\n--- Instantaneous measurement scatter (within-point torque std, Nm) ---")
        print(f"  median : {result['std_median']:.4f}")
        print(f"  mean   : {result['std_mean']:.4f}")
        print(f"  IQR    : [{result['std_iqr'][0]:.4f}, {result['std_iqr'][1]:.4f}]")
        print(f"  range  : [{result['std_range'][0]:.4f}, {result['std_range'][1]:.4f}]")
        print("\n--- Precision of the averaged per-point target (SEM = std/sqrt(N), Nm) ---")
        print("    (i.i.d. lower bound; true value is larger because ripple is autocorrelated)")
        print(f"  median : {result['sem_median']:.4f}")
        print(f"  IQR    : [{result['sem_iqr'][0]:.4f}, {result['sem_iqr'][1]:.4f}]")

    return result


def build_residual_test_set(
    X: np.ndarray,
    y_measured: np.ndarray,
    analytical_model: Any,
    test_size: float = 0.15,
    random_state: int = 42,
    return_train: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Reconstructs the residual-target split the NTM was trained against.

    The NTM is trained on the residual ``y = measured - analytical``, so the
    targets must be built the same way to reproduce the reported RMSE.

    Args:
        X: Input features, columns ``[omega, i_d1, i_q1, i_d3, i_q3, ...]``.
        y_measured: Measured torque, shape ``(N, 1)``.
        analytical_model: Any object exposing
            ``calculate_torque(omega: float, curr_dq: np.ndarray) -> float``
            (e.g. ``ModelAnalytical``). Passed in (not imported) to avoid a
            utils -> optimization import cycle.
        test_size: Fraction held out for testing (must match training).
        random_state: Split seed (must match training).
        return_train: If True, also return the train-val portion (the 1 -
            ``test_size`` fraction the NTM was fit on) for fitting baselines
            on the same data.

    Returns:
        ``(X_test, y_test)`` by default, or
        ``(X_train, X_test, y_train, y_test)`` if ``return_train`` is True.
        ``y_*`` are the residual targets.
    """
    X = np.asarray(X, dtype=np.float64)
    y_measured = np.asarray(y_measured, dtype=np.float64).reshape(-1, 1)

    y_analytical = np.array(
        [analytical_model.calculate_torque(float(row[0]), row[1:]) for row in X]
    ).reshape(-1, 1)
    y = y_measured - y_analytical

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if return_train:
        return X_train, X_test, y_train, y_test
    return X_test, y_test


def input_noise_sensitivity(
    net: torch.nn.Module,
    scaler: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: tuple[float, ...] = (0.0, 0.02, 0.05),
    n_trials: int = 200,
    device: torch.device | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Robustness of a trained network to input (sensor) noise.

    For each noise level ``p``, additive zero-mean Gaussian noise with standard
    deviation ``p * (per-channel std of X_test)`` is added to the test inputs;
    the noisy inputs are scaled and pushed through ``net`` (no retraining) and
    the residual-torque RMSE is recorded, averaged over ``n_trials`` draws.

    Args:
        net: Trained network (e.g. ``NeuralTorquePredictor``) in eval mode.
        scaler: Fitted input scaler exposing ``transform``.
        X_test: Test inputs (raw, unscaled), shape ``(N, n_features)``.
        y_test: Test targets (the residual the net predicts), shape ``(N, 1)``.
        noise_levels: Fractional noise levels; ``0.0`` gives the clean baseline.
        n_trials: Noise realisations to average per non-zero level.
        device: Torch device; defaults to ``net.device`` or CPU.
        seed: Seed for the noise generator (reproducibility).
        verbose: If True, prints a formatted table.

    Returns:
        Dict with ``baseline`` (clean RMSE) and ``rmse_by_level``
        (``{level: mean_rmse}``).
    """
    if device is None:
        device = getattr(net, "device", torch.device("cpu"))
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1, 1)

    def predict(X_raw: np.ndarray) -> np.ndarray:
        Xn = scaler.transform(X_raw)
        with torch.no_grad():
            return net(torch.from_numpy(Xn).float().to(device)).cpu().numpy()

    def rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - y_test) ** 2)))

    col_std = X_test.std(axis=0)
    rng = np.random.default_rng(seed)

    baseline = rmse(predict(X_test))
    rmse_by_level: dict[float, float] = {}
    for p in noise_levels:
        if p == 0.0:
            rmse_by_level[p] = baseline
        else:
            draws = [
                rmse(predict(X_test + rng.normal(0.0, 1.0, X_test.shape) * (p * col_std)))
                for _ in range(n_trials)
            ]
            rmse_by_level[p] = float(np.mean(draws))

    if verbose:
        print(f"Test set: {X_test.shape[0]} points  |  {n_trials} noise realisations per level\n")
        print(f"{'noise':>8} | {'RMSE [Nm]':>11} | {'vs clean':>9}")
        print("-" * 34)
        for p, r in rmse_by_level.items():
            print(f"{p * 100:6.0f}% | {r:11.4f} | {r / baseline * 100:7.1f}%")

    return {"baseline": baseline, "rmse_by_level": rmse_by_level}