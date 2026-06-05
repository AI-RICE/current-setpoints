"""
Diagnostics for the substitution iron-loss models.

Two analyses, both on out-of-fold predictions from a single 5-fold CV
(shuffle, seed 42) over all operating points:

  1. K-fold CV RMSE (mean +/- std across folds) for each model -- a far more
     stable comparison than the single 27-point holdout.
  2. Per-speed-cluster RMSE from the out-of-fold residuals -- tests whether
     Model 2's U/f proxy degrades at low speed (where the resistive drop is a
     larger fraction of U and the 1/omega^2 weighting amplifies it).

The analytical baseline (no correction) is included as a reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from current_setpoints.optimization import ModelAnalytical
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.utils import (
    fit_substitution_loss,
    load_aggregated_csv_data,
    substitution2_loss_features,
    substitution_loss_features,
)

CSV_PATH = REPO_ROOT / "data" / "aggregated_file_means.csv"
COLUMN_MAP = {
    "omega": "omega", "id1": "id1", "iq1": "iq1", "id3": "id3", "iq3": "iq3",
    "ud1": "ud1", "uq1": "uq1", "ud3": "ud3", "uq3": "uq3", "torq": "torq",
}
CURR_COLS = ["id1", "iq1", "id3", "iq3"]
VOLT_COLS = ["ud1", "uq1", "ud3", "uq3"]
N_SPLITS = 5
SEED = 42
CLUSTER_STEP = 335.0  # rad/s; data falls in ~335/670/1005/1340 clusters


def baseline_torque(df, analytical_model):
    omega = df["omega"].to_numpy(dtype=float)
    curr = df[CURR_COLS].to_numpy(dtype=float)
    return np.array([analytical_model.calculate_torque(float(w), c) for w, c in zip(omega, curr)])


def predict_total(df, coeffs, features_fn, analytical_model, p):
    omega = df["omega"].to_numpy(dtype=float)
    volt = df[VOLT_COLS].to_numpy(dtype=float)
    t_base = baseline_torque(df, analytical_model)
    f_v, f_h = features_fn(omega, volt, p)
    return t_base + coeffs["k_v"] * f_v + coeffs["k_h"] * f_h


def rmse(pred, meas):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(meas)) ** 2)))


def main() -> None:
    flux = Flux_IEEEMachine2()
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    analytical_model = ModelAnalytical(machine=machine, flux=flux)
    p = machine.n_ppairs

    df = load_aggregated_csv_data(str(CSV_PATH), COLUMN_MAP).reset_index(drop=True)
    meas = df["torq"].to_numpy(dtype=float)

    models = [
        ("Analytical", None),
        ("Substitution1 (B=U)", substitution_loss_features),
        ("Substitution2 (B=U/f)", substitution2_loss_features),
    ]

    # Out-of-fold predicted total torque per point, and per-fold RMSE.
    oof_pred = {name: np.full(len(df), np.nan) for name, _ in models}
    fold_rmse = {name: [] for name, _ in models}

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in kf.split(df):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        meas_te = meas[te_idx]
        for name, feat in models:
            if feat is None:
                pred_te = baseline_torque(df_te, analytical_model)
            else:
                coeffs = fit_substitution_loss(df_tr, analytical_model, p, features_fn=feat)
                pred_te = predict_total(df_te, coeffs, feat, analytical_model, p)
            oof_pred[name][te_idx] = pred_te
            fold_rmse[name].append(rmse(pred_te, meas_te))

    print(f"\n=== 5-fold CV RMSE (all {len(df)} points, shuffle seed {SEED}) ===")
    print(f"  {'model':<24}{'mean RMSE':>11}{'std':>9}")
    for name, _ in models:
        r = np.array(fold_rmse[name])
        print(f"  {name:<24}{r.mean():>10.4f} {r.std():>8.4f}")

    # Per-speed-cluster RMSE from out-of-fold residuals.
    w = df["omega"].to_numpy(dtype=float)
    cluster_id = np.round(w / CLUSTER_STEP).astype(int)
    print("\n=== Per-speed-cluster out-of-fold RMSE [Nm] ===")
    header = f"  {'speed [rad/s]':<16}{'n':>4}" + "".join(f"{name.split()[0]:>16}" for name, _ in models)
    print(header)
    for cid in sorted(set(cluster_id)):
        mask = cluster_id == cid
        speed = w[mask].mean()
        row = f"  {speed:<16.0f}{int(mask.sum()):>4}"
        for name, _ in models:
            row += f"{rmse(oof_pred[name][mask], meas[mask]):>16.4f}"
        print(row)


if __name__ == "__main__":
    main()
