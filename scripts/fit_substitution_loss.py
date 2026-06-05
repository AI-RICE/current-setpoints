"""
Fit the substitution iron-loss model and report its train/test torque RMSE.

The test RMSE is computed on the same holdout split the neural torque model
(NTM) uses (``test_size=0.15``, ``random_state=42``), so it is directly
comparable to the NTM's reported test RMSE. Run from anywhere:

    python scripts/fit_substitution_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from current_setpoints.optimization import (
    ModelAnalytical,
    ModelLossesSubstitution1,
    ModelLossSubstitution2,
)
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.utils import (
    fit_substitution_loss,
    load_aggregated_csv_data,
    split_like_neural,
    substitution2_loss_features,
    substitution_loss_features,
    substitution_loss_rmse,
)

CSV_PATH = REPO_ROOT / "data" / "aggregated_file_means.csv"

# Identity map: the aggregated CSV already uses the internal column names. The
# voltage columns (ud1, uq1, ud3, uq3) are included so NaN-dropping matches the
# columns the loss model consumes.
COLUMN_MAP = {
    "omega": "omega",
    "id1": "id1",
    "iq1": "iq1",
    "id3": "id3",
    "iq3": "iq3",
    "ud1": "ud1",
    "uq1": "uq1",
    "ud3": "ud3",
    "uq3": "uq3",
    "torq": "torq",
}


def _class_test_rmse(model, df_test) -> float:
    """Test RMSE computed by driving the model class row-by-row (sanity tie-in)."""
    sq_err = []
    for _, row in df_test.iterrows():
        omega = float(row["omega"])
        curr_dq = row[["id1", "iq1", "id3", "iq3"]].to_numpy(dtype=float)
        volt_dq = row[["ud1", "uq1", "ud3", "uq3"]].to_numpy(dtype=float)
        t_pred = model.calculate_torque(omega, curr_dq, volt_dq)
        sq_err.append((t_pred - float(row["torq"])) ** 2)
    return (sum(sq_err) / len(sq_err)) ** 0.5


def _report(name, features_fn, model_cls, df_train, df_test, analytical_model, machine, flux) -> None:
    coeffs = fit_substitution_loss(df_train, analytical_model, machine.n_ppairs, features_fn=features_fn)
    test_rmse = substitution_loss_rmse(coeffs, df_test, analytical_model, machine.n_ppairs, features_fn=features_fn)

    model = model_cls(machine, flux, k_v=coeffs["k_v"], k_h=coeffs["k_h"])
    test_rmse_class = _class_test_rmse(model, df_test)

    print(f"\n=== {name} ===")
    print(f"  k_v = {coeffs['k_v']:.6e}")
    print(f"  k_h = {coeffs['k_h']:.6e}")
    print(f"  train R^2    : {coeffs['r2']:.4f}")
    print(f"  train RMSE   : {coeffs['rmse']:.4f} Nm")
    print(f"  TEST  RMSE   : {test_rmse:.4f} Nm")
    print(f"  TEST  RMSE (via class, sanity) : {test_rmse_class:.4f} Nm")


def main() -> None:
    flux = Flux_IEEEMachine2()
    machine = IEEEMachine2()
    # Match the NTM trainer's limits so the analytical baseline is identical.
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    analytical_model = ModelAnalytical(machine=machine, flux=flux)

    df = load_aggregated_csv_data(str(CSV_PATH), COLUMN_MAP)
    df_train, df_test = split_like_neural(df)
    print(f"\ntrain points : {len(df_train)}   test points : {len(df_test)}")

    _report(
        "ModelLossesSubstitution1", substitution_loss_features, ModelLossesSubstitution1,
        df_train, df_test, analytical_model, machine, flux,
    )
    _report(
        "ModelLossSubstitution2", substitution2_loss_features, ModelLossSubstitution2,
        df_train, df_test, analytical_model, machine, flux,
    )


if __name__ == "__main__":
    main()
