"""
E04 -- Forward-Euler vs centred-difference discretisation.

Same operating point and N, two schemes:
  * 'forward':  v(theta_n) = U x_n + (omega/dtheta) L (x_{n+1} - x_n) + bemf
  * 'central':  v(theta_n) = U x_n + (omega/(2 dtheta)) L (x_{n+1} - x_{n-1}) + bemf

Save before/after (R0 -> active-set) figures for both schemes and a
comparison table of Joule loss, chatter amplitude, and dense voltage
residual.

Predicted outcomes:
  * Chatter vanishes under 'central' at the same N => cause A confirmed.
  * Chatter persists under 'central'             => A excluded; B/C dominant.

Run:
    python experiments/chatter/e04_central_difference.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")

from current_setpoints.utils.plotting_dynamic import plot_dq_phase_combined
from dynamic.active_set_optimizer import run_active_set, voltage_residuals_dense
from experiments.chatter.common import (
    chatter_amplitude,
    figs_dir,
    joule_loss,
    make_setup,
    tail_mass,
)


def main() -> None:
    setup = make_setup()
    n_grid = 128

    rows = []
    for scheme in ("forward", "central"):
        result = run_active_set(
            model=setup.model,
            transform=setup.transform,
            omega=setup.omega_el,
            torq_target=setup.torq_target,
            curr_max=setup.machine.curr_max,
            volt_max=setup.machine.volt_max,
            n_grid=n_grid,
            max_outer_iter=10,
            tol=1e-3,
            rho=0.0,
            scheme=scheme,
        )
        X = result["curr_dq_grid"]
        r_dense, _ = voltage_residuals_dense(
            setup.transform,
            setup.omega_el,
            X,
            setup.machine.volt_max,
        )
        rows.append(
            {
                "scheme": scheme,
                "converged": bool(result["converged"]),
                "active_size": len(result["active"]),
                "J": joule_loss(X),
                "C": chatter_amplitude(X, h_max=7),
                "tail": tail_mass(X, h_max=7),
                "r_dense": r_dense,
            }
        )
        print(
            f"scheme={scheme:8s}  J={rows[-1]['J']:.4f}  C={rows[-1]['C']:.4f}  "
            f"tail={rows[-1]['tail']:.4f}  r_dense={r_dense:+.4f} V  "
            f"|A|={rows[-1]['active_size']:3d}  converged={rows[-1]['converged']}"
        )

        fig, _ = plot_dq_phase_combined(
            setup.transform,
            setup.omega_el,
            X,
            curr_max=setup.machine.curr_max,
            volt_max=setup.machine.volt_max,
            title=f"E04 scheme={scheme} (N={n_grid})  J={rows[-1]['J']:.4f}, C={rows[-1]['C']:.4f}",
        )
        out = os.path.join(figs_dir(), f"e04_{scheme}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    print("\nSummary:")
    print(f"  forward J={rows[0]['J']:.4f}  central J={rows[1]['J']:.4f}  DeltaJ={rows[1]['J'] - rows[0]['J']:+.4f}")
    print(
        f"  forward C={rows[0]['C']:.4f}  central C={rows[1]['C']:.4f}  "
        f"ratio central/forward = {rows[1]['C'] / max(rows[0]['C'], 1e-9):.3f}"
    )


if __name__ == "__main__":
    main()
