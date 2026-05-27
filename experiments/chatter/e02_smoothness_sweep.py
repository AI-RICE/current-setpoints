"""
E02 -- Slew-rate (smoothness) regulariser sweep.

Augment the objective with ``rho * sum_n |x_{n+1} - x_n|^2`` and sweep
rho over several decades. At each rho, re-run the active-set algorithm
from R0 and record:
  - Joule loss J
  - chatter amplitude C (H=7)
  - dense voltage residual

Predicted outcomes:
  * J flat in rho while C drops      => cause B (degenerate optimum).
  * J rises sharply as rho grows     => cause C (real high-harmonic content).
  * C drops smoothly with no surprises => either way, rho > 0 cleans up
    the trajectory if needed.

Saves a single figure with three subpanels and prints a per-rho table.

Run:
    python experiments/chatter/e02_smoothness_sweep.py
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

from dynamic.active_set_optimizer import run_active_set, voltage_residuals_dense
from experiments.chatter.common import (
    chatter_amplitude,
    figs_dir,
    joule_loss,
    make_setup,
)


def main() -> None:
    setup = make_setup()
    n_grid = 128
    rhos = [0.0, 1e-6, 1e-4, 1e-2, 1.0]

    losses: list[float] = []
    chatters: list[float] = []
    residuals: list[float] = []
    convergeds: list[bool] = []

    for rho in rhos:
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
            rho=float(rho),
            scheme="forward",
        )
        X = result["curr_dq_grid"]
        J = joule_loss(X)
        C = chatter_amplitude(X, h_max=7)
        r_dense, _ = voltage_residuals_dense(
            setup.transform,
            setup.omega_el,
            X,
            setup.machine.volt_max,
        )
        losses.append(J)
        chatters.append(C)
        residuals.append(r_dense)
        convergeds.append(bool(result["converged"]))
        print(
            f"rho={rho:.1e}  J={J:.4f}  C={C:.4f}  r_dense={r_dense:+.4f} V  "
            f"|A|={len(result['active']):3d}  converged={result['converged']}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    rho_plot = [max(r, 1e-9) for r in rhos]  # log-safe

    axes[0].semilogx(rho_plot, losses, marker="o")
    axes[0].set_ylabel(r"Joule loss  $J$")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].loglog(rho_plot, [max(c, 1e-6) for c in chatters], marker="o")
    axes[1].set_ylabel(r"chatter amplitude  $C$  (H=7)")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(rho_plot, residuals, marker="o")
    axes[2].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[2].set_ylabel("dense $V$ residual [V]")
    axes[2].set_xlabel(r"slew-rate weight  $\rho$")
    axes[2].grid(True, which="both", alpha=0.3)

    axes[0].set_title(f"E02: smoothness sweep at N={n_grid}\nflat J + dropping C => cause B;  rising J => cause C")
    fig.tight_layout()
    out = os.path.join(figs_dir(), "e02_smoothness_sweep.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
