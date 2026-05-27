"""
E03 -- Grid refinement.

Solve the same operating point at increasing N, all with forward-Euler
and rho=0. Record chatter amplitude, Joule loss, dense voltage residual,
and wall time.

Predicted outcomes (log-log slope of C vs Delta_theta):
  * linear (slope 1)    => cause A (forward-Euler artefact dominant).
  * quadratic (slope 2) => cause D (aliasing).
  * constant            => cause C (genuine high-harmonic content; the
                          chatter is not a discretization artefact).

Run:
    python experiments/chatter/e03_grid_refinement.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    tail_mass,
)


def main() -> None:
    setup = make_setup()
    n_grids = [32, 64, 128, 256]

    deltas: list[float] = []
    losses: list[float] = []
    chatters: list[float] = []
    tails: list[float] = []
    residuals: list[float] = []
    walls: list[float] = []
    convergeds: list[bool] = []

    for N in n_grids:
        t0 = time.time()
        result = run_active_set(
            model=setup.model,
            transform=setup.transform,
            omega=setup.omega_el,
            torq_target=setup.torq_target,
            curr_max=setup.machine.curr_max,
            volt_max=setup.machine.volt_max,
            n_grid=N,
            max_outer_iter=10,
            tol=1e-3,
            rho=0.0,
            scheme="forward",
        )
        elapsed = time.time() - t0
        X = result["curr_dq_grid"]
        deltas.append(2 * np.pi / N)
        losses.append(joule_loss(X))
        chatters.append(chatter_amplitude(X, h_max=7))
        tails.append(tail_mass(X, h_max=7))
        r_dense, _ = voltage_residuals_dense(
            setup.transform,
            setup.omega_el,
            X,
            setup.machine.volt_max,
        )
        residuals.append(r_dense)
        walls.append(elapsed)
        convergeds.append(bool(result["converged"]))
        print(
            f"N={N:4d}  dtheta={deltas[-1]:.5f}  J={losses[-1]:.4f}  "
            f"C={chatters[-1]:.4f}  tail={tails[-1]:.4f}  "
            f"r_dense={residuals[-1]:+.4f} V  t={elapsed:.1f}s  "
            f"converged={convergeds[-1]}"
        )

    # Fit log-log slope of C vs delta_theta
    log_d = np.log(deltas)
    log_c = np.log(np.maximum(chatters, 1e-9))
    slope, intercept = np.polyfit(log_d, log_c, 1)
    print(f"\nlog-log slope of C vs Delta_theta: {slope:.2f}  (1=>A, 2=>D, 0=>C)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].loglog(deltas, np.maximum(chatters, 1e-9), marker="o")
    axes[0, 0].set_xlabel(r"$\Delta\theta$")
    axes[0, 0].set_ylabel(r"chatter amplitude $C$ (H=7)")
    axes[0, 0].set_title(f"log-log slope = {slope:.2f}")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    axes[0, 1].semilogx(deltas, losses, marker="o")
    axes[0, 1].set_xlabel(r"$\Delta\theta$")
    axes[0, 1].set_ylabel(r"Joule loss $J$")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    axes[1, 0].loglog(deltas, np.maximum(tails, 1e-9), marker="o")
    axes[1, 0].set_xlabel(r"$\Delta\theta$")
    axes[1, 0].set_ylabel(r"tail energy fraction (H=7)")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    axes[1, 1].semilogx(n_grids, walls, marker="o")
    axes[1, 1].set_xlabel(r"$N$")
    axes[1, 1].set_ylabel("wall time [s]")
    axes[1, 1].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"E03: forward-Euler grid refinement (N in {n_grids})")
    fig.tight_layout()
    out = os.path.join(figs_dir(), "e03_grid_refinement.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
