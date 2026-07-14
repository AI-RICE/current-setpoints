"""
E01 -- Fourier spectrum of the converged active-set trajectory.

Question: does the trajectory have substantial Fourier energy above
harmonic 7?

Predicted outcomes:
  * Rapid decay above h ~ 7  => chatter is supra-physical (B or E).
  * Slow / no decay          => chatter reflects genuine high-harmonic
                                content (C).

Run:
    python experiments/chatter/e01_fourier_spectrum.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")

from current_setpoints.optimization import ActiveSetOptimizer
from dynamic.voltage_diagnostics import voltage_residuals_dense
from experiments.chatter.common import (
    chatter_amplitude,
    figs_dir,
    fourier_spectrum,
    joule_loss,
    make_setup,
    tail_mass,
)


def main() -> None:
    setup = make_setup()
    n_grid = 128

    optimizer = ActiveSetOptimizer(setup.fwd, n_grid=n_grid, max_outer_iter=8, tol=1e-3)
    sol = optimizer.minimize_current(setup.torq_target, setup.omega_el)
    X = sol.curr_dq
    print(f"converged: {sol.success}, |A| = {len(sol.diagnostics.get('active', frozenset()))}")
    print(f"joule_loss = {joule_loss(X):.6f}")
    print(f"chatter_amplitude (H=7) = {chatter_amplitude(X, h_max=7):.4f}")
    print(f"tail_mass (H=7)         = {tail_mass(X, h_max=7):.4f}")
    r_dense, _ = voltage_residuals_dense(
        setup.fwd,
        setup.omega_el,
        X,
        setup.machine.volt_max,
    )
    print(f"dense voltage residual = {r_dense:+.4f} V")

    h, mag = fourier_spectrum(X)
    labels = [
        rf"$i_{{d^{2 * i + 1}}}$" if j == 0 else rf"$i_{{q^{2 * i + 1}}}$"
        for i in range(setup.machine.dim // 2)
        for j in range(2)
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(setup.machine.dim):
        ax.semilogy(h, np.maximum(mag[:, j], 1e-10), marker="o", ms=4, label=labels[j], linewidth=1.2)
    ax.set_xlabel(r"harmonic number $h$")
    ax.set_ylabel(r"$|\hat{a}_h|$  [A]")
    ax.set_xlim(0, n_grid // 2)
    ax.axvspan(0, 7, alpha=0.08, color="green", label="low-harmonic band")
    ax.axvspan(7, n_grid // 2, alpha=0.08, color="red", label="chatter band")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncols=3, fontsize=10, loc="lower left")
    ax.set_title(
        f"E01: Fourier spectrum of active-set trajectory  (N={n_grid})\n"
        f"chatter amp(H=7)={chatter_amplitude(X, 7):.3f},  "
        f"tail mass(H=7)={tail_mass(X, 7):.3f}"
    )
    out = os.path.join(figs_dir(), "e01_fourier_spectrum.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
