"""
E05 -- Restart sensitivity.

Run the active-set joint solver from K randomised warm-starts at fixed N.
Compare Joule loss and per-restart chatter pattern. If all restarts yield
the same loss but visibly different chatter patterns, the optimum is
degenerate (cause B/E).

Warm-starts are obtained by perturbing the R0 trajectory with sinusoidal
noise of amplitude eps in dq space.

Run:
    python experiments/chatter/e05_restart_sensitivity.py
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
from experiments.chatter.common import (
    chatter_amplitude,
    figs_dir,
    joule_loss,
    make_setup,
)


def _perturb(X0: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Adds smooth random perturbation: low-frequency sinusoids in dq."""
    N, dim = X0.shape
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    perturb = np.zeros_like(X0)
    for h in (1, 2, 3, 5, 7):
        for j in range(dim):
            a = rng.normal()
            b = rng.normal()
            perturb[:, j] += eps * (a * np.cos(h * theta) + b * np.sin(h * theta)) / (h + 1)
    return X0 + perturb


def main() -> None:
    setup = make_setup()
    n_grid = 128
    n_restarts = 5
    eps = 0.5
    seed = 12345

    rng = np.random.default_rng(seed)
    optimizer = ActiveSetOptimizer(setup.fwd, n_grid=n_grid, max_outer_iter=10, tol=1e-3)
    sol = optimizer.minimize_current(setup.torq_target, setup.omega_el)
    A = set(int(n) for n in sol.diagnostics.get("active", frozenset()))
    X_ref = sol.curr_dq
    J_ref = joule_loss(X_ref)
    print(f"reference run: J={J_ref:.6f}  |A|={len(A)}  converged={sol.success}")
    print()

    losses: list[float] = []
    chatters: list[float] = []
    trajectories: list[np.ndarray] = []

    maps = optimizer._precompute_maps(setup.omega_el)
    for k in range(n_restarts):
        X_init = _perturb(X_ref, eps, rng)
        X, ok = optimizer._solve_joint(maps, setup.omega_el, setup.torq_target, X_init, A)
        J = joule_loss(X)
        C = chatter_amplitude(X, h_max=7)
        losses.append(J)
        chatters.append(C)
        trajectories.append(X)
        print(f"restart {k}  J={J:.6f}  dJ_vs_ref={J - J_ref:+.4e}  C={C:.4f}  ok={ok}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    theta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    cmap = plt.get_cmap("tab10")
    for k, X in enumerate(trajectories):
        axes[0].plot(
            theta_grid,
            X[:, 0],
            color=cmap(k),
            lw=1.0,
            alpha=0.85,
            label=f"restart {k}  J={losses[k]:.4f}  C={chatters[k]:.3f}",
        )
        axes[1].plot(theta_grid, X[:, 1], color=cmap(k), lw=1.0, alpha=0.85)
    axes[0].plot(theta_grid, X_ref[:, 0], color="k", lw=1.5, ls="--", label="reference")
    axes[1].plot(theta_grid, X_ref[:, 1], color="k", lw=1.5, ls="--")
    axes[0].set_ylabel(r"$i_{d^1}(\theta)$ [A]")
    axes[1].set_ylabel(r"$i_{q^1}(\theta)$ [A]")
    axes[1].set_xlabel(r"$\theta$ [rad]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title(f"E05 restart sensitivity (N={n_grid}, eps={eps})  J spread = {max(losses) - min(losses):.3e}")
    fig.tight_layout()
    out = os.path.join(figs_dir(), "e05_restart_sensitivity.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")
    print(f"J spread across restarts: {max(losses) - min(losses):.4e}")
    print(f"C spread across restarts: {max(chatters) - min(chatters):.4e}")


if __name__ == "__main__":
    main()
