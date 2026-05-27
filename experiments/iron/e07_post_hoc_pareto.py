"""
E07 -- Phase-1 post-hoc Pareto evaluation.

Reuse the active-set converged trajectories produced by the chatter
sweeps (E02 over rho, E03 over N) and evaluate their iron-loss
components using ``dynamic.iron_loss``. No new optimization. The goal
is to see, at the reference operating point only, whether the
high-harmonic content the optimizer picks up in E03 pays for itself
once iron is accounted for.

Outputs:
  experiments/iron/figs/e07_pareto_rho.png    (Pareto from E02 rho sweep)
  experiments/iron/figs/e07_pareto_N.png      (Pareto from E03 N sweep)

Run:
    python experiments/iron/e07_post_hoc_pareto.py
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

from dynamic.active_set_optimizer import run_active_set
from dynamic.iron_loss import iron_loss
from experiments.chatter.common import (
    chatter_amplitude,
    joule_loss,
    make_setup,
)


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    setup = make_setup()
    out_dir = _figs_dir()

    # --- rho sweep at fixed N=128 -----------------------------------
    rhos = [0.0, 1e-6, 1e-4, 1e-2, 1.0]
    n_grid = 128
    print(f"rho sweep at N={n_grid}:")
    Js: list[float] = []
    Pe: list[float] = []
    Ph: list[float] = []
    Cs: list[float] = []
    for rho in rhos:
        r = run_active_set(
            model=setup.model,
            transform=setup.transform,
            omega=setup.omega_el,
            torq_target=setup.torq_target,
            curr_max=setup.machine.curr_max,
            volt_max=setup.machine.volt_max,
            n_grid=n_grid,
            max_outer_iter=10,
            tol=1e-3,
            rho=rho,
            scheme="forward",
        )
        X = r["curr_dq_grid"]
        J = joule_loss(X)
        loss = iron_loss(setup.transform, setup.omega_el, X)
        C = chatter_amplitude(X, h_max=7)
        Js.append(J)
        Pe.append(loss["eddy"])
        Ph.append(loss["hysteresis"])
        Cs.append(C)
        print(f"  rho={rho:.1e}  J={J:.3f}  P_eddy={loss['eddy']:.3f}  P_hyst={loss['hysteresis']:.3f}  C={C:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(Js, Pe, c=np.log10(np.maximum(rhos, 1e-12)), cmap="viridis", s=80)
    for J, P, rho in zip(Js, Pe, rhos, strict=True):
        axes[0].annotate(f"rho={rho:.0e}", (J, P), fontsize=9, xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel(r"Joule loss $J$")
    axes[0].set_ylabel(r"Eddy iron loss $P_{eddy}$")
    axes[0].set_title(r"E07a: Pareto from $\rho$ sweep at $N=128$")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label=r"$\log_{10}\rho$")

    axes[1].scatter(Js, Ph, c=np.log10(np.maximum(rhos, 1e-12)), cmap="viridis", s=80)
    axes[1].set_xlabel(r"Joule loss $J$")
    axes[1].set_ylabel(r"Hysteresis proxy $P_{hyst}$")
    axes[1].set_title(r"E07a: hysteresis from $\rho$ sweep")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e07_pareto_rho.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.join(out_dir, 'e07_pareto_rho.png')}\n")

    # --- N sweep at fixed rho=0 ------------------------------------
    Ns = [32, 64, 128, 256]
    print("N sweep at rho=0:")
    JsN: list[float] = []
    PeN: list[float] = []
    PhN: list[float] = []
    CsN: list[float] = []
    for N in Ns:
        r = run_active_set(
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
        X = r["curr_dq_grid"]
        J = joule_loss(X)
        loss = iron_loss(setup.transform, setup.omega_el, X)
        C = chatter_amplitude(X, h_max=7)
        JsN.append(J)
        PeN.append(loss["eddy"])
        PhN.append(loss["hysteresis"])
        CsN.append(C)
        print(f"  N={N:4d}  J={J:.3f}  P_eddy={loss['eddy']:.3f}  P_hyst={loss['hysteresis']:.3f}  C={C:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(JsN, PeN, "o-", markersize=8)
    for J, P, N in zip(JsN, PeN, Ns, strict=True):
        axes[0].annotate(f"N={N}", (J, P), fontsize=9, xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel(r"Joule loss $J$")
    axes[0].set_ylabel(r"Eddy iron loss $P_{eddy}$")
    axes[0].set_title("E07b: trajectory cost as N grows")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(Ns, np.array(JsN) + np.array(PeN), "o-", label=r"$J + P_{eddy}$", markersize=8)
    axes[1].semilogx(Ns, JsN, "s--", label=r"$J$", markersize=8)
    axes[1].semilogx(Ns, PeN, "^--", label=r"$P_{eddy}$", markersize=8)
    axes[1].set_xlabel(r"$N$")
    axes[1].set_ylabel("loss")
    axes[1].set_title(r"E07b: total cost vs $N$  ($k_e=k_h=1$)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e07_pareto_N.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.join(out_dir, 'e07_pareto_N.png')}\n")


if __name__ == "__main__":
    main()
