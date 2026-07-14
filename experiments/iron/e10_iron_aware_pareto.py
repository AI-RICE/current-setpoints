"""
E10 -- Phase-4 iron-aware Pareto at p4.

At the iron-dominated cell (p4: T*=1 Nm, n_mech=1500 rpm), where
P_eddy / J ~ 12 in the relative ``k_e=k_h=1`` proxy, augment the
active-set objective with ``iron_weight * P_eddy(X)``. The iron term
is the same eddy quadratic the ``dynamic.iron_loss.eddy_loss`` proxy
computes -- linear in the dq trajectory through the phase-flux
projection and forward-difference -- so it adds an analytical
quadratic to the SLSQP cost.

Sweep ``iron_weight`` over a log range, record:
  J          -- Joule loss (also reported as objective decomposition)
  P_eddy     -- eddy iron-loss proxy
  P_hyst     -- hysteresis proxy
  C          -- chatter amplitude (H=7)
  res_dense  -- max dense voltage residual
  |A|        -- active-set size at convergence

Outputs:
  experiments/iron/figs/e10_pareto_p4.png        (Pareto curve)
  experiments/iron/figs/e10_trajectories_p4.png  (dq panels for selected alpha)
  experiments/iron/figs/e10_iron_aware_table.txt (raw data)

Wall time estimate: ~5 min (7 active-set runs at N=128).

Run:
    python experiments/iron/e10_iron_aware_pareto.py
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

from current_setpoints.optimization import ActiveSetOptimizer
from dynamic.iron_loss import iron_loss
from dynamic.voltage_diagnostics import voltage_residuals_dense
from experiments.chatter.common import chatter_amplitude, joule_loss, make_setup


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    # p4: iron-heavy extreme
    setup = make_setup(n_mech_rpm=1500.0, torq_target=1.0)
    n_grid = 128
    alphas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]

    print(f"p4 iron-aware sweep:  T*=1.0 Nm, n_mech=1500 rpm, N={n_grid}")
    print(f"alphas: {alphas}\n")

    rows = []
    trajectories: dict[float, np.ndarray] = {}

    total_t0 = time.time()
    for alpha in alphas:
        t0 = time.time()
        optimizer = ActiveSetOptimizer(
            setup.fwd,
            n_grid=n_grid,
            max_outer_iter=10,
            tol=1e-3,
            rho=0.0,
            scheme="forward",
            iron_weight=float(alpha),
        )
        sol = optimizer.minimize_current(setup.torq_target, setup.omega_el)
        X = sol.curr_dq
        active = sol.diagnostics.get("active", frozenset())
        J = joule_loss(X)
        loss = iron_loss(setup.fwd, setup.omega_el, X)
        C = chatter_amplitude(X, h_max=7)
        res_dense, _ = voltage_residuals_dense(
            setup.fwd,
            setup.omega_el,
            X,
            setup.machine.volt_max,
        )
        rows.append(
            {
                "alpha": float(alpha),
                "J": J,
                "Pe": loss["eddy"],
                "Ph": loss["hysteresis"],
                "C": C,
                "res": res_dense,
                "A_size": len(active),
                "converged": bool(sol.success),
                "obj_total": J + float(alpha) * loss["eddy"],
            }
        )
        trajectories[float(alpha)] = X
        print(
            f"  alpha={alpha:.1e}  J={J:8.3f}  P_e={loss['eddy']:8.3f}  "
            f"obj=J+alpha*P_e={rows[-1]['obj_total']:8.3f}  "
            f"P_h={loss['hysteresis']:.3f}  C={C:.4f}  "
            f"res={res_dense:+.4f}V  |A|={len(active):3d}  "
            f"t={time.time() - t0:.1f}s"
        )
    print(f"\nTotal wall: {time.time() - total_t0:.1f}s")

    # Save raw table
    table_path = os.path.join(_figs_dir(), "e10_iron_aware_table.txt")
    with open(table_path, "w") as f:
        f.write(f"# p4: T*=1.0 Nm, n_mech=1500 rpm, N={n_grid}\n")
        f.write("# alpha  J  P_eddy  P_hyst  C  res_dense[V]  |A|  obj_total  converged\n")
        for r in rows:
            f.write(
                f"{r['alpha']:.1e} {r['J']:.5f} {r['Pe']:.5f} {r['Ph']:.5f} "
                f"{r['C']:.5f} {r['res']:.5f} {r['A_size']} "
                f"{r['obj_total']:.5f} {r['converged']}\n"
            )
    print(f"saved {table_path}")

    # Pareto plot
    Js = np.array([r["J"] for r in rows])
    Pes = np.array([r["Pe"] for r in rows])
    Cs = np.array([r["C"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc = axes[0].scatter(Js, Pes, c=np.log10(np.maximum(alphas, 1e-12)), cmap="viridis", s=120, edgecolors="k")
    axes[0].plot(Js, Pes, "-", color="grey", alpha=0.5)
    for J_, Pe_, a in zip(Js, Pes, alphas, strict=True):
        axes[0].annotate(rf"$\alpha={a:.0e}$", (J_, Pe_), fontsize=9, xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel(r"Joule loss $J$")
    axes[0].set_ylabel(r"Eddy iron loss $P_{eddy}$")
    axes[0].set_title("p4 Pareto front (J, P_eddy)")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label=r"$\log_{10}\alpha$")

    axes[1].loglog(np.maximum(alphas, 1e-10), Js, "o-", markersize=8, label=r"$J$")
    axes[1].loglog(np.maximum(alphas, 1e-10), Pes, "s-", markersize=8, label=r"$P_{eddy}$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("loss component")
    axes[1].set_title(r"loss vs $\alpha$ at p4")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    axes[2].loglog(np.maximum(alphas, 1e-10), np.maximum(Cs, 1e-6), "o-", markersize=8, color="tab:red")
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"chatter amplitude $C$ (H=7)")
    axes[2].set_title("chatter at p4")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(_figs_dir(), "e10_pareto_p4.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Trajectories at a few alpha values
    sel_alphas = [0.0, 1e-2, 1.0, 10.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=False)
    theta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    cmap = plt.get_cmap("tab10")
    for ax, a in zip(axes.flat, sel_alphas, strict=True):
        X = trajectories[a]
        for j in range(setup.machine.dim):
            ax.plot(
                theta_grid,
                X[:, j],
                color=cmap(j),
                lw=1.5,
                label=[r"$i_{d^1}$", r"$i_{q^1}$", r"$i_{d^3}$", r"$i_{q^3}$"][j],
            )
        rec = next(r for r in rows if r["alpha"] == a)
        ax.set_title(rf"$\alpha = {a:.0e}$    J={rec['J']:.2f}  $P_e$={rec['Pe']:.2f}  C={rec['C']:.3f}")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("dq current [A]")
        if a == sel_alphas[0]:
            ax.legend(fontsize=9, ncols=4, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\theta$ [rad]")
    fig.suptitle("p4 trajectories at selected iron weights")
    fig.tight_layout()
    out = os.path.join(_figs_dir(), "e10_trajectories_p4.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
