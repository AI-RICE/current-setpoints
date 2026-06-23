"""
E11 -- Iron-aware Pareto at p3, p4, p5 (eddy term).

Repeats the iron-aware sweep of E10 across the three cells with binding
active set, to test whether the "iron loss is operating-point dominated,
not trajectory dominated" finding at p4 generalises.

Sweep ``iron_weight`` over a short log range at N=128 for each cell;
record J, P_eddy, P_hyst, C, dense residual, |A|, total objective.

Outputs:
  experiments/iron/figs/e11_pareto_per_cell.png
  experiments/iron/figs/e11_loss_vs_alpha.png
  experiments/iron/figs/e11_table.txt
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
from dynamic.iron_loss import iron_loss
from experiments.chatter.common import (
    OPERATING_POINTS,
    chatter_amplitude,
    joule_loss,
    make_setup,
)


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    n_grid = 128
    alphas = [0.0, 1e-2, 1.0, 10.0]
    # Only the three cells where R0 is infeasible (active-set non-trivial).
    cells = [c for c in OPERATING_POINTS if c[2] in {"p3_curr_edge", "p4_iron_heavy", "p5_corner"}]

    results: dict[str, dict] = {}
    total_t0 = time.time()

    for T_star, n_mech, tag, descr in cells:
        print(f"\n=== {tag}: T*={T_star} Nm, n_mech={n_mech} rpm ===")
        setup = make_setup(n_mech_rpm=n_mech, torq_target=T_star)
        cell_data = {
            "T_star": T_star,
            "n_mech": n_mech,
            "descr": descr,
            "alpha": [],
            "J": [],
            "Pe": [],
            "Ph": [],
            "C": [],
            "res": [],
            "A_size": [],
            "converged": [],
        }
        for alpha in alphas:
            t0 = time.time()
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
                rho=0.0,
                scheme="forward",
                iron_weight=float(alpha),
            )
            X = r["curr_dq_grid"]
            J = joule_loss(X)
            loss = iron_loss(setup.transform, setup.omega_el, X)
            C = chatter_amplitude(X, h_max=7)
            res_dense, _ = voltage_residuals_dense(
                setup.transform,
                setup.omega_el,
                X,
                setup.machine.volt_max,
            )
            cell_data["alpha"].append(float(alpha))
            cell_data["J"].append(J)
            cell_data["Pe"].append(loss["eddy"])
            cell_data["Ph"].append(loss["hysteresis"])
            cell_data["C"].append(C)
            cell_data["res"].append(res_dense)
            cell_data["A_size"].append(len(r["active"]))
            cell_data["converged"].append(bool(r["converged"]))
            print(
                f"  alpha={alpha:.1e}  J={J:8.3f}  P_e={loss['eddy']:8.3f}  "
                f"C={C:.4f}  res={res_dense:+.4f}V  |A|={len(r['active']):3d}  "
                f"t={time.time() - t0:.1f}s"
            )
        results[tag] = cell_data
    print(f"\nTotal wall: {time.time() - total_t0:.1f}s")

    # Save table
    tab = os.path.join(_figs_dir(), "e11_table.txt")
    with open(tab, "w") as f:
        f.write(f"# E11 iron-aware multi-cell (N={n_grid})\n")
        f.write("# tag T* n_mech alpha J P_eddy P_hyst C res |A|\n")
        for tag, d in results.items():
            for k in range(len(d["alpha"])):
                f.write(
                    f"{tag} {d['T_star']:.2f} {d['n_mech']:.0f} "
                    f"{d['alpha'][k]:.1e} {d['J'][k]:.4f} {d['Pe'][k]:.4f} "
                    f"{d['Ph'][k]:.4f} {d['C'][k]:.4f} {d['res'][k]:.4f} "
                    f"{d['A_size'][k]}\n"
                )
    print(f"saved {tab}")

    # per-cell Pareto plot
    fig, axes = plt.subplots(1, len(cells), figsize=(5.5 * len(cells), 5))
    if len(cells) == 1:
        axes = [axes]
    for ax, (T_star, n_mech, tag, _) in zip(axes, cells, strict=True):
        d = results[tag]
        sc = ax.scatter(
            d["J"], d["Pe"], c=np.log10(np.maximum(d["alpha"], 1e-12)), cmap="viridis", s=120, edgecolors="k"
        )
        ax.plot(d["J"], d["Pe"], "-", color="grey", alpha=0.4)
        for J_, P_, a in zip(d["J"], d["Pe"], d["alpha"], strict=True):
            ax.annotate(rf"$\alpha={a:.0e}$", (J_, P_), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(r"Joule loss $J$")
        ax.set_ylabel(r"Eddy iron loss $P_{eddy}$")
        ax.set_title(f"{tag}\nT*={T_star} Nm  n={n_mech} rpm", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label=r"$\log_{10}\alpha$")
    fig.tight_layout()
    out = os.path.join(_figs_dir(), "e11_pareto_per_cell.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # relative loss vs alpha
    fig, axes = plt.subplots(2, len(cells), figsize=(5.5 * len(cells), 9), sharex=True)
    if len(cells) == 1:
        axes = axes.reshape(2, 1)
    for col, (T_star, n_mech, tag, _) in enumerate(cells):
        d = results[tag]
        a_arr = np.maximum(d["alpha"], 1e-12)
        J_rel = np.array(d["J"]) / d["J"][0]
        Pe_rel = np.array(d["Pe"]) / d["Pe"][0]
        axes[0, col].semilogx(a_arr, J_rel, "o-", markersize=8, label=r"$J/J_0$")
        axes[0, col].semilogx(a_arr, Pe_rel, "s-", markersize=8, label=r"$P_{eddy}/P_{eddy,0}$")
        axes[0, col].set_title(f"{tag}")
        axes[0, col].grid(True, which="both", alpha=0.3)
        axes[0, col].legend(fontsize=9)
        axes[0, col].axhline(1.0, color="k", lw=0.8, ls="--")
        axes[0, col].set_ylabel("normalised loss")

        axes[1, col].semilogx(a_arr, np.maximum(d["C"], 1e-6), "o-", markersize=8, color="tab:red")
        axes[1, col].set_xlabel(r"$\alpha$")
        axes[1, col].set_ylabel(r"chatter $C$ (H=7)")
        axes[1, col].grid(True, which="both", alpha=0.3)
    fig.suptitle("E11: iron-aware sweep, relative loss components per cell")
    fig.tight_layout()
    out = os.path.join(_figs_dir(), "e11_loss_vs_alpha.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
