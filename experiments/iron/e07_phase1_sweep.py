"""
E07 -- Phase-1 full sweep over the 5 selected operating points.

For each cell in ``experiments.chatter.common.OPERATING_POINTS`` and
each ``rho`` in a small log-spaced list, run the active-set solver at
N=128 and record:

  J          -- Joule loss
  P_eddy     -- eddy iron-loss proxy
  P_hyst     -- hysteresis proxy
  C          -- chatter amplitude (H=7)
  res_dense  -- max dense voltage residual
  |A|        -- active-set size at convergence

Outputs:
  experiments/iron/figs/e07_pareto_per_cell.png   (5 subplots, one per cell)
  experiments/iron/figs/e07_pareto_combined.png   (all cells overlaid)
  experiments/iron/figs/e07_summary_table.txt     (table of all metrics)

Wall time estimate: ~10-15 min on a laptop (25 active-set runs at N=128).

Run:
    python experiments/iron/e07_phase1_sweep.py
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
    rhos = [0.0, 1e-6, 1e-4, 1e-2, 1.0]
    n_grid = 128
    out_dir = _figs_dir()

    results: dict[str, dict] = {}
    total_t0 = time.time()

    for T_star, n_mech, tag, descr in OPERATING_POINTS:
        print(f"\n=== {tag}: T*={T_star} Nm, n_mech={n_mech} rpm  ({descr}) ===")
        setup = make_setup(n_mech_rpm=n_mech, torq_target=T_star)
        cell_data = {
            "T_star": T_star,
            "n_mech": n_mech,
            "descr": descr,
            "rho": [],
            "J": [],
            "Pe": [],
            "Ph": [],
            "C": [],
            "res": [],
            "A_size": [],
            "converged": [],
        }
        for rho in rhos:
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
                rho=float(rho),
                scheme="forward",
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
            cell_data["rho"].append(float(rho))
            cell_data["J"].append(J)
            cell_data["Pe"].append(loss["eddy"])
            cell_data["Ph"].append(loss["hysteresis"])
            cell_data["C"].append(C)
            cell_data["res"].append(res_dense)
            cell_data["A_size"].append(len(r["active"]))
            cell_data["converged"].append(bool(r["converged"]))
            print(
                f"  rho={rho:.1e}  J={J:8.2f}  P_e={loss['eddy']:8.2f}  "
                f"P_h={loss['hysteresis']:6.3f}  C={C:.4f}  res={res_dense:+.4f}V  "
                f"|A|={len(r['active']):3d}  conv={r['converged']}  "
                f"t={time.time() - t0:.1f}s"
            )
        results[tag] = cell_data

    print(f"\nTotal wall time: {time.time() - total_t0:.1f}s")

    # ---------------- summary table -------------------------------
    table_path = os.path.join(out_dir, "e07_summary_table.txt")
    with open(table_path, "w") as f:
        f.write(f"# E07 Phase-1 sweep (N={n_grid}, rho values: {rhos})\n")
        f.write("# tag | T*[Nm] | n_mech[rpm] | rho | J | P_eddy | P_hyst | C | res_dense[V] | |A| | conv\n")
        for tag, d in results.items():
            for k in range(len(d["rho"])):
                f.write(
                    f"{tag} {d['T_star']:.2f} {d['n_mech']:.0f} "
                    f"{d['rho'][k]:.1e} {d['J'][k]:.4f} {d['Pe'][k]:.4f} "
                    f"{d['Ph'][k]:.4f} {d['C'][k]:.4f} {d['res'][k]:.4f} "
                    f"{d['A_size'][k]} {d['converged'][k]}\n"
                )
    print(f"saved {table_path}")

    # ---------------- per-cell Pareto subplots --------------------
    fig, axes = plt.subplots(1, len(OPERATING_POINTS), figsize=(4 * len(OPERATING_POINTS), 5))
    if len(OPERATING_POINTS) == 1:
        axes = [axes]
    for ax, (T_star, n_mech, tag, descr) in zip(axes, OPERATING_POINTS, strict=True):
        d = results[tag]
        J = np.array(d["J"])
        Pe = np.array(d["Pe"])
        ax.scatter(J, Pe, c=np.log10(np.maximum(d["rho"], 1e-12)), cmap="viridis", s=80, edgecolors="k")
        for k, rho in enumerate(d["rho"]):
            ax.annotate(f"{rho:.0e}", (J[k], Pe[k]), fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel(r"Joule loss $J$")
        ax.set_ylabel(r"$P_{eddy}$")
        ax.set_title(f"{tag}\nT*={T_star} Nm  n={n_mech} rpm", fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"E07: per-cell Pareto (J, P_eddy) over rho sweep   N={n_grid}", y=1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, "e07_pareto_per_cell.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---------------- combined log-log Pareto ---------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = ["o", "s", "^", "v", "D"]
    for (T_star, n_mech, tag, descr), marker in zip(OPERATING_POINTS, markers, strict=True):
        d = results[tag]
        J = np.maximum(d["J"], 1e-9)
        Pe = np.maximum(d["Pe"], 1e-9)
        ax.loglog(J, Pe, marker=marker, linestyle="-", label=f"{tag} (T*={T_star}, n={n_mech})")
    ax.set_xlabel(r"Joule loss $J$")
    ax.set_ylabel(r"$P_{eddy}$ iron loss proxy")
    ax.set_title("E07: combined Pareto across operating points")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, "e07_pareto_combined.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---------------- chatter vs rho per cell ---------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for (T_star, n_mech, tag, descr), marker in zip(OPERATING_POINTS, markers, strict=True):
        d = results[tag]
        rho_plot = np.maximum(d["rho"], 1e-10)
        ax.loglog(
            rho_plot, np.maximum(d["C"], 1e-6), marker=marker, linestyle="-", label=f"{tag} (T*={T_star}, n={n_mech})"
        )
    ax.set_xlabel(r"slew weight $\rho$")
    ax.set_ylabel(r"chatter amplitude $C$ (H=7)")
    ax.set_title("E07: chatter response to smoothness penalty per operating point")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, "e07_chatter_vs_rho.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
