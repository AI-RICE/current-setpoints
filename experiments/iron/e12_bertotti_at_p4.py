"""
E12 -- Bertotti excess loss as iron-aware objective at p4.

Compare three iron-aware optimisations at the iron-heavy cell p4:
  (a) eddy-only objective: ``iron_weight = alpha``
  (b) bertotti-only objective: ``excess_weight = alpha``
  (c) joint: both weighted equally

For each variant, sweep alpha and record J, P_eddy (post-hoc), P_excess
(post-hoc Bertotti integral on the trajectory), P_hyst, chatter, and
residual. If both (a) and (b) yield the same flat Pareto, the
"operating-point-dominated" conclusion is robust to the iron-loss model
(eddy quadratic vs. Bertotti 3/2-power).
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
from dynamic.iron_loss import eddy_loss, excess_loss, hysteresis_loss
from dynamic.voltage_diagnostics import voltage_residuals_dense
from experiments.chatter.common import chatter_amplitude, joule_loss, make_setup


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    setup = make_setup(n_mech_rpm=1500.0, torq_target=1.0)
    n_grid = 128
    alphas = [0.0, 1e-2, 1.0, 10.0, 100.0]

    variants = [
        ("eddy", lambda a: {"iron_weight": a, "excess_weight": 0.0}),
        ("bertotti", lambda a: {"iron_weight": 0.0, "excess_weight": a}),
        ("joint", lambda a: {"iron_weight": a, "excess_weight": a}),
    ]

    results: dict[str, dict] = {}
    t_all = time.time()
    for name, kw_fun in variants:
        print(f"\n=== variant: {name} ===")
        d = {"alpha": [], "J": [], "Pe": [], "Px": [], "Ph": [], "C": [], "res": [], "A_size": []}
        for alpha in alphas:
            t0 = time.time()
            kw = kw_fun(alpha)
            optimizer = ActiveSetOptimizer(
                setup.fwd,
                n_grid=n_grid,
                max_outer_iter=10,
                tol=1e-3,
                rho=0.0,
                scheme="forward",
                **kw,
            )
            sol = optimizer.minimize_current(setup.torq_target, setup.omega_el)
            X = sol.curr_dq
            active = sol.diagnostics.get("active", frozenset())
            J = joule_loss(X)
            Pe = eddy_loss(setup.fwd, setup.omega_el, X)
            Px = excess_loss(setup.fwd, setup.omega_el, X)
            Ph = hysteresis_loss(setup.fwd, setup.omega_el, X)
            C = chatter_amplitude(X, h_max=7)
            res_dense, _ = voltage_residuals_dense(
                setup.fwd,
                setup.omega_el,
                X,
                setup.machine.volt_max,
            )
            d["alpha"].append(float(alpha))
            d["J"].append(J)
            d["Pe"].append(Pe)
            d["Px"].append(Px)
            d["Ph"].append(Ph)
            d["C"].append(C)
            d["res"].append(res_dense)
            d["A_size"].append(len(active))
            print(
                f"  alpha={alpha:.1e}  J={J:8.3f}  P_e={Pe:8.3f}  P_x={Px:8.3f}  "
                f"C={C:.4f}  res={res_dense:+.4f}V  t={time.time() - t0:.1f}s"
            )
        results[name] = d
    print(f"\nTotal wall: {time.time() - t_all:.1f}s")

    # write table
    tab = os.path.join(_figs_dir(), "e12_table.txt")
    with open(tab, "w") as f:
        f.write(f"# E12 Bertotti at p4 (N={n_grid})\n")
        f.write("# variant alpha J P_eddy P_bertotti P_hyst C res |A|\n")
        for name, d in results.items():
            for k in range(len(d["alpha"])):
                f.write(
                    f"{name} {d['alpha'][k]:.1e} {d['J'][k]:.4f} {d['Pe'][k]:.4f} "
                    f"{d['Px'][k]:.4f} {d['Ph'][k]:.4f} {d['C'][k]:.4f} "
                    f"{d['res'][k]:.4f} {d['A_size'][k]}\n"
                )
    print(f"saved {tab}")

    # plot per-variant: J, P_e, P_x normalised vs alpha
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for col, name in enumerate(["eddy", "bertotti", "joint"]):
        d = results[name]
        a_arr = np.maximum(d["alpha"], 1e-12)
        J0 = d["J"][0]
        Pe0 = d["Pe"][0]
        Px0 = d["Px"][0]
        axes[0, col].semilogx(a_arr, np.array(d["J"]) / J0, "o-", label=r"$J/J_0$")
        axes[0, col].semilogx(a_arr, np.array(d["Pe"]) / Pe0, "s-", label=r"$P_e/P_{e,0}$")
        axes[0, col].semilogx(a_arr, np.array(d["Px"]) / Px0, "^-", label=r"$P_x/P_{x,0}$")
        axes[0, col].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
        axes[0, col].set_title(f"variant: {name}")
        axes[0, col].set_ylabel("normalised loss")
        axes[0, col].grid(True, which="both", alpha=0.3)
        axes[0, col].legend(fontsize=9)

        axes[1, col].semilogx(a_arr, np.maximum(d["C"], 1e-6), "o-", color="tab:red", label="C (H=7)")
        axes[1, col].set_xlabel(r"$\alpha$")
        axes[1, col].set_ylabel("chatter")
        axes[1, col].grid(True, which="both", alpha=0.3)
    fig.suptitle("E12: Bertotti vs eddy iron-loss objectives at p4 (T*=1 Nm, n=1500 rpm)")
    fig.tight_layout()
    out = os.path.join(_figs_dir(), "e12_loss_vs_alpha.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
