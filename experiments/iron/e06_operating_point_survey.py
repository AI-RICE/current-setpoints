"""
E06 -- Phase-0 operating-point survey.

Coarse mesh over (T*, n_mech_rpm) using only the cheap R0 (independent
per-angle) solver. For each cell we record:

  J_R0       -- Joule loss of the R0 trajectory
  res_max    -- worst dynamic voltage residual on R0 (positive = R0
                violates the dynamic voltage limit, telling us the
                active-set algorithm would change things here)
  P_eddy     -- eddy iron-loss proxy on R0
  P_hyst     -- Steinmetz hysteresis proxy on R0
  ratio      -- P_eddy / J_R0
  feasible   -- whether the static torque ceiling exceeds T*

Heatmaps over the mesh make the four regions visible (free,
current-limited, voltage-limited, corner) and let us pick ~5
representative cells to scope Phases 1--4 to.

Run:
    python experiments/iron/e06_operating_point_survey.py
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

from current_setpoints.optimization import ModelAnalytical, MotorOptimizer
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform
from dynamic.active_set_optimizer import voltage_residuals
from dynamic.independent_optimizer import run_independent_per_angle
from dynamic.iron_loss import iron_loss
from experiments.chatter.common import joule_loss


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()
    transform = Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=700)
    model = ModelAnalytical(machine=machine, flux=flux)
    optimizer = MotorOptimizer(
        model=model,
        opts={"disp": False, "ftol": 1e-8, "maxiter": 500, "eps": 1e-8},
    )

    torq_targets = np.array([1.0, 3.0, 4.5, 6.0, 7.0])
    rpm_targets = np.array([100.0, 400.0, 800.0, 1200.0, 1500.0])
    omega_targets = rpm_targets * (np.pi / 30.0) * machine.n_ppairs
    n_grid = 32  # cheap

    J = np.full((len(torq_targets), len(rpm_targets)), np.nan)
    res = np.full_like(J, np.nan)
    Pe = np.full_like(J, np.nan)
    Ph = np.full_like(J, np.nan)
    feas = np.zeros_like(J, dtype=bool)
    Tcap = np.full(len(rpm_targets), np.nan)

    print(f"Mesh: {len(torq_targets)} torques x {len(rpm_targets)} speeds, N={n_grid} R0 solver per cell.\n")

    for j, (omega_el, rpm) in enumerate(zip(omega_targets, rpm_targets, strict=True)):
        _, t_cap, ok_top = optimizer.maximize_torque(omega=omega_el, transform=transform)
        Tcap[j] = t_cap if ok_top else np.nan
        for i, T in enumerate(torq_targets):
            if not (ok_top and t_cap > T):
                continue
            _, X, ok = run_independent_per_angle(
                model=model,
                transform=transform,
                omega=omega_el,
                torq_target=float(T),
                curr_max=machine.curr_max,
                volt_max=machine.volt_max,
                n_grid=n_grid,
            )
            if not bool(np.all(ok)):
                continue
            feas[i, j] = True
            J[i, j] = joule_loss(X)
            r, _ = voltage_residuals(transform, omega_el, X, machine.volt_max, scheme="forward")
            res[i, j] = float(np.max(r))
            loss = iron_loss(transform, omega_el, X)
            Pe[i, j] = loss["eddy"]
            Ph[i, j] = loss["hysteresis"]
            print(
                f"  T*={T:4.1f} Nm  n={rpm:6.0f} rpm  Tcap={t_cap:5.2f}  "
                f"J={J[i, j]:7.2f}  res={res[i, j]:+6.2f} V  "
                f"P_e={Pe[i, j]:7.2f}  ratio={Pe[i, j] / J[i, j]:5.3f}"
            )

    print("\nStatic torque ceilings by speed:")
    for rpm, t_cap in zip(rpm_targets, Tcap, strict=True):
        print(f"  n={rpm:6.0f} rpm  Tcap={t_cap:.2f} Nm")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        (J, "Joule loss J", "viridis"),
        (res, "max V residual on R0 [V]", "RdBu_r"),
        (Pe, r"$P_{eddy}$  ($k_e=1$)", "viridis"),
        (Pe / np.where(J > 0, J, np.nan), r"$P_{eddy}/J$", "magma"),
    ]
    for ax, (Z, title, cmap) in zip(axes.flat, panels, strict=True):
        masked = np.ma.array(Z, mask=~feas)
        if cmap == "RdBu_r":
            vmax = np.nanmax(np.abs(Z))
            im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(rpm_targets)))
        ax.set_xticklabels([f"{r:.0f}" for r in rpm_targets])
        ax.set_yticks(range(len(torq_targets)))
        ax.set_yticklabels([f"{t:.1f}" for t in torq_targets])
        ax.set_xlabel("n_mech [rpm]")
        ax.set_ylabel("T* [Nm]")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i_ in range(len(torq_targets)):
            for j_ in range(len(rpm_targets)):
                if feas[i_, j_]:
                    val = Z[i_, j_]
                    if cmap == "RdBu_r":
                        text = f"{val:.2f}"
                    elif val >= 100:
                        text = f"{val:.0f}"
                    else:
                        text = f"{val:.2f}"
                    ax.text(
                        j_,
                        i_,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="w" if not np.isfinite(val) or abs(val) > 10 else "k",
                    )
    fig.suptitle("E06: operating-point survey on R0 trajectories  (N=32)")
    out = os.path.join(_figs_dir(), "e06_operating_point_survey.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")

    # ----- candidate selection: print 5 representative cells -------
    print("\nCandidate operating points for Phase 1--4:")
    cells: list[tuple[float, float, str]] = []
    feas_idx = np.argwhere(feas)
    if not len(feas_idx):
        print("  (no feasible cells found)")
        return

    # 1. lowest J among feasible -- "free" baseline
    i0 = feas_idx[np.argmin(J[feas])]
    cells.append((torq_targets[i0[0]], rpm_targets[i0[1]], "free baseline (low J)"))

    # 2. highest J -- current-limited corner
    i1 = feas_idx[np.argmax(J[feas])]
    cells.append((torq_targets[i1[0]], rpm_targets[i1[1]], "high-J / current-limited"))

    # 3. highest residual -- where active-set matters most
    i2 = feas_idx[np.argmax(np.where(feas, res, -np.inf)[feas])]
    cells.append((torq_targets[i2[0]], rpm_targets[i2[1]], "max V-residual"))

    # 4. highest P_eddy / J -- iron-dominated
    ratio = Pe / np.where(J > 0, J, np.inf)
    i3 = feas_idx[np.argmax(ratio[feas])]
    cells.append((torq_targets[i3[0]], rpm_targets[i3[1]], "highest P_eddy/J"))

    # 5. median residual -- "typical" intermediate point
    med = np.nanmedian(np.where(feas, res, np.nan))
    diffs = np.where(feas, np.abs(res - med), np.inf)
    i4 = np.unravel_index(np.argmin(diffs), diffs.shape)
    cells.append((torq_targets[i4[0]], rpm_targets[i4[1]], "median V-residual"))

    seen: set[tuple[float, float]] = set()
    for T, n, tag in cells:
        if (T, n) in seen:
            continue
        seen.add((T, n))
        print(f"  T*={T:.2f} Nm,  n_mech={n:.0f} rpm   --  {tag}")


if __name__ == "__main__":
    main()
