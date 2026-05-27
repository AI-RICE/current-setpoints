"""
Second dynamic-MTPA experiment: sweep electrical speed at fixed torque.

Mirror of ``01_low_speed_torque_sweep.py``, but with the roles of
``T*`` and ``omega`` swapped. We pick a moderate ``T*`` that is feasible
across a wide speed range and increase ``omega`` until the static MTPA
problem becomes infeasible (the local torque ceiling drops below ``T*``).

At each speed we solve:
  (a) static MTPA (``MotorOptimizer.minimize_current``);
  (b) R0 independent-per-angle relaxation (``run_independent_per_angle``).

Regime detection reuses the arc-fraction + arc-count fingerprint from
``01_low_speed_torque_sweep.py``. The expectation, by analogy with the
torque sweep, is a single transition ``free -> V_all_x?`` once the
inductive + BEMF voltage drives any phase up to the limit on an arc.

Outputs land in ``dynamic/figs/`` next to this script, prefixed
``speedsweep_``.
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")

from current_setpoints.optimization import ModelAnalytical, MotorOptimizer
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform
from current_setpoints.utils.plotting_dynamic import plot_dq_phase_combined
from dynamic.independent_optimizer import run_independent_per_angle
from dynamic.regimes import active_set, active_set_tag, fingerprint


def main() -> None:
    machine = IEEEMachine2()
    nmax_mech_rpm = 1800
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=nmax_mech_rpm)

    flux = Flux_IEEEMachine2()
    transform = Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=700)

    analytical = ModelAnalytical(machine=machine, flux=flux)
    solver_opts = {"disp": False, "ftol": 1e-8, "maxiter": 500, "eps": 1e-8}
    optimizer = MotorOptimizer(model=analytical, opts=solver_opts)

    torq_target = 4.0  # Nm -- well below the low-speed maximum (~7.76 Nm)
    n_grid = 64

    n_sweep = 30
    rpm_min, rpm_max = 50.0, 1500.0
    n_mech_grid = np.linspace(rpm_min, rpm_max, n_sweep)
    omega_grid = n_mech_grid * (np.pi / 30.0) * machine.n_ppairs

    static_dq: list[np.ndarray] = []
    static_ok: list[bool] = []
    dyn_dq: list[np.ndarray] = []
    dyn_ok: list[np.ndarray] = []
    torq_caps: list[float] = []
    actives: list[frozenset[tuple[str, int, int]] | None] = []

    curr_dq_prev = np.array([1.0, 0.0, 0.0, 0.0])
    for omega_el in omega_grid:
        _, torq_max_local, ok_top = optimizer.maximize_torque(omega=omega_el, transform=transform)
        torq_caps.append(torq_max_local if ok_top else float("nan"))

        feasible = ok_top and torq_max_local > torq_target
        if not feasible:
            static_dq.append(np.full(transform.dim, np.nan))
            static_ok.append(False)
            dyn_dq.append(np.full((n_grid, transform.dim), np.nan))
            dyn_ok.append(np.zeros(n_grid, dtype=bool))
            actives.append(None)
            continue

        x_static, ok_s = optimizer.minimize_current(
            torq_target=torq_target,
            omega=omega_el,
            transform=transform,
            vec_curr_guess=curr_dq_prev,
        )
        if ok_s:
            curr_dq_prev = x_static
        static_dq.append(x_static)
        static_ok.append(bool(ok_s))

        _, x_dyn, ok_dyn = run_independent_per_angle(
            model=analytical,
            transform=transform,
            omega=omega_el,
            torq_target=torq_target,
            curr_max=machine.curr_max,
            volt_max=machine.volt_max,
            n_grid=n_grid,
            warm_start=x_static if ok_s else None,
        )
        dyn_dq.append(x_dyn)
        dyn_ok.append(ok_dyn)

        if bool(np.all(ok_dyn)):
            alpha_I, alpha_V, n_I, n_V = fingerprint(transform, omega_el, x_dyn, machine.curr_max, machine.volt_max)
            actives.append(active_set(alpha_I, alpha_V, n_I, n_V))
        else:
            actives.append(None)

    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out_dir, exist_ok=True)

    crossings: list[tuple[int, int]] = []
    for i in range(1, n_sweep):
        a_prev, a_curr = actives[i - 1], actives[i]
        if a_prev is None or a_curr is None:
            continue
        if a_prev != a_curr:
            crossings.append((i - 1, i))

    print(
        f"Sweep done: T*={torq_target:.2f} Nm fixed; n_mech sweep over "
        f"[{rpm_min:.0f}, {rpm_max:.0f}] rpm ({n_sweep} points)"
    )
    print("active sets along sweep:")
    for rpm, T_cap, a in zip(n_mech_grid, torq_caps, actives, strict=True):
        cap_s = f"{T_cap:6.2f}" if np.isfinite(T_cap) else "  nan"
        tag = active_set_tag(a) if a is not None else "FAIL"
        print(f"  n_mech={rpm:7.1f} rpm  T_cap={cap_s} Nm  ->  {tag}")
    if not crossings:
        print("\nNo active-set changes detected along this sweep.")
    else:
        print(f"\nActive-set changes at indices: {crossings}")

    saved: list[str] = []
    for i_before, i_after in crossings:
        for which, idx in (("before", i_before), ("after", i_after)):
            rpm = n_mech_grid[idx]
            tag_active = active_set_tag(actives[idx])
            tag = f"n{rpm:0.0f}rpm_{tag_active}_{which}"
            title = f"T*={torq_target:.2f}Nm, n_mech={rpm:.0f}rpm, active={tag_active} ({which} crossing)"
            fig, _ = plot_dq_phase_combined(
                transform,
                omega_grid[idx],
                dyn_dq[idx],
                curr_dq_static=static_dq[idx] if static_ok[idx] else None,
                curr_max=machine.curr_max,
                volt_max=machine.volt_max,
                title=title,
            )
            out_path = os.path.join(out_dir, f"speedsweep_{tag}.png")
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)
            print(f"  saved {out_path}")

    print(f"\nDone. {len(saved)} figure(s) written to {out_dir}.")


if __name__ == "__main__":
    main()
