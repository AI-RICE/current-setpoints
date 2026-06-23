"""
Active-set demo at the point where R0 fails the voltage constraint.

We pick the operating point where the speed sweep at fixed torque
(``02_speed_sweep_at_fixed_torque.py``) showed the dynamic voltage
exceeding ``V_max`` in the R0 reconstruction -- n_mech = 1400 rpm,
T* = 4 Nm. We run the active-set / successive-relaxation algorithm
(``dynamic/active_set_optimizer.py``) and save two figures:

  * the R0 starting trajectory (same as 02 at this point), and
  * the final active-set converged trajectory.

The max voltage residual is also printed per outer iteration to make
the convergence visible.
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

from current_setpoints.optimization import ModelAnalytical
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform
from current_setpoints.utils.plotting_dynamic import plot_dq_phase_combined
from dynamic.active_set_optimizer import (
    run_active_set,
    voltage_residuals,
    voltage_residuals_dense,
)
from dynamic.regimes import active_set, active_set_tag, fingerprint


def main() -> None:
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()
    transform = Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=700)
    model = ModelAnalytical(machine=machine, flux=flux)

    n_mech_rpm = 1400.0
    torq_target = 4.0
    n_grid = 128

    omega_el = n_mech_rpm * (np.pi / 30.0) * machine.n_ppairs
    print(f"Operating point: n_mech={n_mech_rpm} rpm  -> omega_el={omega_el:.3f} rad/s")
    print(f"                 T*={torq_target} Nm, n_grid={n_grid}")
    print(f"                 V_max={machine.volt_max} V, I_max={machine.curr_max} A\n")

    result = run_active_set(
        model=model,
        transform=transform,
        omega=omega_el,
        torq_target=torq_target,
        curr_max=machine.curr_max,
        volt_max=machine.volt_max,
        n_grid=n_grid,
        max_outer_iter=8,
        tol=1e-3,
    )

    print("History:")
    for h in result["history"]:
        print(
            f"  iter={h['iter']}  max_residual={h['max_residual_V']:+.4f} V  "
            f"n_violating={h['n_violating']:3d}  active_before={h['active_size_before']:3d}  "
            f"{h.get('action', '')}"
        )
    print(f"\nconverged: {result['converged']}, final |A| = {len(result['active'])}")

    r_r0_node, _ = voltage_residuals(transform, omega_el, result["r0_curr_dq_grid"], machine.volt_max)
    r_fin_node, _ = voltage_residuals(transform, omega_el, result["curr_dq_grid"], machine.volt_max)
    r_r0_dense, _ = voltage_residuals_dense(transform, omega_el, result["r0_curr_dq_grid"], machine.volt_max)
    r_fin_dense, _ = voltage_residuals_dense(transform, omega_el, result["curr_dq_grid"], machine.volt_max)
    print("Voltage residuals (positive = overshoot above V_max):")
    print(f"  R0          coarse-node: {r_r0_node.max():+.4f} V   dense plot: {r_r0_dense:+.4f} V")
    print(f"  Active-set  coarse-node: {r_fin_node.max():+.4f} V   dense plot: {r_fin_dense:+.4f} V\n")

    fp_r0 = fingerprint(transform, omega_el, result["r0_curr_dq_grid"], machine.curr_max, machine.volt_max)
    fp_fin = fingerprint(transform, omega_el, result["curr_dq_grid"], machine.curr_max, machine.volt_max)
    tag_r0 = active_set_tag(active_set(*fp_r0))
    tag_fin = active_set_tag(active_set(*fp_fin))
    print(f"R0 active set tag:        {tag_r0}")
    print(f"Active-set active set tag: {tag_fin}")

    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out_dir, exist_ok=True)

    for which, dq, tag in (("r0", result["r0_curr_dq_grid"], tag_r0), ("activeset", result["curr_dq_grid"], tag_fin)):
        title = f"T*={torq_target:.2f}Nm, n_mech={n_mech_rpm:.0f}rpm, {which} (active={tag})"
        fig, _ = plot_dq_phase_combined(
            transform,
            omega_el,
            dq,
            curr_max=machine.curr_max,
            volt_max=machine.volt_max,
            title=title,
        )
        out_path = os.path.join(out_dir, f"actset_n{n_mech_rpm:0.0f}rpm_T{torq_target:0.2f}_{which}.png")
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
