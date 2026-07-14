"""
Active-set demo at the point where R0 fails the voltage constraint.

We pick the operating point where the speed sweep at fixed torque
(``02_speed_sweep_at_fixed_torque.py``) showed the dynamic voltage
exceeding ``V_max`` in the R0 reconstruction -- n_mech = 1400 rpm,
T* = 4 Nm. We run the active-set / successive-relaxation algorithm
(``current_setpoints.optimization.ActiveSetOptimizer``) and save two figures:

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

from current_setpoints.models.forward_model import ForwardModel
from current_setpoints.models.machines import PMSM5Phase
from current_setpoints.optimization import ActiveSetOptimizer
from dynamic._waveforms import plot_dq_phase_combined
from dynamic.voltage_diagnostics import voltage_residuals_dense
from dynamic.regimes import active_set, active_set_tag, fingerprint


def main() -> None:
    machine = PMSM5Phase()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    fwd = ForwardModel(machine, add_volt_0=False, n_theta=700)

    n_mech_rpm = 1400.0
    torq_target = 4.0
    n_grid = 128

    omega_el = n_mech_rpm * (np.pi / 30.0) * machine.n_ppairs
    print(f"Operating point: n_mech={n_mech_rpm} rpm  -> omega_el={omega_el:.3f} rad/s")
    print(f"                 T*={torq_target} Nm, n_grid={n_grid}")
    print(f"                 V_max={machine.volt_max} V, I_max={machine.curr_max} A\n")

    optimizer = ActiveSetOptimizer(fwd, n_grid=n_grid, max_outer_iter=8, tol=1e-3)
    sol = optimizer.minimize_current(torq_target, omega_el)
    diag = sol.diagnostics

    print("History:")
    for h in diag.get("history", []):
        print(
            f"  iter={h['iter']}  max_residual={h['max_V_residual']:+.4f} V  "
            f"n_violating={h['n_violating']:3d}  active_before={h['active_size']:3d}  "
            f"{h.get('action', '')}"
        )
    active = diag.get("active", frozenset())
    r0_grid = diag.get("r0_grid", sol.curr_dq)
    print(f"\nconverged: {sol.success}, final |A| = {len(active)}")

    maps = optimizer._precompute_maps(omega_el)
    r_r0_node, _ = optimizer.voltage_residuals(maps, omega_el, r0_grid)
    r_fin_node, _ = optimizer.voltage_residuals(maps, omega_el, sol.curr_dq)
    r_r0_dense, _ = voltage_residuals_dense(fwd, omega_el, r0_grid, machine.volt_max)
    r_fin_dense, _ = voltage_residuals_dense(fwd, omega_el, sol.curr_dq, machine.volt_max)
    print("Voltage residuals (positive = overshoot above V_max):")
    print(f"  R0          coarse-node: {r_r0_node.max():+.4f} V   dense plot: {r_r0_dense:+.4f} V")
    print(f"  Active-set  coarse-node: {r_fin_node.max():+.4f} V   dense plot: {r_fin_dense:+.4f} V\n")

    fp_r0 = fingerprint(fwd, omega_el, r0_grid, machine.curr_max, machine.volt_max)
    fp_fin = fingerprint(fwd, omega_el, sol.curr_dq, machine.curr_max, machine.volt_max)
    tag_r0 = active_set_tag(active_set(*fp_r0))
    tag_fin = active_set_tag(active_set(*fp_fin))
    print(f"R0 active set tag:        {tag_r0}")
    print(f"Active-set active set tag: {tag_fin}")

    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out_dir, exist_ok=True)

    for which, dq, tag in (("r0", r0_grid, tag_r0), ("activeset", sol.curr_dq, tag_fin)):
        title = f"T*={torq_target:.2f}Nm, n_mech={n_mech_rpm:.0f}rpm, {which} (active={tag})"
        fig, _ = plot_dq_phase_combined(
            fwd,
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
