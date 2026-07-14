"""
First dynamic-MTPA experiment: sweep torque at low speed.

At a single low electrical speed ``omega`` (so the inductive d/dtheta term
is small but non-zero), we sweep the target torque ``T*`` over a uniform
grid from zero to the local maximum and at each step solve:

  (a) the existing static MTPA (``MotorOptimizer.minimize_current``);
  (b) the R0 independent-per-angle relaxation of the dynamic problem,
      see ``dynamic/independent_optimizer.py``.

Regime classification: in the dynamic setting ``count_peaks`` does not
apply -- the limit can be touched on whole arcs, not at isolated points.
We instead summarise each solution by a *fingerprint*:

    alpha_k^I = (1 / 2 pi) * arc-length{ theta : |i_k(theta)| >= (1 - eps) I_max }
    alpha_k^V = (1 / 2 pi) * arc-length{ theta : |v_k(theta)| >= (1 - eps) V_max }

and define the *active set* as the (kind, phase) pairs whose fingerprint
component exceeds a small threshold. A regime change is any change in the
active set between consecutive T* values. For each crossing we save the
combined 3-panel waveform plot for the T* just before and just after, with
the R0 dynamic trajectory as solid lines and the static MTPA solution as
the dashed reference.

Outputs land in ``dynamic/figs/`` next to this script.
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
from current_setpoints.models.machines import ieee_machine2
from current_setpoints.optimization import IndependentOptimizer, StaticOptimizer
from dynamic._waveforms import plot_dq_phase_combined
from dynamic.regimes import active_set, active_set_tag, fingerprint


def main() -> None:
    nmax_mech_rpm = 1800
    machine = ieee_machine2(curr_max=30.0, volt_max=13.0, omega_max=nmax_mech_rpm)

    fwd = ForwardModel(machine, add_volt_0=False, n_theta=700)

    solver_opts = {"disp": False, "ftol": 1e-8, "maxiter": 500, "eps": 1e-8}
    optimizer = StaticOptimizer(fwd, opts=solver_opts)

    # Low speed: 50 RPM mechanical. Convert to electrical rad/s.
    n_mech_rpm = 50.0
    omega_el = n_mech_rpm * (np.pi / 30.0) * machine.n_ppairs

    # Local torque ceiling at this speed.
    sol_top = optimizer.maximize_torque(omega_el)
    if not sol_top.success:
        raise RuntimeError(f"maximize_torque failed at omega_el={omega_el:.3f}")
    torq_max_local = sol_top.torque

    n_sweep = 25
    torq_targets = np.linspace(0.5, 0.98 * torq_max_local, n_sweep)
    n_grid = 64
    # Note: unlike the pre-refactor run_independent_per_angle, IndependentOptimizer
    # does not accept an external warm-start across calls (each solve reseeds from
    # fwd.drive.seeds()[0] and chains internally along its own angle grid).
    ind_optimizer = IndependentOptimizer(fwd, opts=solver_opts, n_grid=n_grid)

    static_dq: list[np.ndarray] = []
    static_ok: list[bool] = []
    dyn_dq: list[np.ndarray] = []
    dyn_ok: list[np.ndarray] = []
    actives: list[frozenset[tuple[str, int]] | None] = []

    curr_dq_prev = np.array([1.0, 0.0, 0.0, 0.0])
    for T in torq_targets:
        sol_s = optimizer.minimize_current(float(T), omega_el, guess=curr_dq_prev)
        if sol_s.success:
            curr_dq_prev = sol_s.curr_dq
        static_dq.append(sol_s.curr_dq)
        static_ok.append(bool(sol_s.success))

        sol_dyn = ind_optimizer.minimize_current(float(T), omega_el)
        x_dyn = sol_dyn.curr_dq
        ok_dyn = sol_dyn.diagnostics["ok_grid"]
        dyn_dq.append(x_dyn)
        dyn_ok.append(ok_dyn)

        if bool(np.all(ok_dyn)):
            alpha_I, alpha_V, n_I, n_V = fingerprint(fwd, omega_el, x_dyn, machine.curr_max, machine.volt_max)
            actives.append(active_set(alpha_I, alpha_V, n_I, n_V))
        else:
            actives.append(None)

    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out_dir, exist_ok=True)

    # Detect active-set changes between consecutive T* and save before/after.
    crossings: list[tuple[int, int]] = []
    for i in range(1, n_sweep):
        a_prev, a_curr = actives[i - 1], actives[i]
        if a_prev is None or a_curr is None:
            continue
        if a_prev != a_curr:
            crossings.append((i - 1, i))

    print(
        f"Sweep done: omega_el={omega_el:.3f} rad/s (n_mech={n_mech_rpm} RPM); torq_max_local={torq_max_local:.3f} Nm"
    )
    print("active sets along sweep:")
    for T, a in zip(torq_targets, actives, strict=True):
        print(f"  T*={T:.2f}Nm  ->  {active_set_tag(a) if a is not None else 'FAIL'}")
    if not crossings:
        print(
            "\nNo active-set changes detected. Widen torq_targets, raise n_sweep, or move omega to provoke a crossing."
        )
    else:
        print(f"\nActive-set changes at indices: {crossings}")

    saved: list[str] = []
    for i_before, i_after in crossings:
        for which, idx in (("before", i_before), ("after", i_after)):
            T = torq_targets[idx]
            tag_active = active_set_tag(actives[idx])
            tag = f"T{T:0.2f}_{tag_active}_{which}"
            title = f"omega={n_mech_rpm:.0f}rpm, T*={T:.2f}Nm, active={tag_active} ({which} crossing)"

            fig, _ = plot_dq_phase_combined(
                fwd,
                omega_el,
                dyn_dq[idx],
                curr_dq_static=static_dq[idx] if static_ok[idx] else None,
                curr_max=machine.curr_max,
                volt_max=machine.volt_max,
                title=title,
            )
            out_path = os.path.join(out_dir, f"lowspeed_{tag}.png")
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)
            print(f"  saved {out_path}")

    print(f"\nDone. {len(saved)} figure(s) written to {out_dir}.")


if __name__ == "__main__":
    main()
