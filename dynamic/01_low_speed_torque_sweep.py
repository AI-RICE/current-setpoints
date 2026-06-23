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

    # Low speed: 50 RPM mechanical. Convert to electrical rad/s.
    n_mech_rpm = 50.0
    omega_el = n_mech_rpm * (np.pi / 30.0) * machine.n_ppairs

    # Local torque ceiling at this speed.
    _, torq_max_local, ok_top = optimizer.maximize_torque(omega=omega_el, transform=transform)
    if not ok_top:
        raise RuntimeError(f"maximize_torque failed at omega_el={omega_el:.3f}")

    n_sweep = 25
    torq_targets = np.linspace(0.5, 0.98 * torq_max_local, n_sweep)
    n_grid = 64

    static_dq: list[np.ndarray] = []
    static_ok: list[bool] = []
    dyn_dq: list[np.ndarray] = []
    dyn_ok: list[np.ndarray] = []
    actives: list[frozenset[tuple[str, int]] | None] = []

    curr_dq_prev = np.array([1.0, 0.0, 0.0, 0.0])
    for T in torq_targets:
        x_static, ok_s = optimizer.minimize_current(
            torq_target=float(T), omega=omega_el, transform=transform, vec_curr_guess=curr_dq_prev
        )
        if ok_s:
            curr_dq_prev = x_static
        static_dq.append(x_static)
        static_ok.append(bool(ok_s))

        _, x_dyn, ok_dyn = run_independent_per_angle(
            model=analytical,
            transform=transform,
            omega=omega_el,
            torq_target=float(T),
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
                transform,
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
