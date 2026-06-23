"""
DIAGNOSTIC 2 -- is the high-speed Dynamic-arm collapse caused by the
ripple-free torque constraint (ripple_budget=1e-2), not initialization?

Fall reaches its torque WITH ripple (0.2-0.34 Nm). The Dynamic arm was
held ripple-free. Re-sweep the Dynamic arm's max feasible torque at three
ripple budgets and compare to Fall. If relaxing the budget restores
T_dyn >= T_Fall, the nesting breaks at the CONSTRAINT level, not the
representation/solver level.

Run:  python experiments/fourier/diag_ripple.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from current_setpoints.optimization import ModelAnalytical  # noqa: E402
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2  # noqa: E402
from current_setpoints.simulation import Transform  # noqa: E402
from dynamic.fourier_optimizer import fourier_design  # noqa: E402
from experiments.fourier import envelope_solver as es  # noqa: E402
from experiments.fourier.e17_fault_envelope import fall_max_torque  # noqa: E402


def max_T_dyn(model, transform, omega, maps, Imax, Vmax, rms_max, ripple, T_hi=5.6):
    """Bisect max feasible ripple-bounded torque for the Dynamic arm at a given ripple budget."""
    lo, hi, warm = 0.0, T_hi, None
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        C, conv, maxI, maxV, rms = es.solve_fault(
            model, transform, omega, mid, es.H_FREE, maps, Imax, Vmax,
            use_voltage=True, rms_max=rms_max, ripple_budget=ripple, warm=warm,
        )
        ok = conv and maxI <= Imax * 1.01 and maxV <= Vmax * 1.01 and rms <= rms_max * 1.01
        if ok:
            lo, warm = mid, C
        else:
            hi = mid
    return lo


def main() -> None:
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    transform = Transform(machine=machine, flux=Flux_IEEEMachine2(), add_volt_0=False, n_theta=700)
    model = ModelAnalytical(machine=machine, flux=Flux_IEEEMachine2())
    Imax, Vmax = machine.curr_max, machine.volt_max
    rms_max = Imax / np.sqrt(2.0)
    dim = transform.dim

    Hf_all = es.fault_phase_map(transform.vec_theta, (0,))
    n_t = transform.vec_theta.size - 1
    idx_s = (np.round(np.linspace(0, 2 * np.pi, es.N_CON, endpoint=False) / (2 * np.pi) * n_t).astype(int)) % n_t
    theta_s = transform.vec_theta[idx_s]

    budgets = [1e-2, 0.25, 1.0]
    print(f"{'rpm':>5} | {'Fall(rip)':>12} | " + " | ".join(f"Dyn rip<={b:g}" for b in budgets))
    for rpm in [500, 700, 900, 1100, 1300]:
        omega = rpm * (np.pi / 30.0) * machine.n_ppairs
        transform._set_omega(omega)
        U = transform.mat_curr_dq_to_volt_dq
        flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
        bemf_dq = omega * machine.mat_crossc @ flux_volt
        Hf_s = Hf_all[idx_s]
        Gf_s = np.einsum("skj,jl->skl", Hf_s, U)
        bemf_ph_s = np.einsum("skj,j->sk", Hf_s, bemf_dq)
        Phi, dPhi = fourier_design(theta_s, es.H_FREE)
        maps = (Hf_s, Gf_s, bemf_ph_s, Phi, dPhi)

        T_fall, fall_d = fall_max_torque(model, transform, omega, Imax, Vmax)
        dyn = [max_T_dyn(model, transform, omega, maps, Imax, Vmax, rms_max, b) for b in budgets]
        print(f"{rpm:5.0f} | {T_fall:6.2f}({fall_d['T_ripple']:.2f}) | "
              + " | ".join(f"{d:9.2f}" for d in dyn))


if __name__ == "__main__":
    main()
