"""
DIAGNOSTIC 3 -- evaluate Fall's exact reconstructed trajectory against the
Dynamic arm's OWN constraint functions. If Fall satisfies peak-I, peak-V,
rms and ripple in the es formulation but the bisection still returns 0,
the failure is the solver/feasibility-search, not the constraints. If Fall
violates one (rms is the prime suspect), that constraint is the real gap.

Run:  python experiments/fourier/diag_fall_feas.py
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
from dynamic.fourier_optimizer import extract_quadratic_torque, fourier_design  # noqa: E402
from experiments.fourier import envelope_solver as es  # noqa: E402
from experiments.fourier.e17_fault_envelope import (  # noqa: E402
    fall_max_torque,
    fall_phase_currents_1f,
    phase_to_amp_inv_dq,
)


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
    Phi, dPhi = fourier_design(theta_s, es.H_FREE)

    print(f"Imax={Imax}  Vmax={Vmax}  rms_max={rms_max:.2f}")
    for rpm in [500, 900, 1100, 1300]:
        omega = rpm * (np.pi / 30.0) * machine.n_ppairs
        transform._set_omega(omega)
        U = transform.mat_curr_dq_to_volt_dq
        L = transform.machine.L_stat
        flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
        bemf_dq = omega * machine.mat_crossc @ flux_volt
        Hf_s = Hf_all[idx_s]
        bemf_ph_s = np.einsum("skj,j->sk", Hf_s, bemf_dq)

        T_fall, fall_d = fall_max_torque(model, transform, omega, Imax, Vmax)
        i_phase = fall_phase_currents_1f(fall_d["i_d1_F"], fall_d["i_q1_F"], theta_s)
        i_dq = phase_to_amp_inv_dq(i_phase, theta_s)  # (4, n_con)
        C_warm, *_ = np.linalg.lstsq(Phi, i_dq.T, rcond=None)

        # es-formulation evaluation of Fall's reconstructed trajectory
        i_s = Phi @ C_warm            # (n_con, dim)
        di_s = dPhi @ C_warm
        i_ph = np.einsum("skj,sj->sk", Hf_s, i_s)       # (n_con, n_surv)
        HU = np.einsum("skj,jl->skl", Hf_s, U)
        HL = np.einsum("skj,jl->skl", Hf_s, L)
        v_ph = np.einsum("skl,sl->sk", HU, i_s) + omega * np.einsum("skl,sl->sk", HL, di_s) + bemf_ph_s
        A_t, b_t, c_t = extract_quadratic_torque(model, omega, dim)
        tq = np.einsum("sj,jk,sk->s", i_s, A_t, i_s) + i_s @ b_t + c_t

        maxI = np.max(np.abs(i_ph))
        maxV = np.max(np.abs(v_ph))
        rms = np.max(np.sqrt(np.mean(i_ph**2, axis=0)))
        tmean = float(np.mean(tq))
        trip = float(np.max(np.abs(tq - tmean)))
        flags = []
        if maxI > Imax * 1.01:
            flags.append("PEAK-I")
        if maxV > Vmax * 1.01:
            flags.append("PEAK-V")
        if rms > rms_max * 1.01:
            flags.append("RMS")
        verdict = "FEASIBLE in es-arm" if not flags else "VIOLATES " + ",".join(flags)
        print(f"\nrpm={rpm}: Fall T_mean={T_fall:.2f}  es-eval of Fall traj: "
              f"maxI={maxI:.2f}/{Imax}  maxV={maxV:.2f}/{Vmax}  "
              f"rms={rms:.2f}/{rms_max:.2f}  Tmean={tmean:.2f} ripple={trip:.2f}  -> {verdict}")


if __name__ == "__main__":
    main()
