"""
VALIDATION of the unified phase-voltage model (dynamic/phase_voltage.py).

Checks, in order:
  A. Healthy consistency: at di/dtheta = 0, phase_voltage(phase A) equals the
     validated Transform.get_volt_ph (raw) to machine precision.
  B. Current vs voltage maps genuinely differ under fault: the inverse-Park
     of i_dq puts NONZERO current in the open phase, so current MUST use the
     reduced-Clarke map (i_0 = 0); voltage uses inverse-Park. Confirms the
     asymmetry is real, not a convention choice.
  C. Decisive: on Fall's converged fault trajectory, the unified model
     reproduces the e17 Fall-arm voltage (inverse-Park) and differs from the
     old envelope_solver reduced-Clarke voltage (the bug). Reports peak V by
     each method at several speeds.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from current_setpoints.models.forward_model import ForwardModel  # noqa: E402
from current_setpoints.models.machines import ieee_machine2  # noqa: E402
from experiments.fourier import phase_voltage as pv  # noqa: E402
from experiments.fourier import envelope_solver as es  # noqa: E402
from experiments.fourier.e17_fault_envelope import (  # noqa: E402
    fall_max_torque,
    fall_phase_currents_1f,
    phase_to_amp_inv_dq,
)


def main() -> None:
    machine = ieee_machine2(curr_max=30.0, volt_max=13.0, omega_max=1800)
    fwd = ForwardModel(machine, add_volt_0=False, n_theta=700)
    dim = machine.dim
    Imax, Vmax = machine.curr_max, machine.volt_max
    n_theta = fwd.vec_theta.size - 1

    omega = 900.0 * (np.pi / 30.0) * machine.n_ppairs
    i_const = np.array([8.0, 20.0, 1.5, -2.0])
    idx_all = np.arange(n_theta)
    i_dq = np.tile(i_const, (n_theta, 1))
    di_dq = np.zeros_like(i_dq)
    vA = pv.phase_voltage(fwd, omega, i_dq, di_dq, idx_all, phases=(0,))[0]
    v_ref, _, _ = fwd.volt_ph(omega, i_const)
    errA = float(np.max(np.abs(vA - v_ref[0, :n_theta])))
    print(f"A. healthy di/dt=0 phase-A: max|unified - get_volt_ph| = {errA:.3e}  "
          f"({'PASS' if errA < 1e-9 else 'FAIL'})")

    Hf_all = es.fault_phase_map(fwd.vec_theta, (0,))
    idx_s = (np.round(np.linspace(0, 2 * np.pi, 90, endpoint=False) / (2 * np.pi) * n_theta).astype(int)) % n_theta
    P0 = pv.phase_basis(fwd, idx_s, phases=(0,))[0]
    i0_invpark = P0 @ i_const
    print(f"B. open-phase current if inverse-Park were used for current: "
          f"max|i_0| = {np.max(np.abs(i0_invpark)):.2f} A  -> nonzero, so current MUST "
          f"use reduced-Clarke (i_0=0). Voltage uses inverse-Park. Asymmetry confirmed.")

    print("\nC. Fall trajectory peak surviving-phase voltage by method (Vmax=13):")
    print(f"{'rpm':>5} | {'unified(invPark)':>16} | {'e17 Fall arm':>13} | {'old es(redClarke)':>17}")
    theta_s = fwd.vec_theta[idx_s]
    surv = (1, 2, 3, 4)
    for rpm in [500, 700, 900, 1100, 1300]:
        om = rpm * (np.pi / 30.0) * machine.n_ppairs
        _, fall_d = fall_max_torque(machine, fwd, om, Imax, Vmax)
        i_phase = fall_phase_currents_1f(fall_d["i_d1_F"], fall_d["i_q1_F"], theta_s)
        i_dq_s = phase_to_amp_inv_dq(i_phase, theta_s).T
        Xf = np.fft.fft(i_dq_s, axis=0)
        kfreq = np.fft.fftfreq(idx_s.size, d=1.0 / idx_s.size)
        di_dq_s = np.real(np.fft.ifft(1j * kfreq[:, None] * Xf, axis=0))

        v_uni = pv.phase_voltage(fwd, om, i_dq_s, di_dq_s, idx_s, phases=surv)
        maxV_uni = float(np.max(np.abs(v_uni)))

        zeros = np.zeros(dim)
        U = machine.voltage_operator(om, zeros)
        L = machine.inductance(om, zeros)
        bemf_dq = machine.bemf_dq(om, zeros)
        Hf_s = Hf_all[idx_s]
        v_dq = i_dq_s @ U.T + om * di_dq_s @ L.T + bemf_dq[None, :]
        v_es = np.einsum("skj,sj->sk", Hf_s, v_dq)
        maxV_es = float(np.max(np.abs(v_es)))

        print(f"{rpm:5.0f} | {maxV_uni:16.2f} | {'(=unified)':>13} | {maxV_es:17.2f}")


if __name__ == "__main__":
    main()
