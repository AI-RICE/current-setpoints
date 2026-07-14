"""
DIAGNOSTIC -- is the high-speed Dynamic-arm collapse an initialization
failure? Compare es.solve_fault (Dynamic arm) from the default cold start
vs a warm start projected from Fall's converged 2-DOF solution.

For each speed we sweep target torque T and, for each init, record whether
the Dynamic arm finds a converged + feasible point. If the Fall-warm start
sustains feasibility where the cold start collapses to 0, the E17 envelope
hole is an init artifact, not physics.

Run:  python experiments/fourier/diag_solver_init.py
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
from dynamic.fourier_math import fourier_design, n_basis  # noqa: E402
from experiments.fourier import envelope_solver as es  # noqa: E402
from experiments.fourier.e17_fault_envelope import (  # noqa: E402
    fall_max_torque,
    fall_phase_currents_1f,
    phase_to_amp_inv_dq,
)
from experiments.fourier.phase_voltage import voltage_linear_maps  # noqa: E402


def dynamic_feasible(drive, omega, T, maps, Imax, Vmax, rms_max, warm):
    """Run the Dynamic arm at torque T from a given warm start; report status."""
    C, conv, maxI, maxV, rms = es.solve_fault(
        drive, omega, T, es.H_FREE, maps, Imax, Vmax,
        use_voltage=True, rms_max=rms_max, warm=warm,
    )
    ok = conv and maxI <= Imax * 1.01 and maxV <= Vmax * 1.01 and rms <= rms_max * 1.01
    return ok, conv, maxI, maxV, rms, C


def main() -> None:
    machine = ieee_machine2(curr_max=30.0, volt_max=13.0, omega_max=1800)
    fwd = ForwardModel(machine, add_volt_0=False, n_theta=700)
    Imax, Vmax = machine.curr_max, machine.volt_max
    rms_max = Imax / np.sqrt(2.0)

    Hf_all = es.fault_phase_map(fwd.vec_theta, (0,))
    n_t = fwd.vec_theta.size - 1
    idx_s = (np.round(np.linspace(0, 2 * np.pi, es.N_CON, endpoint=False) / (2 * np.pi) * n_t).astype(int)) % n_t
    theta_s = fwd.vec_theta[idx_s]
    nb = n_basis(es.H_FREE)

    for rpm in [700, 900, 1100, 1300]:
        omega = rpm * (np.pi / 30.0) * machine.n_ppairs
        Hf_s = Hf_all[idx_s]
        gU_p, gL_p, bV_p = voltage_linear_maps(fwd, omega, idx_s, phases=(1, 2, 3, 4))
        gU_s = gU_p.transpose(1, 0, 2)
        gL_s = gL_p.transpose(1, 0, 2)
        bemf_ph_s = bV_p.T
        Phi, dPhi = fourier_design(theta_s, es.H_FREE)
        maps = (Hf_s, gU_s, gL_s, bemf_ph_s, Phi, dPhi)

        # Fall optimum + project its dq trajectory onto the H_FREE Fourier basis
        T_fall, fall_d = fall_max_torque(machine, fwd, omega, Imax, Vmax)
        i_phase = fall_phase_currents_1f(fall_d["i_d1_F"], fall_d["i_q1_F"], theta_s)
        i_dq = phase_to_amp_inv_dq(i_phase, theta_s)  # (4, n_con)
        C_warm, *_ = np.linalg.lstsq(Phi, i_dq.T, rcond=None)  # (nb, dim)
        recon = float(np.sqrt(np.mean((Phi @ C_warm - i_dq.T) ** 2)))

        print(f"\n=== rpm={rpm}  Fall T_mean={T_fall:.2f} Nm (ripple {fall_d['T_ripple']:.2f})  "
              f"Fall->Fourier reconstruction rms err={recon:.3e} ===")
        print(f"{'T*':>5} | {'cold: conv  maxI  maxV   rms  OK':>34} | {'warm(Fall): conv  maxI  maxV   rms  OK':>40}")
        max_cold = max_warm = 0.0
        for T in np.arange(0.5, 5.51, 0.25):
            okc, cc, ic, vc, rc, _ = dynamic_feasible(machine, omega, T, maps, Imax, Vmax, rms_max, None)
            okw, cw, iw, vw, rw, _ = dynamic_feasible(machine, omega, T, maps, Imax, Vmax, rms_max, C_warm)
            if okc:
                max_cold = T
            if okw:
                max_warm = T
            print(f"{T:5.2f} | {str(cc):>5} {ic:5.1f} {vc:5.2f} {rc:5.1f}  {'OK' if okc else '..':>2}"
                  f" | {str(cw):>5} {iw:5.1f} {vw:5.2f} {rw:5.1f}  {'OK' if okw else '..':>2}")
        print(f"  --> max feasible T:  cold={max_cold:.2f}  warm(Fall)={max_warm:.2f}  (Fall mean={T_fall:.2f})")


if __name__ == "__main__":
    main()
