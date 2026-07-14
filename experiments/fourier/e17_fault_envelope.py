"""
E17 -- Single open-phase fault, realisable torque-speed envelope on
IEEEMachine2, **with a faithful Fall (2016) baseline** (Eqs 16-19,
verified against Fall's Fig 3a in baselines/Fall/).

Arms compared:

  Dynamic   - free dq shaping (Fourier {0..10}) + per-phase current peak
              + per-phase voltage incl. EXACT di/dtheta (this work).
  Yepes     - free dq shaping (Fourier {0..10}) + per-phase peak + per-phase
              rms (no voltage, matching Yepes 2024 explicitly).
  Static    - constant dq (Fourier {0}) + per-phase peak + voltage (the
              static-paper formulation: 4-DOF in (i_d1,i_q1,i_d3,i_q3),
              no equal-magnitude criterion).
  Fall      - 2-DOF closed-form reconfiguration (Eqs 16-19, validated to
              Fig 3a) applied here to IEEEMachine2: surviving currents
              are pure sinusoids parameterised by (i_d1_F, i_q1_F)
              constants in Fall's power-invariant Park; torque and
              voltage computed with IEEEMachine2's full quadratic model
              (3rd-harm back-EMF + saliency). Per-phase current and
              voltage(di/dt) limits as in Fall's Eqs 21/22-25.

Honest caveat for Fall on IEEEMachine2: Fall's Eqs 16-19 assume
sinusoidal back-EMF and no saliency. IEEEMachine2 has both. Applying
Fall's *reconfiguration recipe* to this machine produces some torque
ripple (from the 3rd-harm back-EMF interacting with his pure-fundamental
phase currents). We report mean torque + ripple amplitude, optimise
mean torque under peak constraints.

Run:
    python experiments/fourier/e17_fault_envelope.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from current_setpoints.models.forward_model import ForwardModel  # noqa: E402
from current_setpoints.models.machines import PMSM5Phase  # noqa: E402

# Dynamic/Yepes/Static arms come from the shared Fourier-on-fault-map solver
from experiments.fourier import envelope_solver as es  # noqa: E402
from experiments.fourier.phase_voltage import voltage_linear_maps  # noqa: E402

# -- Fall constants (5-phase) -------------------------------------------
D1_FALL = np.cos(2 * np.pi / 5) - np.cos(4 * np.pi / 5)  # ~1.118
D2_FALL = np.sin(2 * np.pi / 5) + np.sin(4 * np.pi / 5)  # ~1.539
SQRT58 = np.sqrt(5.0 / 8.0)


def fall_phase_currents_1f(i_d1_F: float, i_q1_F: float, theta: np.ndarray) -> np.ndarray:
    """
    Fall Eqs (16)-(19): surviving phase currents for phase-a open,
    parameterised by Fall's constants (i_d1_F, i_q1_F) [power-invariant
    Park]. Returns 5xT array; row 0 = open phase a = 0.
    """
    N1 = i_d1_F * np.cos(theta) - i_q1_F * np.sin(theta)
    N2 = i_d1_F * np.sin(theta) + i_q1_F * np.cos(theta)
    A = N1 / D1_FALL
    B = N2 / D2_FALL
    i_b = SQRT58 * (+A + B)
    i_c = SQRT58 * (-A + B)
    i_d = SQRT58 * (-A - B)
    i_e = SQRT58 * (+A - B)
    return np.stack([np.zeros_like(theta), i_b, i_c, i_d, i_e], axis=0)  # (5, T)


def phase_to_amp_inv_dq(i_phase: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Project physical phase currents (5, T) to IEEEMachine2's amplitude-
    invariant dq frame (4, T): (i_d1, i_q1, i_d3, i_q3) at each theta.
    Uses C_full from envelope_solver (2/5-scaled Clarke) followed by rotation.
    """
    C = es.full_clarke_5phase()  # 5x5, rows: alpha1, beta1, alpha3, beta3, zero
    # i_alphabeta1_3_zero = C @ i_phase  → (5, T)
    i_alphabeta = C @ i_phase  # rows: alpha1, beta1, alpha3, beta3, zero
    # Rotate fundamental to dq1 at theta; 3rd-harmonic to dq3 at 3*theta
    c1, s1 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(3 * theta), np.sin(3 * theta)
    i_d1 = i_alphabeta[0] * c1 + i_alphabeta[1] * s1
    i_q1 = -i_alphabeta[0] * s1 + i_alphabeta[1] * c1
    i_d3 = i_alphabeta[2] * c3 + i_alphabeta[3] * s3
    i_q3 = -i_alphabeta[2] * s3 + i_alphabeta[3] * c3
    return np.stack([i_d1, i_q1, i_d3, i_q3], axis=0)  # (4, T)


def fall_arm_evaluate(i_d1_F: float, i_q1_F: float, theta: np.ndarray, drive, fwd: ForwardModel, omega: float):
    """
    Evaluate Fall's reconfiguration for PMSM5Phase at (i_d1_F, i_q1_F):
    returns (T_array, v_phase_array, i_phase_array). Phase currents from
    Eqs 16-19 (physical), torque + voltage from PMSM5Phase's full model
    on the amplitude-invariant dq projection. Voltage includes the EXACT
    di_dq/dtheta term via spectral derivative on the dense theta grid.
    """
    dim = drive.dim
    zeros = np.zeros(dim)
    U = drive.voltage_operator(omega, zeros)
    L = drive.inductance(omega, zeros)
    bemf_dq = drive.bemf_dq(omega, zeros)  # (4,)

    i_phase = fall_phase_currents_1f(i_d1_F, i_q1_F, theta)  # (5, T)
    i_dq = phase_to_amp_inv_dq(i_phase, theta)  # (4, T)

    # Spectral derivative of i_dq w.r.t. theta on dense grid (Fourier exact)
    n_t = theta.size
    Xf = np.fft.fft(i_dq, axis=1)
    k = np.fft.fftfreq(n_t, d=1.0 / n_t)
    di_dq = np.real(np.fft.ifft(1j * k[None, :] * Xf, axis=1))  # (4, T)

    # Torque at every theta via the analytic quadratic model
    T_per_theta = np.array([drive.torque(omega, i_dq[:, t]) for t in range(n_t)])

    # Phase voltage v_phase_k(theta) = h_k(theta) . (U.i + omega L di/dtheta + bemf)
    v_dq = U @ i_dq + omega * (L @ di_dq) + bemf_dq[:, None]  # (4, T)
    # h_k(theta) via the amp-inv dq->phase map (all 5 phases stored directly by ForwardModel)
    # Snap theta to fwd.vec_theta indices
    n_theta_tr = fwd.vec_theta.size - 1
    idx = (np.round(theta / (2 * np.pi) * n_theta_tr).astype(int)) % n_theta_tr
    H_amp = fwd._mat_dq_to_ph_all[idx].transpose(1, 0, 2)  # (5, T, dim): all 5 phases' dq->phase rows
    v_phase = np.einsum("ktj,jt->kt", H_amp, v_dq)  # (5, T)

    return T_per_theta, v_phase, i_phase


def fall_max_torque(
    drive,
    fwd: ForwardModel,
    omega: float,
    Imax: float,
    Vmax: float,
    use_voltage: bool = True,
    n_theta_dense: int = 360,
    x0: tuple[float, float] = (0.0, 30.0),
) -> tuple[float, dict]:
    """
    Faithful Fall arm: 2-DOF SLSQP over (i_d1_F, i_q1_F) maximising the
    period-MEAN PMSM5Phase torque under Fall's reconfiguration, subject
    to per-phase peak current and (if use_voltage) per-phase peak voltage
    (incl. exact di/dtheta) over the surviving phases. Returns
    (T_max, diagnostics dict).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta_dense, endpoint=False)

    def eval_for_x(x):
        Tarr, varr, iarr = fall_arm_evaluate(x[0], x[1], theta, drive, fwd, omega)
        return Tarr, varr, iarr

    def neg_Tmean(x):
        Tarr, _, _ = eval_for_x(x)
        return -float(np.mean(Tarr))

    def current_margin(x):
        _, _, iarr = eval_for_x(x)
        return Imax - float(np.max(np.abs(iarr[1:, :])))  # surviving phases only (skip phase 0)

    def voltage_margin(x):
        _, varr, _ = eval_for_x(x)
        return Vmax - float(np.max(np.abs(varr[1:, :])))

    cons = [{"type": "ineq", "fun": current_margin}]
    if use_voltage:
        cons.append({"type": "ineq", "fun": voltage_margin})

    res = minimize(
        neg_Tmean,
        np.array(x0),
        method="SLSQP",
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    # Diagnostics at optimum
    Tarr, varr, iarr = eval_for_x(res.x)
    maxI = float(np.max(np.abs(iarr[1:, :])))
    maxV = float(np.max(np.abs(varr[1:, :])))
    T_mean = float(np.mean(Tarr))
    T_ripple = float(np.max(np.abs(Tarr - T_mean)))
    return T_mean, {
        "converged": bool(res.success),
        "i_d1_F": float(res.x[0]),
        "i_q1_F": float(res.x[1]),
        "maxI_surviving": maxI,
        "maxV_surviving": maxV,
        "T_mean": T_mean,
        "T_ripple": T_ripple,
    }


def main() -> None:
    machine = PMSM5Phase()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    fwd = ForwardModel(machine, add_volt_0=False, n_theta=700)
    Imax, Vmax = machine.curr_max, machine.volt_max
    rms_max = Imax / np.sqrt(2.0)

    # Set up the maps once (depends on omega, but the open-phase fault map
    # is omega-independent for the Fourier arms; we recompute Gf, bemf per
    # omega inside the loop, same as envelope_solver).
    Hf_all = es.fault_phase_map(fwd.vec_theta, (0,))
    n_t = fwd.vec_theta.size - 1
    idx_s = (np.round(np.linspace(0, 2 * np.pi, es.N_CON, endpoint=False) / (2 * np.pi) * n_t).astype(int)) % n_t
    theta_s = fwd.vec_theta[idx_s]

    rpm_list = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500])
    arms = ["Dynamic", "Yepes", "Static", "Fall"]
    env = {a: [] for a in arms}
    diag_fall = []

    from dynamic.fourier_math import fourier_design  # noqa: E402

    for rpm in rpm_list:
        omega = rpm * (np.pi / 30.0) * machine.n_ppairs
        Hf_s = Hf_all[idx_s]  # reduced-Clarke CURRENT map (surviving phases 1..4)
        # inverse-Park VOLTAGE linear maps for the surviving physical phases.
        # voltage_linear_maps returns (phase, n, dim); solver wants (n, phase, dim).
        gU_p, gL_p, bV_p = voltage_linear_maps(fwd, omega, idx_s, phases=(1, 2, 3, 4))
        gU = gU_p.transpose(1, 0, 2)
        gL = gL_p.transpose(1, 0, 2)
        bV = bV_p.T
        Phi, dPhi = fourier_design(theta_s, es.H_FREE)
        Phi0, dPhi0 = fourier_design(theta_s, (0,))
        maps_free = (Hf_s, gU, gL, bV, Phi, dPhi)
        maps_dc = (Hf_s, gU, gL, bV, Phi0, dPhi0)

        # Yepes / Dynamic / Static via envelope_solver's existing solver
        T_dyn = es.max_torque(machine, omega, "Dynamic", maps_free, Imax, Vmax, rms_max)
        T_yep = es.max_torque(machine, omega, "Yepes", maps_free, Imax, Vmax, rms_max)
        T_sta = es.max_torque(machine, omega, "Static", maps_dc, Imax, Vmax, rms_max)

        # Fall via the faithful arm
        T_fall, fall_d = fall_max_torque(machine, fwd, omega, Imax, Vmax)
        diag_fall.append((rpm, fall_d))

        env["Dynamic"].append(T_dyn)
        env["Yepes"].append(T_yep)
        env["Static"].append(T_sta)
        env["Fall"].append(T_fall)

        print(
            f"rpm={rpm:5.0f}  Dyn={T_dyn:5.2f}  Yep={T_yep:5.2f}  Sta={T_sta:5.2f}  "
            f"Fall={T_fall:5.2f} (ripple={fall_d['T_ripple']:.2f}Nm, "
            f"i_dq_F=({fall_d['i_d1_F']:.1f},{fall_d['i_q1_F']:.1f}))",
            flush=True,
        )

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 6))
    styles = {
        "Dynamic": ("C0", "o-", "Dynamic: free shape + di/dt-V (this work)"),
        "Yepes": ("C1", "s--", "Yepes: free shape, current-only (no V)"),
        "Static": ("C2", "v-.", "Static: const dq (4-DOF) + V (static paper)"),
        "Fall": ("C3", "^:", "Fall: Eqs 16-19 reconfiguration + V (verbatim)"),
    }
    for a in arms:
        c, st, lab = styles[a]
        ax.plot(rpm_list, env[a], st, color=c, lw=2, label=lab)
    ax.set_xlabel("mechanical speed [rpm]")
    ax.set_ylabel("max mean torque [Nm]")
    ax.set_title(
        "Single open-phase fault, IEEEMachine2: realisable torque-speed envelope\n"
        "(faithful Fall via baselines/Fall; supersedes envelope_solver)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "e17_fault_envelope_v2.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
