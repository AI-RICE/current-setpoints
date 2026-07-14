"""
E13 -- Fourier ansatz vs full-grid optimizer, head-to-head.

Question (from the dynamic-paper discussion): how much of the full
position-dependent (grid) optimizer's Joule advantage over the prior-art
low-order Fourier ansatz is *real*, and in which operating regime?

We run both solvers at two clean cells:

  * FW (voltage-limited): T*=4 Nm, 1400 rpm -- the chatter reference
    point. Binding limit is the per-phase *voltage*, which contains the
    inductive term omega*L*di/dtheta.
  * current-limited:      T*=7 Nm,  200 rpm -- binding limit is the
    per-phase *current* h_k.i, which has NO derivative.

For the Fourier method we sweep the set of dq angular harmonics
``H = {0} -> {0,9,11} -> {0,9,11,19,21} -> {0..7}``. ``H={0}`` is the
static / prior-art ansatz (constant dq = 1st+3rd phase harmonics);
``{9,11,...}`` are the 10k+/-1 harmonics the chatter analysis (E01)
found in the grid optimum.

The decisive diagnostic is the **voltage of the grid solution under the
exact (spectral) derivative** vs under the finite difference the grid
solver uses. If the grid rides V_max under the finite difference but
exceeds it under the exact derivative, its extra Joule reduction is a
discretization artifact, and the Fourier method (which uses the exact
analytic derivative) is right not to reproduce it.

Run:
    python experiments/fourier/e13_headtohead.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from current_setpoints.models.forward_model import ForwardModel  # noqa: E402
from current_setpoints.models.machines import ieee_machine2  # noqa: E402
from current_setpoints.optimization import ActiveSetOptimizer, FourierOptimizer  # noqa: E402
from dynamic._waveforms import evaluate_dq_on_grid  # noqa: E402
from dynamic.voltage_diagnostics import voltage_residuals_dense  # noqa: E402

# Harmonic sets to sweep (each must include the DC term 0).
HARMONIC_SETS: list[tuple[int, ...]] = [
    (0,),
    (0, 9, 11),
    (0, 9, 11, 19, 21),
    (0, 1, 2, 3, 4, 5, 6, 7),
]

CELLS = [
    # (rpm, T*, label)
    (1400.0, 4.0, "FW (voltage-limited)"),
    (200.0, 7.0, "current-limited"),
]


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def joule_dense(fwd: ForwardModel, theta_grid: np.ndarray, X: np.ndarray) -> float:
    """Period-mean Joule loss on the dense vec_theta grid (common metric)."""
    Xd = evaluate_dq_on_grid(theta_grid, X, fwd.vec_theta[:-1])
    return float(np.mean(np.sum(Xd**2, axis=1)))


def spectral_voltage_residual(fwd: ForwardModel, omega: float, X_grid: np.ndarray, volt_max: float) -> float:
    """
    Worst per-phase voltage residual of a grid trajectory under the *exact*
    (spectral) derivative of its band-limited interpolation.

    The grid solver evaluates the inductive term omega*L*di/dtheta with a
    finite difference; here we instead take the trajectory's DFT, multiply
    mode h by i*h to differentiate exactly, and rebuild the phase voltage.
    A positive return means the grid solution violates V_max once the
    inductive term is computed faithfully.
    """
    drive = fwd.drive
    dim = drive.dim
    n_grid, _ = X_grid.shape
    # spectral derivative on the coarse grid (band-limited interpolation)
    Xf = np.fft.fft(X_grid, axis=0)
    h = np.fft.fftfreq(n_grid, d=1.0 / n_grid)  # integer mode numbers
    dX = np.real(np.fft.ifft(1j * h[:, None] * Xf, axis=0))  # d/dtheta on [0,2pi)

    zeros = np.zeros(dim)
    U = drive.voltage_operator(omega, zeros)
    L = drive.inductance(omega, zeros)
    bemf_dq = drive.bemf_dq(omega, zeros)
    v_dq = X_grid @ U.T + omega * dX @ L.T + bemf_dq[None, :]  # (n_grid, dim)

    n_theta = fwd.vec_theta.size - 1
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    idx = (np.round(theta_grid / (2 * np.pi) * n_theta).astype(int)) % n_theta
    Hb = fwd._mat_dq_to_ph_all[idx].transpose(1, 0, 2)  # (n_phases, n_grid, dim)
    v_ph = np.einsum("knj,nj->kn", Hb, v_dq)
    return float(np.max(np.abs(v_ph)) - volt_max)


def main() -> None:
    machine = ieee_machine2(curr_max=30.0, volt_max=13.0, omega_max=1800)
    fwd = ForwardModel(machine, add_volt_0=False, n_theta=700)
    Imax, Vmax = machine.curr_max, machine.volt_max

    n_grid = 128
    n_con = 144
    lines: list[str] = []

    fig, axes = plt.subplots(1, len(CELLS), figsize=(13, 5.0), squeeze=False)

    for col, (rpm, T, label) in enumerate(CELLS):
        omega = rpm * (np.pi / 30.0) * machine.n_ppairs
        header = f"\n=== {label}: T*={T} Nm, {rpm:.0f} rpm (omega={omega:.1f} rad/s) ==="
        print(header)
        lines.append(header)

        active_opt = ActiveSetOptimizer(fwd, n_grid=n_grid)
        sol_g = active_opt.minimize_current(T, omega)
        theta_grid_g = sol_g.diagnostics["theta_grid"]
        Jg = joule_dense(fwd, theta_grid_g, sol_g.curr_dq)
        rv_fd, _ = voltage_residuals_dense(fwd, omega, sol_g.curr_dq, Vmax)
        rv_exact = spectral_voltage_residual(fwd, omega, sol_g.curr_dq, Vmax)
        msg = (
            f"GRID(N={n_grid}): J={Jg:.3f}  conv={sol_g.success}  "
            f"V-residual finite-diff={rv_fd:+.3f} V   exact-derivative={rv_exact:+.3f} V"
        )
        print(msg)
        lines.append(msg)

        Js: list[float] = []
        tags: list[str] = []
        warm = None
        J0 = None
        for H in HARMONIC_SETS:
            fourier_opt = FourierOptimizer(fwd, n_grid=n_con, harmonics=H)
            maps = fourier_opt._precompute_maps(omega)
            fr = fourier_opt._solve(omega, maps, T, warm_coeffs=warm)
            Jf = joule_dense(fwd, fr["theta_out"], fr["curr_dq_grid"])
            if H == (0,):
                J0 = Jf
            closed = 100.0 * (J0 - Jf) / (J0 - Jg) if (J0 is not None and (J0 - Jg) > 1e-9) else float("nan")
            gap = 100.0 * (Jf - Jg) / Jg
            row = (
                f"  Fourier H={str(H):22s} nb*dim={fr['C'].size:3d}  J={Jf:8.3f}  "
                f"gap_vs_grid={gap:+.2f}%  gap_closed={closed:5.1f}%  ripple={fr['torque_ripple']:.1e} Nm  conv={fr['success']}"
            )
            print(row)
            lines.append(row)
            Js.append(Jf)
            tags.append("{" + ",".join(map(str, H)) + "}")
            if H == (0, 9, 11):
                warm = fr["C"]  # warm-start richer sets from the {0,9,11} solution

        ax = axes[0][col]
        x = np.arange(len(Js))
        ax.plot(x, Js, "o-", color="C0", label="Fourier ansatz")
        ax.axhline(Jg, color="C3", ls="--", lw=2, label=f"full grid (N={n_grid})")
        ax.axhline(J0, color="C7", ls=":", lw=1.5, label="static / prior art {0}")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("period-mean Joule loss")
        ax.set_title(f"{label}\nT*={T} Nm, {rpm:.0f} rpm\nGRID V-resid: FD {rv_fd:+.2f} V / exact {rv_exact:+.2f} V")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("E13: Fourier ansatz vs full-grid optimizer -- Joule loss by dq-harmonic set")
    fig.tight_layout()
    out = os.path.join(_figs_dir(), "e13_headtohead.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")

    summary = os.path.join(_figs_dir(), "e13_summary.txt")
    with open(summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {summary}")


if __name__ == "__main__":
    main()
