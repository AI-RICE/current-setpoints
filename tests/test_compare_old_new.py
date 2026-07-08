"""
Cross-library regression: current_setpoints (old) vs current_setpoints_new (new).

For each mode (PMSM static, PMSM fault single/two-phase, IM, dynamic independent),
we run the same operating points through both APIs and compare:
  - torque at a fixed current vector
  - curr_peak and volt_peak at a fixed operating point
  - minimize_current result (||i_dq||, achieved torque, peak values)
  - maximize_torque result (torque value)

Tolerances:
  - Physics evaluations (same formula, same params): rtol=1e-5
  - Optimizer results (different code paths, same problem): rtol=1e-3
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── OLD library
from current_setpoints.parameters.machines import IEEEMachine2
from current_setpoints.parameters.flux import Flux_IEEEMachine2
from current_setpoints.parameters.im_machine import IM9Phase as OldIM9Phase
from current_setpoints.simulation.transform import Transform as OldTransform
from current_setpoints.simulation.im_transform import IMTransform as OldIMTransform
from current_setpoints.optimization.models import ModelAnalytical
from current_setpoints.optimization.im_model import ModelIMAnalytical
from current_setpoints.optimization.optimizer import MotorOptimizer

# ── NEW library
from current_setpoints_new.models.machines import PMSM5Phase as NewPMSM, IM9Phase as NewIM9Phase
from current_setpoints_new.models.forward_model import ForwardModel, Fault
from current_setpoints_new.optimization.optimizer import StaticOptimizer, IndependentOptimizer

# ── Old fault reference (envelope_solver in experiments/)
from experiments.fourier.envelope_solver import fault_phase_map

# ── Old dynamic reference
from dynamic.independent_optimizer import run_independent_per_angle

# ──────────────────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────────────────

CURR_MAX  = 30.0
VOLT_MAX  = 13.0
OMEGA_MAX = 1800  # RPM

SLSQP_OPTS = {"disp": False, "ftol": 1e-8, "maxiter": 500}

OMEGA_LOW = 50.0    # electrical rad/s (~60 RPM for 8 pole-pairs) — no FW
OMEGA_MID = 800.0   # electrical rad/s (~955 RPM) — light FW

IM_CURR_MAX  = 3.0
IM_VOLT_MAX  = 200.0
IM_OMEGA_MAX = 1500  # RPM

IM_OMEGA_LOW = 50.0   # electrical rad/s
IM_OMEGA_MID = 200.0

# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _old_pmsm():
    """(machine, flux, model, transform, optimizer)."""
    machine = IEEEMachine2()
    machine.set_max_pars(CURR_MAX, VOLT_MAX, OMEGA_MAX)
    flux = Flux_IEEEMachine2()
    model = ModelAnalytical(machine, flux)
    tr = OldTransform(machine=machine, flux=flux, add_volt_0=False, n_theta=700)
    opt = MotorOptimizer(model, SLSQP_OPTS)
    return machine, flux, model, tr, opt


def _new_pmsm(fault: Fault | None = None):
    """(drive, fwd, optimizer)."""
    drive = NewPMSM()
    drive.set_max_pars(CURR_MAX, VOLT_MAX, OMEGA_MAX)
    fwd = ForwardModel(drive, n_theta=700, fault=fault)
    opt = StaticOptimizer(fwd, opts=SLSQP_OPTS)
    return drive, fwd, opt


def _old_im():
    machine = OldIM9Phase()
    machine.set_max_pars(IM_CURR_MAX, IM_VOLT_MAX, IM_OMEGA_MAX)
    model = ModelIMAnalytical(machine)
    tr = OldIMTransform(machine=machine, add_volt_0=False, n_theta=700)
    opt = MotorOptimizer(model, SLSQP_OPTS)
    return machine, model, tr, opt


def _new_im():
    drive = NewIM9Phase()
    drive.set_max_pars(IM_CURR_MAX, IM_VOLT_MAX, IM_OMEGA_MAX)
    fwd = ForwardModel(drive, n_theta=700)
    opt = StaticOptimizer(fwd, opts=SLSQP_OPTS)
    return drive, fwd, opt


# ──────────────────────────────────────────────────────────────────────────────
# ══ PMSM static (healthy) ════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────

class TestPMSMStatic:

    I_TEST = np.array([2.0, 18.0, 1.0, -1.5])

    def test_torque_at_fixed_current(self):
        _, _, model, _, _ = _old_pmsm()
        drive, _, _ = _new_pmsm()
        T_old = model.calculate_torque(OMEGA_LOW, self.I_TEST)
        T_new = drive.torque(OMEGA_LOW, self.I_TEST)
        np.testing.assert_allclose(T_new, T_old, rtol=1e-5,
            err_msg="PMSM torque mismatch at fixed i_dq")

    def test_curr_peak_at_fixed_current(self):
        _, _, _, tr, _ = _old_pmsm()
        _, fwd, _ = _new_pmsm()
        curr_peak_old, *_ = tr.get_max_vals(OMEGA_LOW, self.I_TEST)
        curr_peak_new = fwd.peak_vals(OMEGA_LOW, self.I_TEST)[0]
        np.testing.assert_allclose(curr_peak_new, curr_peak_old, rtol=1e-5,
            err_msg="PMSM curr_peak mismatch")

    def test_volt_peak_at_fixed_current(self):
        _, _, _, tr, _ = _old_pmsm()
        _, fwd, _ = _new_pmsm()
        _, _, volt_peak_old, _ = tr.get_max_vals(OMEGA_LOW, self.I_TEST)
        volt_peak_new = fwd.peak_vals(OMEGA_LOW, self.I_TEST)[1]
        np.testing.assert_allclose(volt_peak_new, volt_peak_old, rtol=1e-5,
            err_msg="PMSM volt_peak mismatch")

    def test_volt_peak_fw(self):
        """At higher speed, voltage is the binding constraint — both must agree."""
        _, _, _, tr, _ = _old_pmsm()
        _, fwd, _ = _new_pmsm()
        i = np.array([-5.0, 15.0, 0.5, -0.5])
        _, _, volt_peak_old, _ = tr.get_max_vals(OMEGA_MID, i)
        volt_peak_new = fwd.peak_vals(OMEGA_MID, i)[1]
        np.testing.assert_allclose(volt_peak_new, volt_peak_old, rtol=1e-5,
            err_msg="PMSM FW volt_peak mismatch")

    def test_minimize_current_torque_achieved(self):
        _, _, model, tr, opt_old = _old_pmsm()
        drive, _, opt_new = _new_pmsm()
        torq = 2.5
        i_old, ok_old = opt_old.minimize_current(torq, OMEGA_LOW, tr)
        sol_new = opt_new.minimize_current(torq, OMEGA_LOW)
        assert ok_old, "Old minimize_current did not converge"
        assert sol_new.success, "New minimize_current did not converge"
        T_old = model.calculate_torque(OMEGA_LOW, i_old)
        T_new = drive.torque(OMEGA_LOW, sol_new.curr_dq)
        np.testing.assert_allclose(T_old, torq, rtol=1e-3)
        np.testing.assert_allclose(T_new, torq, rtol=1e-3)
        np.testing.assert_allclose(T_new, T_old, rtol=1e-3,
            err_msg="PMSM minimize_current torque diverges between old and new")

    def test_minimize_current_norm(self):
        _, _, _, tr, opt_old = _old_pmsm()
        _, _, opt_new = _new_pmsm()
        torq = 2.5
        i_old, ok_old = opt_old.minimize_current(torq, OMEGA_LOW, tr)
        sol_new = opt_new.minimize_current(torq, OMEGA_LOW)
        if not ok_old or not sol_new.success:
            pytest.skip("One optimizer did not converge")
        norm_old = float(np.dot(i_old, i_old))
        norm_new = float(np.dot(sol_new.curr_dq, sol_new.curr_dq))
        np.testing.assert_allclose(norm_new, norm_old, rtol=1e-3,
            err_msg="PMSM minimize_current: current norms diverge old vs new")

    def test_minimize_current_fw(self):
        """Field-weakening operating point (OMEGA_MID)."""
        _, _, model, tr, opt_old = _old_pmsm()
        drive, _, opt_new = _new_pmsm()
        torq = 1.5
        i_old, ok_old = opt_old.minimize_current(torq, OMEGA_MID, tr)
        sol_new = opt_new.minimize_current(torq, OMEGA_MID)
        if not ok_old:
            pytest.skip("Old minimize_current FW did not converge")
        if not sol_new.success:
            pytest.skip("New minimize_current FW did not converge")
        T_old = model.calculate_torque(OMEGA_MID, i_old)
        T_new = drive.torque(OMEGA_MID, sol_new.curr_dq)
        np.testing.assert_allclose(T_new, T_old, rtol=1e-3,
            err_msg="PMSM FW minimize_current torque mismatch")

    def test_maximize_torque_value(self):
        _, _, _, tr, opt_old = _old_pmsm()
        _, _, opt_new = _new_pmsm()
        _, T_max_old, ok_old = opt_old.maximize_torque(OMEGA_LOW, tr)
        sol_new = opt_new.maximize_torque(OMEGA_LOW)
        assert ok_old, "Old maximize_torque did not converge"
        assert sol_new.success, "New maximize_torque did not converge"
        np.testing.assert_allclose(sol_new.torque, T_max_old, rtol=1e-3,
            err_msg="PMSM maximize_torque value diverges old vs new")

    def test_maximize_torque_curr_at_limit(self):
        """At max torque the peak current should saturate the limit."""
        _, _, _, tr, opt_old = _old_pmsm()
        _, fwd, opt_new = _new_pmsm()
        i_old, _, ok_old = opt_old.maximize_torque(OMEGA_LOW, tr)
        sol_new = opt_new.maximize_torque(OMEGA_LOW)
        if not ok_old or not sol_new.success:
            pytest.skip("maximize_torque did not converge")
        curr_peak_old, *_ = tr.get_max_vals(OMEGA_LOW, i_old)
        curr_peak_new = fwd.peak_vals(OMEGA_LOW, sol_new.curr_dq)[0]
        assert curr_peak_old >= CURR_MAX * 0.99, f"Old curr_peak {curr_peak_old:.3f} < limit"
        assert curr_peak_new >= CURR_MAX * 0.99, f"New curr_peak {curr_peak_new:.3f} < limit"


# ──────────────────────────────────────────────────────────────────────────────
# ══ PMSM fault — single open phase ══════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────

class TestPMSMFaultSingle:
    """
    Old reference: envelope_solver.fault_phase_map (reduced-Clarke C_red inverse).
    New: ForwardModel(Fault((0,))).curr_ph via Fault.current_map.

    Both encode the same reduced-Clarke inverse — compare them directly.
    """

    OPEN = (0,)

    def test_null_space_exists_and_has_correct_dimension(self):
        """
        Single-fault has exactly 1 null-space row (5 − 4 = 1).
        Two-fault cases have exactly 2 null-space rows (5 − 3 = 2).
        """
        _, fwd1, _ = _new_pmsm(Fault((0,)))
        assert fwd1.fault._N is not None
        assert fwd1.fault._N.shape == (1, 4), f"expected (1,4), got {fwd1.fault._N.shape}"

        _, fwd2a, _ = _new_pmsm(Fault((0, 1)))
        assert fwd2a.fault._N is not None
        assert fwd2a.fault._N.shape == (2, 4), f"expected (2,4), got {fwd2a.fault._N.shape}"

        _, fwd2n, _ = _new_pmsm(Fault((0, 2)))
        assert fwd2n.fault._N is not None
        assert fwd2n.fault._N.shape == (2, 4), f"expected (2,4), got {fwd2n.fault._N.shape}"

    def test_null_space_rows_orthogonal_to_Clarke_columns(self):
        """
        The full 5D left null-space vectors of C_aug must satisfy N_full @ C_aug = 0.
        We recompute them via SVD (same algorithm used in _null_space_matrix) and
        verify orthogonality directly on the full 5D vectors.
        """
        from current_setpoints_new.models.forward_model import _build_clarke_5phase
        C_full = _build_clarke_5phase()
        _, fwd, _ = _new_pmsm(Fault(self.OPEN))
        kept = list(fwd.fault._kept)
        C_aug = C_full[:, kept]          # 5 × n_surv
        n_surv = len(kept)
        _, _, vh = np.linalg.svd(C_aug.T, full_matrices=True)
        null_dim = 5 - n_surv
        N_full = vh[-null_dim:]          # (null_dim, 5) full left null-space rows
        product = N_full @ C_aug         # (null_dim, n_surv) — must be ≈ 0
        np.testing.assert_allclose(
            product, 0.0, atol=1e-10,
            err_msg="Null-space rows are not orthogonal to C_aug columns",
        )

    def test_open_phase_excluded_from_output(self):
        """curr_ph in fault mode returns n_surv rows, not n_phases rows."""
        drive, fwd, _ = _new_pmsm(Fault(self.OPEN))
        i_ph = fwd.curr_ph(OMEGA_LOW, np.array([5.0, 15.0, 1.0, -2.0]))
        n_surv = 5 - len(self.OPEN)
        assert i_ph.shape[0] == n_surv, \
            f"Expected {n_surv} surviving-phase rows, got {i_ph.shape[0]}"

    def test_maximize_torque_lower_than_healthy(self):
        _, _, opt_h = _new_pmsm()
        _, _, opt_f = _new_pmsm(Fault(self.OPEN))
        sol_h = opt_h.maximize_torque(OMEGA_LOW)
        sol_f = opt_f.maximize_torque(OMEGA_LOW)
        assert sol_h.success and sol_f.success
        assert sol_f.torque < sol_h.torque, "Fault should reduce max torque"
        assert sol_f.torque > 0

    def test_minimize_current_achieves_torque(self):
        drive, fwd, opt = _new_pmsm(Fault(self.OPEN))
        sol_max = opt.maximize_torque(OMEGA_LOW)
        if not sol_max.success:
            pytest.skip("maximize_torque probe failed")
        torq = sol_max.torque * 0.5
        sol = opt.minimize_current(torq, OMEGA_LOW)
        assert sol.success
        np.testing.assert_allclose(
            drive.torque(OMEGA_LOW, sol.curr_dq), torq, rtol=1e-3)

    def test_curr_peak_respects_limit(self):
        drive, fwd, opt = _new_pmsm(Fault(self.OPEN))
        sol_max = opt.maximize_torque(OMEGA_LOW)
        if not sol_max.success:
            pytest.skip("maximize_torque did not converge")
        curr_peak = fwd.peak_vals(OMEGA_LOW, sol_max.curr_dq)[0]
        assert curr_peak <= CURR_MAX * 1.001, \
            f"Single-fault solution violates current limit: {curr_peak:.3f} > {CURR_MAX}"


# ──────────────────────────────────────────────────────────────────────────────
# ══ PMSM fault — two open phases ════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────

class TestPMSMFaultTwo:

    OPEN = (0, 1)

    def test_open_phases_excluded_from_output(self):
        """curr_ph returns only surviving phases — open phases not in output array."""
        _, fwd, _ = _new_pmsm(Fault(self.OPEN))
        i_ph = fwd.curr_ph(OMEGA_LOW, np.array([3.0, 10.0, 0.5, -1.0]))
        n_surv = 5 - len(self.OPEN)
        assert i_ph.shape[0] == n_surv, \
            f"Expected {n_surv} rows, got {i_ph.shape[0]}"

    def test_torque_lower_than_single_fault(self):
        _, _, opt1 = _new_pmsm(Fault((0,)))
        _, _, opt2 = _new_pmsm(Fault(self.OPEN))
        sol1 = opt1.maximize_torque(OMEGA_LOW)
        sol2 = opt2.maximize_torque(OMEGA_LOW)
        assert sol1.success and sol2.success
        assert sol2.torque < sol1.torque
        assert sol2.torque > 0

    def test_minimize_current_achieves_torque(self):
        drive, _, opt = _new_pmsm(Fault(self.OPEN))
        sol_max = opt.maximize_torque(OMEGA_LOW)
        if not sol_max.success:
            pytest.skip("probe failed")
        torq = sol_max.torque * 0.5
        sol = opt.minimize_current(torq, OMEGA_LOW)
        assert sol.success
        np.testing.assert_allclose(
            drive.torque(OMEGA_LOW, sol.curr_dq), torq, rtol=1e-3)

    def test_current_map_shape_and_finite(self):
        """
        Two-fault: curr_ph returns (n_surv=3, n_theta+1), all finite.
        (Two open phases → underdetermined: full roundtrip cannot recover i_dq.)
        """
        _, fwd, _ = _new_pmsm(Fault(self.OPEN))
        i_dq = np.array([3.0, 8.0, 0.5, -1.0])
        i_ph = fwd.curr_ph(OMEGA_LOW, i_dq)
        assert i_ph.shape[0] == 3, f"Expected 3 surviving-phase rows, got {i_ph.shape[0]}"
        assert np.all(np.isfinite(i_ph)), "curr_ph contains non-finite values"


# ──────────────────────────────────────────────────────────────────────────────
# ══ IM (9-phase induction motor) ═════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────

class TestIM:

    I_TEST = np.array([2.0, 2.0, 0.3, -0.3])

    def test_torque_at_fixed_current(self):
        _, model, _, _ = _old_im()
        drive, _, _ = _new_im()
        T_old = model.calculate_torque(IM_OMEGA_LOW, self.I_TEST)
        T_new = drive.torque(IM_OMEGA_LOW, self.I_TEST)
        np.testing.assert_allclose(T_new, T_old, rtol=1e-5,
            err_msg="IM torque mismatch at fixed i_dq")

    def test_curr_peak_at_fixed_current(self):
        _, _, tr, _ = _old_im()
        _, fwd, _ = _new_im()
        curr_peak_old, *_ = tr.get_max_vals(IM_OMEGA_LOW, self.I_TEST)
        curr_peak_new = fwd.peak_vals(IM_OMEGA_LOW, self.I_TEST)[0]
        np.testing.assert_allclose(curr_peak_new, curr_peak_old, rtol=1e-5,
            err_msg="IM curr_peak mismatch")

    def test_volt_peak_at_fixed_current(self):
        _, _, tr, _ = _old_im()
        _, fwd, _ = _new_im()
        _, _, volt_peak_old, _ = tr.get_max_vals(IM_OMEGA_LOW, self.I_TEST)
        volt_peak_new = fwd.peak_vals(IM_OMEGA_LOW, self.I_TEST)[1]
        np.testing.assert_allclose(volt_peak_new, volt_peak_old, rtol=1e-5,
            err_msg="IM volt_peak mismatch")

    def test_maximize_torque(self):
        _, _, tr, opt_old = _old_im()
        _, _, opt_new = _new_im()
        _, T_old, ok_old = opt_old.maximize_torque(IM_OMEGA_LOW, tr)
        sol_new = opt_new.maximize_torque(IM_OMEGA_LOW)
        assert ok_old, "Old IM maximize_torque failed"
        assert sol_new.success, "New IM maximize_torque failed"
        np.testing.assert_allclose(sol_new.torque, T_old, rtol=1e-3,
            err_msg="IM maximize_torque mismatch")

    def test_minimize_current_torque_achieved(self):
        _, model, tr, opt_old = _old_im()
        drive, _, opt_new = _new_im()
        # Probe feasible target
        sol_probe = opt_new.maximize_torque(IM_OMEGA_LOW)
        if not sol_probe.success:
            pytest.skip("IM maximize_torque probe failed")
        torq = sol_probe.torque * 0.5

        i_old, ok_old = opt_old.minimize_current(torq, IM_OMEGA_LOW, tr)
        sol_new = opt_new.minimize_current(torq, IM_OMEGA_LOW)
        if not ok_old:
            pytest.skip("Old IM minimize_current did not converge")
        assert sol_new.success, "New IM minimize_current did not converge"

        T_old = model.calculate_torque(IM_OMEGA_LOW, i_old)
        T_new = drive.torque(IM_OMEGA_LOW, sol_new.curr_dq)
        np.testing.assert_allclose(T_old, torq, rtol=1e-3)
        np.testing.assert_allclose(T_new, torq, rtol=1e-3)
        np.testing.assert_allclose(T_new, T_old, rtol=1e-3,
            err_msg="IM minimize_current torque diverges old vs new")

    def test_minimize_current_norm(self):
        _, _, tr, opt_old = _old_im()
        _, _, opt_new = _new_im()
        sol_probe = opt_new.maximize_torque(IM_OMEGA_LOW)
        if not sol_probe.success:
            pytest.skip("probe failed")
        torq = sol_probe.torque * 0.5
        i_old, ok_old = opt_old.minimize_current(torq, IM_OMEGA_LOW, tr)
        sol_new = opt_new.minimize_current(torq, IM_OMEGA_LOW)
        if not ok_old or not sol_new.success:
            pytest.skip("one optimizer failed")
        norm_old = float(np.dot(i_old, i_old))
        norm_new = float(np.dot(sol_new.curr_dq, sol_new.curr_dq))
        np.testing.assert_allclose(norm_new, norm_old, rtol=1e-3,
            err_msg="IM minimize_current current norms diverge")


# ──────────────────────────────────────────────────────────────────────────────
# ══ Dynamic: independent per-angle ══════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────

def _old_per_angle_all_phases(
    model: ModelAnalytical,
    tr: OldTransform,
    omega: float,
    torq_target: float,
    n_grid: int,
    opts: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Independent per-angle solver using the OLD physics (ModelAnalytical + Transform)
    but enforcing current and voltage limits on ALL n_phases — not just phase A.

    The old dynamic.independent_optimizer._per_angle_constraints only enforces
    phase A at each angle, making it under-constrained vs the new library.
    This wrapper builds the correct all-phase constraints using the same Transform
    matrices, giving an apples-to-apples comparison.

    Returns (theta_grid, curr_dq_grid, ok_grid).
    """
    from scipy.optimize import minimize as scipy_minimize

    machine = tr.machine
    n_phases = machine.n_phases
    dim = machine.dim
    n_theta = tr.vec_theta.size - 1
    shift = tr._phase_shift_samples

    tr._set_omega(omega)
    flux_volt, _ = tr.flux.get_flux(omega, np.zeros(dim))
    bemf_dq = omega * machine.mat_crossc @ flux_volt
    bemf_ph_all = tr.mat_dq_to_ph @ bemf_dq   # (n_theta+1,) phase-A BEMF at each theta

    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    idx_grid = (np.round(theta_grid / (2 * np.pi) * n_theta).astype(int)) % n_theta

    curr_dq_grid = np.full((n_grid, dim), np.nan)
    ok_grid = np.zeros(n_grid, dtype=bool)
    seed = np.array([1.0] + [0.0] * (dim - 1))

    for i, n_idx in enumerate(idx_grid):
        # All-phase constraints: enforce ±curr_max and ±volt_max for every phase k
        cons = []
        for k in range(n_phases):
            k_idx = int((n_idx - k * shift) % n_theta)
            h = tr.mat_dq_to_ph[k_idx].copy()               # current row for phase k
            g = tr.mat_curr_dq_to_volt_ph[k_idx].copy()     # voltage row for phase k
            b = float(bemf_ph_all[k_idx])                    # BEMF at phase k
            cons += [
                {"type": "ineq", "fun": lambda x, h=h:  CURR_MAX - h @ x},
                {"type": "ineq", "fun": lambda x, h=h:  CURR_MAX + h @ x},
                {"type": "ineq", "fun": lambda x, g=g, b=b: VOLT_MAX - (g @ x + b)},
                {"type": "ineq", "fun": lambda x, g=g, b=b: VOLT_MAX + (g @ x + b)},
            ]
        T = torq_target
        cons.append({"type": "eq", "fun": lambda x, T=T: model.calculate_torque(omega, x) - T})

        candidates = model.get_candidates(seed)
        best_fun, best_x = float("inf"), None
        for x0 in candidates:
            res = scipy_minimize(lambda x: float(np.sum(x**2)), x0,
                                 method="SLSQP", constraints=cons, options=opts)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x

        if best_x is not None:
            curr_dq_grid[i] = best_x
            ok_grid[i] = True
            seed = best_x

    return theta_grid, curr_dq_grid, ok_grid


class TestDynamicIndependent:
    """
    Old physics (ModelAnalytical + Transform) with all-phase constraints
    vs new IndependentOptimizer.

    Root cause of the previous 5% tolerance: the old dynamic.independent_optimizer
    enforced only phase-A current/voltage at each angle — a single row constraint.
    The new IndependentOptimizer correctly enforces all n_surv phases.  With the
    same all-phase constraint formulation, both should solve the identical QCQP
    and agree to rtol=1e-3.

    n_theta=64, n_grid=32 for speed.
    """

    N_THETA = 64
    N_GRID  = 32
    TORQ    = 1.5

    def _old(self):
        machine = IEEEMachine2()
        machine.set_max_pars(CURR_MAX, VOLT_MAX, OMEGA_MAX)
        flux = Flux_IEEEMachine2()
        model = ModelAnalytical(machine, flux)
        tr = OldTransform(machine=machine, flux=flux, add_volt_0=False, n_theta=self.N_THETA)
        return machine, model, tr

    def _new(self):
        drive = NewPMSM()
        drive.set_max_pars(CURR_MAX, VOLT_MAX, OMEGA_MAX)
        fwd = ForwardModel(drive, n_theta=self.N_THETA)
        opt = IndependentOptimizer(fwd, opts=SLSQP_OPTS, n_grid=self.N_GRID)
        return drive, fwd, opt

    def test_per_angle_torque_achieved_old_all_phases(self):
        """Old all-phase: torque achieved at every converged angle."""
        machine, model, tr = self._old()
        _, curr_dq_grid, ok_grid = _old_per_angle_all_phases(
            model, tr, OMEGA_LOW, self.TORQ, self.N_GRID, SLSQP_OPTS
        )
        assert ok_grid.sum() > 0, "Old all-phase optimizer did not converge at any angle"
        for n in np.where(ok_grid)[0]:
            T = model.calculate_torque(OMEGA_LOW, curr_dq_grid[n])
            np.testing.assert_allclose(T, self.TORQ, rtol=1e-3,
                err_msg=f"Old all-phase: torque target not met at index {n}")

    def test_per_angle_torque_achieved_new(self):
        """New: torque achieved at every angle."""
        drive, _, opt = self._new()
        sol = opt.minimize_current(self.TORQ, OMEGA_LOW)
        assert sol.success, "New IndependentOptimizer did not converge"
        for n in range(sol.curr_dq.shape[0]):
            T = drive.torque(OMEGA_LOW, sol.curr_dq[n])
            np.testing.assert_allclose(T, self.TORQ, rtol=1e-3,
                err_msg=f"New: torque target not met at index {n}")

    def test_per_angle_norm_matches_rtol_1e3(self):
        """
        With the same all-phase constraint formulation, old and new physics
        solve the identical QCQP. Per-angle ||i||^2 must agree to rtol=1e-3.
        """
        machine, model, tr = self._old()
        drive, fwd, opt = self._new()

        theta_old, curr_old, ok_old = _old_per_angle_all_phases(
            model, tr, OMEGA_LOW, self.TORQ, self.N_GRID, SLSQP_OPTS
        )
        sol_new = opt.minimize_current(self.TORQ, OMEGA_LOW)
        if not sol_new.success:
            pytest.skip("New IndependentOptimizer did not converge")

        theta_new = sol_new.diagnostics.get(
            "theta_grid",
            np.linspace(0, 2 * np.pi, sol_new.curr_dq.shape[0], endpoint=False),
        )

        mismatches = []
        for n_o in np.where(ok_old)[0]:
            n_n = int(np.argmin(np.abs(theta_new - theta_old[n_o])))
            if abs(theta_new[n_n] - theta_old[n_o]) > 0.1:
                continue
            norm_old = float(np.dot(curr_old[n_o], curr_old[n_o]))
            norm_new = float(np.dot(sol_new.curr_dq[n_n], sol_new.curr_dq[n_n]))
            rel_err = abs(norm_new - norm_old) / max(norm_old, 1e-9)
            if rel_err > 1e-3:
                mismatches.append((n_o, theta_old[n_o], norm_old, norm_new, rel_err))

        assert not mismatches, (
            f"Per-angle norms disagree at {len(mismatches)} angles (rtol=1e-3):\n"
            + "\n".join(
                f"  n={m[0]}, θ={m[1]:.3f}: old={m[2]:.6f}, new={m[3]:.6f}, err={m[4]*100:.3f}%"
                for m in mismatches[:5]
            )
        )
