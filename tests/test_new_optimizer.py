"""
Tests for StaticOptimizer, IndependentOptimizer, ActiveSetOptimizer, FourierOptimizer.
Verifies: feasibility (torque achieved, constraints satisfied), Solution fields, IM support.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from current_setpoints.models.machines import PMSM5Phase, IM9Phase
from current_setpoints.models.forward_model import Fault, ForwardModel
from current_setpoints.optimization import (
    Solution,
    StaticOptimizer,
    IndependentOptimizer,
    ActiveSetOptimizer,
    FourierOptimizer,
)

CURR_MAX = 30.0
VOLT_MAX = 13.0
OMEGA_MAX = 1800

SLSQP_OPTS = {"disp": False, "ftol": 1e-8, "maxiter": 500}

# electrical rad/s from mechanical RPM
def to_elec(rpm, n_ppairs):
    return rpm * (np.pi / 30) * n_ppairs

OMEGA_LOW  = to_elec(300,  8)
OMEGA_MID  = to_elec(900,  8)
OMEGA_HIGH = to_elec(1500, 8)


@pytest.fixture(scope="module")
def pmsm():
    m = PMSM5Phase()
    m.set_max_pars(CURR_MAX, VOLT_MAX, OMEGA_MAX)
    return m


@pytest.fixture(scope="module")
def im():
    m = IM9Phase()
    m.set_max_pars(curr_max=20.0, volt_max=200.0, omega_max=1500)
    return m


@pytest.fixture(scope="module")
def fwd(pmsm):
    return ForwardModel(pmsm, n_theta=700)


@pytest.fixture(scope="module")
def fwd_f1(pmsm):
    return ForwardModel(pmsm, n_theta=700, fault=Fault((0,)))


@pytest.fixture(scope="module")
def fwd_im(im):
    return ForwardModel(im, n_theta=700)


@pytest.fixture(scope="module")
def static_opt(fwd):
    return StaticOptimizer(fwd, opts=SLSQP_OPTS)


@pytest.fixture(scope="module")
def static_opt_f1(fwd_f1):
    return StaticOptimizer(fwd_f1, opts=SLSQP_OPTS)


@pytest.fixture(scope="module")
def static_opt_im(fwd_im):
    return StaticOptimizer(fwd_im, opts=SLSQP_OPTS)


# ── Solution dataclass ────────────────────────────────────────────────────────

def test_solution_fields():
    s = Solution(curr_dq=np.zeros(4), torque=5.0, success=True)
    assert s.curr_dq.shape == (4,)
    assert s.torque == 5.0
    assert s.success is True
    assert isinstance(s.diagnostics, dict)


# ── StaticOptimizer — minimize_current ────────────────────────────────────────
# Torque targets are derived from maximize_torque to guarantee feasibility.

def _feasible_torque(opt, omega, fraction):
    """Return fraction of max achievable torque at omega, or None if probe fails."""
    sol = opt.maximize_torque(omega)
    return sol.torque * fraction if sol.success else None


@pytest.mark.parametrize("omega,fraction", [
    (OMEGA_LOW, 0.3),
    (OMEGA_LOW, 0.7),
    (OMEGA_MID, 0.3),
    (OMEGA_MID, 0.7),
])
def test_static_minimize_current_converges(static_opt, omega, fraction):
    torq_target = _feasible_torque(static_opt, omega, fraction)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt.minimize_current(torq_target, omega)
    assert sol.success, f"Did not converge for T={torq_target:.2f}, ω={omega:.0f}"


@pytest.mark.parametrize("omega,fraction", [
    (OMEGA_LOW, 0.5),
    (OMEGA_MID, 0.5),
])
def test_static_minimize_current_achieves_torque(static_opt, omega, fraction):
    torq_target = _feasible_torque(static_opt, omega, fraction)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt.minimize_current(torq_target, omega)
    if not sol.success:
        pytest.skip("Optimizer did not converge — skip torque check")
    np.testing.assert_allclose(sol.torque, torq_target, rtol=1e-3,
                               err_msg=f"Torque not achieved at T={torq_target:.2f}, ω={omega:.0f}")


@pytest.mark.parametrize("omega", [OMEGA_LOW, OMEGA_MID])
def test_static_minimize_current_respects_current_limit(static_opt, omega):
    torq_target = _feasible_torque(static_opt, omega, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt.minimize_current(torq_target, omega)
    if not sol.success:
        pytest.skip("Optimizer did not converge")
    cp, _ = static_opt.fwd.peak_vals(omega, sol.curr_dq)
    assert cp <= CURR_MAX + 1e-3, f"Current limit violated: {cp:.4f} > {CURR_MAX}"


@pytest.mark.parametrize("omega", [OMEGA_LOW, OMEGA_MID])
def test_static_minimize_current_respects_voltage_limit(static_opt, omega):
    torq_target = _feasible_torque(static_opt, omega, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt.minimize_current(torq_target, omega)
    if not sol.success:
        pytest.skip("Optimizer did not converge")
    _, vp = static_opt.fwd.peak_vals(omega, sol.curr_dq)
    assert vp <= VOLT_MAX + 1e-3, f"Voltage limit violated: {vp:.4f} > {VOLT_MAX}"


def test_static_minimize_current_curr_dq_shape(static_opt):
    torq_target = _feasible_torque(static_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt.minimize_current(torq_target, OMEGA_LOW)
    assert sol.curr_dq.shape == (4,)


def test_static_minimize_current_with_warm_start(static_opt):
    torq_target = _feasible_torque(static_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol_cold = static_opt.minimize_current(torq_target, OMEGA_LOW)
    guess = sol_cold.curr_dq if sol_cold.success else None
    sol_warm = static_opt.minimize_current(torq_target, OMEGA_LOW, guess=guess)
    if sol_cold.success and sol_warm.success:
        assert np.sum(sol_warm.curr_dq**2) <= np.sum(sol_cold.curr_dq**2) + 1e-6


# ── StaticOptimizer — maximize_torque ─────────────────────────────────────────

@pytest.mark.parametrize("omega", [OMEGA_LOW, OMEGA_MID, OMEGA_HIGH])
def test_static_maximize_torque_converges(static_opt, omega):
    sol = static_opt.maximize_torque(omega)
    assert sol.success, f"maximize_torque did not converge at ω={omega:.0f}"


@pytest.mark.parametrize("omega", [OMEGA_LOW, OMEGA_MID, OMEGA_HIGH])
def test_static_maximize_torque_positive(static_opt, omega):
    sol = static_opt.maximize_torque(omega)
    if not sol.success:
        pytest.skip("Did not converge")
    assert sol.torque > 0, f"maximize_torque returned non-positive torque at ω={omega:.0f}"


@pytest.mark.parametrize("omega", [OMEGA_LOW, OMEGA_MID, OMEGA_HIGH])
def test_static_maximize_torque_limits(static_opt, omega):
    sol = static_opt.maximize_torque(omega)
    if not sol.success:
        pytest.skip("Did not converge")
    cp, vp = static_opt.fwd.peak_vals(omega, sol.curr_dq)
    assert cp <= CURR_MAX + 1e-3
    assert vp <= VOLT_MAX + 1e-3


def test_static_maximize_torque_decreases_with_speed(static_opt):
    """Available torque should drop as speed increases into field-weakening."""
    T_low  = static_opt.maximize_torque(OMEGA_LOW).torque
    T_high = static_opt.maximize_torque(OMEGA_HIGH).torque
    assert T_low > T_high, f"Expected T(low speed) > T(high speed): {T_low:.2f} vs {T_high:.2f}"


# ── StaticOptimizer — single fault ───────────────────────────────────────────

def test_static_fault1_minimize_current_converges(static_opt_f1):
    torq_target = _feasible_torque(static_opt_f1, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt_f1.minimize_current(torq_target, OMEGA_LOW)
    assert sol.success, "Single-fault minimize_current failed to converge"


def test_static_fault1_achieves_torque(static_opt_f1):
    torq_target = _feasible_torque(static_opt_f1, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt_f1.minimize_current(torq_target, OMEGA_LOW)
    if not sol.success:
        pytest.skip("Did not converge")
    np.testing.assert_allclose(sol.torque, torq_target, rtol=1e-3)


def test_static_fault1_limits(static_opt_f1):
    torq_target = _feasible_torque(static_opt_f1, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt_f1.minimize_current(torq_target, OMEGA_LOW)
    if not sol.success:
        pytest.skip("Did not converge")
    cp, vp = static_opt_f1.fwd.peak_vals(OMEGA_LOW, sol.curr_dq)
    assert cp <= CURR_MAX + 1e-3
    assert vp <= VOLT_MAX + 1e-3


def test_static_fault1_maximize_torque(static_opt_f1):
    sol = static_opt_f1.maximize_torque(OMEGA_LOW)
    assert sol.success
    assert sol.torque > 0


# ── StaticOptimizer — IM ─────────────────────────────────────────────────────

def test_static_im_maximize_torque_converges(static_opt_im):
    omega = to_elec(500, 2)
    sol = static_opt_im.maximize_torque(omega)
    assert sol.success, "IM maximize_torque failed to converge"


def test_static_im_maximize_torque_positive(static_opt_im):
    omega = to_elec(500, 2)
    sol = static_opt_im.maximize_torque(omega)
    if not sol.success:
        pytest.skip("Did not converge")
    assert sol.torque > 0


def test_static_im_minimize_current_converges(static_opt_im):
    omega = to_elec(300, 2)
    torq_target = _feasible_torque(static_opt_im, omega, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt_im.minimize_current(torq_target, omega)
    assert sol.success, "IM minimize_current failed to converge"


def test_static_im_minimize_current_achieves_torque(static_opt_im):
    omega = to_elec(300, 2)
    torq_target = _feasible_torque(static_opt_im, omega, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = static_opt_im.minimize_current(torq_target, omega)
    if not sol.success:
        pytest.skip("Did not converge")
    np.testing.assert_allclose(sol.torque, torq_target, rtol=1e-3)


# ── IndependentOptimizer ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fwd_small(pmsm):
    """Small n_theta for dynamic optimizer tests — 700 angles × SLSQP is too slow."""
    return ForwardModel(pmsm, n_theta=50)


@pytest.fixture(scope="module")
def ind_opt(fwd_small):
    return IndependentOptimizer(fwd_small, opts={**SLSQP_OPTS, "maxiter": 100})


def test_ind_minimize_current_converges(ind_opt):
    torq_target = _feasible_torque(ind_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = ind_opt.minimize_current(torq_target, OMEGA_LOW)
    assert sol.success, "IndependentOptimizer minimize_current failed"


def test_ind_minimize_current_trajectory_shape(ind_opt, fwd_small):
    torq_target = _feasible_torque(ind_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = ind_opt.minimize_current(torq_target, OMEGA_LOW)
    if not sol.success:
        pytest.skip("Did not converge")
    n_grid = ind_opt.n_grid
    assert sol.curr_dq.ndim == 2 and sol.curr_dq.shape[1] == 4


def test_ind_maximize_torque_converges(ind_opt):
    sol = ind_opt.maximize_torque(OMEGA_MID)
    assert sol.success, "IndependentOptimizer maximize_torque failed"


def test_ind_maximize_torque_positive(ind_opt):
    sol = ind_opt.maximize_torque(OMEGA_MID)
    if not sol.success:
        pytest.skip("Did not converge")
    assert sol.torque > 0


# ── ActiveSetOptimizer ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def active_opt(fwd_small):
    return ActiveSetOptimizer(fwd_small, opts={**SLSQP_OPTS, "maxiter": 200})


def test_active_maximize_torque_converges(active_opt):
    sol = active_opt.maximize_torque(OMEGA_LOW)
    assert sol.success, "ActiveSetOptimizer maximize_torque failed"


def test_active_maximize_torque_positive(active_opt):
    sol = active_opt.maximize_torque(OMEGA_LOW)
    if not sol.success:
        pytest.skip("Did not converge")
    assert sol.torque > 0


def test_active_minimize_current_converges(active_opt):
    torq_target = _feasible_torque(active_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = active_opt.minimize_current(torq_target, OMEGA_LOW)
    assert sol.success, "ActiveSetOptimizer minimize_current failed"


def test_active_minimize_current_achieves_torque(active_opt):
    torq_target = _feasible_torque(active_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = active_opt.minimize_current(torq_target, OMEGA_LOW)
    if not sol.success:
        pytest.skip("Did not converge")
    np.testing.assert_allclose(sol.torque, torq_target, rtol=1e-3)


# ── FourierOptimizer ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fourier_opt(fwd_small):
    # harmonics=(0,1): nb=3 < dim=4 — avoids the warm-start recursive call in _solve
    return FourierOptimizer(fwd_small, opts={**SLSQP_OPTS, "maxiter": 300}, harmonics=(0, 1))


def test_fourier_maximize_torque_converges(fourier_opt):
    sol = fourier_opt.maximize_torque(OMEGA_MID)
    assert sol.success, "FourierOptimizer maximize_torque failed"


def test_fourier_maximize_torque_positive(fourier_opt):
    sol = fourier_opt.maximize_torque(OMEGA_MID)
    if not sol.success:
        pytest.skip("Did not converge")
    assert sol.torque > 0


def test_fourier_minimize_current_converges(fourier_opt):
    torq_target = _feasible_torque(fourier_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = fourier_opt.minimize_current(torq_target, OMEGA_LOW)
    assert sol.success, "FourierOptimizer minimize_current failed"


def test_fourier_minimize_current_achieves_torque(fourier_opt):
    torq_target = _feasible_torque(fourier_opt, OMEGA_LOW, 0.5)
    if torq_target is None:
        pytest.skip("maximize_torque probe failed")
    sol = fourier_opt.minimize_current(torq_target, OMEGA_LOW)
    if not sol.success:
        pytest.skip("Did not converge")
    np.testing.assert_allclose(sol.torque, torq_target, rtol=1e-3)
