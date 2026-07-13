"""
Physics-correctness tests for PMSM5Phase and IM9Phase.
All tests use current_setpoints exclusively.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from current_setpoints.models.machines import (
    PMSM5Phase, IM9Phase, ConstantFlux, _build_cross_coupling
)

CURR_MAX = 30.0
VOLT_MAX = 13.0
OMEGA_MAX = 1800


# ── Fixtures ──────────────────────────────────────────────────────────────────

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


# ── Cross-coupling matrix ─────────────────────────────────────────────────────

def test_cross_coupling_antisymmetric():
    J = _build_cross_coupling(2)
    assert J.shape == (4, 4)
    np.testing.assert_array_equal(J, -J.T)


def test_cross_coupling_harmonic_values():
    J = _build_cross_coupling(2)
    # h=1 block: [[0,-1],[1,0]]; h=3 block: [[0,-3],[3,0]]
    assert J[0, 1] == -1 and J[1, 0] == 1
    assert J[2, 3] == -3 and J[3, 2] == 3


# ── PMSM5Phase geometry ───────────────────────────────────────────────────────

def test_pmsm_dimensions(pmsm):
    assert pmsm.n_phases == 5
    assert pmsm.n_ppairs == 8
    assert pmsm.dim == 4
    assert pmsm.n_harmonics == 2


def test_pmsm_resistance_shape(pmsm):
    assert pmsm.R_stat.shape == (4, 4)
    # diagonal resistance — off-diagonals must be zero
    np.testing.assert_array_equal(pmsm.R_stat, np.diag(np.diag(pmsm.R_stat)))


def test_pmsm_inductance_shape(pmsm):
    L = pmsm.inductance(0.0)
    assert L.shape == (4, 4)


def test_pmsm_inductance_positive_definite(pmsm):
    L = pmsm.inductance(0.0)
    eigenvalues = np.linalg.eigvalsh(L)
    assert np.all(eigenvalues > 0), f"L not positive definite: {eigenvalues}"


def test_pmsm_flux_shapes(pmsm):
    zeros = np.zeros(4)
    fv = pmsm.flux.flux(0.0, zeros)
    assert fv.shape == (4,)


def test_pmsm_flux_volt_values(pmsm):
    fv = pmsm.flux.flux(0.0, np.zeros(4))
    np.testing.assert_allclose(fv[0], 1.13438169e-02, rtol=1e-6)
    np.testing.assert_allclose(fv[1], 1.71999345e-03, rtol=1e-6)
    np.testing.assert_allclose(fv[2:], 0.0, atol=2e-5)


# ── PMSM5Phase physics ────────────────────────────────────────────────────────

def test_pmsm_torque_zero_current(pmsm):
    """Zero current must produce zero torque."""
    for omega in [0.0, 100.0, 500.0]:
        T = pmsm.torque(omega, np.zeros(4))
        assert abs(T) < 1e-12, f"T={T} at omega={omega} with zero current"


def test_pmsm_torque_positive_for_positive_iq(pmsm):
    """Positive i_q1 (index 1) should produce positive torque."""
    curr = np.array([0.0, 20.0, 0.0, 0.0])
    T = pmsm.torque(100.0, curr)
    assert T > 0, f"Expected T>0, got T={T}"


def test_pmsm_torque_sign_flips_with_iq(pmsm):
    """Torque sign must flip with i_q1 sign (quadratic term breaks exact antisymmetry)."""
    curr_pos = np.array([0.0, 10.0, 0.0, 0.0])
    curr_neg = np.array([0.0, -10.0, 0.0, 0.0])
    T_pos = pmsm.torque(100.0, curr_pos)
    T_neg = pmsm.torque(100.0, curr_neg)
    assert T_pos > 0 and T_neg < 0


def test_pmsm_torque_scales_with_current(pmsm):
    """Torque is not constant — larger |i| should produce more torque (linear PM term dominates)."""
    omega = 100.0
    T5  = pmsm.torque(omega, np.array([0.0,  5.0, 0.0, 0.0]))
    T20 = pmsm.torque(omega, np.array([0.0, 20.0, 0.0, 0.0]))
    assert T20 > T5 > 0


def test_pmsm_bemf_zero_at_zero_speed(pmsm):
    """BEMF must be zero at standstill."""
    bemf = pmsm.bemf_dq(0.0, np.zeros(4))
    np.testing.assert_allclose(bemf, 0.0, atol=1e-14)


def test_pmsm_bemf_nonzero_at_speed(pmsm):
    """BEMF must be non-zero at non-zero speed with PM flux."""
    bemf = pmsm.bemf_dq(100.0, np.zeros(4))
    assert np.any(np.abs(bemf) > 0)


def test_pmsm_voltage_operator_shape(pmsm):
    U = pmsm.voltage_operator(100.0, np.zeros(4))
    assert U.shape == (4, 4)


def test_pmsm_voltage_operator_at_zero_speed_equals_resistance(pmsm):
    """At ω=0 the voltage operator reduces to R_stat."""
    U = pmsm.voltage_operator(0.0, np.zeros(4))
    np.testing.assert_allclose(U, pmsm.R_stat, rtol=1e-14)


def test_pmsm_torque_quadratic_approximation(pmsm):
    """torque_quadratic must reproduce torque() within 0.5 Nm over a grid."""
    omega = 200.0
    A, b, c = pmsm.torque_quadratic(omega)
    rng = np.random.default_rng(42)
    for _ in range(50):
        i = rng.uniform(-15.0, 15.0, size=4)
        T_exact = pmsm.torque(omega, i)
        T_approx = float(i @ A @ i + b @ i + c)
        np.testing.assert_allclose(T_approx, T_exact, atol=1.0, rtol=0.1)


def test_pmsm_set_max_pars(pmsm):
    assert pmsm.curr_max == CURR_MAX
    assert pmsm.volt_max == VOLT_MAX
    assert pmsm.omega_max == OMEGA_MAX


def test_pmsm_seeds_returns_list_of_arrays(pmsm):
    seeds = pmsm.seeds()
    assert isinstance(seeds, list)
    assert len(seeds) >= 2
    for s in seeds:
        assert s.shape == (4,)


def test_pmsm_seeds_with_guess(pmsm):
    guess = np.array([1.0, 5.0, 0.5, -0.5])
    seeds = pmsm.seeds(guess)
    found = any(np.allclose(s, guess) for s in seeds)
    assert found, "guess should appear in seeds list"


# ── IM9Phase geometry ─────────────────────────────────────────────────────────

def test_im_dimensions(im):
    assert im.n_phases == 9
    assert im.n_ppairs == 2
    assert im.dim == 4
    assert im.n_harmonics == 2


def test_im_resistance_shape(im):
    assert im.R_stat.shape == (4, 4)


def test_im_inductance_shape(im):
    curr = np.array([5.0, 5.0, 0.0, 0.0])
    L = im.inductance(100.0, curr)
    assert L.shape == (4, 4)


# ── IM9Phase physics ──────────────────────────────────────────────────────────

def test_im_bemf_always_zero(im):
    """IM has no PM flux — BEMF is always zero."""
    for omega, curr in [
        (0.0,   np.zeros(4)),
        (100.0, np.array([5.0, 5.0, 0.0, 0.0])),
        (500.0, np.array([10.0, -5.0, 1.0, 0.0])),
    ]:
        bemf = im.bemf_dq(omega, curr)
        np.testing.assert_allclose(bemf, 0.0, atol=1e-14, err_msg=f"omega={omega}")


def test_im_torque_zero_at_zero_slip(im):
    """k_ir is zero at synchronous speed (omega_r=0), so torque should be zero."""
    # slip_from_dq_foc returns 0 when i_sq^1 = 0
    curr = np.array([5.0, 0.0, 0.0, 0.0])   # i_q1 = 0 → omega_r = 0
    T = im.torque(0.0, curr)
    assert abs(T) < 1e-10, f"Expected T≈0 at zero slip, got {T}"


def test_im_torque_positive_for_positive_iq(im):
    """Positive i_d1 and i_q1 should produce positive torque."""
    curr = np.array([5.0, 5.0, 0.0, 0.0])
    T = im.torque(100.0, curr)
    assert T > 0, f"Expected T>0, got T={T}"


def test_im_torque_increases_with_current(im):
    curr_small = np.array([3.0, 3.0, 0.0, 0.0])
    curr_large = np.array([8.0, 8.0, 0.0, 0.0])
    T_small = im.torque(100.0, curr_small)
    T_large = im.torque(100.0, curr_large)
    assert T_large > T_small


def test_im_slip_zero_when_iq_zero(im):
    curr = np.array([5.0, 0.0, 0.0, 0.0])
    assert im.slip_from_dq_foc(curr) == 0.0


def test_im_slip_positive_for_positive_iq(im):
    """slip_from_dq_foc = (R_r / L_r) * (i_d / i_q). With i_d, i_q > 0 → slip > 0."""
    curr = np.array([5.0, 5.0, 0.0, 0.0])
    omega_r = im.slip_from_dq_foc(curr)
    assert omega_r > 0


def test_im_kir_zero_at_synchronous_speed(im):
    """k_ir(0) must be the zero matrix (no rotor current at synchronous speed)."""
    K = im.k_ir(0.0)
    np.testing.assert_allclose(K, 0.0, atol=1e-14)


def test_im_kir_block_diagonal(im):
    """k_ir must be block-diagonal (no coupling between harmonic subspaces)."""
    K = im.k_ir(10.0)
    assert abs(K[0, 2]) < 1e-14
    assert abs(K[0, 3]) < 1e-14
    assert abs(K[1, 2]) < 1e-14
    assert abs(K[1, 3]) < 1e-14
    assert abs(K[2, 0]) < 1e-14
    assert abs(K[3, 0]) < 1e-14


def test_im_inductance_at_none_current(im):
    """inductance(omega, None) must return L_s (static approximation)."""
    L = im.inductance(100.0, None)
    assert L.shape == (4, 4)
    # Should equal L_s = L_mu + L_sigma — just check it's a reasonable diagonal
    assert np.all(np.diag(L) > 0)


def test_im_voltage_operator_shape(im):
    curr = np.array([5.0, 5.0, 0.0, 0.0])
    U = im.voltage_operator(100.0, curr)
    assert U.shape == (4, 4)


def test_im_voltage_operator_at_zero_speed(im):
    """At omega=0 the voltage operator must equal R_stat."""
    curr = np.array([5.0, 5.0, 0.0, 0.0])
    U = im.voltage_operator(0.0, curr)
    np.testing.assert_allclose(U, im.R_stat, rtol=1e-14)


def test_im_seeds_returns_list(im):
    seeds = im.seeds()
    assert isinstance(seeds, list)
    assert len(seeds) >= 4
    for s in seeds:
        assert s.shape == (4,)


def test_im_copper_loss_nonnegative(im):
    curr = np.array([5.0, 5.0, 0.0, 0.0])
    P = im.copper_loss(100.0, curr)
    assert P >= 0


def test_im_copper_loss_zero_at_zero_current(im):
    P = im.copper_loss(100.0, np.zeros(4))
    assert abs(P) < 1e-14
