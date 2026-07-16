"""IMDriveLUT: instance-parameterised IM drive.

Regression half: with flat (table-free) parameters the instance must
reproduce the hard-coded IM9Phase bit-for-bit. Feature half: the
saturation lookup table changes the physics in the expected direction.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from current_setpoints.models.im_lut import (
    IMDriveLUT,
    IMHarmonicParams,
    IMLUTParams,
    im9_prototype,
)
from current_setpoints.models.machines import IM9Phase

CURR_MAX, VOLT_MAX, OMEGA_MAX = 20.0, 200.0, 1500.0


@pytest.fixture(scope="module")
def ref() -> IM9Phase:
    m = IM9Phase()
    m.set_max_pars(curr_max=CURR_MAX, volt_max=VOLT_MAX, omega_max=OMEGA_MAX)
    return m


@pytest.fixture(scope="module")
def lut() -> IMDriveLUT:
    return im9_prototype()


def _sample_points(n: int = 20) -> np.ndarray:
    return np.random.default_rng(0).normal(scale=8.0, size=(n, 4))


# ── flat parameters: exact match with IM9Phase ────────────────────────────────


def test_identity_attributes(ref, lut):
    assert lut.n_phases == ref.n_phases
    assert lut.n_harmonics == ref.n_harmonics
    assert lut.dim == ref.dim
    assert lut.n_ppairs == ref.n_ppairs
    assert lut.k_phase == ref.k_phase
    assert (lut.curr_max, lut.volt_max, lut.omega_max) == (CURR_MAX, VOLT_MAX, OMEGA_MAX)
    np.testing.assert_array_equal(lut.R_stat, ref.R_stat)


def test_inductance_matches(ref, lut):
    np.testing.assert_allclose(lut.inductance(0.0), ref.inductance(0.0), rtol=0, atol=0)
    for x in _sample_points():
        np.testing.assert_allclose(lut.inductance(100.0, x), ref.inductance(100.0, x), rtol=1e-15, atol=1e-18)


def test_slip_and_kir_match(ref, lut):
    for x in _sample_points():
        s_l, s_r = lut.slip_from_dq_foc(x), ref.slip_from_dq_foc(x)
        assert s_l == pytest.approx(s_r, rel=1e-15, abs=1e-18)
        np.testing.assert_allclose(lut.k_ir(s_l, x), ref.k_ir(s_r), rtol=1e-15, atol=1e-18)


def test_torque_matches(ref, lut):
    for omega in (10.0, 100.0, 500.0):
        for x in _sample_points():
            assert lut.torque(omega, x) == pytest.approx(ref.torque(omega, x), rel=1e-13, abs=1e-15)


def test_voltage_operator_matches(ref, lut):
    for omega in (10.0, 100.0, 500.0):
        for x in _sample_points(5):
            np.testing.assert_allclose(lut.voltage_operator(omega, x), ref.voltage_operator(omega, x), rtol=1e-13)


def test_copper_loss_matches(ref, lut):
    for x in _sample_points():
        assert lut.copper_loss(0.0, x) == pytest.approx(ref.copper_loss(0.0, x), rel=1e-13)


def test_seeds_match(ref, lut):
    for s_l, s_r in zip(lut.seeds(), ref.seeds()):
        np.testing.assert_array_equal(s_l, s_r)
    g = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(lut.seeds(g)[0], ref.seeds(g)[0])


# ── saturation lookup table ───────────────────────────────────────────────────


def _saturating_prototype() -> IMDriveLUT:
    base = im9_prototype()
    p1 = base.params.harmonics[0]
    sat = IMHarmonicParams(
        R_r=p1.R_r,
        L_mu=p1.L_mu,
        L_s_sigma=p1.L_s_sigma,
        L_r_sigma=p1.L_r_sigma,
        i_mag_table=np.array([0.0, 10.0, 20.0]),
        L_mu_table=np.array([p1.L_mu, 0.8 * p1.L_mu, 0.6 * p1.L_mu]),
    )
    return IMDriveLUT(
        IMLUTParams(
            n_phases=9,
            n_ppairs=2,
            R_s=5.0,
            harmonics=[sat, base.params.harmonics[1]],
            curr_max=CURR_MAX,
            volt_max=VOLT_MAX,
            omega_max=OMEGA_MAX,
        )
    )


def test_lut_interpolation():
    sat = _saturating_prototype().params.harmonics[0]
    assert sat.L_mu_at(0.0) == pytest.approx(sat.L_mu)
    assert sat.L_mu_at(5.0) == pytest.approx(0.9 * sat.L_mu)  # midpoint
    assert sat.L_mu_at(-5.0) == pytest.approx(0.9 * sat.L_mu)  # sign-symmetric
    assert sat.L_mu_at(50.0) == pytest.approx(0.6 * sat.L_mu)  # clamped at table end


def test_saturation_changes_physics(ref, lut):
    m = _saturating_prototype()
    x0 = np.array([0.0, 5.0, 0.0, 0.0])  # zero magnetizing current: unsaturated
    assert m.torque(100.0, x0) == pytest.approx(lut.torque(100.0, x0), rel=1e-13)
    x_sat = np.array([15.0, 5.0, 0.0, 0.0])  # deep in the table
    assert m.torque(100.0, x_sat) != pytest.approx(lut.torque(100.0, x_sat), rel=1e-6)
    L_sat = m.inductance(100.0, x_sat)
    L_lin = lut.inductance(100.0, x_sat)
    assert L_sat[0, 0] < L_lin[0, 0]  # saturation lowers the d1 inductance


def test_table_validation():
    with pytest.raises(ValueError):
        IMHarmonicParams(R_r=1.0, L_mu=0.5, L_s_sigma=0.01, L_r_sigma=0.05, i_mag_table=np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        IMHarmonicParams(
            R_r=1.0,
            L_mu=0.5,
            L_s_sigma=0.01,
            L_r_sigma=0.05,
            i_mag_table=np.array([1.0, 0.0]),
            L_mu_table=np.array([0.5, 0.4]),
        )


# ── five-phase async-paper machine (IM_5f.tex + FEM saturation) ──────────────


def test_im5_async_paper_values():
    from current_setpoints.models.im_lut import im5_async

    m = im5_async(omega_max=1000.0)
    assert (m.n_phases, m.n_ppairs, m.dim, m.k_phase) == (5, 2, 4, 2.5)
    assert m.params.R_s == 0.74
    p1, p3 = m.params.harmonics
    assert (p1.R_r, p1.L_mu, p1.L_s_sigma, p1.L_r_sigma) == (0.61, 367.0e-3, 7.23e-3, 5.07e-3)
    assert (p3.R_r, p3.L_mu, p3.L_s_sigma, p3.L_r_sigma) == (0.48, 36.1e-3, 8.94e-3, 3.22e-3)
    assert (m.curr_max, m.volt_max, m.omega_max) == (13.5, 325.0, 1000.0)
    # motoring point produces positive torque
    assert m.torque(100.0, np.array([3.0, 8.0, 0.0, 0.0])) > 0


def test_im5_saturated_matches_linear_at_low_current():
    from current_setpoints.models.im_lut import im5_async, im5_async_saturated

    lin, sat = im5_async(omega_max=1000.0), im5_async_saturated(omega_max=1000.0)
    x_lo = np.array([1.0, 4.0, 0.0, 0.0])  # below the first table knot: some interp already
    # at i_mag -> 0 the table equals the paper value exactly
    assert sat.params.harmonics[0].L_mu_at(0.0) == pytest.approx(367.0e-3)
    assert sat.params.harmonics[0].L_mu_at(2.0) == pytest.approx(367.0e-3)  # flat first segment
    np.testing.assert_allclose(
        sat.inductance(100.0, x_lo * 0 + np.array([2.0, 0.0, 0.0, 0.0])),
        lin.inductance(100.0, np.array([2.0, 0.0, 0.0, 0.0])),
        rtol=1e-12,
    )


def test_im5_saturation_at_rated_current():
    from current_setpoints.models.im_lut import im5_async, im5_async_saturated

    lin, sat = im5_async(omega_max=1000.0), im5_async_saturated(omega_max=1000.0)
    p1 = sat.params.harmonics[0]
    # at the paper's I_max the measured L_mu is ~39 % of the linear value
    assert p1.L_mu_at(13.5) == pytest.approx(142.084e-3, rel=1e-6)
    assert p1.L_mu_at(13.5) / 367.0e-3 == pytest.approx(0.387, abs=0.005)
    # saturation reduces torque at a high-magnetization operating point
    x = np.array([10.0, 8.0, 0.0, 0.0])
    assert sat.torque(100.0, x) < lin.torque(100.0, x)
