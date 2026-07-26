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
    assert (lut.curr_max, lut.volt_max) == (CURR_MAX, VOLT_MAX)
    assert not hasattr(lut, "omega_max")  # map horizon is a grid-time opt, not machine data
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

    m = im5_async()
    assert (m.n_phases, m.n_ppairs, m.dim, m.k_phase) == (5, 2, 4, 2.5)
    assert m.params.R_s == 0.74
    p1, p3 = m.params.harmonics
    assert (p1.R_r, p1.L_mu, p1.L_s_sigma, p1.L_r_sigma) == (0.61, 367.0e-3, 7.23e-3, 5.07e-3)
    assert (p3.R_r, p3.L_mu, p3.L_s_sigma, p3.L_r_sigma) == (0.48, 36.1e-3, 8.94e-3, 3.22e-3)
    assert (m.curr_max, m.volt_max) == (13.5, 325.0)
    # motoring point produces positive torque
    assert m.torque(100.0, np.array([3.0, 8.0, 0.0, 0.0])) > 0


# FEM clean-slice torques (Id1, Iq1) -> T_FEM [Nm] from the 357-point Ansys
# sweep I_combs_Results_correct_wr_definition.xlsx (Id3 = Iq3 = 0).
_FEM_CLEAN = [
    (40, 40, 20.42), (40, 80, 45.98), (40, 120, 72.96), (40, 160, 100.24), (40, 200, 127.41),
    (80, 40, 21.74), (80, 80, 46.51), (80, 120, 75.00), (80, 160, 105.76), (80, 200, 137.10),
    (120, 80, 35.69), (120, 120, 56.67), (120, 160, 77.86), (120, 200, 99.74),
    (160, 120, 44.56), (160, 160, 61.18),
]


def test_im5_tesla_gen1_known_values():
    from current_setpoints.models.im_lut import im5_tesla_gen1

    # Computed-from-FEM equivalent circuit (Laksar, Ansys, 2026-07-20); the
    # linear form (saturating=False) exposes the raw EC parameters.
    m = im5_tesla_gen1(saturating=False)
    assert (m.n_phases, m.n_ppairs, m.dim) == (5, 3, 4)
    assert m.params.R_s == pytest.approx(0.02194121, rel=1e-12)
    assert (m.curr_max, m.volt_max) == (200.0, 230.0)
    p1, p3 = m.params.harmonics
    assert (p1.R_r, p1.L_mu, p1.L_s_sigma, p1.L_r_sigma) == (
        0.006191796,
        2.193562e-3,
        5.913e-5,
        6.851e-5,
    )
    assert (p3.R_r, p3.L_mu, p3.L_s_sigma, p3.L_r_sigma) == (
        0.004503021,
        0.202479e-3,
        6.465e-5,
        5.166e-5,
    )
    # emailed total stator/rotor inductances L_1, L_2' are reproduced
    assert p1.L_mu + p1.L_s_sigma == pytest.approx(2.253e-3, rel=1e-3)
    assert p1.L_mu + p1.L_r_sigma == pytest.approx(2.262e-3, rel=1e-3)


def test_im5_tesla_gen1_saturation_law():
    """Default (saturating) fundamental plane follows the identified Route B
    law L_mu1(Id1) = L0/(1+(Id1/i_sat)^p): monotone-decreasing, recovering the
    computed EC at low current (2.17 mH @ 40 A) and collapsing under load."""
    from current_setpoints.models.im_lut import im5_tesla_gen1

    p1 = im5_tesla_gen1().params.harmonics[0]
    assert p1.L_mu_at(0.0) == pytest.approx(2.4757e-3, rel=1e-3)
    assert p1.L_mu_at(40.0) == pytest.approx(2.17e-3, rel=1e-2)
    assert p1.L_mu_at(160.0) < 0.7e-3  # deep saturation
    assert p1.L_mu_at(40.0) > p1.L_mu_at(80.0) > p1.L_mu_at(160.0)


def test_im5_tesla_slip_uses_fixed_no_load_constants():
    from current_setpoints.models.im_lut import im5_tesla_gen1

    m = im5_tesla_gen1()  # slip is fixed regardless of saturation mode
    x = np.array([120.0, 80.0, 0.0, 0.0])
    # slip follows the fixed fundamental no-load constants (the FEM wr
    # definition): R_r1 = 6.19 mOhm, L_r1 = L_mu1 + L_r_sigma1 = 2.262 mH
    r_r1, l_r1 = 0.006191796, 2.193562e-3 + 6.851e-5
    assert m.slip_from_dq_foc(x) == pytest.approx((r_r1 / l_r1) * (120.0 / 80.0), rel=1e-12)


def test_im5_tesla_saturating_beats_linear_on_fem_sweep():
    """The saturating model (Route B, ADR 0003) tracks the FEM clean slice to
    a small RMS, while the linear model overshoots grossly once Id1 saturates
    the iron — the reason saturating is the default."""
    from current_setpoints.models.im_lut import im5_tesla_gen1

    sat = im5_tesla_gen1(saturating=True)
    lin = im5_tesla_gen1(saturating=False)

    def rms(machine):
        e = [machine.torque(0.0, np.array([float(d), float(q), 0.0, 0.0])) - t
             for d, q, t in _FEM_CLEAN]
        return float(np.sqrt(np.mean(np.square(e))))

    assert rms(sat) < 15.0  # ~11 Nm in the identification
    assert rms(lin) > 100.0  # linear is unusable across the sweep
    # at the least-saturated point (Id1=40) both are close to FEM
    assert sat.torque(0.0, np.array([40.0, 200.0, 0.0, 0.0])) == pytest.approx(127.41, rel=0.05)
