"""InverterDrop: one per-phase inverter drop, projected by ForwardModel onto any
machine's controlled dq planes (5 or 9 phases, healthy or with an open phase)."""

import numpy as np
import pytest

from current_setpoints.models import Fault, ForwardModel, InverterDrop
from current_setpoints.models.machines import ieee_machine2, im9_prototype


def SQUARE(U, h):
    """harmonic h of a square wave of height U in phase with cos"""
    return (4 * U / (np.pi * h)) * (1 if h % 4 == 1 else -1)


@pytest.fixture(scope="module")
def pmsm5():
    return ieee_machine2(curr_max=30.0, volt_max=13.0)


@pytest.fixture(scope="module")
def im9():
    return im9_prototype()


def rotate(x, a):
    """time shift of a dq1/dq3 quantity: plane h rotated by h*a"""
    out = x.copy()
    for k, h in ((0, 1), (2, 3)):
        c, s = np.cos(h * a), np.sin(h * a)
        out[k], out[k + 1] = c * x[k] - s * x[k + 1], s * x[k] + c * x[k + 1]
    return out


@pytest.mark.parametrize("drive_name", ["pmsm5", "im9"])
def test_hard_dead_time_is_a_square_wave(drive_name, request):
    """I_r -> 0: each phase drop is U_dt * sign(i_k), a square wave in phase with a
    pure d1 current: fundamental 4/pi U_dt, 3rd harmonic -4/(3 pi) U_dt -- for 5 and
    9 phases alike, without telling the drop the phase count."""
    drive = request.getfixturevalue(drive_name)
    fwd = ForwardModel(drive, n_theta=4000, volt_drop=InverterDrop(U_dt=5.0, I_r=1e-6))
    E = fwd.drop_dq(np.r_[10.0, np.zeros(drive.dim - 1)])
    assert E == pytest.approx([SQUARE(5.0, 1), 0.0, SQUARE(5.0, 3), 0.0], abs=5e-3)


@pytest.mark.parametrize("drive_name", ["pmsm5", "im9"])
def test_linear_part_is_a_resistance(drive_name, request):
    drive = request.getfixturevalue(drive_name)
    i = np.array([-12.0, 30.0, 4.0, -7.0])
    fwd = ForwardModel(drive, n_theta=200, volt_drop=InverterDrop(U_dt=0.0, I_r=1.0, R_sw=0.1))
    assert fwd.drop_dq(i) == pytest.approx(0.1 * i, abs=1e-12)


def test_planes_are_consistent_under_a_time_shift(pmsm5):
    """drop(time-shifted current) == time-shifted drop: holds only if dq1 and dq3
    come from the same phase-domain waveform with the same angle convention."""
    fwd = ForwardModel(pmsm5, n_theta=200, volt_drop=InverterDrop(U_dt=7.0, I_r=3.0, R_sw=0.05))
    i = np.array([-20.0, 25.0, 6.0, -9.0])
    a = 2 * np.pi * 13 / 200  # whole number of samples -> exact
    assert fwd.drop_dq(rotate(i, a)) == pytest.approx(rotate(fwd.drop_dq(i), a), abs=1e-10)


def test_open_phase_drop_is_the_best_constant_dq_fit(pmsm5):
    """with phase 0 open the drop comes from the surviving phases only: the dq
    voltage is the least-squares fit of their drop waveforms over the period"""
    E = InverterDrop(U_dt=7.0, I_r=3.0, R_sw=0.05)
    fwd = ForwardModel(pmsm5, n_theta=200, fault=Fault((0,)), volt_drop=E)
    i = np.array([-10.0, 20.0, 2.0, -3.0])
    v = fwd.drop_dq(i)
    kept = list(fwd.fault._kept)
    H = fwd._mat_dq_to_ph_all[:-1, kept, :].reshape(-1, 4)
    e = E.phase(np.einsum("tik, k -> ti", fwd._mat_curr_to_ph[:-1], i)).ravel()
    assert H.T @ (H @ v - e) == pytest.approx(np.zeros(4), abs=1e-8)
    healthy = ForwardModel(pmsm5, n_theta=200, volt_drop=E).drop_dq(i)
    assert not np.allclose(v, healthy, atol=1e-3)  # the open phase changes the drop


def test_zero_drop_reproduces_the_plain_model(pmsm5):
    i = np.array([-5.0, 20.0, 1.0, -2.0])
    w = 2 * np.pi * 50.0
    plain = ForwardModel(pmsm5, n_theta=200)
    zero = ForwardModel(pmsm5, n_theta=200, volt_drop=InverterDrop(0.0, 1.0, 0.0))
    assert zero.volt_dq(w, i) == pytest.approx(plain.volt_dq(w, i), abs=1e-12)
    assert zero.peak_vals(w, i) == pytest.approx(plain.peak_vals(w, i), abs=1e-12)


def test_drop_raises_motoring_and_lowers_generating_voltage(pmsm5):
    """the drop is in phase with the current: it adds to the voltage the
    controller must command when motoring, and subtracts when generating"""
    i = np.array([-5.0, 20.0, 0.0, 0.0])
    E = InverterDrop(U_dt=0.5, I_r=1.0, R_sw=0.01)
    for w, sign in ((2 * np.pi * 50.0, 1), (-2 * np.pi * 50.0, -1)):
        assert pmsm5.torque(w, i) * w * sign > 0
        _, v0 = ForwardModel(pmsm5, n_theta=200).peak_vals(w, i)
        _, v1 = ForwardModel(pmsm5, n_theta=200, volt_drop=E).peak_vals(w, i)
        assert sign * (v1 - v0) > 0.1


def test_per_angle_voltage_map_refuses_the_drop(pmsm5):
    fwd = ForwardModel(pmsm5, n_theta=200, volt_drop=InverterDrop(7.0, 3.0))
    with pytest.raises(NotImplementedError):
        fwd.volt_map_at_theta(0, 100.0)


def test_dead_time_constructor_scales_with_the_dc_link():
    a = InverterDrop.from_dead_time(t_dead=2e-6, f_pwm=10e3, U_dc=350.0, I_r=3.0, U_th=1.0)
    b = InverterDrop.from_dead_time(t_dead=2e-6, f_pwm=10e3, U_dc=700.0, I_r=3.0, U_th=1.0)
    assert (a.U_dt, b.U_dt) == pytest.approx((8.0, 15.0))
