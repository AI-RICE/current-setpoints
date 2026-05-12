import numpy as np
import pytest

from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform


def _build_default_transform(add_volt_0: bool = False, n_theta: int = 700) -> Transform:
    """Returns a Transform tied to a fresh, default-configured IEEEMachine2."""
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()
    return Transform(machine=machine, flux=flux, add_volt_0=add_volt_0, n_theta=n_theta)


def _project_dq(signal: np.ndarray, theta: np.ndarray, h: int) -> tuple[float, float]:
    """Project a phase-A periodic waveform onto the (cos h*theta, -sin h*theta) basis."""
    s = signal[:-1]
    t = theta[:-1]
    d = 2.0 * np.mean(s * np.cos(h * t))
    q = -2.0 * np.mean(s * np.sin(h * t))
    return float(d), float(q)


def test_init_rounds_n_theta_to_multiple_of_2_n_phases():
    """
    n_theta is silently rounded to the nearest multiple of 2 * n_phases
    so a full electrical period samples cleanly across all phases.
    """
    transform = _build_default_transform(n_theta=703)
    assert transform.vec_theta.size == 701  # n_theta + 1


def test_init_rejects_nonpositive_n_theta():
    machine = IEEEMachine2()
    flux = Flux_IEEEMachine2()
    with pytest.raises(ValueError, match="n_theta must be positive"):
        Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=0)


def test_init_rejects_n_theta_that_rounds_to_zero():
    """
    A small n_theta like 3 with n_phases=5 rounds to 0; this must be caught
    instead of silently producing a single-sample theta vector.
    """
    machine = IEEEMachine2()
    flux = Flux_IEEEMachine2()
    with pytest.raises(ValueError, match="too small after rounding"):
        Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=3)


def test_phase_shift_samples_matches_symmetry():
    """
    n_theta / n_phases samples must correspond to one (2*pi / n_phases)
    phase shift; SVPWM injection relies on this.
    """
    transform = _build_default_transform(n_theta=700)
    n_phases = transform.n_phases
    assert transform._phase_shift_samples * n_phases == transform.vec_theta.size - 1


def test_dq_to_ph_matrix_columns_are_orthogonal_harmonics():
    """
    For an n-phase machine the DQ-to-phase matrix consists of cos/sin pairs
    at odd harmonics 1, 3, 5, ... Sampled across a full period these columns
    are mutually orthogonal in the L2 inner product (modulo the wrap point).
    """
    transform = _build_default_transform(n_theta=700)
    M = transform.mat_dq_to_ph[:-1]  # drop wrap sample so dot products are clean
    gram = M.T @ M
    n = M.shape[0]
    assert np.allclose(gram, (n / 2) * np.eye(M.shape[1]), atol=1e-9)


def test_get_curr_ph_pure_d1_produces_pure_cosine():
    """Setting only i_d1 = I should make phase A track I * cos(theta)."""
    transform = _build_default_transform(n_theta=700)
    vec_curr_dq = np.zeros(transform.dim)
    vec_curr_dq[0] = 5.0
    i_phase = transform.get_curr_ph(omega=0.0, curr_dq=vec_curr_dq)
    expected = 5.0 * np.cos(transform.vec_theta)
    assert np.allclose(i_phase, expected)


def test_get_curr_ph_pure_q1_produces_pure_negative_sine():
    """Setting only i_q1 = I should give -I * sin(theta) per the (cos, -sin) convention."""
    transform = _build_default_transform(n_theta=700)
    vec_curr_dq = np.zeros(transform.dim)
    vec_curr_dq[1] = 5.0
    i_phase = transform.get_curr_ph(omega=0.0, curr_dq=vec_curr_dq)
    expected = -5.0 * np.sin(transform.vec_theta)
    assert np.allclose(i_phase, expected)


def test_get_volt_dq_zero_current_zero_speed_is_zero():
    """At i = 0 and omega = 0 there should be no DQ voltage drop."""
    transform = _build_default_transform()
    v_dq = transform.get_volt_dq(omega=0.0, curr_dq=np.zeros(transform.dim))
    assert np.allclose(v_dq, 0.0)


def test_get_volt_dq_zero_current_nonzero_speed_is_pure_bemf():
    """
    With i = 0 and omega != 0, v_dq should equal omega * (mat_crossc @ flux_volt).
    Flux now lives on the Flux provider, not the machine, so we read it via
    `transform.flux.get_flux(omega, curr_dq)`.
    """
    transform = _build_default_transform()
    omega = 50.0
    zero_curr = np.zeros(transform.dim)
    v_dq = transform.get_volt_dq(omega=omega, curr_dq=zero_curr)

    flux_volt, _ = transform.flux.get_flux(omega, zero_curr)
    expected = omega * (transform.machine.mat_crossc @ flux_volt)
    assert np.allclose(v_dq, expected)


def test_set_omega_updates_speed_dependent_matrix():
    """
    `_compute_matrices_init` builds the omega-independent term once;
    `_set_omega` adds the omega-scaled term. Testing the private hook
    directly because that's exactly what's being validated.
    """
    transform = _build_default_transform()
    transform._set_omega(0.0)
    M0 = transform.mat_curr_dq_to_volt_dq.copy()
    transform._set_omega(100.0)
    M_high = transform.mat_curr_dq_to_volt_dq

    diff = M_high - M0
    expected = 100.0 * (transform.machine.mat_crossc @ transform.machine.L_stat)
    assert np.allclose(diff, expected)


def test_get_volt_ph_no_injection_equals_raw():
    """When add_volt_0=False, vec_volt_ph must equal vec_volt_raw exactly."""
    transform = _build_default_transform(add_volt_0=False)
    vec_curr_dq = np.array([5.0, 3.0, 0.0, 0.0])
    vec_volt_ph, vec_volt_0, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    assert np.allclose(vec_volt_ph, vec_volt_raw)
    assert np.allclose(vec_volt_0, 0.0)


# ---------------------------------------------------------------------------
# Zero-sequence injection (SVPWM) is not yet implemented in the library.
# The four tests below describe the contract the implementation should meet
# once it lands. They are kept (marked skip) so they re-enable automatically
# the moment the NotImplementedError in `get_volt_ph` is replaced by a real
# implementation.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_pure_fundamental_injection_matches_theory():
    """
    Min-max common-mode injection on a pure-fundamental phase voltage must
    reduce the peak by exactly cos(pi / (2 * n_phases)) — for 5-phase that
    is cos(pi/10) ≈ 0.9511.
    """
    transform = _build_default_transform(add_volt_0=True, n_theta=700)
    vec_curr_dq = np.zeros(transform.dim)
    vec_curr_dq[0] = 10.0  # pure i_d1
    vec_volt_ph, _, vec_volt_raw = transform.get_volt_ph(omega=0.0, curr_dq=vec_curr_dq)

    raw_peak = np.max(np.abs(vec_volt_raw))
    inj_peak = np.max(np.abs(vec_volt_ph))
    expected_ratio = np.cos(np.pi / (2 * transform.n_phases))

    assert raw_peak > 0
    assert np.isclose(inj_peak / raw_peak, expected_ratio, atol=1e-3)


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_injection_reduces_peak_under_load():
    """
    Under a realistic operating point (fundamental + harmonic content)
    injection must still reduce the peak phase voltage vs the raw waveform.
    """
    transform = _build_default_transform(add_volt_0=True)
    vec_curr_dq = np.array([10.0, 5.0, 1.0, 0.5])
    vec_volt_ph, _, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    assert np.max(np.abs(vec_volt_ph)) < np.max(np.abs(vec_volt_raw))


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_injection_preserves_dq_content():
    """
    SVPWM is a common-mode signal: it must NOT contaminate the DQ subspace.
    Projecting the injected phase-A waveform back onto the cos/sin basis at
    each odd harmonic must give the same DQ values as projecting the raw.
    """
    transform = _build_default_transform(add_volt_0=True)
    vec_curr_dq = np.array([10.0, 5.0, 1.0, 0.5])
    vec_volt_ph, _, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    theta = transform.vec_theta

    for h in (1, 3):
        d_raw, q_raw = _project_dq(vec_volt_raw, theta, h)
        d_inj, q_inj = _project_dq(vec_volt_ph, theta, h)
        assert np.isclose(d_raw, d_inj, atol=1e-9)
        assert np.isclose(q_raw, q_inj, atol=1e-9)


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_injection_v0_returned_consistently():
    """vec_volt_ph == vec_volt_raw + vec_volt_0 sample by sample."""
    transform = _build_default_transform(add_volt_0=True)
    vec_curr_dq = np.array([7.0, 2.0, 0.5, -0.5])
    vec_volt_ph, vec_volt_0, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    assert np.allclose(vec_volt_ph, vec_volt_raw + vec_volt_0)


def test_get_max_vals_returns_four_values():
    """
    `get_max_vals` was cut from eight returns to four:
    (curr_peak, curr_ang_diff, volt_peak, volt_ang_diff).
    """
    transform = _build_default_transform()
    out = transform.get_max_vals(omega=0.0, curr_dq=np.array([5.0, 3.0, 0.0, 0.0]))
    assert len(out) == 4


def test_get_max_vals_curr_peak_matches_analytical_for_pure_d1():
    """For pure i_d1 = I the peak phase current must equal I exactly."""
    transform = _build_default_transform()
    vec_curr_dq = np.array([7.5, 0.0, 0.0, 0.0])
    curr_peak, *_ = transform.get_max_vals(omega=0.0, curr_dq=vec_curr_dq)
    assert np.isclose(curr_peak, 7.5)


def test_harmonic_alignment_diff_returns_value_in_zero_pi():
    """Output must always lie in [0, pi]."""
    diff = Transform._harmonic_alignment_diff(0.5, 1.7, h=3)
    assert 0.0 <= diff <= np.pi


def test_harmonic_alignment_diff_handles_wraparound():
    """
    Two angles that are circularly close but straddle the 0/2*pi boundary
    must report a small distance, not a near-2*pi distance.
    """
    eps = 1e-4
    diff = Transform._harmonic_alignment_diff(-np.pi / 3 + eps, 2 * np.pi - eps, h=3)
    assert 0.0 <= diff <= np.pi
    assert diff < 1e-3


def test_harmonic_alignment_diff_perfectly_aligned_is_zero():
    # 3 * 0 + pi = pi == ang_h
    diff = Transform._harmonic_alignment_diff(0.0, np.pi, h=3)
    assert np.isclose(diff, 0.0)


def test_alignment_diff_returns_zero_for_low_dim():
    """Machines with dim < 4 (i.e., no harmonics above the fundamental) get 0.0."""
    transform = _build_default_transform()
    transform.dim = 2
    assert transform._alignment_diff_all_harmonics(np.array([1.0, 1.0])) == 0.0


def test_alignment_diff_picks_perfect_alignment():
    """
    For a 5-phase machine (dim=4), only h=3 is checked. With phi_1 = 0 and
    phi_3 = pi, we have 3*0 + pi = pi == phi_3, so the alignment diff is 0.
    """
    transform = _build_default_transform()
    vec_dq = np.array([1.0, 0.0, -1.0, 0.0])
    diff = transform._alignment_diff_all_harmonics(vec_dq)
    assert np.isclose(diff, 0.0)


def test_count_peaks_helper_zero_when_below_limit():
    transform = _build_default_transform()
    n = transform._count_peaks_helper(value=5.0, val_max=10.0, angle_diff=0.0, tol=1e-4)
    assert n == 0


def test_count_peaks_helper_one_when_at_limit_no_alignment():
    transform = _build_default_transform()
    n = transform._count_peaks_helper(value=10.0, val_max=10.0, angle_diff=1.0, tol=1e-4)
    assert n == 1


def test_count_peaks_helper_two_when_at_limit_and_aligned():
    transform = _build_default_transform()
    n = transform._count_peaks_helper(value=10.0, val_max=10.0, angle_diff=0.0, tol=1e-4)
    assert n == 2
