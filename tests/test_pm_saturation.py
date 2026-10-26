import numpy as np
import pytest

from current_setpoints.models import PMSMDrive, PMSMParams, SaturatingPMFlux
from current_setpoints.models.pm_saturation import P0, grad_w

P = {**P0, "K": 0.01, "c_d": 50.0, "c_q": -30.0, "theta0": 0.03, "k_xx": 0.3, "k_yy": -0.2, "k_xy": 0.4}
RNG = np.random.default_rng(0)
PSI = np.column_stack([RNG.uniform(0.02, 0.3, 40), RNG.uniform(-0.3, 0.3, 40),
                       RNG.uniform(-0.05, 0.03, 40), RNG.uniform(-0.05, 0.03, 40)])


def test_currents_are_an_energy_gradient():
    """Reciprocity by construction: the raw (unsymmetrised) Hessian of W is symmetric."""
    h = 1e-7
    H = np.empty((len(PSI), 4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        H[:, :, j] = (grad_w(P, PSI + e) - grad_w(P, PSI - e)) / (2 * h)
    assert np.abs(H - np.transpose(H, (0, 2, 1))).max() < 1e-8 * np.abs(H).max()


def test_forward_inverse_round_trip():
    m = SaturatingPMFlux(P)
    assert np.allclose(m.total_flux(m.current(PSI)), PSI, atol=1e-10)


def test_mirror_symmetry_about_d_axis():
    m = SaturatingPMFlux({**P, "theta0": 0.0, "theta3": 0.0})
    i = np.array([[-20.0, 15.0, 3.0, -4.0]])
    mirror = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.allclose(m.total_flux(i * mirror), m.total_flux(i) * mirror, atol=1e-10)


def test_pmsmdrive_composition_is_exact():
    m = SaturatingPMFlux(P)
    params = PMSMParams(n_phases=5, n_ppairs=2, R_stat=0.6 * np.eye(4), L_stat=np.eye(4) * 1e-3,
                        flux_pm=np.zeros(4), curr_max=50.0, volt_max=180.0)
    d = PMSMDrive(params, flux=m)
    i, w = np.array([-23.0, 17.0, 2.5, -6.0]), 400.0
    psi = m.total_flux(i[None, :])[0]
    J = m.cross_coupling
    assert d.torque(w, i) == pytest.approx(5 * 2 / 2 * i @ J @ psi, rel=1e-9)
    v = d.voltage_operator(w, i) @ i + d.bemf_dq(w, i)
    assert np.allclose(v, 0.6 * i + w * J @ psi, atol=1e-9)
    L = d.inductance(w, i)
    assert np.allclose(L, L.T) and np.all(np.linalg.eigvalsh(L) > 0)
