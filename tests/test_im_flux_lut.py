import numpy as np
import pytest

from current_setpoints.models import (
    CoreLossLUT,
    IMFluxLUT,
    SteinmetzIronLoss,
    im5_tesla_gen1_fluxlut,
)
from current_setpoints.models.im_flux_lut import _DATA


def test_fluxlut_identity_and_limits():
    m = im5_tesla_gen1_fluxlut()
    assert isinstance(m, IMFluxLUT)
    assert (m.n_phases, m.n_ppairs, m.dim, m.k_phase) == (5, 3, 4, 2.5)
    assert (m.curr_max, m.volt_max) == (200.0, 230.0)
    assert m.R_stat[0, 0] == pytest.approx(0.02194121, rel=1e-12)


def test_fluxlut_reproduces_fem_torque():
    """The whole point: torque from the interpolated flux map matches the
    357-point Ansys sweep to <1 Nm RMS (co-energy formula, p_p = 3)."""
    m = im5_tesla_gen1_fluxlut()
    d = np.load(_DATA)["raw"]
    err = np.array([m.torque(0.0, row[0:4]) - row[8] for row in d])
    assert np.sqrt(np.mean(err**2)) < 1.0
    assert np.abs(err).max() < 3.0


def test_fluxlut_flux_is_returned_and_interpolates():
    m = im5_tesla_gen1_fluxlut()
    # at a sampled node the flux equals the FEM value
    d = np.load(_DATA)["raw"]
    node = d[100]
    assert m.flux(0.0, node[0:4]) == pytest.approx(node[4:8], rel=1e-6, abs=1e-9)
    # an interior midpoint interpolates to something finite and in range
    psi = m.flux(0.0, np.array([100.0, 100.0, 0.0, 0.0]))
    assert psi.shape == (4,) and np.all(np.isfinite(psi))


def test_fluxlut_hull_flagging():
    m = im5_tesla_gen1_fluxlut()
    assert m.in_hull(np.array([120.0, 150.0, 20.0, -10.0])) is True
    assert m.in_hull(np.array([300.0, 300.0, 0.0, 0.0])) is False
    # outside the hull flux still returns finite (nearest-neighbour) values
    assert np.all(np.isfinite(m.flux(0.0, np.array([300.0, 300.0, 0.0, 0.0]))))


def test_fluxlut_voltage_from_flux():
    """v = R_s*i + omega * J . psi ; bemf carries the flux-induced part."""
    m = im5_tesla_gen1_fluxlut()
    i = np.array([80.0, 120.0, 0.0, 0.0])
    omega = 500.0
    psi = m.flux(0.0, i)
    v = m.voltage_operator(omega, i) @ i + m.bemf_dq(omega, i)
    expect = m.R_stat @ i + omega * m._J @ psi
    assert v == pytest.approx(expect, rel=1e-9)
    # h=3 rotates at 3x: bemf_q1 uses h=1, bemf on plane 3 uses h=3
    assert m.bemf_dq(omega, i)[1] == pytest.approx(omega * 1 * psi[0], rel=1e-9)


def test_fluxlut_agrees_with_analytic_at_low_current():
    """Where the analytic EC is trustworthy (least-saturated, rated q-current),
    the two models agree on torque to a few percent."""
    from current_setpoints.models import im5_tesla_gen1

    lut = im5_tesla_gen1_fluxlut()
    ana = im5_tesla_gen1()
    i = np.array([40.0, 200.0, 0.0, 0.0])
    assert lut.torque(0.0, i) == pytest.approx(ana.torque(0.0, i), rel=0.05)


def test_iron_loss_steinmetz_fits_fem_column():
    """Steinmetz model P = k1|psi1|^2 + k3|psi3|^2 reproduces the FEM CoreLoss
    column to R^2 > 0.99 (the physical B^2 validation)."""
    m = im5_tesla_gen1_fluxlut(iron_loss="steinmetz")
    d = np.load(_DATA)["raw"]
    pred = np.array([m.iron_loss_at(0.0, row[0:4]) for row in d])
    P = d[:, 9]
    r2 = 1.0 - np.sum((P - pred) ** 2) / np.sum((P - P.mean()) ** 2)
    assert r2 > 0.99
    assert np.sqrt(np.mean((pred - P) ** 2)) < 25.0  # ~19 W


def test_iron_loss_lut_reproduces_nodes():
    m = im5_tesla_gen1_fluxlut(iron_loss="lut")
    d = np.load(_DATA)["raw"]
    # at a sampled node the LUT returns the FEM CoreLoss value
    node = d[123]
    assert m.iron_loss_at(0.0, node[0:4]) == pytest.approx(node[9], rel=1e-6)
    assert m.iron_loss_at(0.0, node[0:4]) > 0.0


def test_iron_loss_scales_with_flux_squared():
    """Both terms grow with flux; more magnetizing current -> more iron loss."""
    m = im5_tesla_gen1_fluxlut(iron_loss="steinmetz")
    lo = m.iron_loss_at(0.0, np.array([40.0, 40.0, 0.0, 0.0]))
    hi = m.iron_loss_at(0.0, np.array([160.0, 40.0, 0.0, 0.0]))
    assert hi > lo > 0.0


def test_iron_loss_none_raises():
    m = im5_tesla_gen1_fluxlut(iron_loss=None)
    with pytest.raises(ValueError):
        m.iron_loss_at(0.0, np.array([80.0, 80.0, 0.0, 0.0]))


def test_flux_mlp_backend_matches_fem_torque():
    """The committed MLP surrogate (numpy inference) reproduces FEM torque to
    <1 Nm RMS, on par with the raw interpolant but C-infinity smooth."""
    m = im5_tesla_gen1_fluxlut(flux_backend="mlp")
    d = np.load(_DATA)["raw"]
    err = np.array([m.torque(0.0, row[0:4]) - row[8] for row in d])
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_flux_odd_symmetry_and_canonicalisation():
    """psi(-i) = -psi(i); torque invariant under i -> -i; and a mirror
    (negative-Id1) point reports in-hull iff its canonical image does."""
    for backend in ("linear", "mlp"):
        m = im5_tesla_gen1_fluxlut(flux_backend=backend)
        i = np.array([80.0, 120.0, 20.0, -10.0])
        assert np.allclose(m.flux(0.0, -i), -m.flux(0.0, i), atol=1e-9)
        assert m.torque(0.0, i) == pytest.approx(m.torque(0.0, -i), rel=1e-9)
        assert m.in_hull(-i) == m.in_hull(i)


def test_flux_backend_default_is_linear():
    assert im5_tesla_gen1_fluxlut().flux_backend == "linear"


def test_hull_margins_sign():
    m = im5_tesla_gen1_fluxlut()
    assert np.all(m.hull_margins(np.array([80.0, 120.0, 0.0, 0.0])) >= -1e-9)  # interior
    assert np.any(m.hull_margins(np.array([300.0, 300.0, 0.0, 0.0])) < 0.0)  # exterior


def test_hull_constrained_optimizer_stays_in_envelope():
    """The hull constraint confines the T_max probe to the sampled envelope:
    the solution is in-hull and its torque is bounded by the FEM envelope
    (~145 Nm), not the extrapolated 162 Nm the unconstrained solve can reach."""
    from current_setpoints.models.forward_model import ForwardModel
    from current_setpoints.optimization.optimizer import HullConstrainedOptimizer

    d = im5_tesla_gen1_fluxlut(flux_backend="mlp")
    d.set_max_pars(200.0, 230.0, 12000.0)
    fwd = ForwardModel(d, n_theta=180)
    om = 5000.0 / (30.0 / (np.pi * d.n_ppairs))  # 5000 mech rpm -> electrical rad/s
    hull = HullConstrainedOptimizer(fwd, opts={"ftol": 1e-9, "maxiter": 500}).maximize_torque(om)
    assert hull.success
    # the hull inequality is satisfied (solution may sit on a facet, where the
    # strict Delaunay `in_hull` returns False, so test the margins with tol)
    assert np.all(d.hull_margins(hull.curr_dq) >= -1e-6)
    assert 0.0 < hull.torque <= 150.0


def test_steinmetz_speed_scaling_and_lut_guard():
    # Steinmetz with a reference speed scales ~ (omega/omega_ref)^2
    s = SteinmetzIronLoss(omega_ref=2000.0, freq_exp=2.0)
    flux = np.array([0.08, 0.03, 1e-3, 2e-3])
    p_ref = s.loss(2000.0, np.zeros(4), flux)
    p_2x = s.loss(4000.0, np.zeros(4), flux)
    assert p_2x == pytest.approx(4.0 * p_ref, rel=1e-9)
    # CoreLossLUT warns when queried far from its reference speed
    d = np.load(_DATA)["raw"]
    lut = CoreLossLUT(d[:, 0:4], d[:, 9], omega_ref=2000.0)
    with pytest.warns(UserWarning):
        lut.loss(5000.0, d[0, 0:4], d[0, 4:8])


def test_optimizer_hysteresis_option_runs_clean():
    """`hysteresis` in opts drives warm-start continuity and must be stripped
    before reaching scipy (no unknown-option warning), still returning a valid
    in-hull solution."""
    import warnings

    from current_setpoints.models.forward_model import ForwardModel
    from current_setpoints.optimization.optimizer import HullConstrainedOptimizer

    d = im5_tesla_gen1_fluxlut(flux_backend="mlp")
    d.set_max_pars(170.0, 230.0, 12000.0)
    fwd = ForwardModel(d, n_theta=180)
    om = 3000.0 / (30.0 / (np.pi * d.n_ppairs))
    opt = HullConstrainedOptimizer(fwd, opts={"ftol": 1e-9, "maxiter": 500, "hysteresis": 0.01})
    guess = np.array([80.0, 120.0, 10.0, -5.0])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*[Uu]nknown solver option.*")
        s = opt.minimize_current(80.0, om, guess=guess)
    assert s.success and np.all(d.hull_margins(s.curr_dq) >= -1e-6)
