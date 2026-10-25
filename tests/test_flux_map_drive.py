import numpy as np
import pytest

from current_setpoints.models import im5_tesla_gen1_fluxmap
from current_setpoints.models.flux_map_drive import _DATA, FluxMapDrive


def test_fluxmap_reproduces_fem_torque():
    d = im5_tesla_gen1_fluxmap(curr_max=170.0)
    z = np.load(_DATA)
    err = np.array([d.torque(0.0, c) - t for c, t in zip(z["currents"], z["T_fem"])])
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_fluxmap_odd_symmetry_and_hull():
    d = im5_tesla_gen1_fluxmap()
    i = np.array([80.0, 120.0, 20.0, -10.0])
    assert np.allclose(d.flux(0.0, -i), -d.flux(0.0, i), atol=1e-9)
    assert d.torque(0.0, i) == pytest.approx(d.torque(0.0, -i), rel=1e-9)
    assert d.in_hull(i) is True and d.in_hull(np.array([300.0, 300.0, 0.0, 0.0])) is False
    assert np.all(d.hull_margins(np.array([80.0, 120.0, 0.0, 0.0])) >= -1e-9)


def test_fluxmap_iron_loss_and_none():
    d = im5_tesla_gen1_fluxmap(iron_loss="steinmetz")
    z = np.load(_DATA)
    pred = np.array([d.iron_loss_at(0.0, c) for c in z["currents"]])
    P = z["P_core"]
    r2 = 1.0 - np.sum((P - pred) ** 2) / np.sum((P - P.mean()) ** 2)
    assert r2 > 0.99
    with pytest.raises(ValueError):
        im5_tesla_gen1_fluxmap(iron_loss=None).iron_loss_at(0.0, np.array([80.0, 80.0, 0.0, 0.0]))


def test_fluxmap_differential_inductance_and_offset():
    d = im5_tesla_gen1_fluxmap()
    i = np.array([80.0, 120.0, 20.0, -10.0])
    L = d.inductance(0.0, i)
    assert L.shape == (4, 4)
    assert np.allclose(L, L.T, atol=1e-9)                         # Maxwell reciprocity
    # differential (not secant): psi(i+delta) ~ psi(i) + L @ delta locally
    delta = np.array([0.5, -0.3, 0.2, 0.1])
    lin = d.flux(0.0, i) + L @ delta
    assert np.allclose(d.flux(0.0, i + delta), lin, atol=1e-2)
    # split identity: psi == psi* + L @ i
    assert np.allclose(d.flux(0.0, i), d.flux_offset(0.0, i) + L @ i, atol=1e-9)


def test_fluxmap_from_npz_configurable():
    d = FluxMapDrive.from_npz(_DATA, n_phases=5, n_ppairs=3, R_s=0.02194121,
                              curr_max=170.0, volt_max=230.0, iron_loss="lut")
    assert d.n_phases == 5 and d.n_ppairs == 3 and d.iron_loss_model is not None
    d2 = im5_tesla_gen1_fluxmap(curr_max=170.0, iron_loss="lut")   # same data via the example
    i = np.array([80.0, 120.0, 0.0, 0.0])
    assert d.torque(0.0, i) == pytest.approx(d2.torque(0.0, i), rel=1e-9)


def _synthetic_pmsm_map():
    """Linear PMSM map psi = psi_pm + L i on the motoring quadrant (i_d1 <= 0,
    i_q1 >= 0), with a sample at i = 0 carrying the PM flux."""
    psi_pm = np.array([0.2, 0.0, -0.025, 0.0])
    L = np.diag([4e-3, 9e-3, 1.3e-3, 1.4e-3])
    g = np.meshgrid(np.arange(-50.0, 1, 10), np.arange(0.0, 36, 7), [-5.0, 0, 5], [-5.0, 0, 5], indexing="ij")
    cur = np.column_stack([a.ravel() for a in g])
    return cur, psi_pm + cur @ L.T, psi_pm, L


def _pmsm_fluxmap(cur, flux):
    return FluxMapDrive(cur, flux, n_phases=5, n_ppairs=2, R_s=0.5, curr_max=50.0, volt_max=180.0,
                        symmetry="none", anchor_origin=False, seed_mode="data")


def test_pmsm_fluxmap_matches_analytic_pmsm():
    from current_setpoints.models import PMSMDrive, PMSMParams

    cur, flux, psi_pm, L = _synthetic_pmsm_map()
    d = _pmsm_fluxmap(cur, flux)
    ref = PMSMDrive(PMSMParams(n_phases=5, n_ppairs=2, R_stat=0.5 * np.eye(4), L_stat=L,
                               flux_pm=psi_pm, curr_max=50.0, volt_max=180.0))
    i = np.array([-23.0, 17.0, 1.5, -2.5])                      # interior, off-grid
    assert d.torque(0.0, i) == pytest.approx(ref.torque(0.0, i), rel=1e-6)
    assert np.allclose(d.inductance(0.0, i), L, atol=1e-6)
    assert np.allclose(d.flux_offset(0.0, i), psi_pm, atol=1e-6)  # PM flux recovered


def test_pmsm_map_refuses_im_assumptions():
    cur, flux, _, _ = _synthetic_pmsm_map()
    with pytest.raises(ValueError, match="symmetry"):
        FluxMapDrive(cur, flux, n_phases=5, n_ppairs=2, R_s=0.5, curr_max=50.0, volt_max=180.0)
    with pytest.raises(ValueError, match="anchor_origin"):
        FluxMapDrive(cur, flux, n_phases=5, n_ppairs=2, R_s=0.5, curr_max=50.0, volt_max=180.0,
                     symmetry="none")


def test_data_seeds_stay_in_sampled_region():
    cur, flux, _, _ = _synthetic_pmsm_map()
    d = _pmsm_fluxmap(cur, flux)
    seeds = d.seeds()
    assert len(seeds) == 4
    assert all(s[0] <= 0.0 and s[1] >= 0.0 and d.in_hull(s) for s in seeds)
    assert d.torque(0.0, seeds[-1]) == pytest.approx(max(d.torque(0.0, c) for c in cur), rel=1e-6)
