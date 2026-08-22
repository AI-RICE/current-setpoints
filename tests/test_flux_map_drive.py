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
