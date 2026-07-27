import numpy as np
import pytest

from current_setpoints.models import im5_tesla_gen1_fluxmap
from current_setpoints.models.flux_map_drive import _DATA


def test_fluxmap_reproduces_fem_torque():
    d = im5_tesla_gen1_fluxmap(curr_max=170.0)
    data = np.loadtxt(_DATA, delimiter=",", skiprows=1)
    err = np.array([d.torque(0.0, r[0:4]) - r[8] for r in data])
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
    data = np.loadtxt(_DATA, delimiter=",", skiprows=1)
    pred = np.array([d.iron_loss_at(0.0, r[0:4]) for r in data])
    P = data[:, 9]
    r2 = 1.0 - np.sum((P - pred) ** 2) / np.sum((P - P.mean()) ** 2)
    assert r2 > 0.99
    with pytest.raises(ValueError):
        im5_tesla_gen1_fluxmap(iron_loss=None).iron_loss_at(0.0, np.array([80.0, 80.0, 0.0, 0.0]))
