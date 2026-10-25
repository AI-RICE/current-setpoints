import numpy as np
import pytest

from current_setpoints.models import ForwardModel, pmsm5_prototype_fluxmap
from current_setpoints.models.flux_map_drive import _DATA_PMSM5
from current_setpoints.optimization import HullConstrainedOptimizer


@pytest.fixture(scope="module")
def drive():
    return pmsm5_prototype_fluxmap()


def test_torque_at_samples_is_the_flux_formula(drive):
    z = np.load(_DATA_PMSM5)
    J = np.diag([1.0, 1.0, 3.0, 3.0]) @ np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    k = np.arange(0, len(z["currents"]), 37)
    for c, psi in zip(z["currents"][k], z["flux"][k]):
        assert drive.torque(0.0, c) == pytest.approx(5 * 2 / 2 * c @ J @ psi, abs=1e-6)


def test_flux_torque_vs_measured_shaft_torque(drive):
    """Documents the open ~4.9 % calibration gap; fails if the data file drifts."""
    z = np.load(_DATA_PMSM5)
    T = np.array([drive.torque(0.0, c) for c in z["currents"]])
    gain = np.polyfit(T, z["T_meas"], 1)[0]
    assert gain == pytest.approx(1.049, abs=0.01)
    r = z["T_meas"] - gain * T
    assert np.sqrt(np.mean(r**2)) < 1.0                      # Nm, after the single gain


def test_pm_flux_and_quadrant(drive):
    psi0 = drive.flux_offset(0.0, np.zeros(4))
    assert psi0[0] == pytest.approx(0.21, abs=0.01)
    assert drive.in_hull(np.array([-25.0, 20.0, 0.0, 0.0]))
    assert not drive.in_hull(np.array([-25.0, -20.0, 0.0, 0.0]))   # braking: not measured


def test_max_torque_stays_in_measured_region(drive):
    fwd = ForwardModel(drive, add_volt_0=True, n_theta=200)
    omega = 1000 * np.pi / 30 * drive.n_ppairs                      # 1000 rpm, electrical
    s = HullConstrainedOptimizer(fwd, opts={"disp": False, "ftol": 1e-8, "maxiter": 300}).maximize_torque(omega)
    assert s.success and np.all(drive.hull_margins(s.curr_dq) >= -1e-6)
    assert 40.0 < s.torque < 75.0
