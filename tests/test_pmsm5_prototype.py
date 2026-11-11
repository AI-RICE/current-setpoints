import numpy as np
import pytest

from current_setpoints.models import ForwardModel, pmsm5_prototype_fluxmap, pmsm5_prototype_inverter
from current_setpoints.models.flux_map_drive import _DATA_DIR, _PMSM5_CAMPAIGNS
from current_setpoints.optimization import HullConstrainedOptimizer

# campaign -> (shaft/flux torque gain stored in T_meas, number of setpoints, limits)
EXPECT = {
    "2026-10": dict(gain=1.049, n=959, curr_max=50.0, volt_max=180.0),   # open 4.9 % calibration gap
    "2026-11": dict(gain=1.000, n=1516, curr_max=45.0, volt_max=175.0),  # T_meas recalibrated to the flux torque
}


@pytest.fixture(scope="module", params=sorted(EXPECT))
def case(request):
    c = request.param
    return c, pmsm5_prototype_fluxmap(campaign=c), np.load(_DATA_DIR / _PMSM5_CAMPAIGNS[c]["data"])


def test_default_is_the_second_campaign():
    d = pmsm5_prototype_fluxmap()
    assert (d.curr_max, d.volt_max) == (45.0, 175.0)


def test_torque_at_samples_is_the_flux_formula(case):
    _, drive, z = case
    J = np.diag([1.0, 1.0, 3.0, 3.0]) @ np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    k = np.arange(0, len(z["currents"]), 37)
    for c, psi in zip(z["currents"][k], z["flux"][k]):
        assert drive.torque(0.0, c) == pytest.approx(5 * 2 / 2 * c @ J @ psi, abs=1e-6)


def test_flux_torque_vs_shaft_torque(case):
    """fails if a data file drifts: Oct = open calibration gap, Nov = recalibrated"""
    c, drive, z = case
    assert len(z["currents"]) == EXPECT[c]["n"]
    assert (drive.curr_max, drive.volt_max) == (EXPECT[c]["curr_max"], EXPECT[c]["volt_max"])
    T = np.array([drive.torque(0.0, i) for i in z["currents"]])
    gain = np.polyfit(T, z["T_meas"], 1)[0]
    assert gain == pytest.approx(EXPECT[c]["gain"], abs=0.01)
    assert np.sqrt(np.mean((z["T_meas"] - gain * T) ** 2)) < 1.0   # Nm, after the single gain


def test_pm_flux_and_quadrant(case):
    _, drive, _ = case
    psi0 = drive.flux_offset(0.0, np.zeros(4))
    assert psi0[0] == pytest.approx(0.21, abs=0.01)
    assert drive.in_hull(np.array([-25.0, 20.0, 0.0, 0.0]))
    assert not drive.in_hull(np.array([-25.0, -20.0, 0.0, 0.0]))   # braking: not measured


def test_max_torque_stays_in_measured_region():
    drive = pmsm5_prototype_fluxmap()
    fwd = ForwardModel(drive, add_volt_0=False, n_theta=200, volt_drop=pmsm5_prototype_inverter())
    omega = 1000 * np.pi / 30 * drive.n_ppairs                      # 1000 rpm, electrical
    s = HullConstrainedOptimizer(fwd, opts={"disp": False, "ftol": 1e-8, "maxiter": 300}).maximize_torque(omega)
    assert s.success and np.all(drive.hull_margins(s.curr_dq) >= -1e-6)
    assert 40.0 < s.torque < 75.0


def test_inverter_drop_is_a_few_volts():
    fwd = ForwardModel(pmsm5_prototype_fluxmap(), n_theta=200, volt_drop=pmsm5_prototype_inverter())
    e = fwd.drop_dq(np.array([-10.0, 40.0, 0.0, 0.0]))
    assert 5.0 < np.linalg.norm(e[:2]) < 15.0                       # dead time ~7 V + R_sw i
