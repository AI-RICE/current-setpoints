"""
Tests for calculate_grid, grid_to_data, MachineData, Waveforms, and evaluate.
All use current_setpoints exclusively.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch
from torch import nn

from current_setpoints.models.machines import ieee_machine2, im9_prototype, neural_pmsm5phase
from current_setpoints.models.forward_model import ForwardModel, Fault
from current_setpoints.optimization.optimizer import StaticOptimizer
from current_setpoints.optimization.grid import calculate_grid, get_correction_grid
from current_setpoints.optimization.data import (
    MachineData, Waveforms, evaluate, grid_to_data
)

SLSQP_OPTS = {"disp": False, "ftol": 1e-8, "maxiter": 300}


@pytest.fixture(scope="module")
def pmsm():
    return ieee_machine2(curr_max=30.0, volt_max=13.0, omega_max=1800)


@pytest.fixture(scope="module")
def im():
    return im9_prototype(curr_max=20.0, volt_max=200.0, omega_max=1500)


@pytest.fixture(scope="module")
def fwd(pmsm):
    return ForwardModel(pmsm, n_theta=700)


@pytest.fixture(scope="module")
def fwd_im(im):
    return ForwardModel(im, n_theta=700)


@pytest.fixture(scope="module")
def static_opt(fwd):
    return StaticOptimizer(fwd, opts=SLSQP_OPTS)


@pytest.fixture(scope="module")
def static_opt_im(fwd_im):
    return StaticOptimizer(fwd_im, opts=SLSQP_OPTS)


GRID_OPTS_SMALL = {
    "n_torq": 2,
    "n_omega": 2,
    "torq_min": 0.0,
    "omega_min": 0.0,
}


@pytest.fixture(scope="module")
def grid_pmsm(static_opt, fwd):
    return calculate_grid(static_opt, fwd, opts=GRID_OPTS_SMALL, mode="standard")


@pytest.fixture(scope="module")
def grid_im(static_opt_im, fwd_im):
    opts = {**GRID_OPTS_SMALL, "omega_probe": 50.0}
    return calculate_grid(static_opt_im, fwd_im, opts=opts, mode="standard")


# ── calculate_grid — required keys ───────────────────────────────────────────

REQUIRED_KEYS = {
    "const_mech_speed",
    "vec_torq",
    "vec_omega",
    "vec_torq_max",
    "curr_dq_grid",
    "grid_curr_peak",
    "grid_volt_peak",
    "grid_segments",
}


def test_grid_has_required_keys(grid_pmsm):
    for key in REQUIRED_KEYS:
        assert key in grid_pmsm, f"Missing key: {key}"


def test_grid_curr_dq_shape(grid_pmsm):
    # (dim=4, n_torq=2, n_omega=2)
    assert grid_pmsm["curr_dq_grid"].shape == (4, 2, 2)


def test_grid_curr_peak_shape(grid_pmsm):
    assert grid_pmsm["grid_curr_peak"].shape == (2, 2)


def test_grid_volt_peak_shape(grid_pmsm):
    assert grid_pmsm["grid_volt_peak"].shape == (2, 2)


def test_grid_segments_shape(grid_pmsm):
    assert grid_pmsm["grid_segments"].shape == (2, 2)


def test_grid_vec_torq_length(grid_pmsm):
    assert len(grid_pmsm["vec_torq"]) == 2


def test_grid_vec_omega_length(grid_pmsm):
    assert len(grid_pmsm["vec_omega"]) == 2


def test_grid_torq_max_shape(grid_pmsm):
    assert grid_pmsm["vec_torq_max"].shape == (1, 2)


def test_grid_curr_peak_within_limit(grid_pmsm):
    cp = grid_pmsm["grid_curr_peak"]
    valid = cp[~np.isnan(cp)]
    assert np.all(valid <= 30.0 + 1e-3), f"Current peak exceeds limit: {valid.max():.4f}"


def test_grid_volt_peak_within_limit(grid_pmsm):
    vp = grid_pmsm["grid_volt_peak"]
    valid = vp[~np.isnan(vp)]
    assert np.all(valid <= 13.0 + 1e-3), f"Voltage peak exceeds limit: {valid.max():.4f}"


def test_grid_segments_values(grid_pmsm):
    segs = grid_pmsm["grid_segments"]
    valid = segs[~np.isnan(segs)]
    assert np.all((valid >= 0) & (valid <= 8))


def test_grid_const_mech_speed_positive(grid_pmsm):
    assert grid_pmsm["const_mech_speed"] > 0


def test_grid_invalid_mode_raises(static_opt, fwd):
    with pytest.raises(ValueError):
        calculate_grid(static_opt, fwd, opts=GRID_OPTS_SMALL, mode="invalid")


# ── calculate_grid — 3×3 grid with more coverage ─────────────────────────────

def test_grid_3x3_shape(static_opt, fwd):
    opts = {"n_torq": 3, "n_omega": 3, "torq_min": 0.0, "omega_min": 0.0}
    grid = calculate_grid(static_opt, fwd, opts=opts, mode="standard")
    assert grid["curr_dq_grid"].shape == (4, 3, 3)
    assert grid["grid_curr_peak"].shape == (3, 3)


# ── calculate_grid — IM mode ──────────────────────────────────────────────────

def test_grid_im_has_required_keys(grid_im):
    for key in REQUIRED_KEYS:
        assert key in grid_im, f"Missing key: {key}"


def test_grid_im_curr_dq_shape(grid_im):
    assert grid_im["curr_dq_grid"].shape == (4, 2, 2)


# ── grid_to_data ──────────────────────────────────────────────────────────────

def test_grid_to_data_returns_machine_data(grid_pmsm):
    md = grid_to_data(grid_pmsm, k_skip=1)
    assert isinstance(md, MachineData)


def test_grid_to_data_omega_conversion(grid_pmsm):
    md = grid_to_data(grid_pmsm, k_skip=1)
    # omega in MachineData is mechanical RPM (converted from electrical rad/s in grid)
    assert np.all(md.omega_grid >= 0)


def test_grid_to_data_shapes(grid_pmsm):
    md = grid_to_data(grid_pmsm, k_skip=1)
    # torq_grid and omega_grid are 2D meshgrids
    assert md.torq_grid.ndim == 2
    assert md.omega_grid.ndim == 2
    assert md.curr_dq_grid.ndim == 3


def test_grid_to_data_curr_dq_shape(grid_pmsm):
    md = grid_to_data(grid_pmsm, k_skip=1)
    assert md.curr_dq_grid.shape[0] == 4  # dim


# ── MachineData ───────────────────────────────────────────────────────────────

def test_machine_data_unique_segments(grid_pmsm):
    md = grid_to_data(grid_pmsm, k_skip=1)
    segs = md.unique_segments
    assert isinstance(segs, np.ndarray)


def test_machine_data_select_k(grid_pmsm):
    md = grid_to_data(grid_pmsm, k_skip=1)
    n_torq = md.torq.size
    n_omega = md.omega.size
    # select_k mutates in place and returns None; k_skip=1 is a no-op
    result = md.select_k(1)
    assert result is None
    assert md.torq.size == n_torq
    assert md.omega.size == n_omega


# ── evaluate (Waveforms) ──────────────────────────────────────────────────────

def test_evaluate_returns_waveforms(fwd):
    curr = np.array([0.0, 15.0, 0.0, 0.0])
    w = evaluate(fwd, 300.0, curr)
    assert isinstance(w, Waveforms)


def test_evaluate_waveforms_shapes(fwd):
    curr = np.array([0.0, 15.0, 0.0, 0.0])
    w = evaluate(fwd, 300.0, curr)
    n_t = fwd.vec_theta.size  # n_theta+1
    assert w.theta.shape == (n_t,)
    assert w.curr_ph.shape == (5, n_t)
    assert w.volt_leg.shape == (5, n_t)
    assert w.volt_raw.shape == (5, n_t)
    assert w.volt_0.shape == (5, n_t)
    assert w.curr_dq.shape == (4,)
    assert w.volt_dq.shape == (4,)
    assert isinstance(w.curr_peak, float)
    assert isinstance(w.volt_peak, float)


def test_evaluate_peaks_match_forward_model(fwd):
    curr = np.array([0.0, 15.0, 0.0, 0.0])
    omega = 300.0
    w = evaluate(fwd, omega, curr)
    cp, vp = fwd.peak_vals(omega, curr)
    np.testing.assert_allclose(w.curr_peak, cp, rtol=1e-12)
    np.testing.assert_allclose(w.volt_peak, vp, rtol=1e-12)


def test_evaluate_volt_dq_consistent(fwd):
    curr = np.array([5.0, 15.0, 0.5, -1.0])
    omega = 400.0
    w = evaluate(fwd, omega, curr)
    expected_vdq = fwd.volt_dq(omega, curr)
    np.testing.assert_allclose(w.volt_dq, expected_vdq, rtol=1e-12)


# ── calculate_grid — neural drive + get_correction_grid + recalculated mode ──
# Ported from the old-layout test_grid.py (ModelNeural wrapper is gone in the
# new API: a neural_pmsm5phase() drive is used directly as StaticOptimizer's fwd.drive).

class DummyScaler:
    """A fake scaler that just returns the input exactly as it is."""

    def transform(self, X):
        return X


class DummyNet(nn.Module):
    """A fake neural network that outputs a dummy torque residual."""

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        return torch.sum(x_normed, dim=1, keepdim=True)


@pytest.fixture(scope="module")
def neural_pmsm():
    return neural_pmsm5phase(net=DummyNet(), scaler=DummyScaler(), device=torch.device("cpu"))


@pytest.fixture(scope="module")
def fwd_neural(neural_pmsm):
    return ForwardModel(neural_pmsm, n_theta=700)


@pytest.fixture(scope="module")
def static_opt_neural(fwd_neural):
    return StaticOptimizer(fwd_neural, opts=SLSQP_OPTS)


def test_calculate_grid_neural_runs_successfully(static_opt_neural, fwd_neural):
    opts = {**GRID_OPTS_SMALL, "torq_min": 7.5}
    grid = calculate_grid(static_opt_neural, fwd_neural, opts=opts, mode="standard")
    assert isinstance(grid, dict)
    assert grid["curr_dq_grid"].shape == (4, 2, 2)


def test_get_correction_grid_adds_neural_torque(neural_pmsm):
    dim, n_torq, n_omega = 4, 2, 2
    curr_dq_grid = np.zeros((dim, n_torq, n_omega))
    curr_dq_grid[:, 1, 1] = np.nan

    mock_baseline_grid = {
        "vec_torq": np.array([10.0, 20.0]),
        "vec_omega": np.array([100.0, 200.0]),
        "curr_dq_grid": curr_dq_grid,
        "const_mech_speed": 1.0,
    }

    corr_grid = get_correction_grid(dict_grid=mock_baseline_grid, neural=neural_pmsm)

    assert "grid_torq_neural" in corr_grid
    assert corr_grid["grid_torq_neural"].shape == (n_torq, n_omega)
    assert np.isnan(corr_grid["grid_torq_neural"][1, 1])
    assert not np.isnan(corr_grid["grid_torq_neural"][0, 0])


def test_calculate_grid_recalculated_runs_successfully(static_opt, fwd):
    dim, n_torq, n_omega = fwd.drive.dim, 2, 2
    mock_corr_grid = {
        "vec_torq": np.array([10.0, 20.0]),
        "vec_omega": np.array([1.0, 2.0]),
        "grid_torq_neural": np.array([[9.5, 9.0], [19.5, 19.0]]),
        "curr_dq_grid": np.zeros((dim, n_torq, n_omega)),
    }

    recalculated_grid = calculate_grid(
        optimizer=static_opt,
        fwd=fwd,
        opts={},
        mode="recalculated",
        dict_grid_corr=mock_corr_grid,
    )

    assert recalculated_grid is not None
    assert recalculated_grid["curr_dq_grid"].shape == (dim, n_torq, n_omega)
