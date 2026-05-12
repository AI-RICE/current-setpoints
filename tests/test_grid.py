from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn

from current_setpoints.optimization import (
    ModelAnalytical,
    ModelNeural,
    MotorOptimizer,
    calculate_grid,
    get_correction_grid,
    grid_to_data,
)
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform
from current_setpoints.utils import NeuralTorquePredictor


def test_calculate_grid_analytical_returns_correct_shapes():
    """
    Tests that calculate_grid processes a minimal 2x2 grid
    and returns a dictionary with the correct numpy array dimensions.
    """
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()

    transform = Transform(machine=machine, flux=flux, add_volt_0=False)
    analytical_model = ModelAnalytical(machine=machine, flux=flux)

    solver_opts = {"disp": False, "ftol": 1e-8, "maxiter": 50, "eps": 1e-8}
    optimizer = MotorOptimizer(model=analytical_model, opts=solver_opts)

    tiny_grid_opts = {"n_torq": 2, "n_omega": 2, "torq_min": 0.0, "omega_min": 0.0}

    grid_result = calculate_grid(optimizer=optimizer, transform=transform, opts=tiny_grid_opts, mode="standard")

    assert isinstance(grid_result, dict), "Result should be a dictionary"

    expected_keys = [
        "vec_torq",
        "vec_omega",
        "curr_dq_grid",
        "grid_curr_peak",
        "grid_volt_peak",
        "const_mech_speed",
    ]
    for key in expected_keys:
        assert key in grid_result, f"Missing expected key: {key}"

    assert grid_result["vec_torq"].shape == (2,)
    assert grid_result["vec_omega"].shape == (2,)
    assert grid_result["grid_curr_peak"].shape == (2, 2)

    dim = machine.n_phases - 1
    assert grid_result["curr_dq_grid"].shape == (dim, 2, 2)


def test_grid_to_data_calculates_omega_correctly():
    """
    Tests that grid_to_data correctly maps the dictionary
    and multiplies omega by const_mech_speed.
    """
    mock_grid = {
        "vec_torq": np.array([10.0, 20.0]),
        "vec_omega": np.array([1.0, 2.0]),
        "const_mech_speed": 5.0,  # We expect omega to be multiplied by this
        "grid_segments": np.zeros((2, 2)),
        "curr_dq_grid": np.zeros((2, 2, 2)),
    }

    result = grid_to_data(grid=mock_grid, k_skip=1)

    assert np.array_equal(result.omega, np.array([5.0, 10.0]))
    assert np.array_equal(result.torq, np.array([10.0, 20.0]))


class DummyScaler:
    """A fake scaler that just returns the input exactly as it is."""

    def transform(self, X):
        return X


class DummyNet(NeuralTorquePredictor):
    """A fake neural network that outputs a dummy torque."""

    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        return torch.sum(x_normed, dim=1, keepdim=True)


def test_calculate_grid_neural_runs_successfully():
    """
    Tests that calculate_grid can process a minimal 2x2 grid
    using the Neural model wrapper without crashing.
    """
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()

    transform = Transform(machine=machine, flux=flux, add_volt_0=False)

    device = torch.device("cpu")
    dummy_net = DummyNet()
    dummy_scaler = DummyScaler()

    neural_model_wrapper = ModelNeural(
        machine=machine, flux=flux, neural_model=dummy_net, scaler=dummy_scaler, device=device
    )

    solver_opts = {"disp": False, "ftol": 1e-5, "maxiter": 3, "eps": 1e-4}
    optimizer = MotorOptimizer(model=neural_model_wrapper, opts=solver_opts)

    tiny_grid_opts = {"n_torq": 2, "n_omega": 2, "torq_min": 7.5, "omega_min": 0.0}

    neural_grid = calculate_grid(optimizer=optimizer, transform=transform, opts=tiny_grid_opts, mode="standard")

    assert isinstance(neural_grid, dict)
    assert "curr_dq_grid" in neural_grid
    assert neural_grid["curr_dq_grid"].shape == (machine.n_phases - 1, 2, 2)


def test_get_correction_grid_adds_neural_torque():
    """
    Tests that get_correction_grid correctly flattens the arrays,
    ignores NaNs, runs the neural model, and appends the new matrix.
    """
    dim, n_torq, n_omega = 4, 2, 2

    curr_dq_grid = np.zeros((dim, n_torq, n_omega))
    curr_dq_grid[:, 1, 1] = np.nan

    mock_baseline_grid = {
        "vec_torq": np.array([10.0, 20.0]),
        "vec_omega": np.array([100.0, 200.0]),
        "curr_dq_grid": curr_dq_grid,
    }

    dummy_net = DummyNet()
    dummy_scaler = DummyScaler()
    device = torch.device("cpu")

    mock_flux = MagicMock()
    mock_flux.get_flux.return_value = (np.zeros(dim), np.zeros(dim))

    real_machine = IEEEMachine2()
    real_machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)

    # Wrap the components in the unified ModelNeural class
    neural_wrapper = ModelNeural(
        machine=real_machine,
        flux=mock_flux,
        neural_model=dummy_net,
        scaler=dummy_scaler,
        device=device,
    )

    corr_grid = get_correction_grid(
        dict_grid=mock_baseline_grid,
        model=neural_wrapper,
    )

    assert "grid_torq_neural" in corr_grid, "Should append the new torque matrix"
    assert corr_grid["grid_torq_neural"].shape == (n_torq, n_omega)

    assert np.isnan(corr_grid["grid_torq_neural"][1, 1])
    assert not np.isnan(corr_grid["grid_torq_neural"][0, 0])


def test_calculate_grid_recalculated_runs_successfully():
    """
    Tests that calculate_grid can run in 'recalculated' mode by feeding
    it a correction grid with pre-computed neural torque targets.
    """
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()

    transform = Transform(machine=machine, flux=flux, add_volt_0=False)
    analytical_model = ModelAnalytical(machine=machine, flux=flux)

    solver_opts = {"disp": False, "ftol": 1e-5, "maxiter": 3, "eps": 1e-4}
    optimizer = MotorOptimizer(model=analytical_model, opts=solver_opts)

    dim, n_torq, n_omega = machine.n_phases - 1, 2, 2
    mock_corr_grid = {
        "vec_torq": np.array([10.0, 20.0]),
        "vec_omega": np.array([1.0, 2.0]),
        "grid_torq_neural": np.array([[9.5, 9.0], [19.5, 19.0]]),
        "curr_dq_grid": np.zeros((dim, n_torq, n_omega)),
    }

    opts = {}

    recalculated_grid = calculate_grid(
        optimizer=optimizer,
        transform=transform,
        opts=opts,
        mode="recalculated",
        dict_grid_corr=mock_corr_grid,
    )

    assert recalculated_grid is not None
    assert recalculated_grid["curr_dq_grid"].shape == (dim, n_torq, n_omega)
