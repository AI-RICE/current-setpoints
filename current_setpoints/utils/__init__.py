"""
Motor Analysis Package
Exposes utilities for data loading, neural torque prediction, and visualization.
"""

from .load_data import load_aggregated_csv_data
from .loss_models import (
    EMPIRICAL_POLYNOMIAL_TERMS,
    FULL_TERMS,
    IRON_LOSS_TERMS,
    SPEED_ONLY_TERMS,
    fit_parametric_loss,
    loss_model_rmse,
    parametric_loss_r2,
    predict_parametric_loss,
)
from .neural import (
    NeuralTorquePredictor,
    load_neural_model,
    predict_torque_neural,
)
from .nn_utils import evaluate_model, prepare_fold_dataloaders, train_model
from .param_fit import (
    NTM_RANDOM_STATE,
    NTM_TEST_SIZE,
    core_loss_rmse,
    fit_core_loss_resistances,
    split_like_neural,
)
from .plot_config import PlotConfig
from .plotting import plot_baseline_vs_compensated, plot_global_performance, plot_grid_segments
from .uncertainty import (
    build_residual_test_set,
    input_noise_sensitivity,
    measurement_noise_floor,
)
