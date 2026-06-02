"""
Motor Analysis Package
Exposes utilities for data loading, neural torque prediction, and visualization.
"""

from .load_data import load_aggregated_csv_data
from .neural import (
    NeuralTorquePredictor,
    load_neural_model,
    predict_torque_neural,
)
from .loss_models import (
    FULL_TERMS,
    IRON_LOSS_TERMS,
    SPEED_ONLY_TERMS,
    fit_parametric_loss,
    loss_model_rmse,
    parametric_loss_r2,
    predict_parametric_loss,
)
from .nn_utils import evaluate_model, prepare_fold_dataloaders, train_model
from .plot_config import PlotConfig
from .plotting import plot_global_performance, plot_grid_segments
from .uncertainty import (
    build_residual_test_set,
    input_noise_sensitivity,
    measurement_noise_floor,
)
