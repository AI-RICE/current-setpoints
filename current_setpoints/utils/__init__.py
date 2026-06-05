"""
Motor Analysis Package
Exposes utilities for data loading, neural torque prediction, and visualization.
"""

from .load_data import load_aggregated_csv_data
from .loss_fit import (
    fit_substitution_loss,
    split_like_neural,
    substitution2_loss_features,
    substitution_loss_features,
    substitution_loss_rmse,
)
from .neural import (
    NeuralTorquePredictor,
    load_neural_model,
    predict_torque_neural,
)
from .nn_utils import evaluate_model, prepare_fold_dataloaders, train_model
from .plot_config import PlotConfig
from .plotting import plot_baseline_vs_compensated, plot_global_performance, plot_grid_segments
