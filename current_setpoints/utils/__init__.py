"""
Motor Analysis Package
Exposes utilities for data loading, neural torque prediction, and visualization.
"""

from .load_data import load_aggregated_csv_data
from .neural import (
    NeuralTorquePredictor,
    load_neural_model,
    predict_torque_neural,
    torq_analytical,
)
from .nn_utils import evaluate_model, prepare_fold_dataloaders, train_model
from .plot_config import PlotConfig
from .plotting import plot_global_performance, plot_grid_segments

__all__ = [
    "load_aggregated_csv_data",
    "NeuralTorquePredictor",
    "load_neural_model",
    "predict_torque_neural",
    "torq_analytical",
    "train_model",
    "evaluate_model",
    "prepare_fold_dataloaders",
    "PlotConfig",
    "plot_grid_segments",
    "plot_global_performance",
]
