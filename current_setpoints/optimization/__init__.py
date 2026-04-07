"""
Motor Optimization and Mapping Package

This package provides tools for motor torque modeling, grid-based performance
mapping, and multi-start optimization under physical current and voltage limits.
"""

from .optimizer import MotorOptimizer
from .models import BaseTorqueModel, ModelAnalytical, ModelNeural
from .grid import calculate_grid, grid_to_data
from .constraints import current_constraint, voltage_constraint

__all__ = [
    "MotorOptimizer",
    "BaseTorqueModel",
    "ModelAnalytical",
    "ModelNeural",
    "calculate_grid",
    "grid_to_data",
    "current_constraint",
    "voltage_constraint",
]
