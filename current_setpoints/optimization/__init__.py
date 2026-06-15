"""
Motor Optimization and Mapping Package

This package provides tools for motor torque modeling, grid-based performance
mapping, and multi-start optimization under physical current and voltage limits.
"""

from .constraints import current_constraint, voltage_constraint
from .grid import calculate_grid, get_correction_grid, grid_to_data
from .models import (
    BaseTorqueModel,
    ModelAnalytical,
    ModelLossesSubstitution1,
    ModelLossesSubstitution1Excess,
    ModelLossParametric,
    ModelLossSubstitution2,
    ModelLossSubstitution2Excess,
    ModelNeural,
)
from .optimizer import MotorOptimizer
