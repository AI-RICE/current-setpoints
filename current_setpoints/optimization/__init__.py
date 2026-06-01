"""
Motor Optimization and Mapping Package

This package provides tools for motor torque modeling, grid-based performance
mapping, and multi-start optimization under physical current and voltage limits.
"""

from .constraints import current_constraint, voltage_constraint
from .grid import calculate_grid, get_correction_grid, grid_to_data
from .models import BaseTorqueModel, ModelAnalytical, ModelNeural
from .im_model import ModelIMAnalytical
from .im_grid import calculate_grid_im, calculate_grid_im_vdc_sweep
from .optimizer import MotorOptimizer
