# TODO: (DONE) finish
"""
Motor Optimization and Mapping Package

This package provides tools for motor torque modeling, grid-based performance
mapping, and multi-start optimization under physical current and voltage limits.
"""

# 1. Core Optimization Engine
from .optimizer import MotorOptimizer

# 2. Torque Model Hierarchy
from .models import BaseTorqueModel, ModelAnalytical, ModelNeural

# 3. Grid Calculation and Data Handling
from .grid import calculate_grid, grid_to_data

# 4. Physical Constraints (Useful for custom solvers)
from .constraints import current_constraint, voltage_constraint

# Define what is available when someone uses 'from package import *'
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
