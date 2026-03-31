"""
Motor Model Subpackage

This module provides the core data structures and coordinate transformation
logic required for electromagnetic motor analysis, including:
- MachineData: Containers for grid-based simulation results.
- Transform: Reference frame transformations (DQ to Phase) and voltage models.
"""

from .data import MachineData
from .transform import Transform

# The __all__ list explicitly defines the public API for this subpackage.
# It ensures that 'from model import *' only exposes these two classes.
__all__ = ["MachineData", "Transform"]
