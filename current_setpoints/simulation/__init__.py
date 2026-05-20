"""
Motor Model Subpackage

This module provides the core data structures and coordinate transformation
logic required for electromagnetic motor analysis, including:
- MachineData: Containers for grid-based simulation results.
- Transform: Reference frame transformations (DQ to Phase) and voltage models.
"""

from .data import MachineData
from .transform import Transform
from .im_transform import IMTransform
