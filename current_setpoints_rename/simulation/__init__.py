"""
Motor Model Subpackage

This module provides the core data structures and coordinate transformation
logic required for electromagnetic motor analysis, including:
- MachineData: Containers for grid-based simulation results.
- Transform family: Reference frame transformations (DQ to Phase) and voltage
  models. ``BaseTransform`` is the abstract foundation; the four concrete
  subclasses cover the healthy machine and the three open-phase fault modes
  (single, two adjacent, two non-adjacent).
"""

from .data import MachineData
from .transform import (
    BaseTransform,
    Transform,
    TransformFault1,
    TransformFault2Adjacent,
    TransformFault2NonAdjacent,
)
