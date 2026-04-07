"""
Motor Analysis and Optimization Library.

This package provides a comprehensive suite of tools for:
- Data handling and grid processing (.data)
- Physics-informed neural torque modeling (.model)
- Efficiency and current control optimization (.optimization)
- Common math and coordinate transformation utilities (.utils)
"""

from . import data, model, optimization, utils

__all__ = ["data", "model", "optimization", "utils"]
