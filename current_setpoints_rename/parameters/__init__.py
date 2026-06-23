"""
Motor Machine Models Module

This module provides the base and concrete implementations for different
motor machine topologies, including generic and IEEE-standardized models.
"""

from .flux import ConstantFlux, Flux, Flux_IEEEMachine2
from .machines import BaseMachine, IEEEMachine2
