"""
Motor Machine Models Module

This module provides the base and concrete implementations for different 
motor machine topologies, including generic and IEEE-standardized models.
"""

# TODO: (DONE) finish
from .machines import BaseMachine, GenericMachine, IEEEMachine2

__all__ = [
    'BaseMachine',
    'GenericMachine',
    'IEEEMachine2'
]