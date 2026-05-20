"""
Motor Machine Models Module

This module provides the base and concrete implementations for different
motor machine topologies, including generic and IEEE-standardized models.
"""

from .flux import ConstantFlux, Flux, Flux_IEEEMachine2
from .machines import BaseMachine, IEEEMachine2
from .im_machine import BaseMachineIM, IM9Phase
from .im_coupling import k_ir_matrix, k_ir_block, slip_from_dq_foc
