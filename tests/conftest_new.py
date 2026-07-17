"""Shared fixtures for current_setpoints tests."""
from __future__ import annotations

import sys
import os
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from current_setpoints.models.machines import PMSMDrive, InductionDrive, ieee_machine2, im9_prototype
from current_setpoints.models.forward_model import Fault, ForwardModel
from current_setpoints.optimization.optimizer import StaticOptimizer

CURR_MAX = 30.0
VOLT_MAX = 13.0

SLSQP_OPTS = {"disp": False, "ftol": 1e-9, "maxiter": 500}

OMEGA_LOW  = 300.0  * (np.pi / 30) * 8   # electrical rad/s
OMEGA_MID  = 900.0  * (np.pi / 30) * 8
OMEGA_HIGH = 1500.0 * (np.pi / 30) * 8


@pytest.fixture(scope="session")
def pmsm() -> PMSMDrive:
    return ieee_machine2(curr_max=CURR_MAX, volt_max=VOLT_MAX)


@pytest.fixture(scope="session")
def im() -> InductionDrive:
    return im9_prototype(curr_max=20.0, volt_max=200.0)


@pytest.fixture(scope="session")
def fwd(pmsm) -> ForwardModel:
    return ForwardModel(pmsm, n_theta=700)


@pytest.fixture(scope="session")
def fwd_f1(pmsm) -> ForwardModel:
    return ForwardModel(pmsm, n_theta=700, fault=Fault((0,)))


@pytest.fixture(scope="session")
def fwd_f2(pmsm) -> ForwardModel:
    return ForwardModel(pmsm, n_theta=700, fault=Fault((0, 2)))


@pytest.fixture(scope="session")
def static_opt(fwd) -> StaticOptimizer:
    return StaticOptimizer(fwd, opts=SLSQP_OPTS)


@pytest.fixture(scope="session")
def static_opt_f1(fwd_f1) -> StaticOptimizer:
    return StaticOptimizer(fwd_f1, opts=SLSQP_OPTS)
