import warnings

import pytest

from current_setpoints.models import im9_prototype
from current_setpoints.models.forward_model import ForwardModel
from current_setpoints.optimization import HullConstrainedOptimizer, StaticOptimizer

SLSQP = {"disp": False, "ftol": 1e-8, "maxiter": 300}


def _fwd():
    return ForwardModel(im9_prototype(), n_theta=180)


def test_hysteresis_option_stripped_and_solves():
    """opts['hysteresis'] drives warm-start continuity and must not reach scipy
    (no unknown-option warning); the solve still succeeds."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*[Uu]nknown solver option.*")
        s = StaticOptimizer(_fwd(), opts={**SLSQP, "hysteresis": 0.01}).maximize_torque(200.0)
    assert s.success


def test_hull_constrained_noop_without_hull_margins():
    """With a drive that has no hull (im9_prototype), HullConstrainedOptimizer
    reduces exactly to StaticOptimizer."""
    fwd = _fwd()
    assert not hasattr(fwd.drive, "hull_margins")
    a = StaticOptimizer(fwd, opts=SLSQP).maximize_torque(200.0)
    b = HullConstrainedOptimizer(fwd, opts=SLSQP).maximize_torque(200.0)
    assert b.success and b.torque == pytest.approx(a.torque, rel=1e-6)
