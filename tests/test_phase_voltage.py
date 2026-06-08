"""
Regression tests for the unified per-phase voltage model
(``dynamic.phase_voltage``).

These pin the convention that diverged between experiments and produced the
spurious high-speed fault envelope: per-phase voltage is the inverse-Park of
v_dq (fault-independent), and every code path that computes per-phase
voltage must agree on the same trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform
from dynamic import phase_voltage as pv


def _transform() -> Transform:
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    return Transform(machine=machine, flux=Flux_IEEEMachine2(), add_volt_0=False, n_theta=700)


def test_unified_matches_get_volt_ph_at_constant_dq():
    """At di/dtheta=0 the unified phase-A voltage equals the validated get_volt_ph."""
    tr = _transform()
    omega = 900.0 * (np.pi / 30.0) * tr.machine.n_ppairs
    i_const = np.array([8.0, 20.0, 1.5, -2.0])
    n_theta = tr.vec_theta.size - 1
    idx = np.arange(n_theta)
    i_dq = np.tile(i_const, (n_theta, 1))
    di_dq = np.zeros_like(i_dq)
    vA = pv.phase_voltage(tr, omega, i_dq, di_dq, idx, phases=(0,))[0]
    v_ref, _, _ = tr.get_volt_ph(omega, i_const)
    assert np.allclose(vA, v_ref[:n_theta], atol=1e-9)


def test_linear_maps_match_direct_evaluation():
    """voltage_linear_maps reconstruct exactly what phase_voltage evaluates.

    This is the regression that would have caught the original bug: two code
    paths (linear constraint maps vs direct evaluation) must agree bit-for-bit
    on the same trajectory and the same phases.
    """
    tr = _transform()
    omega = 1100.0 * (np.pi / 30.0) * tr.machine.n_ppairs
    n_theta = tr.vec_theta.size - 1
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n_theta, size=64, replace=False))
    # a non-trivial trajectory with harmonic content (so di/dtheta != 0)
    theta = tr.vec_theta[idx]
    i_dq = np.stack(
        [6 + 2 * np.cos(theta), 18 + 3 * np.sin(2 * theta), np.cos(3 * theta), -np.sin(theta)], axis=1
    )
    di_dq = np.stack(
        [-2 * np.sin(theta), 6 * np.cos(2 * theta), -3 * np.sin(3 * theta), -np.cos(theta)], axis=1
    )
    phases = (1, 2, 3, 4)
    v_direct = pv.phase_voltage(tr, omega, i_dq, di_dq, idx, phases=phases)  # (4, n)
    gU, gL, bV = pv.voltage_linear_maps(tr, omega, idx, phases=phases)
    v_maps = np.einsum("pnj,nj->pn", gU, i_dq) + omega * np.einsum("pnj,nj->pn", gL, di_dq) + bV
    assert np.allclose(v_direct, v_maps, atol=1e-12)


def test_open_phase_current_needs_reduced_clarke():
    """Inverse-Park puts nonzero current in the would-be open phase: hence the
    CURRENT realisation under fault cannot use inverse-Park (it must use the
    reduced-Clarke map), while VOLTAGE does use inverse-Park. The asymmetry is
    physical, not a free convention choice."""
    tr = _transform()
    n_theta = tr.vec_theta.size - 1
    idx = np.arange(0, n_theta, 7)
    i_const = np.array([8.0, 20.0, 1.5, -2.0])
    i_dq = np.tile(i_const, (idx.size, 1))
    P0 = pv.phase_basis(tr, idx, phases=(0,))[0]  # (n, dim)
    i0 = P0 @ i_const if False else np.einsum("nj,j->n", P0, i_const)
    assert np.max(np.abs(i0)) > 1.0  # decisively nonzero


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
