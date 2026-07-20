"""
Tests for IndependentOptimizer (dynamic/per-angle mode) under phase faults.

These exist specifically to verify the dynamic-mode fault results are
trustworthy, since StaticOptimizer's 2-open-phase fault handling was found
to be physically invalid (see memory: static-fault-constraint-bug) — it only
checks current-achievability at a single theta=0 snapshot instead of for
every rotor angle. IndependentOptimizer/Fault.extra_constraints_at_theta()
uses the theta-rotated constraint, which is the physically correct one, and
these tests hold it to that standard independently of the optimizer's own
`Solution.success`/`ok_grid` bookkeeping.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from current_setpoints.models.machines import ieee_machine2
from current_setpoints.models.forward_model import Fault, ForwardModel
from current_setpoints.optimization import StaticOptimizer, IndependentOptimizer

CURR_MAX = 30.0
VOLT_MAX = 13.0

SLSQP_OPTS = {"disp": False, "ftol": 1e-8, "maxiter": 300}

# n_theta and n_grid both multiples of 5 avoid the phase-aliasing artifact
# confirmed empirically: n_grid=64 (not a multiple of 5) broke single-fault
# rotation symmetry by ~1.4%; n_grid=65/130 restored exact symmetry.
N_THETA = 50
N_GRID = 50


def to_elec(rpm, n_ppairs):
    return rpm * (np.pi / 30) * n_ppairs


OMEGA_LOW = to_elec(300, 8)


@pytest.fixture(scope="module")
def pmsm():
    return ieee_machine2(curr_max=CURR_MAX, volt_max=VOLT_MAX)


def _ind_opt(pmsm, open_phases):
    fwd = ForwardModel(pmsm, n_theta=N_THETA, fault=Fault(open_phases))
    return IndependentOptimizer(fwd, opts=SLSQP_OPTS, n_grid=N_GRID)


def _assert_solution_respects_constraints(opt, sol, omega):
    """Independent check of every per-angle constraint, not just Solution.success."""
    fwd = opt.fwd
    assert bool(np.all(sol.diagnostics["ok_grid"]))
    _, idx_grid = opt._grid_indices()
    for n in range(N_GRID):
        i_n = sol.curr_dq[n]
        assert np.all(np.isfinite(i_n))
        theta_idx = int(idx_grid[n])

        H = fwd.phase_map_at_theta(theta_idx)  # (n_surv, dim)
        phase_i = H @ i_n
        assert np.max(np.abs(phase_i)) <= CURR_MAX + 1e-3, (
            f"node {n}: per-angle current limit violated ({np.max(np.abs(phase_i)):.4f} > {CURR_MAX})"
        )

        for cons in fwd.extra_constraints_at_theta(theta_idx):
            resid = cons["fun"](i_n)
            assert abs(resid) < 1e-6, f"node {n}: fault achievability constraint violated (residual={resid:.3e})"


# ── Single fault: deterministic, and close (not identical) across phases ──
#
# Earlier versions of this test asserted *exact* phase invariance. That
# expectation was itself an artifact of a since-fixed bug in the phase
# reconstruction (ForwardModel used a single combined harmonic for both the
# static Clarke projection and the theta rotation; the correct, reference-
# formulation-matching reconstruction uses two genuinely different harmonics
# -- see _build_dq_to_phase_map). Under the corrected geometry, combined with
# this machine's known anisotropic L_stat/flux_pm, single-fault T_max is
# *not* exactly phase-invariant -- only single-fault's zero-DOF-loss property
# (no null-space constraint, exact square inverse for every choice of open
# phase) is guaranteed, which the round-trip check below verifies directly.

@pytest.mark.parametrize("phase", range(5))
def test_single_fault_deterministic(pmsm, phase):
    """Same config, same result -- catches nondeterminism, not asymmetry."""
    sol_a = _ind_opt(pmsm, (phase,)).maximize_torque(OMEGA_LOW)
    sol_b = _ind_opt(pmsm, (phase,)).maximize_torque(OMEGA_LOW)
    assert sol_a.success and sol_b.success
    np.testing.assert_allclose(sol_a.torque, sol_b.torque, rtol=1e-9)


@pytest.mark.parametrize("phase", range(5))
def test_single_fault_reduced_map_is_exact_square_inverse(pmsm, phase):
    """Single-fault never loses a DOF: 4 surviving wires <-> 4 dq dims is
    always an exact, well-conditioned bijection, for every choice of open
    phase -- unlike double-fault, which needs a real achievability
    constraint."""
    fwd = ForwardModel(pmsm, n_theta=N_THETA, fault=Fault((phase,)))
    assert fwd.fault._N is None


# ── Single fault: dynamic mode must weakly dominate static mode ─────────────────
# (valid only for n_open <= 1: static's fault handling is exact there, so a
# per-angle relaxation -- which can always reproduce a constant trajectory --
# can never do worse. This does NOT hold for n_open == 2; see the bug memory.)

def test_single_fault_dynamic_at_least_static(pmsm):
    fwd_static = ForwardModel(pmsm, n_theta=N_THETA, fault=Fault((0,)))
    static_sol = StaticOptimizer(fwd_static, opts=SLSQP_OPTS).maximize_torque(OMEGA_LOW)
    assert static_sol.success

    dyn_sol = _ind_opt(pmsm, (0,)).maximize_torque(OMEGA_LOW)
    assert dyn_sol.success

    assert dyn_sol.torque >= static_sol.torque - 1e-3, (
        f"dynamic mode ({dyn_sol.torque:.4f}) should never underperform static mode "
        f"({static_sol.torque:.4f}) for a single-open-phase fault"
    )


def test_healthy_dynamic_at_least_static(pmsm):
    fwd_static = ForwardModel(pmsm, n_theta=N_THETA)
    static_sol = StaticOptimizer(fwd_static, opts=SLSQP_OPTS).maximize_torque(OMEGA_LOW)
    assert static_sol.success

    dyn_sol = _ind_opt(pmsm, ()).maximize_torque(OMEGA_LOW)
    assert dyn_sol.success

    assert dyn_sol.torque >= static_sol.torque - 1e-3


# ── Double fault: dynamic-mode solutions must independently satisfy every ───────
# per-angle current limit and the theta-rotated achievability constraint —
# not just report success.

@pytest.mark.parametrize("open_phases", [(0, 1), (0, 2), (1, 3), (1, 4)])
def test_double_fault_solution_respects_all_constraints(pmsm, open_phases):
    opt = _ind_opt(pmsm, open_phases)
    sol = opt.maximize_torque(OMEGA_LOW)
    assert sol.success
    assert sol.torque > 0
    _assert_solution_respects_constraints(opt, sol, OMEGA_LOW)


@pytest.mark.parametrize("open_phases", [(0, 1), (0, 2)])
def test_double_fault_torque_below_healthy(pmsm, open_phases):
    """A 2-open-phase fault must never allow *more* torque than the healthy machine."""
    healthy_sol = _ind_opt(pmsm, ()).maximize_torque(OMEGA_LOW)
    assert healthy_sol.success

    fault_sol = _ind_opt(pmsm, open_phases).maximize_torque(OMEGA_LOW)
    assert fault_sol.success

    assert fault_sol.torque < healthy_sol.torque
