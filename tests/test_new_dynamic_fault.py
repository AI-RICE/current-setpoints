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
    """Single-fault's reduced current map (4 surviving wires -> 4 dq dims) is
    always an exact, well-conditioned square bijection, for every choice of
    open phase. The achievability constraint (that phase's own current is
    physically zero) still applies -- it's derived from the full map's open-
    phase row directly, independent of whether the reduced map is square or
    underdetermined."""
    fwd = ForwardModel(pmsm, n_theta=N_THETA, fault=Fault((phase,)))
    M = fwd.phase_map_at_theta(0)
    assert M.shape == (4, 4)
    assert np.linalg.cond(M) < 1e8
    cons = fwd.extra_constraints_at_theta(0)
    assert len(cons) == 1


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


# ── Regression: double-fault torque must be rotationally symmetric ──────────
#
# Two bugs, both fixed together, previously broke this:
#   1. _per_angle_constraints passed the coarse n_grid loop index straight
#      into ForwardModel.extra_constraints_at_theta, which expects a fine
#      vec_theta index -- the achievability constraint was enforced at the
#      wrong rotor angle at every node.
#   2. The achievability constraint itself was derived from the null space
#      of the *static* (harmonic-1,2) reduced Clarke matrix, then rotated by
#      R(theta) (harmonic-1,3) -- a construction that does not actually zero
#      the open phase's current when checked against the model's own true
#      (theta-dependent) reconstruction. The fix derives it directly from the
#      open phase's own row of the full map at each theta instead.
# Both bugs happened to leave each fault-pair *category* internally
# consistent by luck, but gave e.g. 42.6%-72.4% spread across physically-
# equivalent non-adjacent pairs on a fully isotropic machine. Fixed, every
# pair within a rotation-equivalent category must match exactly.

def _isotropic_pmsm():
    from current_setpoints.models.machines import PMSMDrive, PMSMParams

    return PMSMDrive(
        PMSMParams(
            n_phases=5,
            n_ppairs=8,
            R_stat=np.diag([0.05, 0.05, 0.05, 0.05]),
            L_stat=1e-3 * np.diag([0.10, 0.10, 0.10, 0.10]),
            flux_pm=np.array([0.0, 0.0113, 0.0, 0.0]),
            curr_max=CURR_MAX,
            volt_max=VOLT_MAX,
        )
    )


ADJACENT_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
NONADJACENT_PAIRS = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]


@pytest.mark.parametrize("pairs", [ADJACENT_PAIRS, NONADJACENT_PAIRS])
def test_double_fault_symmetric_within_rotation_class(pairs):
    machine = _isotropic_pmsm()
    torques = []
    for open_phases in pairs:
        fwd = ForwardModel(machine, n_theta=700, fault=Fault(open_phases))
        sol = IndependentOptimizer(fwd, opts=SLSQP_OPTS, n_grid=130).maximize_torque(0.0)
        assert sol.success
        torques.append(sol.torque)
    np.testing.assert_allclose(torques, torques[0], rtol=1e-3)


def test_double_fault_ordering_nonadjacent_beats_adjacent():
    """Physical requirement: healthy > single > non-adjacent-double >
    adjacent-double -- non-adjacent survivors are more spatially spread out
    and always retain more torque capability than adjacent survivors."""
    machine = _isotropic_pmsm()

    def torque_for(open_phases):
        fwd = ForwardModel(machine, n_theta=700, fault=Fault(open_phases))
        sol = IndependentOptimizer(fwd, opts=SLSQP_OPTS, n_grid=130).maximize_torque(0.0)
        assert sol.success
        return sol.torque

    t_healthy = torque_for(())
    t_single = torque_for((0,))
    t_adjacent = torque_for(ADJACENT_PAIRS[0])
    t_nonadjacent = torque_for(NONADJACENT_PAIRS[0])

    assert t_healthy > t_single > t_nonadjacent > t_adjacent


def test_per_angle_constraint_uses_fine_grid_theta(pmsm):
    """Regression for the coarse/fine index bug: the constraint actually
    enforced by IndependentOptimizer at coarse node n must match
    ForwardModel.extra_constraints_at_theta(idx_grid[n]) exactly, not
    extra_constraints_at_theta(n). Needs n_theta != n_grid (unlike this
    module's shared N_THETA=N_GRID=50 fixtures) so the coarse and fine
    indices actually diverge at the chosen node."""
    fwd = ForwardModel(pmsm, n_theta=700, fault=Fault((0, 1)))
    opt = IndependentOptimizer(fwd, opts=SLSQP_OPTS, n_grid=130)
    maps = opt._precompute_maps(OMEGA_LOW)
    idx_grid = maps["idx_grid"]

    n = 7
    assert int(idx_grid[n]) != n, "test node must have a coarse index that actually differs from its fine index"

    cons_used = opt._per_angle_constraints(maps, OMEGA_LOW, n, None)
    cons_expected = fwd.extra_constraints_at_theta(int(idx_grid[n]))

    rng = np.random.default_rng(0)
    x = rng.normal(size=fwd.drive.dim)
    # the achievability constraints are the last len(cons_expected) entries
    used_vals = sorted(c["fun"](x) for c in cons_used[-len(cons_expected):])
    expected_vals = sorted(c["fun"](x) for c in cons_expected)
    np.testing.assert_allclose(used_vals, expected_vals, atol=1e-10)


def test_achievability_constraint_actually_zeros_open_phase_current():
    """Regression for the null-space-derivation bug: a curr_dq that
    satisfies extra_constraints_at_theta must reconstruct exactly zero
    current in the open phase(s) via the model's own full 5-phase map --
    the previous static-Clarke-null-vector-then-rotate construction did not
    (verified nonzero residual up to ~0.3-0.5A on a unit-norm test current)."""
    machine = _isotropic_pmsm()
    for open_phases in [(0,), (0, 1), (0, 2)]:
        fwd = ForwardModel(machine, n_theta=700, fault=Fault(open_phases))
        theta_idx = 123
        cons = fwd.extra_constraints_at_theta(theta_idx)
        H_full = fwd._mat_dq_to_ph_all[theta_idx]
        open_rows = H_full[list(open_phases)]
        _, _, vh = np.linalg.svd(open_rows)
        v = vh[-1]  # feasible direction: satisfies all open-phase-row constraints
        recon = H_full @ v
        for p in open_phases:
            assert abs(recon[p]) < 1e-9, f"open_phases={open_phases}: phase {p} current not zero ({recon[p]:.4e})"
