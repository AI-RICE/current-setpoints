"""
Ad-hoc physicality checks for the fault-tolerant forward model / optimizer,
run against current_setpoints_new.

Test 1 (Neutral Wire): sum of realized phase currents for a 1-phase-open fault.
Test 2 (Phantom Torque): optimal curr_dq vs. realized curr_dq (round-trip through
    curr_ph and back through the forward Clarke/Park transform) for a 2-phase
    non-adjacent fault.
Test 3 (Torque Hierarchy): max torque at a fixed omega for healthy vs. 1-fault
    vs. 2-fault (adjacent and non-adjacent).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from current_setpoints_new.models.machines import PMSM5Phase
from current_setpoints_new.models.forward_model import Fault, ForwardModel
from current_setpoints_new.optimization.optimizer import StaticOptimizer

CURR_MAX = 30.0
VOLT_MAX = 13.0
OMEGA_MAX = 1800
SLSQP_OPTS = {"disp": False, "ftol": 1e-8, "maxiter": 500}


def to_elec(rpm, n_ppairs):
    return rpm * (np.pi / 30) * n_ppairs


def make_pmsm():
    m = PMSM5Phase()
    m.set_max_pars(CURR_MAX, VOLT_MAX, OMEGA_MAX)
    return m


def clarke_forward_matrix():
    """Forward (phase -> alpha/beta/3rd-harmonic/zero-seq) Clarke matrix, 5x5,
    consistent with forward_model._build_clarke_5phase."""
    C = np.zeros((5, 5))
    for p in range(5):
        a = p * 2 * np.pi / 5
        C[0, p] = np.cos(a)
        C[1, p] = np.sin(a)
        C[2, p] = np.cos(3 * a)
        C[3, p] = np.sin(3 * a)
        C[4, p] = 0.5
    return C * (2.0 / 5.0)


def forward_park(theta):
    """Inverse of the R(theta) block used in Fault._build_reduced_map /
    ForwardModel.__init__: maps alpha/beta (h=1,3) -> dq (h=1,3)."""
    c1, s1 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(3 * theta), np.sin(3 * theta)
    R = np.array([
        [c1, -s1, 0, 0],
        [s1, c1, 0, 0],
        [0, 0, c3, -s3],
        [0, 0, s3, c3],
    ])
    return R.T  # R maps dq->alphabeta (orthogonal), so R.T is alphabeta->dq


def realized_curr_dq_from_phase(curr_ph_full5, theta):
    """Map full 5-phase currents (open phases = 0) back to dq via forward Clarke+Park."""
    C = clarke_forward_matrix()
    alphabeta = C @ curr_ph_full5          # (5,) = [a1,b1,a3,b3,zero]
    ab4 = alphabeta[:4]
    return forward_park(theta) @ ab4


def full_phase_vector(curr_ph_surviving, kept, n_phases=5):
    full = np.zeros(n_phases)
    full[list(kept)] = curr_ph_surviving
    return full


def test1_neutral_wire():
    print("\n=== Test 1: Neutral Wire Test (1-phase fault) ===")
    pmsm = make_pmsm()
    fault = Fault((0,))
    fwd = ForwardModel(pmsm, fault=fault, n_theta=700)
    opt = StaticOptimizer(fwd, opts=SLSQP_OPTS)

    omega = to_elec(900, pmsm.n_ppairs)
    sol = opt.maximize_torque(omega)
    print("success:", sol.success, "torque:", sol.torque, "curr_dq:", sol.curr_dq)

    curr_ph = fwd.curr_ph(omega, sol.curr_dq)   # (n_surviving=4, n_theta+1)
    phase_sum = np.sum(curr_ph, axis=0)
    print("max|sum of surviving phase currents| over theta:", np.max(np.abs(phase_sum)))
    print("sample values:", phase_sum[:5])
    if np.allclose(phase_sum, 0.0, atol=1e-8):
        print("RESULT: sum == 0 -> isolated neutral, KCL satisfied by construction.")
    else:
        print("RESULT: sum != 0 -> would require a neutral wire.")


def test2_phantom_torque():
    print("\n=== Test 2: Phantom Torque Test (2-phase non-adjacent fault) ===")
    pmsm = make_pmsm()
    fault = Fault((0, 2))  # non-adjacent
    fwd = ForwardModel(pmsm, fault=fault, n_theta=700)
    opt = StaticOptimizer(fwd, opts=SLSQP_OPTS)

    omega = to_elec(900, pmsm.n_ppairs)
    sol = opt.maximize_torque(omega)
    print("success:", sol.success, "torque:", sol.torque)
    print("optimal_curr_dq:", sol.curr_dq)
    print("  i_d1, i_q1, i_d3, i_q3 =", sol.curr_dq)

    curr_ph = fwd.curr_ph(omega, sol.curr_dq)  # (3, n_theta+1)

    max_diff = 0.0
    worst_theta = None
    worst_realized = None
    for theta_idx in range(0, curr_ph.shape[1], 10):
        theta = fwd.vec_theta[theta_idx]
        full5 = full_phase_vector(curr_ph[:, theta_idx], fwd.fault._kept)
        realized = realized_curr_dq_from_phase(full5, theta)
        diff = np.max(np.abs(sol.curr_dq - realized))
        if diff > max_diff:
            max_diff = diff
            worst_theta = theta
            worst_realized = realized

    print("optimal_curr_dq:            ", sol.curr_dq)
    print("realized_curr_dq at theta=0:", realized_curr_dq_from_phase(
        full_phase_vector(curr_ph[:, 0], fwd.fault._kept), fwd.vec_theta[0]))
    print(f"worst-case realized_curr_dq at theta={worst_theta:.4f}: {worst_realized}")
    print("max |optimal - realized| over all theta:", max_diff)
    if max_diff < 1e-6:
        print("RESULT: optimal_curr_dq == realized_curr_dq at every theta -> honest, no phantom torque.")
    else:
        print("RESULT: mismatch at some theta -> phantom torque only hidden at theta=0.")


def test3_torque_hierarchy():
    print("\n=== Test 3: Torque Hierarchy Test ===")
    pmsm = make_pmsm()
    omega = to_elec(900, pmsm.n_ppairs)

    configs = {
        "Healthy": Fault(()),
        "1-Phase Fault": Fault((0,)),
        "2-Phase Adjacent": Fault((0, 1)),
        "2-Phase Non-Adjacent": Fault((0, 2)),
    }

    torques = {}
    for name, fault in configs.items():
        fwd = ForwardModel(pmsm, fault=fault, n_theta=700)
        opt = StaticOptimizer(fwd, opts=SLSQP_OPTS)
        sol = opt.maximize_torque(omega)
        torques[name] = sol.torque
        print(f"{name:22s} success={sol.success!s:5s} max_torque={sol.torque:.4f}")

    print("\nTorques:", torques)
    healthy = torques["Healthy"]
    f1 = torques["1-Phase Fault"]
    f2a = torques["2-Phase Adjacent"]
    f2n = torques["2-Phase Non-Adjacent"]

    ok = healthy > f1 > f2a and healthy > f1 > f2n
    print("Healthy > 1-Phase > 2-Phase (adjacent):", healthy > f1 > f2a)
    print("Healthy > 1-Phase > 2-Phase (non-adjacent):", healthy > f1 > f2n)
    if ok:
        print("RESULT: hierarchy holds.")
    else:
        print("RESULT: hierarchy VIOLATED.")


if __name__ == "__main__":
    test1_neutral_wire()
    test2_phantom_torque()
    test3_torque_hierarchy()
