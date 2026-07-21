import sys
sys.path.insert(0, r"c:\Users\matas\Desktop\CurrentSetpoints_main")
import numpy as np

from current_setpoints.models.machines import ieee_machine2
from current_setpoints.models.forward_model import ForwardModel, Fault

drive = ieee_machine2()

def check(open_phases):
    fwd = ForwardModel(drive, fault=Fault(open_phases), n_theta=700)
    # stack open-phase rows across ALL theta samples
    rows = []
    for t in range(fwd.vec_theta.size - 1):
        H = fwd._mat_dq_to_ph_all[t]
        for p in open_phases:
            rows.append(H[p])
    rows = np.array(rows)  # (n_open * n_theta, 4)
    sv = np.linalg.svd(rows, compute_uv=False)
    rank = np.sum(sv > 1e-9 * sv[0])
    print(f"open_phases={open_phases}: singular values over full theta sweep = {sv}")
    print(f"  effective rank = {rank} / 4  -> only i_dq={'0' if rank==4 else f'a {4-rank}-dim subspace'} satisfies the constraint at EVERY theta")

check((0,))
check((0, 1))
check((0, 2))
