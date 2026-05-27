"""
E08 -- Phase-2 controller-bandwidth axis.

For each h_cutoff in a chosen list, apply ``low_pass_trajectory`` to
the converged reference X (from the rho=0, N=128 active-set run) and
re-evaluate Joule + iron loss on the *tracked* low-passed trajectory.

This models a closed-loop controller that filters the reference at
harmonic h_cutoff. As h_cutoff drops, harmonics of the chatter are
removed before they ever reach the iron, so iron loss should fall.
Joule loss should also change (a lower-harmonic trajectory is *not*
the same as a fresh re-optimization at that truncation).

If iron loss collapses to a floor at moderate h_cutoff and Joule loss
stays close to optimum, then "smooth reference + finite controller"
is competitive with "chattering reference + perfect controller" -- a
useful conclusion for system design.

Run:
    python experiments/iron/e08_controller_bandwidth.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")

from dynamic.active_set_optimizer import run_active_set
from dynamic.iron_loss import iron_loss, low_pass_trajectory
from experiments.chatter.common import chatter_amplitude, joule_loss, make_setup


def _figs_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    setup = make_setup()
    out_dir = _figs_dir()
    n_grid = 128

    print("Reference active-set run at rho=0, N=128 ...")
    r = run_active_set(
        model=setup.model,
        transform=setup.transform,
        omega=setup.omega_el,
        torq_target=setup.torq_target,
        curr_max=setup.machine.curr_max,
        volt_max=setup.machine.volt_max,
        n_grid=n_grid,
        max_outer_iter=10,
        tol=1e-3,
        rho=0.0,
        scheme="forward",
    )
    X_ref = r["curr_dq_grid"]
    J_ref = joule_loss(X_ref)
    loss_ref = iron_loss(setup.transform, setup.omega_el, X_ref)
    print(
        f"  reference J={J_ref:.3f}  P_eddy={loss_ref['eddy']:.3f}  "
        f"P_hyst={loss_ref['hysteresis']:.3f}  C={chatter_amplitude(X_ref, 7):.4f}\n"
    )

    hcs = [3, 5, 7, 11, 21, 31, n_grid // 2]
    Js: list[float] = []
    Pe: list[float] = []
    Ph: list[float] = []
    Cs: list[float] = []
    print("Controller bandwidth (h_cutoff) sweep:")
    for hc in hcs:
        X_lp = low_pass_trajectory(X_ref, h_cutoff=hc)
        J = joule_loss(X_lp)
        loss = iron_loss(setup.transform, setup.omega_el, X_lp)
        C = chatter_amplitude(X_lp, h_max=7)
        Js.append(J)
        Pe.append(loss["eddy"])
        Ph.append(loss["hysteresis"])
        Cs.append(C)
        print(f"  h_c={hc:3d}  J={J:.3f}  P_eddy={loss['eddy']:.3f}  P_hyst={loss['hysteresis']:.3f}  C={C:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogx(hcs, Js, "o-", label=r"$J$", markersize=8)
    axes[0].axhline(J_ref, color="C0", ls="--", lw=1, alpha=0.6, label=r"$J$ (ref)")
    axes[0].set_xlabel(r"controller cutoff $h_c$")
    axes[0].set_ylabel(r"Joule loss")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[0].set_title("E08: J vs controller cutoff")

    axes[1].semilogx(hcs, Pe, "o-", label=r"$P_{eddy}$", markersize=8)
    axes[1].semilogx(hcs, Ph, "s-", label=r"$P_{hyst}$", markersize=8)
    axes[1].axhline(loss_ref["eddy"], color="C0", ls="--", lw=1, alpha=0.6)
    axes[1].axhline(loss_ref["hysteresis"], color="C1", ls="--", lw=1, alpha=0.6)
    axes[1].set_xlabel(r"controller cutoff $h_c$")
    axes[1].set_ylabel("iron loss components")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    axes[1].set_title("E08: iron loss vs controller cutoff")

    fig.tight_layout()
    out = os.path.join(out_dir, "e08_controller_bandwidth.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
