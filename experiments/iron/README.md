# Iron-loss-aware experiments

Companion code for the dynamic torque-control paper
(`2026-Terka-Fault` paper folder, `Dynamic.tex`) — same paper as
[`dynamic/`](../../dynamic/README.md) and [`experiments/chatter/`](../chatter/README.md).
Frozen reproductions of published numbers/figures, not library usage
examples.

## Drivers

| Script | Phase | What it shows |
|---|---|---|
| `e06_operating_point_survey.py` | 0 | Coarse (T*, speed) mesh via the R0 solver: Joule loss, worst dynamic-voltage residual, eddy/hysteresis iron-loss proxies, feasibility. |
| `e07_phase1_sweep.py` | 1 | Active-set sweep over `rho` (slew penalty) at the 5 reference operating points. |
| `e07_post_hoc_pareto.py` | 1 | Reuses chatter's E02/E03 trajectories to evaluate iron-loss components post-hoc — no new optimization. |
| `e08_controller_bandwidth.py` | 2 | Low-pass-filters the converged reference at various cutoffs to model closed-loop tracking bandwidth. |
| `e10_iron_aware_pareto.py` | 4 | Augments the active-set objective with an eddy-loss term at the iron-dominated cell p4. |
| `e11_iron_aware_multi_cell.py` | 4 | Repeats E10 across cells p3/p4/p5 to test generality of the "iron loss is operating-point dominated" finding. |
| `e12_bertotti_at_p4.py` | 4 | Compares eddy-only, Bertotti-excess-only, and joint iron-aware objectives at p4. |

Reference operating points and shared metrics are defined in
`experiments/chatter/common.py`.
