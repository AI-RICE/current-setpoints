# Fourier-ansatz vs. full-grid experiments

Companion code for the dynamic torque-control paper
(`2026-Terka-Fault` paper folder, `Dynamic.tex`) — same paper as
[`dynamic/`](../../dynamic/README.md) and [`experiments/chatter/`](../chatter/README.md).
Frozen reproductions of published numbers/figures, not library usage
examples.

## Drivers

| Script | Question |
|---|---|
| `e13_headtohead.py` | How much of the full position-dependent (grid) optimizer's Joule advantage over the low-order Fourier ansatz (prior art) is real, in the voltage-limited (FW) vs. current-limited regime? |
| `e15_fault_headtohead.py` | Same question, under a single open-phase fault — does the gap widen relative to the <1% seen healthy? |
| `e17_fault_envelope.py` | Single open-phase fault realisable torque-speed envelope on IEEEMachine2, benchmarked against Fall (2016) Eqs 16-19, Yepes (2024), and the static-paper formulation. |

`diag_*.py` are one-off diagnostic scripts used while deriving the
above (solver init, ripple, feasibility, voltage validation) — kept for
provenance, not part of the headline results.

`envelope_solver.py` / `phase_voltage.py` are shared solver/voltage-map
code used by the `e1*` drivers above.
