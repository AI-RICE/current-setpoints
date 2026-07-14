# Dynamic-MTPA experiments

Companion code for the dynamic (position-dependent) torque-control
paper (`2026-Terka-Fault` paper folder, `Dynamic.tex`) — the 3rd article
in the [[library-vision]] series (see `dynamic-mtpa` branch history).

These scripts snapshot the numbers/figures published for that paper;
they are frozen reproductions, not library usage examples — the
library API underneath them may change independently in the future.

## Drivers

| Script | What it shows |
|---|---|
| `01_low_speed_torque_sweep.py` | Static MTPA vs. R0 (independent per-angle) relaxation at a single low speed, swept over target torque. |
| `02_speed_sweep_at_fixed_torque.py` | Fixed torque, swept over speed — shows where R0's reconstructed voltage starts violating `V_max`. |
| `03_active_set_demo.py` | Active-set / successive-relaxation solve at the point found in `02` where R0 fails the voltage constraint; compares the R0 starting trajectory against the converged active-set trajectory. |

Supporting modules (`_waveforms.py`, `voltage_diagnostics.py`,
`fourier_math.py`, `iron_loss.py`, `regimes.py`) are shared math/plotting
used by the numbered drivers above and by `experiments/`.
