# Chatter analysis experiments

Companion code for Section "Trajectory chatter: causes and diagnostics"
in `Dynamic.tex` (`2026-Terka-Fault` paper folder).

## Reference operating point

All experiments use the same point so results are comparable:

- machine: `IEEEMachine2` (5-phase, 8 pole pairs)
- `curr_max = 30 A`, `volt_max = 13 V`
- `n_mech = 1400 rpm` (`omega_el = 1172.86 rad/s`)
- `T* = 4 Nm`
- coarse-grid forward-Euler is the default; `N` is each script's own knob

Common utilities (`common.py`) define the metrics:

- `joule_loss(X) = (1/N) sum_n |x_n|^2`
- `chatter_amplitude(X, H) = ||X - LP_H(X)|| / ||X||`, with `LP_H` keeping only the first `H+1` Fourier harmonics
- `tail_mass(X, H)` — energy fraction above harmonic `H`
- `fourier_spectrum(X)` — per-component magnitude spectrum

## Drivers

| Script | Tests cause(s) | Outputs |
|---|---|---|
| `e01_fourier_spectrum.py` | B / E vs C | `figs/e01_fourier_spectrum.png` |
| `e02_smoothness_sweep.py` | B vs C | `figs/e02_smoothness_sweep.png` |
| `e03_grid_refinement.py`  | A vs D vs C | `figs/e03_grid_refinement.png` |
| `e04_central_difference.py` | A | `figs/e04_forward.png`, `figs/e04_central.png` |
| `e05_restart_sensitivity.py` | B / E | `figs/e05_restart_sensitivity.png` |

Each runs from a fresh Python session and is self-contained:

```bash
python experiments/chatter/e01_fourier_spectrum.py
python experiments/chatter/e02_smoothness_sweep.py
python experiments/chatter/e03_grid_refinement.py
python experiments/chatter/e04_central_difference.py
python experiments/chatter/e05_restart_sensitivity.py
```

Wall-time guide on a 2024 Apple Silicon laptop:

- E01: ~25 s (one active-set run at N=128)
- E02: ~3 min (5 active-set runs)
- E03: ~5 min (N up to 256)
- E04: ~50 s (two active-set runs)
- E05: ~3 min (1 base + 5 restart joint solves)

## Selection rule

See `Dynamic.tex` Section "Selection rule". After running E01–E05,
identify the dominant cause(s) and pick the remedy accordingly. The
choice has direct consequences for the iron-loss calculation, which
sees the trajectory's $d\bm{i}_s/d\theta$ directly.
