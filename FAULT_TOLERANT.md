# Fault-tolerant mode — findings so far

Machine: `ieee_machine2()` (5-phase PMSM, `curr_max=30A`, `volt_max=13V`,
identified `R_stat`/`L_stat`/`flux_pm` from real measured data — **not**
electromagnetically isotropic in dq-space). `Fault(open_phases)` supports 0-2
simultaneous open phases (indices 0-4).

## What's verified correct

- **`IndependentOptimizer` (dynamic/per-angle mode)** is physically correct
  for arbitrary faults. `Fault.extra_constraints_at_theta()` rotates the
  achievability null-space constraint with rotor angle
  (`N @ R(theta) @ i = 0`), which is the right thing to do — confirmed via:
  - Clarke round-trip check (forward → dq0 → inverse, error ~1e-16).
  - Rotation-consistency check (reconstruct phase current from a target dq,
    transform back, recover the exact target at multiple theta).
  - 60-restart independent re-solve of the bottleneck node for several fault
    pairs reproduces the optimizer's reported value exactly — not a
    local-optimum/seeding artifact of the sequential per-node solve.
  - No coefficient/harmonic-order bug: all sites that encode "1st vs 3rd
    harmonic" (`_build_clarke_5phase`, `_build_cross_coupling`'s `h=2*i+1`,
    both `R(theta)` construction sites) consistently use order 1 and 3,
    matching Fall (2016)'s own equations (`pΩ` vs `3pΩ`).
  - See `tests/test_new_dynamic_fault.py` (13 tests) for the regression
    suite locking all of this in.

- **`baselines/Fall/`** already provides a faithful, validated replication of
  Fall (2016)'s closed-form fault-tolerant strategy (see that folder's own
  README) — useful as an independent point of comparison, though it solves a
  more restricted problem (see below).

## What's broken — needs a fix before the next PR

- **`StaticOptimizer` is physically invalid for 2-open-phase faults.**
  `Fault.extra_constraints()` (used by `StaticOptimizer`) checks current
  achievability using a single **theta=0 snapshot** of the null-space
  direction `N`, but the real constraint rotates with rotor angle. A
  constant-in-time dq current can only be exactly realized by 3 surviving
  wires at *isolated instants*, not throughout a full rotation — verified
  numerically: static mode's reported optimum for `(0,2)`-open had a
  constraint residual swinging ±19.9 across the rotation, despite reading
  exactly 0 at theta=0.
  - Single-fault (`n_open==1`) is unaffected: 4 surviving wires ↔ 4 dq
    dims is an exact bijection (no pinv, no achievability gap).
  - Fix not yet implemented: `StaticOptimizer` should raise/refuse for
    `fwd.fault.n_open >= 2` rather than silently returning an invalid
    optimum (or otherwise flag it loudly).
  - Full derivation: see memory `static-fault-constraint-bug`.

## Max-torque comparison (ω=0, current-limited regime in all cases)

**Static mode** (valid only for n_open ≤ 1):

| Scenario | T_max |
|---|---|
| Healthy | 7.576 |
| Single fault (any phase, exactly phase-invariant) | 4.072 |

**Dynamic mode** (`IndependentOptimizer`, `n_grid=130` — a multiple of 5,
see aliasing note below):

| Scenario | T_max |
|---|---|
| Healthy | 7.752 |
| Single fault (any phase, exactly phase-invariant) | 5.615 |
| Double fault (0,1) adjacent | 3.18 |
| Double fault (1,2) adjacent | 4.14 |
| Double fault (2,3) adjacent | 3.81 |
| Double fault (3,4) adjacent | 3.44 |
| Double fault (0,4) adjacent | 4.04 |
| Double fault (0,2) non-adjacent | 2.75 |
| Double fault (0,3) non-adjacent | **5.04 (best)** |
| Double fault (1,3) non-adjacent | 5.42 |
| Double fault (1,4) non-adjacent | **3.23 (worst)** |
| Double fault (2,4) non-adjacent | 2.60 |

### Key findings from this sweep

1. **Single-fault T_max is exactly phase-invariant** in both modes (once the
   `n_grid` aliasing artifact below is controlled for) — losing exactly 1 of
   5 phases never reduces the controllable DOF (4 wires ↔ 4 dq dims, exact
   bijection), so every single-fault configuration solves the literal same
   optimization problem, just relabeled.

2. **Double-fault T_max is NOT simply "adjacent vs. non-adjacent"** and does
   **not** generalize from any one example pair — the full sweep shows
   non-adjacent pairs span the *entire* range (from the single worst case
   (1,4) to the single best case (0,3)), overlapping heavily with the
   adjacent-pair range (3.18–4.14).

3. **Root cause of the double-fault variance**: two things that don't rotate
   together —
   - The torque objective (`A`, `b` from `L_stat`/`flux_pm`) is fixed and
     **not isotropic** in dq-space (real measured saliency, e.g.
     `L_stat[0,1]=-0.0286` vs `L_stat[1,0]=-0.0133`).
   - Losing 2 phases collapses dq control to a 3-D subspace, and *which*
     3-D subspace survives depends on the specific physical winding pair —
     confirmed the null-space row `N` is genuinely different (not a
     relabeling/rotation) across same-shape pairs.
   - So a double fault's achievable subspace can align well or badly with
     the machine's fixed, anisotropic torque-maximizing direction — that
     alignment, not fault count or adjacency, drives T_max.
   - **Investigated 2026-07-14, negative result** — no single simple scalar
     predictor found. Checked:
     - `|b_sub|/|b|` (how much of the fixed linear torque term survives
       projection onto the fault's null space) — ranking doesn't match
       T_max (e.g. (1,3) has low alignment 0.79 but the 2nd-highest T_max
       5.42; (1,2) has the highest alignment 0.996 but only middling T_max
       4.14).
     - Phase-current "gain" (largest singular value of the null-space-to-
       phase-current map) — identical (2.5) across *all* 10 pairs, so not
       discriminating at all.
     - `|i_opt|` (sustainable current magnitude in the subspace before any
       phase saturates at `curr_max`) — varies 20.85 to 41.26 across pairs
       (>2x), but correlation with T_max is ~0 (-0.02).
     - `T_max/|i_opt|` ("torque density") — varies 0.082 to 0.199 (>2x),
       but also doesn't rank-match T_max alone.
     - Conclusion: `T_max` is the *product* of these last two independently-
       varying factors (how much current the subspace sustains × how
       efficiently that current converts to torque), and neither factor
       alone predicts the ranking — e.g. (0,3) wins not by having the most
       sustainable current (it doesn't) nor the highest density (it isn't
       highest either), but by landing simultaneously well on both.
       Capturing this jointly appears to require solving the constrained
       optimization itself; no shortcut scalar found. Not pursued further
       absent a more targeted hypothesis.

4. **A single current-saturated healthy operating point explains the large
   single-fault drop (53.8% retained in static mode).** Healthy `T_max`
   already occurs at `curr_peak=30.0` exactly (the hard limit) — no current
   headroom is left unused, so losing a phase forces the dominant q-axis
   current to roughly halve (from ~33.7 to ~17.9) just to keep the same
   phase-current cap. This machine's `curr_max` is fault-invariant; many
   real fault-tolerant drives instead *increase* the allowed current
   post-fault (freed thermal headroom from the open phase), which isn't
   modeled here.

## Cross-check: constant-flux vs. neural-flux model (2026-07-14)

All the numbers above use `ieee_machine2()` (`ConstantFlux`) — a *single*
hardcoded `L_stat` applied at every operating point, identified once (likely
near the healthy MTPA point). `ieee_machine2_trained_neural_flux()` instead
derives inductance as `L_stat + d(flux)/di` from a trained neural network, so
it's genuinely current-dependent. Printed both directly:

- At zero current, the neural-derived inductance is wildly different from
  the hardcoded `L_stat` (e.g. `[0,1]`: hardcoded `-0.000029` vs. neural
  `+0.000251` — ~10x larger and opposite sign).
- Near the healthy MTPA point `[-8.19, 33.66, -3.82, 4.39]`, the neural value
  converges much closer to the hardcoded one (`[0,1]`: `-0.000041` vs.
  `-0.000029`), consistent with the hardcoded `L_stat` having been fit near
  that regime.

This raised a real concern: fault operating points sit at very different
current magnitudes/directions than healthy MTPA (faults redistribute current
onto a smaller/reshaped subspace), so `ConstantFlux` extrapolates its
healthy-fit `L_stat` into a regime it wasn't characterized for — a plausible
source of the messy, non-monotonic double-fault ranking above.

**Tested this directly** — reran the full healthy/1-fault/2-fault sweep with
`ieee_machine2_trained_neural_flux()` (same `IndependentOptimizer`,
`n_grid=130`, ω=0):

| Scenario | Constant-flux | Neural-flux |
|---|---|---|
| Healthy | 7.752 | 7.668 |
| Single fault (any phase, still exactly phase-invariant) | 5.615 | 5.511 |
| (0,1) adjacent | 3.18 | 4.52 |
| (1,2) adjacent | 4.14 | 3.96 |
| (2,3) adjacent | 3.81 | 3.90 |
| (3,4) adjacent | 3.44 | 4.39 |
| (0,4) adjacent | 4.04 | 3.89 |
| (0,2) non-adjacent | 2.75 | 3.00 |
| (0,3) non-adjacent | 5.04 | 5.04 |
| (1,3) non-adjacent | 5.42 | **5.34 (best)** |
| (1,4) non-adjacent | **3.23 (worst)** | **2.53 (worst)** |
| (2,4) non-adjacent | 2.60 | 2.95 |

**Result: the hypothesis was wrong.** Individual values shift (e.g. (0,1)
moves 3.18→4.52), but the qualitative picture is unchanged: adjacent pairs
still span a wide range that heavily overlaps non-adjacent, non-adjacent
pairs are *still* both the best case ((1,3)) and worst case ((1,4)), and
single-fault remains exactly phase-invariant. Since the constant-`L_stat`
snapshot and the full nonlinear neural-flux model are two independent
characterizations of the same real machine and they **agree** on this messy
ranking, the non-monotonic double-fault behavior is not an artifact of
extrapolating a fixed inductance matrix outside its fit region — it's
consistent evidence of a genuine property of `ieee_machine2`'s real,
identified anisotropy.

## Aliasing gotcha (fixed, keep in mind for future dynamic-mode work)

`IndependentOptimizer`'s default `n_grid=64` is **not a multiple of 5** —
this broke single-fault's expected exact phase-symmetry by ~1.4% purely from
theta-grid discretization not aligning with the 5-fold phase symmetry.
Confirmed: `n_grid=65` or `n_grid=130` restores exact symmetry. Any fault
comparison sweep with `IndependentOptimizer`/`ActiveSetOptimizer` should use
an `n_grid` that's a multiple of 5.

## How this compares to Fall (2016)

Fall's method (see `baselines/Fall/README.md` for the full derivation) is a
**hybrid**, not directly comparable to either of our modes:
- Assumes purely sinusoidal back-EMF (no 3rd harmonic torque contribution),
  which `ieee_machine2` does *not* have.
- Holds only `(id1,iq1)` constant (optimized, 2 DOF) and derives
  `(id3,iq3)` as a **time-varying, non-optimized byproduct** via KCL +
  (for 1-fault only) an equal-phase-current-magnitude heuristic.
- This sidesteps our `StaticOptimizer` bug entirely, since Fall never
  claims `(id3,iq3)` are constant — only the torque-producing pair is.
- Fall's own machine (different parameters, arguably more isotropic) shows
  the "textbook" ranking (adjacent worse than non-adjacent) that does
  **not** generalize to `ieee_machine2`'s real, anisotropic parameters.

## Open items / next steps

- [ ] Fix `StaticOptimizer` to raise/refuse for `fwd.fault.n_open >= 2`.
- [ ] Extend fault testing to `ActiveSetOptimizer` at nonzero omega (only
      `IndependentOptimizer`/R0 has been fault-tested so far; R0 ignores
      inter-angle voltage coupling, fine at omega=0, unvalidated at higher
      speed / field-weakening).
- [ ] Investigate: what geometric quantity about a double-fault pair
      predicts T_max? (in progress)
- [ ] Decide how "fault mode" becomes a documented, first-class part of the
      unified library (docs page? convenience factory/example?).
