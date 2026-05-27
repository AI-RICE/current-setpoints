# Joule vs iron-loss investigation — system axes

The chatter analysis (E01–E05) confirms that the active-set solution
genuinely wants high-harmonic content (cause C). This raises the
practical question: does that content actually pay off when iron
losses are accounted for, and does the trade-off depend on what the
controller can track?

The investigation has **five independent axes**. Each yields a Pareto
front of (Joule, iron) costs; the overall recommendation depends on
how the axes interact.

---

## Axis 1 — Reference parameterization (representation)

How is $\bm{i}_s(\theta)$ encoded for the optimizer to manipulate?

| Variant | Decision space size | Restricts harmonic content? |
|---|---|---|
| Fourier truncation $N_F$ harmonics | $4(2N_F+1)$ | Yes — hard cap at $h\le N_F$ |
| Grid $N$ nodes | $4N$ | Nyquist soft cap at $h \le N/2$ |
| Mixed: Fourier coefficients + grid residuals | both | Adjustable |

**Question**: at what $N$ (or $N_F$) does the per-axis Pareto front
saturate? E03 already shows monotonic improvement in $J$ up to $N=256$.
Repeat the same study with $P_\text{iron}$ added.

---

## Axis 2 — Reference smoothness (regularization)

How much chatter is the optimizer allowed to put in?

| Variant | Implementation |
|---|---|
| No penalty | `rho = 0` |
| Slew penalty | `+ rho * sum |x_{n+1} - x_n|^2` (E02) |
| Hard spectral cap | post-hoc `low_pass_trajectory(X, h_cutoff)` |
| Soft spectral cap | weighted Fourier penalty in the objective |

**Question**: do "smooth" references have meaningfully higher Joule
loss, or is the smoothness essentially free? E02 already showed a
~17% chatter reduction at $\rho=1$ costs only 0.02% in $J$.

---

## Axis 3 — Controller bandwidth (post-hoc filter)

The reference is just a desired trajectory. The actual current that
flows is what the closed-loop controller produces while tracking this
reference. A finite-bandwidth controller filters the reference,
attenuating harmonics above its closed-loop cutoff $h_c$.

| Knob | Range |
|---|---|
| First-order low-pass cutoff $h_c$ | 3, 5, 10, 30, 64 (harmonic index) |
| Filter type | first-order, Bessel, ideal brick-wall |
| **Deadbeat (ZOH)** sample count $N_s$ | 16, 32, 64, 128 samples per electrical period |
| Deadbeat one-step delay | 0 (idealised) or 1 sample (realistic digital) |

Three structurally distinct controller models to compare:
1. **Ideal (no filter)**: tracked = reference. Upper bound on iron loss.
2. **First-order low-pass at $h_c$**: smooth attenuation, models analog
   or PI-loop closure with bandwidth $h_c\,\omega$.
3. **Deadbeat (ZOH)**: piecewise-constant tracking over $N_s$ samples
   per period; harsher discontinuities, models a digital current
   controller with sample rate $N_s\omega/2\pi$.

The deadbeat model is implemented as
``dynamic.iron_loss.deadbeat_tracking``.

**Question**: if the controller already filters at $h_c \approx 10$,
the chatter at $h=19, 21, \ldots$ never makes it into the iron and we
gain only Joule-side benefits from the high-harmonic reference. In
that case the "smooth reference + perfect controller" path is
identical to the "chattering reference + finite-bandwidth controller"
path and we should prefer the smooth reference for robustness.

This axis matters most because it is *outside* the optimizer and
encodes a physical bandwidth limitation of the real plant.

---

## Axis 4 — Iron-loss model (loss components)

How sophisticated is the iron-loss accounting?

| Component | Where to find it | Coefficient |
|---|---|---|
| Eddy | `dynamic.iron_loss.eddy_loss` | $k_e$ |
| Hysteresis (Steinmetz) | `dynamic.iron_loss.hysteresis_loss` | $k_h, \beta$ |
| Excess (Bertotti) | `dynamic.iron_loss.excess_loss` | $k_x$ |
| Total | `dynamic.iron_loss.iron_loss` | sum |

Coefficients are machine-specific. Without measured fits, use
$k_e = k_h = 1$, $\beta = 1.8$, $k_x = 0$ for *relative* comparisons —
absolute numbers require a finite-element reference or a magnetic
measurement.

**Question**: does the qualitative shape of the (Joule, iron) Pareto
depend on which components are included? Eddy alone is a clean
quadratic and FFT-free; the moment hysteresis is in, the optimum
becomes a max-min on per-phase $B_\text{max}$.

---

## Axis 5 — Operating point

The Pareto curve depends strongly on $(T^*, \omega)$.

- **Low $\omega$, low $T^*$**: iron loss ≪ Joule loss; reference can
  be anything; chatter doesn't matter.
- **Low $\omega$, high $T^*$** (current-limited): static MTPA arc
  saturation; chatter has small amplitude.
- **High $\omega$, low $T^*$** (BEMF-dominated): eddy iron loss
  dominant; trade-off relevant. **This is where E04+ landed.**
- **High $\omega$, high $T^*$** (corner): all constraints active,
  many DoF locked; biggest payoff for finer harmonic control.

A 2D scan over $(T^*, n_\text{mech})$ on a coarse grid (5x5) reveals
where the trade-off study is meaningful and where it isn't.

---

## Proposed experimental schedule

The five axes are independent; controlled experiments fix all but one.

### Phase 0 (operating-point survey, preselect ~5 representative points)

Before committing computational effort to Phases 1--4 at a single point,
do a quick sweep over a coarse $(T^*, n_\text{mech})$ mesh and identify
which cells are *interesting* (i.e. where the trade-off study is
meaningful) versus which cells are boring (R0 already solves it).

For each mesh cell, run only the cheap R0 (per-angle independent)
solver and record:
- Joule loss $J_{R0}$;
- dynamic voltage residual on the R0 trajectory (how much it would
  violate the full dynamic constraint -- a *score* for "active-set
  needed here");
- eddy iron-loss proxy $P_e$ and ratio $P_e/J$;
- chatter amplitude in R0 (typically negligible -- R0 doesn't chatter
  much, only the active-set refined solution does).

Heatmaps of these four scalars over the mesh highlight:
- "boring" cells (low residual, low $P_e/J$): R0 is good enough;
- "current-limited" cells (high $J$, low $P_e$): static-style;
- "voltage-limited" cells (low $J$, high $P_e/J$, large residual):
  the dynamic formulation matters but iron dominates;
- "corner" cells (high $J$, high $P_e$, large residual): all axes
  interact; richest study.

Select 5 representative cells -- one boring, one mid-corner, one
current-edge, one voltage-edge, one corner -- and run Phases 1--4 at
each.

The mesh used is intentionally coarse (5x5) for the initial pass; cells
adjacent to interesting ones can be filled in afterwards.

→ Driver: `experiments/iron/e06_operating_point_survey.py`

### Phase 1 (single operating point, post-hoc evaluation only)

Reference operating point: $T^*=4$\,Nm, $n_\text{mech}=1400$\,rpm
(same as E01–E05). Reuse the *existing* converged trajectories from
E02 and E03 (no new optimization). Evaluate
`iron_loss(transform, omega, X)` on each and report the
(Joule, eddy, hysteresis, total) tuple per trajectory. **Outcome**: a
first-pass Pareto front from $\rho$ and $N$ sweeps already done.

→ Driver: `experiments/iron/e07_post_hoc_pareto.py`

### Phase 2 (controller bandwidth axis)

Apply `low_pass_trajectory(X, h_c)` for
$h_c \in \{3,5,7,11,21,\infty\}$ to each Phase-1 trajectory. Plot how
iron loss decreases with reducing $h_c$ and at what point the
trade-off collapses (iron-loss flat).

→ Driver: `experiments/iron/e08_controller_bandwidth.py`

### Phase 3 (operating-point envelope)

For 5×5 grid of $(T^*, n_\text{mech})$ run the active-set solver with
$\rho \in \{0, 1\}$ and evaluate the iron-loss components. Identify
where the trade-off is large enough to matter.

→ Driver: `experiments/iron/e09_envelope.py`

### Phase 4 (iron-aware optimization)

Add the eddy term as a soft penalty in the objective and resolve.
This is the first time iron enters the optimization. Sweep the
penalty weight and trace a true Pareto front.

→ Driver: `experiments/iron/e10_iron_in_objective.py`

### Phase 5 (writeup)

Decide on the recommended numerical recipe based on Phase 1–4. Update
`Dynamic.tex` with the conclusions and add the comparison to the
low-order Fourier prior art.

---

## Notes on coefficient choice

Absolute iron-loss numbers require a fit. Two cheap options:

1. **Match the static MTPA point.** At the static d-q operating
   point, run the proxy and compare against any reported iron-loss
   number for IEEEMachine2 from prior literature. Scale $k_e, k_h$ so
   the proxy matches the literature.
2. **Match a finite-element point.** Run the static d-q trajectory
   through a FE solver (e.g., FEMM at a few representative angles)
   and fit the proxy to the FE iron-loss output. More work but more
   defensible.

For Phase 1–3 we can stay with $k_e = k_h = 1$ since the *shape* of
the trade-off, not its absolute level, is what we need.
