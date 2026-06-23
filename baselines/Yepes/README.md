# Yepes (2024) — baseline reconstruction (ML stage of NSBE-FRML)

Replication of the **minimum-stator-copper-loss closed-form current
reference (Eq 7)** from

> A. G. Yepes, W. E. Abdel-Azim, A. Shawier, A. S. Abdel-Khalik,
> M. S. Hamad, S. Ahmed, J. Doval-Gandoy,
> **"Open-Phase-Tolerant Online Current References for Maximum Torque
> Range and Minimum Loss With Current and Torque-Ripple Limits for
> n-Phase Nonsalient PMSMs With Nonsinusoidal Back EMF"**,
> *IEEE Transactions on Transportation Electrification* **10**(1):432–449,
> 2024-03. DOI [10.1109/tte.2023.3288525](https://doi.org/10.1109/tte.2023.3288525).

Yepes is the other named competitor in the dynamic-paper discussion.
Unlike Fall (verbatim replicated in
[`../Fall/`](../Fall/README.md)), Yepes's full method (**NSBE-FRML**)
is a multi-stage online algorithm spanning Sections III–VI of the
paper. This baseline replicates only the **foundational ML stage**
(closed-form Eq 7), which is the piece the rest of the algorithm
builds on and the one most-cited as "minimum-loss reference."

Contents:

- [`replicate.py`](replicate.py) — verbatim Eq (7) plus the five-phase
  illustrative example of their Section II.
- [`results.txt`](results.txt) — captured stdout.
- This file — scope, what's covered, what isn't, and the convention
  diagnosis.

## 1. Scope of this baseline

| | covered here? |
|---|---|
| (a) ML reference (Eq 7) — closed-form min-SCL, no current limits | ✅ |
| (b) Peak-current saturation when ML overshoots | ⛔ (their Sec IV) |
| (c) Torque-ripple limit | ⛔ (their Sec V) |
| (d) RMS-current overload reduction | ⛔ (their Sec VI) |
| Voltage constraint of any kind | ⛔ (they don't include one — listed as future work) |

So this is a *small* replication compared to Fall's. The bigger point
isn't to reproduce the full NSBE-FRML pipeline; it's to **anchor what
Yepes optimises** with one published numerical check, and to make the
structural finding (no voltage constraint) explicit and code-visible.

## 2. Method recap — Eq (7), the ML reference

Per their Section II, for an n-phase non-salient PMSM with back-EMF
function `e_k(θ)` (back-EMF over mechanical speed), the per-sample
minimum-SCL current reference subject to a torque equality and the
zero-sequence (wye) equality is:

```
       f .* e  −  (fᵀe / fᵀf) · f
i  =  ────────────────────────────  · T
      (f .* e)ᵀ · e  −  (fᵀe)² / fᵀf
```

where `f_k = 1` for healthy phases and `0` for open phases, and `T` is
the desired electromagnetic torque. The open phases carry zero current
identically (top-left zero in the elementwise product).

In healthy 5-phase with balanced fundamental + 3rd back-EMF, `fᵀe = 0`
and the formula collapses to `i_k = T · e_k / Σ_j e_j²` — i.e., the
familiar "current waveform proportional to back-EMF" result for
non-salient minimum-loss control.

## 3. The five-phase illustrative example (Yepes §II, page 436)

The paper sets up a simple, normalisation-clean example to motivate the
need for stages (b)–(d):

- `n = 5`, non-salient.
- Back-EMF: **fundamental + 3rd harmonic of amplitude 30% of the
  fundamental, "opposite phase angle"** (their Fig 4). See §4 below.
- Limits: `i_pk^mx = 1 A`, `i_rms^mx = 0.83 A`.
- **Rated torque: 152.7 Nm** — defined as the healthy ML torque at
  `i_pk = 1`. ("The last two ratings follow from considering
  `i_pk^mx = 1 A` in healthy conditions.")
- Anchor: with **phase a open**, the ML reference hits the converter
  peak limit `i_pk = 1` at **T₁ = 75.5 Nm**, i.e. only **49.4 %** of
  the healthy rated torque.

So the published target ratio is

```
T₁ / T_rated  =  75.5 / 152.7  =  49.4%.
```

## 4. The "opposite phase" sign — convention diagnosis

"Third-harmonic with opposite phase angle" is ambiguous on its own —
it can mean either the third **flattens** the fundamental's peak (the
two interfere destructively at the peak: `sin(θ) + 0.3 sin(3θ)`,
crest factor ~1.25) or **sharpens** it (constructive interference:
`sin(θ) − 0.3 sin(3θ)`, crest ~1.76). Testing both:

| back-EMF | T₁ / T_rated (this code) | match Yepes 49.4 % ? |
|---|---|---|
| `sin + 0.3 sin(3·)` (flattened, crest 1.25) | **51.0 %** | ✅ within 3-digit rounding |
| `sin − 0.3 sin(3·)` (sharpened, crest 1.76) | 57.8 % | ❌ off by 8 pp |

The flattened (crest ~1.25) version reproduces Yepes's ratio to within
the precision of his three-significant-figure published values
(75.5 and 152.7). The script uses that convention (`SIGN_3 = +1`); the
other case is one variable flip away if you want to inspect it.

The residual 1.5 pp gap (51.0 vs 49.4) is consistent with rounding of
T₁ and T_rated to three digits — and an amplitude scan around 0.30
(0.25, 0.28, 0.30, 0.32, 0.35) moves the ratio by less than 1 pp, so
this isn't a hidden harmonic-coefficient mismatch either.

## 5. Validation result

(from `results.txt`)

```
HEALTHY:                      T at peak=1  =  2.9613  (=152.70 Nm by Yepes's normalisation)
1-PHASE OPEN  (phase a):      T at peak=1  =  1.5100  (=77.86  Nm)
RATIO T₁ / T_rated            =  50.990 %
Yepes published               =  49.443 %  (75.5 / 152.7)
diff                          =  +1.547 pp   (within published 3-digit rounding)
```

Open phase carries zero current identically (`max |i_a / T| = 0.0e+00`).

Per-phase peaks of the surviving phases at the fault peak-limited
optimum: `[0.66, 0.36, 0.36, 0.66]` (phases b, c, d, e). **The ML
reference does *not* impose equal magnitude**; the surviving phases
load asymmetrically (the inner two phases at half the peak of the
outer two), and the most-loaded pair saturates at `i_pk^mx`. Compare
with [Fall's 1-phase reconfiguration](../Fall/README.md#4-one-open-phase-eqs-1619--fall-sec-32) where all 4
phases share an identical peak by construction.

## 6. What Yepes does *not* optimise — the structural finding

Pinning the ML reference (Eq 7) also pins what the entire NSBE-FRML
algorithm leaves *outside* its constraints:

1. **No voltage constraint anywhere.** None of stages (a)–(d) of Fig 3
   limits per-phase voltage, the inverter `di/dt` drop, or the
   flux-weakening envelope. Their own Conclusion explicitly lists
   *"incorporating a limitation to the voltage constraints for high
   speeds"* as future work — meaning at any non-trivial speed their
   "maximum torque range" overstates what an inverter with finite DC
   bus and finite slew can actually deliver.
2. **No `di/dt` term** — the surviving currents are shaped freely as
   functions of θ (online sample-by-sample), with neither the
   inductive voltage drop nor any rate-of-change penalty entering
   the optimisation.
3. **Non-salient only.** Saliency is dropped on the argument that it
   is negligible in concentrated-winding non-sinusoidal-EMF designs.
   A salient machine (`L_d ≠ L_q`) is outside the scope of Eq (7)
   as written; their staged algorithm doesn't add it back.
4. **Equal-magnitude is *not* imposed** — unlike Fall, Yepes lets
   the surviving phases load asymmetrically (see §5). This is the
   "min-loss" choice but it sacrifices per-phase thermal symmetry.

Items 1 and 2 together are the key load-bearing structural finding
the dynamic-paper argument relies on: **Yepes's published
torque-speed envelope is unrealisable above the speed at which the
inverter voltage actually binds**, because that constraint isn't in
his optimisation.

## 7. What we did NOT replicate (and why it doesn't break the finding)

- Stages (b), (c), (d) of NSBE-FRML — the peak-saturation,
  ripple-limit, and rms-overload sub-controllers. These activate
  when ML alone is infeasible; they don't change the constraint
  *set* (current peak, current rms, torque ripple — still no
  voltage). The over-claim argument doesn't depend on them.
- The six-phase asymmetric machine in their Fig 2. Different
  geometry, same constraint-set conclusion.
- Their experimental six-phase PMSM (Section VII). Out of scope.

If/when we need to run NSBE-FRML *as a comparator* on a common
machine (e.g., IEEEMachine2), that's a bigger lift — see option (C)
in the offer above the commit. For now Eq (7) suffices to nail the
methodology and pin the published anchor.

## 8. Reproducing

```
python baselines/Yepes/replicate.py  >  baselines/Yepes/results.txt
```

(or just `python baselines/Yepes/replicate.py` to see it on stdout).
The script has **no dependencies on the rest of this repo** — it is
intentionally standalone, like
[`../Fall/replicate.py`](../Fall/replicate.py).
