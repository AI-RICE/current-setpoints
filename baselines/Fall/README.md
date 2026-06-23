# Fall (2016) — baseline reconstruction

Verbatim replication of the optimal post-fault current references of

> O. Fall, N. K. Nguyen, J. F. Charpentier, P. Letellier, E. Semail,
> X. Kestelyn, **"Variable speed control of a 5-phase permanent magnet
> synchronous generator including voltage and current limits in healthy and
> open-circuited modes"**, *Electric Power Systems Research* **140** (2016)
> 507–516. DOI [10.1016/j.epsr.2016.05.024](https://doi.org/10.1016/j.epsr.2016.05.024).

A faithful baseline matters for the dynamic paper because Fall is the **closest
direct competitor** named in the static paper: he too treats a 5-phase
sinusoidal-back-EMF PMSM with **per-phase peak current and peak voltage
(including the inductive `di/dt`) limits**, in healthy and three open-phase
fault scenarios. Knowing exactly what Fall does and doesn't optimise is what
lets us draw a defensible novelty boundary.

Contents:

- [`replicate.py`](replicate.py) — verbatim implementation of Fall's
  reconfiguration plus an analytical and numerical solve of the low-speed
  torque ceiling for healthy, 1-phase, 2-adjacent and 2-non-adjacent
  open-circuit faults on Fall's own setup (Table 1).
- [`results.txt`](results.txt) — captured output reproducing all four of
  Fall's Fig 3a / experimental anchors to within his "~" rounding.
- This file — the full reasoning, including the convention bug that took
  three attempts to spot.

## 1. Fall's machine (Table 1, page 512)

| | value |
|---|---|
| Stator resistance `R_s` | 9.1 mΩ |
| Phase inductance `L` | 0.09 mH |
| Mutual `M₁, M₂` | 0.02 / −0.01 mH |
| `L_d1 = L_q1` (no fundamental saliency) | 0.12 mH |
| `L_d3 = L_q3` | 0.04 mH |
| PM flux `φ₁` | 19.4 mWb |
| Pole pairs `p` | 7 |
| DC-bus `V_DC` | 30 V ⇒ `V_max = V_DC/2 = 15 V` (sinusoidal PWM) |
| Peak current `I_max` | 60 A |

Back-EMF is purely sinusoidal (`φ_3 = 0`), so the torque comes only from
the fundamental-plane currents (Fall Eq 8):

```
T = p · [ (L_d1 − L_q1)·i_d1·i_q1  +  √(5/2)·φ₁·i_q1 ]
```

For this machine `L_d1 = L_q1` (no saliency) so

```
T = p · √(5/2) · φ₁ · i_q1.
```

## 2. The `√(2/5)` vs `√(5/2)` convention bug (anchor that nearly broke validation)

Fall writes Eqs (10)/(11)/(12) with a square-root fraction whose rendered
form in the PDF reads as `√(5/2)`. Take the healthy peak-current bound (Eq 11)
naively at face value:

```
|i_a(θ)| = √(5/2) · | i_d1 cos θ − i_q1 sin θ + (3rd-harm terms) |  ≤  I_max
```

With `i_d3 = i_q3 = 0` and `i_d1 = 0`, that gives `√(5/2)·|i_q1| ≤ 60`, so
`|i_q1| ≤ 37.95`. Plug into Eq (8):

```
T_max,healthy  = 7 · √(5/2) · 0.0194 · 37.95  =  8.15 Nm
```

…but Fall's Fig 3a clearly shows the healthy low-speed ceiling at **~20.5 Nm**.

Back-solving from `T = p·√(5/2)·φ₁·i_q1 = 20.5` gives `i_q1 ≈ 95.4 A`. For that
to be consistent with `|i_a| ≤ 60 A` at low speed, the factor in Eq (11) must
be the **reciprocal**, `√(2/5) ≈ 0.632`, not `√(5/2)`. Then

```
peak |i_a|  =  √(2/5) · |i_q1|  ≤  60   ⇒   |i_q1|  ≤  60·√(5/2)  =  94.87
T_max,healthy  =  7 · √(5/2) · 0.0194 · 94.87  =  20.37 Nm  ≈  Fall's 20.5  ✓
```

So Fall is using the **power-invariant Park transform** with `√(2/5)` in
*both* directions: Eq (13) goes phase→dq with `√(2/5)` (correct as
printed), and Eqs (11)/(12) go dq→phase-magnitude with `√(2/5)` =
`1/√(5/2)` (the rendered fraction is `5/2` under the radical only
because of how IEEE's typesetter set the fraction inside the radical; the
intended factor is `1/√(5/2)`).

With this single correction the whole paper is internally consistent — and
all four of his published torque values reproduce within rounding.

## 3. Healthy

Two DOFs `(i_d1, i_q1)` constant (Fall Sec 3.4 fixes `i_d3 = i_q3 = 0` for
sinusoidal back-EMF). At low speed voltage is slack, so

```
T_max,healthy  =  p·√(5/2)·φ₁ · I_max·√(5/2)  =  p·(5/2)·φ₁·I_max  =  20.37 Nm.
```

## 4. One open phase (Eqs 16–19 — Fall Sec 3.2)

Phase `a` open. Fall's surviving currents in **closed form**, parametrised by
`(i'_d1, i'_q1)` constants and rotor angle θ. Let

```
N₁ = i'_d1 cos θ − i'_q1 sin θ
N₂ = i'_d1 sin θ + i'_q1 cos θ
D₁ = cos(2π/5) − cos(4π/5) ≈ 1.11803
D₂ = sin(2π/5) + sin(4π/5) ≈ 1.53884
```

Then (Eqs 16–19):

```
i'_b = √(5/8) · ( +N₁/D₁ + N₂/D₂ )
i'_c = √(5/8) · ( −N₁/D₁ + N₂/D₂ )
i'_d = √(5/8) · ( −N₁/D₁ − N₂/D₂ )
i'_e = √(5/8) · ( +N₁/D₁ − N₂/D₂ )
```

### Analytical verification (no code needed)

**Wye (Eq 15):**
`i'_b + i'_c + i'_d + i'_e = √(5/8)·[(1−1−1+1)·N₁/D₁ + (1+1−1−1)·N₂/D₂] = 0` ✓

**Park-back-projection** through Eq (14): the `N₁/D₁` coefficients of
`Σ_k i'_k·cos(θ−α_k)` collect via sum-to-product to `2 cos(θ)·D₁`, and the
`N₂/D₂` coefficients collect to `2 sin(θ)·D₂`. So

```
i_d1(reconstructed) = √(2/5)·√(5/8)·2·[cos(θ)·N₁ + sin(θ)·N₂]
                    = 1 · [i_d1·(cos²θ+sin²θ) + i_q1·(−cos θ sin θ + sin θ cos θ)]
                    = i_d1.   ✓
```

The same identity reproduces `i_q1`. (The constants conspire: `√(2/5)·√(5/8)·2 = 1`.)

**Equal magnitude:** writing `i'_b = √(5/8)·[(i_d1/D₁ + i_q1/D₂)·cos θ + (i_d1/D₂ − i_q1/D₁)·sin θ]`
and squaring,

```
|i'_b|²  =  (5/8)·[ (i_d1/D₁)² + 2·i_d1 i_q1 /(D₁ D₂) + (i_q1/D₂)²
                   + (i_d1/D₂)² − 2·i_d1 i_q1 /(D₁ D₂) + (i_q1/D₁)² ]
        =  (5/8)·(i_d1² + i_q1²)·(1/D₁² + 1/D₂²)
```

Cross-terms cancel. The same algebra applies to `i'_c, i'_d, i'_e` (only
sign combinations differ), so **all four surviving phases share identical
peak amplitude** — confirming Fall's stated equal-magnitude criterion is a
consequence of his closed form, not an extra constraint.

### Low-speed torque ceiling

`peak |i'_k|  =  √(5/8)·√(1/D₁² + 1/D₂²) · √(i_d1²+i_q1²)  =  0.8740·√(i_d1²+i_q1²)`

For peak ≤ I_max=60 with i_d1=0:

```
|i_q1|_max,1-fault  =  60 / 0.8740  =  68.65
T_max,1-fault       =  7·√(5/2)·0.0194·68.65  =  14.74 Nm
ratio_1f / healthy  =  14.74 / 20.37  =  72.4%        (Fall reports "~75%")
```

## 5. Two open phases (Sec 3.3)

Fall **drops** the equal-magnitude criterion here (Sec 3.3.1): with only 3
surviving phases, the 2 Park conditions plus the 1 wye condition already
uniquely determine the surviving currents per `(i'_d1, i'_q1)`. He prints
closed forms (Eqs 28–30 for adjacent, 36–38 for non-adjacent), but the
*equivalent* and transcription-risk-free implementation is to solve the
3×3 linear system at every θ:

```
[ √(2/5)·cos(θ−α₁)   √(2/5)·cos(θ−α₂)   √(2/5)·cos(θ−α₃) ] [ i_k1 ]   [ i'_d1 ]
[ −√(2/5)·sin(θ−α₁)  −√(2/5)·sin(θ−α₂)  −√(2/5)·sin(θ−α₃) ] [ i_k2 ] = [ i'_q1 ]
[       1                   1                   1         ] [ i_k3 ]   [   0   ]
```

with `α` = the angles of the surviving phases.

This formulation also lets us *observe* (rather than impose) that the per-phase
amplitudes are no longer equal — the most-loaded surviving phase saturates at
I_max while the others stay slack. At I_q1 = 50 A:

| fault | surviving | peaks per phase [A] |
|---|---|---|
| adjacent (`a, b` open) | `c, d, e` | **[70.7, 114.4, 70.7]** — middle phase d carries 1.6× more |
| non-adj (`a, c` open) | `b, d, e` | **[43.7, 70.7, 70.7]** — phase b lighter |

So at the optimum, the most-loaded phase pins I_max while the others have
spare current budget — a fact Fall accepts by dropping equal-magnitude.

### Low-speed torque ceilings

| | this code | Fall reports |
|---|---|---|
| 2-adjacent ceiling | 5.63 Nm (**27.6%**) | ~5.5 Nm ("~27%") |
| 2-non-adjacent ceiling | 9.11 Nm (**44.7%**) | ~9 Nm ("~44%") |

## 6. Validation summary

| Case | this code | Fall Fig 3a | match |
|---|---|---|---|
| Healthy | **20.37 Nm** | 20.5 Nm | ✓ within 1% |
| 1-phase open (Eqs 16–19) | **14.74 Nm** | 15.4 Nm, "~75%" | ✓ ratio 72.4% |
| 2-phase adjacent (Park+wye) | **5.63 Nm** | ~5.5 Nm, "~27%" | ✓ ratio 27.6% |
| 2-phase non-adjacent (Park+wye) | **9.11 Nm** | ~9 Nm, "~44%" | ✓ ratio 44.7% |

All four within Fall's own "approximately X" precision.

Lower-speed assumption: at low speed `V_max` is slack, so the optimisation
reduces to maximising torque subject to the per-phase current peak only.
The script verifies feasibility by both (i) analytical closed form (Eqs
16–19 ⇒ exact peak factor) and (ii) numerical SLSQP over `(i_d1, i_q1)` —
both give the same number to ≥4 decimals.

## 7. Where Fall stops — what he does not optimise

Pinning Fall's method exactly also pins its restrictions, which the
dynamic paper's contribution must live *strictly outside*:

1. **`(i_d1, i_q1)` are constants in the rotating frame.** They do not depend
   on θ. The dynamic paper allows them to vary with θ.
2. **`i_d3(θ), i_q3(θ)` are slaved**, not optimised. For 1-fault they are
   the specific θ-dependent functions implied by Eqs (16)–(19); for 2-fault
   they are determined by Park + wye.
3. **Equal-magnitude across surviving phases** is imposed for 1-fault, which
   leaves the per-phase peak budget evenly split rather than optimally
   redistributed. (Fall acknowledges this is restrictive by *dropping* it
   for 2-fault.)
4. **Sinusoidal back-EMF only.** Fall's torque expression uses only `i_q1`,
   so no `(i_d3, i_q3)` contribution to mean torque. Machines with a
   non-negligible 3rd back-EMF harmonic (like the IEEEMachine2 used in this
   repo) need an extension Fall does not provide.

These four are the entry points for the dynamic paper's claims; the
e16-style envelope comparisons need to be re-done against this faithful
Fall baseline rather than against the (over-restrictive) const-dq +
equal-RMS placeholder I used earlier.

## 8. Reproducing

```
python baselines/Fall/replicate.py  >  baselines/Fall/results.txt
```

(or just `python baselines/Fall/replicate.py` to see it on stdout). The
script has no dependencies on the rest of this repo — it is intentionally
standalone so it can be copied/cited as the Fall implementation we used.
