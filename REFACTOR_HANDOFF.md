# Unified Library Refactor — Handoff / Context

Paste-into-a-fresh-chat context for executing the `current_setpoints` refactor. Read this **plus**
the four design docs in the repo root: **`ARCHITECTURE_PLAN.md`** (authoritative target skeleton —
every file/class/function signature), **`ARCHITECTURE.md`** (rationale + module structure +
old→new mapping), **`TRANSFORM_ARCHITECTURE.md`** (the reconstruction layer in detail),
**`MERGE_PLAN.md`** (how the branches were consolidated).

---

## 0. Current repo state (start here)

- Branch **`integration/unified-library`** is at commit **`ed0dc67`** = the clean, verified merge of
  `main` + `dynamic-mtpa` + `im-9phase-public`. **The refactor itself is NOT done** — two attempts were
  made and fully reverted (they built *facades wrapping the old code* instead of the planned modules; see
  §6). We are back at the merge.
- The four design `.md` docs are present (untracked). They are the spec; this file summarizes + adds the
  context that isn't in them.
- **The merge is good and worth keeping**: `main` + `dynamic-mtpa` + `im-9phase`, one `__init__.py`
  conflict (`optimization/__init__.py`) resolved by **union** (keep all loss-model exports AND add the IM
  exports). Test suite verified **behavior-identical** to the source branches.

### Environment / how to run things
- Python env: **`conda run -n current-setpoints`**. `conda.exe` is at
  `C:\Users\matas\miniconda3\Scripts\conda.exe` (not on PATH).
- **Do NOT call the env's bare `python.exe`** — it DLL-crashes (numpy/scipy/torch native libs not on path
  without activation). Always go through `conda run`. `conda run` cannot take multi-line `python -c`; use a
  temp script file.
- Tests (PowerShell): `$env:PYTHONPATH=(Get-Location).Path; & "C:\Users\matas\miniconda3\Scripts\conda.exe" run -n current-setpoints --no-capture-output python -m pytest -q`
- **Baseline test result at the merge = `24 passed, 4 skipped, 3 failed`.** The 3 failures are
  **pre-existing** (not the merge's fault): 2× a neural test whose `DummyNet` fixture is incompatible with
  the installed torch (`_backward_hooks`/`_modules`), 1× a `count_peaks` plateau/edge case. Any refactor
  must keep behavior equivalent to this baseline.

### Branch map (what each line of work is)
- `main` — PMSM healthy base (5-phase), neural torque model, loss models (Bertotti/Substitution), 1st article.
- **`dynamic-mtpa` — THE GROUND TRUTH.** PMSM dynamic/"pointwise" mode + the canonical fault model
  (`dynamic/phase_voltage.py`, inverse-Park voltage) + `experiments/`, `baselines/`. 3rd article.
- `im-9phase-public` — 9-phase induction-motor stack (`im_*` modules), ZSC injection, efficiency map. 4th
  (IM) article. Additive.
- `fault-tolerant` — PMSM faults baked into `transform.py` (reduced-Clarke); **superseded/retired**, BUT it
  (and the user's local `current_setpoints_rename/` folder) hold the **two-open-phase null-space logic**
  that dynamic-mtpa lacks.
- `dev`, `fix-peak-detector-type-classification` — already contained in `main`.

---

## 1. Project & vision

One unified, public (GitHub) library for multiphase electric-drive setpoint optimization. **2-D feature
matrix:** drive types {5/7/9-phase PMSM, 9-phase IM} × modes {healthy, fault-tolerant, static, dynamic}.
End users: model any drive, compute setpoint/current maps, plot them, train their own neural models from
data — across all modes. Reused across multiple articles; couples to external tools (Ansys flux maps).
Needs a clean UI + docs (not yet built). **Refactor freedom is total**: the articles only *link* to the
library, they are not coupled to its API/structure — so renaming/restructuring/deleting is free, no
backward-compat needed.

---

## 2. Target architecture (summary of the plan docs)

**Two hard boundaries:** `models/` = *what the machine is* (physics); `optimization/` = *how we solve it*
(no physics). A `DriveModel` is **self-contained** (its magnetic/electric/mechanical parts belong to one
machine) and is *passed to* the reconstruction/optimizer through a thin **facade**; they never reach into
its internals.

```
current_setpoints/                         (~22 files target)
  models/        magnetic.py  electric.py  mechanical.py  drive.py   ← ALL physics
  simulation/    reconstruction.py  data.py
  optimization/  optimizer.py  representation.py  constraints.py  grid.py  regimes.py  efficiency.py
  utils/         plotting.py  plot_config.py  load_data.py  neural.py  nn_utils.py  loss_fit.py
examples/        (was experiments/ + baselines/ + dynamic/ drivers)
tests/  docs/  data/  weights/
```

- **`models/`** — `DriveModel` (ABC) holds params + a `MagneticModel` (flux λ(i), inductance L=∂λ/∂i, PM
  flux, cross-coupling J, IM `k_ir`/slip), an `ElectricModel` (R_s, iron loss), a `MechanicalModel`
  (torque). Facade: `voltage_operator(ω,i)`, `inductance()`, `bemf_dq(ω)`, `torque(ω,i)`,
  `torque_quadratic(ω)`, `seeds(guess)`. Concrete drives: **`PMSM5Phase`**, **`IM9Phase`** — they hold the
  real machine parameter values directly.
- **`simulation/reconstruction.py`** — `Reconstruction` (ONE class; drive type is in the `DriveModel`),
  `Fault` (value object), `Waveforms` (dataclass). Always all-phase; voltage = inverse-Park
  (fault-independent) + ZSC; current realization from `Fault`. Exposes `voltage_linear_maps` for the
  dynamic optimizer and `evaluate()→Waveforms` for out-of-optimizer assessment.
- **`optimization/`** — `Solution` (dataclass); `BaseOptimizer` → `StaticOptimizer`, `DynamicOptimizer` →
  `{IndependentOptimizer, ActiveSetOptimizer, FourierOptimizer}`; `representation.py`
  (`PerAngle`/`Fourier`); `constraints.py` (current/voltage/rms/ripple); `grid.py`
  (`calculate_grid`, optimizer-agnostic; `grid_to_data`; `get_correction_grid`); `regimes.py`
  (`count_peaks` + active-set tagging); `efficiency.py` (copper-loss map + the `dynamic/iron_loss.py`
  trajectory proxies).

---

## 3. Resolved design decisions (the contract)

- **Drive type → inheritance** (`PMSM5Phase`/`IM9Phase`). **Fault → a value object** `Fault(open_phases)`
  (healthy = `Fault()`). **Static vs dynamic → which Optimizer subclass + vector-vs-schedule input** (NO
  "mode" classes on the transform).
- **Per-phase voltage = inverse-Park, fault-independent.** This is what `dynamic-mtpa` does, and it
  **contradicts the user's paper** (which uses the reduced-Clarke `h_k` map, eqs 18/29/40). Working
  assumption: inverse-Park is correct (the paper's reduced-Clarke over-counts surviving-phase voltage at
  high speed and collapses the envelope). It's isolated in `Reconstruction` so it's a one-place fix if ever
  wrong. **NOT yet empirically confirmed** against the user's two-fault numbers — flag.
- **Faults: single + two-open-phase.** `Fault.current_map`: exact reduced-Clarke inverse when
  `surviving == dim` (single fault, 5-phase), Moore–Penrose pseudoinverse when `surviving < dim` (two
  faults). `Fault.extra_constraints`: null-space realizability `N·iₛ=0` (static) / `N·R(θ)·iₛ=0` (dynamic),
  added only when `surviving < dim`. The two-fault logic must be **ported from the
  `fault-tolerant`/`current_setpoints_rename` lineage** (dynamic-mtpa is **single-open-phase only** —
  its `fault_phase_map` inverts a square reduced-Clarke). Generalize `_full_clarke` to any `n_phases`
  (must reproduce `full_clarke_5phase` bit-for-bit at n=5).
- **Losses: keep ONLY `ModelLossesSubstitution1`** → `ElectricModel.iron_loss` (`k_v·f_v + k_h·f_h` from
  `volt_dq`). Drop `ModelLossParametric`, `…Substitution1Excess`, `…Substitution2`, `…Substitution2Excess`
  (and `utils/param_fit.py`, which only served them). Iron loss is **post-hoc** (benchmarking/efficiency),
  never in the optimizer. `dynamic/iron_loss.py` (trajectory eddy/hyst/excess proxies) is a *separate* tool
  — **keep it** (→ `efficiency.py`), it is NOT part of the pruned `ModelLoss*` family.
- **`count_peaks` → `regimes.py`** (it's classification). `get_max_vals` (peak magnitude) **stays** in
  `Reconstruction` — it's the optimizer's constraint primitive.
- **`torque_quadratic` = model-agnostic least-squares fit** of `torque()` (relocated
  `extract_quadratic_torque`: seed 0, scale 8, n=300, symmetric A). Exact for analytical/IM (torque is
  quadratic), best-fit surrogate for neural — so the Fourier path works with any model.
- **`Solution`** (optimizer return) and **`Waveforms`** (assessment) are dataclasses, built once per call
  (not in the hot loop). Hot loop stays on raw arrays / cached `(A,b,c)`.

---

## 4. The fault math (from the user's PDF — 5-phase PMSM)

Four cases, all sharing torque `T = iₛᵀA iₛ + 2bᵀiₛ`, `A = pp(m/4)(JLₛ+LₛJᵀ)` and
`vₛ = U iₛ + u`, `U = Rₛ + ωJLₛ`, `u = ωJΨ_PM`:
- **I Fault-free** — full 5×5 Clarke; **ZSC injection** `v_s0 = -½(min+max)` (eq 9); voltage limit uses it.
- **II Single open phase** — reduced 4×4 Clarke, **exact inverse**; per-phase current `iₖ = c⁻¹_fault,k R⁻¹(θ) iₛ`; no null-space constraint; ZSC disabled.
- **III Two adjacent open** / **IV Two non-adjacent open** — reduced 4×3 Clarke, **Moore–Penrose pseudoinverse**; plus **`N iₛ = 0`** where `N C_fault = 0` (left null-space); ZSC disabled.
- Paper's per-phase **voltage** uses `vₖ = hₖ(θ)vₛ` with the *reduced-Clarke* `hₖ`. **The code (and our
  decision) uses inverse-Park instead** — this is the one open physics question (see §3). The user reported
  the paper's reduced-Clarke voltage gives non-physical results; inverse-Park is the hypothesized fix.

---

## 5. Old → new file mapping (for "same results" traceability)

| Today (merged branch) | New home |
|---|---|
| `parameters/flux.py`, `machines.py` (L_stat, mat_crossc), `im_machine.py`, `im_coupling.py` | `models/magnetic.py` (+ params on the concrete drives in `models/drive.py`); `R_stat` → `models/electric.py` |
| `optimization/models.py` `ModelAnalytical`/`ModelNeural`; `im_model.py` `ModelIMAnalytical` | `models/mechanical.py` (`PMSMMechanical`/`NeuralMechanical`/`IMMechanical`) |
| `ModelLossesSubstitution1` (+ `substitution_loss_features`) | `models/electric.py`; **drop the other 4 loss models + `param_fit.py`** |
| `simulation/transform.py` + `im_transform.py` | `simulation/reconstruction.py` (one class via `DriveModel` facade) |
| `dynamic/phase_voltage.py`, `experiments/fourier/envelope_solver.py` (`fault_phase_map`) | `Reconstruction` / `Fault` |
| `transform.count_peaks` + `dynamic/regimes.py` | `optimization/regimes.py` |
| `optimization/optimizer.py` `MotorOptimizer` | `optimization/optimizer.py` `StaticOptimizer` (+ `BaseOptimizer`) |
| `dynamic/{independent,active_set,fourier}_optimizer.py` | `optimization/optimizer.py` `{Independent,ActiveSet,Fourier}Optimizer` + `representation.py` |
| `optimization/grid.py` + `im_grid.py` + `dynamic/0*_*.py` | `optimization/grid.py` (unified) + `examples/` |
| `utils/plotting.py` + `plotting_dynamic.py` | `utils/plotting.py` (merged) |
| `experiments/`, `baselines/`, `dynamic/` drivers | `examples/` (imports repointed) |
| `dynamic/iron_loss.py` (trajectory proxies) | `optimization/efficiency.py` (kept) |

---

## 6. CRITICAL — how to execute (lessons from the failed attempts)

1. **BUILD THE PLAN FOR REAL. Do NOT build facades that wrap the old code.** Both failed attempts created
   `models/` etc. as thin shells delegating to `parameters/` + `optimization/models.py`, leaving both
   layers in place → **more files, not fewer**, and unusable. The new modules must *contain* the inlined
   physics (the R/L values, flux, torque A,b, k_ir), and `parameters/`, `transform.py`,
   `optimization/models.py`, `im_model.py` must be **replaced/deleted**, not wrapped. The goal is **fewer,
   cleaner files (~22)**.
2. **Verify every stage — and READ THE ACTUAL CODE.** A green test suite is NOT proof of quality: it can
   pass simply because a facade wraps working old code. After each module: run `pytest` (must stay
   `24 passed / 4 skipped / 3 failed`) AND do a numeric **equivalence check vs the legacy** (e.g.
   `Reconstruction` vs `Transform`, `StaticOptimizer` vs `MotorOptimizer`, dynamic optimizers vs the
   `run_*` functions, and an end-to-end `calculate_grid` comparison). Inspect the generated files yourself.
3. **Go module-by-module and show the real files.** Don't mass-generate via unread subagents.
4. **Equivalence expectation:** the new pipeline matched the legacy to **~1e-6** in the interior operating
   region; the only divergence was torque-ceiling cells (flat-maximum non-uniqueness in `maximize_torque`
   + knife-edge feasibility at `T*=Tmax`) — that is expected, not a regression. Bonus: doing the all-phase
   `Reconstruction` makes PMSM ZSC injection work (4 previously-skipped tests pass).
5. **Suggested order:** models/ (real physics) → reconstruction.py → regimes.py + representation.py →
   optimizer.py (Static, verify vs MotorOptimizer) → optimizer.py (Dynamic, port `run_*`) → grid.py
   (verify end-to-end) → migrate tests + consumers to the new API → delete the old modules → folder reorg
   (examples/). Verify after each.

---

## 7. Numbers you'll need (transcribed from source — confirm against the repo)

- **PMSM `IEEEMachine2`**: `n_phases=5, n_ppairs=8`, limits `curr_max=30, volt_max=13, omega_max=1800`;
  `R_stat=diag([0.0191,0.0514,0.0805,0.0801])`; `L_stat=1e-3*[[0.0920,-0.0286,-0.0141,0.0010],[-0.0133,0.1090,-0.0008,-0.0092],[-0.0088,0.0037,0.0725,-0.0466],[-0.0041,-0.0053,0.0475,0.0722]]`.
  Flux: `flux_volt=[0.0115,0.0018,0,0]`, `flux_torq=[1.18255974e-2,-1.36757644e-3,8.94095382e-5,-4.58552615e-5]`.
- **IM `IM9Phase`**: `n_phases=9, n_ppairs=2, n_harmonics=2 (dim=4)`, limits `3 / 200 / 1500`;
  `R_s=5.0`, `R_r=diag([1.54,1.54,1.57,1.57])`, `L_mu=1e-3*diag([496,496,58.2,58.2])`,
  `L_s_sigma=1e-3*diag([15.1,15.1,13.6,13.6])`, `L_r_sigma=1e-3*diag([53.3,53.3,33.4,33.4])`;
  `L_s=L_mu+L_s_sigma`, `L_r=L_mu+L_r_sigma`, `R_stat=R_s*I`.
- `mat_crossc` (cross-coupling J): block-diagonal, harmonic `h=2i+1` block `[[0,-h],[h,0]]` along dq pairs.
- Torque: PMSM `A=(m·pp/4)(J·L_stat + L_stat·Jᵀ)`, `b=(m·pp/4)(J·flux_torq)`, `T=iᵀA i + 2bᵀi`.
  IM `A(ω_r)=(m·pp/2)(J·L_mu·k_ir(ω_r))`, `T=iᵀA(ω_r)i`, slip `ω_r=(R_r¹/L_r¹)(i_sd¹/i_sq¹)`.
- `n_theta` is rounded to a multiple of `2·n_phases` (exactness of the per-phase cyclic shift).

---

*Always confirm the above against the actual files on `integration/unified-library@ed0dc67` before relying
on it — this doc is a summary, the repo is the source of truth.*
