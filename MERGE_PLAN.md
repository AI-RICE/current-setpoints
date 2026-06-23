# Merge & Unification Plan — CurrentSetpoints

**Goal:** consolidate all active branches into `main` as one clean, unified, public
multiphase-drive setpoint-optimization library. Drive types **5/7/9-phase PMSM + induction (IM)**
× operating modes **healthy / fault-tolerant / dynamic (pointwise)**, with setpoint-map
computation, plotting, and neural-model training available across the matrix.

**Decision (agreed):** full *merge + architecture unification* — shared base classes so each
mode is written once and inherited by every drive type. Salvage items (`utils/sweep.py`
animations, `notebooks/fault_grids*.ipynb`) are **deferred** — explicitly out of scope for now.

---

## 1. Source inventory

| Branch | Role | Status into `main` |
|---|---|---|
| `main` | PMSM healthy base + neural model + loss models (Bertotti) + 1st article | target |
| `dynamic-mtpa` | Dynamic/pointwise mode + **canonical** fault model (`dynamic/phase_voltage.py`, inverse-Park; faults as current constraints). 3rd article. | **merges cleanly (0 conflicts)** |
| `im-9phase-public` | 9-phase induction stack (`im_*` parallel modules), ZSC injection, efficiency map. 4th article. | **1 conflict**: `optimization/__init__.py` |
| `fault-tolerant` | PMSM faults baked into `transform.py` (reduced-Clarke). **Superseded** by `dynamic-mtpa`. | **retire** (do not merge) |
| `dev`, `fix-peak-detector-type-classification` | already contained in `main` (0 ahead) | delete after confirming |

Local untracked folder `current_setpoints_rename` (= `fault-tolerant` package + 3 refined files)
is fully superseded — archive/remove it so it stops causing confusion.

**Guiding principle:** `dynamic-mtpa`'s *layered* fault model is canonical (cleaner physics,
generalizes to IM). The `fault-tolerant` transform-baked approach is dropped.

---

## 2. Phase 0 — Baseline & safety net  *(½ day)*

A unification refactor is only safe with a green regression suite to merge against.

1. Create integration branch: `git switch -c integration/unified-library main`.
2. Establish the test baseline in the project env (package not pip-installed — add repo root to
   `sys.path`): `conda run -n current-setpoints pytest -q`. Record pass/fail per test.
3. Inventory each branch's tests so none are lost: `main`/`im` have `tests/test_grid.py`,
   `tests/test_transform.py`; `dynamic-mtpa` adds `tests/test_phase_voltage.py`.
4. **Gate:** baseline tests green (or known-failing documented) before any merge.

---

## 3. Phase 1 — Clean merge (no behaviour change)  *(1 day)*

Get all code coexisting in `main`'s lineage, tests green, **before** touching architecture.
Use real merge commits (`--no-ff`) to preserve multi-author attribution and article provenance.

1. **Merge `dynamic-mtpa`** → integration branch. Confirmed conflict-free. Brings in
   `dynamic/`, `experiments/`, `baselines/`, `utils/plotting_dynamic.py`, `tests/test_phase_voltage.py`,
   plus `pyproject.toml`/`.gitignore` tweaks (review these two).
2. **Merge `im-9phase-public`** → integration branch. Resolve the single conflict in
   `optimization/__init__.py` by **keeping main's loss-model exports AND adding the IM exports**
   (the IM branch predates the loss models, so its version only *looks* like a deletion).
3. Run the full suite. Add minimal smoke tests for the IM stack (currently validated only by
   commit-message smoke notes on the 15 kW prototype) and for the dynamic optimizers.
4. **Gate:** all branches incorporated, `pytest` green, every public symbol importable.
   At this point `main` could ship as "everything coexists" even if Phase 2 slips.

---

## 4. Phase 2 — Architecture unification  *(the real work — 1–2 weeks)*

Eliminate the parallel PMSM/IM duplication and promote modes into first-class, shared layers.
Each step is independently test-guarded; commit per step so any regression is bisectable.

### 4a. Unify the drive-type axis (PMSM ↔ IM)

- **Machines:** extract a common `BaseMachine` ABC; make PMSM machines and `IM9Phase` subclasses.
  Fold `im_coupling.py` in as IM-specific physics.
- **Transform (highest value):** `Transform` and `IMTransform` already share `get_curr_ph`,
  `get_volt_ph` (ZSC injection), `get_max_vals`, and an *identical* `count_peaks`. They differ
  **only** in the voltage operator (PMSM flux/BEMF vs IM slip-dependent `ω_s·J·(L_s+L_μ·k_ir)`).
  → Extract `BaseTransform` with the shared methods + an abstract `get_volt_dq`; `PMSMTransform`
  and `IMTransform` override only the operator. This deletes the largest block of duplication.
- **Torque models:** already both subclass `BaseTorqueModel` — only normalise the
  `get_candidates` / `calculate_torque` interfaces.
- **Grid:** `calculate_grid_im` is `calculate_grid` + 3 adjustments (state dim, positive-`ω_s`
  max-torque probe, d+q seeding). Parametrise the single `calculate_grid` to derive these from
  the machine/transform, retiring `im_grid.py`.

### 4b. Unify the mode axis (healthy / fault / dynamic)

- Introduce a `current_setpoints/modes/` layer (or mixins) so a mode is written once and applies
  to any drive type:
  - **healthy** — the base constraint path.
  - **fault** — promote `dynamic/phase_voltage.py` + the `fault_phase_map` logic (currently in
    `experiments/fourier/envelope_solver.py`) into a reusable fault layer that wraps any
    `BaseTransform`. Verify it composes with both PMSM and IM.
  - **dynamic** — promote `dynamic/{independent,active_set,fourier}_optimizer.py` from the
    top-level research folder into the package (e.g. `optimization/dynamic/`).
- Keep the `experiments/` and `baselines/` article drivers, but repoint their imports at the
  unified API (and consider renaming `experiments/` → `examples/` for a public audience).

### 4c. Cross-cutting

- **Plotting:** merge `plotting.py` + `plotting_dynamic.py` into one mode-aware plotting module.
- **Neural training:** confirm `ModelNeural` + `nn_trainer.ipynb` data path works for every
  drive type / mode (it is the cross-cutting capability most likely to silently break).

**Proposed target layout:**
```
current_setpoints/
  parameters/   BaseMachine + PMSM/IM subclasses, flux, coupling
  simulation/   BaseTransform + PMSMTransform/IMTransform, data
  optimization/ models, optimizer, constraints, grid (unified), dynamic/, efficiency
  modes/        healthy / fault / dynamic  (shared across drive types)
  utils/        plotting (unified), plot_config, load_data, neural, nn_utils
examples/       article drivers + baselines
tests/  docs/
```

**Gate:** tests green after each sub-step; public API stable and documented in docstrings.

---

## 5. Phase 3 — Public polish  *(noted; schedule separately)*

User interface, README/quickstart, API docs (e.g. mkdocs/sphinx), worked examples per
drive-type×mode, and the Ansys-flux-map coupling example. Not part of the merge itself but the
reason the library is going public — flagged so it isn't forgotten.

---

## 6. Git mechanics & history

- Work entirely on `integration/unified-library`; never commit directly to `main`.
- Phase 1: `--no-ff` merges to preserve authorship (incl. supervisor's IM work) and article links.
- Phase 2: small, scoped, individually-reviewable commits (one per unification step).
- Land via a single PR `integration/unified-library → main` so the whole consolidation is
  reviewed as a unit. Delete merged branches (`dev`, `fix-peak-detector-type-classification`,
  `fault-tolerant`, `dynamic-mtpa`, `im-9phase-public`) only after the PR lands and tags are cut.
- Tag a release (e.g. `v1.0-unified`) once merged.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| IM stack has no dedicated tests | Add IM smoke/regression tests in Phase 1 before refactoring it |
| Refactor silently changes numerics | Snapshot reference grids pre-refactor; assert equality after each step |
| `BaseTransform` extraction breaks ZSC/count_peaks edge cases | They're already near-identical; extract behind existing tests, add fault/IM cases |
| Neural training path breaks under new structure | Explicit Phase 2c check on `nn_trainer` for each drive type |
| Example drivers stop running after refactor | Articles only *link* to the library — no API contract — but keep `examples/` runnable as a usability check |

> **Refactor freedom (confirmed):** the articles merely link to the library; they are **not** coupled
> to its API or structure. Phase 2 may therefore rename classes/modules, change signatures, and even
> rename the package freely — **no backward-compatibility shims, deprecation period, or frozen IM API.**

---

## 8. Open questions for supervisor

1. Confirm `fault-tolerant` is retired (its physics is superseded by `dynamic-mtpa`).
2. ~~Is the IM API frozen?~~ **Resolved — refactor is unconstrained (articles only link to the library).**
3. Package/public naming: keep `current_setpoints`, or rename to reflect the broader scope
   (drives + faults + dynamic + neural)? Now a pure branding choice — no reproducibility blocker.
4. Deferred salvage (`sweep.py` animations, `fault_grids` notebooks) — re-add later or drop?
