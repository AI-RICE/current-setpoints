# Refactor Architecture — Unified Drive Setpoint Library

**Plan:** (1) merge `dynamic-mtpa` + `im-9phase-public` (additive; one `__init__.py` conflict), then
(2) refactor into the structure below. **Hard invariant: identical functionality, logic, and numerical
results** — this is a *structural* refactor (DRY + speed + complete `DriveModel`s), not a behaviour change.

**Two boundaries:**
- `models/` = what the machine *is* (physics) + how it *responds* (forward model). `optimization/` = how we solve it (no physics).
- A `DriveModel` is **self-contained** and is *passed to* the forward model/optimizer through a thin **facade** — they never reach into its internals.

---

## 1. Package layout

```
current_setpoints/
  models/                           ← physics (machine description + forward model)
    machines.py      FluxModel, DriveModel, PMSM5Phase, IM9Phase, NeuralPMSM5Phase
    forward_model.py Fault, Waveforms, ForwardModel   [full detail in TRANSFORM_ARCHITECTURE.md]
  optimization/                     ← PURE solver machinery; zero physics
    optimizer.py     BaseOptimizer → StaticOptimizer, DynamicOptimizer{Independent,ActiveSet,Fourier};
                     Representation → PerAngle, Fourier; constraint primitives; Solution
    grid.py          calculate_grid, Grid, grid_to_data, get_correction_grid,
                     count_peaks, active_set, active_set_tag
    efficiency.py    add_efficiency_map (copper-loss map) + iron_loss trajectory proxies
    data.py          MachineData
  utils/
    plotting.py      (merges plotting.py + plotting_dynamic.py + plot_config.py)
    neural_model.py  NeuralTorquePredictor, load_neural_model, predict_torque_neural
    training_utils.py  train_model, evaluate_model, prepare_fold_dataloaders
    loss_fit.py      fit_substitution_loss (Substitution1 only)
    load_data.py     load_aggregated_csv_data
examples/                           (was experiments/ {chatter,fourier,iron} + baselines/ {Fall,Yepes} + dynamic drivers)
tests/   docs/   data/   weights/
```

---

## 2. `models/models.py` — FluxModel + DriveModel (one file)

Everything in one file, top-to-bottom: flux provider → drive base → concrete drives.

```python
# ── Flux (magnetic) ───────────────────────────────────────────────────────
# Flux and L_stat are a computational unit: L = ∂λ/∂i is derived FROM the flux model
# (constant Jacobian, numerical diff of LUT, or autograd through neural net).
# cross_coupling (J) is magnetic geometry — lives here alongside flux/L.

class FluxModel(ABC):
    cross_coupling: np.ndarray                          # (dim, dim)  J — rotation/coupling geometry
    def flux(self, curr_dq) -> tuple[np.ndarray, np.ndarray]: ...   # (flux_volt, flux_torq)  λ(i)
    def inductance(self, curr_dq=None) -> np.ndarray: ...           # L = ∂λ/∂i  (const for ConstantFlux)

class ConstantFlux(FluxModel): ...          # current case — arrays stored directly
class LUTFlux(FluxModel): ...               # FEM/Ansys map — L = numerical diff of table
# class NeuralFlux(FluxModel): ...          # future — neural λ(i), L via autograd Jacobian

# ── Drive (electric + mechanical, flat on DriveModel) ─────────────────────
# R_stat is a true constant — no reason to encapsulate it.
# torque / torque_quadratic / seeds are mechanical, overridden by NeuralPMSM5Phase.
# iron_loss / copper_loss are post-hoc only (never called by the optimizer).

class DriveModel(ABC):
    # identity / limits
    n_phases: int; n_harmonics: int; dim: int; n_ppairs: int
    curr_max: float; volt_max: float; omega_max: float
    # electrical
    R_stat: np.ndarray                                  # (dim, dim) stator resistance
    k_v: float; k_h: float                              # ModelLossesSubstitution1 coefficients (fit via loss_fit)
    # magnetic — encapsulated because L = ∂λ/∂i must be computed alongside λ
    flux: FluxModel
    # FACADE — the only surface reconstruction/optimizer touch:
    def voltage_operator(self, omega, curr_dq) -> np.ndarray: ...   # R_stat + ω·J·L·i  (PMSM const; IM adds L_mu·k_ir)
    def inductance(self, curr_dq=None) -> np.ndarray: ...           # flux.inductance(curr_dq)  (for ω·L·di/dθ)
    def bemf_dq(self, omega) -> np.ndarray: ...                     # ω·J·flux_volt  (zeros for IM)
    # mechanical
    @abstractmethod
    def torque(self, omega, curr_dq) -> float: ...                  # exact, pointwise
    def torque_quadratic(self, omega) -> tuple[np.ndarray, np.ndarray, float]: ...
        # SHARED concrete impl = relocated extract_quadratic_torque: LS fit of self.torque()
        # (seed 0, scale 8, n=300). Exact for analytical/IM; best-fit surrogate for neural.
    def seeds(self, guess=None) -> list[np.ndarray]: ...            # multi-start candidates (MTPA/FW)
    # loss (post-hoc only — never in optimizer):
    def iron_loss(self, omega, volt_dq) -> float: ...               # k_v·f_v + k_h·f_h  (ModelLossesSubstitution1)
    def copper_loss(self, curr_ph) -> float: ...                    # (used by efficiency.py)
    def set_max_pars(self, curr_max, volt_max, omega_max) -> None: ...

class PMSM5Phase(DriveModel): ...           # IEEEMachine2 prototype — ConstantFlux
class IM9Phase(DriveModel): ...             # 15 kW IM prototype — IMMagneticFlux (k_ir, slip, L_mu)
class NeuralPMSM5Phase(PMSM5Phase): ...     # overrides torque() with NeuralTorquePredictor
```

> **`torque_quadratic`:** today's `fourier_optimizer.extract_quadratic_torque` relocated — a
> model-agnostic LS fit of `torque()`. Exact for analytical/IM; best-fit surrogate for neural.
> The exact `torque(...)` is still used by static/per-angle optimizers. Both preserved → identical results.
>
> **Flux extensibility:** `LUTFlux` computes `L = numerical_diff(table)`; a future `NeuralFlux`
> computes `L = autograd_jacobian(net)`. Both satisfy the `FluxModel` interface — `DriveModel` never changes.
> The neural training loop is `calculate_grid → grid_to_data → MachineData → utils/nn_utils.train_model → weights/ → NeuralPMSM5Phase`.

---

## 3. `simulation/reconstruction.py` — forward-model evaluator

Full signatures in **TRANSFORM_ARCHITECTURE.md**. Summary:

```python
class Fault:                          # single value object; healthy = empty open_phases
    def __init__(self, open_phases=()): ...
    def current_map(self, recon) -> np.ndarray: ...        # surviving == dim: exact inverse;  surviving < dim: pseudoinverse C⁺
    def extra_constraints(self, recon, theta_idx) -> list: ...  # surviving < dim: null-space  (static N·iₛ=0 / dynamic N·R(θ)·iₛ=0)

@dataclass
class Waveforms: theta; curr_ph; volt_leg; volt_raw; volt_0; curr_dq; volt_dq; curr_peak; volt_peak

class Reconstruction:                 # ONE class (drive type encapsulated in DriveModel)
    def __init__(self, drive: DriveModel, n_theta=700, add_volt_0=False, fault: Fault = Fault()): ...
    def get_curr_ph(...); get_volt_dq(...); get_volt_ph(...); get_max_vals(...)   # all-phase; physics via drive facade
    def voltage_linear_maps(self, omega, theta_idx) -> (gU, gL, bV): ...          # per-angle optimizer seam
    def evaluate(self, omega, setpoint, dcurr_dtheta=None) -> Waveforms: ...      # out-of-optimizer assessment
```

- static vs dynamic = vector vs schedule passed in (no mode class).
- healthy vs fault = the `fault` value object.
- `count_peaks` lives in `regimes.py`; `get_max_vals` (magnitude) stays here as the constraint primitive.

---

## 4. `optimization/` — pure solver machinery

```python
@dataclass
class Solution:
    curr_dq: np.ndarray        # (dim,) static  OR  (N_opt, dim) dynamic trajectory
    torque: float; success: bool; diagnostics: dict

class BaseOptimizer(ABC):
    def __init__(self, recon: Reconstruction, opts=None): ...   # recon carries drive + fault (single source of truth)
    def minimize_current(self, torq_target, omega, guess=None) -> Solution: ...   # template
    def maximize_torque(self, omega, guess=None) -> Solution: ...                 # template
    def _solve(self, x0, cons): ...                                              # shared SLSQP (multi-start)
    @abstractmethod
    def _build_constraints(self, torq_target, omega): ...
    @abstractmethod
    def _initial_guess(self, torq_target, omega, guess): ...

class StaticOptimizer(BaseOptimizer): ...        # was MotorOptimizer; single vector, worst-case-over-θ
class DynamicOptimizer(BaseOptimizer):           # trajectory; per-angle + inductive ω·L·di/dθ
    representation: Representation
class IndependentOptimizer(DynamicOptimizer): ...   # was run_independent_per_angle (R0)
class ActiveSetOptimizer(IndependentOptimizer): ... # was run_active_set
class FourierOptimizer(DynamicOptimizer): ...       # was run_fourier + envelope_solver.solve_fault

# representation.py
class Representation(ABC):
    def curr_field(self, recon, x) -> np.ndarray: ...     # decision vars -> i_dq(θ)
    def dcurr_dtheta(self, recon, x) -> np.ndarray: ...   # finite-diff (PerAngle) | analytic (Fourier)
    def n_vars(self, dim) -> int: ...
class PerAngle(Representation): ...
class Fourier(Representation): ...                # was fourier_optimizer basis (fourier_design, reconstruct, Parseval)

# constraints.py — primitives + analytical Jacobians
def current_constraint(...); def voltage_constraint(...); def rms_constraint(...); def ripple_constraint(...)

# grid.py — ONE grid, optimizer-agnostic (unifies grid.py + im_grid.py + the dynamic drivers)
def calculate_grid(recon, optimizer, opts) -> "Grid": ...
def grid_to_data(grid, k_skip) -> MachineData: ...
def get_correction_grid(...): ...
class Grid: ...

# regimes.py — classification (count_peaks + active-set)
def count_peaks(...); def fingerprint(...); def active_set(...); def active_set_tag(...)

# efficiency.py
def add_efficiency_map(grid, drive) -> "Grid": ...     # was im efficiency.py
```

---

## 5. Old → new mapping (traceability for "same results")

| Today (`dynamic-mtpa` / `im-9phase-public`) | New home |
|---|---|
| `parameters/flux.py` (Flux, ConstantFlux, Flux_IEEEMachine2) | `models/machines.py` `ConstantFlux(FluxModel)` |
| `parameters/machines.py` `L_stat`, `mat_crossc` | `models/machines.py` `FluxModel.inductance` / `cross_coupling`;  `R_stat` → flat on `DriveModel`;  limits/`n_*` → `DriveModel` |
| `parameters/im_machine.py`, `im_coupling.py` (k_ir, slip) | `models/machines.py` `IM9Phase` + `IMMagneticFlux(FluxModel)` |
| `optimization/models.py` ModelAnalytical / ModelIMAnalytical | `models/machines.py` `PMSM5Phase.torque` / `IM9Phase.torque` |
| `optimization/models.py` ModelNeural | `models/machines.py` `NeuralPMSM5Phase.torque` |
| `ModelLossesSubstitution1` (**main only** — the ONE loss model kept) + `substitution_loss_features` + `utils/loss_fit` | `models/machines.py` `DriveModel.iron_loss` (k_v·f_v + k_h·f_h, post-hoc) + `utils/loss_fit.py` |
| `optimization/efficiency.py` (copper-loss map) | `optimization/efficiency.py` |
| `dynamic/iron_loss.py` (trajectory eddy/hyst/excess proxies, post-hoc) | **kept** → `optimization/efficiency.py` (loss/efficiency analysis for the `iron/` studies); unrelated to the `ModelLoss*` family |
| `ModelLossParametric`, `…Substitution1Excess`, `…Substitution2`, `…Substitution2Excess` | **dropped** |
| `utils/param_fit.py` (core-loss resistance fitting for ModelLossParametric) | **dropped** (ModelLossParametric dropped) |
| `simulation/transform.py` + `simulation/im_transform.py` | `models/forward_model.py` `ForwardModel` (one class via DriveModel facade) |
| `dynamic/phase_voltage.py` (phase_basis, voltage_linear_maps) | `ForwardModel.get_volt_ph` / `.voltage_linear_maps` |
| `experiments/fourier/envelope_solver.py` (fault_phase_map, solve_fault) | `Fault.current_map` + `FourierOptimizer` + constraints in `optimizer.py` |
| `transform.count_peaks` + `dynamic/regimes.py` | `optimization/grid.py` (count_peaks, active_set, active_set_tag merged in) |
| `optimization/optimizer.py` MotorOptimizer | `optimization/optimizer.py` StaticOptimizer |
| `dynamic/{independent,active_set,fourier}_optimizer.py` | `optimization/optimizer.py` {Independent,ActiveSet,Fourier} + Representation/PerAngle/Fourier + constraint primitives (all merged into optimizer.py) |
| `optimization/grid.py` + `optimization/im_grid.py` + `dynamic/0*_*.py` | `optimization/grid.py` (unified) + `examples/` |
| `simulation/data.py` MachineData | `optimization/data.py` |
| `utils/plotting.py` + `utils/plotting_dynamic.py` + `utils/plot_config.py` | `utils/plotting.py` (merged) |
| `utils/neural.py` | `utils/neural_model.py` (inference: NeuralTorquePredictor, load, predict) |
| `utils/nn_utils.py` | `utils/training_utils.py` (training: train_model, evaluate_model, dataloaders) |
| `experiments/`, `baselines/`, `dynamic/0*_*.py` | `examples/` (imports repointed) |

---

## 6. Behaviour-preservation checklist (must hold after refactor)

- **Voltage = inverse-Park, fault-independent.** This IS what `dynamic-mtpa` does (so it's preserved, not a
  deviation from ground truth) — but it CONTRADICTS the paper's reduced-Clarke `hₖ` (eqs 18/29/40).
  **Working assumption: inverse-Park is correct** (the reduced-Clarke voltage over-counts at high speed →
  envelope collapse). Isolated entirely in `Reconstruction`/`voltage_linear_maps`, so if it ever proves
  wrong the fix is one place — no restructuring (this is the placeholder).
- **Faults: single + two-open-phase, both in scope.** Single-fault current is preserved from `dynamic-mtpa`
  (exact reduced-Clarke inverse). **Two-fault is an extension ported from the
  `current_setpoints_rename`/`fault-tolerant` lineage**: `Fault.current_map` = pseudoinverse `C⁺`;
  `Fault.extra_constraints` = null-space realizability (`N·iₛ=0` static / `N·R(θ)·iₛ=0` dynamic), paired
  with inverse-Park voltage. `_full_clarke` generalized to any n_phases (must reproduce
  `full_clarke_5phase` bit-for-bit at n=5); adjacent vs non-adjacent = same code, different `open_phases`.
- **Preserve numerics:** the dq↔phase matrices, BEMF (`flux.get_flux(ω,0)`), the `(A,b,c)` quadratic torque,
  worst-case-over-θ static constraints, and per-angle dynamic constraints — relocated, not rederived.
- **Preserve solver behaviour:** multi-start seeds (`get_candidates`→`seeds`), warm-starting, **analytical
  Jacobians**, **per-phase RMS** limit, **ripple budget**, SLSQP options.
- **Regression gate:** snapshot reference grids (PMSM static/dynamic/single-fault + IM static) on the merged
  branch *before* refactoring; assert equality after each step.

## 7. Performance

- Precompute all θ-sampled matrices once in `ForwardModel.__init__`; cache `(A,b,c)` per ω.
- Optimizer inner loop stays on raw arrays / cached `(A,b,c)` — no facade calls per SLSQP eval; `Solution`
  built once per solve.
- All-phase ops are single vectorized matmuls; keep the analytical Jacobians; `calculate_grid` parallelizable per cell.

## 8. Resolved decisions (recap)

Drive type via inheritance · fault via value object · static/dynamic via optimizer + vector/schedule input ·
`Solution` dataclass · `count_peaks`/`active_set` merged into `grid.py` · `DriveModel` facade is the boundary ·
per-phase voltage = inverse-Park (fault-independent, isolated in `ForwardModel`) · two-fault = pseudoinverse
`C⁺` + null-space (`N·iₛ=0` / `N·R(θ)·iₛ=0`) in `Fault`, ported from the rename lineage.

**File structure:** `models/machines.py` (no sub-files). `FluxModel` (+ `cross_coupling`) is encapsulated
because `L = ∂λ/∂i` is computed from the flux model. `R_stat`, `k_v`, `k_h`, `torque()`, `iron_loss()` are flat
on `DriveModel` — no `ElectricModel` / `MechanicalModel` wrappers. `simulation/` folder eliminated —
`ForwardModel` lives in `models/forward_model.py`; `MachineData` moves to `optimization/data.py`.
`Representation`, `PerAngle`, `Fourier`, and constraint primitives merged into `optimization/optimizer.py`.
`utils/neural.py` → `neural_model.py` (inference); `utils/nn_utils.py` → `training_utils.py` (training);
`utils/plot_config.py` + `plotting_dynamic.py` merged into `plotting.py`; `param_fit.py` dropped.

**Loss model:** keep **only `ModelLossesSubstitution1`** (ported from main → `DriveModel.iron_loss(k_v·f_v + k_h·f_h)` + a
vs-neural benchmarking util); drop `ModelLossParametric` / `…Sub1Excess` / `…Sub2` / `…Sub2Excess`. Iron loss
is **post-hoc** (benchmarking/efficiency), never in the optimizer.
`dynamic/iron_loss.py` (trajectory eddy/hysteresis/excess proxies for the `iron/` studies) is a *separate*
post-hoc tool — **kept** as part of dynamic functionality, lives with `efficiency.py`; NOT part of the pruned
`ModelLoss*` family.
```
