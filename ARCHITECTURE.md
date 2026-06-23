# Refactor Architecture — Unified Drive Setpoint Library

**Plan:** (1) merge `dynamic-mtpa` + `im-9phase-public` (additive; one `__init__.py` conflict), then
(2) refactor into the structure below. **Hard invariant: identical functionality, logic, and numerical
results** — this is a *structural* refactor (DRY + speed + complete `DriveModel`s), not a behaviour change.

**Two boundaries:**
- `models/` = what the machine *is* (physics). `optimization/` = how we solve it (no physics).
- A `DriveModel` is **self-contained** (its magnetic/electric/mechanical parts belong to one machine) and is
  *passed to* the reconstruction/optimizer through a thin **facade** — they never reach into its internals.

---

## 1. Package layout

```
current_setpoints/
  models/                           ← ALL physics; one self-contained DriveModel per drive type
    magnetic.py     MagneticModel    : PMSMMagnetic, IMMagnetic
    electric.py     ElectricModel    : R_s, iron/copper loss
    mechanical.py   MechanicalModel  : PMSMMechanical, IMMechanical, NeuralMechanical
    drive.py        DriveModel       : PMSM5Phase (IEEEMachine2), IM9Phase   (7/9-ph PMSM = same classes, diff params)
  simulation/
    reconstruction.py  Reconstruction, Fault, Waveforms     [full detail in TRANSFORM_ARCHITECTURE.md]
    data.py            MachineData
  optimization/                     ← PURE solver machinery; zero physics
    optimizer.py    BaseOptimizer → StaticOptimizer, DynamicOptimizer{Independent,ActiveSet,Fourier}; Solution
    representation.py  Representation → PerAngle, Fourier
    constraints.py  current / voltage / rms / ripple primitives (+ analytical Jacobians)
    grid.py         calculate_grid (optimizer-agnostic), Grid, grid_to_data, get_correction_grid
    regimes.py      count_peaks + active-set / regime classification
    efficiency.py   add_efficiency_map (copper-loss map)
  utils/
    plotting.py (merges plotting + plotting_dynamic)  plot_config.py  load_data.py  neural.py  nn_utils.py
examples/                           (was experiments/ {chatter,fourier,iron} + baselines/ {Fall,Yepes} + dynamic drivers)
tests/   docs/   data/   weights/
```

---

## 2. `models/` — DriveModel + three sub-models

```python
# ── magnetic.py ───────────────────────────────────────────────────────────
class MagneticModel(ABC):
    L_stat: np.ndarray                                  # (dim, dim) stator inductance  (L = ∂λ/∂i)
    cross_coupling: np.ndarray                          # (dim, dim) rotation/coupling J
    def operator(self, omega, curr_dq) -> np.ndarray: ...   # ω·J·L  (PMSM const; IM adds L_mu·k_ir(ω_r))
    def bemf(self, omega) -> np.ndarray: ...                # ω·J·λ_pm  (PMSM);  zeros (IM)
    def flux(self, omega, curr_dq) -> tuple[np.ndarray, np.ndarray]: ...   # λ(i) = (flux_volt, flux_torq)
class PMSMMagnetic(MagneticModel): ...     # holds a Flux provider:  ConstantFlux / Flux_IEEEMachine2 / LUTFlux(Ansys)
class IMMagnetic(MagneticModel): ...       # holds k_ir(ω_r), slip_from_dq_foc, L_mu  (was im_coupling.py)

# ── electric.py ───────────────────────────────────────────────────────────
class ElectricModel(ABC):
    R_stat: np.ndarray
    def iron_loss(self, omega, volt_dq) -> float: ...       # ONLY ModelLossesSubstitution1 (ported from main):
                                                            #   k_v·f_v + k_h·f_h (eddy+hysteresis from volt_dq).
                                                            #   POST-HOC (vs-neural benchmark / efficiency), NOT in optimizer.
    def copper_loss(self, curr_ph) -> float: ...            # (used by efficiency.py)

# ── mechanical.py ─────────────────────────────────────────────────────────
class MechanicalModel(ABC):
    @abstractmethod
    def torque(self, omega, curr_dq) -> float: ...                                  # exact, pointwise
    def torque_quadratic(self, omega) -> tuple[np.ndarray, np.ndarray, float]: ...  # (A,b,c): T = iᵀA i + bᵀi + c
        # SHARED concrete impl = relocated extract_quadratic_torque: least-squares fit of self.torque over
        # random samples (seed 0, scale 8, n=300). EXACT for analytical/IM (torque is quadratic); best-fit
        # SURROGATE for neural → the Fourier path works with ANY mechanical model, no per-subclass override.
    def seeds(self, guess=None) -> list[np.ndarray]: ...                            # multi-start candidates
class PMSMMechanical(MechanicalModel): ...     # was ModelAnalytical
class IMMechanical(MechanicalModel): ...       # was ModelIMAnalytical; A(ω_r) slip-dependent
class NeuralMechanical(MechanicalModel): ...   # was ModelNeural; wraps utils/neural.NeuralTorquePredictor

# ── drive.py ──────────────────────────────────────────────────────────────
class DriveModel(ABC):
    n_phases: int; n_harmonics: int; dim: int; n_ppairs: int
    curr_max: float; volt_max: float; omega_max: float
    magnetic: MagneticModel; electric: ElectricModel; mechanical: MechanicalModel
    # FACADE — the only surface reconstruction/optimizer touch:
    def voltage_operator(self, omega, curr_dq) -> np.ndarray: ...   # = electric.R_stat + magnetic.operator(...)
    def inductance(self) -> np.ndarray: ...                         # = magnetic.L_stat   (for ω·L·di/dθ)
    def bemf_dq(self, omega) -> np.ndarray: ...                     # = magnetic.bemf(omega)
    def torque(self, omega, curr_dq) -> float: ...                  # = mechanical.torque(...)
    def torque_quadratic(self, omega) -> tuple[np.ndarray, np.ndarray, float]: ...  # = mechanical.torque_quadratic
    def seeds(self, guess=None) -> list[np.ndarray]: ...            # = mechanical.seeds
class PMSM5Phase(DriveModel): ...     # the IEEEMachine2 prototype
class IM9Phase(DriveModel): ...       # the 15 kW IM prototype
```

> **`torque_quadratic` (incl. IM & neural):** it is today's `fourier_optimizer.extract_quadratic_torque`
> relocated — a **model-agnostic least-squares fit** of `torque()` (it calls `calculate_torque`, never
> reads `A`/`b`). Exact for analytical/IM (torque is quadratic); a best-fit **surrogate** for neural. The
> *exact* nonlinear `torque(...)` is still used by the static/per-angle optimizers. Both preserved verbatim
> → identical results, and the Fourier path works with any mechanical model (incl. neural) with no override.
>
> **Data-driven sub-models:** the three-way split makes each sub-model source-agnostic behind the same
> facade — magnetic can be analytical or `LUTFlux` (Ansys/FEM flux map); mechanical can be `PMSMMechanical`
> or `NeuralMechanical` (a trained `NeuralTorquePredictor`). The neural training loop is
> `calculate_grid → grid_to_data → MachineData → utils/nn_utils.train_model → weights/ → NeuralMechanical`.

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
| `parameters/flux.py` (Flux, ConstantFlux, Flux_IEEEMachine2) | `models/magnetic.py` (PMSMMagnetic holds Flux) |
| `parameters/machines.py` `L_stat`, `mat_crossc` | `models/magnetic.py`;  `R_stat` → `models/electric.py`;  limits/`n_*` → `models/drive.py` |
| `parameters/im_machine.py`, `im_coupling.py` (k_ir, slip) | `models/magnetic.py` IMMagnetic + `models/drive.py` IM9Phase |
| `optimization/models.py` ModelAnalytical / ModelIMAnalytical | `models/mechanical.py` PMSMMechanical / IMMechanical |
| `optimization/models.py` ModelNeural | `models/mechanical.py` NeuralMechanical |
| `ModelLossesSubstitution1` (**main only** — the ONE loss model kept) + `substitution_loss_features` + `utils/loss_fit` | `models/electric.py` `iron_loss` + vs-neural benchmarking util (port from main) |
| `optimization/efficiency.py` (copper-loss map) | `optimization/efficiency.py` |
| `dynamic/iron_loss.py` (trajectory eddy/hyst/excess proxies, post-hoc) | **kept** → `optimization/efficiency.py` (loss/efficiency analysis for the `iron/` studies); unrelated to the `ModelLoss*` family |
| `ModelLossParametric`, `…Substitution1Excess`, `…Substitution2`, `…Substitution2Excess` | **dropped** |
| `simulation/transform.py` + `simulation/im_transform.py` | `simulation/reconstruction.py` (one class via DriveModel facade) |
| `dynamic/phase_voltage.py` (phase_basis, voltage_linear_maps) | `Reconstruction.get_volt_ph` / `.voltage_linear_maps` |
| `experiments/fourier/envelope_solver.py` (fault_phase_map, solve_fault) | `Fault.current_map` + `FourierOptimizer` + `constraints.py` |
| `transform.count_peaks` + `dynamic/regimes.py` | `optimization/regimes.py` |
| `optimization/optimizer.py` MotorOptimizer | `optimization/optimizer.py` StaticOptimizer |
| `dynamic/{independent,active_set,fourier}_optimizer.py` | `optimization/optimizer.py` {Independent,ActiveSet,Fourier} + `representation.py` |
| `optimization/grid.py` + `optimization/im_grid.py` + `dynamic/0*_*.py` | `optimization/grid.py` (unified) + `examples/` |
| `utils/plotting.py` + `utils/plotting_dynamic.py` | `utils/plotting.py` (merged) |
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

- Precompute all θ-sampled matrices once in `Reconstruction.__init__`; cache `(A,b,c)` per ω.
- Optimizer inner loop stays on raw arrays / cached `(A,b,c)` — no facade calls per SLSQP eval; `Solution`
  built once per solve.
- All-phase ops are single vectorized matmuls; keep the analytical Jacobians; `calculate_grid` parallelizable per cell.

## 8. Resolved decisions (recap)

Drive type via inheritance · fault via value object · static/dynamic via optimizer + vector/schedule input ·
losses in `ElectricModel` · `Solution` dataclass · `count_peaks`→`regimes` · `DriveModel` facade is the boundary ·
per-phase voltage = inverse-Park (fault-independent, isolated in `Reconstruction`) · two-fault = pseudoinverse
`C⁺` + null-space (`N·iₛ=0` / `N·R(θ)·iₛ=0`) in `Fault`, ported from the rename lineage.

**Loss model:** keep **only `ModelLossesSubstitution1`** (ported from main → `ElectricModel.iron_loss` + a
vs-neural benchmarking util); drop `ModelLossParametric` / `…Sub1Excess` / `…Sub2` / `…Sub2Excess`. Iron loss
is **post-hoc** (benchmarking/efficiency), never in the optimizer.
`dynamic/iron_loss.py` (trajectory eddy/hysteresis/excess proxies for the `iron/` studies) is a *separate*
post-hoc tool — **kept** as part of dynamic functionality, lives with `efficiency.py`; NOT part of the pruned
`ModelLoss*` family.
```
