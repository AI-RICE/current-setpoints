# ARCHITECTURE_PLAN — full skeleton (signatures only)

Final target codebase, every file/class/function as a skeleton, for one last review before the refactor.
Consolidates `ARCHITECTURE.md` + `TRANSFORM_ARCHITECTURE.md`. Comments map each item to today's code.
Scope: merge `dynamic-mtpa` + `im-9phase-public`, refactor to this structure, **same results**; plus port
`ModelLossesSubstitution1` from `main`. Faults = single + two-open-phase; voltage = inverse-Park.
`simulation/` folder eliminated — physics lives in `models/`, grid data in `optimization/`.

```
current_setpoints/
  models/        machines.py  forward_model.py
  optimization/  optimizer.py  grid.py  efficiency.py  data.py
  utils/         plotting.py  neural_model.py  training_utils.py  loss_fit.py  load_data.py
examples/   tests/   docs/   data/   weights/
```

---

## models/machines.py
```python
# Flux and L_stat are a computational unit: L = ∂λ/∂i is derived FROM the flux model.
# cross_coupling (J) is magnetic geometry — lives here alongside flux/L.
# R_stat, k_v, k_h, torque(), iron_loss() are flat on DriveModel — no sub-model wrappers.

# ── Flux (magnetic) ───────────────────────────────────────────────────────

class FluxModel(ABC):
    cross_coupling: np.ndarray                            # (dim,dim) J — rotation/coupling geometry
    def flux(self, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (flux_volt, flux_torq)
    def inductance(self, curr_dq: np.ndarray | None = None) -> np.ndarray: ... # L = ∂λ/∂i

class ConstantFlux(FluxModel):                            # current PMSM case — L and λ are stored directly
    def __init__(self, flux_volt: np.ndarray, flux_torq: np.ndarray,
                 L_stat: np.ndarray, cross_coupling: np.ndarray) -> None: ...

class LUTFlux(FluxModel):                                 # Ansys/FEM flux map — L = numerical diff of table
    def __init__(self, flux_table: np.ndarray, axes: tuple[np.ndarray, ...],
                 cross_coupling: np.ndarray) -> None: ...

class IMMagneticFlux(FluxModel):                          # IM "flux" provider — holds k_ir, slip, L_mu
    def __init__(self, L_s: np.ndarray, L_mu: np.ndarray,
                 cross_coupling: np.ndarray, rotor_params: dict) -> None: ...
    def k_ir(self, omega_r: float) -> np.ndarray: ...     # was im_coupling.py
    def slip_from_dq_foc(self, curr_dq: np.ndarray) -> float: ...  # ω_r = (R_r¹/L_r¹)(i_sd¹/i_sq¹)

# ── Drive ─────────────────────────────────────────────────────────────────

def substitution_loss_features(omega: float, volt_dq: np.ndarray, n_ppairs: int) -> tuple[float, float]: ...  # (f_v, f_h)

class DriveModel(ABC):
    # identity / limits
    n_phases: int; n_harmonics: int; dim: int; n_ppairs: int
    curr_max: float; volt_max: float; omega_max: float
    # electrical — flat constants (no sub-model wrapper)
    R_stat: np.ndarray                                    # (dim,dim) stator resistance
    k_v: float; k_h: float                                # ModelLossesSubstitution1 coefficients
    # magnetic — encapsulated because L = ∂λ/∂i must be computed alongside λ
    flux: FluxModel
    # FACADE — the only surface reconstruction/optimizer touch:
    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray: ...  # R_stat + ω·J·L(·i)
    def inductance(self, curr_dq: np.ndarray | None = None) -> np.ndarray: ...        # flux.inductance(curr_dq)
    def bemf_dq(self, omega: float) -> np.ndarray: ...    # ω·J·flux_volt  (zeros for IM)
    # mechanical — flat methods, overridden by NeuralPMSM5Phase
    @abstractmethod
    def torque(self, omega: float, curr_dq: np.ndarray) -> float: ...                 # exact, pointwise
    def torque_quadratic(self, omega: float) -> tuple[np.ndarray, np.ndarray, float]: ...
        # SHARED concrete = relocated extract_quadratic_torque: LS fit of self.torque()
        # (seed 0, scale 8, n=300). Exact for analytical/IM; best-fit surrogate for neural.
    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]: ...         # multi-start (MTPA/FW)
    # loss — post-hoc only, never called by optimizer
    def iron_loss(self, omega: float, volt_dq: np.ndarray) -> float: ...              # k_v·f_v + k_h·f_h
    def copper_loss(self, curr_ph: np.ndarray) -> float: ...                          # (used by efficiency.py)
    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None: ...

class PMSM5Phase(DriveModel):                             # IEEEMachine2 prototype — ConstantFlux
    def __init__(self) -> None: ...
    def torque(self, omega, curr_dq) -> float: ...        # iᵀA i + 2bᵀi ; A = (m·pp/4)(JLₛ+LₛJᵀ), b = (m·pp/4)(J·flux_torq)

class IM9Phase(DriveModel):                               # 15 kW IM — IMMagneticFlux
    def __init__(self) -> None: ...
    def torque(self, omega, curr_dq) -> float: ...        # A(ω_r) slip-dependent

class NeuralPMSM5Phase(PMSM5Phase):                       # overrides torque() with NeuralTorquePredictor
    def __init__(self, net: "NeuralTorquePredictor", scaler, device) -> None: ...
    def torque(self, omega, curr_dq) -> float: ...        # predict_torque_neural; torque_quadratic → fitted surrogate
```

---

## models/forward_model.py
```python
PhaseArray = np.ndarray   # (n_phases, n_theta+1)

def _full_clarke(n_phases: int, n_harmonics: int) -> np.ndarray: ...      # generalizes full_clarke_5phase (exact at n=5)
def _reduced_clarke(n_phases: int, n_harmonics: int, open_phases: tuple[int, ...]) -> np.ndarray: ...

class Fault:                                              # single value object (healthy = empty)
    def __init__(self, open_phases: tuple[int, ...] = ()) -> None: ...
    @property
    def surviving_phases(self) -> tuple[int, ...]: ...
    def current_map(self, recon: "Reconstruction") -> np.ndarray:        # (n_theta+1, n_surv, dim)
        ...   # surviving == dim: exact reduced-Clarke inverse;  surviving < dim: Moore–Penrose pseudoinverse C⁺
    def extra_constraints(self, recon: "Reconstruction", theta_idx: np.ndarray) -> list:
        ...   # surviving < dim only: static N·iₛ=0  |  dynamic N·R(θ)·iₛ=0   (N = left null-space of C_fault)

@dataclass
class Waveforms:
    theta: np.ndarray; curr_ph: PhaseArray; volt_leg: PhaseArray; volt_raw: PhaseArray; volt_0: np.ndarray
    curr_dq: np.ndarray; volt_dq: np.ndarray; curr_peak: float; volt_peak: float

class ForwardModel:
    def __init__(self, drive: DriveModel, n_theta: int = 700, add_volt_0: bool = False, fault: Fault = Fault()) -> None: ...
    drive: DriveModel; fault: Fault
    vec_theta: np.ndarray; mat_dq_to_ph_all: np.ndarray; _phase_shift_samples: int; omega: float | None
    def _build_phase_mapping(self) -> None: ...
    def _set_omega(self, omega: float) -> None: ...
    # forward model (all-phase; physics via drive facade; current via fault.current_map):
    def get_curr_ph(self, omega, curr_dq) -> PhaseArray: ...
    def get_volt_dq(self, omega, curr_dq, dcurr_dtheta=None) -> np.ndarray: ...        # U·i + ω·L·di + bemf
    def get_volt_ph(self, omega, curr_dq, dcurr_dtheta=None) -> tuple[PhaseArray, np.ndarray, PhaseArray]: ...  # inverse-Park + ZSC
    def get_max_vals(self, omega, curr_dq, dcurr_dtheta=None) -> tuple[float, float]: ...   # (I_peak, V_peak) — constraint primitive
    def _zsc_min_max(self, volt_all: PhaseArray) -> np.ndarray: ...                    # -0.5(min+max) over phases
    def voltage_linear_maps(self, omega, theta_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # (gU, gL, bV)
    def evaluate(self, omega, setpoint, dcurr_dtheta=None) -> Waveforms: ...           # out-of-optimizer assessment
```

## simulation/data.py  (unchanged from today)
```python
class MachineData:                                        # grid → training/analysis container
    def __init__(self, torq, omega, segments: np.ndarray, curr_dq_grid: np.ndarray, k_skip: int | None = None) -> None: ...
        # torq (n_torq,); omega (n_omega, mech RPM); segments (n_torq,n_omega) = 3·n_volt_peaks + n_curr_peaks;
        # curr_dq_grid (dim, n_torq, n_omega); NaN = unfilled
    def _check_dimensions(self) -> None: ...
    def select_k(self, k_skip: int) -> None: ...          # subsample both axes
    @property
    def torq_grid(self) -> np.ndarray: ...
    @property
    def omega_grid(self) -> np.ndarray: ...
    @property
    def unique_segments(self) -> np.ndarray: ...
```

---

## optimization/optimizer.py
```python
@dataclass
class Solution:
    curr_dq: np.ndarray            # (dim,) static  OR  (N_opt, dim) trajectory
    torque: float; success: bool; diagnostics: dict

class BaseOptimizer(ABC):
    def __init__(self, recon: Reconstruction, opts: dict | None = None) -> None: ...   # recon = drive + fault
    def minimize_current(self, torq_target, omega, guess=None) -> Solution: ...        # template
    def maximize_torque(self, omega, guess=None) -> Solution: ...                      # template
    def _run_optimization(self, objective, constraints, candidates, opts) -> tuple: ...# shared SLSQP multi-start
    @abstractmethod
    def _build_constraints(self, torq_target, omega) -> list: ...
    @abstractmethod
    def _initial_guess(self, torq_target, omega, guess) -> list[np.ndarray]: ...

class StaticOptimizer(BaseOptimizer): ...                 # was MotorOptimizer; single vector, worst-case-over-θ
class DynamicOptimizer(BaseOptimizer):                    # trajectory; per-angle + inductive term
    representation: "Representation"
class IndependentOptimizer(DynamicOptimizer): ...         # was run_independent_per_angle (R0)
class ActiveSetOptimizer(IndependentOptimizer): ...       # was run_active_set
class FourierOptimizer(DynamicOptimizer): ...             # was run_fourier + envelope_solver.solve_fault
```

## optimization/optimizer.py  (also contains Representation, constraints)
```python
# ── Representation (merged in — only used by DynamicOptimizer subclasses) ─
class Representation(ABC):
    def curr_field(self, fwd: ForwardModel, x: np.ndarray) -> np.ndarray: ...      # decision vars → i_dq(θ)
    def dcurr_dtheta(self, fwd: ForwardModel, x: np.ndarray) -> np.ndarray: ...
    def n_vars(self, dim: int) -> int: ...
class PerAngle(Representation): ...                       # finite-diff dcurr_dtheta
class Fourier(Representation):                            # was fourier_optimizer basis machinery
    def design(self, theta, harmonics) -> tuple[np.ndarray, np.ndarray]: ...           # (Phi, dPhi)
    def reconstruct(self, coeffs, theta, harmonics) -> np.ndarray: ...
    def parseval_weights(self, harmonics) -> np.ndarray: ...

# ── Constraint primitives (merged in — only used by _build_constraints) ────
def current_constraint(omega, curr_dq, curr_max, fwd: ForwardModel) -> float: ...  # margin ≥ 0
def voltage_constraint(omega, curr_dq, volt_max, fwd: ForwardModel) -> float: ...  # margin ≥ 0
def rms_constraint(omega, curr_dq, rms_max, fwd: ForwardModel) -> float: ...       # per-phase thermal
def ripple_constraint(torque_fn, T_target, budget) -> float: ...                   # dynamic torque ripple

# ── Optimizers ───────────────────────────────────────────────────────────────
class StaticOptimizer(BaseOptimizer): ...                 # was MotorOptimizer; single vector, worst-case-over-θ
class DynamicOptimizer(BaseOptimizer):                    # trajectory; per-angle + inductive term
    representation: Representation
class IndependentOptimizer(DynamicOptimizer): ...         # was run_independent_per_angle (R0)
class ActiveSetOptimizer(IndependentOptimizer): ...       # was run_active_set
class FourierOptimizer(DynamicOptimizer): ...             # was run_fourier + envelope_solver.solve_fault
```

## optimization/grid.py  (also contains regime classification)
```python
# ── Regime classification (merged in — only consumed by _fill_grid_point) ──
def count_peaks(fwd: ForwardModel, omega, curr_dq, rel_tol: float = 1e-3) -> tuple[int, int]: ...  # was Transform.count_peaks
def _count_waveform_peaks_at_limit(waveform, limit, rel_tol) -> int: ...
def active_set(...) -> frozenset: ...
def active_set_tag(active) -> str: ...

# ── Grid ─────────────────────────────────────────────────────────────────────
# A grid is a dict[str, np.ndarray]. Keys:
#   const_mech_speed, vec_torq, vec_omega, vec_torq_max (1,n_omega),
#   curr_dq_grid (dim,n_torq,n_omega), grid_curr_peak, grid_volt_peak,
#   grid_segments (= 3·n_volt_peaks + n_curr_peaks)   [recalculated mode adds grid_torq_neural]
#   NOTE: legacy grid_curr_ang_diff / grid_volt_ang_diff dropped (get_max_vals → (I_peak, V_peak)).

def _init_grid_arrays(dim: int, n_torq: int, n_omega: int) -> dict[str, np.ndarray]: ...
def _fill_grid_point(grid, fwd: ForwardModel, omega, curr_dq, idx_torq, idx_omega) -> None: ...
def calculate_grid(optimizer: BaseOptimizer, fwd: ForwardModel, opts: dict,
                   mode: str = "standard", dict_grid_corr: dict | None = None) -> dict: ...
    # unifies calculate_grid + calculate_grid_im + _vdc_sweep.  mode: "standard" | "recalculated"
    # opts: n_torq, n_omega, torq_min, omega_min ; warm-starts cell→cell ; max-torque probe per ω
def get_correction_grid(dict_grid: dict, neural: "NeuralPMSM5Phase") -> dict: ...
    # adds grid_torq_neural = neural torque predicted on baseline currents (analytical→neural flow)
def grid_to_data(grid: dict, k_skip: int) -> "MachineData": ...
```
> Neural map flow (preserved): `calculate_grid(standard)` → `get_correction_grid` → `calculate_grid(mode="recalculated", dict_grid_corr=…)`.

## optimization/efficiency.py
```python
def add_efficiency_map(grid: dict, drive: DriveModel) -> dict: ...        # was im efficiency.py (copper-loss map)
# dynamic/iron_loss.py — post-hoc trajectory proxies (kept; for the iron/ studies):
def phase_flux_waveforms(fwd, omega, curr_dq_grid) -> np.ndarray: ...
def eddy_loss(fwd, omega, curr_dq_grid, k_e=1.0) -> float: ...
def hysteresis_loss(fwd, omega, curr_dq_grid, k_h=1.0, beta=2.0) -> float: ...
def excess_loss(fwd, omega, curr_dq_grid, k_x=1.0) -> float: ...
def iron_loss(fwd, omega, curr_dq_grid, k_e=1.0, k_h=1.0, k_x=1.0, beta=2.0) -> float: ...
```

## optimization/data.py  (moved from simulation/data.py — produced by grid_to_data)
```python
class MachineData:                                        # grid → training/analysis container
    def __init__(self, torq, omega, segments, curr_dq_grid, k_skip=None) -> None: ...
    def select_k(self, k_skip: int) -> None: ...
    @property
    def torq_grid(self) -> np.ndarray: ...
    @property
    def omega_grid(self) -> np.ndarray: ...
    @property
    def unique_segments(self) -> np.ndarray: ...
```

---

## utils/
```python
# neural_model.py  (inference — used at runtime by NeuralPMSM5Phase)
class NeuralTorquePredictor(nn.Module):
    def __init__(self, ...) -> None: ...
    def forward(self, x_normed: torch.Tensor) -> torch.Tensor: ...
def load_neural_model(...) -> tuple: ...
def predict_torque_neural(...) -> float: ...

# training_utils.py  (training — used only in notebooks, never at runtime)
def train_model(...) -> Any: ...
def evaluate_model(...) -> Any: ...
def prepare_fold_dataloaders(...) -> Any: ...

# loss_fit.py  (ported from main, Substitution1 only)
def fit_substitution_loss(data, omega, n_ppairs) -> tuple[float, float]: ...           # least-squares (k_v, k_h)

# plotting.py  (merges plotting.py + plotting_dynamic.py + plot_config.py)
def plot_grid_segments(...) -> None: ...                   # regime/active-set map
def plot_global_performance(...) -> None: ...
def plot_residual_torque_error(dict_grid: dict, machine) -> None: ...
def plot_current_trajectories_split(...) -> None: ...
def plot_joule_losses_reduction(...) -> None: ...
def plot_dq_phase_combined(...) -> None: ...               # dq planes + phase waveforms (was plotting_dynamic)

# load_data.py
def load_aggregated_csv_data(file_path: str, col_map: dict[str, str]) -> "pd.DataFrame": ...
```

---

## examples/   (was experiments/ {chatter,fourier,iron} + baselines/ {Fall,Yepes} + dynamic drivers 01–03)
```
fourier/  iron/  chatter/  baselines/Fall  baselines/Yepes  low_speed_sweep.py  speed_sweep.py  active_set_demo.py
# library code only — these scripts import the package; not part of the installed package.
```

## tests/
```python
test_forward_model.py    # was test_transform.py + test_phase_voltage.py
test_machines.py         # DriveModel facade, torque A,b, torque_quadratic fit
test_optimizer.py        # static + dynamic (independent/active-set/fourier)
test_fault.py            # current_map (single inverse / two pseudoinverse), null-space realizability (open phase ≈ 0)
test_grid.py             # was test_grid.py ;  IM smoke tests (new)
```

---

### Review checklist (the deltas vs today)
- `transform.py` + `im_transform.py` → **one** `ForwardModel` in `models/forward_model.py`; drive type lives in `DriveModel`.
- `parameters/*` + `optimization/models.py` → **`models/machines.py`** (one file: `FluxModel` + `DriveModel` + concrete drives).
- `simulation/` folder eliminated — `ForwardModel` in `models/`; `MachineData` in `optimization/data.py`.
- No `ElectricModel`/`MechanicalModel`/`MagneticModel` wrappers — `R_stat`/losses/torque are flat on `DriveModel`; only `FluxModel` is encapsulated (because `L = ∂λ/∂i` is computed from it).
- `Representation`, `PerAngle`, `Fourier`, constraint primitives merged into `optimizer.py` (only used internally).
- `count_peaks`, `active_set`, `active_set_tag` merged into `grid.py` (only consumed by `_fill_grid_point`).
- dynamic free functions (`run_independent/active_set/fourier`) → **Optimizer subclasses** in `optimizer.py`.
- `utils/neural.py` → `neural_model.py` (inference); `utils/nn_utils.py` → `training_utils.py` (training).
- `utils/plot_config.py` + `plotting_dynamic.py` merged into `plotting.py`; `param_fit.py` **dropped** (ModelLossParametric dropped).
- `count_peaks` → **`regimes.py`**;  `get_max_vals` stays in `Reconstruction`.
- Faults: `Fault` value object; **two-fault = pseudoinverse + null-space** (ported from rename lineage), inverse-Park voltage.
- Losses: **only `ModelLossesSubstitution1`** → `ElectricModel.iron_loss` (+ ported from main); `dynamic/iron_loss.py` kept in `efficiency.py`.
- Returns: `Solution` / `Waveforms` dataclasses.
```
