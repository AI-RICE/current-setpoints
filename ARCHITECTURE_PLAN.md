# ARCHITECTURE_PLAN — full skeleton (signatures only)

Final target codebase, every file/class/function as a skeleton, for one last review before the refactor.
Consolidates `ARCHITECTURE.md` + `TRANSFORM_ARCHITECTURE.md`. Comments map each item to today's code.
Scope: merge `dynamic-mtpa` + `im-9phase-public`, refactor to this structure, **same results**; plus port
`ModelLossesSubstitution1` from `main`. Faults = single + two-open-phase; voltage = inverse-Park.

```
current_setpoints/
  models/        magnetic.py  electric.py  mechanical.py  drive.py
  simulation/    reconstruction.py  data.py
  optimization/  optimizer.py  representation.py  constraints.py  grid.py  regimes.py  efficiency.py
  utils/         plotting.py  plot_config.py  load_data.py  neural.py  nn_utils.py  loss_fit.py
examples/   tests/   docs/   data/   weights/
```

---

## models/magnetic.py
```python
class Flux(ABC):                                          # λ(i) provider (PMSM)
    def get_flux(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (flux_volt, flux_torq)
class ConstantFlux(Flux):
    def __init__(self, flux_volt: np.ndarray, flux_torq: np.ndarray) -> None: ...
class Flux_IEEEMachine2(ConstantFlux):
    def __init__(self) -> None: ...
class LUTFlux(Flux):                                      # NEW: Ansys/FEM flux map
    def __init__(self, flux_table: np.ndarray, axes: tuple[np.ndarray, ...]) -> None: ...

class MagneticModel(ABC):
    L_stat: np.ndarray                                    # (dim,dim) inductance L = ∂λ/∂i
    cross_coupling: np.ndarray                            # (dim,dim) J
    def operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray: ...   # ω·J·L  (the inductive part of U)
    def inductance(self) -> np.ndarray: ...               # L  (for ω·L·di/dθ)
    def bemf(self, omega: float) -> np.ndarray: ...       # ω·J·λ_pm
    def flux(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

class PMSMMagnetic(MagneticModel):                        # was parameters/flux.py + machines L_stat/mat_crossc
    def __init__(self, L_stat, cross_coupling, flux: Flux) -> None: ...
class IMMagnetic(MagneticModel):                          # was im_coupling.py (k_ir, slip) + L_mu
    def __init__(self, L_s, L_mu, cross_coupling, rotor_params) -> None: ...
    def k_ir(self, omega_r: float) -> np.ndarray: ...
    def slip_from_dq_foc(self, curr_dq: np.ndarray) -> float: ...   # ω_r = (R_r¹/L_r¹)(i_sd¹/i_sq¹)
```

## models/electric.py
```python
def substitution_loss_features(omega: float, volt_dq: np.ndarray, n_ppairs: int) -> tuple[float, float]: ...  # (f_v, f_h)

class ElectricModel(ABC):
    R_stat: np.ndarray
    k_v: float; k_h: float                                # ModelLossesSubstitution1 coefficients (fit via loss_fit)
    def iron_loss(self, omega: float, volt_dq: np.ndarray) -> float: ...   # k_v·f_v + k_h·f_h  (POST-HOC only)
    def copper_loss(self, curr_ph: np.ndarray) -> float: ...
class PMSMElectric(ElectricModel):
    def __init__(self, R_stat, k_v=0.0, k_h=0.0) -> None: ...
class IMElectric(ElectricModel):                          # + rotor copper (via k_ir coupling, see efficiency.py)
    def __init__(self, R_stat, R_rot, k_v=0.0, k_h=0.0) -> None: ...
```

## models/mechanical.py
```python
class MechanicalModel(ABC):
    n_phases: int; n_ppairs: int; curr_max: float
    @abstractmethod
    def torque(self, omega: float, curr_dq: np.ndarray) -> float: ...                 # exact, pointwise
    def torque_quadratic(self, omega: float) -> tuple[np.ndarray, np.ndarray, float]: # SHARED concrete
        ...   # = relocated extract_quadratic_torque: least-squares fit of self.torque (seed 0, scale 8, n=300)
    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]: ...          # was get_candidates (MTPA/FW)

class PMSMMechanical(MechanicalModel):                    # was ModelAnalytical
    def __init__(self, magnetic: "PMSMMagnetic", n_phases, n_ppairs) -> None: ...      # A = (m·pp/4)(JLₛ+LₛJᵀ)
    def torque(self, omega, curr_dq) -> float: ...        # iᵀA i + 2bᵀi ; b = (m·pp/4)(J·flux_torq)
class IMMechanical(MechanicalModel):                      # was im_model.ModelIMAnalytical
    def __init__(self, magnetic: "IMMagnetic", n_phases, n_ppairs) -> None: ...
    def _build_A(self, omega_r: float) -> np.ndarray: ...                              # A(ω_r) slip-dependent
    def torque(self, omega, curr_dq) -> float: ...
class NeuralMechanical(MechanicalModel):                  # was ModelNeural
    def __init__(self, net: "NeuralTorquePredictor", scaler, device) -> None: ...
    def torque(self, omega, curr_dq) -> float: ...        # predict_torque_neural; torque_quadratic → fitted surrogate
```

## models/drive.py
```python
class DriveModel(ABC):
    n_phases: int; n_harmonics: int; dim: int; n_ppairs: int
    curr_max: float; volt_max: float; omega_max: float
    magnetic: MagneticModel; electric: ElectricModel; mechanical: MechanicalModel
    # facade (only surface reconstruction/optimizer touch):
    def voltage_operator(self, omega, curr_dq) -> np.ndarray: ...   # electric.R_stat + magnetic.operator(omega,curr_dq) = U
    def inductance(self) -> np.ndarray: ...                         # magnetic.inductance()
    def bemf_dq(self, omega) -> np.ndarray: ...                     # magnetic.bemf(omega) = u
    def torque(self, omega, curr_dq) -> float: ...                  # mechanical.torque(...)
    def torque_quadratic(self, omega) -> tuple[np.ndarray, np.ndarray, float]: ...   # mechanical.torque_quadratic
    def seeds(self, guess=None) -> list[np.ndarray]: ...            # mechanical.seeds
    def set_max_pars(self, curr_max, volt_max, omega_max) -> None: ...

class PMSM5Phase(DriveModel):                             # the IEEEMachine2 prototype
    def __init__(self) -> None: ...
class IM9Phase(DriveModel):                               # the 15 kW IM prototype
    def __init__(self) -> None: ...
```

---

## simulation/reconstruction.py
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

class Reconstruction:
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

## optimization/representation.py
```python
class Representation(ABC):
    def curr_field(self, recon: Reconstruction, x: np.ndarray) -> np.ndarray: ...      # decision vars → i_dq(θ)
    def dcurr_dtheta(self, recon: Reconstruction, x: np.ndarray) -> np.ndarray: ...
    def n_vars(self, dim: int) -> int: ...
class PerAngle(Representation): ...                       # finite-diff dcurr_dtheta
class Fourier(Representation):                            # was fourier_optimizer basis machinery
    def design(self, theta, harmonics) -> tuple[np.ndarray, np.ndarray]: ...           # (Phi, dPhi)
    def reconstruct(self, coeffs, theta, harmonics) -> np.ndarray: ...
    def parseval_weights(self, harmonics) -> np.ndarray: ...
```

## optimization/constraints.py
```python
def current_constraint(omega, curr_dq, curr_max, recon: Reconstruction) -> float: ...  # margin ≥ 0 (eq 10c/20c)
def voltage_constraint(omega, curr_dq, volt_max, recon: Reconstruction) -> float: ...  # margin ≥ 0 (eq 10d/20d)
def rms_constraint(omega, curr_dq, rms_max, recon: Reconstruction) -> float: ...       # per-phase thermal
def ripple_constraint(torque_fn, T_target, budget) -> float: ...                       # dynamic torque ripple
```

## optimization/grid.py
```python
# A grid is a dict[str, np.ndarray] (optionally wrapped in a Grid dataclass). Keys today:
#   const_mech_speed, vec_torq, vec_omega, vec_torq_max (1,n_omega),
#   curr_dq_grid (dim,n_torq,n_omega), grid_curr_peak, grid_volt_peak,
#   grid_segments (= 3·n_volt_peaks + n_curr_peaks)   [recalculated mode adds grid_torq_neural]
#   NOTE: legacy grid_curr_ang_diff / grid_volt_ang_diff dropped (get_max_vals now → (I_peak, V_peak)).

def _init_grid_arrays(dim: int, n_torq: int, n_omega: int) -> dict[str, np.ndarray]: ...
def _fill_grid_point(grid, recon: Reconstruction, omega, curr_dq, idx_torq, idx_omega) -> None: ...
    # recon.get_max_vals → (I_peak, V_peak);  regimes.count_peaks → (n_curr, n_volt)
def calculate_grid(optimizer: BaseOptimizer, recon: Reconstruction, opts: dict,
                   mode: str = "standard", dict_grid_corr: dict | None = None) -> dict: ...
    # unifies calculate_grid + calculate_grid_im + _vdc_sweep.  mode: "standard" | "recalculated"
    # opts: n_torq, n_omega, torq_min, omega_min ; warm-starts cell→cell ; max-torque probe per ω
def get_correction_grid(dict_grid: dict, neural: "NeuralMechanical") -> dict: ...
    # adds grid_torq_neural = neural torque predicted on the baseline currents (analytical→neural flow)
def grid_to_data(grid: dict, k_skip: int) -> MachineData: ...
```
> Neural map flow (preserved): `calculate_grid(standard)` → `get_correction_grid` → `calculate_grid(mode="recalculated", dict_grid_corr=…)`.

## optimization/regimes.py
```python
def count_peaks(recon: Reconstruction, omega, curr_dq, rel_tol: float = 1e-3) -> tuple[int, int]: ...  # was Transform.count_peaks
def _count_waveform_peaks_at_limit(waveform, limit, rel_tol) -> int: ...
def fingerprint(...) -> Any: ...
def active_set(...) -> frozenset: ...
def active_set_tag(active) -> str: ...
```

## optimization/efficiency.py
```python
def add_efficiency_map(grid: dict, drive: DriveModel) -> dict: ...        # was im efficiency.py (copper-loss map)
# dynamic/iron_loss.py — post-hoc trajectory proxies (kept; for the iron/ studies):
def phase_flux_waveforms(recon, omega, curr_dq_grid) -> np.ndarray: ...
def eddy_loss(recon, omega, curr_dq_grid, k_e=1.0) -> float: ...
def hysteresis_loss(recon, omega, curr_dq_grid, k_h=1.0, beta=2.0) -> float: ...
def excess_loss(recon, omega, curr_dq_grid, k_x=1.0) -> float: ...
def iron_loss(recon, omega, curr_dq_grid, k_e=1.0, k_h=1.0, k_x=1.0, beta=2.0) -> float: ...
```

---

## utils/
```python
# neural.py
class NeuralTorquePredictor(nn.Module):
    def __init__(self, ...) -> None: ...
    def forward(self, x_normed: torch.Tensor) -> torch.Tensor: ...
def load_neural_model(...) -> tuple: ...
def predict_torque_neural(...) -> float: ...

# nn_utils.py
def train_model(...) -> Any: ...
def evaluate_model(...) -> Any: ...
def prepare_fold_dataloaders(...) -> Any: ...

# loss_fit.py  (ported from main, Substitution1 only)
def fit_substitution_loss(data, omega, n_ppairs) -> tuple[float, float]: ...           # least-squares (k_v, k_h)

# plotting.py  (merges plotting.py + plotting_dynamic.py — actual functions)
def plot_grid_segments(...) -> None: ...                   # regime/active-set map
def plot_global_performance(...) -> None: ...
def plot_residual_torque_error(dict_grid: dict, machine) -> None: ...
def plot_current_trajectories_split(...) -> None: ...
def plot_joule_losses_reduction(...) -> None: ...
def plot_dq_phase_combined(...) -> None: ...               # from plotting_dynamic: dq planes + phase waveforms
# plot_config.py — rcParams/styles
# load_data.py:
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
test_reconstruction.py   # was test_transform.py + test_phase_voltage.py
test_models.py           # DriveModel facade, torque A,b, torque_quadratic fit
test_optimizer.py        # static + dynamic (independent/active-set/fourier)
test_fault.py            # current_map (single inverse / two pseudoinverse), null-space realizability (open phase ≈ 0)
test_grid.py             # was test_grid.py ;  test_im.py — IM smoke tests (new)
```

---

### Review checklist (the deltas vs today)
- `transform.py` + `im_transform.py` → **one** `Reconstruction`; drive type lives in `DriveModel`.
- `parameters/*` + `optimization/models.py` → **`models/`** (magnetic/electric/mechanical/drive).
- dynamic free functions (`run_independent/active_set/fourier`) → **Optimizer subclasses** + `representation.py`.
- `count_peaks` → **`regimes.py`**;  `get_max_vals` stays in `Reconstruction`.
- Faults: `Fault` value object; **two-fault = pseudoinverse + null-space** (ported from rename lineage), inverse-Park voltage.
- Losses: **only `ModelLossesSubstitution1`** → `ElectricModel.iron_loss` (+ ported from main); `dynamic/iron_loss.py` kept in `efficiency.py`.
- Returns: `Solution` / `Waveforms` dataclasses.
```
