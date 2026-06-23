# `simulation/reconstruction.py` — architecture (signatures only, no implementation)

`Reconstruction` is a pure forward-model **evaluator**: given a `DriveModel` and a current setpoint,
it produces all-phase current/voltage waveforms, peaks, and the linear maps the optimizer needs.

- **Drive type** is encapsulated in the `DriveModel` → `Reconstruction` is **one class** (no PMSM/IM subclasses).
- **Healthy ↔ fault** is one injected value object, `fault` (swap it; not a class hierarchy).
- **Static ↔ dynamic** is **not a mode at all** — it's whether you pass a single dq vector or a
  per-angle schedule. `di/dθ` is zero for the former, computed for the latter.

```python
import numpy as np
from abc import ABC
from dataclasses import dataclass
from ..models import DriveModel               # the complete drive (magnetic + electric + mechanical)

PhaseArray = np.ndarray   # shape (n_phases, n_theta+1) — every phase, full electrical period


# ───────────────────────────────────────────────────────────────────────────
# module-level helpers
# ───────────────────────────────────────────────────────────────────────────
def _full_clarke(n_phases: int, n_harmonics: int) -> np.ndarray: ...
def _reduced_clarke(n_phases: int, n_harmonics: int, open_phases: tuple[int, ...]) -> np.ndarray: ...


# ───────────────────────────────────────────────────────────────────────────
# FAULT  —  single value object (healthy = empty open_phases); injected via `fault=`
# ───────────────────────────────────────────────────────────────────────────
class Fault:
    def __init__(self, open_phases: tuple[int, ...] = ()) -> None: ...      # () = healthy
    @property
    def surviving_phases(self) -> tuple[int, ...]: ...
    def current_map(self, recon: "Reconstruction") -> np.ndarray: ...       # (n_theta+1, n_surv, dim)
        # surviving == dim: exact reduced-Clarke inverse;  surviving < dim: Moore–Penrose pseudoinverse C⁺
    def extra_constraints(self, recon: "Reconstruction", theta_idx: np.ndarray) -> list: ...
        # null-space realizability, added IFF surviving < dim:
        #   static  → N·i_s = 0        (N = left null-space of C_fault)
        #   dynamic → N·R(θ)·i_s = 0    (per-angle; the static form is only valid at θ=0)


# ───────────────────────────────────────────────────────────────────────────
# WAVEFORMS  —  rich result for assessment OUTSIDE the optimizer (plotting/diagnostics)
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class Waveforms:
    theta:     np.ndarray      # (n_theta+1,)
    curr_ph:   PhaseArray      # per-phase current waveforms
    volt_leg:  PhaseArray      # per-phase leg voltage (with ZSC)
    volt_raw:  PhaseArray      # per-phase voltage (no ZSC)
    volt_0:    np.ndarray      # zero-sequence component (1-D)
    curr_dq:   np.ndarray      # (n_theta+1, dim) dq current trajectory  → d/q-plane
    volt_dq:   np.ndarray      # (n_theta+1, dim) dq voltage trajectory  → d/q-plane
    curr_peak: float
    volt_peak: float


# ───────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION  —  one class; physics comes from the injected DriveModel facade
# ───────────────────────────────────────────────────────────────────────────
class Reconstruction:
    def __init__(
        self,
        drive: DriveModel,
        n_theta: int = 700,                 # rounded to a multiple of 2*n_phases
        add_volt_0: bool = False,           # zero-sequence (SVPWM) injection
        fault: Fault = Fault(),             # healthy by default; swap to change fault mode
    ) -> None: ...

    # ---- state ----
    drive: DriveModel
    fault: Fault
    vec_theta: np.ndarray                   # (n_theta+1,)
    mat_dq_to_ph_all: np.ndarray            # (n_phases, n_theta+1, dim): dq -> EVERY physical phase
    _phase_shift_samples: int
    omega: float | None

    # ---- setup ----
    def _build_phase_mapping(self) -> None: ...
    def _set_omega(self, omega: float) -> None: ...

    # ---- low-level forward model (used by the optimizer hot loop) ----
    #   curr_dq : (dim,) single setpoint  OR  (n_theta+1, dim) per-angle field
    #   dcurr_dtheta : None -> static (no inductive term)  |  supplied -> dynamic (ω·L·di/dθ)
    #   physics pulled from the drive facade: drive.voltage_operator / .inductance / .bemf_dq
    def get_curr_ph(self, omega: float, curr_dq: np.ndarray) -> PhaseArray: ...
    def get_volt_dq(self, omega: float, curr_dq: np.ndarray,
                    dcurr_dtheta: np.ndarray | None = None) -> np.ndarray: ...
    def get_volt_ph(self, omega: float, curr_dq: np.ndarray,
                    dcurr_dtheta: np.ndarray | None = None) -> tuple[PhaseArray, np.ndarray, PhaseArray]: ...
    def get_max_vals(self, omega: float, curr_dq: np.ndarray,
                     dcurr_dtheta: np.ndarray | None = None) -> tuple[float, float]: ...   # (I_peak, V_peak)
    def _zsc_min_max(self, volt_all: PhaseArray) -> np.ndarray: ...        # -0.5*(min+max) over phases
    #   NOTE: count_peaks (active-set classification) lives in optimization/regimes.py, not here.

    # ---- linear maps for the optimizer's analytical constraint Jacobians ----
    def voltage_linear_maps(self, omega: float, theta_idx: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...           # (gU, gL, bV)

    # ---- high-level assessment OUTSIDE the optimizer ----
    #   setpoint (dim,)        -> static  (dq-plane point, constant waveforms)
    #   setpoint (N_opt, dim)  -> dynamic (dq-plane curve; di/dθ via finite-diff unless supplied)
    def evaluate(self, omega: float, setpoint: np.ndarray,
                 dcurr_dtheta: np.ndarray | None = None) -> Waveforms: ...
```

## Selecting drive type / fault / static-vs-dynamic

```python
recon = Reconstruction(PMSM5Phase())                     # PMSM, healthy
recon = Reconstruction(IM9Phase(), fault=Fault((0,)))    # IM, phase 0 open
recon.fault = Fault((0, 2))                              # swap fault mode at will

wf = recon.evaluate(omega, i_vector)                     # static  assessment
wf = recon.evaluate(omega, i_schedule)                   # dynamic assessment (waveforms, dq planes, V)
```

- drive type = which `DriveModel`
- fault = the `fault` value object (swappable)
- static vs dynamic = vector vs schedule passed in (no mode class)
```
