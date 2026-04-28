# current-setpoints

A Python library for computing optimal current setpoints in multiphase permanent
magnet synchronous motor (PMSM) drives. The library combines a steady-state
analytical motor model with a lightweight neural torque correction (NTM) to
account for speed-dependent losses (predominantly mechanical and iron) that the
analytical model alone does not capture. Optimal setpoint maps are then
generated under the resulting hybrid model, subject to peak current and voltage
constraints.

The library accompanies the paper:

> T. Matasova, L. Adam, V. Šmídl, J. Laksar, T. Komrska. *Reducing Joule Losses
> in Five-Phase Synchronous Drives Using a Neural Torque Model and
> Third-Harmonic Injection.* ECCE Europe 2026.

If you use this library in academic work, please cite the paper above.

The supporting taxonomy of operating regions and the steady-state baseline model
follow:

> J. Laksar, T. Komrska, V. Šmídl, O. Suchý, Z. Frank, L. Adam, Z. Peroutka.
> *Analysis and Taxonomy of Optimal Current Setpoints for Five-Phase
> Synchronous Motor Drives.* IEEE Transactions on Industrial Electronics, 2025.

---

## What is this for?

If you have a multiphase PMSM and you want to know which currents to inject for
a given torque demand at a given speed — while respecting the inverter's voltage
limit and the motor's current limit, and while keeping Joule losses low — this
library will produce that map for you. It will also let you train a small
neural network on experimental data to compensate for losses your analytical
model misses.

Concretely, the library lets you:

- Build a parameterized motor model for any number of phases (5-phase
  `IEEEMachine2` is included as a worked example).
- Train a small residual neural network on measured operating points to
  compensate for unmodeled mechanical and iron losses.
- Generate full operating maps (torque × speed) of optimal current setpoints,
  using either the analytical model alone or the loss-aware neural model.
- Visualize the resulting MTPA and field-weakening regions, the third-harmonic
  injection patterns, and the achieved Joule loss reduction.

## Installation

This library targets Python 3.11+. From a fresh environment:

```bash
git clone https://github.com/AI-RICE/current-setpoints.git
cd CurrentSetpoints
pip install -e .
```

For development (pytest, ruff):

```bash
pip install -e ".[dev]"
```

The main runtime dependencies are NumPy, SciPy, PyTorch, scikit-learn,
pandas, and matplotlib.

## Quick start

Generate a baseline (analytical-only) setpoint map for the included 5-phase
benchmark machine:

```python
from current_setpoints.data import IEEEMachine2, FluxValues
from current_setpoints.model import Transform
from current_setpoints.optimization import (
    ModelAnalytical, MotorOptimizer, calculate_grid, grid_to_data,
)
from current_setpoints.utils import plot_grid_segments

# 1. Build the machine and physical limits
flux_registry = FluxValues()
machine = IEEEMachine2(flux_values=flux_registry)
machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)

# 2. Set up the dq-to-phase transform and the analytical torque model
transform = Transform(machine=machine, omega=0.0, add_volt_0=False)
analytical_model = ModelAnalytical(machine=machine)

# 3. Wrap the model in the optimizer
solver_opts = {"disp": False, "ftol": 1e-8, "maxiter": 500, "eps": 1e-8}
optimizer = MotorOptimizer(model=analytical_model, opts=solver_opts)

# 4. Sweep the operating envelope
grid_opts = {"n_torq": 51, "n_omega": 51, "torq_min": 0.0, "omega_min": 0.0}
grid = calculate_grid(
    optimizer=optimizer, transform=transform, opts=grid_opts, mode="standard",
)

# 5. Visualize
plot_grid_segments(grid_to_data(grid, k_skip=1), title="Baseline setpoint map")
```

This produces an MTPA / FW segmentation map analogous to Figure 4 (top) of the
paper.

## Using the neural torque model

The neural correction is a small feedforward network (one hidden layer,
GeLU activation) that learns the residual between the analytical baseline and
measured torque. To use a pretrained network during optimization:

```python
import torch
from current_setpoints.optimization import ModelNeural
from current_setpoints.utils import load_neural_model

device = torch.device("cpu")

neural_net, scaler = load_neural_model(
    weights_path="weights/NTM_Cloned_Corrected.pth",
    scaler_path="weights/NTM_Cloned_Corrected_Scaler.npy",
    hidden_size=12,
    input_size=5,            # 1 (omega) + 4 (currents) for 5-phase
    machine=machine,
    device=device,
)

neural_model_wrapper = ModelNeural(
    machine=machine,
    neural_model=neural_net,
    scaler=scaler,
    device=device,
)
neural_optimizer = MotorOptimizer(model=neural_model_wrapper, opts=solver_opts)

compensated_grid = calculate_grid(
    optimizer=neural_optimizer, transform=transform, opts=grid_opts,
    mode="standard",
)
```

The compensated map will have shifted region boundaries and lower Joule losses
under the loss-aware model.

## Training your own neural correction

Training requires a CSV of measured operating points with columns for speed,
DQ-frame currents, and measured torque. The included `nn_trainer.ipynb`
notebook is a worked example that:

1. Loads the CSV and computes the dynamic flux vector at each operating point.
2. Splits the data, fits a scaler, and runs 5-fold cross-validation over a
   small hyperparameter grid.
3. Retrains the best architecture on the full training set with early stopping.
4. Saves the trained weights and scaler for use with `load_neural_model`.

For your own machine, you will typically also want to register a new flux
provider via `FluxValues.register_constant(...)` (or, for a flux map,
`register_map(...)` once the map evaluation is implemented) and create a
subclass of `GenericMachine` analogous to `IEEEMachine2`.

## Workflow notebooks

End-to-end examples live in `notebooks/`:

- `main.ipynb` — full pipeline: baseline map, correction grid, recalculated
  grid, compensated map, plus the diagnostic plots.
- `nn_trainer.ipynb` — neural network training pipeline.

## Project structure

```text
current_setpoints/
├── data/             # Machine models, parameters, flux registry
├── model/            # Transforms, dq-frame data containers
├── optimization/     # Constraints, torque models, MotorOptimizer, grid sweep
└── utils/            # Data loading, NN architecture & training, plotting
```

## Tests

Run the test suite with:

```bash
pytest
```

## License

MIT — see [`LICENSE`](LICENSE).
