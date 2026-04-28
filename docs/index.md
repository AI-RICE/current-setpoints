# current-setpoints

A Python library for computing optimal current setpoints in multiphase
permanent magnet synchronous motor (PMSM) drives. The library combines a
steady-state analytical motor model with a lightweight neural torque correction
to account for speed-dependent losses (predominantly mechanical and iron) that
the analytical model alone does not capture. Optimal setpoint maps are then
generated under the resulting hybrid model, subject to peak current and voltage
constraints.

The library accompanies the paper:

> T. Matasova, L. Adam, V. Šmídl, J. Laksar, T. Komrska. *Reducing Joule Losses
> in Five-Phase Synchronous Drives Using a Neural Torque Model and
> Third-Harmonic Injection.* ECCE Europe 2026.

If you use this library in academic work, please cite the paper above.

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

For building this documentation site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

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

flux_registry = FluxValues()
machine = IEEEMachine2(flux_values=flux_registry)
machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)

transform = Transform(machine=machine, omega=0.0, add_volt_0=False)
analytical_model = ModelAnalytical(machine=machine)

solver_opts = {"disp": False, "ftol": 1e-8, "maxiter": 500, "eps": 1e-8}
optimizer = MotorOptimizer(model=analytical_model, opts=solver_opts)

grid_opts = {"n_torq": 51, "n_omega": 51, "torq_min": 0.0, "omega_min": 0.0}
grid = calculate_grid(
    optimizer=optimizer, transform=transform, opts=grid_opts, mode="standard",
)

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
    input_size=5,
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

## Workflow notebooks

End-to-end examples live in `notebooks/`:

- `main.ipynb` — full pipeline: baseline map, correction grid, recalculated
  grid, compensated map, plus diagnostic plots.
- `nn_trainer.ipynb` — neural network training pipeline.

For full source code and additional documentation, see the
[GitHub repository](https://github.com/AI-RICE/current-setpoints.git).

## License

Released under the MIT License.
