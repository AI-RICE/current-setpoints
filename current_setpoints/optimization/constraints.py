import numpy as np
from ..model import Transform
from ..data import BaseMachine


def current_constraint(
    vec_curr_dq: np.ndarray, machine: BaseMachine, transform: Transform
) -> float:
    """
    Ensures the peak physical phase current does not exceed the machine's maximum rating.
    In Scipy's SLSQP, a positive return value means the constraint is satisfied (val >= 0).

    Args:
        vec_curr_dq: N-element current vector (e.g., length 4 for 5-phase, 8 for 9-phase).
        machine: The Machine object containing 'curr_max'.
        transform: The Transform instance providing the DQ-to-Phase mapping.

    Returns:
        float: The margin between current limit and peak phase current.
    """
    vec_curr_ph = transform.get_curr_ph(vec_curr_dq)
    curr_peak = np.max(np.abs(vec_curr_ph))

    return machine.curr_max - curr_peak


def voltage_constraint(
    vec_curr_dq: np.ndarray, machine: BaseMachine, transform: Transform
) -> float:
    """
    Ensures the peak phase voltage (including zero-sequence injection if enabled)
    does not exceed the available DC-link/inverter voltage limit.

    Args:
        vec_curr_dq: N-element current vector (e.g., length 4 for 5-phase, 8 for 9-phase).
        machine: The Machine object containing 'volt_max'.
        transform: The Transform instance providing voltage matrices and SVPWM logic.

    Returns:
        float: The margin between voltage limit and peak phase voltage.
    """
    vec_volt_ph, _, _ = transform.get_volt_ph(vec_curr_dq)
    volt_peak = np.max(np.abs(vec_volt_ph))

    return machine.volt_max - volt_peak
