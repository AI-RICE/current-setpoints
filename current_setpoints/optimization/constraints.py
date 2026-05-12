import numpy as np

from ..simulation import Transform


def current_constraint(
    omega: float, curr_dq: np.ndarray, curr_max: float, transform: Transform
) -> float:
    """
    Ensures the peak physical phase current does not exceed the machine's maximum rating.
    In Scipy's SLSQP, a positive return value means the constraint is satisfied (val >= 0).

    Args:
        omega: Electrical speed [rad/s].
        curr_dq: N-element DQ current vector, length ``n_phases - 1``
            (e.g., 4 for 5-phase, 8 for 9-phase).
        curr_max: Maximum allowable peak phase current [A].
        transform: The Transform instance providing the DQ-to-Phase mapping.

    Returns:
        float: The margin between current limit and peak phase current.
    """
    curr_ph = transform.get_curr_ph(omega, curr_dq)
    curr_peak = np.max(np.abs(curr_ph))

    return curr_max - curr_peak


def voltage_constraint(
    omega: float, curr_dq: np.ndarray, volt_max: float, transform: Transform
) -> float:
    """
    Ensures the peak phase voltage does not exceed the available DC-link /
    inverter voltage limit. When zero-sequence injection (SVPWM) is
    implemented, the peak is taken over the post-injection waveform; with
    injection currently disabled, this is the raw phase voltage.

    Args:
        omega: Electrical speed [rad/s].
        curr_dq: N-element DQ current vector, length ``n_phases - 1``
            (e.g., 4 for 5-phase, 8 for 9-phase).
        volt_max: Maximum allowable peak phase voltage [V].
        transform: The Transform instance providing the DQ-to-Phase mapping.

    Returns:
        float: The margin between voltage limit and peak phase voltage.
    """
    vec_volt_ph, _, _ = transform.get_volt_ph(omega, curr_dq)
    volt_peak = np.max(np.abs(vec_volt_ph))

    return volt_max - volt_peak
