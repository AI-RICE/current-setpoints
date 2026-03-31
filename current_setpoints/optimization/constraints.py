# TODO: (DONE) finish
import numpy as np

def current_constraint(vec_curr_dq, machine, transform):
    """Ensures current magnitude does not exceed Imax."""
    vec_curr_ph = np.abs(transform.get_curr_ph(vec_curr_dq))
    return machine.curr_max - vec_curr_ph

def voltage_constraint(vec_curr_dq, machine, transform):
    """Ensures voltage magnitude does not exceed Umax."""
    vec_volt_ph = transform.get_volt_ph(vec_curr_dq)
    vec_volt_max = np.max(np.abs(vec_volt_ph))
    return machine.volt_max - vec_volt_max