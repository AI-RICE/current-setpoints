import numpy as np
from scipy.optimize import minimize
from constraints import current_constraint, voltage_constraint

# TODO: (DONE) do we want to have it here?
# TODO: (DONE) normal name here and later everywhere
# TODO: (DONE) lots of code is written by chatGPT. simplify it. double check.
# TODO: (DONE) this is far too complicated
# TODO: (DONE) this is far too complicated
# TODO: (DONE) the same comments as above
# TODO: when I think about it, it should be a class with two functions (they need to have the same API)
# TODO: rewrite this and the two functions above and merge them. they need to have the same arguments. all the other arguments should go into __init__

def torque_optimizer(objective_fun, constraints, candidates, opts):
    """
    Core optimization engine handling multi-start evaluation and exception catching.
    """
    best_res = None
    best_val = float('inf')

    for start_vec in candidates:
        try:
            res = minimize(
                objective_fun, 
                start_vec, 
                method="SLSQP", 
                constraints=constraints, 
                options=opts
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is not None:
        return best_res.x, best_res.fun, True

    # Fallback to the second candidate (usually MTPA) if all multi-starts fail
    fallback_guess = candidates[1] if len(candidates) > 1 else candidates[0]
    res = minimize(
        objective_fun, 
        fallback_guess, 
        method="SLSQP", 
        constraints=constraints, 
        options=opts
    )
    return res.x, res.fun, res.success


def maximize_torque(model, W, is0=None, opts=None):
    """
    API Wrapper: Maximizes torque by minimizing negative torque.
    """
    if opts is None: 
        opts = {"disp": False, "ftol": 1e-8, "maxiter": 500}

    # Objective: Minimize negative torque
    def objective(is_vec):
        return -model.calculate_torque(is_vec)

    constraints = [
        {"type": "ineq", "fun": current_constraint, "args": (model.IPM, W)},
        {"type": "ineq", "fun": voltage_constraint, "args": (model.IPM, W)},
    ]

    candidates = model.get_candidates(is0)
    
    best_x, best_val, success = torque_optimizer(objective, constraints, candidates, opts)
    
    # Return positive torque
    return best_x, -best_val, success


def minimize_current(model, target_torque, W, is0=None, opts=None):
    """
    API Wrapper: Finds the minimum current vector to hit a specific target torque.
    """
    if opts is None: 
        opts = {"disp": False, "ftol": 1e-9, "maxiter": 500}

    # Objective: Minimize sum of squared currents
    def objective(is_vec):
        return np.sum(is_vec**2)

    # Equality Constraint: Achieved torque must equal target torque
    def eq_cons(is_vec):
        return model.calculate_torque(is_vec) - target_torque

    constraints = [
        {"type": "eq", "fun": eq_cons},
        {"type": "ineq", "fun": current_constraint, "args": (model.IPM, W)},
        {"type": "ineq", "fun": voltage_constraint, "args": (model.IPM, W)},
    ]

    candidates = model.get_candidates(is0)
    
    best_x, _, success = torque_optimizer(objective, constraints, candidates, opts)
    
    return best_x, success