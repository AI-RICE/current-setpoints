import numpy as np
from scipy.optimize import minimize
from constraints import current_constraint, voltage_constraint

# TODO: (DONE) do we want to have it here?
# TODO: (DONE) normal name here and later everywhere
# TODO: (DONE) lots of code is written by chatGPT. simplify it. double check.
# TODO: (DONE) this is far too complicated
# TODO: (DONE) this is far too complicated
# TODO: (DONE) the same comments as above
# TODO: (DONE) when I think about it, it should be a class with two functions (they need to have the same API)
# TODO: (DONE) rewrite this and the two functions above and merge them. they need to have the same arguments. all the other arguments should go into __init__

class MotorOptimizer:
    def __init__(self, model, opts=None):
        """
        Initializes the optimizer with a specific torque model strategy.
        Applies a unified set of default solver options across all methods.
        """
        self.model = model
        self.opts = opts if opts is not None else {"disp": False, "ftol": 1e-8, "maxiter": 500}

    def _get_base_constraints(self, transform):
        """
        Returns the physical limits shared by all optimization targets.
        """
        return [
            {"type": "ineq", "fun": current_constraint, "args": (self.model.machine, transform)},
            {"type": "ineq", "fun": voltage_constraint, "args": (self.model.machine, transform)},
        ]

    def _run_optimization(self, objective_fun, constraints, candidates, opts):
        """
        Core multi-start solver loop.
        """
        best_res = None
        best_val = float('inf')

        for vec_start in candidates:
            try:
                res = minimize(
                    objective_fun, 
                    vec_start, 
                    method="SLSQP", 
                    constraints=constraints, 
                    options=opts
                )
                if res.success and res.fun < best_val:
                    best_val = res.fun
                    best_res = res
            except Exception:
                continue

        # If at least one start was successful, return the best one
        if best_res is not None:
            return best_res.x, best_res.fun, True

        # If we reach this point, all multi-starts failed.
        # Return a safe failure state (the initial guess, infinite cost, and False success flag).
        return candidates[0], float('inf'), False

    def maximize_torque(self, transform, vec_curr_guess=None, opts=None):
        """
        Maximizes torque by minimizing negative torque.
        Uses method-specific opts if provided, otherwise defaults to self.opts.
        """
        opts = opts if opts is not None else self.opts
        constraints = self._get_base_constraints(transform)
        candidates = self.model.get_candidates(vec_curr_guess)
        
        def objective(vec_curr_dq):
            return -self.model.calculate_torque(vec_curr_dq)
            
        vec_curr_dq_best, best_val, success = self._run_optimization(objective, constraints, candidates, opts)
        
        return vec_curr_dq_best, -best_val, success

    def minimize_current(self, torq_target, transform, vec_curr_guess=None, opts=None):
        """
        Finds the minimum current vector to hit a specific target torque.
        Uses method-specific opts if provided, otherwise defaults to self.opts.
        """
        opts = opts if opts is not None else self.opts
        constraints = self._get_base_constraints(transform)
        candidates = self.model.get_candidates(vec_curr_guess)
        
        # Append the specific equality constraint for hitting target torque
        def eq_cons(vec_curr_dq):
            return self.model.calculate_torque(vec_curr_dq) - torq_target
            
        constraints.append({"type": "eq", "fun": eq_cons})
        
        def objective(vec_curr_dq):
            return np.sum(vec_curr_dq**2)
            
        vec_curr_dq_best, _, success = self._run_optimization(objective, constraints, candidates, opts)
        
        return vec_curr_dq_best, success