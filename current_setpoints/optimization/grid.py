from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from ..models.forward_model import ForwardModel, count_peaks_at_limit
from .optimizer import BaseOptimizer

if TYPE_CHECKING:
    from ..models.machines import PMSMDrive


def _call_with_timeout(timeout: float, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, str | None]:
    """Run fn(*args, **kwargs) with a hard wall-clock cap, returning
    (None, "timeout") if it doesn't finish in time -- so ONE pathological
    SLSQP call (no cell-level cap otherwise exists in calculate_grid; see
    its docstring/opts["cell_timeout_seconds"]) can never block the whole
    grid indefinitely.

    Uses a plain daemon thread, NOT concurrent.futures.ThreadPoolExecutor.
    An executor's worker threads are registered with
    concurrent.futures.thread's own atexit handler, which BLOCKS
    interpreter/kernel shutdown until every such thread finishes -- even
    ones already abandoned via shutdown(wait=False). Measured directly: an
    exploratory benchmark script built the executor way kept running at
    ~100% CPU with zero new output for 47 minutes after its actual work
    was done, solely because several timed-out solves' worker threads
    were still grinding away and the interpreter would not exit until
    they did. A plain daemon thread has no such handler and never blocks
    process exit.

    This does NOT make an abandoned solve stop running -- Windows has no
    reliable way to force-stop a blocking C-extension call from another
    thread, so a timed-out SLSQP call keeps consuming CPU in the
    background until it finishes on its own. What this bounds is how long
    the CALLER (the grid loop) waits on any one cell, not how long the
    abandoned computation itself runs -- if many cells time out, their
    orphaned threads accumulate and compete for CPU with later cells,
    slowing the whole run down even though it keeps making progress.
    """
    result: dict[str, Any] = {}

    def _run() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as a string, not raised here
            result["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    if "error" in result:
        return None, f"exception: {result['error']!r}"
    return result.get("value"), None


def count_peaks(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    rel_tol: float = 1e-3,
) -> tuple[int, int]:
    return fwd.count_peaks(omega, curr_dq, rel_tol)


def _init_grid_arrays(dim: int, n_torq: int, n_omega: int) -> dict[str, np.ndarray]:
    grid: dict[str, np.ndarray] = {
        k: np.full((n_torq, n_omega), np.nan)
        for k in ("grid_curr_peak", "grid_volt_peak", "grid_segments", "grid_joule_loss")
    }
    grid["vec_torq_max"] = np.full((1, n_omega), np.nan)
    grid["curr_dq_grid"] = np.full((dim, n_torq, n_omega), np.nan)
    return grid


def _joule_loss_static(fwd: ForwardModel, omega: float, curr_dq: np.ndarray) -> float:
    """Cycle-averaged Joule-loss proxy sum_k i_k(theta)^2 for a CONSTANT dq
    setpoint, swept over the full fine theta grid -- exact for static mode
    (current never changes over the cycle). Deliberately the cycle-average
    of the SQUARED PHASE current, not the squared dq norm and not the
    square of a mean -- see _trajectory_waveforms below for why that
    distinction matters for a valid static-vs-dynamic comparison."""
    curr_ph = fwd.curr_ph(omega, curr_dq)  # (n_kept, n_theta+1)
    return float(np.mean(np.sum(curr_ph**2, axis=0)))


def _trajectory_waveforms(
    fwd: ForwardModel,
    omega: float,
    curr_dq_traj: np.ndarray,
    rel_tol: float,
) -> tuple[float, float, int, int, float]:
    """Same diagnostics ForwardModel.peak_vals/count_peaks compute for a
    constant current, but for a dynamic-mode per-angle trajectory
    (curr_dq_traj shape (n_grid, dim)) -- reconstructed via each node's OWN
    rotor angle and OWN current, rather than sweeping one constant current
    over a full cycle.

    Voltage: the PDF's eq. (7) with the current running linearly between
    nodes, i.e. INCLUDING the dynamic term (omega/dtheta) L (i_next - i_n).
    Two values per node: the end of the incoming segment (slope
    (i_n - i_prev)/dtheta, just before theta_n) and the start of the outgoing
    one (slope (i_next - i_n)/dtheta, just after theta_n) -- the same two
    voltages ActiveSetOptimizer limits, so for any trajectory (also R0 /
    IndependentOptimizer ones, which don't limit them) this is the voltage the
    drive actually needs. U/L/u are evaluated at each node's own current
    (exact for any drive). For a constant trajectory both values reduce to
    the static voltage.

    Also returns the cycle-averaged Joule-loss proxy sum_k i_k(theta)^2,
    averaged over the N_GRID nodes -- the mean of the SQUARED phase
    current at each node, not the squared mean dq current (curr_dq_grid,
    which _fill_grid_point stores separately and only for a representative
    setpoint). The two differ whenever the trajectory actually varies over
    the cycle (Jensen's inequality: mean(i^2) >= mean(i)^2), which dynamic
    trajectories generally do -- using the mean-dq-based quantity here
    would silently understate copper loss relative to _joule_loss_static's
    exact per-theta computation for static mode, making the two grids not
    validly comparable."""
    n_grid = curr_dq_traj.shape[0]
    n_theta = fwd.vec_theta.size - 1
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    idx_grid = np.round(theta_grid / (2 * np.pi) * n_theta).astype(int) % n_theta
    kept = list(fwd.fault._kept)
    coupling = omega / (2 * np.pi / n_grid)  # omega / dtheta

    n_curr_ph = fwd.phase_map_at_theta(0).shape[0]
    curr_wave = np.full((n_curr_ph, n_grid), np.nan)
    # Column 2n: end of segment n-1 (just before theta_n); column 2n+1: start of segment n.
    volt_wave = np.full((len(kept), 2 * n_grid), np.nan)

    for n in range(n_grid):
        i_n = curr_dq_traj[n]
        if not np.all(np.isfinite(i_n)):
            continue
        i_prev = curr_dq_traj[(n - 1) % n_grid]
        i_next = curr_dq_traj[(n + 1) % n_grid]
        theta_idx = int(idx_grid[n])
        H = fwd.phase_map_at_theta(theta_idx)
        curr_wave[:, n] = H @ i_n
        gU, gL, bV = fwd.volt_map_at_theta(theta_idx, omega, i_n)
        v_steady = gU @ i_n + bV
        volt_wave[:, 2 * n] = (v_steady + coupling * (gL @ (i_n - i_prev)))[kept]
        volt_wave[:, 2 * n + 1] = (v_steady + coupling * (gL @ (i_next - i_n)))[kept]

    # count_peaks_at_limit expects a trailing "wrap" sample (theta=2pi == theta=0,
    # as the constant-current n_theta+1 sweep has); append the first column so
    # the same peak-touching logic applies unchanged.
    curr_wave_wrapped = np.concatenate([curr_wave, curr_wave[:, :1]], axis=1)
    volt_wave_wrapped = np.concatenate([volt_wave, volt_wave[:, :1]], axis=1)

    curr_peak = float(np.nanmax(np.abs(curr_wave)))
    volt_peak = float(np.nanmax(np.abs(volt_wave)))
    n_curr = count_peaks_at_limit(curr_wave_wrapped, fwd.drive.curr_max, rel_tol)
    n_volt = count_peaks_at_limit(volt_wave_wrapped, fwd.drive.volt_max, rel_tol)
    joule_loss = float(np.nanmean(np.sum(curr_wave**2, axis=0)))
    return curr_peak, volt_peak, n_curr, n_volt, joule_loss


def _fill_grid_point(
    grid: dict[str, Any],
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    idx_torq: int,
    idx_omega: int,
    rel_tol: float = 1e-3,
) -> None:
    if curr_dq.ndim == 1:
        curr_peak, volt_peak = fwd.peak_vals(omega, curr_dq)
        n_curr, n_volt = fwd.count_peaks(omega, curr_dq, rel_tol)
        joule_loss = _joule_loss_static(fwd, omega, curr_dq)
        grid["curr_dq_grid"][:, idx_torq, idx_omega] = curr_dq
    else:
        curr_peak, volt_peak, n_curr, n_volt, joule_loss = _trajectory_waveforms(fwd, omega, curr_dq, rel_tol)
        # curr_dq_grid holds one representative current per cell (the mean over the nodes -- NOT an
        # operating point, only a summary); the full per-angle trajectory, which is the actual
        # dynamic setpoint, is kept in curr_dq_traj (n_torq, n_omega, n_grid, dim), allocated on the
        # first dynamic cell. grid_joule_loss is NOT derived from the mean -- see _trajectory_waveforms.
        grid["curr_dq_grid"][:, idx_torq, idx_omega] = np.nanmean(curr_dq, axis=0)
        if "curr_dq_traj" not in grid:
            n_torq, n_omega = grid["grid_joule_loss"].shape
            grid["curr_dq_traj"] = np.full((n_torq, n_omega) + curr_dq.shape, np.nan)
        grid["curr_dq_traj"][idx_torq, idx_omega] = curr_dq

    grid["grid_curr_peak"][idx_torq, idx_omega] = curr_peak
    grid["grid_volt_peak"][idx_torq, idx_omega] = volt_peak
    grid["grid_segments"][idx_torq, idx_omega] = 3 * n_volt + n_curr
    grid["grid_joule_loss"][idx_torq, idx_omega] = joule_loss


def calculate_grid(
    optimizer: BaseOptimizer,
    fwd: ForwardModel,
    opts: dict[str, Any],
    mode: str = "standard",
    dict_grid_corr: dict[str, Any] | None = None,
    on_cell_solved: Any = None,
) -> dict[str, Any]:
    """... (see module docstring / existing behavior above for the grid contract).

    on_cell_solved: optional callback (idx_torq, idx_omega, omega, torq_target, sol) -> None,
    invoked once per ATTEMPTED cell (both mode branches), immediately after that cell's solve
    returns and _fill_grid_point runs -- sol is the actual Solution used for the numeric grid
    (None if the solve timed out, see cell_timeout_seconds), so a caller can do its own
    per-cell work (e.g. save a full waveform diagnostic plot from the SAME solve) without a
    second, separate pass re-solving every cell. Purely additive: default None, zero behavior
    change for every existing caller.
    """
    if mode not in ("standard", "recalculated"):
        raise ValueError(f"mode must be 'standard' or 'recalculated', got {mode!r}")
    if mode == "recalculated" and dict_grid_corr is None:
        raise ValueError("dict_grid_corr must be provided for recalculated mode.")

    drive = fwd.drive
    dim = drive.dim
    const_mech_speed = 30.0 / (np.pi * drive.n_ppairs)
    omega_probe = float(opts.get("omega_probe", 0.0))

    # Optional hard per-solve wall-clock cap -- off (None) by default, matching every existing
    # caller's behavior exactly. See _call_with_timeout's docstring for what this does and does
    # not protect against. A timed-out cell is treated exactly like a converged-but-failed one
    # (sol is None below wherever a plain "not sol.success" check used to be sufficient).
    cell_timeout = opts.get("cell_timeout_seconds")

    def _solve(fn: Any, *args: Any, **kwargs: Any) -> Any:
        if cell_timeout is None:
            return fn(*args, **kwargs)
        sol, err = _call_with_timeout(cell_timeout, fn, *args, **kwargs)
        if sol is None:
            print(f"    solve did not finish within {cell_timeout}s ({err}) -- treating as failed")
        return sol

    grid: dict[str, Any] = {"const_mech_speed": const_mech_speed}

    if mode == "standard":
        if "torq_max" in opts and opts["torq_max"] is not None:
            torq_max_global = float(opts["torq_max"])
        else:
            sol = _solve(optimizer.maximize_torque, omega_probe)
            if sol is None or not sol.success:
                raise RuntimeError(
                    f"Global T_max probe at omega={omega_probe} rad/s failed or timed out. "
                    "Adjust omega_probe in opts or extend seeds."
                )
            torq_max_global = sol.torque

        # The speed horizon is a property of the map being computed, not of
        # the machine; opts["omega_max"] is preferred, drive.omega_max is the
        # legacy fallback for drives that still carry it.
        if "omega_max" in opts and opts["omega_max"] is not None:
            omega_max_grid = float(opts["omega_max"])
        else:
            omega_max_grid = drive.omega_max

        n_torq, n_omega = opts["n_torq"], opts["n_omega"]
        grid["vec_torq"] = np.linspace(opts["torq_min"], torq_max_global, n_torq)
        grid["vec_omega"] = np.linspace(
            opts["omega_min"] / const_mech_speed,
            omega_max_grid / const_mech_speed,
            n_omega,
        )
        grid_torq_targets = None
    else:
        n_torq = len(dict_grid_corr["vec_torq"])  # type: ignore[index]
        n_omega = len(dict_grid_corr["vec_omega"])  # type: ignore[index]
        grid["vec_torq"] = dict_grid_corr["vec_torq"]  # type: ignore[index]
        grid["vec_omega"] = dict_grid_corr["vec_omega"]  # type: ignore[index]
        grid_torq_targets = dict_grid_corr["grid_torq_neural"]  # type: ignore[index]
        grid["grid_torq_neural"] = grid_torq_targets
        torq_max_global = float(grid["vec_torq"][-1])

    grid.update(_init_grid_arrays(dim, n_torq, n_omega))

    print(f"Starting {mode} grid: {n_torq} torques × {n_omega} speeds")

    # Warm-start for maximize_torque carried across speeds (FW corner convergence)
    prev_max_curr: np.ndarray | None = None

    for idx_omega in range(n_omega):
        omega = float(grid["vec_omega"][idx_omega])
        rpm = omega * const_mech_speed
        print(f"  Speed {idx_omega + 1}/{n_omega}: {rpm:.1f} RPM")

        # Per-speed T_max probe (warm-started from previous speed)
        sol_max = _solve(optimizer.maximize_torque, omega, guess=prev_max_curr)
        if sol_max is not None and sol_max.success:
            torq_max_local = sol_max.torque
            prev_max_curr = sol_max.curr_dq
        else:
            # Fall back to global estimate so cells aren't silently skipped
            torq_max_local = torq_max_global if mode == "standard" else float("inf")
        grid["vec_torq_max"][0, idx_omega] = torq_max_local

        # Initial warm-start for the torque sweep: first seed from drive
        prev_curr = drive.seeds()[0]

        for idx_torq in range(n_torq):
            if mode == "standard":
                torq_target = float(grid["vec_torq"][idx_torq])
                guess = prev_curr.copy()
                if torq_target > min(torq_max_local, torq_max_global):
                    break
            else:
                torq_target = float(grid_torq_targets[idx_torq, idx_omega])  # type: ignore[index]
                corr_curr = dict_grid_corr["curr_dq_grid"][:, idx_torq, idx_omega]  # type: ignore[index]
                if np.isnan(torq_target) or np.any(np.isnan(corr_curr)) or torq_target > torq_max_local:
                    continue
                guess = corr_curr

            sol = _solve(optimizer.minimize_current, torq_target, omega, guess=guess)

            if mode == "standard":
                if sol is not None and sol.success:
                    prev_curr = sol.curr_dq
                    _fill_grid_point(grid, fwd, omega, sol.curr_dq, idx_torq, idx_omega)
            else:
                # Recalculated: only a converged solve fills the cell. A failed or timed-out solve
                # leaves it empty (NaN) -- no fallback to the warm-start guess (2026-10-05): the
                # guess is not a solution for this cell's torque target, and filling it in made
                # failed cells look solved on the maps.
                if sol is not None and sol.success:
                    _fill_grid_point(grid, fwd, omega, sol.curr_dq, idx_torq, idx_omega)

            if on_cell_solved is not None:
                on_cell_solved(idx_torq, idx_omega, omega, torq_target, sol)

    print(f"{mode.capitalize()} grid complete.")
    return grid


def get_correction_grid(
    dict_grid: dict[str, Any],
    neural: "PMSMDrive",
) -> dict[str, Any]:
    print("Computing neural correction grid...")

    out = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in dict_grid.items()}

    dim, n_torq, n_omega = dict_grid["curr_dq_grid"].shape
    vec_omega = dict_grid["vec_omega"]
    const_mech_speed = dict_grid["const_mech_speed"]
    grid_torq_neural = np.full((n_torq, n_omega), np.nan)

    for idx_omega in range(n_omega):
        omega = float(vec_omega[idx_omega])
        for idx_torq in range(n_torq):
            curr = dict_grid["curr_dq_grid"][:, idx_torq, idx_omega]
            if np.any(np.isnan(curr)):
                continue
            grid_torq_neural[idx_torq, idx_omega] = neural.torque(omega, curr)

    out["grid_torq_neural"] = grid_torq_neural
    print("Neural correction grid complete.")
    return out
