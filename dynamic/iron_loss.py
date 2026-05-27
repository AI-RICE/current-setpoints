"""
Iron-loss proxies for post-hoc evaluation of a periodic dq trajectory.

These functions are **not** intended to enter the optimization at this
stage. They evaluate energy components of an already-converged
trajectory so the trade-off against Joule loss can be studied.

Three proxy components, all in the time domain (no FFT required):

* ``eddy_loss``        --  proportional to the period-mean of
  ``|dlambda/dt|^2``, where ``lambda`` is the phase flux linkage seen
  by the stator iron. Quadratic in the trajectory; dominant at high
  speed.
* ``hysteresis_loss``  --  Steinmetz-style proxy on the per-phase
  peak flux ``B_max``. Captures the dominant low-frequency loss but is
  non-smooth in the trajectory (involves a ``max`` over angle), hence
  post-hoc only.
* ``excess_loss``      --  Bertotti's excess term
  ``proportional to mean(|dB/dt|^1.5)``. Included for completeness;
  amplitude often comparable to eddy at intermediate frequencies.

The flux model is intentionally minimal: ``lambda(theta) = L_s x(theta)
+ Psi_PM(theta)``. ``Psi_PM`` is taken from
``transform.flux.get_flux(omega, x)`` at the mean operating dq so the
PM contribution is a constant offset in time (it's the rotor's PM
linkage, which is a periodic function of ``theta`` in the stator frame
but is fixed-amplitude per phase). The constants ``k_e``, ``k_h``,
``k_x`` are machine-specific Steinmetz / Bertotti coefficients. They
are left at 1.0 for relative comparisons; absolute numbers require
fitting against a measurement or a finite-element reference.

The proxy is a *system-level* iron loss summed across all stator
phases. For per-tooth or per-yoke local iron-loss densities a richer
magnetic model would be needed; that is outside the scope of this
proxy.
"""

from __future__ import annotations

import numpy as np

from current_setpoints.simulation import Transform


def _grid_indices(transform: Transform, n_grid: int) -> np.ndarray:
    """Coarse-grid sample indices into ``transform.vec_theta``."""
    n_theta = transform.vec_theta.size - 1
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    return (np.round(theta_grid / (2 * np.pi) * n_theta).astype(int)) % n_theta


def _phase_basis(transform: Transform, idx_grid: np.ndarray) -> np.ndarray:
    """``h_k(theta_n)`` rows, shape ``(n_phases, n_grid, dim)``."""
    n_theta = transform.vec_theta.size - 1
    shift = transform._phase_shift_samples
    return np.stack(
        [transform.mat_dq_to_ph[(idx_grid - k * shift) % n_theta] for k in range(transform.n_phases)],
        axis=0,
    )


def phase_flux_waveforms(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
) -> np.ndarray:
    """
    Per-phase flux-linkage waveforms ``lambda_k(theta_n)`` on the
    coarse grid; shape ``(n_phases, n_grid)``.

    ``lambda_dq = L_s @ x + Psi_PM``; project with ``h_k(theta_n)``.
    """
    transform._set_omega(omega)
    n_grid, dim = curr_dq_grid.shape

    L = transform.machine.L_stat
    flux_volt, _ = transform.flux.get_flux(omega, curr_dq_grid.mean(axis=0))
    psi_pm_dq = flux_volt  # (dim,)

    lambda_dq = curr_dq_grid @ L.T + psi_pm_dq[None, :]  # (n_grid, dim)
    idx_grid = _grid_indices(transform, n_grid)
    H = _phase_basis(transform, idx_grid)
    lambda_ph = np.einsum("knj,nj->kn", H, lambda_dq)  # (n_phases, n_grid)
    return lambda_ph


def eddy_loss(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
    k_e: float = 1.0,
) -> float:
    """
    ``P_eddy = k_e * (2*pi/N) * sum_n sum_k (d lambda_k / dt|_n)^2``.

    The d/dt is evaluated by forward finite difference on the coarse
    grid, using ``d/dt = omega * d/dtheta``. Quadratic in the
    trajectory.
    """
    lambda_ph = phase_flux_waveforms(transform, omega, curr_dq_grid)
    n_grid = lambda_ph.shape[1]
    delta_theta = 2 * np.pi / n_grid
    dlambda_dth = (np.roll(lambda_ph, -1, axis=1) - lambda_ph) / delta_theta  # (n_phases, n_grid)
    dlambda_dt = omega * dlambda_dth
    return float(k_e * np.mean(np.sum(dlambda_dt**2, axis=0)))


def hysteresis_loss(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
    k_h: float = 1.0,
    beta: float = 1.8,
) -> float:
    """
    Steinmetz proxy ``P_hyst = k_h * (omega / 2pi) * sum_k B_max_k^beta``.

    ``B_max_k = max_n |lambda_k(theta_n)|``. Non-smooth in the
    trajectory; use post-hoc.
    """
    lambda_ph = phase_flux_waveforms(transform, omega, curr_dq_grid)
    b_max_per_phase = np.max(np.abs(lambda_ph), axis=1)
    return float(k_h * (omega / (2 * np.pi)) * np.sum(b_max_per_phase**beta))


def excess_loss(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
    k_x: float = 1.0,
) -> float:
    """
    Bertotti excess term ``P_excess = k_x * (1/N) * sum_n sum_k |d lambda_k/dt|^1.5``.

    Time domain, no FFT. Sub-quadratic dependence on the trajectory's
    slope, so it weights moderately-fast variations more than the
    pure eddy term does.
    """
    lambda_ph = phase_flux_waveforms(transform, omega, curr_dq_grid)
    n_grid = lambda_ph.shape[1]
    delta_theta = 2 * np.pi / n_grid
    dlambda_dt = omega * (np.roll(lambda_ph, -1, axis=1) - lambda_ph) / delta_theta
    return float(k_x * np.mean(np.sum(np.abs(dlambda_dt) ** 1.5, axis=0)))


def iron_loss(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
    *,
    k_e: float = 1.0,
    k_h: float = 1.0,
    k_x: float = 0.0,
    beta: float = 1.8,
) -> dict[str, float]:
    """
    Sum of components with their machine-specific coefficients.

    Returns a dict ``{"eddy": ..., "hysteresis": ..., "excess": ..., "total": ...}``.
    Set ``k_x=0`` (default) to disable the Bertotti excess term.
    """
    p_e = eddy_loss(transform, omega, curr_dq_grid, k_e=k_e)
    p_h = hysteresis_loss(transform, omega, curr_dq_grid, k_h=k_h, beta=beta)
    p_x = excess_loss(transform, omega, curr_dq_grid, k_x=k_x) if k_x != 0.0 else 0.0
    return {"eddy": p_e, "hysteresis": p_h, "excess": p_x, "total": p_e + p_h + p_x}


def deadbeat_tracking(
    curr_dq_grid: np.ndarray,
    n_samples: int,
    delay: int = 0,
) -> np.ndarray:
    """
    Zero-order-hold model of a deadbeat current controller running at
    ``n_samples`` updates per electrical period.

    The reference is sampled at uniformly spaced angles
    ``theta_k = k * 2*pi / n_samples`` for ``k = 0..n_samples-1``;
    each sampled dq vector is held constant over the
    ``N / n_samples``-wide angular interval until the next sample. The
    output has the same shape as the input (held values on the fine
    grid).

    Setting ``delay = 1`` shifts the sampled values forward by one
    interval to model the one-step computational latency of a digital
    deadbeat scheme. ``delay = 0`` is the idealised, latency-free
    version.

    If ``n_samples >= N``, returns a copy of the input unchanged (the
    sampler resolves every fine-grid sample already).

    Differs from ``low_pass_trajectory`` by inducing sharp
    discontinuities at the sample boundaries; high harmonics survive
    in the *step* shape, unlike the smooth attenuation of an LP filter.
    """
    X = np.asarray(curr_dq_grid)
    N = X.shape[0]
    if n_samples >= N or n_samples <= 0:
        return X.copy()
    sample_idx = (np.linspace(0, N, n_samples + 1, endpoint=True).astype(int))[:-1]
    samples = X[sample_idx]
    if delay:
        samples = np.roll(samples, delay, axis=0)
    rep_counts = np.diff(np.concatenate([sample_idx, [N]]))
    return np.repeat(samples, rep_counts, axis=0)


def low_pass_trajectory(curr_dq_grid: np.ndarray, h_cutoff: int) -> np.ndarray:
    """
    Project ``curr_dq_grid`` onto its first ``h_cutoff`` Fourier
    harmonics (in the theta direction) and reconstruct. This models a
    finite-bandwidth controller that cannot follow harmonics above
    ``h_cutoff`` -- the *tracked* trajectory used for iron-loss
    evaluation is this low-passed version, not the chattering
    reference.
    """
    N = curr_dq_grid.shape[0]
    coeffs = np.fft.rfft(curr_dq_grid, axis=0)
    coeffs[h_cutoff + 1 :] = 0.0
    return np.fft.irfft(coeffs, n=N, axis=0)


__all__ = [
    "phase_flux_waveforms",
    "eddy_loss",
    "hysteresis_loss",
    "excess_loss",
    "iron_loss",
    "low_pass_trajectory",
    "deadbeat_tracking",
]
