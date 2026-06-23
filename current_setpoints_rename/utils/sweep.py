"""
Operating-point sweep animation utilities.

Given a precomputed grid (from ``calculate_grid``) and a ``BaseTransform``,
build a sequence of frames that walk a chosen path through the operating
map and render them as an animation. Three traversal kinds are supported:

- ``fixed_torque``: hold T constant, sweep omega left-right.
- ``fixed_omega``: hold omega constant, sweep T bottom-up.
- ``envelope``: walk the max-torque envelope from low omega to high.

Each frame carries phase-domain waveforms, the DQ harmonic vectors, and
the segment label, so the animator can render the four-panel figure
consistently across all frames.
"""

from __future__ import annotations

import os
from typing import Any, Literal
from matplotlib.figure import Figure
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..parameters import BaseMachine
from ..simulation import BaseTransform
from .plot_config import PlotConfig


TraversalKind = Literal["fixed_torque", "fixed_omega", "envelope"]


def find_feasible_traversal_center(grid: dict[str, Any]) -> tuple[float, float]:
    """
    Returns ``(T_center_Nm, omega_center_rpm)`` at the median of the grid's
    feasible cells.

    Useful as a default ``fixed_value`` for ``compute_sweep_data`` so that
    "fixed_torque" / "fixed_omega" sweeps land inside the feasibility
    envelope across all four operating modes. A naive "50% of nominal
    machine limit" can miss the envelope entirely on heavily-faulted
    modes whose feasibility shrinks well below the machine rating.

    Args:
        grid: Output of ``calculate_grid``.

    Returns:
        Tuple ``(T_center, omega_center_rpm)`` in physical units.

    Raises:
        ValueError: If the grid contains no feasible cells.
    """
    segments = np.asarray(grid["grid_segments"])
    feasible = ~np.isnan(segments)
    if not np.any(feasible):
        raise ValueError("Grid has no feasible cells.")

    i_indices, j_indices = np.where(feasible)
    i_med = int(np.median(i_indices))
    j_med = int(np.median(j_indices))

    vec_torq = np.asarray(grid["vec_torq"])
    vec_omega_rpm = np.asarray(grid["vec_omega"]) * grid["const_mech_speed"]

    return float(vec_torq[i_med]), float(vec_omega_rpm[j_med])


def compute_sweep_data(
    grid: dict[str, Any],
    transform: BaseTransform,
    kind: TraversalKind,
    fixed_value: float | None = None,
) -> dict[str, Any]:
    """
    Walks the operating map along one traversal and computes everything
    needed to render an animation frame at each step.

    Args:
        grid: Output of ``calculate_grid``. Must contain ``vec_torq``,
            ``vec_omega``, ``curr_dq_grid``, ``grid_segments``,
            ``const_mech_speed``. For ``kind="envelope"`` also requires
            ``vec_torq_max``.
        transform: A configured ``BaseTransform`` subclass instance.
        kind: Traversal type.
        fixed_value: Required for ``"fixed_torque"`` (in Nm) and
            ``"fixed_omega"`` (in mechanical RPM). Ignored for
            ``"envelope"``. The nearest grid value is used.

    Returns:
        A dict containing arrays of shape ``(n_frames, ...)`` for every
        per-frame quantity, plus metadata: ``kind``, ``label``,
        ``vec_theta``, the original grid axes, and ``n_phases_healthy``.
    """
    vec_torq = np.asarray(grid["vec_torq"])
    vec_omega_elec = np.asarray(grid["vec_omega"])
    const_mech = grid["const_mech_speed"]
    vec_omega_rpm = vec_omega_elec * const_mech
    curr_dq_grid = np.asarray(grid["curr_dq_grid"])
    grid_segments = np.asarray(grid["grid_segments"])

    if kind == "fixed_torque":
        if fixed_value is None:
            raise ValueError("fixed_value (Nm) required for kind='fixed_torque'.")
        idx_T = int(np.argmin(np.abs(vec_torq - fixed_value)))
        actual_T = float(vec_torq[idx_T])
        cell_indices = [(idx_T, j) for j in range(len(vec_omega_elec))]
        label = f"Fixed T = {actual_T:.2f} Nm, sweep \u03c9"
    elif kind == "fixed_omega":
        if fixed_value is None:
            raise ValueError("fixed_value (RPM) required for kind='fixed_omega'.")
        idx_omega = int(np.argmin(np.abs(vec_omega_rpm - fixed_value)))
        actual_omega = float(vec_omega_rpm[idx_omega])
        cell_indices = [(i, idx_omega) for i in range(len(vec_torq))]
        label = f"Fixed \u03c9 = {actual_omega:.0f} RPM, sweep T"
    elif kind == "envelope":
        vec_torq_max = np.asarray(grid["vec_torq_max"]).flatten()
        cell_indices = []
        for j, T_max in enumerate(vec_torq_max):
            if np.isnan(T_max):
                continue
            i = int(np.argmin(np.abs(vec_torq - T_max)))
            cell_indices.append((i, j))
        label = "Max-torque envelope"
    else:
        raise ValueError(f"Unknown traversal kind: {kind!r}.")

    feasible = []
    for i, j in cell_indices:
        if np.isnan(grid_segments[i, j]):
            continue
        if np.any(np.isnan(curr_dq_grid[:, i, j])):
            continue
        feasible.append((i, j))

    if not feasible:
        n_filled = int(np.sum(~np.isnan(grid_segments)))
        if kind == "envelope":
            n_omega_with_torq_max = int(np.sum(~np.isnan(np.asarray(grid["vec_torq_max"]).flatten())))
            hint = (
                f"Diagnostic: {n_filled} filled cells in grid_segments; "
                f"{n_omega_with_torq_max} of {len(vec_omega_elec)} omega columns "
                f"have a non-NaN vec_torq_max entry; {len(cell_indices)} candidate "
                f"(i, j) cells were considered. "
                f"If both counts are zero, the grid is empty — check that "
                f"motor_optimizer.maximize_torque is succeeding at the speeds "
                f"in vec_omega. If vec_torq_max has entries but they point to "
                f"NaN grid cells, the cached grid pickle predates the post-hoc "
                f"vec_torq_max write in calculate_grid_pointwise — delete and "
                f"recompute."
            )
        else:
            hint = (
                f"Diagnostic: {n_filled} filled cells in grid_segments. "
                f"fixed_value={fixed_value!r} selected the {'torque' if kind == 'fixed_torque' else 'speed'} "
                f"slice with {len(cell_indices)} candidate cells, all NaN."
            )
        raise ValueError(
            f"No feasible cells along traversal {kind!r} with fixed_value={fixed_value!r}. {hint}"
        )

    vec_theta = transform.vec_theta
    n_frames = len(feasible)
    dim = curr_dq_grid.shape[0]
    n_harmonics = dim // 2

    i0, j0 = feasible[0]
    curr_dq0 = curr_dq_grid[:, i0, j0]
    omega0 = float(vec_omega_elec[j0])
    curr_ph0 = transform.get_curr_ph(omega0, curr_dq0)
    if curr_ph0.ndim == 1:
        n_healthy_phases = 1
    else:
        n_healthy_phases = curr_ph0.shape[0]

    omega_elec_arr = np.zeros(n_frames)
    omega_rpm_arr = np.zeros(n_frames)
    T_arr = np.zeros(n_frames)
    segment_arr = np.zeros(n_frames, dtype=int)
    curr_dq_arr = np.zeros((n_frames, dim))
    curr_ph_arr = np.zeros((n_frames, n_healthy_phases, vec_theta.size))
    volt_ph_arr = np.zeros((n_frames, n_healthy_phases, vec_theta.size))

    for k, (i, j) in enumerate(feasible):
        omega_e = float(vec_omega_elec[j])
        curr_dq = curr_dq_grid[:, i, j]

        omega_elec_arr[k] = omega_e
        omega_rpm_arr[k] = omega_e * const_mech
        T_arr[k] = float(vec_torq[i])
        segment_arr[k] = int(grid_segments[i, j])
        curr_dq_arr[k, :] = curr_dq

        curr_ph = transform.get_curr_ph(omega_e, curr_dq)
        volt_ph, _, _ = transform.get_volt_ph(omega_e, curr_dq)

        if curr_ph.ndim == 1:
            curr_ph = curr_ph[None, :]
            volt_ph = volt_ph[None, :]
        curr_ph_arr[k] = curr_ph
        volt_ph_arr[k] = volt_ph

    return {
        "kind": kind,
        "label": label,
        "vec_theta": vec_theta,
        "vec_torq": vec_torq,
        "vec_omega_rpm": vec_omega_rpm,
        "grid_segments": grid_segments,
        "n_phases_healthy": n_healthy_phases,
        "n_harmonics": n_harmonics,
        "omega_elec": omega_elec_arr,
        "omega_rpm": omega_rpm_arr,
        "T": T_arr,
        "segment": segment_arr,
        "curr_dq": curr_dq_arr,
        "curr_ph": curr_ph_arr,
        "volt_ph": volt_ph_arr,
        "curr_max": transform.machine.curr_max,
        "volt_max": transform.machine.volt_max,
    }


_PHASE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_HARMONIC_COLOURS = ["#1f77b4", "#d62728"]


def animate_sweep(
    sweep_data_list: list[dict[str, Any]],
    machine: BaseMachine,
    output_path: str,
    fps: int = 20,
    target_duration_sec: float | None = 30.0,
    segment_weights: dict[int, float] | None = None,
    save_frames_dir: str | None = None,
    title: str = "",
) -> None:
    """
    Renders an animation walking through one or more sweeps back-to-back.

    Args:
        sweep_data_list: Output of ``compute_sweep_data``, in playback order.
        machine: For axis scaling (n_ppairs is used to compute RPM ceilings).
        output_path: mp4 (or gif) filename for the rendered animation.
        fps: Frames per second.
        target_duration_sec: If set, every sweep is resampled along its
            time axis so that the concatenated playback lasts roughly
            this many seconds. Frames within each sweep are redistributed
            proportionally to the sweep's original frame count, so a
            three-sweep input has the same per-sweep relative pacing
            it had before but a fixed total duration. ``None`` disables
            resampling and plays the raw per-cell frames.
        save_frames_dir: If set, also save individual PNG frames here
            (zero-padded numbering) for use with LaTeX's ``\\animategraphics``.
        title: Figure suptitle, e.g. the fault-mode name.
    """
    if not sweep_data_list:
        raise ValueError("sweep_data_list must contain at least one sweep.")

    if target_duration_sec is not None:
        target_total = int(round(target_duration_sec * fps))
        sweep_data_list = _resample_sweeps_to_total(sweep_data_list, target_total, segment_weights=segment_weights)

    sweeps = []
    cumulative = 0
    for s in sweep_data_list:
        n = len(s["omega_elec"])
        sweeps.append({"data": s, "start": cumulative, "n": n, "label": s["label"]})
        cumulative += n
    total_frames = cumulative

    max_curr = max(np.max(np.abs(s["curr_ph"])) for s in sweep_data_list)
    max_volt = max(np.max(np.abs(s["volt_ph"])) for s in sweep_data_list)
    max_dq = max(np.max(np.abs(s["curr_dq"])) for s in sweep_data_list)
    omega_max_rpm = max(np.max(s["vec_omega_rpm"]) for s in sweep_data_list)
    T_max = max(np.max(s["vec_torq"]) for s in sweep_data_list)
    curr_lim = sweep_data_list[0]["curr_max"]
    volt_lim = sweep_data_list[0]["volt_max"]

    curr_y = 1.1 * max(max_curr, curr_lim)
    volt_y = 1.1 * max(max_volt, volt_lim)
    dq_lim = 1.1 * max(max_dq, 1e-3)

    seg_colours = PlotConfig.get_colors()
    seg_labels = PlotConfig.get_labels()

    fig, axes_dict = _build_figure_layout(title=title)

    base_grid_segments = sweep_data_list[0]["grid_segments"]
    base_vec_omega_rpm = sweep_data_list[0]["vec_omega_rpm"]
    base_vec_torq = sweep_data_list[0]["vec_torq"]
    _draw_map_background(
        axes_dict["map"],
        vec_omega_rpm=base_vec_omega_rpm,
        vec_torq=base_vec_torq,
        grid_segments=base_grid_segments,
        seg_colours=seg_colours,
        seg_labels=seg_labels,
    )
    axes_dict["map"].set_xlim(0, omega_max_rpm * 1.05)
    axes_dict["map"].set_ylim(0, T_max * 1.05)
    axes_dict["map"].set_xlabel("Speed [r/min]")
    axes_dict["map"].set_ylabel("Torque [Nm]")

    vec_theta = sweep_data_list[0]["vec_theta"]
    axes_dict["curr"].set_xlim(0, 2 * np.pi)
    axes_dict["curr"].set_ylim(-curr_y, curr_y)
    axes_dict["curr"].axhline(curr_lim, color="k", linestyle="--", linewidth=1)
    axes_dict["curr"].axhline(-curr_lim, color="k", linestyle="--", linewidth=1)
    axes_dict["curr"].set_xlabel(r"$\theta$ [rad]")
    axes_dict["curr"].set_ylabel("Phase current [A]")
    axes_dict["curr"].set_title("Current waveforms")
    axes_dict["curr"].grid(True, alpha=0.3)

    axes_dict["volt"].set_xlim(0, 2 * np.pi)
    axes_dict["volt"].set_ylim(-volt_y, volt_y)
    axes_dict["volt"].axhline(volt_lim, color="k", linestyle="--", linewidth=1)
    axes_dict["volt"].axhline(-volt_lim, color="k", linestyle="--", linewidth=1)
    axes_dict["volt"].set_xlabel(r"$\theta$ [rad]")
    axes_dict["volt"].set_ylabel("Phase voltage [V]")
    axes_dict["volt"].set_title("Voltage waveforms")
    axes_dict["volt"].grid(True, alpha=0.3)

    axes_dict["dq"].set_xlim(-dq_lim, dq_lim)
    axes_dict["dq"].set_ylim(-dq_lim, dq_lim)
    axes_dict["dq"].axhline(0, color="grey", linewidth=0.5)
    axes_dict["dq"].axvline(0, color="grey", linewidth=0.5)
    axes_dict["dq"].set_xlabel("$i_d$ [A]")
    axes_dict["dq"].set_ylabel("$i_q$ [A]")
    axes_dict["dq"].set_title("DQ plane (h=1 blue, h=3 red)")
    axes_dict["dq"].set_aspect("equal")
    axes_dict["dq"].grid(True, alpha=0.3)

    artists: dict[str, Any] = {}

    artists["marker"] = axes_dict["map"].scatter(
        [0], [0], s=120, c="white", edgecolors="black", linewidths=1.5, zorder=10
    )

    (artists["trail"],) = axes_dict["map"].plot([], [], color="black", linewidth=0.8, alpha=0.5, zorder=5)

    n_healthy = sweep_data_list[0]["n_phases_healthy"]
    artists["curr_lines"] = []
    artists["volt_lines"] = []
    for p in range(n_healthy):
        colour = _PHASE_COLOURS[p % len(_PHASE_COLOURS)]
        (l_curr,) = axes_dict["curr"].plot(
            vec_theta,
            np.zeros_like(vec_theta),
            color=colour,
            linewidth=1.5,
            label=f"Phase {chr(ord('a') + p)}",
        )
        (l_volt,) = axes_dict["volt"].plot(
            vec_theta,
            np.zeros_like(vec_theta),
            color=colour,
            linewidth=1.5,
            label=f"Phase {chr(ord('a') + p)}",
        )
        artists["curr_lines"].append(l_curr)
        artists["volt_lines"].append(l_volt)
    axes_dict["curr"].legend(loc="upper right", fontsize=9)

    n_harmonics = sweep_data_list[0]["n_harmonics"]
    artists["dq_arrows"] = []
    for h in range(n_harmonics):
        colour = _HARMONIC_COLOURS[h % len(_HARMONIC_COLOURS)]
        arr = axes_dict["dq"].annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colour, lw=2.5),
        )
        artists["dq_arrows"].append(arr)

    artists["label_text"] = fig.text(0.5, 0.93, "", ha="center", fontsize=13, fontweight="normal")

    def update(frame_idx: int):
        """Advance to frame `frame_idx`."""
        sweep_idx, local_idx = _frame_to_sweep(frame_idx, sweeps)
        s = sweeps[sweep_idx]["data"]
        sweep_label = sweeps[sweep_idx]["label"]

        omega = s["omega_rpm"][local_idx]
        T = s["T"][local_idx]
        seg = int(s["segment"][local_idx])
        artists["marker"].set_offsets(np.array([[omega, T]]))
        artists["marker"].set_facecolor(seg_colours.get(seg, "grey"))

        trail_omega = s["omega_rpm"][: local_idx + 1]
        trail_T = s["T"][: local_idx + 1]
        artists["trail"].set_data(trail_omega, trail_T)

        for p in range(n_healthy):
            artists["curr_lines"][p].set_ydata(s["curr_ph"][local_idx, p])
            artists["volt_lines"][p].set_ydata(s["volt_ph"][local_idx, p])

        curr_dq = s["curr_dq"][local_idx]
        for h in range(n_harmonics):
            d_idx, q_idx = 2 * h, 2 * h + 1
            artists["dq_arrows"][h].xy = (curr_dq[d_idx], curr_dq[q_idx])

        artists["label_text"].set_text(f"Sweep {sweep_idx + 1}/{len(sweeps)}: {sweep_label}")

        return ()

    anim = manimation.FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps, blit=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        writer = manimation.PillowWriter(fps=fps)
    elif ext == ".mp4":
        writer = manimation.FFMpegWriter(fps=fps, bitrate=2000)
    else:
        raise ValueError(f"Unsupported animation extension {ext!r}. Use '.gif' (Pillow) or '.mp4' (ffmpeg).")

    if save_frames_dir is not None:
        os.makedirs(save_frames_dir, exist_ok=True)
        for k in range(total_frames):
            update(k)
            fig.savefig(
                os.path.join(save_frames_dir, f"frame_{k:04d}.png"),
                dpi=100,
            )

    anim.save(output_path, writer=writer)

    plt.close(fig)


def _frame_to_sweep(frame_idx: int, sweeps: list[dict[str, Any]]) -> tuple[int, int]:
    """Maps a global frame index to (sweep index, local index within sweep)."""
    for i, sw in enumerate(sweeps):
        if frame_idx < sw["start"] + sw["n"]:
            return i, frame_idx - sw["start"]
    return len(sweeps) - 1, sweeps[-1]["n"] - 1


def _build_figure_layout(title: str) -> tuple[Figure, dict[str, Axes]]:
    """
    Builds a 2-row figure: top row is the operating map (wide), bottom
    row has three panels (current waveforms, voltage waveforms, dq
    vectors). Returns the figure and a dict keyed by panel name.
    """
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.0, 1.0],
        hspace=0.45,
        wspace=0.35,
        top=0.90,
        bottom=0.08,
        left=0.07,
        right=0.98,
    )
    ax_map = fig.add_subplot(gs[0, :])
    ax_curr = fig.add_subplot(gs[1, 0])
    ax_volt = fig.add_subplot(gs[1, 1])
    ax_dq = fig.add_subplot(gs[1, 2])

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    return fig, {
        "map": ax_map,
        "curr": ax_curr,
        "volt": ax_volt,
        "dq": ax_dq,
    }


def _draw_map_background(
    ax: Axes,
    vec_omega_rpm: np.ndarray,
    vec_torq: np.ndarray,
    grid_segments: np.ndarray,
    seg_colours: dict[int, str],
    seg_labels: dict[int, str],
) -> None:
    """Renders the operating-map background as a coloured scatter."""
    omega_2d, torq_2d = np.meshgrid(vec_omega_rpm, vec_torq)
    valid = ~np.isnan(grid_segments)
    unique_segs = np.unique(grid_segments[valid]).astype(int)
    for seg_val in unique_segs:
        mask = valid & (grid_segments == seg_val)
        ax.scatter(
            omega_2d[mask],
            torq_2d[mask],
            s=8,
            c=seg_colours.get(int(seg_val), "grey"),
            marker=".",
            alpha=0.4,
            label=seg_labels.get(int(seg_val), str(seg_val)),
        )
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)


_LINEAR_KEYS = (
    "omega_elec",
    "omega_rpm",
    "T",
    "curr_dq",
    "curr_ph",
    "volt_ph",
)

_NEAREST_KEYS = ("segment",)


def _resample_sweeps_to_total(
    sweep_data_list: list[dict[str, Any]],
    target_total: int,
    segment_weights: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Returns a new list of sweep dicts whose combined per-frame arrays
    total ``target_total`` frames.

    The target frame budget is split **equally** across the input sweeps
    (each gets roughly ``target_total / n_sweeps`` frames) so every sweep
    takes the same wall-clock time at playback. Within each sweep,
    continuous per-frame quantities (waveforms, dq vectors, omega, T)
    are linearly interpolated; the integer segment label is
    nearest-neighbour.

    ``segment_weights`` lets you spend less screen time in less
    interesting operating regimes. Each cell's contribution to the
    sweep's playback timeline is scaled by the weight of its segment
    label (defaulting to 1.0). For example, ``{0: 0.3}`` means MTPA_0
    cells take only 30% of the time they would have at uniform weight,
    so the animation skates through them faster and lingers in the
    surrounding regions.

    Metadata keys (``kind``, ``label``, ``vec_theta``, ``vec_torq``,
    ``vec_omega_rpm``, ``grid_segments``, ``n_phases_healthy``,
    ``n_harmonics``, ``curr_max``, ``volt_max``) are passed through.
    """
    if target_total < len(sweep_data_list):
        target_total = len(sweep_data_list)

    n_sweeps = len(sweep_data_list)
    if n_sweeps == 0:
        return sweep_data_list

    base = target_total // n_sweeps
    remainder = target_total - base * n_sweeps
    alloc = [max(2, base + (1 if i < remainder else 0)) for i in range(n_sweeps)]

    return [
        _resample_one_sweep(s, n_target, segment_weights=segment_weights) for s, n_target in zip(sweep_data_list, alloc)
    ]


def _resample_one_sweep(
    sweep: dict[str, Any],
    n_target: int,
    segment_weights: dict[int, float] | None = None,
) -> dict[str, Any]:
    """
    Resamples a single sweep to length ``n_target``. Continuous arrays
    use linear interpolation; the discrete segment label uses
    nearest-neighbour.

    When ``segment_weights`` is provided, the original sample positions
    are spaced unevenly along the playback axis so that cells in
    low-weight segments occupy less of the timeline. The result is that
    a uniformly-spaced ``n_target`` resample lands more frames in
    high-weight regions and fewer in low-weight ones.
    """
    n_original = len(sweep["omega_elec"])
    if n_original < 2:
        out = dict(sweep)
        for key in _LINEAR_KEYS + _NEAREST_KEYS:
            arr = np.asarray(sweep[key])
            out[key] = np.broadcast_to(arr, (n_target,) + arr.shape[1:]).copy()
        return out

    if segment_weights:
        segs = np.asarray(sweep["segment"], dtype=int)
        costs = np.array([segment_weights.get(int(s), 1.0) for s in segs], dtype=np.float64)
        cum = np.concatenate([[0.0], np.cumsum(costs)])
        total = cum[-1]
        if total == 0:
            x_old = np.linspace(0.0, 1.0, n_original)
        else:
            midpoints = 0.5 * (cum[:-1] + cum[1:]) / total
            midpoints[0] = 0.0
            midpoints[-1] = 1.0
            x_old = midpoints
    else:
        x_old = np.linspace(0.0, 1.0, n_original)

    x_new = np.linspace(0.0, 1.0, n_target)

    out = dict(sweep)
    for key in _LINEAR_KEYS:
        out[key] = _interp_axis0(np.asarray(sweep[key]), x_old, x_new)
    nearest_idx = np.array([int(np.argmin(np.abs(x_old - xn))) for xn in x_new], dtype=int)
    for key in _NEAREST_KEYS:
        out[key] = np.asarray(sweep[key])[nearest_idx]
    return out


def _interp_axis0(arr: np.ndarray, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Linearly interpolates ``arr`` along axis 0 (frame axis) from
    ``x_old`` sample positions to ``x_new``. Works for any trailing
    shape. ``x_old`` need not be uniformly spaced.
    """
    flat = arr.reshape(arr.shape[0], -1)
    out_flat = np.empty((len(x_new), flat.shape[1]), dtype=flat.dtype)
    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(x_new, x_old, flat[:, j])
    return out_flat.reshape((len(x_new),) + arr.shape[1:])
