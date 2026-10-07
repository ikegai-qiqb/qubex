"""Pi-pair rotary-echo ZX90 pulse construction and amplitude calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
)
from qubex.experiment.models import Result
from qubex.pulse import Arbitrary, FlatTop, PulseArray, PulseSchedule, Waveform
from qubex.typing import TargetMap
from qubex.visualization import make_figure

__all__ = ["calibrate_pi_pair_rotary_zx90", "pi_pair_rotary_zx90"]

_DURATION_UNIT = 16.0
_DEFAULT_SWEEP_FRACTION = 0.16
_DEFAULT_SWEEP_POINTS = 17
_DEFAULT_ADIABATIC_SAFE_FACTOR = 0.75
_TOLERANCE = 1e-12
_MIN_SLOPE = 1e-9


def pi_pair_rotary_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    cr_amplitude: float,
    duration: float,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float = 0.0,
) -> PulseSchedule:
    """
    Build an echoed ZX90 schedule with a pi-pair target rotary echo.

    The cancellation IQ is scaled from the stored CR calibration in proportion
    to `cr_amplitude`. Each CR half contains the rotary pattern
    `blank, +pi, blank, +pi, -pi, blank, -pi, blank`.

    Parameters
    ----------
    exp
        Experiment used to load CR parameters and calibrated X180 pulses.
    control_qubit
        Control qubit label.
    target_qubit
        Target qubit label.
    cr_amplitude
        Positive absolute CR amplitude.
    duration
        Duration of one CR half in ns. It must leave four equal blank regions,
        each at least as long as the CR ramp, around four target X180 pulses.
    x180
        Optional control X180 waveform or target mapping.
    x180_margin
        Zero-amplitude margin before and after each control X180 in ns. It must
        align to the experiment sampling period.

    Returns
    -------
    PulseSchedule
        Echoed pi-pair rotary ZX90 schedule.

    Raises
    ------
    ValueError
        If stored calibration is missing, timing is incompatible with the
        pi-pair geometry, or a channel exceeds unit amplitude.
    TypeError
        If a scalar, waveform, or target mapping has an incompatible type.
    """
    _validate_qubit_labels(control_qubit, target_qubit)
    amplitude = _as_positive_float(cr_amplitude, name="cr_amplitude")
    half_duration = _as_positive_float(duration, name="duration")
    margin = _as_nonnegative_float(x180_margin, name="x180_margin")
    sampling_period = _sampling_period(exp)
    params = _load_cr_params(exp, control_qubit, target_qubit)
    target_pi = exp.pulse.x180(target_qubit)
    control_pi = _resolve_x180(exp, control_qubit, x180)
    _validate_waveform(target_pi, sampling_period=sampling_period, name="target X180")
    _validate_waveform(control_pi, sampling_period=sampling_period, name="control X180")
    _require_aligned(half_duration, sampling_period, name="duration")
    _require_aligned(margin, sampling_period, name="x180_margin")

    blank_duration = (half_duration - 4.0 * target_pi.duration) / 4.0
    if blank_duration + _TOLERANCE < params["ramptime"]:
        raise ValueError("The pi-pair blank duration must be at least the CR ramptime.")
    _require_aligned(blank_duration, sampling_period, name="pi-pair blank duration")

    scale = amplitude / params["cr_amplitude"]
    cancellation = (
        params["cancel_amplitude"] * np.exp(1j * params["cancel_phase"]) * scale
    )
    cr_waveform = FlatTop(
        duration=half_duration,
        amplitude=amplitude,
        tau=params["ramptime"],
        phase=params["cr_phase"],
        beta=params["cr_beta"],
        sampling_period=sampling_period,
    )
    cancellation_waveform = FlatTop(
        duration=half_duration,
        amplitude=float(abs(cancellation)),
        tau=params["ramptime"],
        phase=float(np.angle(cancellation)),
        beta=params["cancel_beta"],
        sampling_period=sampling_period,
    )
    rotary_values = _pi_pair_values(
        target_pi,
        blank_duration=blank_duration,
        sampling_period=sampling_period,
    )
    cr_values = np.asarray(cr_waveform.values, dtype=np.complex128)
    target_values = (
        np.asarray(cancellation_waveform.values, dtype=np.complex128) + rotary_values
    )
    _validate_channel(cr_values, name="CR channel")
    _validate_channel(target_values, name="target channel")

    cr_label = f"{control_qubit}-{target_qubit}"
    echoed_control_pi = _with_margin(
        control_pi,
        margin=margin,
        sampling_period=sampling_period,
    )
    with PulseSchedule([control_qubit, cr_label, target_qubit]) as schedule:
        schedule.add(
            cr_label,
            Arbitrary(cr_values, sampling_period=sampling_period),
        )
        schedule.add(
            target_qubit,
            Arbitrary(target_values, sampling_period=sampling_period),
        )
        schedule.barrier()
        schedule.add(control_qubit, echoed_control_pi)
        schedule.barrier()
        schedule.add(
            cr_label,
            Arbitrary(-cr_values, sampling_period=sampling_period),
        )
        schedule.add(
            target_qubit,
            Arbitrary(-target_values, sampling_period=sampling_period),
        )
        schedule.barrier()
        schedule.add(control_qubit, echoed_control_pi)
    return schedule


def calibrate_pi_pair_rotary_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    duration: float | None = None,
    amplitude_range: ArrayLike | None = None,
    sweep_fraction: float = _DEFAULT_SWEEP_FRACTION,
    sweep_points: int = _DEFAULT_SWEEP_POINTS,
    adiabatic_safe_factor: float | None = None,
    max_amplitude: float = 1.0,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float = 0.0,
    shots: int | None = None,
    interval: float | None = None,
    plot: bool = True,
) -> Result:
    """
    Calibrate the CR amplitude of a pi-pair rotary-echo ZX90 gate.

    Four target-Z traces are measured from initial states 00, 01, 10, and 11.
    All measured sign-changing intervals of
    `(z00 - z01 + z10 - z11) / 4` are linearly interpolated, and the root
    nearest the amplitude predicted from the stored ZX rotation rate is used.

    Parameters
    ----------
    exp
        Experiment used for pulse construction and measurement.
    control_qubit
        Control qubit label.
    target_qubit
        Target qubit label.
    duration
        Duration of one CR half in ns. When omitted, a 16 ns-aligned duration
        is predicted and enlarged as needed for the pi-pair geometry.
    amplitude_range
        Optional strictly increasing positive absolute CR amplitudes within
        the configured safe limit. Defaults to 17 points spanning +/-16%
        around the predicted amplitude.
    sweep_fraction
        Fractional half-width of the automatic sweep. Defaults to `0.16`.
    sweep_points
        Number of automatic sweep points. Defaults to `17`.
    adiabatic_safe_factor
        Maximum CR Rabi rate as a fraction of control-target detuning.
        Defaults to `0.75`.
    max_amplitude
        Additional upper bound on absolute CR amplitude. Defaults to `1.0`.
    x180
        Optional control X180 waveform or target mapping.
    x180_margin
        Margin around each control X180 in ns.
    shots
        Shots per initial-state sweep. Defaults to calibration shots.
    interval
        Shot interval in ns. Defaults to the experiment-wide default.
    plot
        Whether to display the zero-crossing plot.

    Returns
    -------
    Result
        Measurements, fit diagnostics, calibration mapping, and final schedule.

    Raises
    ------
    ValueError
        If stored CR data, timing, sweep bounds, pulse amplitudes, measurements,
        or the measured zero crossing are invalid.
    TypeError
        If an input has an incompatible scalar, array, waveform, or mapping
        type.

    Notes
    -----
    This function performs hardware measurements but does not update the
    ordinary CR calibration note.
    """
    _validate_qubit_labels(control_qubit, target_qubit)
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be a boolean.")
    fraction = _as_positive_float(sweep_fraction, name="sweep_fraction")
    if fraction >= 1.0:
        raise ValueError("sweep_fraction must be less than one.")
    points = _as_integer_at_least(sweep_points, minimum=2, name="sweep_points")
    resolved_shots = (
        CALIBRATION_SHOTS
        if shots is None
        else _as_integer_at_least(shots, minimum=1, name="shots")
    )
    resolved_interval = (
        float(DEFAULT_INTERVAL)
        if interval is None
        else _as_positive_float(interval, name="interval")
    )
    safe_factor = (
        _DEFAULT_ADIABATIC_SAFE_FACTOR
        if adiabatic_safe_factor is None
        else _as_positive_float(
            adiabatic_safe_factor,
            name="adiabatic_safe_factor",
        )
    )
    hardware_limit = _as_positive_float(max_amplitude, name="max_amplitude")
    if hardware_limit > 1.0 + _TOLERANCE:
        raise ValueError("max_amplitude must not exceed one.")

    params = _load_cr_params(exp, control_qubit, target_qubit)
    sampling_period = _sampling_period(exp)
    target_pi = exp.pulse.x180(target_qubit)
    _validate_waveform(target_pi, sampling_period=sampling_period, name="target X180")
    resolved_duration = _resolve_duration(
        duration,
        reference_amplitude=params["cr_amplitude"],
        zx_rotation_rate=params["zx_rotation_rate"],
        ramptime=params["ramptime"],
        pi_duration=target_pi.duration,
        sampling_period=sampling_period,
    )
    predicted = 1.0 / (
        8.0 * params["zx_rotation_rate"] * (resolved_duration - params["ramptime"])
    )
    max_cr_amplitude = _max_cr_amplitude(
        exp,
        control_qubit,
        target_qubit,
        adiabatic_safe_factor=safe_factor,
        max_amplitude=hardware_limit,
    )
    if predicted > max_cr_amplitude + _TOLERANCE:
        raise ValueError(
            "The predicted CR amplitude exceeds the configured safe CR limit; "
            "increase the CR-half duration."
        )
    amplitudes = _resolve_amplitude_range(
        amplitude_range,
        center=predicted,
        fraction=fraction,
        points=points,
        upper=max_cr_amplitude,
    )
    # Validate both endpoints before starting hardware measurements. Channel
    # magnitude is convex in the linearly scaled CR/cancellation amplitude, so
    # safe endpoints imply a safe interval.
    for endpoint in (amplitudes[0], amplitudes[-1]):
        pi_pair_rotary_zx90(
            exp,
            control_qubit,
            target_qubit,
            cr_amplitude=float(endpoint),
            duration=resolved_duration,
            x180=x180,
            x180_margin=x180_margin,
        )

    state_pairs = (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"))
    state_values = np.empty((4, amplitudes.size), dtype=np.float64)
    raw_results: list[Any] = []

    def sequence(candidate: float) -> PulseSchedule:
        return pi_pair_rotary_zx90(
            exp,
            control_qubit,
            target_qubit,
            cr_amplitude=float(candidate),
            duration=resolved_duration,
            x180=x180,
            x180_margin=x180_margin,
        )

    for index, (control_state, target_state) in enumerate(state_pairs):
        measured = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=amplitudes,
            initial_states={
                control_qubit: control_state,
                target_qubit: target_state,
            },
            n_shots=resolved_shots,
            shot_interval=resolved_interval,
            plot=False,
            title=f"Pi-pair rotary ZX90 calibration: {control_qubit}-{target_qubit}",
            xlabel="CR amplitude (arb. units)",
            ylabel="Normalized target Z",
        )
        raw_results.append(measured)
        if target_qubit not in measured.data:
            raise ValueError(
                f"Calibration measurement is missing target `{target_qubit}`."
            )
        values = _as_finite_vector(
            measured.data[target_qubit].normalized,
            name="normalized target-Z measurement",
        )
        if values.shape != amplitudes.shape:
            raise ValueError(
                "Each state measurement must return one value per amplitude."
            )
        state_values[index] = values

    z00, z01, z10, z11 = state_values
    s_zx = (z00 - z01 + z10 - z11) / 4.0
    s_ix_from_z = (z00 - z01 - z10 + z11) / 4.0
    slope, intercept, root, root_bracket, fitted = _nearest_zero_crossing(
        amplitudes,
        s_zx,
        reference=predicted,
    )
    if root <= 0.0 or root > max_cr_amplitude + _TOLERANCE:
        raise ValueError(
            "The interpolated ZX90 root is outside the safe amplitude range."
        )

    final_schedule = pi_pair_rotary_zx90(
        exp,
        control_qubit,
        target_qubit,
        cr_amplitude=root,
        duration=resolved_duration,
        x180=x180,
        x180_margin=x180_margin,
    )
    cancel_amplitude = params["cancel_amplitude"] * root / params["cr_amplitude"]
    calibration = {
        "status": "completed",
        "control_qubit": control_qubit,
        "target_qubit": target_qubit,
        "duration": resolved_duration,
        "ramptime": params["ramptime"],
        "cr_amplitude": root,
        "cr_phase": params["cr_phase"],
        "cr_beta": params["cr_beta"],
        "cancel_amplitude": cancel_amplitude,
        "cancel_phase": params["cancel_phase"],
        "cancel_beta": params["cancel_beta"],
        "predicted_cr_amplitude": predicted,
        "max_cr_amplitude": max_cr_amplitude,
        "sampling_period": sampling_period,
        "x180_margin": float(x180_margin),
    }
    figure = _plot_calibration(
        amplitudes,
        s_zx,
        fitted,
        root=root,
        root_bracket=root_bracket,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
    )
    if plot:
        figure.show()
    return Result(
        data={
            "pi_pair_rotary_zx90_calibration": calibration,
            "amplitude_range": amplitudes,
            "state_values": state_values,
            "s_zx": s_zx,
            "s_ix_from_z": s_ix_from_z,
            "fit_slope": slope,
            "fit_intercept": intercept,
            "fitted_signal": fitted,
            "root": root,
            "root_bracket": root_bracket,
            "raw_results": tuple(raw_results),
            "pulse_schedule": final_schedule,
        },
        figure=figure,
    )


def _load_cr_params(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
) -> dict[str, float]:
    cr_label = f"{control_qubit}-{target_qubit}"
    stored = exp.ctx.calib_note.get_cr_param(
        cr_label,
        valid_days=exp.ctx.calibration_valid_days,
    )
    if stored is None:
        raise ValueError(f"CR parameters for {cr_label} are not stored.")
    if not isinstance(stored, Mapping):
        raise TypeError("Stored CR parameters must be a mapping.")
    return {
        "ramptime": _stored_float(stored, "ramptime", nonnegative=True),
        "cr_amplitude": _stored_float(stored, "cr_amplitude", positive=True),
        "cr_phase": _stored_float(stored, "cr_phase"),
        "cr_beta": _stored_float(stored, "cr_beta"),
        "cancel_amplitude": _stored_float(
            stored,
            "cancel_amplitude",
            nonnegative=True,
        ),
        "cancel_phase": _stored_float(stored, "cancel_phase"),
        "cancel_beta": _stored_float(stored, "cancel_beta"),
        "zx_rotation_rate": _stored_float(
            stored,
            "zx_rotation_rate",
            positive=True,
        ),
    }


def _stored_float(
    stored: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        value = stored[key]
    except KeyError as exc:
        raise ValueError(f"Stored CR parameters are missing {key!r}.") from exc
    if positive:
        return _as_positive_float(value, name=key)
    if nonnegative:
        return _as_nonnegative_float(value, name=key)
    return _as_finite_float(value, name=key)


def _sampling_period(exp: Experiment) -> float:
    value = exp.ctx.util.resolve_sampling_period(exp.ctx.measurement.sampling_period)
    return _as_positive_float(value, name="sampling_period")


def _resolve_x180(
    exp: Experiment,
    control_qubit: str,
    x180: TargetMap[Waveform] | Waveform | None,
) -> Waveform:
    if x180 is None:
        result = exp.pulse.x180(control_qubit)
    elif isinstance(x180, Waveform):
        result = x180
    elif isinstance(x180, Mapping):
        try:
            result = x180[control_qubit]
        except KeyError as exc:
            raise ValueError(
                f"x180 mapping does not contain control qubit {control_qubit!r}."
            ) from exc
    else:
        raise TypeError("x180 must be a Waveform, a target mapping, or None.")
    if not isinstance(result, Waveform):
        raise TypeError("x180 must resolve to a Waveform.")
    return result


def _pi_pair_values(
    pi_pulse: Waveform,
    *,
    blank_duration: float,
    sampling_period: float,
) -> NDArray[np.complex128]:
    blank = np.zeros(round(blank_duration / sampling_period), dtype=np.complex128)
    pi_values = np.asarray(pi_pulse.values, dtype=np.complex128)
    return np.concatenate(
        (
            blank,
            pi_values,
            blank,
            pi_values,
            -pi_values,
            blank,
            -pi_values,
            blank,
        )
    )


def _with_margin(
    waveform: Waveform,
    *,
    margin: float,
    sampling_period: float,
) -> Waveform:
    if margin == 0.0:
        return waveform
    zeros = np.zeros(round(margin / sampling_period), dtype=np.complex128)
    return PulseArray(
        (
            Arbitrary(zeros, sampling_period=sampling_period),
            waveform,
            Arbitrary(zeros, sampling_period=sampling_period),
        )
    )


def _resolve_duration(
    requested: float | None,
    *,
    reference_amplitude: float,
    zx_rotation_rate: float,
    ramptime: float,
    pi_duration: float,
    sampling_period: float,
) -> float:
    _require_aligned(_DURATION_UNIT, sampling_period, name="duration unit")
    minimum_geometry = 4.0 * pi_duration + 4.0 * ramptime
    if requested is None:
        predicted = 1.0 / (8.0 * reference_amplitude * zx_rotation_rate) + ramptime
        duration = (
            math.ceil((max(predicted, minimum_geometry) - _TOLERANCE) / _DURATION_UNIT)
            * _DURATION_UNIT
        )
    else:
        duration = _as_positive_float(requested, name="duration")
        units = duration / _DURATION_UNIT
        if not np.isclose(units, round(units), rtol=0.0, atol=_TOLERANCE):
            raise ValueError("duration must be a multiple of 16 ns.")
    _require_aligned(duration, sampling_period, name="duration")
    if duration + _TOLERANCE < minimum_geometry:
        raise ValueError(
            "duration is too short for four target pi pulses and ramp-safe blanks."
        )
    return float(duration)


def _max_cr_amplitude(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    adiabatic_safe_factor: float,
    max_amplitude: float,
) -> float:
    detuning = abs(
        float(exp.ctx.qubits[target_qubit].frequency)
        - float(exp.ctx.qubits[control_qubit].frequency)
    )
    max_rabi_rate = adiabatic_safe_factor * detuning
    adiabatic_limit = _as_nonnegative_float(
        exp.pulse.calc_control_amplitude(control_qubit, max_rabi_rate),
        name="adiabatic CR amplitude limit",
    )
    return float(np.clip(adiabatic_limit, 0.0, max_amplitude))


def _resolve_amplitude_range(
    values: ArrayLike | None,
    *,
    center: float,
    fraction: float,
    points: int,
    upper: float,
) -> NDArray[np.float64]:
    if values is None:
        half_span = center * fraction
        lower = center - half_span
        high = center + half_span
        if high > upper:
            high = upper
            lower = high - 2.0 * half_span
        if lower <= 0.0:
            raise ValueError("The safe CR amplitude range is too narrow for the sweep.")
        result = np.linspace(lower, high, points)
    else:
        result = _as_finite_vector(values, name="amplitude_range")
    if result.size < 2:
        raise ValueError("amplitude_range must contain at least two points.")
    if np.any(result <= 0.0) or np.any(np.diff(result) <= 0.0):
        raise ValueError("amplitude_range must be strictly increasing and positive.")
    if result[-1] > upper + _TOLERANCE:
        raise ValueError("amplitude_range exceeds the safe CR amplitude limit.")
    return result


def _nearest_zero_crossing(
    amplitudes: NDArray[np.float64],
    signal: NDArray[np.float64],
    *,
    reference: float,
) -> tuple[float, float, float, tuple[float, float], NDArray[np.float64]]:
    """Interpolate every measured crossing and select the nearest reference."""
    candidates: list[tuple[float, float, float, tuple[float, float]]] = []

    for index in np.flatnonzero(signal == 0.0):
        neighbor = index - 1 if index > 0 else index + 1
        neighbor_amplitude = float(amplitudes[neighbor])
        neighbor_signal = float(signal[neighbor])
        crossing_amplitude = float(amplitudes[index])
        crossing_signal = float(signal[index])
        slope, intercept = _line_through_samples(
            neighbor_amplitude,
            neighbor_signal,
            crossing_amplitude,
            crossing_signal,
        )
        bracket = (
            min(neighbor_amplitude, crossing_amplitude),
            max(neighbor_amplitude, crossing_amplitude),
        )
        candidates.append(
            (
                crossing_amplitude,
                slope,
                intercept,
                bracket,
            )
        )

    for index in range(amplitudes.size - 1):
        if signal[index] * signal[index + 1] >= 0.0:
            continue
        slope, intercept = _line_through_samples(
            amplitudes[index],
            signal[index],
            amplitudes[index + 1],
            signal[index + 1],
        )
        root = float(-intercept / slope)
        candidates.append(
            (
                root,
                slope,
                intercept,
                (float(amplitudes[index]), float(amplitudes[index + 1])),
            )
        )

    if not candidates:
        raise ValueError("S_ZX has no zero crossing in the measured sweep range.")
    root, slope, intercept, bracket = min(
        candidates,
        key=lambda candidate: abs(candidate[0] - reference),
    )
    return slope, intercept, root, bracket, slope * amplitudes + intercept


def _line_through_samples(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float]:
    """Return the line through two distinct measured samples."""
    slope = float((y1 - y0) / (x1 - x0))
    if not np.isfinite(slope) or abs(slope) < _MIN_SLOPE:
        raise ValueError("The measured ZX zero-crossing slope is too small.")
    intercept = float(y0 - slope * x0)
    return slope, intercept


def _plot_calibration(
    amplitudes: NDArray[np.float64],
    signal: NDArray[np.float64],
    fitted: NDArray[np.float64],
    *,
    root: float,
    root_bracket: tuple[float, float],
    control_qubit: str,
    target_qubit: str,
) -> go.Figure:
    figure = make_figure()
    figure.add_scatter(x=amplitudes, y=signal, mode="markers", name="S_ZX")
    bracket = np.asarray(root_bracket)
    bracket_indices = np.searchsorted(amplitudes, bracket)
    figure.add_scatter(
        x=bracket,
        y=fitted[bracket_indices],
        mode="lines",
        name="local interpolation",
    )
    figure.add_vline(x=root, line_dash="dash")
    figure.update_layout(
        title=f"Pi-pair rotary ZX90 calibration: {control_qubit}-{target_qubit}",
        xaxis_title="CR amplitude (arb. units)",
        yaxis_title="S_ZX",
    )
    return figure


def _validate_waveform(
    waveform: Waveform,
    *,
    sampling_period: float,
    name: str,
) -> None:
    if not isinstance(waveform, Waveform):
        raise TypeError(f"{name} must be a Waveform.")
    if not np.isclose(
        waveform.sampling_period,
        sampling_period,
        rtol=0.0,
        atol=_TOLERANCE,
    ):
        raise ValueError(f"{name} sampling period does not match the experiment.")
    _validate_channel(np.asarray(waveform.values), name=name)


def _validate_channel(values: NDArray[Any], *, name: str) -> None:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite samples.")
    if np.max(np.abs(values)) > 1.0 + _TOLERANCE:
        raise ValueError(f"{name} amplitude exceeds the hardware limit of one.")


def _require_aligned(value: float, unit: float, *, name: str) -> None:
    ratio = value / unit
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=_TOLERANCE):
        raise ValueError(f"{name} must align to the sampling period.")


def _validate_qubit_labels(control_qubit: str, target_qubit: str) -> None:
    if not isinstance(control_qubit, str) or not control_qubit:
        raise ValueError("control_qubit must be a non-empty string.")
    if not isinstance(target_qubit, str) or not target_qubit:
        raise ValueError("target_qubit must be a non-empty string.")
    if control_qubit == target_qubit:
        raise ValueError("control_qubit and target_qubit must be different.")


def _as_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _as_positive_float(value: Any, *, name: str) -> float:
    result = _as_finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _as_nonnegative_float(value: Any, *, name: str) -> float:
    result = _as_finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _as_integer_at_least(value: Any, *, minimum: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _as_finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real array.") from exc
    if np.iscomplexobj(source):
        raise TypeError(f"{name} must be a one-dimensional real array.")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real array.") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result.copy()
