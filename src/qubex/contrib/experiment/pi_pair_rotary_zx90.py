"""Pi-pair rotary-echo ZX90 pulse construction and amplitude calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from numbers import Real
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from qubex.analysis import util
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
_DEFAULT_SWEEP_POINTS = 33
_DEFAULT_REPETITIONS = (1, 3, 5)
_DEFAULT_ADIABATIC_SAFE_FACTOR = 0.75
_TOLERANCE = 1e-12
_TARGET_STATES = ("0", "1")


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
    repetitions: Sequence[int] = _DEFAULT_REPETITIONS,
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

    The control is prepared in state 0. For each requested odd repetition
    count, target-Z traces are measured with the target prepared in states 0
    and 1, and their half-difference defines `S_N`. All traces are jointly fit
    to `S_N(A) = C_N cos[N(pi/2 + k(A - A_0))]`, where the ZX90 amplitude
    `A_0` and angle slope `k` are shared and each repetition has an independent
    contrast `C_N`.

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
        the configured safe limit. Defaults to 33 points spanning +/-16%
        around the predicted amplitude.
    sweep_fraction
        Fractional half-width of the automatic sweep. Defaults to `0.16`.
    sweep_points
        Number of automatic sweep points. Defaults to `33`.
    repetitions
        Strictly increasing positive odd gate repetition counts. Defaults to
        `(1, 3, 5)`. Include `7` explicitly when coherence permits.
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
        Whether to display the repeated-gate joint fit and final pulse schedule.

    Returns
    -------
    Result
        A result containing repetition-indexed measurements and fitted signals,
        joint-fit diagnostics, the calibration mapping, a coherence-limit
        estimate when T1/T2 data are available, and the final pulse schedule.

    Raises
    ------
    ValueError
        If stored CR data, timing, sweep bounds, pulse amplitudes, measurements,
        or the repeated-gate joint fit is invalid.
    TypeError
        If an input has an incompatible scalar, array, waveform, or mapping
        type.

    Notes
    -----
    This function performs `2 * len(repetitions)` hardware sweeps but does not
    update the ordinary CR calibration note. A fit whose `A_0` is constrained
    to a sweep boundary is rejected because it does not establish an in-range
    ZX90 point. The reported `rotary_amplitude` is the peak absolute sample
    amplitude of the calibrated target X180 waveform.
    """
    _validate_qubit_labels(control_qubit, target_qubit)
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be a boolean.")
    fraction = _as_positive_float(sweep_fraction, name="sweep_fraction")
    if fraction >= 1.0:
        raise ValueError("sweep_fraction must be less than one.")
    points = _as_integer_at_least(sweep_points, minimum=2, name="sweep_points")
    resolved_repetitions = _resolve_repetitions(repetitions)
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

    state_values: dict[int, NDArray[np.float64]] = {}
    signals: dict[int, NDArray[np.float64]] = {}
    raw_results: dict[int, tuple[Any, ...]] = {}
    for repetition in resolved_repetitions:
        repeated_state_values = np.empty((2, amplitudes.size), dtype=np.float64)
        repeated_raw_results: list[Any] = []

        def sequence(candidate: float, repetition: int = repetition) -> PulseSchedule:
            gate = pi_pair_rotary_zx90(
                exp,
                control_qubit,
                target_qubit,
                cr_amplitude=float(candidate),
                duration=resolved_duration,
                x180=x180,
                x180_margin=x180_margin,
            )
            return gate.repeated(repetition)

        for state_index, target_state in enumerate(_TARGET_STATES):
            measured = exp.measurement_service.sweep_parameter(
                sequence=sequence,
                sweep_range=amplitudes,
                initial_states={
                    control_qubit: "0",
                    target_qubit: target_state,
                },
                n_shots=resolved_shots,
                shot_interval=resolved_interval,
                plot=False,
                title=(
                    "Pi-pair rotary ZX90 calibration: "
                    f"{control_qubit}-{target_qubit} (N={repetition})"
                ),
                xlabel="CR amplitude (arb. units)",
                ylabel="Normalized target Z",
            )
            repeated_raw_results.append(measured)
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
            repeated_state_values[state_index] = values

        state_values[repetition] = repeated_state_values
        signals[repetition] = (
            repeated_state_values[0] - repeated_state_values[1]
        ) / 2.0
        raw_results[repetition] = tuple(repeated_raw_results)

    fit = _fit_repeated_gate_signals(
        amplitudes,
        signals,
        repetitions=resolved_repetitions,
        predicted_amplitude=predicted,
    )
    root = float(fit["zx90_amplitude"])
    fit_range = (float(amplitudes[0]), float(amplitudes[-1]))

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
    rotary_amplitude = float(np.max(np.abs(target_pi.values)))
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
        "rotary_amplitude": rotary_amplitude,
        "predicted_cr_amplitude": predicted,
        "max_cr_amplitude": max_cr_amplitude,
        "sampling_period": sampling_period,
        "x180_margin": float(x180_margin),
        "angle_slope": fit["angle_slope"],
        "contrasts": fit["contrasts"],
        "repetitions": resolved_repetitions,
    }
    coherence_limit = _calculate_coherence_limit(
        exp,
        control_qubit,
        target_qubit,
        gate_time=final_schedule.duration,
    )
    _print_calibrated_parameters(calibration)
    _print_joint_fit(fit)
    if coherence_limit:
        _print_coherence_limit(coherence_limit)
    figure = _plot_calibration(
        amplitudes,
        signals,
        fit["fitted_signals"],
        root=root,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
    )
    if plot:
        figure.show()
        final_schedule.plot(
            title=f"Pi-pair rotary ZX90 sequence : {control_qubit}-{target_qubit}",
            show_physical_pulse=True,
        )
    return Result(
        data={
            "pi_pair_rotary_zx90_calibration": calibration,
            "amplitude_range": amplitudes,
            "target_state_order": _TARGET_STATES,
            "state_values": state_values,
            "signals": signals,
            "fitted_signals": fit["fitted_signals"],
            "fit": fit,
            "root": root,
            "fit_range": fit_range,
            "raw_results": raw_results,
            "pulse_schedule": final_schedule,
            "coherence_limit": coherence_limit,
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


def _resolve_repetitions(values: Sequence[int]) -> tuple[int, ...]:
    """Validate repeated-gate counts and return an immutable sequence."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("repetitions must be a sequence of integers.")
    result = tuple(
        _as_integer_at_least(value, minimum=1, name="repetitions entry")
        for value in values
    )
    if not result:
        raise ValueError("repetitions must contain at least one value.")
    if any(value % 2 == 0 for value in result):
        raise ValueError("repetitions must contain only odd integers.")
    if any(left >= right for left, right in pairwise(result)):
        raise ValueError("repetitions must be strictly increasing and unique.")
    return result


def _fit_repeated_gate_signals(
    amplitudes: NDArray[np.float64],
    signals: Mapping[int, NDArray[np.float64]],
    *,
    repetitions: tuple[int, ...],
    predicted_amplitude: float,
) -> dict[str, Any]:
    """Jointly fit repeated-gate signals to one ZX90 amplitude and slope."""
    signal_variation = max(float(np.ptp(signals[value])) for value in repetitions)
    if signal_variation <= 10.0 * np.finfo(float).eps:
        raise ValueError(
            "Repeated-gate joint fit is not identifiable from flat measured signals."
        )
    angle_slope_initial = np.pi / (2.0 * predicted_amplitude)
    initial_contrasts: list[float] = []
    for repetition in repetitions:
        basis = np.cos(
            repetition
            * (np.pi / 2.0 + angle_slope_initial * (amplitudes - predicted_amplitude))
        )
        denominator = float(np.dot(basis, basis))
        contrast = (
            float(np.dot(signals[repetition], basis) / denominator)
            if denominator > 0.0
            else 1.0
        )
        initial_contrasts.append(float(np.clip(contrast, 0.05, 1.0)))

    def residuals(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        zx90_amplitude = float(parameters[0])
        angle_slope = float(parameters[1])
        return np.concatenate(
            [
                _repeated_gate_model(
                    amplitudes,
                    repetition=repetition,
                    zx90_amplitude=zx90_amplitude,
                    angle_slope=angle_slope,
                    contrast=float(parameters[index + 2]),
                )
                - signals[repetition]
                for index, repetition in enumerate(repetitions)
            ]
        )

    lower_amplitude = float(amplitudes[0])
    upper_amplitude = float(amplitudes[-1])
    amplitude_epsilon = np.finfo(float).eps * max(1.0, abs(predicted_amplitude))
    initial_amplitude = float(
        np.clip(
            predicted_amplitude,
            lower_amplitude + amplitude_epsilon,
            upper_amplitude - amplitude_epsilon,
        )
    )
    initial = np.asarray(
        [initial_amplitude, angle_slope_initial, *initial_contrasts],
        dtype=np.float64,
    )
    lower_bounds = np.asarray(
        [lower_amplitude, 0.0, *([0.0] * len(repetitions))],
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        [
            upper_amplitude,
            4.0 * angle_slope_initial,
            *([1.2] * len(repetitions)),
        ],
        dtype=np.float64,
    )
    result = least_squares(
        residuals,
        initial,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=10_000,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Repeated-gate joint fit failed: {result.message}")
    if result.active_mask[0] != 0:
        raise ValueError(
            "The fitted ZX90 amplitude is constrained to a sweep boundary; "
            "expand or recenter amplitude_range."
        )

    zx90_amplitude = float(result.x[0])
    angle_slope = float(result.x[1])
    contrasts = {
        repetition: float(result.x[index + 2])
        for index, repetition in enumerate(repetitions)
    }
    fitted_signals = {
        repetition: _repeated_gate_model(
            amplitudes,
            repetition=repetition,
            zx90_amplitude=zx90_amplitude,
            angle_slope=angle_slope,
            contrast=contrasts[repetition],
        )
        for repetition in repetitions
    }
    residual_vector = residuals(result.x)
    jacobian = np.asarray(result.jac, dtype=np.float64)
    if np.linalg.matrix_rank(jacobian) < result.x.size:
        raise ValueError(
            "Repeated-gate joint fit is not identifiable from the measured signals."
        )
    residual_rms = float(np.sqrt(np.mean(np.square(residual_vector))))
    amplitude_stderr = _parameter_standard_error(
        jacobian,
        residual_vector,
        parameter_index=0,
    )
    return {
        "success": True,
        "message": str(result.message),
        "zx90_amplitude": zx90_amplitude,
        "angle_slope": angle_slope,
        "contrasts": contrasts,
        "amplitude_stderr": amplitude_stderr,
        "residual_rms": residual_rms,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "repetitions": repetitions,
        "fitted_signals": fitted_signals,
    }


def _repeated_gate_model(
    amplitudes: NDArray[np.float64],
    *,
    repetition: int,
    zx90_amplitude: float,
    angle_slope: float,
    contrast: float,
) -> NDArray[np.float64]:
    """Evaluate one repeated-gate calibration signal."""
    angle = np.pi / 2.0 + angle_slope * (amplitudes - zx90_amplitude)
    return contrast * np.cos(repetition * angle)


def _parameter_standard_error(
    jacobian: NDArray[np.float64],
    residuals: NDArray[np.float64],
    *,
    parameter_index: int,
) -> float:
    """Estimate one parameter standard error from the fit Jacobian."""
    degrees_of_freedom = residuals.size - jacobian.shape[1]
    if degrees_of_freedom <= 0:
        return float("nan")
    residual_variance = float(np.dot(residuals, residuals) / degrees_of_freedom)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * residual_variance
    variance = float(covariance[parameter_index, parameter_index])
    return float(np.sqrt(max(variance, 0.0)))


def _plot_calibration(
    amplitudes: NDArray[np.float64],
    signals: Mapping[int, NDArray[np.float64]],
    fitted_signals: Mapping[int, NDArray[np.float64]],
    *,
    root: float,
    control_qubit: str,
    target_qubit: str,
) -> go.Figure:
    figure = make_figure()
    colors = ("#636EFA", "#EF553B", "#00CC96", "#AB63FA")
    for index, repetition in enumerate(signals):
        color = colors[index % len(colors)]
        figure.add_scatter(
            x=amplitudes,
            y=signals[repetition],
            mode="markers",
            name=f"S_{repetition} data",
            marker={"color": color},
        )
        figure.add_scatter(
            x=amplitudes,
            y=fitted_signals[repetition],
            mode="lines",
            name=f"N={repetition} fit",
            line={"color": color},
        )
    figure.add_annotation(
        x=root,
        y=0.0,
        text=f"ZX90 amplitude: {root:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    figure.update_layout(
        title=f"Pi-pair rotary ZX90 calibration : {control_qubit}-{target_qubit}",
        xaxis_title="Amplitude (arb. units)",
        yaxis_title="Repeated-gate signal S_N",
    )
    return figure


def _calculate_coherence_limit(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    gate_time: float,
) -> dict[str, float | str]:
    """Calculate the coherence limit using the emitted schedule duration."""
    try:
        config_loader = exp.ctx.system_manager.config_loader
        t1_data = config_loader.load_param_data("t1")
        t2_data = config_loader.load_param_data("t2_echo")
        t1 = (
            _as_positive_float(t1_data[control_qubit], name="control T1"),
            _as_positive_float(t1_data[target_qubit], name="target T1"),
        )
        t2 = (
            _as_positive_float(t2_data[control_qubit], name="control echo T2"),
            _as_positive_float(t2_data[target_qubit], name="target echo T2"),
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}

    duration = _as_positive_float(gate_time, name="ZX90 gate time")
    return {
        "control_qubit": control_qubit,
        "target_qubit": target_qubit,
        "gate_time": duration,
        "t1_control": t1[0],
        "t1_target": t1[1],
        "t2_control": t2[0],
        "t2_target": t2[1],
        **util.calc_2q_gate_coherence_limit(
            gate_time=duration,
            t1=t1,
            t2=t2,
        ),
    }


def _print_calibrated_parameters(calibration: Mapping[str, Any]) -> None:
    """Print the calibrated pi-pair rotary ZX90 parameters."""
    print()
    print("Calibrated CR parameters:")
    print(f"  CR duration      : {float(calibration['duration']):.1f} ns")
    print(f"  CR ramptime      : {float(calibration['ramptime']):.1f} ns")
    print(f"  CR amplitude     : {float(calibration['cr_amplitude']):.6f}")
    print(f"  CR phase         : {float(calibration['cr_phase']):.6f}")
    print(f"  CR beta          : {float(calibration['cr_beta']):.6f}")
    print(f"  Cancel amplitude : {float(calibration['cancel_amplitude']):.6f}")
    print(f"  Cancel phase     : {float(calibration['cancel_phase']):.6f}")
    print(f"  Cancel beta      : {float(calibration['cancel_beta']):.6f}")
    print(f"  Rotary amplitude : {float(calibration['rotary_amplitude']):.6f}")
    print()


def _print_joint_fit(fit: Mapping[str, Any]) -> None:
    """Print repeated-gate joint-fit parameters."""
    repetitions = tuple(int(value) for value in fit["repetitions"])
    contrasts = fit["contrasts"]
    if not isinstance(contrasts, Mapping):
        raise TypeError("fit contrasts must be a mapping.")
    print("Repeated-gate joint fit:")
    print(f"  Repetitions      : {', '.join(str(value) for value in repetitions)}")
    print(f"  Angle slope      : {float(fit['angle_slope']):.6f} rad / amplitude")
    for repetition in repetitions:
        print(f"  Contrast (N={repetition})  : {float(contrasts[repetition]):.6f}")
    print(f"  Amplitude stderr : {float(fit['amplitude_stderr']):.6g}")
    print(f"  Residual RMS     : {float(fit['residual_rms']):.6g}")
    print()


def _print_coherence_limit(coherence_limit: Mapping[str, float | str]) -> None:
    """Print the pi-pair rotary ZX90 coherence-limit estimate."""
    print("ZX90 coherence limit:")
    print(f"  Gate time       : {float(coherence_limit['gate_time']):.0f} ns")
    print(f"  T1 (control)    : {float(coherence_limit['t1_control']) * 1e-3:.1f} μs")
    print(f"  T1 (target)     : {float(coherence_limit['t1_target']) * 1e-3:.1f} μs")
    print(f"  T2 (control)    : {float(coherence_limit['t2_control']) * 1e-3:.1f} μs")
    print(f"  T2 (target)     : {float(coherence_limit['t2_target']) * 1e-3:.1f} μs")
    print(f"  Coherence limit : {float(coherence_limit['fidelity']) * 100:.2f} %")
    print()


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
