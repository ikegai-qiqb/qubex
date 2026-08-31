"""One-qubit zero-crossing calibration for SRRE amplitude."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import CALIBRATION_SHOTS
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import Arbitrary, PulseSchedule, Waveform

from .srre_waveform import predict_srre_amplitude, srre_waveform

__all__ = ["calibrate_srre"]

_DEFAULT_SWEEP_FRACTIONS = np.linspace(-0.08, 0.08, 17)
_MIN_ABSOLUTE_SLOPE = 1e-9
_SAMPLING_PERIOD_TOLERANCE = 1e-12
_HARDWARE_AMPLITUDE_LIMIT = 1.0
_HARDWARE_AMPLITUDE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _ZeroCrossingAnalysis:
    root: float
    root_bracket: tuple[float, float]
    fit_slope: float
    fit_intercept: float
    fitted_signal: NDArray[np.float64]
    differential_signal: NDArray[np.float64]


def calibrate_srre(
    exp: Experiment,
    target: str,
    *,
    block_duration: float,
    ramp_time: float,
    amplitude_range: ArrayLike | None = None,
    amplitude_bounds: tuple[float, float] = (0.0, 1.0),
    probe_detuning: float | None = None,
    repetitions: int = 4,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
) -> Result:
    """
    Calibrate one SRRE block amplitude from a differential zero crossing.

    For each candidate amplitude, this experiment prepares `|+X>`, applies
    repeated SRRE under `+probe_detuning` and `-probe_detuning`, projects along
    the predicted analysis axis, and measures target Z. The accepted amplitude
    is the zero crossing of one least-squares line fitted to every measured
    differential-signal point.

    Parameters
    ----------
    exp : Experiment
        Experiment instance used for pulse generation and measurement.
    target : str
        Target qubit label.
    block_duration : float
        Duration of one complete SRRE block in ns.
    ramp_time : float
        Ramp-up and ramp-down duration of each lobe in ns.
    amplitude_range : ArrayLike | None, optional
        Strictly increasing candidate peak amplitudes. Defaults to 17 points
        spanning ±8% around the predicted root, clipped to `amplitude_bounds`.
    amplitude_bounds : tuple[float, float], optional
        Non-negative hardware amplitude bounds used for root prediction.
    probe_detuning : float | None, optional
        Positive detuning magnitude in GHz. Defaults to a value accumulating
        pi/2 phase over the complete repeated SRRE sequence. An explicit value
        must be below the waveform sampling Nyquist frequency.
    repetitions : int, optional
        Positive number of contiguous SRRE blocks. Defaults to `4`.
    n_shots : int | None, optional
        Shots per detuning and amplitude point. Defaults to calibration shots.
    shot_interval : float | None, optional
        Shot interval in ns. Defaults to the current experiment context.
    plot : bool, optional
        Whether to plot differential signal against amplitude together with
        the fitted line.

    Returns
    -------
    Result
        Result containing the `srre_calibration` data contract.

    Raises
    ------
    ValueError
        If inputs, reference pulses, or measured signals are invalid, or the
        fitted root is outside the measured amplitude range.
    TypeError
        If an array or scalar input has an incompatible type.

    Notes
    -----
    This function performs hardware measurements but does not update the
    calibration note or any stored pulse parameters.
    """
    _validate_label(target, name="target")
    repetitions = _validate_repetitions(repetitions)
    resolved_shots = _resolve_optional_positive_integer(
        n_shots,
        default=CALIBRATION_SHOTS,
        name="n_shots",
    )
    resolved_interval = _resolve_optional_positive_float(
        shot_interval,
        default=_context_shot_interval(exp),
        name="shot_interval",
    )
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be a boolean.")
    plot = bool(plot)
    sampling_period = _as_positive_float(
        exp.ctx.util.resolve_sampling_period(exp.ctx.measurement.sampling_period),
        name="sampling_period",
    )

    exp.pulse.validate_rabi_params([target])

    def rabi_rate_from_amplitude(amplitude: float) -> float:
        return float(exp.pulse.calc_rabi_rate(target, amplitude))

    prediction = predict_srre_amplitude(
        block_duration=block_duration,
        ramp_time=ramp_time,
        rabi_rate_from_amplitude=rabi_rate_from_amplitude,
        amplitude_bounds=amplitude_bounds,
        sampling_period=sampling_period,
    )
    amplitudes = _resolve_amplitude_range(
        amplitude_range,
        predicted_amplitude=prediction.amplitude,
        amplitude_bounds=amplitude_bounds,
    )
    detuning = _resolve_probe_detuning(
        probe_detuning,
        block_duration=block_duration,
        repetitions=repetitions,
        sampling_period=sampling_period,
    )
    analysis_angle = _wrap_to_small_angle(np.pi / 2 + prediction.phi_pred)

    sweep_points = [
        (float(amplitude), detuning_sign * detuning)
        for amplitude in amplitudes
        for detuning_sign in (1.0, -1.0)
    ]

    def sequence(point_index: float) -> PulseSchedule:
        amplitude, signed_detuning = sweep_points[int(point_index)]
        return _build_srre_calibration_sequence(
            exp,
            target,
            block_duration=block_duration,
            ramp_time=ramp_time,
            amplitude=amplitude,
            detuning=signed_detuning,
            repetitions=repetitions,
            analysis_angle=analysis_angle,
            sampling_period=sampling_period,
        )

    sweep_result = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=np.arange(len(sweep_points)),
        n_shots=resolved_shots,
        shot_interval=resolved_interval,
        plot=False,
        title=f"SRRE amplitude calibration: {target}",
        xlabel="Interleaved amplitude/detuning point",
        ylabel="Normalized target Z",
    )
    if target not in sweep_result.data:
        raise ValueError(f"SRRE measurement result is missing target `{target}`.")
    normalized = _as_finite_vector(
        sweep_result.data[target].normalized,
        name="normalized SRRE measurement",
    )
    expected_points = 2 * amplitudes.size
    if normalized.shape != (expected_points,):
        raise ValueError(
            "SRRE measurement must return one scalar for every interleaved "
            f"detuning point; expected {expected_points}, got shape {normalized.shape}."
        )
    signal_plus = normalized[0::2].copy()
    signal_minus = normalized[1::2].copy()
    analysis = _analyze_zero_crossing(
        amplitudes=amplitudes,
        signal_plus=signal_plus,
        signal_minus=signal_minus,
    )
    if plot:
        _plot_linear_fit(
            amplitudes,
            analysis.differential_signal,
            analysis.fitted_signal,
            title=f"SRRE amplitude calibration: {target}",
            xlabel="SRRE amplitude",
            ylabel="Differential signal",
        )
    calibrated_rabi_rate = rabi_rate_from_amplitude(analysis.root)
    if not np.isfinite(calibrated_rabi_rate):
        raise ValueError("The calibrated amplitude must map to a finite Rabi rate.")

    return Result(
        data={
            "srre_calibration": {
                "target": target,
                "amplitude": analysis.root,
                "predicted_amplitude": prediction.amplitude,
                "rabi_rate": calibrated_rabi_rate,
                "block_duration": float(block_duration),
                "ramp_time": float(ramp_time),
                "sampling_period": sampling_period,
                "positive_lobe_angle": prediction.positive_lobe_angle,
                "phi_pred": prediction.phi_pred,
                "analysis_angle": analysis_angle,
                "probe_detuning": detuning,
                "repetitions": repetitions,
                "f0_predicted": prediction.f0,
                "f1_predicted": prediction.f1,
                "root_bracket": analysis.root_bracket,
                "fit_slope": analysis.fit_slope,
                "fit_intercept": analysis.fit_intercept,
                "fitted_signal": analysis.fitted_signal,
                "amplitude_range": amplitudes,
                "signal_plus": signal_plus,
                "signal_minus": signal_minus,
                "differential_signal": analysis.differential_signal,
                "raw_results": sweep_result,
            }
        }
    )


def _build_srre_calibration_sequence(
    exp: Experiment,
    target: str,
    *,
    block_duration: float,
    ramp_time: float,
    amplitude: float,
    detuning: float,
    repetitions: int,
    analysis_angle: float,
    sampling_period: float,
) -> PulseSchedule:
    """Build one detuned SRRE measurement sequence with continuous phase."""
    repetitions = _validate_repetitions(repetitions)
    if not np.isfinite(detuning):
        raise ValueError("detuning must be finite.")
    if not np.isfinite(analysis_angle):
        raise ValueError("analysis_angle must be finite.")

    preparation_pulse = exp.pulse.y90(target)
    analysis_reference = exp.pulse.x90(target)
    _validate_reference_pulse(
        preparation_pulse,
        name="preparation pulse",
        sampling_period=sampling_period,
    )
    _validate_reference_pulse(
        analysis_reference,
        name="analysis reference pulse",
        sampling_period=sampling_period,
    )

    block = srre_waveform(
        block_duration=block_duration,
        ramp_time=ramp_time,
        amplitude=amplitude,
        sampling_period=sampling_period,
    )
    repeated_values = np.tile(block.values, repetitions)
    sample_times = np.arange(repeated_values.size) * sampling_period
    phase_ramp = np.exp(-2j * np.pi * detuning * sample_times)
    detuned_srre = Arbitrary(
        repeated_values * phase_ramp,
        sampling_period=sampling_period,
    )

    total_srre_duration = block_duration * repetitions
    phase_correction = -2.0 * np.pi * detuning * total_srre_duration
    analysis_pulse = analysis_reference.scaled(analysis_angle / (np.pi / 2)).shifted(
        phase_correction
    )
    _validate_waveform_peak(analysis_pulse, name="analysis pulse")

    with PulseSchedule([target]) as schedule:
        schedule.add(target, preparation_pulse)
        schedule.add(target, detuned_srre)
        schedule.add(target, analysis_pulse)
    return schedule


def _analyze_zero_crossing(
    *,
    amplitudes: ArrayLike,
    signal_plus: ArrayLike,
    signal_minus: ArrayLike,
) -> _ZeroCrossingAnalysis:
    """Fit one line to the measured differential signal and return its root."""
    amplitude_values = _as_finite_vector(amplitudes, name="amplitudes")
    plus_values = _as_finite_vector(signal_plus, name="signal_plus")
    minus_values = _as_finite_vector(signal_minus, name="signal_minus")
    if plus_values.shape != amplitude_values.shape:
        raise ValueError("signal_plus must have the same shape as amplitudes.")
    if minus_values.shape != amplitude_values.shape:
        raise ValueError("signal_minus must have the same shape as amplitudes.")
    if amplitude_values.size < 2:
        raise ValueError("amplitudes must contain at least two points.")
    if np.any(np.diff(amplitude_values) <= 0):
        raise ValueError("amplitudes must be strictly increasing.")

    differential = (plus_values - minus_values) / 2.0
    slope, intercept, fitted_signal = _fit_line(amplitude_values, differential)
    if not np.isfinite(slope) or abs(slope) < _MIN_ABSOLUTE_SLOPE:
        raise ValueError("Measured SRRE zero-crossing slope is too small.")
    root = -intercept / slope
    lower = float(amplitude_values[0])
    upper = float(amplitude_values[-1])
    if not lower <= root <= upper:
        raise ValueError("Fitted SRRE root lies outside the measured amplitude range.")
    if root <= 0.0:
        raise ValueError("Measured SRRE amplitude root must be positive.")

    return _ZeroCrossingAnalysis(
        root=float(root),
        root_bracket=(lower, upper),
        fit_slope=float(slope),
        fit_intercept=float(intercept),
        fitted_signal=fitted_signal,
        differential_signal=differential,
    )


def _fit_line(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64]]:
    """Return a least-squares line and its values on the supplied grid."""
    centered_x = x_values - np.mean(x_values)
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0.0:
        raise ValueError("Linear fit requires at least two distinct x values.")
    slope = float(np.dot(centered_x, y_values - np.mean(y_values)) / denominator)
    intercept = float(np.mean(y_values) - slope * np.mean(x_values))
    return slope, intercept, slope * x_values + intercept


def _plot_linear_fit(
    x_values: NDArray[np.float64],
    signal: NDArray[np.float64],
    fitted_signal: NDArray[np.float64],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Show measured zero-crossing data together with its fitted line."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=x_values, y=signal, mode="markers", name="Measurement")
    )
    figure.add_trace(
        go.Scatter(x=x_values, y=fitted_signal, mode="lines", name="Linear fit")
    )
    figure.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    figure.show()


def _resolve_amplitude_range(
    amplitude_range: ArrayLike | None,
    *,
    predicted_amplitude: float,
    amplitude_bounds: tuple[float, float],
) -> NDArray[np.float64]:
    lower, upper = amplitude_bounds
    if amplitude_range is None:
        values = predicted_amplitude * (1.0 + _DEFAULT_SWEEP_FRACTIONS)
        values = np.clip(values, lower, upper)
        values = np.unique(values)
    else:
        values = _as_finite_vector(amplitude_range, name="amplitude_range")

    if values.size < 2:
        raise ValueError("amplitude_range must contain at least two distinct points.")
    if np.any(np.diff(values) <= 0):
        raise ValueError("amplitude_range must be strictly increasing.")
    if values[0] < lower or values[-1] > upper:
        raise ValueError("amplitude_range must lie within amplitude_bounds.")
    return values


def _resolve_probe_detuning(
    probe_detuning: float | None,
    *,
    block_duration: float,
    repetitions: int,
    sampling_period: float,
) -> float:
    if probe_detuning is None:
        return 1.0 / (4.0 * block_duration * repetitions)
    value = _as_finite_float(probe_detuning, name="probe_detuning")
    if value == 0:
        raise ValueError("probe_detuning must be non-zero.")
    if value < 0:
        raise ValueError("probe_detuning must be positive.")
    nyquist_frequency = 0.5 / sampling_period
    if value >= nyquist_frequency:
        raise ValueError(
            "probe_detuning must be below the sampling Nyquist frequency "
            f"({nyquist_frequency} GHz)."
        )
    return value


def _validate_repetitions(repetitions: int) -> int:
    return _as_positive_integer(repetitions, name="repetitions")


def _context_shot_interval(exp: Experiment) -> float:
    """Resolve the configured shot interval for the current experiment context."""
    experiment_system = getattr(exp.ctx, "experiment_system", None)
    configured_defaults = getattr(experiment_system, "measurement_defaults", None)
    defaults = resolve_measurement_defaults(configured_defaults)
    return float(defaults.execution.shot_interval_ns)


def _validate_label(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _as_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _as_positive_float(value: object, *, name: str) -> float:
    result = _as_finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _resolve_optional_positive_integer(
    value: object,
    *,
    default: int,
    name: str,
) -> int:
    return default if value is None else _as_positive_integer(value, name=name)


def _as_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _resolve_optional_positive_float(
    value: object,
    *,
    default: float,
    name: str,
) -> float:
    return default if value is None else _as_positive_float(value, name=name)


def _as_finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real array.") from exc
    if np.iscomplexobj(source):
        raise TypeError(f"{name} must be a one-dimensional real array.")
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real array.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_reference_pulse(
    pulse: Waveform,
    *,
    name: str,
    sampling_period: float,
) -> None:
    if not isinstance(pulse, Waveform):
        raise TypeError(f"{name} must be a Waveform.")
    if not np.isclose(
        pulse.sampling_period,
        sampling_period,
        rtol=0.0,
        atol=_SAMPLING_PERIOD_TOLERANCE,
    ):
        raise ValueError(
            f"{name} sampling period ({pulse.sampling_period} ns) must match "
            f"the SRRE sampling period ({sampling_period} ns)."
        )
    _validate_waveform_peak(pulse, name=name)


def _validate_waveform_peak(pulse: Waveform, *, name: str) -> None:
    values = np.asarray(pulse.values, dtype=np.complex128)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite samples.")
    if np.max(np.abs(values)) > (
        _HARDWARE_AMPLITUDE_LIMIT + _HARDWARE_AMPLITUDE_TOLERANCE
    ):
        raise ValueError(f"{name} amplitude must not exceed the hardware limit of 1.")


def _wrap_to_small_angle(angle: float) -> float:
    if not np.isfinite(angle):
        raise ValueError("analysis angle must be finite.")
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
