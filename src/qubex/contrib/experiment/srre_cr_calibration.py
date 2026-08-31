"""SRRE-assisted ZX90 calibration orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import CALIBRATION_SHOTS
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import Arbitrary, PulseArray, PulseSchedule, Waveform

from .srre_calibration import calibrate_srre
from .srre_cross_resonance import _build_srre_cross_resonance

__all__ = ["calibrate_srre_zx90"]

_Stage = Literal["zx", "zy", "iy", "ix"]

_REFERENCE_ANGLE = np.pi / 2.0
_DURATION_UNIT = 16.0
_NUMERIC_TOLERANCE = 1e-12
_MIN_ABSOLUTE_SLOPE = 1e-9
_SIGNAL_TOLERANCE = 0.02
_DEFAULT_CR_SWEEP_FACTORS = np.linspace(0.84, 1.16, 9)
_FINE_CR_SWEEP_FACTORS = np.linspace(0.92, 1.08, 17)
_DEFAULT_PHASE_OFFSETS = np.linspace(-0.16, 0.16, 17)
_DEFAULT_CANCEL_SWEEP_POINTS = 21
_DEFAULT_CANCEL_EDGE_ROTATION = 0.2
_MAX_SWEEP_EXPANSION_FACTOR = 2.0
_STAGE_CONFIGURATIONS: dict[_Stage, tuple[bool, bool]] = {
    "zx": (True, True),
    "zy": (True, False),
    "iy": (False, False),
    "ix": (False, True),
}
_STAGE_PARAMETER_KEYS: dict[_Stage, str] = {
    "zx": "cr_amplitude",
    "zy": "cr_phase",
    "iy": "cancel_y",
    "ix": "cancel_x",
}
_FINE_STAGE_DATA_KEYS: tuple[tuple[str, _Stage], ...] = (
    ("phase_stage", "zy"),
    ("cancel_y_stage", "iy"),
    ("cancel_x_stage", "ix"),
)


@dataclass(frozen=True)
class _ZxSignals:
    zx: float
    ix_from_z: float


@dataclass(frozen=True)
class _ZeroCrossingAnalysis:
    root: float
    root_bracket: tuple[float, float]
    fit_slope: float
    fit_intercept: float
    fitted_signal: NDArray[np.float64]


@dataclass(frozen=True)
class _DurationResolution:
    requested_duration: float | None
    predicted_duration: float
    predicted_cr_amplitude: float
    resolved_duration: float
    duration_unit: float
    sampling_period: float
    srre_lobe_duration: float
    srre_flat_time: float
    source: str


@dataclass(frozen=True)
class _StageMeasurement:
    sweep_values: NDArray[np.float64]
    state_values: NDArray[np.float64]
    error_signal: NDArray[np.float64]
    diagnostic_signals: dict[str, NDArray[np.float64]]
    raw_results: tuple[Any, ...]


@dataclass(frozen=True)
class _StageRun:
    data: dict[str, Any]
    accepted_calibration: dict[str, Any]


@dataclass(frozen=True)
class _FineRoundRun:
    data: dict[str, Any]
    accepted_calibration: dict[str, Any]
    final_verification: dict[str, Any] | None
    failed: bool
    converged: bool


def calibrate_srre_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    cr_half_duration: float | None = None,
    srre_ramp_time: float,
    srre_calibration: Mapping[str, Any] | None = None,
    srre_amplitude_range: ArrayLike | None = None,
    probe_detuning: float | None = None,
    cr_amplitude_range: ArrayLike | None = None,
    cr_phase_offsets: ArrayLike | None = None,
    cancel_y_offsets: ArrayLike | None = None,
    cancel_x_offsets: ArrayLike | None = None,
    error_amplification_n: int = 1,
    max_fine_rounds: int = 2,
    scale_cancellation_with_cr: bool = True,
    n_shots: int | None = None,
    verification_n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
) -> Result:
    """
    Calibrate an SRRE-assisted ZX90 gate without updating stored CR parameters.

    The workflow consumes CR parameters previously produced by
    `obtain_cr_params`, fixes one CR-half/SRRE geometry, calibrates or reuses the
    one-qubit SRRE amplitude, and performs the documented ZX, ZY, IY, and IX
    four-state zero-crossing stages. Every proposed parameter is remeasured at
    a fresh verification point before it is accepted.

    Parameters
    ----------
    exp : Experiment
        Experiment instance used for pulse construction and measurement.
    control_qubit : str
        Control qubit label.
    target_qubit : str
        Target qubit label.
    cr_half_duration : float | None, optional
        Fixed duration of one CR half in ns. When omitted, the value is
        predicted from the stored ZX rate and aligned to 16 ns.
    srre_ramp_time : float
        Ramp-up and ramp-down time of each SRRE lobe in ns.
    srre_calibration : Mapping[str, Any] | None, optional
        Reusable `srre_calibration` mapping. When omitted, `calibrate_srre` is
        run once for the fixed geometry.
    srre_amplitude_range : ArrayLike | None, optional
        One-qubit SRRE amplitude sweep passed to `calibrate_srre`.
    probe_detuning : float | None, optional
        Positive one-qubit SRRE probe detuning in GHz.
    cr_amplitude_range : ArrayLike | None, optional
        Strictly increasing positive absolute CR amplitudes not exceeding one
        for stages 8 and 11. When omitted, nine points spanning +/-16% around
        the ZX90 amplitude predicted for the fixed CR-half duration are used.
        The fitted root is subsequently fine-tuned over 17 points spanning
        +/-8%.
    cr_phase_offsets : ArrayLike | None, optional
        Strictly increasing phase offsets in radians for stage 9a. Defaults to
        17 points spanning +/-0.16 rad.
    cancel_y_offsets : ArrayLike | None, optional
        Strictly increasing cancellation-Y offsets for stage 9b. By default,
        target Rabi parameters set 21 symmetric points whose sweep edges rotate
        by `0.2 / N` rad in one candidate gate.
    cancel_x_offsets : ArrayLike | None, optional
        Strictly increasing cancellation-X offsets for stage 9c. Its default is
        resolved in the same way as `cancel_y_offsets`.
    error_amplification_n : int, optional
        Positive `N`; fine-stage sequences contain exactly `2N` cycles.
    max_fine_rounds : int, optional
        Positive maximum number of phase/cancellation/final-angle rounds.
    scale_cancellation_with_cr : bool, optional
        Whether accepted cancellation IQ scales with CR amplitude candidates.
    n_shots : int | None, optional
        Shots per four-state sweep point. Defaults to calibration shots.
    verification_n_shots : int | None, optional
        Shots per fresh verification point. Defaults to `n_shots`.
    shot_interval : float | None, optional
        Shot interval in ns. Defaults to the current experiment context.
    plot : bool, optional
        Whether to plot every differential calibration signal together with its
        linear fit.

    Returns
    -------
    Result
        Result containing the complete `srre_cr_calibration` data contract.

    Raises
    ------
    ValueError
        If stored parameters, geometry, reused SRRE metadata, or sweep inputs
        are invalid.
    TypeError
        If an input has an incompatible scalar or mapping type.

    Notes
    -----
    This function performs hardware measurements but does not run
    `obtain_cr_params`, update the calibration note, or mutate reused
    calibration mappings.
    """
    _validate_label(control_qubit, name="control_qubit")
    _validate_label(target_qubit, name="target_qubit")
    if control_qubit == target_qubit:
        raise ValueError("control_qubit and target_qubit must be different.")
    error_amplification_n = _as_positive_integer(
        error_amplification_n, name="error_amplification_n"
    )
    max_fine_rounds = _as_positive_integer(max_fine_rounds, name="max_fine_rounds")
    if not isinstance(scale_cancellation_with_cr, (bool, np.bool_)):
        raise TypeError("scale_cancellation_with_cr must be a boolean.")
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be a boolean.")
    scale_cancellation_with_cr = bool(scale_cancellation_with_cr)
    plot = bool(plot)
    resolved_shots = _resolve_optional_positive_integer(
        n_shots,
        default=CALIBRATION_SHOTS,
        name="n_shots",
    )
    resolved_verification_shots = _resolve_optional_positive_integer(
        verification_n_shots,
        default=resolved_shots,
        name="verification_n_shots",
    )
    resolved_interval = _resolve_optional_positive_float(
        shot_interval,
        default=_context_shot_interval(exp),
        name="shot_interval",
    )

    sampling_period = _as_positive_float(
        exp.ctx.util.resolve_sampling_period(exp.ctx.measurement.sampling_period),
        name="sampling_period",
    )
    cr_parameters = _load_cr_parameters(exp, control_qubit, target_qubit)
    duration_resolution = _resolve_cr_half_duration(
        requested_duration=cr_half_duration,
        cr_amplitude=cr_parameters["cr_amplitude"],
        zx_rotation_rate=cr_parameters["zx_rotation_rate"],
        cr_ramptime=cr_parameters["cr_ramptime"],
        srre_ramp_time=srre_ramp_time,
        sampling_period=sampling_period,
    )
    amplitude_values = _resolve_cr_amplitude_range(
        cr_amplitude_range,
        center_amplitude=duration_resolution.predicted_cr_amplitude,
    )
    default_cancel_offsets: NDArray[np.float64] | None = None
    if cancel_y_offsets is None or cancel_x_offsets is None:
        default_cancel_offsets = _default_cancel_offsets(
            exp,
            target_qubit,
            cr_half_duration=duration_resolution.resolved_duration,
            cr_ramptime=cr_parameters["cr_ramptime"],
            error_amplification_n=error_amplification_n,
        )
    fine_offsets: dict[_Stage, NDArray[np.float64]] = {
        "zy": _resolve_offsets(
            cr_phase_offsets,
            default=_DEFAULT_PHASE_OFFSETS,
            name="cr_phase_offsets",
        ),
        "iy": _resolve_offsets(
            cancel_y_offsets,
            default=default_cancel_offsets,
            name="cancel_y_offsets",
        ),
        "ix": _resolve_offsets(
            cancel_x_offsets,
            default=default_cancel_offsets,
            name="cancel_x_offsets",
        ),
    }
    srre_data, srre_stage = _resolve_srre_calibration(
        exp,
        target_qubit,
        supplied=srre_calibration,
        block_duration=duration_resolution.resolved_duration,
        ramp_time=srre_ramp_time,
        sampling_period=sampling_period,
        amplitude_range=srre_amplitude_range,
        probe_detuning=probe_detuning,
        n_shots=resolved_shots,
        shot_interval=resolved_interval,
        plot=plot,
    )
    accepted = _initial_calibration(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        cr_parameters=cr_parameters,
        cr_half_duration=duration_resolution.resolved_duration,
        srre_calibration=srre_data,
        scale_cancellation_with_cr=scale_cancellation_with_cr,
    )

    initial_angle = _run_parameter_stage(
        exp,
        control_qubit,
        target_qubit,
        accepted=accepted,
        stage="zx",
        sweep_values=amplitude_values,
        root_reference=duration_resolution.predicted_cr_amplitude,
        error_amplification_n=error_amplification_n,
        scale_cancellation_with_cr=scale_cancellation_with_cr,
        n_shots=resolved_shots,
        verification_n_shots=resolved_verification_shots,
        shot_interval=resolved_interval,
        plot=plot,
        allow_range_expansion=True,
        fine_tune_cr=True,
    )
    accepted = initial_angle.accepted_calibration

    fine_rounds: list[dict[str, Any]] = []
    final_verification: dict[str, Any] = _verification_summary(
        initial_angle.data,
        None,
        None,
        None,
    )
    failed = initial_angle.data["status"] == "failed"
    converged = False

    if not failed:
        for round_index in range(max_fine_rounds):
            fine_round = _run_fine_round(
                exp,
                control_qubit,
                target_qubit,
                accepted=accepted,
                round_number=round_index + 1,
                cr_amplitude_range=cr_amplitude_range,
                fine_offsets=fine_offsets,
                error_amplification_n=error_amplification_n,
                scale_cancellation_with_cr=scale_cancellation_with_cr,
                n_shots=resolved_shots,
                verification_n_shots=resolved_verification_shots,
                shot_interval=resolved_interval,
                plot=plot,
            )
            accepted = fine_round.accepted_calibration
            fine_rounds.append(fine_round.data)
            failed = fine_round.failed
            converged = fine_round.converged
            if fine_round.final_verification is not None:
                final_verification = fine_round.final_verification
            if failed or converged:
                break

    if failed:
        status = "failed"
    elif converged:
        status = "converged"
    else:
        status = "max_fine_rounds_reached"

    result_data = _finalize_calibration_data(
        accepted,
        duration_resolution=duration_resolution,
        srre_stage=srre_stage,
        initial_angle_stage=initial_angle.data,
        fine_rounds=fine_rounds,
        final_verification=final_verification,
        converged=converged,
        status=status,
    )
    return Result(data={"srre_cr_calibration": result_data})


def _calculate_zx_signals(measurements: ArrayLike) -> _ZxSignals:
    """Calculate `S_ZX` and its `S_IX_from_Z` diagnostic."""
    values = _as_four_state_values(measurements)
    d0 = (values[0] - values[1]) / 2.0
    d1 = (values[2] - values[3]) / 2.0
    return _ZxSignals(zx=float((d0 + d1) / 2.0), ix_from_z=float((d0 - d1) / 2.0))


def _calculate_zy_signal(measurements: ArrayLike) -> float:
    """Calculate the stage-9a conditional-Y signal `S_ZY`."""
    values = _as_four_state_values(measurements)
    d0 = (values[0] - values[1]) / 2.0
    d1 = (values[2] - values[3]) / 2.0
    return float((d0 - d1) / 2.0)


def _calculate_iy_signal(measurements: ArrayLike) -> float:
    """Calculate the stage-9b unconditional-Y signal `S_IY`."""
    values = _as_four_state_values(measurements)
    d0 = (values[0] - values[1]) / 2.0
    d1 = (values[2] - values[3]) / 2.0
    return float((d0 + d1) / 2.0)


def _calculate_ix_signal(measurements: ArrayLike) -> float:
    """Calculate the stage-9c unconditional-X signal `S_IX`."""
    values = _as_four_state_values(measurements)
    d0 = (values[0] - values[1]) / 2.0
    d1 = (values[2] - values[3]) / 2.0
    return float((d0 + d1) / 2.0)


def _fit_zero_crossing(
    *,
    sweep_values: ArrayLike,
    error_signal: ArrayLike,
    allow_outside: bool = False,
) -> _ZeroCrossingAnalysis:
    """Fit one line to a calibration signal and return its zero crossing."""
    values = _as_finite_vector(sweep_values, name="sweep_values")
    signal = _as_finite_vector(error_signal, name="error_signal")
    if values.shape != signal.shape:
        raise ValueError("error_signal must have the same shape as sweep_values.")
    if values.size < 2:
        raise ValueError("sweep_values must contain at least two points.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("sweep_values must be strictly increasing.")

    centered_values = values - np.mean(values)
    denominator = float(np.dot(centered_values, centered_values))
    if denominator == 0.0:
        raise ValueError("Linear fit requires at least two distinct sweep values.")
    slope = float(np.dot(centered_values, signal - np.mean(signal)) / denominator)
    if not np.isfinite(slope) or abs(slope) < _MIN_ABSOLUTE_SLOPE:
        raise ValueError("Measured zero-crossing slope is too small.")
    intercept = float(np.mean(signal) - slope * np.mean(values))
    root = -intercept / slope
    lower = float(values[0])
    upper = float(values[-1])
    if not allow_outside and not lower <= root <= upper:
        raise ValueError("Fitted root lies outside the measured sweep range.")
    return _ZeroCrossingAnalysis(
        root=float(root),
        root_bracket=(lower, upper),
        fit_slope=float(slope),
        fit_intercept=intercept,
        fitted_signal=slope * values + intercept,
    )


def _resolve_cr_half_duration(
    *,
    requested_duration: float | None,
    cr_amplitude: float,
    zx_rotation_rate: float,
    cr_ramptime: float,
    srre_ramp_time: float,
    sampling_period: float,
) -> _DurationResolution:
    """Resolve a fixed, hardware-aligned CR-half and SRRE geometry."""
    amplitude = _as_positive_float(cr_amplitude, name="cr_amplitude")
    rate = _as_positive_float(zx_rotation_rate, name="zx_rotation_rate")
    ramptime = _as_nonnegative_float(cr_ramptime, name="cr_ramptime")
    ramp_time = _as_nonnegative_float(srre_ramp_time, name="srre_ramp_time")
    sampling_period = _as_positive_float(sampling_period, name="sampling_period")
    zx_frequency = amplitude * rate
    predicted_duration = 1.0 / (8.0 * zx_frequency) + ramptime

    duration_samples = _DURATION_UNIT / sampling_period
    if not np.isclose(
        duration_samples,
        round(duration_samples),
        rtol=0.0,
        atol=_NUMERIC_TOLERANCE,
    ):
        raise ValueError("The 16 ns duration unit must align to the sampling period.")

    if requested_duration is None:
        resolved_duration = (
            math.ceil((predicted_duration - _NUMERIC_TOLERANCE) / _DURATION_UNIT)
            * _DURATION_UNIT
        )
        requested = None
        source = "zx_rate_prediction"
    else:
        requested = _as_positive_float(requested_duration, name="cr_half_duration")
        units = requested / _DURATION_UNIT
        if not np.isclose(
            units,
            round(units),
            rtol=0.0,
            atol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError("cr_half_duration must align to the 16 ns duration unit.")
        resolved_duration = requested
        source = "explicit"

    sample_count = resolved_duration / sampling_period
    if not np.isclose(
        sample_count,
        round(sample_count),
        rtol=0.0,
        atol=_NUMERIC_TOLERANCE,
    ):
        raise ValueError("cr_half_duration must align to the sampling period.")
    if round(sample_count) % 2 != 0:
        raise ValueError("cr_half_duration must contain an even number of samples.")
    ramp_samples = ramp_time / sampling_period
    if not np.isclose(
        ramp_samples,
        round(ramp_samples),
        rtol=0.0,
        atol=_NUMERIC_TOLERANCE,
    ):
        raise ValueError("srre_ramp_time must align to the sampling period.")
    if 2.0 * ramptime > resolved_duration + _NUMERIC_TOLERANCE:
        raise ValueError("CR ramptime is too long for cr_half_duration.")

    lobe_duration = resolved_duration / 2.0
    if lobe_duration + _NUMERIC_TOLERANCE < 2.0 * ramp_time:
        raise ValueError("Each SRRE lobe must be at least twice the SRRE ramp time.")
    effective_duration = resolved_duration - ramptime
    if effective_duration <= 0.0:
        raise ValueError("CR flat-top area must be positive for ZX90 prediction.")
    predicted_cr_amplitude = 1.0 / (8.0 * rate * effective_duration)
    if predicted_cr_amplitude > 1.0 + _NUMERIC_TOLERANCE:
        raise ValueError(
            "The predicted ZX90 CR amplitude exceeds the hardware limit of 1."
        )
    return _DurationResolution(
        requested_duration=requested,
        predicted_duration=float(predicted_duration),
        predicted_cr_amplitude=float(predicted_cr_amplitude),
        resolved_duration=float(resolved_duration),
        duration_unit=_DURATION_UNIT,
        sampling_period=sampling_period,
        srre_lobe_duration=float(lobe_duration),
        srre_flat_time=float(lobe_duration - 2.0 * ramp_time),
        source=source,
    )


def _build_stage_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    calibration: Mapping[str, Any],
    stage: _Stage,
    control_state: str,
    error_amplification_n: int,
) -> PulseSchedule:
    """Build one four-state calibration sequence for a fixed candidate."""
    if stage not in _STAGE_CONFIGURATIONS:
        raise ValueError(f"Unsupported calibration stage {stage!r}.")
    if control_state not in ("0", "1"):
        raise ValueError("control_state must be '0' or '1'.")
    echo, include_srre = _STAGE_CONFIGURATIONS[stage]
    gate = _build_srre_cross_resonance(
        exp,
        control_qubit,
        target_qubit,
        _REFERENCE_ANGLE,
        calibration=calibration,
        echo=echo,
        include_srre=include_srre,
    )
    if stage == "zx":
        return gate

    repetitions = 2 * _as_positive_integer(
        error_amplification_n, name="error_amplification_n"
    )
    if stage in ("zy", "iy"):
        interleaved_pulse = exp.pulse.y180(target_qubit)
    else:
        interleaved_pulse = exp.pulse.x90(target_qubit)
        if control_state == "1":
            interleaved_pulse = interleaved_pulse.scaled(-1.0)
    if not isinstance(interleaved_pulse, Waveform):
        raise TypeError("The fine-stage reference pulse must be a Waveform.")
    srre_metadata = _required(
        calibration,
        "srre_calibration",
        context="SRRE-CR calibration",
    )
    if not isinstance(srre_metadata, Mapping):
        raise TypeError("calibration['srre_calibration'] must be a mapping.")
    sampling_period = _as_positive_float(
        _required(
            srre_metadata,
            "sampling_period",
            context="SRRE calibration",
        ),
        name="srre_calibration.sampling_period",
    )
    _validate_reference_pulse_sampling(
        interleaved_pulse,
        name="fine-stage reference pulse",
        sampling_period=sampling_period,
    )

    with PulseSchedule() as cycle:
        cycle.call(gate)
        cycle.add(target_qubit, interleaved_pulse)
        idle = Arbitrary(
            np.zeros(interleaved_pulse.length, dtype=np.complex128),
            sampling_period=sampling_period,
        )
        for label in gate.labels:
            if label != target_qubit:
                cycle.add(label, idle)
    return cycle.repeated(repetitions)


def _validate_reference_pulse_sampling(
    pulse: Waveform,
    *,
    name: str,
    sampling_period: float,
) -> None:
    waveforms = (
        pulse.get_flattened_waveforms(apply_frame_shifts=False)
        if isinstance(pulse, PulseArray)
        else [pulse]
    )
    if not waveforms:
        raise ValueError(f"{name} must contain at least one waveform.")
    for waveform in waveforms:
        if not np.isclose(
            waveform.sampling_period,
            sampling_period,
            rtol=0.0,
            atol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError(
                f"{name} sampling period must match the SRRE sampling period."
            )


def _measure_stage(
    *,
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    calibration: Mapping[str, Any],
    stage: _Stage,
    sweep_values: ArrayLike,
    error_amplification_n: int,
    scale_cancellation_with_cr: bool,
    n_shots: int,
    shot_interval: float,
) -> _StageMeasurement:
    """Measure the four state series for one parameter-specific stage."""
    values = _as_finite_vector(sweep_values, name="sweep_values")
    if values.size < 1:
        raise ValueError("sweep_values must contain at least one point.")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("sweep_values must be strictly increasing.")
    state_pairs = _stage_state_pairs(stage)
    state_values = np.empty((4, values.size), dtype=np.float64)
    raw_results: list[Any] = []

    for state_index, (control_state, target_state) in enumerate(state_pairs):

        def sequence(
            candidate: float,
            control_state: str = control_state,
        ) -> PulseSchedule:
            candidate_calibration = _apply_stage_candidate(
                calibration,
                stage=stage,
                candidate=float(candidate),
                scale_cancellation_with_cr=scale_cancellation_with_cr,
            )
            return _build_stage_sequence(
                exp,
                control_qubit,
                target_qubit,
                calibration=candidate_calibration,
                stage=stage,
                control_state=control_state,
                error_amplification_n=error_amplification_n,
            )

        result = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=values,
            initial_states={
                control_qubit: control_state,
                target_qubit: target_state,
            },
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            title=f"SRRE-CR {stage.upper()} calibration: {control_qubit}-{target_qubit}",
            xlabel=_stage_parameter_name(stage),
            ylabel="Normalized target Z",
        )
        raw_results.append(result)
        if target_qubit not in result.data:
            raise ValueError(
                f"Calibration measurement is missing target `{target_qubit}`."
            )
        normalized = _as_finite_vector(
            result.data[target_qubit].normalized,
            name="normalized four-state measurement",
        )
        if normalized.shape != values.shape:
            raise ValueError(
                "Each four-state series must return one scalar per sweep point; "
                f"expected {values.shape}, got {normalized.shape}."
            )
        state_values[state_index] = normalized

    error_signal = np.empty(values.size, dtype=np.float64)
    diagnostics: dict[str, NDArray[np.float64]] = {}
    if stage == "zx":
        ix_from_z = np.empty(values.size, dtype=np.float64)
        for index in range(values.size):
            signals = _calculate_zx_signals(state_values[:, index])
            error_signal[index] = signals.zx
            ix_from_z[index] = signals.ix_from_z
        diagnostics["ix_from_z"] = ix_from_z
    else:
        calculator = {
            "zy": _calculate_zy_signal,
            "iy": _calculate_iy_signal,
            "ix": _calculate_ix_signal,
        }[stage]
        for index in range(values.size):
            error_signal[index] = calculator(state_values[:, index])

    return _StageMeasurement(
        sweep_values=values.copy(),
        state_values=state_values,
        error_signal=error_signal,
        diagnostic_signals=diagnostics,
        raw_results=tuple(raw_results),
    )


def _measure_fitted_stage(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    calibration: Mapping[str, Any],
    stage: _Stage,
    sweep_values: ArrayLike,
    root_reference: float,
    error_amplification_n: int,
    scale_cancellation_with_cr: bool,
    n_shots: int,
    shot_interval: float,
    plot: bool,
    allow_range_expansion: bool,
) -> tuple[_StageMeasurement, _ZeroCrossingAnalysis, list[dict[str, Any]]]:
    """Measure and fit a stage, extending once toward an out-of-range root."""
    values = _as_finite_vector(sweep_values, name="sweep_values")
    fit_history: list[dict[str, Any]] = []
    for attempt in range(2):
        measurement = _measure_stage(
            exp=exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            calibration=calibration,
            stage=stage,
            sweep_values=values,
            error_amplification_n=error_amplification_n,
            scale_cancellation_with_cr=scale_cancellation_with_cr,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        analysis = _fit_zero_crossing(
            sweep_values=measurement.sweep_values,
            error_signal=measurement.error_signal,
            allow_outside=True,
        )
        fit_history.append(_fit_measurement_data(measurement, analysis))
        if plot:
            _plot_stage_fit(measurement, analysis, stage=stage)
        lower, upper = analysis.root_bracket
        if lower <= analysis.root <= upper:
            return measurement, analysis, fit_history
        if not allow_range_expansion or attempt == 1:
            return measurement, analysis, fit_history
        values = _expand_sweep_toward_root(
            values,
            center=root_reference,
            root=analysis.root,
            bounds=(0.0, 1.0) if stage == "zx" else None,
        )
    raise RuntimeError("Unreachable fitted-stage state.")


def _run_parameter_stage(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    accepted: Mapping[str, Any],
    stage: _Stage,
    sweep_values: ArrayLike,
    root_reference: float,
    error_amplification_n: int,
    scale_cancellation_with_cr: bool,
    n_shots: int,
    verification_n_shots: int,
    shot_interval: float,
    plot: bool,
    allow_range_expansion: bool = False,
    fine_tune_cr: bool = False,
) -> _StageRun:
    """Sweep, propose, freshly verify, and conditionally accept one parameter."""
    if fine_tune_cr and stage != "zx":
        raise ValueError("fine_tune_cr is supported only for the ZX stage.")
    current = _copy_calibration(accepted)
    current_parameter = _stage_parameter_value(current, stage=stage)
    input_params = _parameter_snapshot(current)
    configuration = _configuration_data(stage)
    measurement: _StageMeasurement | None = None
    analysis: _ZeroCrossingAnalysis | None = None
    fit_history: list[dict[str, Any]] = []
    try:
        measurement, analysis, fit_history = _measure_fitted_stage(
            exp,
            control_qubit,
            target_qubit,
            calibration=current,
            stage=stage,
            sweep_values=sweep_values,
            root_reference=root_reference,
            error_amplification_n=error_amplification_n,
            scale_cancellation_with_cr=scale_cancellation_with_cr,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            allow_range_expansion=allow_range_expansion,
        )
        _require_root_in_measured_range(analysis)
        if fine_tune_cr:
            measurement, analysis, fine_history = _measure_fitted_stage(
                exp,
                control_qubit,
                target_qubit,
                calibration=current,
                stage=stage,
                sweep_values=_fine_cr_amplitude_range(analysis.root),
                root_reference=analysis.root,
                error_amplification_n=error_amplification_n,
                scale_cancellation_with_cr=scale_cancellation_with_cr,
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=plot,
                allow_range_expansion=allow_range_expansion,
            )
            fit_history.extend(fine_history)
            _require_root_in_measured_range(analysis)
        proposed = _apply_stage_candidate(
            current,
            stage=stage,
            candidate=_validated_stage_root(analysis.root, stage=stage),
            scale_cancellation_with_cr=scale_cancellation_with_cr,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return _StageRun(
            data=_failed_stage_data(
                input_params=input_params,
                configuration=configuration,
                measurement=measurement,
                analysis=analysis,
                reason=str(exc),
                fit_history=fit_history,
            ),
            accepted_calibration=current,
        )

    verification_values = _verification_sweep_values(
        current_parameter=current_parameter,
        candidate_parameter=analysis.root,
    )
    verification: _StageMeasurement | None = None
    try:
        verification = _measure_stage(
            exp=exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            calibration=current,
            stage=stage,
            sweep_values=verification_values,
            error_amplification_n=error_amplification_n,
            scale_cancellation_with_cr=scale_cancellation_with_cr,
            n_shots=verification_n_shots,
            shot_interval=shot_interval,
        )
        candidate_index = _matching_index(
            verification.sweep_values,
            analysis.root,
            name="candidate root",
        )
        candidate_signal = float(verification.error_signal[candidate_index])
        if verification.sweep_values.size == 1:
            current_signal = candidate_signal
        else:
            current_index = _matching_index(
                verification.sweep_values,
                current_parameter,
                name="current parameter",
            )
            current_signal = float(verification.error_signal[current_index])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return _failed_verification_run(
            current=current,
            proposed=proposed,
            measurement=measurement,
            analysis=analysis,
            configuration=configuration,
            reason=str(exc),
            current_parameter=current_parameter,
            verification=verification,
            fit_history=fit_history,
        )

    same_parameter = np.isclose(
        analysis.root,
        current_parameter,
        rtol=0.0,
        atol=_NUMERIC_TOLERANCE,
    )
    improved = abs(candidate_signal) + _NUMERIC_TOLERANCE < abs(current_signal)
    verified = improved or (
        same_parameter and abs(candidate_signal) <= _SIGNAL_TOLERANCE
    )
    if not verified:
        return _failed_verification_run(
            current=current,
            proposed=proposed,
            measurement=measurement,
            analysis=analysis,
            configuration=configuration,
            reason="Fresh verification did not improve the calibration signal.",
            verification=verification,
            current_parameter=current_parameter,
            current_signal=current_signal,
            candidate_signal=candidate_signal,
            fit_history=fit_history,
        )

    converged = abs(candidate_signal) <= _SIGNAL_TOLERANCE
    verification_data = {
        **_stage_measurement_data(verification),
        "current_parameter": current_parameter,
        "candidate_parameter": analysis.root,
        "current_signal": current_signal,
        "candidate_signal": candidate_signal,
        "improved": improved,
        "status": "accepted",
        "reason": None,
    }
    return _StageRun(
        data={
            **_stage_measurement_data(measurement),
            "input_params": input_params,
            "proposed_params": _parameter_snapshot(proposed),
            "accepted_params": _parameter_snapshot(proposed),
            "root": analysis.root,
            "root_bracket": analysis.root_bracket,
            "fit_slope": analysis.fit_slope,
            "fit_intercept": analysis.fit_intercept,
            "fitted_signal": analysis.fitted_signal.copy(),
            "fit_history": fit_history,
            "fit_quality": _fit_quality(analysis),
            "verification": verification_data,
            "configuration": configuration,
            "converged": converged,
            "status": "accepted",
            "reason": None,
        },
        accepted_calibration=proposed,
    )


def _run_fine_round(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    accepted: Mapping[str, Any],
    round_number: int,
    cr_amplitude_range: ArrayLike | None,
    fine_offsets: Mapping[_Stage, NDArray[np.float64]],
    error_amplification_n: int,
    scale_cancellation_with_cr: bool,
    n_shots: int,
    verification_n_shots: int,
    shot_interval: float,
    plot: bool,
) -> _FineRoundRun:
    """Run one phase, cancellation-IQ, and final-angle calibration round."""
    current = _copy_calibration(accepted)
    round_data: dict[str, Any] = {"round": round_number}

    for data_key, stage in _FINE_STAGE_DATA_KEYS:
        current_parameter = _stage_parameter_value(current, stage=stage)
        stage_run = _run_parameter_stage(
            exp,
            control_qubit,
            target_qubit,
            accepted=current,
            stage=stage,
            sweep_values=current_parameter + fine_offsets[stage],
            root_reference=current_parameter,
            error_amplification_n=error_amplification_n,
            scale_cancellation_with_cr=scale_cancellation_with_cr,
            n_shots=n_shots,
            verification_n_shots=verification_n_shots,
            shot_interval=shot_interval,
            plot=plot,
            allow_range_expansion=stage in ("iy", "ix"),
        )
        round_data[data_key] = stage_run.data
        current = stage_run.accepted_calibration
        if stage_run.data["status"] == "failed":
            _fill_skipped_stages(round_data, after=data_key)
            round_data["converged"] = False
            return _FineRoundRun(
                data=round_data,
                accepted_calibration=current,
                final_verification=None,
                failed=True,
                converged=False,
            )

    current_amplitude = _stage_parameter_value(current, stage="zx")
    final_angle = _run_parameter_stage(
        exp,
        control_qubit,
        target_qubit,
        accepted=current,
        stage="zx",
        sweep_values=_resolve_cr_amplitude_range(
            cr_amplitude_range,
            center_amplitude=current_amplitude,
        ),
        root_reference=current_amplitude,
        error_amplification_n=error_amplification_n,
        scale_cancellation_with_cr=scale_cancellation_with_cr,
        n_shots=n_shots,
        verification_n_shots=verification_n_shots,
        shot_interval=shot_interval,
        plot=plot,
        allow_range_expansion=True,
        fine_tune_cr=True,
    )
    round_data["final_angle_stage"] = final_angle.data
    current = final_angle.accepted_calibration
    if final_angle.data["status"] == "failed":
        round_data["converged"] = False
        return _FineRoundRun(
            data=round_data,
            accepted_calibration=current,
            final_verification=None,
            failed=True,
            converged=False,
        )

    verification = _verification_summary(
        final_angle.data,
        round_data["phase_stage"],
        round_data["cancel_y_stage"],
        round_data["cancel_x_stage"],
    )
    converged = bool(verification["converged"])
    round_data["verification"] = verification
    round_data["converged"] = converged
    return _FineRoundRun(
        data=round_data,
        accepted_calibration=current,
        final_verification=verification,
        failed=False,
        converged=converged,
    )


def _resolve_srre_calibration(
    exp: Experiment,
    target_qubit: str,
    *,
    supplied: Mapping[str, Any] | None,
    block_duration: float,
    ramp_time: float,
    sampling_period: float,
    amplitude_range: ArrayLike | None,
    probe_detuning: float | None,
    n_shots: int,
    shot_interval: float,
    plot: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if supplied is None:
        result = calibrate_srre(
            exp,
            target_qubit,
            block_duration=block_duration,
            ramp_time=ramp_time,
            amplitude_range=amplitude_range,
            probe_detuning=probe_detuning,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
        )
        value = result.data.get("srre_calibration")
        status = "calibrated"
        raw_result: Result | None = result
    else:
        if not isinstance(supplied, Mapping):
            raise TypeError("srre_calibration must be a mapping or None.")
        if "srre_calibration" in supplied:
            nested = supplied["srre_calibration"]
            if not isinstance(nested, Mapping):
                raise TypeError(
                    "srre_calibration['srre_calibration'] must be a mapping."
                )
            value = nested
        else:
            value = supplied
        status = "reused"
        raw_result = None
    if not isinstance(value, Mapping):
        raise TypeError("SRRE calibration result is missing `srre_calibration` data.")
    calibration = deepcopy(dict(value))
    _validate_srre_metadata(
        calibration,
        target_qubit=target_qubit,
        block_duration=block_duration,
        ramp_time=ramp_time,
        sampling_period=sampling_period,
    )
    return calibration, {
        "status": status,
        "reason": None,
        "calibration": deepcopy(calibration),
        "raw_result": raw_result,
    }


def _validate_srre_metadata(
    calibration: Mapping[str, Any],
    *,
    target_qubit: str,
    block_duration: float,
    ramp_time: float,
    sampling_period: float,
) -> None:
    target = _required(calibration, "target", context="SRRE calibration")
    if target != target_qubit:
        raise ValueError("SRRE calibration target does not match target_qubit.")
    comparisons = {
        "block_duration": block_duration,
        "ramp_time": ramp_time,
        "sampling_period": sampling_period,
    }
    for field, expected in comparisons.items():
        actual = _as_finite_float(
            _required(calibration, field, context="SRRE calibration"),
            name=f"srre_calibration.{field}",
        )
        if not np.isclose(
            actual,
            expected,
            rtol=0.0,
            atol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError(
                f"SRRE calibration {field} does not match the fixed CR geometry."
            )
    amplitude = _as_positive_float(
        _required(calibration, "amplitude", context="SRRE calibration"),
        name="srre_calibration.amplitude",
    )
    if abs(amplitude) > 1.0 + _NUMERIC_TOLERANCE:
        raise ValueError("SRRE calibration amplitude exceeds the hardware limit.")


def _load_cr_parameters(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
) -> dict[str, float]:
    cr_label = f"{control_qubit}-{target_qubit}"
    stored = exp.ctx.calib_note.get_cr_param(cr_label)
    if stored is None:
        raise ValueError(
            f"CR parameters for {cr_label} are not stored; run obtain_cr_params first."
        )
    if not isinstance(stored, Mapping):
        raise TypeError("Stored CR parameters must be a mapping.")
    cancel_amplitude = _as_nonnegative_float(
        _required(stored, "cancel_amplitude", context="stored CR parameters"),
        name="cancel_amplitude",
    )
    cancel_phase = _as_finite_float(
        _required(stored, "cancel_phase", context="stored CR parameters"),
        name="cancel_phase",
    )
    cancel = cancel_amplitude * np.exp(1j * cancel_phase)
    return {
        "cr_ramptime": _as_nonnegative_float(
            _required(stored, "ramptime", context="stored CR parameters"),
            name="cr_ramptime",
        ),
        "cr_amplitude": _as_positive_float(
            _required(stored, "cr_amplitude", context="stored CR parameters"),
            name="cr_amplitude",
        ),
        "cr_phase": _as_finite_float(
            _required(stored, "cr_phase", context="stored CR parameters"),
            name="cr_phase",
        ),
        "cr_beta": _as_finite_float(
            _required(stored, "cr_beta", context="stored CR parameters"),
            name="cr_beta",
        ),
        "cancel_x": float(cancel.real),
        "cancel_y": float(cancel.imag),
        "cancel_beta": _as_finite_float(
            _required(stored, "cancel_beta", context="stored CR parameters"),
            name="cancel_beta",
        ),
        "zx_rotation_rate": _as_finite_float(
            _required(stored, "zx_rotation_rate", context="stored CR parameters"),
            name="zx_rotation_rate",
        ),
    }


def _initial_calibration(
    *,
    control_qubit: str,
    target_qubit: str,
    cr_parameters: Mapping[str, float],
    cr_half_duration: float,
    srre_calibration: Mapping[str, Any],
    scale_cancellation_with_cr: bool,
) -> dict[str, Any]:
    return {
        "control_qubit": control_qubit,
        "target_qubit": target_qubit,
        "cr_half_duration": float(cr_half_duration),
        "cr_ramptime": cr_parameters["cr_ramptime"],
        "cr_amplitude": cr_parameters["cr_amplitude"],
        "cr_phase": cr_parameters["cr_phase"],
        "cr_beta": cr_parameters["cr_beta"],
        "cancel_x": cr_parameters["cancel_x"],
        "cancel_y": cr_parameters["cancel_y"],
        "cancel_beta": cr_parameters["cancel_beta"],
        "zx_rotation_rate": cr_parameters["zx_rotation_rate"],
        "srre_calibration": deepcopy(dict(srre_calibration)),
        "scale_cancellation_with_cr": scale_cancellation_with_cr,
    }


def _apply_stage_candidate(
    calibration: Mapping[str, Any],
    *,
    stage: _Stage,
    candidate: float,
    scale_cancellation_with_cr: bool,
) -> dict[str, Any]:
    result = _copy_calibration(calibration)
    value = _as_finite_float(candidate, name="candidate")
    if stage == "zx":
        current_amplitude = _as_positive_float(
            result["cr_amplitude"], name="current cr_amplitude"
        )
        if value <= 0.0:
            raise ValueError("CR amplitude candidates must be positive.")
        if scale_cancellation_with_cr:
            scale = value / current_amplitude
            result["cancel_x"] = float(result["cancel_x"] * scale)
            result["cancel_y"] = float(result["cancel_y"] * scale)
        result["cr_amplitude"] = value
    elif stage == "zy":
        result["cr_phase"] = value
    elif stage == "iy":
        result["cancel_y"] = value
    elif stage == "ix":
        result["cancel_x"] = value
    else:
        raise ValueError(f"Unsupported calibration stage {stage!r}.")
    return result


def _validated_stage_root(root: float, *, stage: _Stage) -> float:
    if stage == "zx" and root <= 0.0:
        raise ValueError("The measured ZX90 CR amplitude root must be positive.")
    return root


def _require_root_in_measured_range(analysis: _ZeroCrossingAnalysis) -> None:
    lower, upper = analysis.root_bracket
    if not lower <= analysis.root <= upper:
        raise ValueError("Fitted root lies outside the measured sweep range.")


def _stage_parameter_value(
    calibration: Mapping[str, Any],
    *,
    stage: _Stage,
) -> float:
    return _as_finite_float(
        calibration[_STAGE_PARAMETER_KEYS[stage]],
        name=f"current {_STAGE_PARAMETER_KEYS[stage]}",
    )


def _resolve_cr_amplitude_range(
    values: ArrayLike | None,
    *,
    center_amplitude: float,
) -> NDArray[np.float64]:
    center = _as_positive_float(center_amplitude, name="center cr_amplitude")
    if values is None:
        result = np.unique(np.clip(center * _DEFAULT_CR_SWEEP_FACTORS, 0.0, 1.0))
    else:
        result = _as_finite_vector(values, name="cr_amplitude_range")
    if result.size < 2:
        raise ValueError("cr_amplitude_range must contain at least two points.")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("cr_amplitude_range must be strictly increasing.")
    if result[0] <= 0.0 or result[-1] > 1.0 + _NUMERIC_TOLERANCE:
        raise ValueError("cr_amplitude_range values must be positive and not exceed 1.")
    return result.copy()


def _fine_cr_amplitude_range(center_amplitude: float) -> NDArray[np.float64]:
    """Return the default 17-point ±8% fine CR amplitude sweep."""
    center = _as_positive_float(center_amplitude, name="fine center cr_amplitude")
    values = np.unique(np.clip(center * _FINE_CR_SWEEP_FACTORS, 0.0, 1.0))
    if values.size < 2:
        raise ValueError("Fine CR amplitude range must contain at least two points.")
    return values


def _expand_sweep_toward_root(
    values: NDArray[np.float64],
    *,
    center: float,
    root: float,
    bounds: tuple[float, float] | None,
) -> NDArray[np.float64]:
    """Extend one side of a sweep to twice its original distance from center."""
    center = _as_finite_float(center, name="sweep center")
    root = _as_finite_float(root, name="fitted root")
    if values.size < 2:
        raise ValueError("A sweep must contain at least two points before expansion.")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("A sweep must be strictly increasing before expansion.")
    step = float(np.min(differences))
    lower = float(values[0])
    upper = float(values[-1])
    if root < lower:
        span = center - lower
        if span <= 0.0:
            raise ValueError(
                "Cannot extend a sweep whose center is not above its lower end."
            )
        limit = center - _MAX_SWEEP_EXPANSION_FACTOR * span
        if bounds is not None:
            limit = max(limit, bounds[0])
        extension = lower - limit
        count = int(np.ceil(extension / step - _NUMERIC_TOLERANCE))
        extra = np.linspace(limit, lower, count + 1, dtype=np.float64)[:-1]
        expanded = np.concatenate((extra, values))
    elif root > upper:
        span = upper - center
        if span <= 0.0:
            raise ValueError(
                "Cannot extend a sweep whose center is not below its upper end."
            )
        limit = center + _MAX_SWEEP_EXPANSION_FACTOR * span
        if bounds is not None:
            limit = min(limit, bounds[1])
        extension = limit - upper
        count = int(np.ceil(extension / step - _NUMERIC_TOLERANCE))
        extra = np.linspace(upper, limit, count + 1, dtype=np.float64)[1:]
        expanded = np.concatenate((values, extra))
    else:
        return values.copy()
    expanded = np.unique(expanded)
    if expanded.size == values.size:
        raise ValueError("The fitted root is outside the maximum sweep range.")
    return expanded


def _default_cancel_offsets(
    exp: Experiment,
    target_qubit: str,
    *,
    cr_half_duration: float,
    cr_ramptime: float,
    error_amplification_n: int,
) -> NDArray[np.float64]:
    """Resolve cancellation offsets giving 0.2/N radians per gate at each edge."""
    repetitions = _as_positive_integer(
        error_amplification_n, name="error_amplification_n"
    )
    effective_duration = _as_positive_float(
        cr_half_duration - cr_ramptime,
        name="cancellation effective duration",
    )
    exp.pulse.validate_rabi_params([target_qubit])
    edge_rabi_rate = _DEFAULT_CANCEL_EDGE_ROTATION / (
        2.0 * np.pi * repetitions * effective_duration
    )
    edge_offset = abs(
        _as_finite_float(
            exp.pulse.calc_control_amplitude(target_qubit, edge_rabi_rate),
            name="cancellation edge offset",
        )
    )
    if edge_offset == 0.0:
        raise ValueError(
            "The target Rabi parameters produced a zero cancellation range."
        )
    return np.linspace(
        -edge_offset,
        edge_offset,
        _DEFAULT_CANCEL_SWEEP_POINTS,
    )


def _resolve_offsets(
    values: ArrayLike | None,
    *,
    default: NDArray[np.float64] | None,
    name: str,
) -> NDArray[np.float64]:
    if values is None:
        if default is None:
            raise RuntimeError(f"Default {name} could not be resolved.")
        result = default.copy()
    else:
        result = _as_finite_vector(values, name=name)
    if result.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return result


def _stage_state_pairs(stage: _Stage) -> tuple[tuple[str, str], ...]:
    if stage == "zx":
        target_states = ("0", "1")
    elif stage in ("zy", "iy"):
        target_states = ("+", "-")
    elif stage == "ix":
        target_states = ("+i", "-i")
    else:
        raise ValueError(f"Unsupported calibration stage {stage!r}.")
    return (
        ("0", target_states[0]),
        ("0", target_states[1]),
        ("1", target_states[0]),
        ("1", target_states[1]),
    )


def _stage_parameter_name(stage: _Stage) -> str:
    return {
        "zx": "CR amplitude",
        "zy": "CR phase (rad)",
        "iy": "Cancellation Y",
        "ix": "Cancellation X",
    }[stage]


def _configuration_data(stage: _Stage) -> dict[str, bool]:
    echo, include_srre = _STAGE_CONFIGURATIONS[stage]
    return {"echo": echo, "include_srre": include_srre}


def _stage_measurement_data(measurement: _StageMeasurement) -> dict[str, Any]:
    return {
        "sweep_values": measurement.sweep_values.copy(),
        "state_values": measurement.state_values.copy(),
        "error_signal": measurement.error_signal.copy(),
        "diagnostic_signals": {
            key: value.copy() for key, value in measurement.diagnostic_signals.items()
        },
        "raw_results": measurement.raw_results,
    }


def _fit_measurement_data(
    measurement: _StageMeasurement,
    analysis: _ZeroCrossingAnalysis,
) -> dict[str, Any]:
    """Return one measured sweep and its linear-fit metadata."""
    return {
        **_stage_measurement_data(measurement),
        "root": analysis.root,
        "root_bracket": analysis.root_bracket,
        "fit_slope": analysis.fit_slope,
        "fit_intercept": analysis.fit_intercept,
        "fitted_signal": analysis.fitted_signal.copy(),
        "fit_quality": _fit_quality(analysis),
    }


def _plot_stage_fit(
    measurement: _StageMeasurement,
    analysis: _ZeroCrossingAnalysis,
    *,
    stage: _Stage,
) -> None:
    """Show one CR calibration signal together with its fitted line."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=measurement.sweep_values,
            y=measurement.error_signal,
            mode="markers",
            name="Measurement",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=measurement.sweep_values,
            y=analysis.fitted_signal,
            mode="lines",
            name="Linear fit",
        )
    )
    figure.update_layout(
        title=f"SRRE-CR {stage.upper()} calibration",
        xaxis_title=_stage_parameter_name(stage),
        yaxis_title=f"S_{stage.upper()}",
    )
    figure.show()


def _empty_measurement_data() -> dict[str, Any]:
    return {
        "sweep_values": np.array([], dtype=np.float64),
        "state_values": np.empty((4, 0), dtype=np.float64),
        "error_signal": np.array([], dtype=np.float64),
        "diagnostic_signals": {},
        "raw_results": (),
    }


def _failed_stage_data(
    *,
    input_params: Mapping[str, Any],
    configuration: Mapping[str, bool],
    measurement: _StageMeasurement | None,
    analysis: _ZeroCrossingAnalysis | None,
    reason: str,
    fit_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data = (
        _empty_measurement_data()
        if measurement is None
        else _stage_measurement_data(measurement)
    )
    return {
        **data,
        "input_params": dict(input_params),
        "proposed_params": None,
        "accepted_params": dict(input_params),
        "root": None if analysis is None else analysis.root,
        "root_bracket": None if analysis is None else analysis.root_bracket,
        "fit_slope": None if analysis is None else analysis.fit_slope,
        "fit_intercept": None if analysis is None else analysis.fit_intercept,
        "fitted_signal": (None if analysis is None else analysis.fitted_signal.copy()),
        "fit_history": [] if fit_history is None else fit_history,
        "fit_quality": None if analysis is None else _fit_quality(analysis),
        "verification": None,
        "configuration": dict(configuration),
        "converged": False,
        "status": "failed",
        "reason": reason,
    }


def _failed_verification_run(
    *,
    current: Mapping[str, Any],
    proposed: Mapping[str, Any],
    measurement: _StageMeasurement,
    analysis: _ZeroCrossingAnalysis,
    configuration: Mapping[str, bool],
    reason: str,
    current_parameter: float,
    verification: _StageMeasurement | None = None,
    current_signal: float | None = None,
    candidate_signal: float | None = None,
    fit_history: list[dict[str, Any]] | None = None,
) -> _StageRun:
    verification_data: dict[str, Any] = (
        {"status": "failed", "reason": reason}
        if verification is None
        else {
            **_stage_measurement_data(verification),
            "status": "failed",
            "reason": reason,
        }
    )
    verification_data.update(
        {
            "current_parameter": current_parameter,
            "candidate_parameter": analysis.root,
            "current_signal": current_signal,
            "candidate_signal": candidate_signal,
            "improved": False,
        }
    )
    return _StageRun(
        data={
            **_stage_measurement_data(measurement),
            "input_params": _parameter_snapshot(current),
            "proposed_params": _parameter_snapshot(proposed),
            "accepted_params": _parameter_snapshot(current),
            "root": analysis.root,
            "root_bracket": analysis.root_bracket,
            "fit_slope": analysis.fit_slope,
            "fit_intercept": analysis.fit_intercept,
            "fitted_signal": analysis.fitted_signal.copy(),
            "fit_history": [] if fit_history is None else fit_history,
            "fit_quality": _fit_quality(analysis),
            "verification": verification_data,
            "configuration": dict(configuration),
            "converged": False,
            "status": "failed",
            "reason": reason,
        },
        accepted_calibration=_copy_calibration(current),
    )


def _fit_quality(analysis: _ZeroCrossingAnalysis) -> dict[str, float]:
    return {
        "absolute_slope": abs(analysis.fit_slope),
        "bracket_width": analysis.root_bracket[1] - analysis.root_bracket[0],
    }


def _matching_index(values: NDArray[np.float64], target: float, *, name: str) -> int:
    indices = np.flatnonzero(
        np.isclose(values, target, rtol=0.0, atol=_NUMERIC_TOLERANCE)
    )
    if indices.size != 1:
        raise ValueError(f"Fresh verification is missing the {name} point.")
    return int(indices[0])


def _verification_sweep_values(
    *,
    current_parameter: float,
    candidate_parameter: float,
) -> NDArray[np.float64]:
    if np.isclose(
        current_parameter,
        candidate_parameter,
        rtol=0.0,
        atol=_NUMERIC_TOLERANCE,
    ):
        return np.asarray([current_parameter], dtype=np.float64)
    return np.sort(
        np.asarray([current_parameter, candidate_parameter], dtype=np.float64)
    )


def _parameter_snapshot(calibration: Mapping[str, Any]) -> dict[str, float]:
    return {
        "cr_amplitude": float(calibration["cr_amplitude"]),
        "cr_phase": float(calibration["cr_phase"]),
        "cancel_x": float(calibration["cancel_x"]),
        "cancel_y": float(calibration["cancel_y"]),
    }


def _copy_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(calibration))


def _verification_summary(
    angle_stage: Mapping[str, Any],
    phase_stage: Mapping[str, Any] | None,
    cancel_y_stage: Mapping[str, Any] | None,
    cancel_x_stage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stage_map = {
        "s_zx": angle_stage,
        "s_zy": phase_stage,
        "s_iy": cancel_y_stage,
        "s_ix": cancel_x_stage,
    }
    signals: dict[str, float | None] = {}
    checks: dict[str, bool] = {}
    for name, stage in stage_map.items():
        signal = _accepted_verification_signal(stage)
        signals[name] = signal
        checks[name] = signal is not None and abs(signal) <= _SIGNAL_TOLERANCE
    diagnostics = angle_stage.get("verification")
    ix_from_z: float | None = None
    ix_from_z_current: float | None = None
    ix_from_z_not_worse = False
    if isinstance(diagnostics, Mapping):
        diagnostic_signals = diagnostics.get("diagnostic_signals")
        candidate = diagnostics.get("candidate_parameter")
        current = diagnostics.get("current_parameter")
        sweep_values = diagnostics.get("sweep_values")
        if (
            isinstance(diagnostic_signals, Mapping)
            and "ix_from_z" in diagnostic_signals
            and candidate is not None
            and current is not None
            and sweep_values is not None
        ):
            values = np.asarray(sweep_values, dtype=np.float64)
            candidate_index = _matching_index(
                values, float(candidate), name="candidate root"
            )
            current_index = _matching_index(
                values, float(current), name="current parameter"
            )
            ix_diagnostic = np.asarray(diagnostic_signals["ix_from_z"])
            ix_from_z = float(ix_diagnostic[candidate_index])
            ix_from_z_current = float(ix_diagnostic[current_index])
            ix_from_z_not_worse = (
                abs(ix_from_z)
                <= abs(ix_from_z_current) + _SIGNAL_TOLERANCE + _NUMERIC_TOLERANCE
            )
    signals["s_ix_from_z"] = ix_from_z
    signals["s_ix_from_z_current"] = ix_from_z_current
    checks["s_ix_from_z_not_worse"] = ix_from_z_not_worse
    complete = all(stage is not None for stage in stage_map.values())
    return {
        "signals": signals,
        "checks": checks,
        "signal_tolerance": _SIGNAL_TOLERANCE,
        "converged": complete and all(checks.values()),
    }


def _accepted_verification_signal(stage: Mapping[str, Any] | None) -> float | None:
    if stage is None:
        return None
    verification = stage.get("verification")
    if not isinstance(verification, Mapping):
        return None
    value = verification.get("candidate_signal")
    if value is None:
        return None
    return float(value)


def _fill_skipped_stages(round_data: dict[str, Any], *, after: str) -> None:
    order = ["phase_stage", "cancel_y_stage", "cancel_x_stage", "final_angle_stage"]
    start = order.index(after) + 1
    for name in order[start:]:
        round_data[name] = {
            **_empty_measurement_data(),
            "input_params": None,
            "proposed_params": None,
            "accepted_params": None,
            "root": None,
            "root_bracket": None,
            "fit_slope": None,
            "fit_intercept": None,
            "fitted_signal": None,
            "fit_history": [],
            "fit_quality": None,
            "verification": None,
            "configuration": None,
            "converged": False,
            "status": "skipped",
            "reason": f"Skipped after {after} failed.",
        }


def _finalize_calibration_data(
    accepted: Mapping[str, Any],
    *,
    duration_resolution: _DurationResolution,
    srre_stage: Mapping[str, Any],
    initial_angle_stage: Mapping[str, Any],
    fine_rounds: list[dict[str, Any]],
    final_verification: Mapping[str, Any],
    converged: bool,
    status: str,
) -> dict[str, Any]:
    result = _copy_calibration(accepted)
    cancel = complex(result["cancel_x"], result["cancel_y"])
    result.update(
        {
            "cancel_amplitude": float(abs(cancel)),
            "cancel_phase": float(np.angle(cancel)),
            "duration_resolution": asdict(duration_resolution),
            "srre_stage": dict(srre_stage),
            "initial_angle_stage": dict(initial_angle_stage),
            "fine_rounds": fine_rounds,
            "fine_round_count": len(fine_rounds),
            "final_verification": dict(final_verification),
            "converged": converged,
            "status": status,
        }
    )
    return result


def _as_four_state_values(values: ArrayLike) -> NDArray[np.float64]:
    result = _as_finite_vector(values, name="measurements")
    if result.shape != (4,):
        raise ValueError("measurements must contain exactly four state values.")
    return result


def _as_finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real array.") from exc
    if np.iscomplexobj(source):
        raise TypeError(f"{name} must be a one-dimensional real array.")
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real array.") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _required(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"{context} is missing required field {key!r}.") from exc


def _validate_label(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _as_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
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


def _as_positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _resolve_optional_positive_integer(
    value: Any,
    *,
    default: int,
    name: str,
) -> int:
    return default if value is None else _as_positive_integer(value, name=name)


def _resolve_optional_positive_float(
    value: Any,
    *,
    default: float,
    name: str,
) -> float:
    return default if value is None else _as_positive_float(value, name=name)


def _context_shot_interval(exp: Experiment) -> float:
    """Resolve the configured shot interval for the current experiment context."""
    experiment_system = getattr(exp.ctx, "experiment_system", None)
    configured_defaults = getattr(experiment_system, "measurement_defaults", None)
    defaults = resolve_measurement_defaults(configured_defaults)
    return float(defaults.execution.shot_interval_ns)
