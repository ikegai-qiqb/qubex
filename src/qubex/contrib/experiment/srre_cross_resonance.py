"""SRRE cross-resonance gate construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qubex.experiment import Experiment
from qubex.pulse import (
    Arbitrary,
    FlatTop,
    PhaseShift,
    PulseArray,
    PulseSchedule,
    Waveform,
)
from qubex.typing import TargetMap

from .srre_waveform import srre_waveform

__all__ = ["srre_rzx"]

_REFERENCE_ANGLE = np.pi / 2.0
_HARDWARE_AMPLITUDE_LIMIT = 1.0
_AMPLITUDE_TOLERANCE = 1e-12
_METADATA_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _SrreParameters:
    target: str
    amplitude: float
    block_duration: float
    ramp_time: float
    sampling_period: float


@dataclass(frozen=True)
class _SrreCrParameters:
    control_qubit: str
    target_qubit: str
    cr_half_duration: float
    cr_ramptime: float
    cr_amplitude: float
    cr_phase: float
    cr_beta: float
    cancel: complex
    cancel_beta: float
    srre: _SrreParameters


@dataclass(frozen=True)
class _ResolvedEcho:
    values: NDArray[np.complex128]
    final_frame_shift: float


def srre_rzx(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    angle: float,
    *,
    calibration: Mapping[str, Any],
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float = 0.0,
) -> PulseSchedule:
    """
    Build an echoed RZX schedule using a calibrated SRRE waveform.

    CR amplitude and cancellation IQ scale linearly relative to the calibrated
    ZX90 angle. SRRE amplitude, shape, and half duration remain fixed. Any
    finite angle, including zero and negative angles, is supported as long as
    the resulting samples remain inside the hardware amplitude limit.

    Parameters
    ----------
    exp : Experiment
        Experiment used to obtain the default control X180 waveform and current
        hardware sampling period.
    control_qubit : str
        Control qubit label.
    target_qubit : str
        Target qubit label.
    angle : float
        Requested RZX rotation angle in radians.
    calibration : Mapping[str, Any]
        Successful `srre_cr_calibration` mapping returned by
        `calibrate_srre_zx90`. If the mapping contains a `status` field, it
        must be `"completed"`.
    x180 : TargetMap[Waveform] | Waveform, optional
        Control X180 waveform or target mapping. The calibrated control X180 is
        used by default.
    x180_margin : float, optional
        Zero-amplitude margin placed before and after each X180 pulse in ns.
        Defaults to `0.0`.

    Returns
    -------
    PulseSchedule
        Final echo-on, SRRE-on RZX schedule suitable for the existing `zx90`
        override accepted by CNOT and benchmarking APIs.

    Raises
    ------
    ValueError
        If calibration metadata is inconsistent, a value is non-finite, pulse
        sampling periods differ, or a final channel sample exceeds magnitude
        one.
    TypeError
        If an input scalar, `calibration`, or `x180` has an incompatible type.
    """
    return _build_srre_cross_resonance(
        exp,
        control_qubit,
        target_qubit,
        angle,
        calibration=calibration,
        echo=True,
        include_srre=True,
        x180=x180,
        x180_margin=x180_margin,
    )


def _build_srre_cross_resonance(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    angle: float,
    *,
    calibration: Mapping[str, Any],
    echo: bool,
    include_srre: bool,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float = 0.0,
) -> PulseSchedule:
    """Build a full-angle gate in one echo/SRRE calibration configuration."""
    control_qubit = _as_label(control_qubit, name="control_qubit")
    target_qubit = _as_label(target_qubit, name="target_qubit")
    if control_qubit == target_qubit:
        raise ValueError("control_qubit and target_qubit must be different.")
    angle = _as_finite_float(angle, name="angle")
    parameters = _parse_calibration(calibration)
    _validate_calibration_metadata(
        exp,
        parameters,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
    )

    coefficient = angle / _REFERENCE_ANGLE
    half = _build_half_schedule(
        cr_label=f"{control_qubit}-{target_qubit}",
        target_qubit=target_qubit,
        parameters=parameters,
        coefficient=coefficient,
        include_srre=include_srre,
    )
    if not echo:
        # ``angle`` is the full-gate angle; each calibrated half contributes half.
        return half.repeated(2)

    resolved_echo = _resolve_echo(
        exp,
        control_qubit=control_qubit,
        x180=x180,
        x180_margin=x180_margin,
        sampling_period=parameters.srre.sampling_period,
    )
    return _build_echoed_schedule(
        half,
        control_qubit=control_qubit,
        resolved_echo=resolved_echo,
        sampling_period=parameters.srre.sampling_period,
    )


def _parse_calibration(calibration: Mapping[str, Any]) -> _SrreCrParameters:
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must be a mapping.")
    status = calibration.get("status")
    if status is not None and status != "completed":
        raise ValueError(
            "calibration status must be 'completed' before building an SRRE gate."
        )
    cancel_x = _as_finite_float(_required(calibration, "cancel_x"), name="cancel_x")
    cancel_y = _as_finite_float(_required(calibration, "cancel_y"), name="cancel_y")

    return _SrreCrParameters(
        control_qubit=_as_label(
            _required(calibration, "control_qubit"), name="control_qubit"
        ),
        target_qubit=_as_label(
            _required(calibration, "target_qubit"), name="target_qubit"
        ),
        cr_half_duration=_as_positive_float(
            _required(calibration, "cr_half_duration"),
            name="cr_half_duration",
        ),
        cr_ramptime=_as_nonnegative_float(
            _required(calibration, "cr_ramptime"), name="cr_ramptime"
        ),
        cr_amplitude=_as_positive_float(
            _required(calibration, "cr_amplitude"), name="cr_amplitude"
        ),
        cr_phase=_as_finite_float(_required(calibration, "cr_phase"), name="cr_phase"),
        cr_beta=_as_finite_float(_required(calibration, "cr_beta"), name="cr_beta"),
        cancel=complex(cancel_x, cancel_y),
        cancel_beta=_as_finite_float(
            _required(calibration, "cancel_beta"), name="cancel_beta"
        ),
        srre=_parse_srre_calibration(_required(calibration, "srre_calibration")),
    )


def _parse_srre_calibration(value: Any) -> _SrreParameters:
    if not isinstance(value, Mapping):
        raise TypeError("calibration['srre_calibration'] must be a mapping.")
    amplitude = _as_positive_float(
        _required(value, "amplitude"), name="srre_calibration.amplitude"
    )
    if amplitude > _HARDWARE_AMPLITUDE_LIMIT + _AMPLITUDE_TOLERANCE:
        raise ValueError("SRRE absolute amplitude must not exceed 1.")
    return _SrreParameters(
        target=_as_label(_required(value, "target"), name="srre_calibration.target"),
        amplitude=amplitude,
        block_duration=_as_positive_float(
            _required(value, "block_duration"),
            name="srre_calibration.block_duration",
        ),
        ramp_time=_as_nonnegative_float(
            _required(value, "ramp_time"), name="srre_calibration.ramp_time"
        ),
        sampling_period=_as_positive_float(
            _required(value, "sampling_period"),
            name="srre_calibration.sampling_period",
        ),
    )


def _validate_calibration_metadata(
    exp: Experiment,
    parameters: _SrreCrParameters,
    *,
    control_qubit: str,
    target_qubit: str,
) -> None:
    if parameters.control_qubit != control_qubit:
        raise ValueError(
            "calibration control_qubit does not match the requested control qubit."
        )
    if parameters.target_qubit != target_qubit:
        raise ValueError(
            "calibration target_qubit does not match the requested target qubit."
        )
    if parameters.srre.target != target_qubit:
        raise ValueError("SRRE calibration target does not match target_qubit.")
    if not np.isclose(
        parameters.srre.block_duration,
        parameters.cr_half_duration,
        rtol=0.0,
        atol=_METADATA_TOLERANCE,
    ):
        raise ValueError("SRRE block_duration must match cr_half_duration.")

    current_sampling_period = _as_positive_float(
        exp.ctx.util.resolve_sampling_period(exp.ctx.measurement.sampling_period),
        name="experiment sampling_period",
    )
    if not np.isclose(
        parameters.srre.sampling_period,
        current_sampling_period,
        rtol=0.0,
        atol=_METADATA_TOLERANCE,
    ):
        raise ValueError(
            "SRRE calibration sampling_period does not match the experiment "
            "sampling period."
        )


def _build_half_schedule(
    *,
    cr_label: str,
    target_qubit: str,
    parameters: _SrreCrParameters,
    coefficient: float,
    include_srre: bool,
) -> PulseSchedule:
    sampling_period = parameters.srre.sampling_period
    cr_waveform = FlatTop(
        duration=parameters.cr_half_duration,
        amplitude=parameters.cr_amplitude * coefficient,
        tau=parameters.cr_ramptime,
        phase=parameters.cr_phase,
        beta=parameters.cr_beta,
        sampling_period=sampling_period,
    )
    scaled_cancel = parameters.cancel * coefficient
    cancellation_waveform = FlatTop(
        duration=parameters.cr_half_duration,
        amplitude=abs(scaled_cancel),
        tau=parameters.cr_ramptime,
        phase=float(np.angle(scaled_cancel)),
        beta=parameters.cancel_beta,
        sampling_period=sampling_period,
    )
    cr_values = np.asarray(cr_waveform.values, dtype=np.complex128)
    target_values = np.asarray(cancellation_waveform.values, dtype=np.complex128).copy()
    if include_srre:
        rotary = srre_waveform(
            block_duration=parameters.srre.block_duration,
            ramp_time=parameters.srre.ramp_time,
            amplitude=parameters.srre.amplitude,
            sampling_period=sampling_period,
        )
        if rotary.length != target_values.size:
            raise ValueError(
                "SRRE block_duration must match cr_half_duration sample for sample."
            )
        target_values += rotary.values

    _validate_channel_samples(cr_values, name="CR channel")
    _validate_channel_samples(target_values, name="target channel")

    with PulseSchedule([cr_label, target_qubit]) as half:
        half.add(cr_label, Arbitrary(cr_values, sampling_period=sampling_period))
        half.add(
            target_qubit, Arbitrary(target_values, sampling_period=sampling_period)
        )
    return half


def _resolve_echo(
    exp: Experiment,
    *,
    control_qubit: str,
    x180: TargetMap[Waveform] | Waveform | None,
    x180_margin: float,
    sampling_period: float,
) -> _ResolvedEcho:
    margin = _as_nonnegative_float(x180_margin, name="x180_margin")
    margin_samples = round(margin / sampling_period)
    if not np.isclose(
        margin,
        margin_samples * sampling_period,
        rtol=0.0,
        atol=_METADATA_TOLERANCE,
    ):
        raise ValueError("x180_margin must be a multiple of the sampling period.")

    if x180 is None:
        pi_pulse = exp.pulse.x180(control_qubit)
    elif isinstance(x180, Waveform):
        pi_pulse = x180
    elif isinstance(x180, Mapping):
        try:
            pi_pulse = x180[control_qubit]
        except KeyError as exc:
            raise ValueError(
                f"x180 mapping does not contain control qubit {control_qubit!r}."
            ) from exc
    else:
        raise TypeError("x180 must be a Waveform, a target mapping, or None.")
    if not isinstance(pi_pulse, Waveform):
        raise TypeError("x180 must resolve to a Waveform.")
    _validate_waveform_sampling_period(
        pi_pulse,
        name="x180",
        sampling_period=sampling_period,
    )

    pi_values = np.asarray(pi_pulse.values, dtype=np.complex128)
    if pi_values.size == 0:
        raise ValueError("x180 must contain at least one sample.")
    _validate_channel_samples(pi_values, name="control echo channel")
    final_frame_shift = (
        pi_pulse.final_frame_shift if isinstance(pi_pulse, PulseArray) else 0.0
    )
    if not np.isfinite(final_frame_shift):
        raise ValueError("x180 final frame shift must be finite.")
    return _ResolvedEcho(
        values=np.pad(pi_values, (margin_samples, margin_samples)),
        final_frame_shift=float(final_frame_shift),
    )


def _build_echoed_schedule(
    half: PulseSchedule,
    *,
    control_qubit: str,
    resolved_echo: _ResolvedEcho,
    sampling_period: float,
) -> PulseSchedule:
    half_values = half.get_sampled_sequences()
    half_samples = len(next(iter(half_values.values())))
    echo_samples = resolved_echo.values.size
    zeros_for_half = np.zeros(half_samples, dtype=np.complex128)
    zeros_for_echo = np.zeros(echo_samples, dtype=np.complex128)

    with PulseSchedule([control_qubit, *half.labels]) as schedule:
        for _ in range(2):
            schedule.add(
                control_qubit,
                Arbitrary(zeros_for_half, sampling_period=sampling_period),
            )
            schedule.add(
                control_qubit,
                Arbitrary(resolved_echo.values, sampling_period=sampling_period),
            )
            if resolved_echo.final_frame_shift != 0.0:
                schedule.add(
                    control_qubit,
                    PhaseShift(resolved_echo.final_frame_shift),
                )
        for label, values in half_values.items():
            schedule.add(
                label,
                Arbitrary(
                    np.concatenate([values, zeros_for_echo, -values, zeros_for_echo]),
                    sampling_period=sampling_period,
                ),
            )
    return schedule


def _validate_waveform_sampling_period(
    waveform: Waveform,
    *,
    name: str,
    sampling_period: float,
) -> None:
    waveforms = (
        waveform.get_flattened_waveforms(apply_frame_shifts=False)
        if isinstance(waveform, PulseArray)
        else [waveform]
    )
    for nested_waveform in waveforms:
        if not np.isclose(
            nested_waveform.sampling_period,
            sampling_period,
            rtol=0.0,
            atol=_METADATA_TOLERANCE,
        ):
            raise ValueError(
                f"{name} sampling period must match the SRRE sampling period."
            )


def _validate_channel_samples(values: np.ndarray, *, name: str) -> None:
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite samples.")
    if np.max(np.abs(values)) > (_HARDWARE_AMPLITUDE_LIMIT + _AMPLITUDE_TOLERANCE):
        raise ValueError(f"{name} amplitude must not exceed the hardware limit of 1.")


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"calibration is missing required field {key!r}.") from exc


def _as_label(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _as_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _as_positive_float(value: Any, *, name: str) -> float:
    scalar = _as_finite_float(value, name=name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def _as_nonnegative_float(value: Any, *, name: str) -> float:
    scalar = _as_finite_float(value, name=name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return scalar
