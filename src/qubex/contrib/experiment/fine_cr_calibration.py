"""
Composable, non-mutating calibration helpers for echoed CR gates.

The functions in this module intentionally separate measurement, parameter
proposal, verification, and persistence.  Calibration stages return candidate
parameters in a `Result`; only `commit_cr_calibration` may update the active
calibration note.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
)
from qubex.experiment.models import Result
from qubex.pulse import CrossResonance, PhaseShift, PulseSchedule, VirtualZ, Waveform

if TYPE_CHECKING:
    from qubex.experiment import Experiment

__all__ = [
    "CrCalibrationOptions",
    "CrCalibrationTolerances",
    "CrGateParameters",
    "build_ecr_gate",
    "calibrate_bare_cr",
    "calibrate_ecr_angle",
    "calibrate_ecr_local_z",
    "calibrate_ecr_phase",
    "calibrate_un_echoed_cancellation",
    "characterize_bare_cr",
    "characterize_ecr_gate",
    "check_cr_prerequisites",
    "commit_cr_calibration",
    "measure_ecr_leakage",
    "optimize_ecr_rotary",
    "scout_cr_operating_points",
    "validate_ecr_gate",
]

_PAULI_TERMS = ("IX", "IY", "IZ", "ZX", "ZY", "ZZ")
_EPS = 1e-12


@dataclass(frozen=True)
class CrGateParameters:
    """
    Store one explicit candidate definition of an echoed CR gate.

    Amplitudes use hardware-normalized units, phases are in radians,
    durations are in nanoseconds, and `cr_detuning` is in GHz.
    """

    cr_amplitude: float
    cr_phase: float
    cr_lobe_duration: float
    cr_ramptime: float
    cr_beta: float = 0.0
    cancel_x: float = 0.0
    cancel_y: float = 0.0
    cancel_beta: float = 0.0
    rotary_x: float = 0.0
    rotary_y: float = 0.0
    cr_detuning: float = 0.0
    control_frame_z: float = 0.0
    target_frame_z: float = 0.0
    zx_rotation_rate: float | None = None

    def __post_init__(self) -> None:
        """Validate finite pulse parameters and physical timing constraints."""
        numeric = {
            key: value
            for key, value in asdict(self).items()
            if key != "zx_rotation_rate" or value is not None
        }
        if not all(np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("CR gate parameters must contain only finite values.")
        if self.cr_amplitude <= 0:
            raise ValueError("cr_amplitude must be positive.")
        if self.cr_ramptime < 0:
            raise ValueError("cr_ramptime must be non-negative.")
        if self.cr_lobe_duration <= 2 * self.cr_ramptime:
            raise ValueError("cr_lobe_duration must be greater than twice cr_ramptime.")
        if self.zx_rotation_rate is not None and self.zx_rotation_rate == 0:
            raise ValueError("zx_rotation_rate must be nonzero when provided.")

    @property
    def cancel_complex(self) -> complex:
        """Return the cancellation tone as a complex Cartesian amplitude."""
        return complex(self.cancel_x, self.cancel_y)

    @property
    def rotary_complex(self) -> complex:
        """Return the rotary tone as a complex Cartesian amplitude."""
        return complex(self.rotary_x, self.rotary_y)


@dataclass(frozen=True)
class CrCalibrationOptions:
    """Store common acquisition and repetition settings for CR calibration."""

    n_shots: int = CALIBRATION_SHOTS
    shot_interval: float = DEFAULT_INTERVAL
    repetition_counts: tuple[int, ...] = tuple(range(9))
    max_iterations: int = 3
    duration_unit: float = 16.0
    verification_shots: int = CALIBRATION_SHOTS
    plot: bool = False

    def __post_init__(self) -> None:
        """Validate acquisition and discretization options."""
        _validate_positive_integer(self.n_shots, name="n_shots")
        _validate_positive_integer(
            self.verification_shots,
            name="verification_shots",
        )
        if not np.isfinite(self.shot_interval) or self.shot_interval <= 0:
            raise ValueError("shot_interval must be positive and finite.")
        _validate_repetition_counts(self.repetition_counts)
        _validate_positive_integer(self.max_iterations, name="max_iterations")
        if not np.isfinite(self.duration_unit) or self.duration_unit <= 0:
            raise ValueError("duration_unit must be positive and finite.")


@dataclass(frozen=True)
class CrCalibrationTolerances:
    """Store convergence and final validation thresholds for CR calibration."""

    coarse_zy_ratio: float = 0.10
    coarse_ix_ratio: float = 0.10
    coarse_iy_ratio: float = 0.10
    fine_angle_error: float = 0.02
    fine_error_angle: float = 0.02
    leakage_per_gate: float = 1e-3
    irb_error: float = 1e-2
    bell_fidelity: float = 0.95
    fit_r2: float = 0.90

    def __post_init__(self) -> None:
        """Validate non-negative errors and probability-like thresholds."""
        values = asdict(self)
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise ValueError("calibration tolerances must be finite.")
        for key, value in values.items():
            if key in {"bell_fidelity", "fit_r2"}:
                if not 0 <= float(value) <= 1:
                    raise ValueError(f"{key} must be in [0, 1].")
            elif float(value) < 0:
                raise ValueError(f"{key} must be non-negative.")


def _result_mapping(result: Result | Mapping[str, object]) -> Mapping[str, object]:
    """Return the payload mapping from a `Result` or mapping."""
    if isinstance(result, Result):
        return result.data
    return result


def _stage_result(
    *,
    stage: str,
    input_params: CrGateParameters | None,
    proposed_params: CrGateParameters | None,
    converged: bool,
    verified: bool,
    supported: bool = True,
    status: str,
    reason: str | None = None,
    metrics_before: Mapping[str, object] | None = None,
    metrics_after: Mapping[str, object] | None = None,
    uncertainties: Mapping[str, object] | None = None,
    fit_quality: Mapping[str, object] | None = None,
    sweep: object = None,
    raw_results: object = None,
    **extra: object,
) -> Result:
    """Build the consistent payload shared by every calibration stage."""
    payload: dict[str, object] = {
        "stage": stage,
        "input_params": input_params,
        "proposed_params": proposed_params,
        "converged": converged,
        "verified": verified,
        "supported": supported,
        "status": status,
        "reason": reason,
        "metrics_before": dict(metrics_before or {}),
        "metrics_after": dict(metrics_after or {}),
        "uncertainties": dict(uncertainties or {}),
        "fit_quality": dict(fit_quality or {}),
        "sweep": sweep,
        "raw_results": raw_results,
    }
    payload.update(extra)
    return Result(data=payload)


def _as_1d_finite(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a non-empty one-dimensional finite float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_float(value: object) -> float | None:
    """Return a finite float conversion or `None` when conversion fails."""
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _validate_positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer while rejecting booleans and fractional values."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _as_integer_counts(values: Sequence[int]) -> tuple[int, ...]:
    """Return repetition counts without silently coercing fractional values."""
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError("repetition_counts must contain integers.")
    return tuple(int(value) for value in values)


def _validate_repetition_counts(values: Sequence[int]) -> tuple[int, ...]:
    """Validate sorted, distinct, non-negative repetition counts."""
    counts = _as_integer_counts(values)
    if len(counts) < 3:
        raise ValueError("repetition_counts must contain at least three values.")
    if any(value < 0 for value in counts):
        raise ValueError("repetition_counts must be non-negative.")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("repetition_counts must be sorted and distinct.")
    return counts


def _parameter_fingerprint(params: CrGateParameters) -> str:
    """
    Return a deterministic token binding validation to exact parameters.

    Parameters
    ----------
    params
        Candidate gate parameters.

    Returns
    -------
    str
        SHA-256 hexadecimal digest of the dataclass payload.
    """
    serialized = json.dumps(asdict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _gate_fingerprint(schedule: PulseSchedule) -> str:
    """Return a digest of sampled IQ and explicit frame shifts in a schedule."""
    digest = hashlib.sha256()
    for label in schedule.labels:
        digest.update(label.encode())
        sequence = schedule.get_sequence(label)
        for element in sequence.flattened_elements:
            digest.update(type(element).__qualname__.encode())
            if isinstance(element, PhaseShift):
                digest.update(np.float64(element.theta).tobytes())
            elif isinstance(element, Waveform):
                values = np.asarray(element.values, dtype=np.complex128)
                digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
                digest.update(values.tobytes())
    return digest.hexdigest()


def _validation_token(params: CrGateParameters, schedule: PulseSchedule) -> str:
    """Bind a validation token to parameters and the realized pulse schedule."""
    digest = hashlib.sha256()
    digest.update(_parameter_fingerprint(params).encode())
    digest.update(_gate_fingerprint(schedule).encode())
    return digest.hexdigest()


def _x180(exp: Experiment, target: str) -> Waveform:
    """Return a calibrated control X180 through the public experiment facade."""
    method = getattr(exp, "x180", None)
    if callable(method):
        return cast(Waveform, method(target))
    pulse_service = getattr(exp, "pulse", None)
    method = getattr(pulse_service, "x180", None)
    if callable(method):
        return cast(Waveform, method(target))
    raise ValueError("The experiment does not expose a calibrated x180 pulse.")


def _build_cr_schedule(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    echo: bool,
    include_rotary: bool,
    include_local_frames: bool,
) -> PulseSchedule:
    """Build one CR primitive without reading implicit CR calibration values."""
    target_tone = params.cancel_complex
    if include_rotary:
        target_tone += params.rotary_complex
    kwargs: dict[str, object] = {
        "control_qubit": control_qubit,
        "target_qubit": target_qubit,
        "cr_amplitude": params.cr_amplitude,
        "cr_duration": params.cr_lobe_duration,
        "cr_ramptime": params.cr_ramptime,
        "cr_phase": params.cr_phase,
        "cr_beta": params.cr_beta,
        "cr_detuning": params.cr_detuning,
        "cancel_amplitude": abs(target_tone),
        "cancel_phase": float(np.angle(target_tone)),
        "cancel_beta": params.cancel_beta,
        "echo": echo,
    }
    if echo:
        kwargs["pi_pulse"] = _x180(exp, control_qubit)
    schedule = CrossResonance(**kwargs)  # type: ignore[arg-type]
    if not include_local_frames:
        return schedule
    if params.control_frame_z == 0 and params.target_frame_z == 0:
        return schedule
    with PulseSchedule(schedule.labels) as framed:
        framed.call(schedule)
        framed.barrier()
        if params.control_frame_z != 0:
            framed.add(control_qubit, VirtualZ(params.control_frame_z))
        if params.target_frame_z != 0:
            framed.add(target_qubit, VirtualZ(params.target_frame_z))
            framed.add(
                f"{control_qubit}-{target_qubit}", VirtualZ(params.target_frame_z)
            )
    return framed


def build_ecr_gate(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    repetitions: int = 1,
) -> PulseSchedule:
    """
    Build the exact echoed CR gate represented by explicit parameters.

    The function never reads or updates stored CR parameters.  CR detuning is
    applied only to the CR lobes, while local frame corrections are appended
    once per completed echoed gate.

    Parameters
    ----------
    exp
        Experiment providing the calibrated control X180 pulse.
    control_qubit
        Control-qubit label.
    target_qubit
        Target-qubit label.
    params
        Explicit candidate gate definition.
    repetitions
        Number of complete gates to concatenate. Must be positive.

    Returns
    -------
    PulseSchedule
        Echoed CR pulse schedule.
    """
    repetitions = _validate_positive_integer(repetitions, name="repetitions")
    gate = _build_cr_schedule(
        exp,
        control_qubit,
        target_qubit,
        params=params,
        echo=True,
        include_rotary=True,
        include_local_frames=True,
    )
    return gate.repeated(repetitions)


def _property_value(exp: Experiment, key: str, target: str) -> object | None:
    """Read one optional device property without triggering measurement."""
    ctx = getattr(exp, "ctx", None)
    manager = getattr(ctx, "system_manager", None)
    loader = getattr(manager, "config_loader", None)
    load = getattr(loader, "load_param_data", None)
    if not callable(load):
        return None
    try:
        values = load(key)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if isinstance(values, Mapping):
        return values.get(target)
    return None


def check_cr_prerequisites(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    require_qutrit_classifier: bool = True,
) -> Result:
    """
    Inspect prerequisites for CR calibration without running hardware.

    Returns a checklist for qubit metadata, calibrated X180 pulses,
    classifiers, anharmonicities, and stored coherence properties. Static ZZ
    is reported as requiring a separate JAZZ measurement because no canonical
    persisted field currently exists.
    """
    ctx = getattr(exp, "ctx", None)
    qubits = getattr(ctx, "qubits", {})
    classifiers = getattr(ctx, "classifiers", {})
    checks: dict[str, dict[str, object]] = {}

    for role, target in (("control", control_qubit), ("target", target_qubit)):
        qubit = qubits.get(target) if isinstance(qubits, Mapping) else None
        frequency = getattr(qubit, "frequency", None)
        anharmonicity = getattr(qubit, "anharmonicity", None)
        checks[f"{role}_qubit"] = {
            "passed": qubit is not None,
            "value": qubit,
            "required": True,
        }
        checks[f"{role}_frequency"] = {
            "passed": frequency is not None and np.isfinite(frequency),
            "value": frequency,
            "required": True,
        }
        checks[f"{role}_anharmonicity"] = {
            "passed": anharmonicity is not None and np.isfinite(anharmonicity),
            "value": anharmonicity,
            "required": True,
        }
        try:
            pulse = _x180(exp, target)
        except (KeyError, TypeError, ValueError):
            pulse = None
        checks[f"{role}_x180"] = {
            "passed": pulse is not None,
            "value": pulse,
            "required": True,
        }
        classifier = (
            classifiers.get(target) if isinstance(classifiers, Mapping) else None
        )
        n_states = getattr(classifier, "n_states", None)
        classifier_ok = classifier is not None and (
            not require_qutrit_classifier or n_states == 3
        )
        checks[f"{role}_classifier"] = {
            "passed": classifier_ok,
            "value": n_states,
            "required": True,
        }
        for property_key in ("t1", "t2_echo"):
            value = _property_value(exp, property_key, target)
            finite_value = _finite_float(value)
            checks[f"{role}_{property_key}"] = {
                "passed": finite_value is not None,
                "value": finite_value,
                "required": True,
            }

    checks["static_zz"] = {
        "passed": False,
        "value": None,
        "required": False,
        "status": "not_run",
        "reason": "Run jazz_experiment and record the result before scouting.",
    }
    missing = [
        name
        for name, check in checks.items()
        if bool(check["required"]) and not bool(check["passed"])
    ]
    passed = not missing
    return _stage_result(
        stage="check_cr_prerequisites",
        input_params=None,
        proposed_params=None,
        converged=passed,
        verified=passed,
        status="success" if passed else "missing_prerequisites",
        reason=None if passed else f"Missing prerequisites: {', '.join(missing)}",
        metrics_after={"n_missing": len(missing)},
        raw_results=checks,
        checks=checks,
        missing=missing,
        passed=passed,
    )


def _extract_coefficients(result: Result | Mapping[str, object]) -> dict[str, float]:
    """Extract and validate six finite CR Hamiltonian coefficients."""
    data = _result_mapping(result)
    raw = data.get("coeffs")
    if not isinstance(raw, Mapping):
        raise TypeError("CR tomography result does not contain a coefficient mapping.")
    coeffs: dict[str, float] = {}
    for term in _PAULI_TERMS:
        value = raw.get(term)
        if value is None or not np.isfinite(value):
            raise ValueError(f"CR tomography coefficient {term} is missing or invalid.")
        coeffs[term] = float(value)
    return coeffs


def _fit_r2_value(result: Result | Mapping[str, object]) -> float | None:
    """Return the minimum available fit R² from one experiment result."""
    data = _result_mapping(result)
    quality = data.get("fit_quality")
    if isinstance(quality, Mapping):
        direct = _finite_float(quality.get("r2"))
        if direct is not None:
            return direct
    values: list[float] = []
    for branch in ("result_0", "result_1"):
        branch_result = data.get(branch)
        if not isinstance(branch_result, (Result, Mapping)):
            continue
        branch_data = _result_mapping(branch_result)
        fit_result = branch_data.get("fit_result")
        if isinstance(fit_result, (Result, Mapping)):
            r2 = _finite_float(_result_mapping(fit_result).get("r2"))
            if r2 is not None:
                values.append(r2)
    return min(values) if values else None


def _coefficient_metrics(coeffs: Mapping[str, float]) -> dict[str, float]:
    """Return normalized coarse-calibration error metrics."""
    zx = abs(coeffs["ZX"])
    denominator = max(zx, _EPS)
    return {
        **{term: float(coeffs[term]) for term in _PAULI_TERMS},
        "zy_ratio": abs(coeffs["ZY"]) / denominator,
        "ix_ratio": abs(coeffs["IX"]) / denominator,
        "iy_ratio": abs(coeffs["IY"]) / denominator,
        "transverse_error_ratio": math.hypot(coeffs["IX"], coeffs["IY"]) / denominator,
    }


def characterize_bare_cr(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    time_range: ArrayLike,
    n_shots: int = CALIBRATION_SHOTS,
    shot_interval: float = DEFAULT_INTERVAL,
    plot: bool = False,
) -> Result:
    """
    Measure the six bare-CR Hamiltonian coefficients without mutation.

    The current public tomography API cannot apply CR/cancellation DRAG or CR
    detuning.  Nonzero values are therefore rejected explicitly instead of
    silently characterizing a different waveform. Rotary tones and local frame
    corrections are intentionally omitted in the bare-CR stage.
    """
    times = _as_1d_finite(time_range, name="time_range")
    unsupported = []
    if params.cr_beta != 0:
        unsupported.append("cr_beta")
    if params.cancel_beta != 0:
        unsupported.append("cancel_beta")
    if params.cr_detuning != 0:
        unsupported.append("cr_detuning")
    if unsupported:
        reason = "The public bare-CR tomography API cannot apply: " + ", ".join(
            unsupported
        )
        return _stage_result(
            stage="characterize_bare_cr",
            input_params=params,
            proposed_params=params,
            converged=False,
            verified=False,
            supported=False,
            status="unsupported_waveform_parameters",
            reason=reason,
        )
    cancel_amplitude = abs(params.cancel_complex)
    cancel_phase = float(np.angle(params.cancel_complex))
    try:
        raw = exp.cr_hamiltonian_tomography(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=times,
            ramptime=params.cr_ramptime,
            cr_amplitude=params.cr_amplitude,
            cr_phase=params.cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
        )
        coeffs = _extract_coefficients(raw)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return _stage_result(
            stage="characterize_bare_cr",
            input_params=params,
            proposed_params=params,
            converged=False,
            verified=False,
            status="measurement_failed",
            reason=str(exc),
        )
    metrics = _coefficient_metrics(coeffs)
    raw_data = _result_mapping(raw)
    uncertainties = raw_data.get("uncertainties", {})
    fit_quality = raw_data.get("fit_quality", {})
    r2 = _fit_r2_value(raw)
    if r2 is not None:
        fit_quality = {
            **(fit_quality if isinstance(fit_quality, Mapping) else {}),
            "r2": r2,
        }
    return _stage_result(
        stage="characterize_bare_cr",
        input_params=params,
        proposed_params=params,
        converged=True,
        verified=True,
        status="success",
        metrics_before=metrics,
        metrics_after=metrics,
        uncertainties=cast(Mapping[str, object], uncertainties)
        if isinstance(uncertainties, Mapping)
        else {},
        fit_quality=cast(Mapping[str, object], fit_quality)
        if isinstance(fit_quality, Mapping)
        else {},
        raw_results=raw,
        coeffs=coeffs,
        xt_rotation_amplitude_hw=raw_data.get("xt_rotation_amplitude_hw"),
        xt_rotation_phase=raw_data.get("xt_rotation_phase"),
    )


def _duration_for_zx90(
    zx_rate: float,
    *,
    ramptime: float,
    duration_unit: float,
) -> float:
    """Estimate and discretize one ECR lobe duration from a bare ZX rate."""
    if not np.isfinite(zx_rate) or zx_rate <= 0:
        raise ValueError("A positive finite ZX rate is required.")
    duration = ramptime + 1 / (8 * zx_rate)
    duration = max(duration, 2 * ramptime + duration_unit)
    return float(np.ceil(duration / duration_unit) * duration_unit)


def scout_cr_operating_points(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    cr_amplitudes: ArrayLike,
    time_range: ArrayLike,
    initial_params: CrGateParameters | None = None,
    ramptime: float = 32.0,
    options: CrCalibrationOptions | None = None,
    tolerances: CrCalibrationTolerances | None = None,
) -> Result:
    """
    Characterize and rank bare-CR operating points without selecting storage.

    Leakage scouting is marked `not_run` because the existing bare tomography
    primitive returns averaged Bloch vectors rather than qutrit populations.
    It must be measured separately before accepting a recommended point.
    """
    amplitudes = _as_1d_finite(cr_amplitudes, name="cr_amplitudes")
    if np.any(amplitudes <= 0):
        raise ValueError("cr_amplitudes must be positive.")
    times = _as_1d_finite(time_range, name="time_range")
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    if initial_params is None:
        initial_params = CrGateParameters(
            cr_amplitude=float(amplitudes[0]),
            cr_phase=0.0,
            cr_lobe_duration=max(float(np.max(times)), 2 * ramptime + 16),
            cr_ramptime=ramptime,
        )
    candidates: list[dict[str, object]] = []
    raw_results: list[Result] = []
    for amplitude in amplitudes:
        candidate = replace(initial_params, cr_amplitude=float(amplitude))
        result = characterize_bare_cr(
            exp,
            control_qubit,
            target_qubit,
            params=candidate,
            time_range=times,
            n_shots=settings.n_shots,
            shot_interval=settings.shot_interval,
            plot=settings.plot,
        )
        raw_results.append(result)
        data = result.data
        r2 = _fit_r2_value(result)
        if not bool(data.get("converged")) or r2 is None or r2 < thresholds.fit_r2:
            candidates.append(
                {
                    "params": candidate,
                    "valid": False,
                    "score": float("-inf"),
                    "reason": (
                        data.get("reason")
                        if not bool(data.get("converged"))
                        else "Tomography fit R² is below tolerance or unavailable."
                    ),
                }
            )
            continue
        coeffs = cast(Mapping[str, float], data["coeffs"])
        zx = abs(float(coeffs["ZX"]))
        error = math.sqrt(
            float(coeffs["IX"]) ** 2
            + float(coeffs["IY"]) ** 2
            + float(coeffs["ZY"]) ** 2
        ) / max(zx, _EPS)
        score = zx / (1 + 10 * error)
        duration = _duration_for_zx90(
            zx,
            ramptime=candidate.cr_ramptime,
            duration_unit=settings.duration_unit,
        )
        recommended = replace(
            candidate,
            cr_lobe_duration=duration,
            zx_rotation_rate=float(coeffs["ZX"]) / candidate.cr_amplitude,
        )
        candidates.append(
            {
                "params": recommended,
                "valid": True,
                "score": score,
                "zx_rate": zx,
                "error_ratio": error,
                "estimated_lobe_duration": duration,
                "leakage": {
                    "status": "not_run",
                    "reason": "Bare CR tomography does not return qutrit populations.",
                },
            }
        )
    ranked_indices = sorted(
        (index for index, item in enumerate(candidates) if bool(item["valid"])),
        key=lambda index: cast(float, candidates[index]["score"]),
        reverse=True,
    )
    recommended_params = (
        cast(CrGateParameters, candidates[ranked_indices[0]]["params"])
        if ranked_indices
        else initial_params
    )
    return _stage_result(
        stage="scout_cr_operating_points",
        input_params=initial_params,
        proposed_params=recommended_params,
        converged=bool(ranked_indices),
        verified=False,
        status="candidates_ranked" if ranked_indices else "no_valid_candidate",
        reason=None
        if ranked_indices
        else "No operating point produced valid tomography.",
        sweep={"cr_amplitudes": amplitudes},
        raw_results=raw_results,
        candidates=candidates,
        recommended_indices=ranked_indices,
        recommended_params=recommended_params,
        leakage_status="not_run",
    )


def _coarse_converged(
    coeffs: Mapping[str, float],
    tolerances: CrCalibrationTolerances,
) -> bool:
    """Return whether bare transverse errors satisfy coarse thresholds."""
    metrics = _coefficient_metrics(coeffs)
    return bool(
        metrics["zy_ratio"] <= tolerances.coarse_zy_ratio
        and metrics["ix_ratio"] <= tolerances.coarse_ix_ratio
        and metrics["iy_ratio"] <= tolerances.coarse_iy_ratio
    )


def _calibration_result_coeffs(result: Result) -> dict[str, float] | None:
    """Return coefficients from a successful stage result."""
    if not bool(result.data.get("converged")):
        return None
    raw = result.data.get("coeffs")
    if not isinstance(raw, Mapping):
        return None
    return {term: float(raw[term]) for term in _PAULI_TERMS}


def _hardware_cancellation_correction(
    exp: Experiment,
    target_qubit: str,
    characterization: Result,
    coeffs: Mapping[str, float],
) -> complex:
    """Return the cancellation tone that opposes measured IX/IY."""
    amplitude = characterization.data.get("xt_rotation_amplitude_hw")
    phase = characterization.data.get("xt_rotation_phase")
    if amplitude is None or phase is None:
        calc = getattr(exp, "calc_control_amplitude", None)
        if not callable(calc):
            pulse_service = getattr(exp, "pulse", None)
            calc = getattr(pulse_service, "calc_control_amplitude", None)
        if not callable(calc):
            raise ValueError(
                "Tomography did not return a hardware IX/IY amplitude and the "
                "experiment has no calc_control_amplitude method."
            )
        amplitude = calc(target_qubit, math.hypot(coeffs["IX"], coeffs["IY"]))
        phase = math.atan2(coeffs["IY"], coeffs["IX"])
    finite_amplitude = _finite_float(amplitude)
    finite_phase = _finite_float(phase)
    if finite_amplitude is None or finite_phase is None:
        raise ValueError("The inferred cancellation correction is not finite.")
    return -finite_amplitude * np.exp(1j * finite_phase)


def _rotate_common_phase(
    params: CrGateParameters,
    phase_delta: float,
) -> CrGateParameters:
    """Rotate CR, cancellation, and rotary IQ by one common phase."""
    phase_factor = np.exp(1j * phase_delta)
    cancellation = params.cancel_complex * phase_factor
    rotary = params.rotary_complex * phase_factor
    return replace(
        params,
        cr_phase=params.cr_phase + phase_delta,
        cancel_x=float(cancellation.real),
        cancel_y=float(cancellation.imag),
        rotary_x=float(rotary.real),
        rotary_y=float(rotary.imag),
    )


def calibrate_bare_cr(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    initial_params: CrGateParameters,
    time_range: ArrayLike,
    max_iterations: int | None = None,
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Coarse-calibrate bare CR phase and cancellation IQ iteratively.

    Each phase proposal is measured before the cancellation update. The full
    update is accepted only after a final tomography measurement confirms that
    the aggregate normalized transverse error did not increase. No parameters
    are persisted by this function.
    """
    times = _as_1d_finite(time_range, name="time_range")
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    iterations = settings.max_iterations if max_iterations is None else max_iterations
    iterations = _validate_positive_integer(iterations, name="max_iterations")
    current = initial_params
    history: list[dict[str, object]] = []
    initial_measurement = characterize_bare_cr(
        exp,
        control_qubit,
        target_qubit,
        params=current,
        time_range=times,
        n_shots=settings.n_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    initial_coeffs = _calibration_result_coeffs(initial_measurement)
    initial_r2 = _fit_r2_value(initial_measurement)
    if initial_coeffs is None or initial_r2 is None or initial_r2 < thresholds.fit_r2:
        return _stage_result(
            stage="calibrate_bare_cr",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            supported=bool(initial_measurement.data.get("supported", True)),
            status="initial_characterization_failed",
            reason=(
                cast(str | None, initial_measurement.data.get("reason"))
                if initial_coeffs is None
                else "Initial tomography fit R² is below tolerance or unavailable."
            ),
            raw_results=[initial_measurement],
            iterations=history,
        )
    before_metrics = _coefficient_metrics(initial_coeffs)
    last_measurement = initial_measurement
    last_coeffs = initial_coeffs
    if _coarse_converged(last_coeffs, thresholds):
        current = replace(
            current,
            zx_rotation_rate=last_coeffs["ZX"] / current.cr_amplitude,
        )
        return _stage_result(
            stage="calibrate_bare_cr",
            input_params=initial_params,
            proposed_params=current,
            converged=True,
            verified=True,
            status="already_converged",
            metrics_before=before_metrics,
            metrics_after=before_metrics,
            raw_results=[initial_measurement],
            iterations=history,
            verification=initial_measurement,
        )

    for index in range(iterations):
        cr_vector = complex(last_coeffs["ZX"], last_coeffs["ZY"])
        if abs(cr_vector) <= _EPS:
            return _stage_result(
                stage="calibrate_bare_cr",
                input_params=initial_params,
                proposed_params=current,
                converged=False,
                verified=False,
                status="zero_zx_rate",
                reason="Cannot calibrate phase because ZX/ZY magnitude is zero.",
                metrics_before=before_metrics,
                metrics_after=_coefficient_metrics(last_coeffs),
                raw_results=[initial_measurement, last_measurement],
                iterations=history,
            )
        phase_delta = -float(np.angle(cr_vector))
        phase_candidate = _rotate_common_phase(current, phase_delta)
        phase_measurement = characterize_bare_cr(
            exp,
            control_qubit,
            target_qubit,
            params=phase_candidate,
            time_range=times,
            n_shots=settings.n_shots,
            shot_interval=settings.shot_interval,
            plot=settings.plot,
        )
        phase_coeffs = _calibration_result_coeffs(phase_measurement)
        phase_r2 = _fit_r2_value(phase_measurement)
        if phase_coeffs is None or phase_r2 is None or phase_r2 < thresholds.fit_r2:
            return _stage_result(
                stage="calibrate_bare_cr",
                input_params=initial_params,
                proposed_params=current,
                converged=False,
                verified=False,
                status="phase_verification_failed",
                reason=(
                    cast(str | None, phase_measurement.data.get("reason"))
                    if phase_coeffs is None
                    else "Phase-verification fit R² is below tolerance or unavailable."
                ),
                metrics_before=before_metrics,
                metrics_after=_coefficient_metrics(last_coeffs),
                raw_results=[initial_measurement, phase_measurement],
                iterations=history,
            )
        try:
            correction = _hardware_cancellation_correction(
                exp,
                target_qubit,
                phase_measurement,
                phase_coeffs,
            )
        except ValueError as exc:
            return _stage_result(
                stage="calibrate_bare_cr",
                input_params=initial_params,
                proposed_params=current,
                converged=False,
                verified=False,
                status="cancellation_inference_failed",
                reason=str(exc),
                metrics_before=before_metrics,
                metrics_after=_coefficient_metrics(last_coeffs),
                raw_results=[initial_measurement, phase_measurement],
                iterations=history,
            )
        cancellation = phase_candidate.cancel_complex + correction
        cancel_candidate = replace(
            phase_candidate,
            cancel_x=float(cancellation.real),
            cancel_y=float(cancellation.imag),
        )
        verification = characterize_bare_cr(
            exp,
            control_qubit,
            target_qubit,
            params=cancel_candidate,
            time_range=times,
            n_shots=settings.verification_shots,
            shot_interval=settings.shot_interval,
            plot=settings.plot,
        )
        verified_coeffs = _calibration_result_coeffs(verification)
        verification_r2 = _fit_r2_value(verification)
        if (
            verified_coeffs is None
            or verification_r2 is None
            or verification_r2 < thresholds.fit_r2
        ):
            return _stage_result(
                stage="calibrate_bare_cr",
                input_params=initial_params,
                proposed_params=current,
                converged=False,
                verified=False,
                status="update_verification_failed",
                reason=(
                    cast(str | None, verification.data.get("reason"))
                    if verified_coeffs is None
                    else "Update-verification fit R² is below tolerance or unavailable."
                ),
                metrics_before=before_metrics,
                metrics_after=_coefficient_metrics(last_coeffs),
                raw_results=[initial_measurement, phase_measurement, verification],
                iterations=history,
            )
        previous_error = math.sqrt(
            last_coeffs["ZY"] ** 2 + last_coeffs["IX"] ** 2 + last_coeffs["IY"] ** 2
        ) / max(abs(last_coeffs["ZX"]), _EPS)
        verified_error = math.sqrt(
            verified_coeffs["ZY"] ** 2
            + verified_coeffs["IX"] ** 2
            + verified_coeffs["IY"] ** 2
        ) / max(abs(verified_coeffs["ZX"]), _EPS)
        accepted = verified_error <= previous_error + _EPS
        history.append(
            {
                "index": index,
                "input_params": current,
                "phase_delta": phase_delta,
                "cancellation_correction": correction,
                "candidate_params": cancel_candidate,
                "phase_measurement": phase_measurement,
                "verification": verification,
                "accepted": accepted,
                "error_before": previous_error,
                "error_after": verified_error,
            }
        )
        if not accepted:
            return _stage_result(
                stage="calibrate_bare_cr",
                input_params=initial_params,
                proposed_params=current,
                converged=False,
                verified=False,
                status="verification_worsened",
                reason="The proposed phase/cancellation update increased residual error.",
                metrics_before=before_metrics,
                metrics_after=_coefficient_metrics(last_coeffs),
                raw_results=history,
                iterations=history,
                verification=verification,
            )
        current = replace(
            cancel_candidate,
            zx_rotation_rate=(verified_coeffs["ZX"] / cancel_candidate.cr_amplitude),
        )
        last_measurement = verification
        last_coeffs = verified_coeffs
        if _coarse_converged(last_coeffs, thresholds):
            return _stage_result(
                stage="calibrate_bare_cr",
                input_params=initial_params,
                proposed_params=current,
                converged=True,
                verified=True,
                status="success",
                metrics_before=before_metrics,
                metrics_after=_coefficient_metrics(last_coeffs),
                raw_results=history,
                iterations=history,
                verification=last_measurement,
            )

    return _stage_result(
        stage="calibrate_bare_cr",
        input_params=initial_params,
        proposed_params=current,
        converged=False,
        verified=True,
        status="max_iterations_reached",
        reason="Coarse tolerances were not reached within max_iterations.",
        metrics_before=before_metrics,
        metrics_after=_coefficient_metrics(last_coeffs),
        raw_results=history,
        iterations=history,
        verification=last_measurement,
    )


def _rotate_z_initial_state(
    counts: NDArray[np.float64],
    rotation_vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ideal Bloch vectors for repeated rotations starting from +Z."""
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= _EPS:
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(counts), 1))
    axis = rotation_vector / angle
    theta = counts * angle
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(axis, z)
    dot = float(axis[2])
    return (
        np.cos(theta)[:, None] * z
        + np.sin(theta)[:, None] * cross
        + (1 - np.cos(theta))[:, None] * axis * dot
    )


def _initial_rotation_guess(
    counts: NDArray[np.float64],
    states: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Estimate an XY-axis rotation from the first nonzero repetition."""
    indices = np.flatnonzero(counts > 0)
    if not len(indices):
        return np.array([np.pi / 2, 0.0, 0.0])
    index = int(indices[0])
    vector = states[index]
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return np.array([np.pi / 2, 0.0, 0.0])
    vector = vector / norm
    step = float(counts[index])
    theta = math.acos(float(np.clip(vector[2], -1, 1))) / step
    sine = math.sin(theta * step)
    if abs(sine) <= 1e-6:
        return np.array([np.pi / 2, 0.0, 0.0])
    axis_x = -float(vector[1]) / sine
    axis_y = float(vector[0]) / sine
    axis_norm = math.hypot(axis_x, axis_y)
    if axis_norm <= _EPS:
        return np.array([np.pi / 2, 0.0, 0.0])
    return theta * np.array([axis_x / axis_norm, axis_y / axis_norm, 0.0])


def _fit_repeated_rotation(
    counts: NDArray[np.float64],
    states: ArrayLike,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Fit one conditional rotation vector in radians per completed gate."""
    vectors = np.asarray(states, dtype=np.float64)
    if vectors.shape != (len(counts), 3):
        raise ValueError("Tomography data must have shape (len(repetition_counts), 3).")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Tomography data contains non-finite values.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= _EPS):
        raise ValueError("Tomography contains a zero-length Bloch vector.")
    normalized = vectors / norms[:, None]
    guess = _initial_rotation_guess(counts, normalized)

    def residual(rotation_vector: NDArray[np.float64]) -> NDArray[np.float64]:
        predicted = _rotate_z_initial_state(counts, rotation_vector)
        return (predicted - normalized).ravel()

    fit = least_squares(
        residual,
        guess,
        bounds=(-np.pi * np.ones(3), np.pi * np.ones(3)),
        max_nfev=4000,
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Repeated-rotation fit failed: {fit.message}")
    prediction = _rotate_z_initial_state(counts, fit.x)
    residual_sum = float(np.sum((normalized - prediction) ** 2))
    total_sum = float(np.sum((normalized - np.mean(normalized, axis=0)) ** 2))
    r2 = 1 - residual_sum / total_sum if total_sum > _EPS else 1.0
    return fit.x.astype(np.float64), {
        "r2": r2,
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "nfev": float(fit.nfev),
    }


def _characterize_repeated_schedule(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    schedule: PulseSchedule,
    params: CrGateParameters,
    repetition_counts: Sequence[int],
    n_shots: int,
    shot_interval: float,
    plot: bool,
    stage: str,
) -> Result:
    """Measure target XYZ for two control states and fit conditional rotations."""
    counts_tuple = _validate_repetition_counts(repetition_counts)
    counts = np.asarray(counts_tuple, dtype=np.float64)
    sequences = [
        PulseSchedule(schedule.labels) if count == 0 else schedule.repeated(count)
        for count in counts_tuple
    ]
    raw_results: dict[str, Result] = {}
    states_by_control: dict[str, NDArray[np.float64]] = {}
    try:
        for state in ("0", "1"):
            raw = exp.state_evolution_tomography(
                sequences=sequences,
                initial_state={control_qubit: state, target_qubit: "0"},
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=plot,
            )
            raw_results[state] = raw
            raw_data = _require_target_tomography(raw, target_qubit)
            states_by_control[state] = np.asarray(
                raw_data[target_qubit], dtype=np.float64
            )
        omega_0, quality_0 = _fit_repeated_rotation(counts, states_by_control["0"])
        omega_1, quality_1 = _fit_repeated_rotation(counts, states_by_control["1"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return _stage_result(
            stage=stage,
            input_params=params,
            proposed_params=params,
            converged=False,
            verified=False,
            status="measurement_or_fit_failed",
            reason=str(exc),
            sweep={"repetition_counts": counts_tuple},
            raw_results=raw_results,
        )
    identity = 0.5 * (omega_0 + omega_1)
    conditional = 0.5 * (omega_0 - omega_1)
    error_angles = dict(
        zip(
            _PAULI_TERMS,
            np.concatenate([identity, conditional]).tolist(),
            strict=True,
        )
    )
    fit_quality = {
        "control_0": quality_0,
        "control_1": quality_1,
        "r2": min(quality_0["r2"], quality_1["r2"]),
    }
    metrics = {**error_angles, "zx_angle": error_angles["ZX"]}
    return _stage_result(
        stage=stage,
        input_params=params,
        proposed_params=params,
        converged=True,
        verified=True,
        status="success",
        metrics_before=metrics,
        metrics_after=metrics,
        fit_quality=fit_quality,
        sweep={"repetition_counts": counts_tuple},
        raw_results=raw_results,
        error_angles=error_angles,
        zx_angle=error_angles["ZX"],
        conditional_rotation_vectors={"0": omega_0, "1": omega_1},
        state_vectors=states_by_control,
    )


def _require_target_tomography(
    result: Result | Mapping[str, object],
    target_qubit: str,
) -> Mapping[str, object]:
    """Return tomography payload that contains the requested target."""
    raw_data = _result_mapping(result)
    if target_qubit not in raw_data:
        raise ValueError(f"Tomography result does not contain target {target_qubit}.")
    return raw_data


def characterize_ecr_gate(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    repetition_counts: Sequence[int] | None = None,
    n_shots: int = CALIBRATION_SHOTS,
    shot_interval: float = DEFAULT_INTERVAL,
    plot: bool = False,
) -> Result:
    """
    Characterize a completed ECR through repeated target XYZ tomography.

    The fit returns one conditional rotation vector for each control state and
    combines them into IX, IY, IZ, ZX, ZY, and ZZ error angles in radians per
    completed gate.
    """
    counts = tuple(range(9)) if repetition_counts is None else repetition_counts
    gate = build_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=params,
    )
    return _characterize_repeated_schedule(
        exp,
        control_qubit,
        target_qubit,
        schedule=gate,
        params=params,
        repetition_counts=counts,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        stage="characterize_ecr_gate",
    )


def _error_angles(
    result: Result,
    *,
    min_r2: float | None = None,
) -> dict[str, float] | None:
    """Return finite repeated-gate error angles from a successful result."""
    if not bool(result.data.get("converged")):
        return None
    if min_r2 is not None:
        fit_quality = result.data.get("fit_quality")
        r2 = (
            _finite_float(fit_quality.get("r2"))
            if isinstance(fit_quality, Mapping)
            else None
        )
        if r2 is None or r2 < min_r2:
            return None
    raw = result.data.get("error_angles")
    if not isinstance(raw, Mapping):
        return None
    try:
        values = {term: float(raw[term]) for term in _PAULI_TERMS}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in values.values()):
        return None
    return values


def _fit_interpolated_zero(
    x: ArrayLike,
    y: ArrayLike,
    *,
    reference: float,
) -> tuple[float | None, dict[str, object]]:
    """Interpolate the nearest zero between adjacent opposite-sign points."""
    x_array = _as_1d_finite(x, name="x")
    y_array = _as_1d_finite(y, name="y")
    if x_array.shape != y_array.shape or len(x_array) < 2:
        raise ValueError("x and y must have matching lengths of at least two.")
    if len(np.unique(x_array)) != len(x_array):
        raise ValueError("x must contain distinct sweep points.")
    if not np.isfinite(reference):
        raise ValueError("reference must be finite.")

    order = np.argsort(x_array)
    x_sorted = x_array[order]
    y_sorted = y_array[order]
    centered_x = x_sorted - np.mean(x_sorted)
    denominator = float(np.sum(centered_x**2))
    global_slope = (
        float(np.sum(centered_x * (y_sorted - np.mean(y_sorted)))) / denominator
    )
    global_intercept = float(np.mean(y_sorted) - global_slope * np.mean(x_sorted))
    predicted = global_slope * x_sorted + global_intercept
    residual_sum = float(np.sum((y_sorted - predicted) ** 2))
    total_sum = float(np.sum((y_sorted - np.mean(y_sorted)) ** 2))
    r2 = 1 - residual_sum / total_sum if total_sum > _EPS else 1.0

    zero_indices = np.flatnonzero(np.isclose(y_sorted, 0.0, rtol=0, atol=_EPS))
    if len(zero_indices):
        index = int(zero_indices[np.argmin(np.abs(x_sorted[zero_indices] - reference))])
        root = float(x_sorted[index])
        return root, {
            "status": "success",
            "r2": r2,
            "root": root,
            "bracket": (root, root),
        }

    brackets = [
        index
        for index in range(len(x_sorted) - 1)
        if y_sorted[index] * y_sorted[index + 1] < 0
    ]
    if not brackets:
        return None, {
            "status": "no_interpolated_root",
            "r2": r2,
            "root": None,
            "bracket": None,
        }
    roots = {
        index: float(
            x_sorted[index]
            - y_sorted[index]
            * (x_sorted[index + 1] - x_sorted[index])
            / (y_sorted[index + 1] - y_sorted[index])
        )
        for index in brackets
    }
    index = min(
        brackets,
        key=lambda item: (
            abs(roots[item] - reference),
            abs(y_sorted[item]) + abs(y_sorted[item + 1]),
        ),
    )
    x0 = float(x_sorted[index])
    x1 = float(x_sorted[index + 1])
    y0 = float(y_sorted[index])
    y1 = float(y_sorted[index + 1])
    slope = (y1 - y0) / (x1 - x0)
    root = roots[index]
    quality: dict[str, object] = {
        "status": "success",
        "r2": r2,
        "root": root,
        "slope": float(slope),
        "bracket": (x0, x1),
    }
    return root, quality


def _measure_ecr_sweep(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    candidates: Sequence[CrGateParameters],
    repetition_counts: Sequence[int],
    settings: CrCalibrationOptions,
) -> list[Result]:
    """Characterize a list of exact candidate ECR gates."""
    return [
        characterize_ecr_gate(
            exp,
            control_qubit,
            target_qubit,
            params=candidate,
            repetition_counts=repetition_counts,
            n_shots=settings.n_shots,
            shot_interval=settings.shot_interval,
            plot=settings.plot,
        )
        for candidate in candidates
    ]


def calibrate_ecr_phase(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    initial_params: CrGateParameters,
    phase_offsets: ArrayLike,
    repetition_counts: Sequence[int] | None = None,
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Calibrate common ECR phase by interpolating the repeated-gate ZY zero.

    CR, cancellation, and rotary IQ vectors are rotated together. A candidate
    is accepted only when the zero is bracketed by measured offsets and a fresh
    verification confirms both improvement and the configured ZY tolerance.
    """
    offsets = _as_1d_finite(phase_offsets, name="phase_offsets")
    if len(offsets) < 2:
        raise ValueError("phase_offsets must contain at least two points.")
    if not np.any(np.isclose(offsets, 0.0, rtol=0, atol=_EPS)):
        raise ValueError(
            "phase_offsets must include 0 to measure the input parameters."
        )
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    counts = (
        settings.repetition_counts
        if repetition_counts is None
        else _validate_repetition_counts(repetition_counts)
    )
    candidates = [
        _rotate_common_phase(initial_params, float(value)) for value in offsets
    ]
    measurements = _measure_ecr_sweep(
        exp,
        control_qubit,
        target_qubit,
        candidates=candidates,
        repetition_counts=counts,
        settings=settings,
    )
    errors = [
        _error_angles(result, min_r2=thresholds.fit_r2) for result in measurements
    ]
    if any(value is None for value in errors):
        return _stage_result(
            stage="calibrate_ecr_phase",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="sweep_measurement_failed",
            reason="At least one phase point did not produce valid tomography.",
            sweep={"phase_offsets": offsets},
            raw_results=measurements,
        )
    zy = np.asarray([cast(dict[str, float], value)["ZY"] for value in errors])
    baseline_index = int(np.argmin(np.abs(offsets)))
    before = cast(dict[str, float], errors[baseline_index])
    if abs(before["ZY"]) <= thresholds.fine_error_angle:
        return _stage_result(
            stage="calibrate_ecr_phase",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=True,
            verified=True,
            status="already_converged",
            metrics_before=before,
            metrics_after=before,
            sweep={"phase_offsets": offsets, "ZY": zy},
            raw_results=measurements,
            phase_correction=0.0,
            verification=measurements[baseline_index],
        )
    root, fit_quality = _fit_interpolated_zero(offsets, zy, reference=0.0)
    if root is None:
        return _stage_result(
            stage="calibrate_ecr_phase",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="no_interpolated_root",
            reason="The measured phase sweep does not bracket a ZY zero.",
            metrics_before=before,
            metrics_after=before,
            fit_quality=fit_quality,
            sweep={"phase_offsets": offsets, "ZY": zy},
            raw_results=measurements,
        )
    candidate = _rotate_common_phase(initial_params, root)
    verification = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=candidate,
        repetition_counts=counts,
        n_shots=settings.verification_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    after = _error_angles(verification, min_r2=thresholds.fit_r2)
    improved = after is not None and abs(after["ZY"]) <= abs(before["ZY"]) + _EPS
    converged = bool(
        improved
        and abs(cast(dict[str, float], after)["ZY"]) <= thresholds.fine_error_angle
    )
    return _stage_result(
        stage="calibrate_ecr_phase",
        input_params=initial_params,
        proposed_params=candidate if improved else initial_params,
        converged=converged,
        verified=bool(improved),
        status="success" if converged else "verification_failed",
        reason=None
        if converged
        else "The proposed phase did not verify within the ZY tolerance.",
        metrics_before=before,
        metrics_after=after or before,
        fit_quality=fit_quality,
        sweep={"phase_offsets": offsets, "ZY": zy},
        raw_results={"sweep": measurements, "verification": verification},
        phase_correction=root,
        verification=verification,
    )


def _scale_cr_and_cancellation(
    params: CrGateParameters,
    scale: float,
) -> CrGateParameters:
    """Scale CR and cancellation tones while leaving rotary independent."""
    return replace(
        params,
        cr_amplitude=params.cr_amplitude * scale,
        cancel_x=params.cancel_x * scale,
        cancel_y=params.cancel_y * scale,
    )


def _zx_rotation_rate_from_angle(
    params: CrGateParameters,
    zx_angle: float,
) -> float:
    """Convert a completed-ECR ZX angle to rate per CR amplitude in GHz."""
    effective_duration = 2 * (params.cr_lobe_duration - params.cr_ramptime)
    return float(zx_angle / (2 * np.pi * effective_duration * params.cr_amplitude))


def calibrate_ecr_angle(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    initial_params: CrGateParameters,
    amplitude_scales: ArrayLike,
    repetition_counts: Sequence[int] | None = None,
    target_angle: float = np.pi / 2,
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Calibrate the completed ECR ZX angle using repeated XYZ tomography.

    Duration is fixed. CR and cancellation amplitudes are scaled together;
    rotary IQ remains independent. Roots outside the measured scale range are
    rejected and no fallback amplitude is applied.
    """
    scales = _as_1d_finite(amplitude_scales, name="amplitude_scales")
    if len(scales) < 2 or np.any(scales <= 0):
        raise ValueError("amplitude_scales must contain at least two positive points.")
    if not np.any(np.isclose(scales, 1.0, rtol=0, atol=_EPS)):
        raise ValueError(
            "amplitude_scales must include 1 to measure the input parameters."
        )
    if not np.isfinite(target_angle):
        raise ValueError("target_angle must be finite.")
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    counts = (
        settings.repetition_counts
        if repetition_counts is None
        else _validate_repetition_counts(repetition_counts)
    )
    candidates = [
        _scale_cr_and_cancellation(initial_params, float(value)) for value in scales
    ]
    measurements = _measure_ecr_sweep(
        exp,
        control_qubit,
        target_qubit,
        candidates=candidates,
        repetition_counts=counts,
        settings=settings,
    )
    errors = [
        _error_angles(result, min_r2=thresholds.fit_r2) for result in measurements
    ]
    if any(value is None for value in errors):
        return _stage_result(
            stage="calibrate_ecr_angle",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="sweep_measurement_failed",
            reason="At least one amplitude point did not produce valid tomography.",
            sweep={"amplitude_scales": scales},
            raw_results=measurements,
        )
    angles = np.asarray([cast(dict[str, float], value)["ZX"] for value in errors])
    angle_errors = angles - target_angle
    baseline_index = int(np.argmin(np.abs(scales - 1)))
    before = cast(dict[str, float], errors[baseline_index])
    if abs(before["ZX"] - target_angle) <= thresholds.fine_angle_error:
        proposed = replace(
            initial_params,
            zx_rotation_rate=_zx_rotation_rate_from_angle(
                initial_params,
                before["ZX"],
            ),
        )
        return _stage_result(
            stage="calibrate_ecr_angle",
            input_params=initial_params,
            proposed_params=proposed,
            converged=True,
            verified=True,
            status="already_converged",
            metrics_before=before,
            metrics_after=before,
            sweep={"amplitude_scales": scales, "ZX": angles},
            raw_results=measurements,
            amplitude_scale=1.0,
            verification=measurements[baseline_index],
        )
    root, fit_quality = _fit_interpolated_zero(scales, angle_errors, reference=1.0)
    if root is None:
        return _stage_result(
            stage="calibrate_ecr_angle",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="no_interpolated_root",
            reason="The measured amplitude sweep does not bracket the target angle.",
            metrics_before=before,
            metrics_after=before,
            fit_quality=fit_quality,
            sweep={"amplitude_scales": scales, "ZX": angles},
            raw_results=measurements,
        )
    candidate = _scale_cr_and_cancellation(initial_params, root)
    verification = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=candidate,
        repetition_counts=counts,
        n_shots=settings.verification_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    after = _error_angles(verification, min_r2=thresholds.fit_r2)
    before_error = abs(before["ZX"] - target_angle)
    improved = (
        after is not None and abs(after["ZX"] - target_angle) <= before_error + _EPS
    )
    converged = bool(
        improved
        and abs(cast(dict[str, float], after)["ZX"] - target_angle)
        <= thresholds.fine_angle_error
    )
    proposed = candidate if improved else initial_params
    if improved:
        observed_rate = _zx_rotation_rate_from_angle(
            proposed,
            cast(dict[str, float], after)["ZX"],
        )
        proposed = replace(proposed, zx_rotation_rate=float(observed_rate))
    return _stage_result(
        stage="calibrate_ecr_angle",
        input_params=initial_params,
        proposed_params=proposed,
        converged=converged,
        verified=bool(improved),
        status="success" if converged else "verification_failed",
        reason=None
        if converged
        else "The proposed amplitude did not verify within the ZX-angle tolerance.",
        metrics_before=before,
        metrics_after=after or before,
        fit_quality=fit_quality,
        sweep={"amplitude_scales": scales, "ZX": angles},
        raw_results={"sweep": measurements, "verification": verification},
        amplitude_scale=root,
        verification=verification,
    )


def _characterize_un_echoed_gate(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    repetition_counts: Sequence[int],
    settings: CrCalibrationOptions,
) -> Result:
    """Characterize repeated un-echoed CR primitives without rotary drive."""
    primitive = _build_cr_schedule(
        exp,
        control_qubit,
        target_qubit,
        params=params,
        echo=False,
        include_rotary=False,
        include_local_frames=False,
    )
    return _characterize_repeated_schedule(
        exp,
        control_qubit,
        target_qubit,
        schedule=primitive,
        params=params,
        repetition_counts=repetition_counts,
        n_shots=settings.n_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
        stage="characterize_un_echoed_cr",
    )


def _calibrate_one_cancellation_axis(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    axis: str,
    offsets: NDArray[np.float64],
    repetition_counts: Sequence[int],
    settings: CrCalibrationOptions,
    min_fit_r2: float,
    error_tolerance: float,
) -> tuple[CrGateParameters, dict[str, object]]:
    """Calibrate one cancellation Cartesian component through an IX/IY zero."""
    field = "cancel_x" if axis == "IX" else "cancel_y"
    candidates = [
        replace(params, **{field: getattr(params, field) + float(value)})
        for value in offsets
    ]
    measurements = [
        _characterize_un_echoed_gate(
            exp,
            control_qubit,
            target_qubit,
            params=candidate,
            repetition_counts=repetition_counts,
            settings=settings,
        )
        for candidate in candidates
    ]
    errors = [_error_angles(result, min_r2=min_fit_r2) for result in measurements]
    if any(value is None for value in errors):
        return params, {
            "axis": axis,
            "status": "sweep_measurement_failed",
            "converged": False,
            "measurements": measurements,
        }
    values = np.asarray([cast(dict[str, float], value)[axis] for value in errors])
    baseline_index = int(np.argmin(np.abs(offsets)))
    before = cast(dict[str, float], errors[baseline_index])
    if abs(before[axis]) <= error_tolerance:
        return params, {
            "axis": axis,
            "status": "already_converged",
            "converged": True,
            "root": 0.0,
            "before": before,
            "after": before,
            "measurements": measurements,
            "verification": measurements[baseline_index],
            "values": values,
        }
    root, fit_quality = _fit_interpolated_zero(offsets, values, reference=0.0)
    if root is None:
        return params, {
            "axis": axis,
            "status": "no_interpolated_root",
            "converged": False,
            "measurements": measurements,
            "values": values,
            "fit_quality": fit_quality,
        }
    candidate = replace(params, **{field: getattr(params, field) + root})
    verification = _characterize_un_echoed_gate(
        exp,
        control_qubit,
        target_qubit,
        params=candidate,
        repetition_counts=repetition_counts,
        settings=replace(settings, n_shots=settings.verification_shots),
    )
    after = _error_angles(verification, min_r2=min_fit_r2)
    improved = after is not None and abs(after[axis]) <= abs(before[axis]) + _EPS
    return (candidate if improved else params), {
        "axis": axis,
        "status": "success" if improved else "verification_failed",
        "converged": bool(improved),
        "root": root,
        "before": before,
        "after": after,
        "measurements": measurements,
        "verification": verification,
        "values": values,
        "fit_quality": fit_quality,
    }


def calibrate_un_echoed_cancellation(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    initial_params: CrGateParameters,
    cancel_x_offsets: ArrayLike,
    cancel_y_offsets: ArrayLike,
    repetition_counts: Sequence[int] | None = None,
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Sequentially null un-echoed IX and IY with cancellation IQ sweeps.

    Rotary drive and local frame corrections are disabled for these primitive
    measurements. The returned update is not persisted, and a completed ECR
    angle calibration must be rerun after this stage.
    """
    x_offsets = _as_1d_finite(cancel_x_offsets, name="cancel_x_offsets")
    y_offsets = _as_1d_finite(cancel_y_offsets, name="cancel_y_offsets")
    if len(x_offsets) < 2 or len(y_offsets) < 2:
        raise ValueError("Each cancellation offset sweep needs at least two points.")
    if not np.any(np.isclose(x_offsets, 0.0, rtol=0, atol=_EPS)):
        raise ValueError(
            "cancel_x_offsets must include 0 to measure the input parameters."
        )
    if not np.any(np.isclose(y_offsets, 0.0, rtol=0, atol=_EPS)):
        raise ValueError(
            "cancel_y_offsets must include 0 to measure the input parameters."
        )
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    counts = (
        settings.repetition_counts
        if repetition_counts is None
        else _validate_repetition_counts(repetition_counts)
    )
    after_x, x_detail = _calibrate_one_cancellation_axis(
        exp,
        control_qubit,
        target_qubit,
        params=initial_params,
        axis="IX",
        offsets=x_offsets,
        repetition_counts=counts,
        settings=settings,
        min_fit_r2=thresholds.fit_r2,
        error_tolerance=thresholds.fine_error_angle,
    )
    if not bool(x_detail["converged"]):
        return _stage_result(
            stage="calibrate_un_echoed_cancellation",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status=cast(str, x_detail["status"]),
            reason="The cancellation-X update could not be verified.",
            sweep={"cancel_x_offsets": x_offsets, "cancel_y_offsets": y_offsets},
            raw_results={"x": x_detail},
            requires_ecr_angle_recalibration=True,
        )
    after_y, y_detail = _calibrate_one_cancellation_axis(
        exp,
        control_qubit,
        target_qubit,
        params=after_x,
        axis="IY",
        offsets=y_offsets,
        repetition_counts=counts,
        settings=settings,
        min_fit_r2=thresholds.fit_r2,
        error_tolerance=thresholds.fine_error_angle,
    )
    if not bool(y_detail["converged"]):
        return _stage_result(
            stage="calibrate_un_echoed_cancellation",
            input_params=initial_params,
            proposed_params=after_x,
            converged=False,
            verified=False,
            status=cast(str, y_detail["status"]),
            reason="The cancellation-Y update could not be verified.",
            sweep={"cancel_x_offsets": x_offsets, "cancel_y_offsets": y_offsets},
            raw_results={"x": x_detail, "y": y_detail},
            requires_ecr_angle_recalibration=True,
        )
    verification = _characterize_un_echoed_gate(
        exp,
        control_qubit,
        target_qubit,
        params=after_y,
        repetition_counts=counts,
        settings=replace(settings, n_shots=settings.verification_shots),
    )
    errors = _error_angles(verification, min_r2=thresholds.fit_r2)
    converged = bool(
        errors is not None
        and abs(errors["IX"]) <= thresholds.fine_error_angle
        and abs(errors["IY"]) <= thresholds.fine_error_angle
    )
    return _stage_result(
        stage="calibrate_un_echoed_cancellation",
        input_params=initial_params,
        proposed_params=after_y,
        converged=converged,
        verified=errors is not None,
        status="success" if converged else "final_verification_failed",
        reason=None
        if converged
        else "Final IX/IY values did not satisfy the configured tolerance.",
        metrics_after=errors or {},
        sweep={"cancel_x_offsets": x_offsets, "cancel_y_offsets": y_offsets},
        raw_results={"x": x_detail, "y": y_detail, "verification": verification},
        verification=verification,
        requires_ecr_angle_recalibration=True,
    )


def _rotary_objective(
    errors: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Return a weighted squared residual objective for rotary optimization."""
    return float(
        sum(float(weights[term]) * float(errors[term]) ** 2 for term in weights)
    )


def optimize_ecr_rotary(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    initial_params: CrGateParameters,
    rotary_x_values: ArrayLike,
    rotary_y_values: ArrayLike | None = None,
    repetition_counts: Sequence[int] | None = None,
    objective_weights: Mapping[str, float] | None = None,
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Minimize echoed residuals over measured rotary IQ candidates.

    This is a safe repeated-ECR error-amplification baseline, not the complete
    HEAT sequence from the literature. The payload marks the dedicated HEAT
    experiment as `not_run`; no unmeasured optimum is interpolated.
    """
    x_values = _as_1d_finite(rotary_x_values, name="rotary_x_values")
    y_values = (
        np.asarray([initial_params.rotary_y], dtype=np.float64)
        if rotary_y_values is None
        else _as_1d_finite(rotary_y_values, name="rotary_y_values")
    )
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    counts = (
        settings.repetition_counts
        if repetition_counts is None
        else _validate_repetition_counts(repetition_counts)
    )
    weights = dict(objective_weights or {"IY": 1.0, "IZ": 1.0, "ZY": 1.0, "ZZ": 1.0})
    if not weights or any(term not in _PAULI_TERMS for term in weights):
        raise ValueError("objective_weights must use CR Pauli-term keys.")
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("objective weights must be finite and non-negative.")
    if not any(value > 0 for value in weights.values()):
        raise ValueError("At least one objective weight must be positive.")

    baseline_measurement = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=initial_params,
        repetition_counts=counts,
        n_shots=settings.n_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    baseline_errors = _error_angles(baseline_measurement, min_r2=thresholds.fit_r2)
    if baseline_errors is None:
        return _stage_result(
            stage="optimize_ecr_rotary",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="initial_characterization_failed",
            reason=cast(str | None, baseline_measurement.data.get("reason")),
            raw_results={"baseline": baseline_measurement},
            heat={
                "status": "not_run",
                "reason": "Dedicated HEAT sequences are not implemented by this baseline.",
            },
        )
    baseline_objective = _rotary_objective(baseline_errors, weights)
    if math.sqrt(baseline_objective) <= thresholds.fine_error_angle:
        return _stage_result(
            stage="optimize_ecr_rotary",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=True,
            verified=True,
            status="already_converged",
            metrics_before={"objective": baseline_objective, **baseline_errors},
            metrics_after={"objective": baseline_objective, **baseline_errors},
            raw_results={"baseline": baseline_measurement},
            heat={
                "status": "not_run",
                "reason": "Dedicated HEAT sequences are not implemented by this baseline.",
            },
            method="repeated_ecr_error_amplification",
        )
    points: list[dict[str, object]] = []
    for x_value in x_values:
        for y_value in y_values:
            candidate = replace(
                initial_params,
                rotary_x=float(x_value),
                rotary_y=float(y_value),
            )
            if candidate == initial_params:
                measurement = baseline_measurement
                errors = baseline_errors
            else:
                measurement = characterize_ecr_gate(
                    exp,
                    control_qubit,
                    target_qubit,
                    params=candidate,
                    repetition_counts=counts,
                    n_shots=settings.n_shots,
                    shot_interval=settings.shot_interval,
                    plot=settings.plot,
                )
                errors = _error_angles(measurement, min_r2=thresholds.fit_r2)
            objective = (
                _rotary_objective(errors, weights)
                if errors is not None
                else float("inf")
            )
            points.append(
                {
                    "params": candidate,
                    "errors": errors,
                    "objective": objective,
                    "measurement": measurement,
                }
            )
    valid_points = [
        point for point in points if _finite_float(point["objective"]) is not None
    ]
    if not valid_points:
        return _stage_result(
            stage="optimize_ecr_rotary",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="sweep_measurement_failed",
            reason="No rotary candidate produced a valid repeated-gate fit.",
            metrics_before={"objective": baseline_objective},
            sweep={"rotary_x_values": x_values, "rotary_y_values": y_values},
            raw_results={"baseline": baseline_measurement, "points": points},
            heat={
                "status": "not_run",
                "reason": "Dedicated HEAT sequences are not implemented by this baseline.",
            },
        )
    best = min(valid_points, key=lambda point: cast(float, point["objective"]))
    best_objective = cast(float, best["objective"])
    if best_objective >= baseline_objective - _EPS:
        return _stage_result(
            stage="optimize_ecr_rotary",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="no_improvement",
            reason="No measured rotary candidate improved the baseline objective.",
            metrics_before={"objective": baseline_objective, **baseline_errors},
            metrics_after={"objective": baseline_objective, **baseline_errors},
            sweep={"rotary_x_values": x_values, "rotary_y_values": y_values},
            raw_results={"baseline": baseline_measurement, "points": points},
            heat={
                "status": "not_run",
                "reason": "Dedicated HEAT sequences are not implemented by this baseline.",
            },
            method="repeated_ecr_error_amplification",
        )
    candidate = cast(CrGateParameters, best["params"])
    verification = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=candidate,
        repetition_counts=counts,
        n_shots=settings.verification_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    after = _error_angles(verification, min_r2=thresholds.fit_r2)
    after_objective = (
        _rotary_objective(after, weights) if after is not None else float("inf")
    )
    verified = after_objective < baseline_objective - _EPS
    converged = verified and math.sqrt(after_objective) <= thresholds.fine_error_angle
    return _stage_result(
        stage="optimize_ecr_rotary",
        input_params=initial_params,
        proposed_params=candidate if verified else initial_params,
        converged=converged,
        verified=verified,
        status="success" if converged else "verification_failed",
        reason=None
        if converged
        else "Rotary improvement did not verify within the residual tolerance.",
        metrics_before={"objective": baseline_objective, **baseline_errors},
        metrics_after={"objective": after_objective, **(after or baseline_errors)},
        sweep={"rotary_x_values": x_values, "rotary_y_values": y_values},
        raw_results={
            "baseline": baseline_measurement,
            "points": points,
            "verification": verification,
        },
        heat={
            "status": "not_run",
            "reason": "Dedicated HEAT sequences are not implemented by this baseline.",
        },
        method="repeated_ecr_error_amplification",
        requires_phase_recalibration=verified,
        requires_angle_recalibration=verified,
    )


def calibrate_ecr_local_z(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    initial_params: CrGateParameters,
    repetition_counts: Sequence[int] | None = None,
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Apply and verify a target local-Z correction from repeated ECR tomography.

    The current target-only tomography cannot identify the control-qubit local
    Z phase. That component is left unchanged and explicitly marked `not_run`;
    the stage therefore reports `partial` even when target IZ is corrected.
    """
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    counts = (
        settings.repetition_counts
        if repetition_counts is None
        else _validate_repetition_counts(repetition_counts)
    )
    before_measurement = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=initial_params,
        repetition_counts=counts,
        n_shots=settings.n_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    before = _error_angles(before_measurement, min_r2=thresholds.fit_r2)
    if before is None:
        return _stage_result(
            stage="calibrate_ecr_local_z",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=False,
            status="initial_characterization_failed",
            reason=cast(str | None, before_measurement.data.get("reason")),
            raw_results={"before": before_measurement},
            components={
                "target": {"status": "not_run", "verified": False},
                "control": {"status": "not_run", "verified": False},
            },
        )
    if abs(before["IZ"]) <= thresholds.fine_error_angle:
        components = {
            "target": {
                "supported": True,
                "status": "already_converged",
                "verified": True,
                "correction": 0.0,
            },
            "control": {
                "supported": False,
                "status": "not_run",
                "verified": False,
                "reason": "Control local-Z requires control-qubit tomography or Ramsey data.",
            },
        }
        return _stage_result(
            stage="calibrate_ecr_local_z",
            input_params=initial_params,
            proposed_params=initial_params,
            converged=False,
            verified=True,
            status="partial",
            reason="Target local-Z is within tolerance; control local-Z was not measured.",
            metrics_before=before,
            metrics_after=before,
            raw_results={"before": before_measurement},
            components=components,
            verification=before_measurement,
        )
    candidate = replace(
        initial_params,
        target_frame_z=initial_params.target_frame_z - before["IZ"],
    )
    verification = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=candidate,
        repetition_counts=counts,
        n_shots=settings.verification_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    after = _error_angles(verification, min_r2=thresholds.fit_r2)
    target_verified = bool(
        after is not None
        and abs(after["IZ"]) <= abs(before["IZ"]) + _EPS
        and abs(after["IZ"]) <= thresholds.fine_error_angle
    )
    components = {
        "target": {
            "supported": True,
            "status": "success" if target_verified else "verification_failed",
            "verified": target_verified,
            "correction": -before["IZ"],
        },
        "control": {
            "supported": False,
            "status": "not_run",
            "verified": False,
            "reason": "Control local-Z requires control-qubit tomography or Ramsey data.",
        },
    }
    return _stage_result(
        stage="calibrate_ecr_local_z",
        input_params=initial_params,
        proposed_params=candidate if target_verified else initial_params,
        converged=False,
        verified=target_verified,
        status="partial" if target_verified else "verification_failed",
        reason=(
            "Target local-Z verified; control local-Z was not measured."
            if target_verified
            else "The target local-Z correction did not verify."
        ),
        metrics_before=before,
        metrics_after=after or before,
        raw_results={"before": before_measurement, "verification": verification},
        components=components,
        verification=verification,
    )


def _normalize_initial_states(initial_states: Sequence[str]) -> tuple[str, ...]:
    """Validate computational two-qubit initial-state labels."""
    states = tuple(initial_states)
    if not states:
        raise ValueError("initial_states must not be empty.")
    if any(len(state) != 2 or set(state) - {"0", "1"} for state in states):
        raise ValueError("initial_states must contain two-character 0/1 labels.")
    return states


def _classified_probabilities(
    measurement: object, target: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract classified populations and standard deviations for one target."""
    data = getattr(measurement, "data", None)
    if not isinstance(data, Mapping) or target not in data:
        raise ValueError(f"Measurement result does not contain {target}.")
    target_data = data[target]
    probabilities = np.asarray(
        getattr(target_data, "probabilities", None), dtype=np.float64
    )
    deviations = np.asarray(
        getattr(target_data, "standard_deviations", np.zeros_like(probabilities)),
        dtype=np.float64,
    )
    if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError(f"Classified probabilities for {target} are invalid.")
    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(f"Classified probabilities for {target} are unphysical.")
    if (
        deviations.shape != probabilities.shape
        or not np.all(np.isfinite(deviations))
        or np.any(deviations < 0.0)
    ):
        raise ValueError(f"Probability deviations for {target} are invalid.")
    return probabilities, deviations


def measure_ecr_leakage(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    repetition_counts: Sequence[int] = (1, 2, 4, 8),
    initial_states: Sequence[str] = ("00", "01", "10", "11"),
    tolerances: CrCalibrationTolerances | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Measure qutrit leakage versus completed-ECR repetition count.

    The third classifier population is interpreted as `P_f`. If either qubit
    returns fewer than three populations, the function fails safely and marks
    qutrit classification as unsupported.
    """
    counts = _as_integer_counts(repetition_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("repetition_counts must contain positive integers.")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("repetition_counts must be sorted and distinct.")
    states = _normalize_initial_states(initial_states)
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    gate = build_ecr_gate(exp, control_qubit, target_qubit, params=params)
    records: dict[str, dict[str, list[float]]] = {}
    raw_results: dict[str, list[object]] = {}
    try:
        for state in states:
            records[state] = {
                control_qubit: [],
                target_qubit: [],
                f"{control_qubit}_std": [],
                f"{target_qubit}_std": [],
            }
            raw_results[state] = []
            for count in counts:
                measurement = exp.measure(
                    gate.repeated(count),
                    initial_states={control_qubit: state[0], target_qubit: state[1]},
                    mode="single",
                    enable_dsp_classification=True,
                    n_shots=settings.n_shots,
                    shot_interval=settings.shot_interval,
                    plot=False,
                )
                raw_results[state].append(measurement)
                for qubit in (control_qubit, target_qubit):
                    probabilities, deviations = _classified_probabilities(
                        measurement, qubit
                    )
                    if len(probabilities) < 3:
                        return _stage_result(
                            stage="measure_ecr_leakage",
                            input_params=params,
                            proposed_params=params,
                            converged=False,
                            verified=False,
                            supported=False,
                            status="qutrit_classifier_required",
                            reason=f"{qubit} returned fewer than three populations.",
                            sweep={
                                "repetition_counts": counts,
                                "initial_states": states,
                            },
                            raw_results=raw_results,
                        )
                    records[state][qubit].append(float(probabilities[2]))
                    records[state][f"{qubit}_std"].append(float(deviations[2]))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return _stage_result(
            stage="measure_ecr_leakage",
            input_params=params,
            proposed_params=params,
            converged=False,
            verified=False,
            status="measurement_failed",
            reason=str(exc),
            sweep={"repetition_counts": counts, "initial_states": states},
            raw_results=raw_results,
        )
    slopes: dict[str, dict[str, float]] = {}
    per_gate = 0.0
    x = np.asarray(counts, dtype=np.float64)
    for state, state_records in records.items():
        slopes[state] = {}
        for qubit in (control_qubit, target_qubit):
            y = np.asarray(state_records[qubit], dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float(y[0] / x[0])
            slope = max(0.0, slope)
            slopes[state][qubit] = slope
            per_gate = max(per_gate, slope)
    passed = per_gate <= thresholds.leakage_per_gate
    return _stage_result(
        stage="measure_ecr_leakage",
        input_params=params,
        proposed_params=params,
        converged=passed,
        verified=True,
        status="success" if passed else "leakage_above_tolerance",
        reason=None if passed else "Estimated leakage per gate exceeds tolerance.",
        metrics_after={"leakage_per_gate": per_gate},
        uncertainties={},
        sweep={"repetition_counts": counts, "initial_states": states},
        raw_results=raw_results,
        populations=records,
        slopes=slopes,
        leakage_per_gate=per_gate,
        passed=passed,
    )


def _not_run(reason: str, *, required: bool) -> dict[str, object]:
    """Return a standardized validation component that was not executed."""
    return {
        "status": "not_run",
        "passed": not required,
        "required": required,
        "reason": reason,
    }


def _extract_gate_error(
    result: Result | Mapping[str, object], cr_label: str
) -> float | None:
    """Extract a finite gate error from an IRB result payload."""
    data = _result_mapping(result)
    entry: object = data.get(cr_label, data)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("gate_error")
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def validate_ecr_gate(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    params: CrGateParameters,
    tolerances: CrCalibrationTolerances | None = None,
    spectator_qubits: Sequence[str] = (),
    run_leakage: bool = True,
    run_bell_tomography: bool = True,
    run_irb: bool = True,
    waive_control_local_z: bool = False,
    repetition_counts: Sequence[int] | None = None,
    options: CrCalibrationOptions | None = None,
) -> Result:
    """
    Validate an exact ECR candidate without updating calibration storage.

    Repeated tomography is always required. Leakage, Bell tomography, and IRB
    are required when enabled. Spectator-aware completed-ECR validation has no
    standardized public primitive yet; requesting spectators therefore marks
    validation incomplete rather than silently passing.
    """
    settings = options or CrCalibrationOptions()
    thresholds = tolerances or CrCalibrationTolerances()
    counts = (
        settings.repetition_counts
        if repetition_counts is None
        else _validate_repetition_counts(repetition_counts)
    )
    cr_label = f"{control_qubit}-{target_qubit}"
    gate = build_ecr_gate(exp, control_qubit, target_qubit, params=params)
    characterization = characterize_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=params,
        repetition_counts=counts,
        n_shots=settings.verification_shots,
        shot_interval=settings.shot_interval,
        plot=settings.plot,
    )
    errors = _error_angles(characterization, min_r2=thresholds.fit_r2)
    if errors is None:
        repeated_criterion = {
            "status": "failed",
            "passed": False,
            "required": True,
            "reason": characterization.data.get("reason"),
        }
    else:
        angle_error = abs(errors["ZX"] - np.pi / 2)
        residuals = {term: abs(errors[term]) for term in ("ZY", "IX", "IY", "IZ", "ZZ")}
        passed = angle_error <= thresholds.fine_angle_error and all(
            value <= thresholds.fine_error_angle for value in residuals.values()
        )
        repeated_criterion = {
            "status": "success" if passed else "outside_tolerance",
            "passed": passed,
            "required": True,
            "angle_error": angle_error,
            "residuals": residuals,
            "fit_r2": cast(Mapping[str, object], characterization.data["fit_quality"])[
                "r2"
            ],
        }

    if waive_control_local_z:
        control_local_z_criterion = {
            "status": "waived",
            "passed": True,
            "required": True,
            "reason": (
                "Caller explicitly accepted that target-only tomography does not "
                "measure the control ZI/local-Z phase."
            ),
        }
    else:
        control_local_z_criterion = _not_run(
            "Control ZI/local-Z requires an independent control Ramsey/tomography measurement; "
            "set waive_control_local_z=True only after an external check.",
            required=True,
        )
    rotation_rate_criterion = {
        "status": "success"
        if params.zx_rotation_rate is not None
        else "missing_measured_value",
        "passed": params.zx_rotation_rate is not None,
        "required": True,
        "value": params.zx_rotation_rate,
        "reason": (
            None
            if params.zx_rotation_rate is not None
            else "Run scouting or ECR angle calibration before final validation."
        ),
    }

    leakage_result: Result | None = None
    if run_leakage:
        leakage_result = measure_ecr_leakage(
            exp,
            control_qubit,
            target_qubit,
            params=params,
            tolerances=thresholds,
            options=settings,
        )
        leakage_criterion = {
            "status": leakage_result.data.get("status"),
            "passed": bool(leakage_result.data.get("passed", False)),
            "required": True,
            "leakage_per_gate": leakage_result.data.get("leakage_per_gate"),
            "reason": leakage_result.data.get("reason"),
        }
    else:
        leakage_criterion = _not_run("Leakage validation was disabled.", required=False)

    bell_result: object = None
    if run_bell_tomography:
        bell_method = getattr(exp, "bell_state_tomography", None)
        if not callable(bell_method):
            bell_criterion = _not_run(
                "The experiment does not expose bell_state_tomography.", required=True
            )
        else:
            try:
                bell_result = bell_method(
                    control_qubit,
                    target_qubit,
                    zx90=gate,
                    n_shots=settings.verification_shots,
                    shot_interval=settings.shot_interval,
                    plot=settings.plot,
                    save_image=False,
                )
                bell_data = _result_mapping(cast(Any, bell_result))
                fidelity_value = bell_data.get("fidelity")
                fidelity = _finite_float(fidelity_value)
                if fidelity is None:
                    fidelity_is_physical = False
                    bell_passed = False
                else:
                    fidelity_is_physical = 0.0 <= fidelity <= 1.0
                    bell_passed = (
                        fidelity_is_physical and fidelity >= thresholds.bell_fidelity
                    )
                if not fidelity_is_physical:
                    bell_status = "invalid_result"
                elif bell_passed:
                    bell_status = "success"
                else:
                    bell_status = "outside_tolerance"
                bell_criterion = {
                    "status": bell_status,
                    "passed": bell_passed,
                    "required": True,
                    "fidelity": fidelity,
                }
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                bell_criterion = {
                    "status": "measurement_failed",
                    "passed": False,
                    "required": True,
                    "reason": str(exc),
                }
    else:
        bell_criterion = _not_run("Bell tomography was disabled.", required=False)

    irb_result: object = None
    if run_irb:
        irb_method = getattr(exp, "interleaved_randomized_benchmarking", None)
        if not callable(irb_method):
            irb_criterion = _not_run(
                "The experiment does not expose interleaved randomized benchmarking.",
                required=True,
            )
        else:
            try:
                irb_result = irb_method(
                    targets=cr_label,
                    interleaved_clifford="ZX90",
                    interleaved_waveform={cr_label: gate},
                    zx90={cr_label: gate},
                    n_shots=settings.verification_shots,
                    shot_interval=settings.shot_interval,
                    plot=settings.plot,
                    save_image=False,
                )
                gate_error = _extract_gate_error(cast(Any, irb_result), cr_label)
                if gate_error is None:
                    gate_error_is_physical = False
                    irb_passed = False
                else:
                    gate_error_is_physical = 0.0 <= gate_error <= 1.0
                    irb_passed = (
                        gate_error_is_physical and gate_error <= thresholds.irb_error
                    )
                if not gate_error_is_physical:
                    irb_status = "invalid_result"
                elif irb_passed:
                    irb_status = "success"
                else:
                    irb_status = "outside_tolerance"
                irb_criterion = {
                    "status": irb_status,
                    "passed": irb_passed,
                    "required": True,
                    "gate_error": gate_error,
                }
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                irb_criterion = {
                    "status": "measurement_failed",
                    "passed": False,
                    "required": True,
                    "reason": str(exc),
                }
    else:
        irb_criterion = _not_run("IRB was disabled.", required=False)

    spectators = tuple(spectator_qubits)
    if spectators:
        spectator_criterion = _not_run(
            "Completed-ECR spectator validation has no standardized public primitive.",
            required=True,
        )
        spectator_criterion["qubits"] = spectators
    else:
        spectator_criterion = _not_run(
            "No spectator qubits were requested.", required=False
        )
        spectator_criterion["qubits"] = spectators

    criteria = {
        "repeated_tomography": repeated_criterion,
        "control_local_z": control_local_z_criterion,
        "zx_rotation_rate": rotation_rate_criterion,
        "leakage": leakage_criterion,
        "bell_tomography": bell_criterion,
        "irb": irb_criterion,
        "spectators": spectator_criterion,
    }
    passed = all(
        bool(criterion["passed"])
        for criterion in criteria.values()
        if bool(criterion["required"])
    )
    waveform_fingerprint = _gate_fingerprint(gate)
    token = _validation_token(params, gate)
    return _stage_result(
        stage="validate_ecr_gate",
        input_params=params,
        proposed_params=params,
        converged=passed,
        verified=passed,
        status="success" if passed else "validation_failed",
        reason=None if passed else "One or more required validation criteria failed.",
        metrics_after=errors or {},
        raw_results={
            "characterization": characterization,
            "leakage": leakage_result,
            "bell_tomography": bell_result,
            "irb": irb_result,
        },
        passed=passed,
        committable=passed,
        criteria=criteria,
        residual_errors=errors,
        leakage=leakage_result,
        bell_tomography=bell_result,
        irb=irb_result,
        spectators=spectator_criterion,
        validated_params=params,
        validation_token=token,
        waveform_fingerprint=waveform_fingerprint,
        cr_label=cr_label,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
    )


def _stored_cr_parameters(params: CrGateParameters, cr_label: str) -> dict[str, object]:
    """Convert Cartesian candidate parameters to the persisted CR schema."""
    cancellation = params.cancel_complex
    if params.zx_rotation_rate is None:
        raise ValueError(
            "zx_rotation_rate must be measured before CR calibration is committed."
        )
    return {
        "target": cr_label,
        "duration": params.cr_lobe_duration,
        "ramptime": params.cr_ramptime,
        "cr_amplitude": params.cr_amplitude,
        "cr_phase": params.cr_phase,
        "cr_beta": params.cr_beta,
        "cr_detuning": params.cr_detuning,
        "cancel_amplitude": abs(cancellation),
        "cancel_phase": float(np.angle(cancellation)),
        "cancel_beta": params.cancel_beta,
        "rotary_amplitude": params.rotary_x,
        "rotary_y": params.rotary_y,
        "control_frame_z": params.control_frame_z,
        "target_frame_z": params.target_frame_z,
        "zx_rotation_rate": params.zx_rotation_rate,
    }


def commit_cr_calibration(
    exp: Experiment,
    validation_result: Result | Mapping[str, object],
) -> Result:
    """
    Commit the exact parameters bound to a final validation result.

    This is the only public function in the module that mutates calibration
    state. Failed, stale, or relabeled validations are rejected. The update is
    intentionally in-memory only because the calibration-note writer does not
    provide an atomic, read-back-verified persistence contract. Call
    `exp.calib_note.save()` explicitly after inspecting the committed entry.
    """
    data = _result_mapping(validation_result)
    if data.get("stage") != "validate_ecr_gate":
        raise ValueError("commit requires a validate_ecr_gate result.")
    params = data.get("validated_params")
    if not isinstance(params, CrGateParameters):
        raise TypeError("Validation result does not contain CrGateParameters.")
    control_qubit = data.get("control_qubit")
    target_qubit = data.get("target_qubit")
    if not isinstance(control_qubit, str) or not isinstance(target_qubit, str):
        raise TypeError("Validation result does not identify the calibrated qubits.")
    realized_gate = build_ecr_gate(
        exp,
        control_qubit,
        target_qubit,
        params=params,
    )
    expected_token = _validation_token(params, realized_gate)
    if data.get("validation_token") != expected_token:
        raise ValueError(
            "Validation token does not match the parameters and realized waveform."
        )
    if not bool(data.get("passed")) or not bool(data.get("committable")):
        raise ValueError("Calibration did not pass validation and cannot be committed.")
    ctx = getattr(exp, "ctx", None)
    note = getattr(ctx, "calib_note", None)
    update = getattr(note, "update_cr_param", None)
    if not callable(update):
        raise TypeError("The experiment does not expose a writable calibration note.")
    validation_label = data.get("cr_label")
    if not isinstance(validation_label, str) or not validation_label:
        raise ValueError("Validation result does not contain a CR target label.")
    cr_label = f"{control_qubit}-{target_qubit}"
    if validation_label != cr_label:
        raise ValueError(
            "Validation CR label does not match the validated control and target qubits."
        )
    stored = _stored_cr_parameters(params, cr_label)
    get = getattr(note, "get_cr_param", None)
    previous = copy.deepcopy(get(cr_label) if callable(get) else None)
    remove = getattr(note, "remove_cr_param", None)
    if not callable(remove):
        raise TypeError(
            "The experiment does not expose removable CR calibration entries."
        )
    remove(cr_label)
    try:
        update(cr_label, stored)
    except Exception:
        if previous is not None:
            update(cr_label, previous)
        raise
    return _stage_result(
        stage="commit_cr_calibration",
        input_params=params,
        proposed_params=params,
        converged=True,
        verified=True,
        status="success",
        raw_results={"previous": previous, "stored": stored},
        committed=True,
        cr_label=cr_label,
        stored_params=stored,
    )
