"""SRRE waveform generation and discrete self-refocusing analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import spherical_jn

from qubex.pulse import Arbitrary

__all__ = [
    "SrreMoments",
    "SrrePrediction",
    "calculate_srre_moments",
    "predict_srre_amplitude",
    "srre_waveform",
]

_SAMPLING_GRID_TOLERANCE = 1e-9
_ROOT_SCAN_POINTS = 2049
_NONTRIVIAL_AMPLITUDE_TOLERANCE = 1e-12
_ROOT_VALUE_TOLERANCE = 1e-12
_ROOT_RESIDUAL_TOLERANCE = 1e-9
_RABI_SYMMETRY_RTOL = 1e-10
_RABI_SYMMETRY_ATOL = 1e-12


@dataclass(frozen=True)
class SrreMoments:
    """
    Store discrete self-refocusing moments for one SRRE block.

    Parameters
    ----------
    f0 : complex
        Dimensionless zeroth modulation moment.
    f1 : complex
        Dimensionless centered first modulation moment.
    positive_lobe_angle : float
        Rotation accumulated by the positive lobe in radians.
    """

    f0: complex
    f1: complex
    positive_lobe_angle: float


@dataclass(frozen=True)
class SrrePrediction:
    """
    Store a predicted first non-trivial SRRE amplitude root.

    Parameters
    ----------
    amplitude : float
        Predicted peak hardware amplitude.
    rabi_rate : float
        Peak Rabi rate at `amplitude` in GHz.
    positive_lobe_angle : float
        Rotation accumulated by the positive lobe in radians.
    phi_pred : float
        Predicted modulation phase in radians.
    f0 : complex
        Zeroth modulation moment at the predicted root.
    f1 : complex
        Centered first modulation moment at the predicted root.
    root_bracket : tuple[float, float]
        Scan interval containing the predicted root.
    """

    amplitude: float
    rabi_rate: float
    positive_lobe_angle: float
    phi_pred: float
    f0: complex
    f1: complex
    root_bracket: tuple[float, float]


@dataclass(frozen=True)
class _SrreGeometry:
    block_duration: float
    sampling_period: float
    lobe_samples: int
    ramp_samples: int


def srre_waveform(
    *,
    block_duration: float,
    ramp_time: float,
    amplitude: float,
    phase: float = 0.0,
    sampling_period: float,
) -> Arbitrary:
    """
    Build one sampled raised-cosine flat-top SRRE block.

    Parameters
    ----------
    block_duration : float
        Duration of the complete positive/negative block in ns. It must contain
        an even number of samples.
    ramp_time : float
        Ramp-up and ramp-down duration of each lobe in ns. Zero produces
        rectangular lobes.
    amplitude : float
        Signed peak hardware amplitude in the range [-1, 1].
    phase : float, optional
        Common rotary-axis phase in radians. Defaults to `0.0`.
    sampling_period : float
        AWG sampling period in ns.

    Returns
    -------
    Arbitrary
        One waveform containing the positive lobe followed by its exact
        sample-wise negative.

    Raises
    ------
    ValueError
        If a scalar is non-finite, a duration is off the sampling grid, the
        block cannot be split into equal lobes, or the lobe geometry is invalid.
    TypeError
        If a scalar input is not real.
    """
    geometry = _validate_geometry(
        block_duration=block_duration,
        ramp_time=ramp_time,
        sampling_period=sampling_period,
    )
    amplitude = _validate_waveform_amplitude(amplitude)
    phase = _as_finite_float(phase, name="phase")

    sample_amplitudes = _srre_sample_amplitudes(geometry, amplitude)
    values = sample_amplitudes * np.exp(1j * phase)
    return Arbitrary(values, sampling_period=geometry.sampling_period)


def calculate_srre_moments(
    *,
    block_duration: float,
    ramp_time: float,
    amplitude: float,
    rabi_rate_from_amplitude: Callable[[float], float],
    sampling_period: float,
) -> SrreMoments:
    """
    Calculate `F0`, `F1`, and positive-lobe angle from sampled SRRE values.

    The waveform samples are treated as piecewise-constant AWG values and each
    sample interval is integrated analytically. Rabi rates are expressed in
    GHz, equivalent to cycles/ns.

    Parameters
    ----------
    block_duration : float
        Complete SRRE block duration in ns.
    ramp_time : float
        Ramp-up and ramp-down duration of each lobe in ns.
    amplitude : float
        Signed peak hardware amplitude in the range [-1, 1].
    rabi_rate_from_amplitude : Callable[[float], float]
        Relation mapping every signed instantaneous hardware amplitude to a
        signed Rabi rate in GHz.
    sampling_period : float
        AWG sampling period in ns.

    Returns
    -------
    SrreMoments
        Dimensionless complex moments and positive-lobe rotation angle in
        radians.

    Raises
    ------
    ValueError
        If waveform parameters are invalid or the Rabi relation returns a
        non-finite or non-real value.
    TypeError
        If a waveform scalar input is not real.
    """
    geometry = _validate_geometry(
        block_duration=block_duration,
        ramp_time=ramp_time,
        sampling_period=sampling_period,
    )
    amplitude = _validate_waveform_amplitude(amplitude)
    sample_amplitudes = _srre_sample_amplitudes(geometry, amplitude)
    return _calculate_moments(
        sample_amplitudes=sample_amplitudes,
        lobe_samples=geometry.lobe_samples,
        block_duration=geometry.block_duration,
        rabi_rate_from_amplitude=rabi_rate_from_amplitude,
        sampling_period=geometry.sampling_period,
        require_rabi_symmetry=False,
    )


def predict_srre_amplitude(
    *,
    block_duration: float,
    ramp_time: float,
    rabi_rate_from_amplitude: Callable[[float], float],
    amplitude_bounds: tuple[float, float],
    sampling_period: float,
) -> SrrePrediction:
    """
    Predict the first non-trivial sampled SRRE `F0=0` amplitude.

    The search uses the real scalar quadrature obtained by rotating `F0` by
    half of the positive-lobe angle. A sign-changing bracket is required; a
    minimum of `abs(F0)` is never substituted for a root.

    Parameters
    ----------
    block_duration : float
        Complete SRRE block duration in ns.
    ramp_time : float
        Ramp-up and ramp-down duration of each lobe in ns.
    rabi_rate_from_amplitude : Callable[[float], float]
        Relation mapping every signed instantaneous hardware amplitude to a
        signed Rabi rate in GHz.
    amplitude_bounds : tuple[float, float]
        Strictly increasing, non-negative hardware amplitude bounds within
        [0, 1].
    sampling_period : float
        AWG sampling period in ns.

    Returns
    -------
    SrrePrediction
        Predicted amplitude, peak Rabi rate, moments, geometric phase, and the
        measured numerical bracket used by the solver.

    Raises
    ------
    ValueError
        If geometry or bounds are invalid, the Rabi relation violates the
        required signed rotary-echo symmetry, or no non-trivial root is
        bracketed.
    TypeError
        If a geometry or bound scalar is not real.
    """
    geometry = _validate_geometry(
        block_duration=block_duration,
        ramp_time=ramp_time,
        sampling_period=sampling_period,
    )
    lower, upper = _validate_amplitude_bounds(amplitude_bounds)
    normalized_samples = _srre_sample_amplitudes(geometry, amplitude=1.0)

    def moments_at(candidate: float) -> SrreMoments:
        return _calculate_moments(
            sample_amplitudes=candidate * normalized_samples,
            lobe_samples=geometry.lobe_samples,
            block_duration=geometry.block_duration,
            rabi_rate_from_amplitude=rabi_rate_from_amplitude,
            sampling_period=geometry.sampling_period,
            require_rabi_symmetry=True,
        )

    def objective(candidate: float) -> float:
        moments = moments_at(candidate)
        rotated_f0 = moments.f0 * np.exp(-0.5j * moments.positive_lobe_angle)
        return float(rotated_f0.real)

    scan_amplitudes = np.linspace(lower, upper, _ROOT_SCAN_POINTS)
    scan_values = np.array([objective(value) for value in scan_amplitudes])
    root, bracket = _find_first_root(scan_amplitudes, scan_values, objective)

    moments = moments_at(root)
    if abs(moments.f0) > _ROOT_RESIDUAL_TOLERANCE:
        raise ValueError(
            "The candidate root does not satisfy complex F0=0; "
            "check rotary-echo symmetry in rabi_rate_from_amplitude."
        )
    rabi_rate = _evaluate_rabi_rates(
        np.array([root], dtype=np.float64),
        rabi_rate_from_amplitude,
    )[0]
    return SrrePrediction(
        amplitude=root,
        rabi_rate=float(rabi_rate),
        positive_lobe_angle=moments.positive_lobe_angle,
        phi_pred=moments.positive_lobe_angle / 2,
        f0=moments.f0,
        f1=moments.f1,
        root_bracket=bracket,
    )


def _validate_geometry(
    *,
    block_duration: float,
    ramp_time: float,
    sampling_period: float,
) -> _SrreGeometry:
    sampling_period = _as_positive_float(sampling_period, name="sampling_period")
    block_duration = _as_positive_float(block_duration, name="block_duration")
    ramp_time = _as_nonnegative_float(ramp_time, name="ramp_time")

    block_samples = _aligned_sample_count(
        block_duration,
        sampling_period,
        name="block_duration",
    )
    if block_samples < 2:
        raise ValueError("block_duration must contain at least two samples.")
    if block_samples % 2 != 0:
        raise ValueError("block_duration must contain an even number of samples.")
    ramp_samples = _aligned_sample_count(
        ramp_time,
        sampling_period,
        name="ramp_time",
    )
    lobe_samples = block_samples // 2
    if lobe_samples < 2 * ramp_samples:
        raise ValueError("lobe duration must be at least twice ramp_time.")
    return _SrreGeometry(
        block_duration=block_duration,
        sampling_period=sampling_period,
        lobe_samples=lobe_samples,
        ramp_samples=ramp_samples,
    )


def _aligned_sample_count(value: float, sampling_period: float, *, name: str) -> int:
    samples = value / sampling_period
    sample_count = round(samples)
    if abs(samples - sample_count) > _SAMPLING_GRID_TOLERANCE:
        raise ValueError(f"{name} must be a multiple of sampling_period.")
    return sample_count


def _positive_lobe_envelope(geometry: _SrreGeometry) -> NDArray[np.float64]:
    envelope = np.ones(geometry.lobe_samples, dtype=np.float64)
    if geometry.ramp_samples == 0:
        return envelope
    ramp_positions = (
        np.arange(geometry.ramp_samples, dtype=np.float64) + 0.5
    ) / geometry.ramp_samples
    ramp = 0.5 * (1 - np.cos(np.pi * ramp_positions))
    envelope[: geometry.ramp_samples] = ramp
    envelope[-geometry.ramp_samples :] = ramp[::-1]
    return envelope


def _srre_sample_amplitudes(
    geometry: _SrreGeometry,
    amplitude: float,
) -> NDArray[np.float64]:
    positive_lobe = amplitude * _positive_lobe_envelope(geometry)
    return np.concatenate((positive_lobe, -positive_lobe))


def _calculate_moments(
    *,
    sample_amplitudes: NDArray[np.float64],
    lobe_samples: int,
    block_duration: float,
    rabi_rate_from_amplitude: Callable[[float], float],
    sampling_period: float,
    require_rabi_symmetry: bool,
) -> SrreMoments:
    rabi_rates = _evaluate_rabi_rates(
        sample_amplitudes,
        rabi_rate_from_amplitude,
    )
    if require_rabi_symmetry:
        _require_rabi_symmetry(rabi_rates, lobe_samples)

    with np.errstate(over="ignore", invalid="ignore"):
        angle_steps = 2 * np.pi * rabi_rates * sampling_period
    if not np.all(np.isfinite(angle_steps)):
        raise ValueError(
            "rabi_rate_from_amplitude must produce finite rotation angles."
        )
    f0, f1 = _integrate_modulation(
        angle_steps=angle_steps,
        block_duration=block_duration,
        sampling_period=sampling_period,
    )
    positive_lobe_angle = float(np.sum(angle_steps[:lobe_samples]))
    return SrreMoments(
        f0=f0,
        f1=f1,
        positive_lobe_angle=positive_lobe_angle,
    )


def _integrate_modulation(
    *,
    angle_steps: NDArray[np.float64],
    block_duration: float,
    sampling_period: float,
) -> tuple[complex, complex]:
    """Integrate `exp(i * theta)` exactly over constant-rate AWG samples."""
    theta_starts = np.cumsum(angle_steps) - angle_steps
    phase_starts = np.exp(1j * theta_starts)
    half_angles = angle_steps / 2
    centered_phases = np.exp(1j * half_angles)
    # For one normalized sample interval with angle step `s`:
    # integral(exp(i*s*u), u=0..1) = exp(i*s/2) * j0(s/2),
    # integral(u*exp(i*s*u), u=0..1) = exp(i*s/2) * (j0+i*j1) / 2.
    spherical_j0 = spherical_jn(0, half_angles)
    zeroth_interval_moments = centered_phases * spherical_j0
    first_interval_moments = (
        centered_phases * (spherical_j0 + 1j * spherical_jn(1, half_angles)) / 2
    )

    modulation_integrals = sampling_period * phase_starts * zeroth_interval_moments
    interval_starts = np.arange(len(angle_steps)) * sampling_period
    centered_modulation_integrals = (
        sampling_period
        * phase_starts
        * (
            (interval_starts - block_duration / 2) * zeroth_interval_moments
            + sampling_period * first_interval_moments
        )
    )
    f0 = complex(np.sum(modulation_integrals) / block_duration)
    f1 = complex(np.sum(centered_modulation_integrals) / block_duration**2)
    return f0, f1


def _evaluate_rabi_rates(
    amplitudes: NDArray[np.float64],
    rabi_rate_from_amplitude: Callable[[float], float],
) -> NDArray[np.float64]:
    values: list[float] = []
    for amplitude in amplitudes:
        rate = rabi_rate_from_amplitude(float(amplitude))
        try:
            value = _as_finite_float(rate, name="rabi_rate_from_amplitude result")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rabi_rate_from_amplitude must return finite real values."
            ) from exc
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _validate_waveform_amplitude(amplitude: float) -> float:
    value = _as_finite_float(amplitude, name="amplitude")
    if abs(value) > 1:
        raise ValueError("absolute amplitude must not exceed 1.")
    return value


def _validate_amplitude_bounds(
    amplitude_bounds: tuple[float, float],
) -> tuple[float, float]:
    if len(amplitude_bounds) != 2:
        raise ValueError("amplitude_bounds must contain exactly two values.")
    lower = _as_finite_float(amplitude_bounds[0], name="amplitude_bounds lower")
    upper = _as_finite_float(amplitude_bounds[1], name="amplitude_bounds upper")
    if lower < 0:
        raise ValueError("amplitude_bounds must be non-negative.")
    if lower >= upper:
        raise ValueError("amplitude_bounds must be strictly increasing.")
    if upper > 1:
        raise ValueError("amplitude_bounds must not exceed 1.")
    return lower, upper


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
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _as_nonnegative_float(value: object, *, name: str) -> float:
    result = _as_finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return result


def _require_rabi_symmetry(
    rabi_rates: NDArray[np.float64],
    lobe_samples: int,
) -> None:
    if not np.allclose(
        rabi_rates[lobe_samples:],
        -rabi_rates[:lobe_samples],
        rtol=_RABI_SYMMETRY_RTOL,
        atol=_RABI_SYMMETRY_ATOL,
    ):
        raise ValueError(
            "rabi_rate_from_amplitude must preserve signed rotary-echo symmetry."
        )


def _find_first_root(
    amplitudes: NDArray[np.float64],
    values: NDArray[np.float64],
    objective: Callable[[float], float],
) -> tuple[float, tuple[float, float]]:
    for index, (amplitude, value) in enumerate(zip(amplitudes, values, strict=True)):
        if (
            amplitude > _NONTRIVIAL_AMPLITUDE_TOLERANCE
            and abs(value) <= _ROOT_VALUE_TOLERANCE
        ):
            exact_root = float(amplitude)
            lower_index = max(0, index - 1)
            upper_index = min(len(amplitudes) - 1, index + 1)
            return exact_root, (
                float(amplitudes[lower_index]),
                float(amplitudes[upper_index]),
            )
        if index == 0 or values[index - 1] * value >= 0:
            continue
        lower = float(amplitudes[index - 1])
        upper = float(amplitude)
        root = cast(
            float,
            brentq(
                objective,
                lower,
                upper,
                xtol=1e-13,
                rtol=np.float64(1e-14),
                full_output=False,
            ),
        )
        if root > _NONTRIVIAL_AMPLITUDE_TOLERANCE:
            return root, (lower, upper)
    raise ValueError("No non-trivial F0 root is bracketed by amplitude_bounds.")
