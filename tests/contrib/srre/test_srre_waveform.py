"""Tests for SRRE waveform generation and numerical analysis."""

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib import (
    calculate_srre_moments,
    predict_srre_amplitude,
    srre_waveform,
)
from qubex.contrib.experiment.srre_waveform import (
    SrreMoments,
    SrrePrediction,
)
from qubex.pulse import Arbitrary


def _linear_rabi_rate(amplitude: float) -> float:
    """Return a signed linear Rabi rate in GHz."""
    return 0.02 * amplitude


def test_srre_waveform_builds_antisymmetric_raised_cosine_lobes() -> None:
    """SRRE samples should be raised-cosine lobes with exact antisymmetry."""
    waveform = srre_waveform(
        block_duration=16.0,
        ramp_time=2.0,
        amplitude=0.4,
        phase=np.pi / 2,
        sampling_period=1.0,
    )

    positive = np.asarray(waveform.values[:8])
    negative = np.asarray(waveform.values[8:])
    expected_envelope = np.array(
        [
            0.5 * (1 - np.cos(np.pi / 4)),
            0.5 * (1 - np.cos(3 * np.pi / 4)),
            1.0,
            1.0,
            1.0,
            1.0,
            0.5 * (1 - np.cos(3 * np.pi / 4)),
            0.5 * (1 - np.cos(np.pi / 4)),
        ]
    )

    assert isinstance(waveform, Arbitrary)
    assert waveform.duration == pytest.approx(16.0, abs=1e-12)
    assert waveform.sampling_period == pytest.approx(1.0, abs=1e-12)
    assert_allclose(positive, 0.4j * expected_envelope, rtol=1e-14, atol=1e-14)
    assert_allclose(negative, -positive, rtol=0.0, atol=0.0)
    assert np.sum(waveform.values) == pytest.approx(0.0j, abs=1e-15)


def test_srre_waveform_supports_zero_ramp_rectangular_lobes() -> None:
    """A zero ramp should produce rectangular positive and negative lobes."""
    waveform = srre_waveform(
        block_duration=8.0,
        ramp_time=0.0,
        amplitude=-0.25,
        sampling_period=1.0,
    )

    assert_allclose(
        waveform.values,
        [-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"block_duration": 0.0}, "block_duration must be positive and finite"),
        ({"block_duration": 1e-12}, "at least two samples"),
        ({"block_duration": 7.0}, "even number of samples"),
        ({"block_duration": 8.5}, "multiple of sampling_period"),
        ({"ramp_time": -1.0}, "ramp_time must be non-negative and finite"),
        ({"ramp_time": 1.5}, "ramp_time must be a multiple"),
        ({"ramp_time": 3.0}, "lobe duration must be at least twice ramp_time"),
        ({"amplitude": np.inf}, "amplitude must be finite"),
        ({"amplitude": 1.01}, "absolute amplitude must not exceed 1"),
        ({"phase": np.nan}, "phase must be finite"),
        ({"sampling_period": 0.0}, "sampling_period must be positive and finite"),
    ],
)
def test_srre_waveform_rejects_invalid_parameters(
    overrides: dict[str, float], message: str
) -> None:
    """Invalid waveform geometry and scalar parameters should fail clearly."""
    parameters = {
        "block_duration": 8.0,
        "ramp_time": 1.0,
        "amplitude": 0.5,
        "phase": 0.0,
        "sampling_period": 1.0,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        srre_waveform(**parameters)


@pytest.mark.parametrize("field", ["amplitude", "phase"])
def test_srre_waveform_rejects_complex_scalar_parameters(field: str) -> None:
    """Complex amplitude or phase should not be accepted as a real pulse scalar."""
    parameters: dict[str, Any] = {
        "block_duration": 8.0,
        "ramp_time": 1.0,
        "amplitude": 0.5,
        "phase": 0.0,
        "sampling_period": 1.0,
    }
    parameters[field] = 0.5j

    with pytest.raises(TypeError, match=f"{field} must be a real number"):
        srre_waveform(**parameters)


def test_calculate_srre_moments_preserves_symmetry_and_lobe_area() -> None:
    """Discrete moments should report zero F1 and the sampled positive-lobe area."""
    moments = calculate_srre_moments(
        block_duration=16.0,
        ramp_time=2.0,
        amplitude=0.4,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        sampling_period=1.0,
    )

    expected_positive_angle = 2 * np.pi * _linear_rabi_rate(0.4) * 6.0
    assert isinstance(moments, SrreMoments)
    assert moments.positive_lobe_angle == pytest.approx(
        expected_positive_angle, rel=1e-14, abs=1e-14
    )
    assert moments.f1 == pytest.approx(0.0j, abs=1e-15)
    assert np.isfinite(moments.f0.real)
    assert np.isfinite(moments.f0.imag)


def test_moments_use_the_realized_sampling_grid_duration() -> None:
    """Accepted duration roundoff must not spoil discrete SRRE symmetry."""
    exact = calculate_srre_moments(
        block_duration=8.0,
        ramp_time=1.0,
        amplitude=0.5,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        sampling_period=1.0,
    )
    rounded = calculate_srre_moments(
        block_duration=8.0 + 5e-10,
        ramp_time=1.0,
        amplitude=0.5,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        sampling_period=1.0,
    )

    assert rounded.f0 == pytest.approx(exact.f0, abs=1e-15)
    assert rounded.f1 == pytest.approx(exact.f1, abs=1e-15)
    assert rounded.positive_lobe_angle == pytest.approx(
        exact.positive_lobe_angle, abs=1e-15
    )


def test_negative_lobe_moments_remain_finite_and_symmetric() -> None:
    """Negative SRRE angles should keep both moments finite and F1 near zero."""
    moments = calculate_srre_moments(
        block_duration=64.0,
        ramp_time=8.0,
        amplitude=0.8,
        rabi_rate_from_amplitude=lambda amplitude: 0.08 * amplitude,
        sampling_period=1.0,
    )

    assert np.isfinite(
        [moments.f0.real, moments.f0.imag, moments.f1.real, moments.f1.imag]
    ).all()
    assert moments.f1 == pytest.approx(0.0j, abs=1e-14)


def test_calculate_srre_moments_integrates_piecewise_constant_samples() -> None:
    """Moment calculation should integrate each constant AWG sample exactly."""
    rabi_rate = 0.05
    positive_lobe_angle = 2 * np.pi * rabi_rate * 2.0
    expected_f0 = np.exp(0.5j * positive_lobe_angle) * np.sinc(
        positive_lobe_angle / (2 * np.pi)
    )

    moments = calculate_srre_moments(
        block_duration=4.0,
        ramp_time=0.0,
        amplitude=0.5,
        rabi_rate_from_amplitude=lambda amplitude: 0.1 * amplitude,
        sampling_period=1.0,
    )

    assert moments.f0 == pytest.approx(expected_f0, rel=1e-13, abs=1e-13)
    assert moments.f1 == pytest.approx(0.0j, abs=1e-14)


def test_calculate_srre_moments_uses_instantaneous_nonlinear_rabi_rate() -> None:
    """Moment integration should apply a nonlinear Rabi relation to every sample."""
    nonlinear_rate = lambda amplitude: amplitude + amplitude**3
    waveform = srre_waveform(
        block_duration=8.0,
        ramp_time=1.0,
        amplitude=0.5,
        sampling_period=1.0,
    )
    positive_rates = [nonlinear_rate(value.real) for value in waveform.values[:4]]

    moments = calculate_srre_moments(
        block_duration=8.0,
        ramp_time=1.0,
        amplitude=0.5,
        rabi_rate_from_amplitude=nonlinear_rate,
        sampling_period=1.0,
    )

    assert moments.positive_lobe_angle == pytest.approx(
        2 * np.pi * sum(positive_rates), rel=1e-14, abs=1e-14
    )
    assert moments.f1 == pytest.approx(0.0j, abs=1e-14)


def test_calculate_srre_moments_batches_vectorized_rabi_rate() -> None:
    """A vector-capable Rabi relation should be evaluated once per waveform."""
    call_shapes: list[tuple[int, ...]] = []

    def vectorized_rabi_rate(amplitude: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(amplitude)
        call_shapes.append(values.shape)
        return 0.02 * values

    calculate_srre_moments(
        block_duration=8.0,
        ramp_time=1.0,
        amplitude=0.5,
        rabi_rate_from_amplitude=vectorized_rabi_rate,  # type: ignore[arg-type]
        sampling_period=1.0,
    )

    assert call_shapes == [(8,)]


@pytest.mark.parametrize(
    "bad_rate",
    [
        lambda _amplitude: np.nan,
        lambda _amplitude: np.inf,
        lambda _amplitude: 1j,
        lambda _amplitude: [0.1],
    ],
)
def test_calculate_srre_moments_rejects_invalid_rabi_rates(bad_rate) -> None:
    """A non-finite or non-real amplitude-to-Rabi result should fail safely."""
    with pytest.raises(ValueError, match="finite real values"):
        calculate_srre_moments(
            block_duration=8.0,
            ramp_time=1.0,
            amplitude=0.5,
            rabi_rate_from_amplitude=bad_rate,
            sampling_period=1.0,
        )


def test_calculate_srre_moments_rejects_overflowing_rotation_angles() -> None:
    """A finite Rabi rate that overflows angle conversion should fail safely."""
    with pytest.raises(ValueError, match="finite rotation angles"):
        calculate_srre_moments(
            block_duration=8.0,
            ramp_time=1.0,
            amplitude=0.5,
            rabi_rate_from_amplitude=lambda _amplitude: float(np.finfo(np.float64).max),
            sampling_period=1.0,
        )


def test_calculate_srre_moments_rejects_overflowing_cumulative_rotation() -> None:
    """Finite per-sample angles must not turn into NaN moments after accumulation."""
    with pytest.raises(ValueError, match="Cumulative SRRE rotation angles"):
        calculate_srre_moments(
            block_duration=8.0,
            ramp_time=0.0,
            amplitude=1.0,
            rabi_rate_from_amplitude=lambda amplitude: 1e307 * amplitude,
            sampling_period=1.0,
        )


def test_predict_srre_amplitude_finds_first_nontrivial_root() -> None:
    """Prediction should select the first F0 root when multiple roots are bounded."""
    prediction = predict_srre_amplitude(
        block_duration=200.0,
        ramp_time=0.0,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        amplitude_bounds=(0.0, 1.0),
        sampling_period=1.0,
    )

    assert isinstance(prediction, SrrePrediction)
    assert prediction.amplitude == pytest.approx(0.5, abs=1e-10)
    assert prediction.rabi_rate == pytest.approx(0.01, abs=1e-12)
    assert prediction.positive_lobe_angle == pytest.approx(2 * np.pi, abs=1e-10)
    assert prediction.phi_pred == pytest.approx(np.pi, abs=1e-10)
    assert prediction.f0 == pytest.approx(0.0j, abs=1e-10)
    assert prediction.f1 == pytest.approx(0.0j, abs=1e-14)
    assert prediction.root_bracket[0] <= prediction.amplitude
    assert prediction.amplitude <= prediction.root_bracket[1]
    assert prediction.root_bracket[1] < 0.75


def test_predict_srre_amplitude_solves_shaped_waveform_not_naive_area() -> None:
    """A raised-cosine prediction should solve F0 instead of imposing a 2-pi area."""
    prediction = predict_srre_amplitude(
        block_duration=200.0,
        ramp_time=20.0,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        amplitude_bounds=(0.0, 1.0),
        sampling_period=1.0,
    )

    assert prediction.f0 == pytest.approx(0.0j, abs=1e-10)
    assert prediction.f1 == pytest.approx(0.0j, abs=1e-14)
    assert prediction.positive_lobe_angle != pytest.approx(2 * np.pi, abs=0.1)
    assert prediction.root_bracket[0] < prediction.amplitude
    assert prediction.amplitude < prediction.root_bracket[1]


def test_srre_result_dataclasses_are_frozen() -> None:
    """The Thread A data contracts should be immutable."""
    moments = calculate_srre_moments(
        block_duration=8.0,
        ramp_time=1.0,
        amplitude=0.5,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        sampling_period=1.0,
    )
    prediction = predict_srre_amplitude(
        block_duration=200.0,
        ramp_time=0.0,
        rabi_rate_from_amplitude=_linear_rabi_rate,
        amplitude_bounds=(0.4, 0.6),
        sampling_period=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        moments.f0 = 0.0j  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        prediction.amplitude = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ((-0.1, 0.5), "non-negative"),
        ((0.5, 0.5), "strictly increasing"),
        ((0.6, 0.5), "strictly increasing"),
        ((0.0, 1.01), "must not exceed 1"),
        ((0.0, np.inf), "finite"),
    ],
)
def test_predict_srre_amplitude_rejects_invalid_bounds(
    bounds: tuple[float, float], message: str
) -> None:
    """Prediction should reject invalid hardware amplitude bounds."""
    with pytest.raises(ValueError, match=message):
        predict_srre_amplitude(
            block_duration=20.0,
            ramp_time=2.0,
            rabi_rate_from_amplitude=_linear_rabi_rate,
            amplitude_bounds=bounds,
            sampling_period=1.0,
        )


def test_predict_srre_amplitude_fails_when_bounds_have_no_root() -> None:
    """Prediction should not substitute a minimum when no F0 root is bracketed."""
    with pytest.raises(ValueError, match="No non-trivial F0 root"):
        predict_srre_amplitude(
            block_duration=20.0,
            ramp_time=2.0,
            rabi_rate_from_amplitude=lambda amplitude: 1e-4 * amplitude,
            amplitude_bounds=(0.0, 1.0),
            sampling_period=1.0,
        )


def test_predict_srre_amplitude_rejects_non_odd_rabi_relation() -> None:
    """Prediction should fail if the Rabi model breaks rotary-echo symmetry."""
    with pytest.raises(ValueError, match="rotary-echo symmetry"):
        predict_srre_amplitude(
            block_duration=200.0,
            ramp_time=0.0,
            rabi_rate_from_amplitude=lambda amplitude: 0.02 * amplitude + 1e-4,
            amplitude_bounds=(0.1, 1.0),
            sampling_period=1.0,
        )


def test_predict_srre_amplitude_checks_rabi_symmetry_throughout_bounds() -> None:
    """Prediction should reject a Rabi model that is odd only at the upper bound."""
    with pytest.raises(ValueError, match="rotary-echo symmetry"):
        predict_srre_amplitude(
            block_duration=200.0,
            ramp_time=0.0,
            rabi_rate_from_amplitude=lambda amplitude: (
                0.02 * amplitude + 1e-4 * (1 - amplitude**2)
            ),
            amplitude_bounds=(0.0, 1.0),
            sampling_period=1.0,
        )
