"""Tests for one-qubit SRRE amplitude calibration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib import calibrate_srre
from qubex.contrib.experiment.srre_calibration import (
    _analyze_zero_crossing,
    _build_srre_calibration_sequence,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Arbitrary, PulseSchedule


class _PulseService:
    def __init__(
        self,
        *,
        sampling_period: float = 1.0,
        x90_amplitude: float = 0.25,
        y90_amplitude: float = 0.25,
    ) -> None:
        self.validated_targets: list[list[str]] = []
        self.sampling_period = sampling_period
        self.x90_amplitude = x90_amplitude
        self.y90_amplitude = y90_amplitude

    def validate_rabi_params(self, targets: list[str]) -> None:
        self.validated_targets.append(targets)

    @staticmethod
    def calc_rabi_rate(_target: str, amplitude: float) -> float:
        return 0.02 * amplitude

    def y90(self, _target: str) -> Arbitrary:
        return Arbitrary(
            [self.y90_amplitude * 1j],
            sampling_period=self.sampling_period,
        )

    def x90(self, _target: str) -> Arbitrary:
        return Arbitrary(
            [self.x90_amplitude],
            sampling_period=self.sampling_period,
        )


class _SweepData:
    def __init__(self, normalized: np.ndarray) -> None:
        self.normalized = normalized


class _MeasurementService:
    def __init__(self, *, target: str, root: float) -> None:
        self.target = target
        self.root = root
        self.calls: list[dict[str, Any]] = []
        self.schedules: list[PulseSchedule] = []

    def sweep_parameter(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        sweep_range = np.asarray(kwargs["sweep_range"])
        sequence = kwargs["sequence"]
        self.schedules = [sequence(int(index)) for index in sweep_range]

        amplitudes = np.array(
            [
                np.max(np.abs(schedule.get_sampled_sequences()[self.target][1:-1]))
                for schedule in self.schedules[::2]
            ]
        )
        differential = amplitudes - self.root
        interleaved = np.empty(2 * amplitudes.size)
        interleaved[0::2] = differential
        interleaved[1::2] = -differential
        return SimpleNamespace(data={self.target: _SweepData(interleaved)})


class _Experiment:
    def __init__(self, *, target: str = "Q00", root: float = 0.52) -> None:
        self.pulse = _PulseService()
        self.measurement_service = _MeasurementService(target=target, root=root)
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            util=SimpleNamespace(resolve_sampling_period=float),
        )


def test_zero_crossing_recovers_known_root_from_differential_signal() -> None:
    """A bracketed mock differential signal should recover its known root."""
    amplitudes = np.array([0.4, 0.5, 0.6])
    signal_plus = 3.0 * (amplitudes - 0.53) + 0.2
    signal_minus = -3.0 * (amplitudes - 0.53) + 0.2

    analysis = _analyze_zero_crossing(
        amplitudes=amplitudes,
        signal_plus=signal_plus,
        signal_minus=signal_minus,
        predicted_amplitude=0.5,
    )

    assert analysis.root == pytest.approx(0.53, abs=1e-14)
    assert analysis.root_bracket == pytest.approx((0.5, 0.6), abs=1e-14)
    assert analysis.fit_slope == pytest.approx(3.0, abs=1e-14)
    assert_allclose(
        analysis.differential_signal,
        3.0 * (amplitudes - 0.53),
        rtol=1e-14,
        atol=1e-14,
    )


def test_zero_crossing_selects_bracket_closest_to_prediction() -> None:
    """Multiple crossings should select the root nearest the prediction."""
    amplitudes = np.array([0.2, 0.4, 0.6, 0.8])
    differential = np.array([-1.0, 1.0, -1.0, 1.0])

    analysis = _analyze_zero_crossing(
        amplitudes=amplitudes,
        signal_plus=differential,
        signal_minus=-differential,
        predicted_amplitude=0.68,
    )

    assert analysis.root == pytest.approx(0.7, abs=1e-14)
    assert analysis.root_bracket == pytest.approx((0.6, 0.8), abs=1e-14)


def test_zero_crossing_rejects_trivial_zero_amplitude_root() -> None:
    """A measured crossing at zero amplitude should not replace a nontrivial root."""
    with pytest.raises(ValueError, match="positive"):
        _analyze_zero_crossing(
            amplitudes=[0.0, 0.1, 0.2],
            signal_plus=[0.0, 0.1, 0.2],
            signal_minus=[0.0, -0.1, -0.2],
            predicted_amplitude=0.5,
        )


@pytest.mark.parametrize(
    ("signal_plus", "signal_minus", "message"),
    [
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], "does not bracket"),
        ([-1e-12, 1e-12, 2e-12], [0.0, 0.0, 0.0], "slope is too small"),
        ([-1.0, np.nan, 1.0], [0.0, 0.0, 0.0], "finite"),
    ],
)
def test_zero_crossing_rejects_unsafe_signal_data(
    signal_plus: list[float],
    signal_minus: list[float],
    message: str,
) -> None:
    """Missing roots, low slopes, and non-finite data should fail safely."""
    with pytest.raises(ValueError, match=message):
        _analyze_zero_crossing(
            amplitudes=np.array([0.4, 0.5, 0.6]),
            signal_plus=np.asarray(signal_plus),
            signal_minus=np.asarray(signal_minus),
            predicted_amplitude=0.5,
        )


def test_zero_crossing_rejects_complex_signal_data() -> None:
    """Complex measurement values should fail instead of losing their imaginary part."""
    with pytest.raises(TypeError, match="real array"):
        _analyze_zero_crossing(
            amplitudes=[0.4, 0.5, 0.6],
            signal_plus=np.array([-1.0 + 0.1j, 0.0, 1.0]),
            signal_minus=[0.0, 0.0, 0.0],
            predicted_amplitude=0.5,
        )


def test_repeated_srre_uses_continuous_detuning_and_corrected_analysis_phase() -> None:
    """Repeated SRRE should keep one phase ramp and compensate its final phase."""
    pulse = _PulseService()
    detuning = 0.001
    schedule = _build_srre_calibration_sequence(
        cast(Any, SimpleNamespace(pulse=pulse)),
        "Q00",
        block_duration=200.0,
        ramp_time=0.0,
        amplitude=0.5,
        detuning=detuning,
        repetitions=2,
        analysis_angle=np.pi / 2,
        sampling_period=1.0,
    )

    sampled = schedule.get_sampled_sequences()["Q00"]
    srre_values = sampled[1:401]
    expected_srre = np.tile(np.r_[np.full(100, 0.5), np.full(100, -0.5)], 2)
    expected_srre = expected_srre * np.exp(-2j * np.pi * detuning * np.arange(400))

    assert_allclose(srre_values, expected_srre, rtol=1e-14, atol=1e-14)
    assert srre_values[200] != pytest.approx(srre_values[0], abs=1e-6)
    expected_analysis = 0.25 * np.exp(-2j * np.pi * detuning * 400.0)
    assert sampled[-1] == pytest.approx(expected_analysis, abs=1e-14)


def test_sequence_rejects_reference_pulses_on_a_different_sampling_grid() -> None:
    """Preparation and analysis pulses should share the SRRE sampling period."""
    pulse = _PulseService(sampling_period=2.0)

    with pytest.raises(ValueError, match="sampling period"):
        _build_srre_calibration_sequence(
            cast(Any, SimpleNamespace(pulse=pulse)),
            "Q00",
            block_duration=200.0,
            ramp_time=0.0,
            amplitude=0.5,
            detuning=0.001,
            repetitions=1,
            analysis_angle=np.pi / 2,
            sampling_period=1.0,
        )


def test_sequence_rejects_analysis_pulse_amplitude_overflow() -> None:
    """A scaled analysis pulse exceeding the hardware limit should fail."""
    pulse = _PulseService(x90_amplitude=0.75)

    with pytest.raises(ValueError, match="analysis pulse amplitude"):
        _build_srre_calibration_sequence(
            cast(Any, SimpleNamespace(pulse=pulse)),
            "Q00",
            block_duration=200.0,
            ramp_time=0.0,
            amplitude=0.5,
            detuning=0.001,
            repetitions=1,
            analysis_angle=np.pi,
            sampling_period=1.0,
        )


def test_calibrate_srre_interleaves_detuning_and_returns_data_contract() -> None:
    """Calibration should pair detunings and return the complete SRRE metadata."""
    exp = _Experiment(root=0.52)
    amplitude_range = np.array([0.45, 0.50, 0.55])

    result = calibrate_srre(
        cast(Any, exp),
        "Q00",
        block_duration=200.0,
        ramp_time=0.0,
        amplitude_range=amplitude_range,
        probe_detuning=0.001,
        repetitions=2,
        n_shots=256,
        shot_interval=1024.0,
        plot=False,
    )

    assert isinstance(result, Result)
    calibration = cast(dict[str, Any], result.data["srre_calibration"])
    assert calibration["target"] == "Q00"
    assert calibration["amplitude"] == pytest.approx(0.52, abs=1e-12)
    assert calibration["predicted_amplitude"] == pytest.approx(0.5, abs=1e-10)
    assert calibration["rabi_rate"] == pytest.approx(0.0104, abs=1e-12)
    assert calibration["block_duration"] == pytest.approx(200.0, abs=1e-14)
    assert calibration["ramp_time"] == pytest.approx(0.0, abs=1e-14)
    assert calibration["sampling_period"] == pytest.approx(1.0, abs=1e-14)
    assert calibration["positive_lobe_angle"] == pytest.approx(2 * np.pi, abs=1e-10)
    assert calibration["phi_pred"] == pytest.approx(np.pi, abs=1e-10)
    assert calibration["analysis_angle"] == pytest.approx(-np.pi / 2, abs=1e-10)
    assert calibration["probe_detuning"] == pytest.approx(0.001, abs=1e-14)
    assert calibration["repetitions"] == 2
    assert calibration["f0_predicted"] == pytest.approx(0.0j, abs=1e-10)
    assert calibration["f1_predicted"] == pytest.approx(0.0j, abs=1e-14)
    assert calibration["root_bracket"] == pytest.approx((0.5, 0.55), abs=1e-12)
    assert calibration["fit_slope"] == pytest.approx(1.0, abs=1e-12)
    assert_allclose(calibration["amplitude_range"], amplitude_range)
    assert_allclose(calibration["signal_plus"], amplitude_range - 0.52)
    assert_allclose(calibration["signal_minus"], -(amplitude_range - 0.52))
    assert_allclose(calibration["differential_signal"], amplitude_range - 0.52)

    call = exp.measurement_service.calls[0]
    assert_allclose(call["sweep_range"], np.arange(6))
    assert call["n_shots"] == 256
    assert call["shot_interval"] == pytest.approx(1024.0)
    assert call["plot"] is False
    assert exp.pulse.validated_targets == [["Q00"]]
    for plus_schedule, minus_schedule in zip(
        exp.measurement_service.schedules[::2],
        exp.measurement_service.schedules[1::2],
        strict=True,
    ):
        plus = plus_schedule.get_sampled_sequences()["Q00"][1:-1]
        minus = minus_schedule.get_sampled_sequences()["Q00"][1:-1]
        assert_allclose(minus, np.conj(plus), rtol=1e-14, atol=1e-14)


def test_calibrate_srre_detaches_returned_amplitude_range_from_caller() -> None:
    """Returned calibration data should not alias the caller's sweep array."""
    exp = _Experiment(root=0.52)
    amplitude_range = np.array([0.45, 0.50, 0.55])

    result = calibrate_srre(
        cast(Any, exp),
        "Q00",
        block_duration=200.0,
        ramp_time=0.0,
        amplitude_range=amplitude_range,
        plot=False,
    )
    amplitude_range[:] = 0.1

    calibration = cast(dict[str, Any], result.data["srre_calibration"])
    assert_allclose(calibration["amplitude_range"], [0.45, 0.50, 0.55])


def test_calibrate_srre_does_not_return_fallback_without_a_bracket() -> None:
    """Calibration should fail rather than adopt a fallback amplitude."""
    exp = _Experiment(root=0.9)

    with pytest.raises(ValueError, match="does not bracket"):
        calibrate_srre(
            cast(Any, exp),
            "Q00",
            block_duration=200.0,
            ramp_time=0.0,
            amplitude_range=[0.45, 0.50, 0.55],
            plot=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"repetitions": 0}, "repetitions must be a positive integer"),
        ({"repetitions": 1.5}, "repetitions must be a positive integer"),
        ({"probe_detuning": 0.0}, "probe_detuning must be non-zero"),
        ({"probe_detuning": -0.001}, "probe_detuning must be positive"),
        ({"probe_detuning": np.nan}, "probe_detuning must be finite"),
        ({"probe_detuning": 0.5}, "below the sampling Nyquist frequency"),
        ({"amplitude_range": [0.5]}, "at least two"),
        ({"amplitude_range": [0.5, 0.4]}, "strictly increasing"),
        ({"amplitude_range": [0.4, np.inf]}, "finite"),
        ({"amplitude_range": [0.4, 1.1]}, "amplitude_bounds"),
        ({"n_shots": 0}, "n_shots must be a positive integer"),
        ({"n_shots": 1.5}, "n_shots must be a positive integer"),
        ({"shot_interval": 0.0}, "shot_interval must be positive"),
        ({"shot_interval": np.nan}, "shot_interval must be finite"),
        ({"plot": "yes"}, "plot must be a boolean"),
    ],
)
def test_calibrate_srre_rejects_invalid_calibration_inputs(
    overrides: dict[str, Any], message: str
) -> None:
    """Invalid calibration inputs should fail before measurement."""
    parameters: dict[str, Any] = {
        "block_duration": 200.0,
        "ramp_time": 0.0,
        "amplitude_bounds": (0.0, 1.0),
        "plot": False,
    }
    parameters.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        calibrate_srre(cast(Any, _Experiment()), "Q00", **parameters)


def test_calibrate_srre_rejects_empty_target_before_measurement() -> None:
    """The calibration target should be a non-empty qubit label."""
    exp = _Experiment()

    with pytest.raises(ValueError, match="target must be a non-empty string"):
        calibrate_srre(
            cast(Any, exp),
            "",
            block_duration=200.0,
            ramp_time=0.0,
            plot=False,
        )

    assert exp.measurement_service.calls == []
