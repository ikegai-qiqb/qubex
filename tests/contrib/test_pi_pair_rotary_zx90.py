"""Tests for pi-pair rotary-echo ZX90 construction and calibration."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib import (
    calibrate_pi_pair_rotary_zx90,
    pi_pair_rotary_zx90,
)
from qubex.experiment.models import Result
from qubex.pulse import Rect


class _CalibrationNote:
    def __init__(self, params: dict[str, float]) -> None:
        self.params = params

    def get_cr_param(self, _label: str, **_kwargs: Any) -> dict[str, float]:
        return self.params


class _PulseService:
    def __init__(self, *, pi_amplitude: float = 0.25) -> None:
        self.pi_amplitude = pi_amplitude

    def x180(self, _target: str) -> Rect:
        return Rect(
            duration=16.0,
            amplitude=self.pi_amplitude,
            sampling_period=2.0,
        )

    def calc_control_amplitude(self, _target: str, rabi_rate: float) -> float:
        return 10.0 * rabi_rate


class _MeasurementService:
    def __init__(
        self,
        *,
        root: float,
        signal_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.root = root
        self.signal_function = signal_function
        self.calls: list[dict[str, Any]] = []

    def sweep_parameter(self, **kwargs: Any) -> Result:
        self.calls.append(kwargs)
        amplitudes = np.asarray(kwargs["sweep_range"], dtype=float)
        target_state = kwargs["initial_states"]["Q1"]
        sign = 1.0 if target_state == "0" else -1.0
        base_signal = (
            amplitudes - self.root
            if self.signal_function is None
            else self.signal_function(amplitudes)
        )
        signal = sign * base_signal
        return Result(data={"Q1": SimpleNamespace(normalized=signal)})


class _Experiment:
    def __init__(
        self,
        *,
        root: float = 0.5,
        pi_amplitude: float = 0.25,
        cancel_amplitude: float = 0.08,
        signal_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        rate = 1.0 / (8.0 * root * (128.0 - 8.0))
        params = {
            "duration": 0.0,
            "ramptime": 8.0,
            "cr_amplitude": 0.4,
            "cr_phase": 0.1,
            "cr_beta": 0.0,
            "cancel_amplitude": cancel_amplitude,
            "cancel_phase": -0.2,
            "cancel_beta": 0.0,
            "zx_rotation_rate": rate,
        }
        self.pulse = _PulseService(pi_amplitude=pi_amplitude)
        self.measurement_service = _MeasurementService(
            root=root,
            signal_function=signal_function,
        )
        self.ctx = SimpleNamespace(
            calib_note=_CalibrationNote(params),
            calibration_valid_days=14,
            measurement=SimpleNamespace(sampling_period=2.0),
            util=SimpleNamespace(resolve_sampling_period=lambda value: value),
            qubits={
                "Q0": SimpleNamespace(frequency=5.0),
                "Q1": SimpleNamespace(frequency=5.2),
            },
            experiment_system=None,
        )


def test_calibration_uses_default_four_state_sweep_and_finds_zx90_root() -> None:
    """The default calibration should fit the four-state ZX zero crossing."""
    exp = _Experiment(root=0.5)

    result = calibrate_pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        duration=128.0,
        plot=False,
        shots=100,
        interval=200.0,
    )

    calibration = result.data["pi_pair_rotary_zx90_calibration"]
    assert calibration["duration"] == pytest.approx(128.0, abs=1e-12)
    assert calibration["cr_amplitude"] == pytest.approx(0.5, abs=1e-12)
    assert calibration["cancel_amplitude"] == pytest.approx(0.1, abs=1e-12)
    assert_allclose(
        result.data["amplitude_range"],
        np.linspace(0.42, 0.58, 17),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(result.data["s_zx"], np.linspace(-0.08, 0.08, 17))
    assert len(exp.measurement_service.calls) == 4
    assert [call["initial_states"] for call in exp.measurement_service.calls] == [
        {"Q0": "0", "Q1": "0"},
        {"Q0": "0", "Q1": "1"},
        {"Q0": "1", "Q1": "0"},
        {"Q0": "1", "Q1": "1"},
    ]


def test_pi_pair_schedule_places_signed_pi_pulses_in_each_half() -> None:
    """The target waveform should contain the signed pi-pair rotary pattern."""
    exp = _Experiment(cancel_amplitude=0.0)

    schedule = pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        cr_amplitude=0.5,
        duration=128.0,
    )

    target = schedule.get_sampled_sequence("Q1")
    assert schedule.duration == pytest.approx(288.0, abs=1e-12)
    first_half = target[:64]
    expected_signs = np.concatenate(
        (
            np.zeros(8),
            np.ones(8),
            np.zeros(8),
            np.ones(8),
            -np.ones(8),
            np.zeros(8),
            -np.ones(8),
            np.zeros(8),
        )
    )
    assert_allclose(first_half.real, 0.25 * expected_signs, rtol=0.0, atol=1e-12)
    assert_allclose(first_half.imag, 0.0, rtol=0.0, atol=1e-12)
    assert_allclose(target[72:136], -first_half, rtol=0.0, atol=1e-12)


def test_calibration_selects_zero_crossing_nearest_prediction() -> None:
    """Multiple crossings should resolve to the one nearest the predicted amplitude."""
    exp = _Experiment(
        root=0.5,
        signal_function=lambda amplitude: (
            (amplitude - 0.44) * (amplitude - 0.52) * (amplitude - 0.56)
        ),
    )

    result = calibrate_pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        duration=128.0,
        amplitude_range=np.linspace(0.4, 0.6, 11),
        plot=False,
    )

    assert result.data["root"] == pytest.approx(0.52, abs=1e-12)
    assert result.data["root_bracket"] == pytest.approx((0.5, 0.52), abs=1e-12)


def test_calibration_predicts_aligned_duration_and_recenters_amplitude() -> None:
    """Automatic duration alignment should be reflected in the amplitude prediction."""
    exp = _Experiment(root=0.5)
    rate = exp.ctx.calib_note.params["zx_rotation_rate"]
    aligned_root = 1.0 / (8.0 * rate * (160.0 - 8.0))
    exp.measurement_service.root = aligned_root

    result = calibrate_pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        plot=False,
    )

    calibration = result.data["pi_pair_rotary_zx90_calibration"]
    assert calibration["duration"] == pytest.approx(160.0, abs=1e-12)
    assert calibration["predicted_cr_amplitude"] == pytest.approx(
        aligned_root,
        abs=1e-12,
    )
    assert calibration["cr_amplitude"] == pytest.approx(aligned_root, abs=1e-12)


def test_pi_pair_schedule_rejects_duration_without_ramp_safe_blanks() -> None:
    """A half duration should leave four ramp-safe blanks around four pi pulses."""
    exp = _Experiment()

    with pytest.raises(ValueError, match="blank duration"):
        pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            cr_amplitude=0.5,
            duration=80.0,
        )


def test_calibration_rejects_prediction_above_safe_cr_limit() -> None:
    """Calibration should fail when the predicted root exceeds the safe CR limit."""
    exp = _Experiment(root=0.5)

    with pytest.raises(ValueError, match="predicted CR amplitude"):
        calibrate_pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            duration=128.0,
            max_amplitude=0.45,
            plot=False,
        )


def test_default_sweep_preserves_width_below_safe_upper_limit() -> None:
    """A capped default sweep should shift down without losing points or width."""
    exp = _Experiment(root=0.5)

    result = calibrate_pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        duration=128.0,
        max_amplitude=0.55,
        plot=False,
    )

    assert_allclose(
        result.data["amplitude_range"],
        np.linspace(0.39, 0.55, 17),
        rtol=0.0,
        atol=1e-12,
    )


def test_pi_pair_schedule_rejects_target_channel_overflow() -> None:
    """The composite cancellation and rotary waveform should stay within one."""
    exp = _Experiment(pi_amplitude=0.95)

    with pytest.raises(ValueError, match="target channel"):
        pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            cr_amplitude=0.5,
            duration=128.0,
        )


def test_calibration_validates_composite_waveforms_before_measurement() -> None:
    """Unsafe composite waveforms should fail before any hardware sweep starts."""
    exp = _Experiment(pi_amplitude=0.95)

    with pytest.raises(ValueError, match="target channel"):
        calibrate_pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            duration=128.0,
            plot=False,
        )

    assert exp.measurement_service.calls == []
