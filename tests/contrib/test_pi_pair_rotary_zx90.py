"""Tests for pi-pair rotary-echo ZX90 construction and calibration."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

from qubex.contrib import (
    calibrate_pi_pair_rotary_zx90,
    pi_pair_rotary_zx90,
)
from qubex.experiment.models import Result
from qubex.pulse import PulseSchedule, Rect


class _CalibrationNote:
    def __init__(self, params: dict[str, float]) -> None:
        self.params = params

    def get_cr_param(self, _label: str, **_kwargs: Any) -> dict[str, float]:
        return self.params


class _ConfigLoader:
    def load_param_data(self, name: str) -> dict[str, float]:
        if name == "t1":
            return {"Q0": 169_100.0, "Q1": 80_800.0}
        if name == "t2_echo":
            return {"Q0": 53_500.0, "Q1": 57_000.0}
        raise KeyError(name)


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
        angle_slope: float | None = None,
        contrasts: dict[int, float] | None = None,
        signal_function: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ) -> None:
        self.root = root
        self.angle_slope = angle_slope or np.pi / (2.0 * root)
        self.contrasts = contrasts or {1: 0.95, 3: 0.8, 5: 0.65, 7: 0.5}
        self.signal_function = signal_function
        self.calls: list[dict[str, Any]] = []

    def sweep_parameter(self, **kwargs: Any) -> Result:
        self.calls.append(kwargs)
        amplitudes = np.asarray(kwargs["sweep_range"], dtype=float)
        repetition = int(kwargs["title"].rsplit("N=", maxsplit=1)[1].rstrip(")"))
        assert kwargs["initial_states"]["Q0"] == "0"
        target_state = kwargs["initial_states"]["Q1"]
        sign = 1.0 if target_state == "0" else -1.0
        base_signal = (
            self.contrasts[repetition]
            * np.cos(
                repetition * (np.pi / 2.0 + self.angle_slope * (amplitudes - self.root))
            )
            if self.signal_function is None
            else self.signal_function(amplitudes, repetition)
        )
        signal = sign * base_signal
        return Result(data={"Q1": SimpleNamespace(normalized=signal)})


class _Experiment:
    def __init__(
        self,
        *,
        root: float = 0.5,
        angle_slope: float | None = None,
        contrasts: dict[int, float] | None = None,
        pi_amplitude: float = 0.25,
        cancel_amplitude: float = 0.08,
        signal_function: Callable[[np.ndarray, int], np.ndarray] | None = None,
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
            angle_slope=angle_slope,
            contrasts=contrasts,
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
            system_manager=SimpleNamespace(config_loader=_ConfigLoader()),
        )


def test_calibration_uses_default_repeated_gate_sweeps_and_finds_zx90() -> None:
    """The default calibration should jointly fit N=1,3,5 target-state signals."""
    contrasts = {1: 0.95, 3: 0.8, 5: 0.65}
    exp = _Experiment(root=0.5, angle_slope=3.2, contrasts=contrasts)

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
        np.linspace(0.42, 0.58, 33),
        rtol=0.0,
        atol=1e-12,
    )
    assert calibration["angle_slope"] == pytest.approx(3.2, abs=1e-8)
    assert calibration["contrasts"] == pytest.approx(contrasts, abs=1e-8)
    assert calibration["repetitions"] == (1, 3, 5)
    assert result.data["target_state_order"] == ("0", "1")
    assert set(result.data["signals"]) == {1, 3, 5}
    assert len(exp.measurement_service.calls) == 6
    assert [call["initial_states"] for call in exp.measurement_service.calls] == [
        {"Q0": "0", "Q1": "0"},
        {"Q0": "0", "Q1": "1"},
        {"Q0": "0", "Q1": "0"},
        {"Q0": "0", "Q1": "1"},
        {"Q0": "0", "Q1": "0"},
        {"Q0": "0", "Q1": "1"},
    ]
    for repetition, call in zip(
        (1, 3, 5), exp.measurement_service.calls[::2], strict=True
    ):
        schedule = call["sequence"](0.5)
        assert schedule.duration == pytest.approx(288.0 * repetition, abs=1e-12)


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


def test_calibration_accepts_explicit_n7_repetition() -> None:
    """An explicit N=7 request should participate in the joint fit."""
    exp = _Experiment(root=0.5)

    result = calibrate_pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        duration=128.0,
        repetitions=(1, 3, 5, 7),
        plot=False,
    )

    assert result.data["fit"]["repetitions"] == (1, 3, 5, 7)
    assert set(result.data["signals"]) == {1, 3, 5, 7}
    assert len(exp.measurement_service.calls) == 8


def test_calibration_rejects_unidentifiable_joint_fit() -> None:
    """Flat zero-contrast traces should not produce an arbitrary calibration."""
    exp = _Experiment(
        signal_function=lambda amplitudes, _repetition: np.zeros_like(amplitudes)
    )

    with pytest.raises(ValueError, match="not identifiable"):
        calibrate_pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            duration=128.0,
            plot=False,
        )


def test_calibration_rejects_zx90_amplitude_at_sweep_boundary() -> None:
    """A boundary-constrained amplitude should not be accepted as calibrated."""
    exp = _Experiment(root=0.5)
    exp.measurement_service.root = 0.62

    with pytest.raises(ValueError, match="sweep boundary"):
        calibrate_pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            duration=128.0,
            plot=False,
        )


@pytest.mark.parametrize("repetitions", [(), (1, 2, 3), (1, 3, 3), (3, 1)])
def test_calibration_rejects_invalid_repetitions(
    repetitions: tuple[int, ...],
) -> None:
    """Repetitions should be nonempty, unique, increasing positive odd integers."""
    exp = _Experiment()

    with pytest.raises((TypeError, ValueError), match="repetitions"):
        calibrate_pi_pair_rotary_zx90(
            exp,  # type: ignore[arg-type]
            "Q0",
            "Q1",
            duration=128.0,
            repetitions=repetitions,
            plot=False,
        )


def test_calibration_predicts_aligned_duration_and_recenters_amplitude() -> None:
    """Automatic duration alignment should be reflected in the amplitude prediction."""
    exp = _Experiment(root=0.5)
    rate = exp.ctx.calib_note.params["zx_rotation_rate"]
    aligned_root = 1.0 / (8.0 * rate * (160.0 - 8.0))
    exp.measurement_service.root = aligned_root
    exp.measurement_service.angle_slope = np.pi / (2.0 * aligned_root)

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
        np.linspace(0.39, 0.55, 33),
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


def test_calibration_prints_summary_and_plots_fit_and_schedule(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful calibration should show its fit, schedule, and text summary."""
    exp = _Experiment(root=0.5)
    shown_figures: list[go.Figure] = []
    schedule_plots: list[dict[str, Any]] = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: shown_figures.append(figure))
    monkeypatch.setattr(
        PulseSchedule,
        "plot",
        lambda _schedule, **kwargs: schedule_plots.append(kwargs),
    )

    result = calibrate_pi_pair_rotary_zx90(
        exp,  # type: ignore[arg-type]
        "Q0",
        "Q1",
        duration=128.0,
        plot=True,
    )

    output = capsys.readouterr().out
    assert "Calibrated CR parameters:" in output
    assert "Rotary amplitude" in output
    assert "ZX90 coherence limit:" in output
    assert result.figure is shown_figures[0]
    assert result.figure.layout.annotations[0].showarrow is True
    assert len(result.figure.data) == 6
    assert schedule_plots == [
        {
            "title": "Pi-pair rotary ZX90 sequence : Q0-Q1",
            "show_physical_pulse": True,
        }
    ]
    assert result.data["coherence_limit"]["gate_time"] == pytest.approx(
        result.data["pulse_schedule"].duration,
        abs=1e-12,
    )
