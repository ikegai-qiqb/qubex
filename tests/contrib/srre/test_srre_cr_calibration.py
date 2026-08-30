"""Tests for SRRE-assisted ZX90 calibration."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import qubex.contrib.experiment.srre_cr_calibration as calibration_module
from qubex.contrib import calibrate_srre_zx90
from qubex.contrib.experiment.srre_cr_calibration import (
    _build_stage_sequence,
    _calculate_ix_signal,
    _calculate_iy_signal,
    _calculate_zx_signals,
    _calculate_zy_signal,
    _find_closest_zero_crossing,
    _measure_stage,
    _resolve_cr_half_duration,
    _run_parameter_stage,
    _verification_summary,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Arbitrary, PulseSchedule

CONTROL = "Q00"
TARGET = "Q01"
SAMPLING_PERIOD = 1.0


class _CalibrationNote:
    def __init__(self) -> None:
        self.cr_param = {
            "target": f"{CONTROL}-{TARGET}",
            "duration": 0.0,
            "ramptime": 16.0,
            "cr_amplitude": 0.4,
            "cr_phase": 0.2,
            "cr_beta": 0.01,
            "cancel_amplitude": np.hypot(0.1, -0.05),
            "cancel_phase": np.angle(0.1 - 0.05j),
            "cancel_beta": -0.02,
            "rotary_amplitude": 0.0,
            "zx_rotation_rate": 0.01,
        }
        self.calls: list[str] = []

    def get_cr_param(self, target: str) -> dict[str, Any] | None:
        self.calls.append(target)
        return self.cr_param


class _PulseService:
    def x180(self, _target: str) -> Arbitrary:
        return Arbitrary([0.8], sampling_period=SAMPLING_PERIOD)

    def y180(self, _target: str) -> Arbitrary:
        return Arbitrary([0.6j], sampling_period=SAMPLING_PERIOD)

    def x90(self, _target: str) -> Arbitrary:
        return Arbitrary([0.3], sampling_period=SAMPLING_PERIOD)


class _Experiment:
    def __init__(self) -> None:
        self.note = _CalibrationNote()
        self.pulse: Any = _PulseService()
        self.measurement_service: Any = SimpleNamespace()
        self.ctx = SimpleNamespace(
            calib_note=self.note,
            measurement=SimpleNamespace(sampling_period=SAMPLING_PERIOD),
            util=SimpleNamespace(resolve_sampling_period=float),
        )


@pytest.fixture
def srre_calibration() -> dict[str, Any]:
    """Return reusable SRRE metadata matching the test CR geometry."""
    return {
        "target": TARGET,
        "amplitude": 0.3,
        "predicted_amplitude": 0.29,
        "rabi_rate": 0.006,
        "block_duration": 192.0,
        "ramp_time": 20.0,
        "sampling_period": SAMPLING_PERIOD,
        "positive_lobe_angle": np.pi,
        "phi_pred": np.pi / 2,
        "analysis_angle": np.pi,
        "probe_detuning": 0.001,
        "repetitions": 1,
        "f0_predicted": 0.0j,
        "f1_predicted": 0.0j,
        "root_bracket": (0.28, 0.31),
        "fit_slope": 1.0,
        "amplitude_range": np.array([0.28, 0.29, 0.31]),
        "signal_plus": np.array([-0.02, -0.01, 0.01]),
        "signal_minus": np.array([0.02, 0.01, -0.01]),
        "differential_signal": np.array([-0.02, -0.01, 0.01]),
    }


def test_zx_signal_uses_the_documented_four_state_signs() -> None:
    """ZX90 signals should use sums and differences of control-conditioned pairs."""
    signals = _calculate_zx_signals([0.8, 0.2, -0.4, -0.8])

    assert signals.zx == pytest.approx(0.25, abs=1e-15)
    assert signals.ix_from_z == pytest.approx(0.05, abs=1e-15)


def test_zy_signal_uses_the_documented_four_state_signs() -> None:
    """CR phase calibration should subtract the two target-state differentials."""
    assert _calculate_zy_signal([0.8, 0.2, -0.4, -0.8]) == pytest.approx(
        0.05, abs=1e-15
    )


def test_iy_signal_uses_the_documented_four_state_signs() -> None:
    """Cancellation-Y calibration should add the two target-state differentials."""
    assert _calculate_iy_signal([0.8, 0.2, -0.4, -0.8]) == pytest.approx(
        0.25, abs=1e-15
    )


def test_ix_signal_uses_the_documented_four_state_signs() -> None:
    """Cancellation-X calibration should add the two target-state differentials."""
    assert _calculate_ix_signal([0.8, 0.2, -0.4, -0.8]) == pytest.approx(
        0.25, abs=1e-15
    )


def test_signal_functions_reject_missing_or_nonfinite_states() -> None:
    """Every stage signal should require exactly four finite state measurements."""
    with pytest.raises(ValueError, match="exactly four"):
        _calculate_zx_signals([1.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="finite"):
        _calculate_zy_signal([1.0, 0.0, np.nan, -1.0])


def test_signal_functions_reject_complex_states() -> None:
    """Complex state values should fail instead of losing their imaginary part."""
    with pytest.raises(TypeError, match="real array"):
        _calculate_zx_signals(np.array([1.0 + 0.1j, 0.0, 0.0, -1.0]))


def test_zero_crossing_selects_the_root_closest_to_current_parameter() -> None:
    """Multiple brackets should select the local crossing nearest the current value."""
    analysis = _find_closest_zero_crossing(
        sweep_values=[0.0, 0.2, 0.4, 0.6],
        error_signal=[-1.0, 1.0, -1.0, 1.0],
        reference=0.48,
    )

    assert analysis.root == pytest.approx(0.5, abs=1e-15)
    assert analysis.root_bracket == pytest.approx((0.4, 0.6), abs=1e-15)
    assert analysis.fit_slope == pytest.approx(10.0, abs=1e-14)


def test_zero_crossing_fails_without_a_safe_measured_bracket() -> None:
    """An unbracketed calibration signal should not produce a fallback parameter."""
    with pytest.raises(ValueError, match="does not bracket"):
        _find_closest_zero_crossing(
            sweep_values=[0.0, 0.1, 0.2],
            error_signal=[1.0, 2.0, 3.0],
            reference=0.1,
        )


def test_duration_resolution_aligns_prediction_and_validates_geometry() -> None:
    """Predicted CR duration should align to 16 ns and preserve two SRRE lobes."""
    resolution = _resolve_cr_half_duration(
        requested_duration=None,
        cr_amplitude=0.4,
        zx_rotation_rate=0.01,
        cr_ramptime=16.0,
        srre_ramp_time=8.0,
        sampling_period=1.0,
    )

    assert resolution.resolved_duration == pytest.approx(48.0, abs=1e-15)
    assert resolution.predicted_cr_amplitude == pytest.approx(0.390625, abs=1e-15)
    assert resolution.srre_lobe_duration == pytest.approx(24.0, abs=1e-15)
    assert resolution.srre_flat_time == pytest.approx(8.0, abs=1e-15)
    assert resolution.source == "zx_rate_prediction"

    with pytest.raises(ValueError, match="duration unit"):
        _resolve_cr_half_duration(
            requested_duration=50.0,
            cr_amplitude=0.4,
            zx_rotation_rate=0.01,
            cr_ramptime=16.0,
            srre_ramp_time=8.0,
            sampling_period=1.0,
        )
    with pytest.raises(ValueError, match="SRRE ramp"):
        _resolve_cr_half_duration(
            requested_duration=32.0,
            cr_amplitude=0.4,
            zx_rotation_rate=0.01,
            cr_ramptime=16.0,
            srre_ramp_time=9.0,
            sampling_period=1.0,
        )


def test_duration_resolution_rejects_negative_zx_rotation_rate() -> None:
    """A negative stored ZX rate should not be silently treated as positive ZX90."""
    with pytest.raises(ValueError, match="positive"):
        _resolve_cr_half_duration(
            requested_duration=None,
            cr_amplitude=0.4,
            zx_rotation_rate=-0.01,
            cr_ramptime=16.0,
            srre_ramp_time=8.0,
            sampling_period=1.0,
        )


def test_explicit_duration_centers_default_cr_sweep_on_predicted_zx90_amplitude(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """An explicit duration should recenter the initial sweep using the stored ZX rate."""
    measured_ranges: list[np.ndarray] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        measured_ranges.append(values)
        return SimpleNamespace(
            sweep_values=values,
            state_values=np.zeros((4, values.size)),
            error_signal=np.ones(values.size),
            diagnostic_signals={"ix_from_z": np.zeros(values.size)},
            raw_results=(),
        )

    srre_calibration["block_duration"] = 96.0
    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    result = calibrate_srre_zx90(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        cr_half_duration=96.0,
        srre_ramp_time=20.0,
        srre_calibration=srre_calibration,
        plot=False,
    )

    expected_amplitude = 1.0 / (8.0 * 0.01 * (96.0 - 16.0))
    assert_allclose(
        measured_ranges[0],
        expected_amplitude * np.linspace(0.9, 1.1, 5),
        rtol=0.0,
        atol=1e-15,
    )
    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["duration_resolution"][
        "predicted_cr_amplitude"
    ] == pytest.approx(expected_amplitude, abs=1e-15)
    assert calibration["status"] == "failed"


@pytest.mark.parametrize(
    ("stage", "control_state", "expected_echo", "expected_srre", "interleaved"),
    [
        ("zx", "0", True, True, None),
        ("zy", "0", True, False, 0.6j),
        ("iy", "0", False, False, 0.6j),
        ("ix", "0", False, True, 0.3),
        ("ix", "1", False, True, -0.3),
    ],
)
def test_stage_sequences_fix_gate_configuration_and_interleaved_pulse(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    control_state: str,
    expected_echo: bool,
    expected_srre: bool,
    interleaved: complex | None,
) -> None:
    """Each calibration stage should use its specified gate and analysis pulse."""
    calls: list[tuple[bool, bool]] = []

    def fake_builder(*_args: Any, **kwargs: Any) -> PulseSchedule:
        calls.append((kwargs["echo"], kwargs["include_srre"]))
        with PulseSchedule([f"{CONTROL}-{TARGET}", TARGET]) as schedule:
            schedule.add(
                f"{CONTROL}-{TARGET}",
                Arbitrary([0.2], sampling_period=SAMPLING_PERIOD),
            )
            schedule.add(
                TARGET,
                Arbitrary([0.1], sampling_period=SAMPLING_PERIOD),
            )
        return schedule

    monkeypatch.setattr(calibration_module, "_build_srre_cross_resonance", fake_builder)
    schedule = _build_stage_sequence(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        calibration={
            "candidate": 1.0,
            "srre_calibration": {"sampling_period": SAMPLING_PERIOD},
        },
        stage=cast(Any, stage),
        control_state=control_state,
        error_amplification_n=2,
    )

    assert calls == [(expected_echo, expected_srre)]
    sampled = schedule.get_sampled_sequences()[TARGET]
    if interleaved is None:
        assert_allclose(sampled, [0.1], rtol=0.0, atol=0.0)
    else:
        assert_allclose(
            sampled,
            np.tile([0.1, interleaved], 4),
            rtol=0.0,
            atol=1e-15,
        )


def test_fine_stage_rejects_reference_pulse_on_another_sampling_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fine-stage analysis pulses should use the calibrated SRRE sampling grid."""

    def fake_builder(*_args: Any, **_kwargs: Any) -> PulseSchedule:
        with PulseSchedule([f"{CONTROL}-{TARGET}", TARGET]) as schedule:
            schedule.add(
                f"{CONTROL}-{TARGET}",
                Arbitrary([0.2], sampling_period=SAMPLING_PERIOD),
            )
            schedule.add(
                TARGET,
                Arbitrary([0.1], sampling_period=SAMPLING_PERIOD),
            )
        return schedule

    exp = _Experiment()
    exp.pulse = SimpleNamespace(
        y180=lambda _target: Arbitrary([0.6j], sampling_period=2.0),
    )
    monkeypatch.setattr(calibration_module, "_build_srre_cross_resonance", fake_builder)

    with pytest.raises(ValueError, match="sampling period"):
        _build_stage_sequence(
            cast(Any, exp),
            CONTROL,
            TARGET,
            calibration={"srre_calibration": {"sampling_period": SAMPLING_PERIOD}},
            stage="zy",
            control_state="0",
            error_amplification_n=1,
        )


@pytest.mark.parametrize(
    ("stage", "target_states", "parameter"),
    [
        ("zx", ("0", "1"), "cr_amplitude"),
        ("zy", ("+", "-"), "cr_phase"),
        ("iy", ("+", "-"), "cancel_y"),
        ("ix", ("+i", "-i"), "cancel_x"),
    ],
)
def test_measure_stage_collects_the_four_states_and_calculates_error_signal(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    target_states: tuple[str, str],
    parameter: str,
) -> None:
    """Stage measurement should preserve state order and return its signed signal."""
    root = 0.45
    values = np.array([0.4, 0.5])
    calls: list[dict[str, Any]] = []

    def fake_build_stage_sequence(*_args: Any, **kwargs: Any) -> PulseSchedule:
        calls.append(kwargs)
        with PulseSchedule([TARGET]) as schedule:
            schedule.add(
                TARGET,
                Arbitrary([0.1], sampling_period=SAMPLING_PERIOD),
            )
        return schedule

    class _MeasurementService:
        def __init__(self) -> None:
            self.initial_states: list[dict[str, str]] = []

        def sweep_parameter(self, **kwargs: Any) -> Any:
            initial_states = kwargs["initial_states"]
            self.initial_states.append(initial_states)
            sweep = np.asarray(kwargs["sweep_range"], dtype=float)
            for candidate in sweep:
                kwargs["sequence"](candidate)
            error = sweep - root
            control_state = initial_states[CONTROL]
            target_state = initial_states[TARGET]
            target_sign = 1.0 if target_state == target_states[0] else -1.0
            control_sign = -1.0 if stage == "zy" and control_state == "1" else 1.0
            normalized = target_sign * control_sign * error
            return SimpleNamespace(
                data={TARGET: SimpleNamespace(normalized=normalized)}
            )

    exp = _Experiment()
    exp.measurement_service = _MeasurementService()
    monkeypatch.setattr(
        calibration_module,
        "_build_stage_sequence",
        fake_build_stage_sequence,
    )
    measurement = _measure_stage(
        exp=cast(Any, exp),
        control_qubit=CONTROL,
        target_qubit=TARGET,
        calibration={
            "cr_amplitude": 0.4,
            "cr_phase": 0.2,
            "cancel_x": 0.1,
            "cancel_y": -0.05,
        },
        stage=cast(Any, stage),
        sweep_values=values,
        error_amplification_n=2,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
    )

    assert exp.measurement_service.initial_states == [
        {CONTROL: "0", TARGET: target_states[0]},
        {CONTROL: "0", TARGET: target_states[1]},
        {CONTROL: "1", TARGET: target_states[0]},
        {CONTROL: "1", TARGET: target_states[1]},
    ]
    assert_allclose(measurement.error_signal, values - root, rtol=0.0, atol=1e-15)
    assert len(calls) == 4 * values.size
    assert_allclose(
        [call["calibration"][parameter] for call in calls[: values.size]],
        values,
        rtol=0.0,
        atol=0.0,
    )
    if stage == "zx":
        assert_allclose(
            measurement.diagnostic_signals["ix_from_z"],
            0.0,
            rtol=0.0,
            atol=0.0,
        )


def _mock_stage_measurement(
    *,
    sweep_values: Any,
    stage: str,
    root: float,
    candidate_signal: float = 0.0,
) -> Any:
    values = np.asarray(sweep_values, dtype=float)
    error_signal = values - root
    if values.size <= 2 and candidate_signal != 0.0:
        error_signal = np.array([0.1, candidate_signal])
    diagnostics = {"ix_from_z": np.zeros(values.size)} if stage == "zx" else {}
    return SimpleNamespace(
        sweep_values=values,
        state_values=np.zeros((4, values.size)),
        error_signal=error_signal,
        diagnostic_signals=diagnostics,
        raw_results=(),
    )


def test_verification_rejects_a_distinct_candidate_that_worsens_the_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worse candidate should be rejected even when it is inside tolerance."""
    measurement_count = 0

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal measurement_count
        measurement_count += 1
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        if measurement_count == 1:
            error_signal = np.array([-0.1, 0.1])
        else:
            error_signal = np.array([0.001, 0.01])
        return SimpleNamespace(
            sweep_values=values,
            state_values=np.zeros((4, values.size)),
            error_signal=error_signal,
            diagnostic_signals={"ix_from_z": np.zeros(values.size)},
            raw_results=(),
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    accepted = {
        "cr_amplitude": 0.3,
        "cr_phase": 0.2,
        "cancel_x": 0.1,
        "cancel_y": -0.05,
    }

    stage = _run_parameter_stage(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        accepted=accepted,
        stage="zx",
        sweep_values=[0.3, 0.5],
        root_reference=0.3,
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        verification_n_shots=256,
        shot_interval=1024.0,
        plot=False,
    )

    assert stage.data["status"] == "failed"
    assert "did not improve" in stage.data["reason"]
    assert stage.accepted_calibration["cr_amplitude"] == pytest.approx(0.3)


def test_zx_stage_rejects_a_nonpositive_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZX90 calibration should never accept a zero drive amplitude."""
    measurement_count = 0

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal measurement_count
        measurement_count += 1
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        return SimpleNamespace(
            sweep_values=values,
            state_values=np.zeros((4, values.size)),
            error_signal=values,
            diagnostic_signals={"ix_from_z": np.zeros(values.size)},
            raw_results=(),
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    accepted = {
        "cr_amplitude": 0.3,
        "cr_phase": 0.2,
        "cancel_x": 0.1,
        "cancel_y": -0.05,
    }

    stage = _run_parameter_stage(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        accepted=accepted,
        stage="zx",
        sweep_values=[0.0, 0.1],
        root_reference=0.05,
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        verification_n_shots=256,
        shot_interval=1024.0,
        plot=False,
    )

    assert measurement_count == 1
    assert stage.data["status"] == "failed"
    assert "positive" in stage.data["reason"]
    assert stage.accepted_calibration["cr_amplitude"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("candidate_ix", "expected_converged"),
    [(0.01, True), (0.03, False)],
)
def test_ix_from_z_diagnostic_uses_measurement_tolerance(
    candidate_ix: float,
    expected_converged: bool,
) -> None:
    """Small IX diagnostic fluctuations should not force another fine round."""

    def stage(signal: float) -> dict[str, Any]:
        return {"verification": {"candidate_signal": signal}}

    angle_stage = {
        "verification": {
            "candidate_signal": 0.0,
            "current_parameter": 0.4,
            "candidate_parameter": 0.5,
            "sweep_values": np.array([0.4, 0.5]),
            "diagnostic_signals": {"ix_from_z": np.array([0.0, candidate_ix])},
        }
    }

    summary = _verification_summary(
        angle_stage,
        stage(0.0),
        stage(0.0),
        stage(0.0),
    )

    assert summary["checks"]["s_ix_from_z_not_worse"] is expected_converged
    assert summary["converged"] is expected_converged


def test_calibrate_srre_zx90_orchestrates_all_stages_and_returns_contract(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """The public workflow should accept verified roots in the documented order."""
    exp = _Experiment()
    original_srre = deepcopy(srre_calibration)
    calls: list[dict[str, Any]] = []
    zx_sweep_index = 0

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal zx_sweep_index
        calls.append(kwargs)
        stage = kwargs["stage"]
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        if stage == "zx" and values.size > 2:
            root = (0.42, 0.44)[zx_sweep_index]
            zx_sweep_index += 1
        elif stage == "zx":
            root = (0.42, 0.44)[zx_sweep_index - 1]
        else:
            root = {"zy": 0.1, "iy": 0.02, "ix": 0.08}[stage]
        assert kwargs["calibration"]["srre_calibration"]["amplitude"] == 0.3
        assert kwargs["calibration"]["cr_half_duration"] == 192.0
        return _mock_stage_measurement(
            sweep_values=values,
            stage=stage,
            root=root,
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    result = calibrate_srre_zx90(
        cast(Any, exp),
        CONTROL,
        TARGET,
        cr_half_duration=192.0,
        srre_ramp_time=20.0,
        srre_calibration=srre_calibration,
        cr_amplitude_range=[0.35, 0.42, 0.5],
        cr_phase_offsets=[-0.2, -0.1, 0.0, 0.1, 0.2],
        cancel_y_offsets=[-0.1, -0.05, 0.0, 0.05, 0.1],
        cancel_x_offsets=[-0.1, -0.05, 0.0, 0.05, 0.1],
        error_amplification_n=2,
        max_fine_rounds=2,
        n_shots=128,
        verification_n_shots=512,
        shot_interval=2048.0,
        plot=False,
    )

    assert isinstance(result, Result)
    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["control_qubit"] == CONTROL
    assert calibration["target_qubit"] == TARGET
    assert calibration["cr_half_duration"] == pytest.approx(192.0)
    assert calibration["cr_amplitude"] == pytest.approx(0.44, abs=1e-14)
    assert calibration["cr_phase"] == pytest.approx(0.1, abs=1e-14)
    assert calibration["cancel_x"] == pytest.approx(0.08 * 0.44 / 0.42)
    assert calibration["cancel_y"] == pytest.approx(0.02 * 0.44 / 0.42)
    assert calibration["cancel_amplitude"] == pytest.approx(
        abs(calibration["cancel_x"] + 1j * calibration["cancel_y"])
    )
    assert calibration["cancel_phase"] == pytest.approx(
        np.angle(calibration["cancel_x"] + 1j * calibration["cancel_y"])
    )
    assert calibration["srre_calibration"]["amplitude"] == 0.3
    assert calibration["fine_round_count"] == 1
    assert len(calibration["fine_rounds"]) == 1
    assert calibration["converged"] is True
    assert calibration["status"] == "converged"
    assert calibration["initial_angle_stage"]["configuration"] == {
        "echo": True,
        "include_srre": True,
    }
    round_data = calibration["fine_rounds"][0]
    assert round_data["phase_stage"]["configuration"] == {
        "echo": True,
        "include_srre": False,
    }
    assert round_data["cancel_y_stage"]["configuration"] == {
        "echo": False,
        "include_srre": False,
    }
    assert round_data["cancel_x_stage"]["configuration"] == {
        "echo": False,
        "include_srre": True,
    }
    assert round_data["final_angle_stage"]["configuration"] == {
        "echo": True,
        "include_srre": True,
    }
    assert [call["stage"] for call in calls] == [
        "zx",
        "zx",
        "zy",
        "zy",
        "iy",
        "iy",
        "ix",
        "ix",
        "zx",
        "zx",
    ]
    assert all(call["error_amplification_n"] == 2 for call in calls)
    assert calls[0]["n_shots"] == 128
    assert calls[1]["n_shots"] == 512
    assert exp.note.calls == [f"{CONTROL}-{TARGET}"]
    assert srre_calibration.keys() == original_srre.keys()
    for key, expected in original_srre.items():
        if isinstance(expected, np.ndarray):
            assert_allclose(srre_calibration[key], expected, rtol=0.0, atol=0.0)
        else:
            assert srre_calibration[key] == expected
    assert exp.note.cr_param["cr_amplitude"] == 0.4


def test_calibration_runs_a_second_round_only_when_verification_needs_it(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """Persistent verified phase error should stop exactly at max_fine_rounds."""
    fine_phase_sweeps = 0

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal fine_phase_sweeps
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        stage = kwargs["stage"]
        root = float((values[0] + values[-1]) / 2)
        if stage == "zy" and values.size > 2:
            root += 0.01
        if stage == "zy" and values.size > 2:
            fine_phase_sweeps += 1
        candidate_signal = 0.05 if stage == "zy" and values.size <= 2 else 0.0
        return _mock_stage_measurement(
            sweep_values=values,
            stage=stage,
            root=root,
            candidate_signal=candidate_signal,
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    result = calibrate_srre_zx90(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        cr_half_duration=192.0,
        srre_ramp_time=20.0,
        srre_calibration=srre_calibration,
        cr_amplitude_range=[0.3, 0.4, 0.5],
        max_fine_rounds=2,
        plot=False,
    )

    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert fine_phase_sweeps == 2
    assert calibration["fine_round_count"] == 2
    assert len(calibration["fine_rounds"]) == 2
    assert calibration["converged"] is False
    assert calibration["status"] == "max_fine_rounds_reached"


def test_failed_root_or_verification_preserves_last_accepted_parameters(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """A failed fine-stage root should retain the verified initial CR state."""
    zx_sweep = True

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal zx_sweep
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        stage = kwargs["stage"]
        if stage == "zx":
            root = 0.42
            if values.size > 2:
                zx_sweep = False
            return _mock_stage_measurement(
                sweep_values=values,
                stage=stage,
                root=root,
            )
        assert not zx_sweep
        measurement = _mock_stage_measurement(
            sweep_values=values,
            stage=stage,
            root=float(values.mean()),
        )
        measurement.error_signal = np.ones(values.size)
        return measurement

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    result = calibrate_srre_zx90(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        cr_half_duration=192.0,
        srre_ramp_time=20.0,
        srre_calibration=srre_calibration,
        cr_amplitude_range=[0.35, 0.42, 0.5],
        plot=False,
    )

    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["cr_amplitude"] == pytest.approx(0.42, abs=1e-14)
    assert calibration["cr_phase"] == pytest.approx(0.2, abs=1e-14)
    assert calibration["status"] == "failed"
    assert calibration["fine_rounds"][0]["phase_stage"]["status"] == "failed"
    assert "does not bracket" in calibration["fine_rounds"][0]["phase_stage"]["reason"]


def test_reused_srre_metadata_must_match_target_and_fixed_geometry(
    srre_calibration: dict[str, Any],
) -> None:
    """Reused SRRE calibration should match target, duration, ramp, and sampling."""
    srre_calibration["block_duration"] = 176.0

    with pytest.raises(ValueError, match="block_duration"):
        calibrate_srre_zx90(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            cr_half_duration=192.0,
            srre_ramp_time=20.0,
            srre_calibration=srre_calibration,
            plot=False,
        )


def test_reused_srre_metadata_rejects_negative_calibrated_amplitude(
    srre_calibration: dict[str, Any],
) -> None:
    """Reused SRRE metadata should retain the non-negative calibration contract."""
    srre_calibration["amplitude"] = -0.3

    with pytest.raises(ValueError, match="positive"):
        calibrate_srre_zx90(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            cr_half_duration=192.0,
            srre_ramp_time=20.0,
            srre_calibration=srre_calibration,
            plot=False,
        )


def test_invalid_fine_offsets_fail_before_any_stage_measurement(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """Invalid fine-stage offsets should be rejected before hardware measurement."""

    def fail_if_measured(**_kwargs: Any) -> Any:
        raise AssertionError("measurement must not start for invalid input")

    monkeypatch.setattr(calibration_module, "_measure_stage", fail_if_measured)

    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_srre_zx90(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            cr_half_duration=192.0,
            srre_ramp_time=20.0,
            srre_calibration=srre_calibration,
            cr_phase_offsets=[0.0, 0.0],
            plot=False,
        )


def test_invalid_cr_range_fails_before_automatic_srre_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid CR sweeps should not trigger one-qubit SRRE measurements."""

    def fail_if_calibrated(*_args: Any, **_kwargs: Any) -> Result:
        raise AssertionError("SRRE calibration must not start for invalid input")

    monkeypatch.setattr(calibration_module, "calibrate_srre", fail_if_calibrated)

    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_srre_zx90(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            cr_half_duration=192.0,
            srre_ramp_time=20.0,
            cr_amplitude_range=[0.5, 0.4],
            plot=False,
        )


def test_srre_is_auto_calibrated_once_when_reuse_data_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """Missing SRRE metadata should trigger the one-qubit calibration stage once."""
    auto_calls: list[dict[str, Any]] = []

    def fake_calibrate_srre(*_args: Any, **kwargs: Any) -> Result:
        auto_calls.append(kwargs)
        return Result(data={"srre_calibration": srre_calibration})

    def fake_measure_stage(**kwargs: Any) -> Any:
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        return _mock_stage_measurement(
            sweep_values=values,
            stage=kwargs["stage"],
            root=float(values.mean()),
        )

    monkeypatch.setattr(calibration_module, "calibrate_srre", fake_calibrate_srre)
    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    result = calibrate_srre_zx90(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        cr_half_duration=192.0,
        srre_ramp_time=20.0,
        srre_amplitude_range=[0.2, 0.3, 0.4],
        probe_detuning=0.001,
        cr_amplitude_range=[0.3, 0.4, 0.5],
        plot=False,
    )

    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert len(auto_calls) == 1
    assert auto_calls[0]["block_duration"] == pytest.approx(192.0)
    assert auto_calls[0]["ramp_time"] == pytest.approx(20.0)
    assert auto_calls[0]["amplitude_range"] == [0.2, 0.3, 0.4]
    assert auto_calls[0]["probe_detuning"] == pytest.approx(0.001)
    assert calibration["srre_stage"]["status"] == "calibrated"
