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
    _calculate_zx90_coherence_limit,
    _calculate_zx_signals,
    _calculate_zy_signal,
    _default_cancel_offsets,
    _fit_zero_crossing,
    _measure_stage,
    _report_completed_calibration,
    _resolve_cr_half_duration,
    _root_centered_retry_grid,
    _run_parameter_stage,
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
    def __init__(self) -> None:
        self.validated_targets: list[list[str]] = []

    def validate_rabi_params(self, targets: list[str]) -> None:
        self.validated_targets.append(targets)

    def calc_control_amplitude(self, _target: str, rabi_rate: float) -> float:
        return rabi_rate / 0.02

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
            experiment_system=SimpleNamespace(
                measurement_defaults={"execution": {"shot_interval_ns": 4096.0}}
            ),
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


def test_zero_crossing_refits_points_near_the_preliminary_root() -> None:
    """A preliminary root between points 11 and 12 should exclude points 1--3."""
    values = np.arange(17, dtype=np.float64)
    signal = values - 10.0
    signal[:3] -= 136.0 / 13.0

    analysis = _fit_zero_crossing(
        sweep_values=values,
        error_signal=signal,
    )

    assert analysis.root == pytest.approx(10.0, abs=1e-14)
    assert analysis.root_bracket == pytest.approx((0.0, 16.0), abs=1e-15)
    assert analysis.fit_slope == pytest.approx(1.0, abs=1e-14)
    assert_allclose(
        analysis.fitted_signal,
        values - 10.0,
        rtol=0.0,
        atol=1e-14,
    )


def test_zero_crossing_uses_the_original_sweep_span_after_expansion() -> None:
    """An expanded fit should keep the distance cutoff from the original sweep."""
    values = np.arange(25, dtype=np.float64)
    signal = values - 18.0
    signal[:11] -= 100.0 / 11.0

    analysis = _fit_zero_crossing(
        sweep_values=values,
        error_signal=signal,
        reference_sweep_span=16.0,
    )

    assert analysis.root == pytest.approx(18.0, abs=1e-14)
    assert analysis.root_bracket == pytest.approx((0.0, 24.0), abs=1e-15)
    assert_allclose(
        analysis.fitted_signal,
        values - 18.0,
        rtol=0.0,
        atol=1e-14,
    )


def test_zero_crossing_fails_when_the_fitted_root_is_outside_the_sweep() -> None:
    """An out-of-range fitted root should fail unless expansion was requested."""
    with pytest.raises(ValueError, match="outside"):
        _fit_zero_crossing(
            sweep_values=[0.0, 0.1, 0.2],
            error_signal=[1.0, 2.0, 3.0],
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


def test_default_cancel_offsets_use_target_rabi_rate_and_requested_rotation() -> None:
    """Each edge should add 0.16/N per ZX90 and 0.32 over the sequence."""
    exp = _Experiment()
    error_amplification_n = 4
    half_effective_duration = 192.0 - 16.0

    offsets = _default_cancel_offsets(
        cast(Any, exp),
        TARGET,
        cr_half_duration=192.0,
        cr_ramptime=16.0,
        error_amplification_n=error_amplification_n,
    )

    assert offsets.size == 17
    assert offsets[0] == pytest.approx(-offsets[-1], abs=1e-15)
    previous_edge_rabi_rate = 0.2 / (
        2.0 * np.pi * error_amplification_n * 2.0 * half_effective_duration
    )
    previous_edge_offset = previous_edge_rabi_rate / 0.02
    assert offsets[-1] == pytest.approx(0.8 * previous_edge_offset, abs=1e-15)
    edge_rabi_rate = offsets[-1] * 0.02
    one_gate_rotation = 2.0 * np.pi * edge_rabi_rate * 2.0 * half_effective_duration
    sequence_rotation = one_gate_rotation * 2 * error_amplification_n
    assert one_gate_rotation == pytest.approx(
        0.16 / error_amplification_n,
        abs=1e-15,
    )
    assert sequence_rotation == pytest.approx(0.32, abs=1e-15)
    assert exp.pulse.validated_targets == [[TARGET]]


@pytest.mark.parametrize("root", [0.62, 0.38])
def test_retry_sweep_preserves_grid_and_centers_on_extrapolated_root(
    root: float,
) -> None:
    """A retry should shift, rather than enlarge, the original sweep grid."""
    values = 0.5 * np.linspace(0.84, 1.16, 17)

    retry = _root_centered_retry_grid(
        values,
        root=root,
        bounds=(0.0, 1.0),
    )

    assert (retry[0] + retry[-1]) / 2 == pytest.approx(root, abs=1e-15)
    assert retry[-1] - retry[0] == pytest.approx(values[-1] - values[0], abs=1e-15)
    assert_allclose(np.diff(retry), np.diff(values), rtol=0.0, atol=1e-15)


def test_retry_sweep_preserves_a_nonuniform_custom_grid() -> None:
    """A shifted custom sweep should retain every original adjacent interval."""
    values = np.array([0.4, 0.44, 0.5])
    retry = _root_centered_retry_grid(
        values,
        root=0.25,
        bounds=(0.0, 1.0),
    )

    assert_allclose(retry, [0.2, 0.24, 0.3], rtol=0.0, atol=1e-15)


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
        expected_amplitude * np.linspace(0.84, 1.16, 17),
        rtol=0.0,
        atol=1e-15,
    )
    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["duration_resolution"][
        "predicted_cr_amplitude"
    ] == pytest.approx(expected_amplitude, abs=1e-15)
    assert calibration["status"] == "failed"


def test_default_sweeps_and_context_interval_are_used_in_the_public_workflow(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """Defaults should use the specified point counts and context shot interval."""
    calls: list[dict[str, Any]] = []
    fitted_roots: dict[str, float] = {}

    def fake_measure_stage(**kwargs: Any) -> Any:
        calls.append(kwargs)
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        stage = kwargs["stage"]
        if values.size > 2:
            fitted_roots[stage] = float(values.mean())
        return _mock_stage_measurement(
            sweep_values=values,
            stage=stage,
            root=fitted_roots[stage],
        )

    exp = _Experiment()
    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    result = calibrate_srre_zx90(
        cast(Any, exp),
        CONTROL,
        TARGET,
        cr_half_duration=192.0,
        srre_ramp_time=20.0,
        srre_calibration=srre_calibration,
        plot=False,
    )

    fit_calls = [call for call in calls if len(call["sweep_values"]) > 2]
    assert [call["stage"] for call in fit_calls] == [
        "zx",
        "zy",
        "iy",
        "ix",
        "zx",
    ]
    assert [len(call["sweep_values"]) for call in fit_calls] == [
        17,
        17,
        17,
        17,
        17,
    ]
    assert_allclose(
        fit_calls[0]["sweep_values"],
        fit_calls[0]["sweep_values"].mean() * np.linspace(0.84, 1.16, 17),
        rtol=0.0,
        atol=1e-15,
    )
    assert_allclose(
        fit_calls[1]["sweep_values"] - 0.2,
        np.linspace(-0.16, 0.16, 17),
        rtol=0.0,
        atol=1e-15,
    )
    assert all(call["shot_interval"] == pytest.approx(4096.0) for call in calls)
    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["requested_fine_rounds"] == 1
    assert calibration["completed_fine_rounds"] == 1
    assert calibration["status"] == "completed"


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
    ("samples", "message"),
    [
        ([], "at least one sample"),
        ([np.nan], "finite samples"),
        ([1.1], "hardware limit"),
    ],
)
def test_fine_stage_rejects_invalid_reference_pulse_samples(
    monkeypatch: pytest.MonkeyPatch,
    samples: list[float],
    message: str,
) -> None:
    """Fine-stage reference pulses must be safe finite hardware waveforms."""

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
        y180=lambda _target: Arbitrary(
            samples,
            sampling_period=SAMPLING_PERIOD,
        ),
    )
    monkeypatch.setattr(calibration_module, "_build_srre_cross_resonance", fake_builder)

    with pytest.raises(ValueError, match=message):
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
) -> Any:
    values = np.asarray(sweep_values, dtype=float)
    error_signal = values - root
    diagnostics = {"ix_from_z": np.zeros(values.size)} if stage == "zx" else {}
    return SimpleNamespace(
        sweep_values=values,
        state_values=np.zeros((4, values.size)),
        error_signal=error_signal,
        diagnostic_signals=diagnostics,
        raw_results=(),
    )


def test_valid_root_is_accepted_without_fresh_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid fitted root should be accepted without another measurement."""
    measurement_count = 0

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal measurement_count
        measurement_count += 1
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        return SimpleNamespace(
            sweep_values=values,
            state_values=np.zeros((4, values.size)),
            error_signal=np.array([-0.1, 0.1]),
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
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
    )

    assert measurement_count == 1
    assert stage.data["status"] == "accepted"
    assert "verification" not in stage.data
    assert stage.accepted_calibration["cr_amplitude"] == pytest.approx(0.4)


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
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
    )

    assert measurement_count == 1
    assert stage.data["status"] == "failed"
    assert "positive" in stage.data["reason"]
    assert stage.accepted_calibration["cr_amplitude"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("stage_name", "parameter"),
    [("iy", "cancel_y"), ("ix", "cancel_x")],
)
def test_cancel_stage_retries_around_root_without_remeasuring_overlap(
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    parameter: str,
) -> None:
    """A retry should measure only centered-grid points outside the first range."""
    measured_ranges: list[np.ndarray] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        measured_ranges.append(values)
        return _mock_stage_measurement(
            sweep_values=values,
            stage=kwargs["stage"],
            root=0.15,
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    stage = _run_parameter_stage(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        accepted={
            "cr_amplitude": 0.3,
            "cr_phase": 0.2,
            "cancel_x": 0.0,
            "cancel_y": 0.0,
        },
        stage=cast(Any, stage_name),
        sweep_values=[-0.1, 0.0, 0.1],
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
        allow_root_centered_retry=True,
    )

    assert len(measured_ranges) == 2
    assert_allclose(measured_ranges[0], [-0.1, 0.0, 0.1], atol=1e-15)
    assert_allclose(measured_ranges[1], [0.15, 0.25], atol=1e-15)
    assert_allclose(
        stage.data["fit_history"][1]["sweep_values"],
        [0.1, 0.15, 0.25],
        atol=1e-15,
    )
    assert stage.data["root"] == pytest.approx(0.15, abs=1e-15)
    assert stage.accepted_calibration[parameter] == pytest.approx(0.15, abs=1e-15)


def test_cr_stage_retries_on_same_width_grid_without_remeasuring_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR retry points should lie outside the first sweep's overlapping part."""
    measured_ranges: list[np.ndarray] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        measured_ranges.append(values)
        return _mock_stage_measurement(
            sweep_values=values,
            stage=kwargs["stage"],
            root=0.62,
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    stage = _run_parameter_stage(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        accepted={
            "cr_amplitude": 0.5,
            "cr_phase": 0.2,
            "cancel_x": 0.1,
            "cancel_y": 0.0,
        },
        stage="zx",
        sweep_values=0.5 * np.linspace(0.84, 1.16, 17),
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
        allow_root_centered_retry=True,
    )

    assert [values.size for values in measured_ranges] == [17, 12]
    assert measured_ranges[1][0] == pytest.approx(0.59, abs=1e-15)
    assert measured_ranges[1][-1] == pytest.approx(0.70, abs=1e-15)
    retry_fit_values = stage.data["fit_history"][1]["sweep_values"]
    assert retry_fit_values[0] == pytest.approx(0.54, abs=1e-15)
    assert retry_fit_values[-1] == pytest.approx(0.70, abs=1e-15)
    assert stage.data["root"] == pytest.approx(0.62, abs=1e-15)
    assert stage.accepted_calibration["cr_amplitude"] == pytest.approx(0.62, abs=1e-15)


@pytest.mark.parametrize(
    ("root", "expected_lower", "expected_upper"),
    [(0.37, 0.37, 0.53), (0.03, -0.13, 0.03)],
)
def test_phase_stage_retries_around_root_without_remeasuring_overlap(
    monkeypatch: pytest.MonkeyPatch,
    root: float,
    expected_lower: float,
    expected_upper: float,
) -> None:
    """CR phase should retry outside the overlap on a root-centered grid."""
    measured_ranges: list[np.ndarray] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        measured_ranges.append(values)
        return _mock_stage_measurement(
            sweep_values=values,
            stage=kwargs["stage"],
            root=root,
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    stage = _run_parameter_stage(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        accepted={
            "cr_amplitude": 0.3,
            "cr_phase": 0.2,
            "cancel_x": 0.1,
            "cancel_y": 0.0,
        },
        stage="zy",
        sweep_values=0.2 + np.linspace(-0.16, 0.16, 17),
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
        allow_root_centered_retry=True,
    )

    assert len(measured_ranges) == 2
    assert_allclose(
        measured_ranges[0],
        0.2 + np.linspace(-0.16, 0.16, 17),
        rtol=0.0,
        atol=1e-15,
    )
    assert measured_ranges[1][0] == pytest.approx(expected_lower, abs=1e-15)
    assert measured_ranges[1][-1] == pytest.approx(expected_upper, abs=1e-15)
    assert stage.data["root"] == pytest.approx(root, abs=1e-15)
    assert stage.accepted_calibration["cr_phase"] == pytest.approx(root, abs=1e-15)
    assert len(stage.data["fit_history"]) == 2


@pytest.mark.parametrize(("root", "expected_edge"), [(0.37, 0.53), (0.03, -0.13)])
def test_public_workflow_centers_phase_retry_on_extrapolated_root(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
    root: float,
    expected_edge: float,
) -> None:
    """The default phase retry should reuse overlap and center on the first root."""
    fitted_roots: dict[str, float] = {}
    phase_fit_ranges: list[np.ndarray] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        stage = kwargs["stage"]
        if values.size > 2:
            fitted_roots[stage] = root if stage == "zy" else float(values.mean())
            if stage == "zy":
                phase_fit_ranges.append(values)
        return _mock_stage_measurement(
            sweep_values=values,
            stage=stage,
            root=fitted_roots[stage],
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
        cancel_y_offsets=[-0.1, 0.0, 0.1],
        cancel_x_offsets=[-0.1, 0.0, 0.1],
        fine_rounds=1,
        plot=False,
    )

    assert len(phase_fit_ranges) == 2
    assert_allclose(
        phase_fit_ranges[0],
        0.2 + np.linspace(-0.16, 0.16, 17),
        rtol=0.0,
        atol=1e-15,
    )
    if root > 0.2:
        assert phase_fit_ranges[1][-1] == pytest.approx(expected_edge, abs=1e-15)
    else:
        assert phase_fit_ranges[1][0] == pytest.approx(expected_edge, abs=1e-15)
    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["fine_rounds"][0]["phase_stage"]["status"] == "accepted"


def test_stage_plot_contains_measurements_and_linear_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each plotted CR calibration fit should show points and its fitted line."""
    shown_figures: list[Any] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        return _mock_stage_measurement(
            sweep_values=kwargs["sweep_values"],
            stage=kwargs["stage"],
            root=0.1,
        )

    monkeypatch.setattr(calibration_module, "_measure_stage", fake_measure_stage)
    monkeypatch.setattr(
        calibration_module.go.Figure,
        "show",
        lambda figure: shown_figures.append(figure),
    )
    stage = _run_parameter_stage(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        accepted={
            "cr_amplitude": 0.3,
            "cr_phase": 0.2,
            "cancel_x": 0.1,
            "cancel_y": 0.0,
        },
        stage="zy",
        sweep_values=[0.0, 0.1, 0.2],
        error_amplification_n=1,
        scale_cancellation_with_cr=True,
        n_shots=128,
        shot_interval=1024.0,
        plot=True,
    )

    assert stage.data["status"] == "accepted"
    assert len(shown_figures) == 1
    assert [trace.mode for trace in shown_figures[0].data] == ["markers", "lines"]
    assert len(shown_figures[0].layout.annotations) == 1
    annotation = shown_figures[0].layout.annotations[0]
    assert annotation.x == pytest.approx(0.1, abs=1e-15)
    assert annotation.y == pytest.approx(0.0, abs=1e-15)
    assert annotation.text == "root: 0.1"
    assert annotation.showarrow is True
    assert shown_figures[0].layout.yaxis.title.text == "S_ZY"
    assert shown_figures[0].layout.template.layout.width == 600
    assert shown_figures[0].layout.template.layout.height == 300


def test_completed_calibration_prints_summary_coherence_and_plots_sequence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    srre_calibration: dict[str, Any],
) -> None:
    """The final report should match the standard ZX90 calibration summary."""
    plot_calls: list[dict[str, Any]] = []
    zx90 = SimpleNamespace(
        get_sampled_sequences=lambda: {TARGET: np.zeros(512)},
        plot=lambda **kwargs: plot_calls.append(kwargs),
    )
    monkeypatch.setattr(
        calibration_module,
        "_build_srre_cross_resonance",
        lambda *_args, **_kwargs: zx90,
    )
    exp = _Experiment()
    exp.ctx.system_manager = SimpleNamespace(
        config_loader=SimpleNamespace(
            load_param_data=lambda name: {
                "t1": {CONTROL: 100_000.0, TARGET: 120_000.0},
                "t2_echo": {CONTROL: 80_000.0, TARGET: 90_000.0},
            }[name]
        )
    )
    calibration = {
        "cr_half_duration": 192.0,
        "cr_ramptime": 16.0,
        "cr_amplitude": 0.42,
        "cr_phase": 0.1,
        "cr_beta": 0.01,
        "cancel_amplitude": 0.08,
        "cancel_phase": -0.2,
        "cancel_beta": -0.02,
        "srre_calibration": srre_calibration,
    }

    coherence_limit = _report_completed_calibration(
        cast(Any, exp),
        CONTROL,
        TARGET,
        calibration=calibration,
        plot=True,
    )

    output = capsys.readouterr().out
    assert "Calibrated CR parameters:" in output
    assert "CR amplitude     : 0.420000" in output
    assert "SRRE amplitude   : 0.300000" in output
    assert "ZX90 coherence limit:" in output
    assert "Gate time       : 512 ns" in output
    assert "Coherence limit :" in output
    assert coherence_limit["gate_time"] == pytest.approx(512.0)
    assert 0.0 < float(coherence_limit["fidelity"]) < 1.0
    assert plot_calls == [
        {
            "title": f"SRRE ZX90 sequence : {CONTROL}-{TARGET}",
            "show_physical_pulse": True,
        }
    ]


@pytest.mark.parametrize("invalid_t1", [0.0, np.nan, "invalid"])
def test_invalid_optional_coherence_data_does_not_discard_calibration(
    invalid_t1: Any,
) -> None:
    """Malformed optional T1/T2 data should suppress only the final estimate."""
    exp = _Experiment()
    exp.ctx.system_manager = SimpleNamespace(
        config_loader=SimpleNamespace(
            load_param_data=lambda name: {
                "t1": {CONTROL: invalid_t1, TARGET: 120_000.0},
                "t2_echo": {CONTROL: 80_000.0, TARGET: 90_000.0},
            }[name]
        )
    )

    assert (
        _calculate_zx90_coherence_limit(
            cast(Any, exp),
            CONTROL,
            TARGET,
            gate_time=512.0,
        )
        == {}
    )


def test_calibrate_srre_zx90_orchestrates_all_stages_and_returns_contract(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """The public workflow should accept fitted roots in the documented order."""
    exp = _Experiment()
    original_srre = deepcopy(srre_calibration)
    calls: list[dict[str, Any]] = []
    zx_fit_index = 0
    last_zx_root = 0.4

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal last_zx_root, zx_fit_index
        calls.append(kwargs)
        stage = kwargs["stage"]
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        if stage == "zx" and values.size > 2:
            root = (0.41, 0.42)[zx_fit_index]
            last_zx_root = root
            zx_fit_index += 1
        elif stage == "zx":
            root = last_zx_root
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
        fine_rounds=1,
        n_shots=128,
        shot_interval=2048.0,
        plot=False,
    )

    assert isinstance(result, Result)
    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert calibration["control_qubit"] == CONTROL
    assert calibration["target_qubit"] == TARGET
    assert calibration["cr_half_duration"] == pytest.approx(192.0)
    assert calibration["cr_amplitude"] == pytest.approx(0.42, abs=1e-14)
    assert calibration["cr_phase"] == pytest.approx(0.1, abs=1e-14)
    assert calibration["cancel_x"] == pytest.approx(0.08 * 0.42 / 0.41)
    assert calibration["cancel_y"] == pytest.approx(0.02 * 0.42 / 0.41)
    assert calibration["cancel_amplitude"] == pytest.approx(
        abs(calibration["cancel_x"] + 1j * calibration["cancel_y"])
    )
    assert calibration["cancel_phase"] == pytest.approx(
        np.angle(calibration["cancel_x"] + 1j * calibration["cancel_y"])
    )
    assert calibration["srre_calibration"]["amplitude"] == 0.3
    assert calibration["requested_fine_rounds"] == 1
    assert calibration["completed_fine_rounds"] == 1
    assert calibration["coherence_limit"] == {}
    assert len(calibration["fine_rounds"]) == 1
    assert "final_verification" not in calibration
    assert "converged" not in calibration
    assert calibration["status"] == "completed"
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
        "zy",
        "iy",
        "ix",
        "zx",
    ]
    assert all(call["error_amplification_n"] == 2 for call in calls)
    assert calls[0]["n_shots"] == 128
    assert all(call["n_shots"] == 128 for call in calls)
    assert exp.note.calls == [f"{CONTROL}-{TARGET}"]
    assert srre_calibration.keys() == original_srre.keys()
    for key, expected in original_srre.items():
        if isinstance(expected, np.ndarray):
            assert_allclose(srre_calibration[key], expected, rtol=0.0, atol=0.0)
        else:
            assert srre_calibration[key] == expected
    assert exp.note.cr_param["cr_amplitude"] == 0.4


def test_calibration_runs_exactly_the_requested_number_of_fine_rounds(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """Fine rounds should be controlled only by the user-provided count."""
    fine_phase_sweeps = 0
    stages: list[str] = []

    def fake_measure_stage(**kwargs: Any) -> Any:
        nonlocal fine_phase_sweeps
        values = np.asarray(kwargs["sweep_values"], dtype=float)
        stage = kwargs["stage"]
        stages.append(stage)
        root = float((values[0] + values[-1]) / 2)
        if stage == "zy" and values.size > 2:
            fine_phase_sweeps += 1
        return _mock_stage_measurement(
            sweep_values=values,
            stage=stage,
            root=root,
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
        fine_rounds=2,
        plot=False,
    )

    calibration = cast(dict[str, Any], result.data["srre_cr_calibration"])
    assert fine_phase_sweeps == 2
    assert stages == ["zx", "zy", "iy", "ix", "zx", "zy", "iy", "ix", "zx"]
    assert calibration["requested_fine_rounds"] == 2
    assert calibration["completed_fine_rounds"] == 2
    assert len(calibration["fine_rounds"]) == 2
    assert all(item["status"] == "completed" for item in calibration["fine_rounds"])
    assert calibration["status"] == "completed"


def test_failed_root_preserves_last_accepted_parameters(
    monkeypatch: pytest.MonkeyPatch,
    srre_calibration: dict[str, Any],
) -> None:
    """A failed fine-stage root should retain the accepted initial CR state."""
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
    assert calibration["requested_fine_rounds"] == 1
    assert calibration["completed_fine_rounds"] == 0
    assert calibration["fine_rounds"][0]["status"] == "failed"
    assert calibration["fine_rounds"][0]["phase_stage"]["status"] == "failed"
    assert (
        "slope is too small" in calibration["fine_rounds"][0]["phase_stage"]["reason"]
    )


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


@pytest.mark.parametrize(
    ("fine_rounds", "error_type"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_fine_rounds_must_be_a_positive_integer(
    srre_calibration: dict[str, Any],
    fine_rounds: Any,
    error_type: type[Exception],
) -> None:
    """The fixed fine-round count should reject zero and non-integers."""
    with pytest.raises(error_type, match="fine_rounds"):
        calibrate_srre_zx90(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            cr_half_duration=192.0,
            srre_ramp_time=20.0,
            srre_calibration=srre_calibration,
            fine_rounds=fine_rounds,
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


def test_zero_cr_range_fails_before_automatic_srre_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR sweep values should be positive like the builder's reference amplitude."""

    def fail_if_calibrated(*_args: Any, **_kwargs: Any) -> Result:
        raise AssertionError("SRRE calibration must not start for invalid input")

    monkeypatch.setattr(calibration_module, "calibrate_srre", fail_if_calibrated)

    with pytest.raises(ValueError, match="must be positive"):
        calibrate_srre_zx90(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            cr_half_duration=192.0,
            srre_ramp_time=20.0,
            cr_amplitude_range=[0.0, 0.1],
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
