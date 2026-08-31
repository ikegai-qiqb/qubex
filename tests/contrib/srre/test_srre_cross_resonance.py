"""Tests for SRRE cross-resonance gate construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import qubex.contrib.experiment.srre_cross_resonance as cross_resonance_module
from qubex.contrib import srre_rzx
from qubex.contrib.experiment.srre_cross_resonance import _build_srre_cross_resonance
from qubex.experiment.services.pulse_service import PulseService
from qubex.pulse import Arbitrary, PhaseShift, PulseArray, PulseSchedule, Waveform

CONTROL = "Q00"
TARGET = "Q01"
CR_LABEL = f"{CONTROL}-{TARGET}"
SAMPLING_PERIOD = 1.0
HALF_SAMPLES = 8


class _PulseService:
    def __init__(self) -> None:
        self.x180_calls: list[str] = []

    def x180(self, target: str) -> Arbitrary:
        self.x180_calls.append(target)
        return Arbitrary([0.8, 0.8], sampling_period=SAMPLING_PERIOD)


class _Experiment:
    def __init__(self) -> None:
        self.pulse = _PulseService()
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=SAMPLING_PERIOD),
            util=SimpleNamespace(resolve_sampling_period=float),
        )


@pytest.fixture
def calibration() -> dict[str, Any]:
    """Return a compact valid SRRE-CR calibration contract."""
    return {
        "control_qubit": CONTROL,
        "target_qubit": TARGET,
        "cr_half_duration": 8.0,
        "cr_ramptime": 0.0,
        "cr_amplitude": 0.2,
        "cr_phase": np.pi / 2,
        "cr_beta": 0.0,
        "cancel_x": 0.1,
        "cancel_y": -0.05,
        "cancel_beta": 0.0,
        "srre_calibration": {
            "target": TARGET,
            "amplitude": 0.3,
            "block_duration": 8.0,
            "ramp_time": 0.0,
            "sampling_period": SAMPLING_PERIOD,
        },
    }


def _sampled_halves(schedule: PulseSchedule) -> tuple[dict[str, np.ndarray], int]:
    sampled = schedule.get_sampled_sequences()
    echo_samples = 2
    second_half_start = HALF_SAMPLES + echo_samples
    return sampled, second_half_start


def test_srre_rzx_builds_signed_echo_halves_and_two_control_echoes(
    calibration: dict[str, Any],
) -> None:
    """Final RZX should negate every half sample around two control echoes."""
    exp = _Experiment()

    schedule = srre_rzx(
        cast(Any, exp),
        CONTROL,
        TARGET,
        np.pi / 2,
        calibration=calibration,
    )

    sampled, second_half_start = _sampled_halves(schedule)
    expected_cr = np.full(HALF_SAMPLES, 0.2j)
    expected_srre = np.r_[np.full(4, 0.3), np.full(4, -0.3)]
    expected_target = 0.1 - 0.05j + expected_srre

    assert schedule.labels == [CONTROL, CR_LABEL, TARGET]
    assert_allclose(sampled[CR_LABEL][:HALF_SAMPLES], expected_cr, rtol=0.0, atol=1e-15)
    assert_allclose(
        sampled[TARGET][:HALF_SAMPLES], expected_target, rtol=0.0, atol=1e-15
    )
    assert_allclose(
        sampled[CR_LABEL][second_half_start : second_half_start + HALF_SAMPLES],
        -expected_cr,
        rtol=0.0,
        atol=1e-15,
    )
    assert_allclose(
        sampled[TARGET][second_half_start : second_half_start + HALF_SAMPLES],
        -expected_target,
        rtol=0.0,
        atol=1e-15,
    )
    assert_allclose(sampled[CONTROL][:HALF_SAMPLES], 0.0, rtol=0.0, atol=0.0)
    assert_allclose(
        sampled[CONTROL][HALF_SAMPLES:second_half_start],
        [0.8, 0.8],
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(sampled[CONTROL][-2:], [0.8, 0.8], rtol=0.0, atol=0.0)
    assert exp.pulse.x180_calls == [CONTROL]


@pytest.mark.parametrize(
    ("echo", "include_srre", "expected_labels", "expected_target"),
    [
        (
            True,
            True,
            [CONTROL, CR_LABEL, TARGET],
            0.1 - 0.05j + np.r_[np.full(4, 0.3), np.full(4, -0.3)],
        ),
        (True, False, [CONTROL, CR_LABEL, TARGET], np.full(8, 0.1 - 0.05j)),
        (
            False,
            True,
            [CR_LABEL, TARGET],
            0.1 - 0.05j + np.r_[np.full(4, 0.3), np.full(4, -0.3)],
        ),
        (False, False, [CR_LABEL, TARGET], np.full(8, 0.1 - 0.05j)),
    ],
)
def test_internal_builder_supports_all_calibration_configurations(
    calibration: dict[str, Any],
    echo: bool,
    include_srre: bool,
    expected_labels: list[str],
    expected_target: np.ndarray,
) -> None:
    """Every configuration should contain two halves with the requested echo."""
    exp = _Experiment()
    schedule = _build_srre_cross_resonance(
        cast(Any, exp),
        CONTROL,
        TARGET,
        np.pi / 2,
        calibration=calibration,
        echo=echo,
        include_srre=include_srre,
    )

    sampled = schedule.get_sampled_sequences()

    assert schedule.labels == expected_labels
    assert_allclose(
        sampled[TARGET][:HALF_SAMPLES], expected_target, rtol=0.0, atol=1e-15
    )
    if echo:
        second_half_start = HALF_SAMPLES + 2
        assert_allclose(
            sampled[TARGET][second_half_start : second_half_start + HALF_SAMPLES],
            -expected_target,
            rtol=0.0,
            atol=1e-15,
        )
    else:
        assert sampled[TARGET].size == 2 * HALF_SAMPLES
        assert_allclose(
            sampled[TARGET][HALF_SAMPLES:],
            expected_target,
            rtol=0.0,
            atol=1e-15,
        )
        assert_allclose(
            sampled[CR_LABEL][:HALF_SAMPLES],
            sampled[CR_LABEL][HALF_SAMPLES:],
            rtol=0.0,
            atol=1e-15,
        )
        assert exp.pulse.x180_calls == []


def test_srre_off_builder_does_not_construct_an_unused_srre_waveform(
    monkeypatch: pytest.MonkeyPatch,
    calibration: dict[str, Any],
) -> None:
    """SRRE-off calibration gates should not generate a discarded waveform."""

    def fail_if_called(**_kwargs: Any) -> None:
        raise AssertionError("srre_waveform must not be called when SRRE is off")

    monkeypatch.setattr(cross_resonance_module, "srre_waveform", fail_if_called)

    schedule = _build_srre_cross_resonance(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        np.pi / 2,
        calibration=calibration,
        echo=False,
        include_srre=False,
    )

    assert schedule.get_sampled_sequences()[TARGET].size == 2 * HALF_SAMPLES


@pytest.mark.parametrize("angle", [0.0, -np.pi / 4, np.pi / 4])
def test_angle_scales_cr_and_cancellation_but_not_srre(
    calibration: dict[str, Any], angle: float
) -> None:
    """Finite angles should scale CR and cancellation while SRRE stays fixed."""
    schedule = _build_srre_cross_resonance(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        angle,
        calibration=calibration,
        echo=False,
        include_srre=True,
    )

    sampled = schedule.get_sampled_sequences()
    coefficient = angle / (np.pi / 2)
    expected_srre = np.r_[np.full(4, 0.3), np.full(4, -0.3)]

    expected_cr_half = np.full(HALF_SAMPLES, 0.2j * coefficient)
    expected_target_half = (0.1 - 0.05j) * coefficient + expected_srre
    assert_allclose(
        sampled[CR_LABEL],
        np.tile(expected_cr_half, 2),
        rtol=0.0,
        atol=1e-15,
    )
    assert_allclose(
        sampled[TARGET],
        np.tile(expected_target_half, 2),
        rtol=0.0,
        atol=1e-15,
    )


def test_srre_rzx_accepts_custom_echo_mapping_and_margin(
    calibration: dict[str, Any],
) -> None:
    """A mapped X180 and symmetric margin should define both echo slots."""
    x180 = Arbitrary([0.7, 0.7], sampling_period=SAMPLING_PERIOD)

    schedule = srre_rzx(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        np.pi / 2,
        calibration=calibration,
        x180={CONTROL: x180},
        x180_margin=1.0,
    )

    sampled = schedule.get_sampled_sequences()
    expected_echo = np.array([0.0, 0.7, 0.7, 0.0])
    second_half_start = HALF_SAMPLES + expected_echo.size

    assert_allclose(
        sampled[CONTROL][HALF_SAMPLES:second_half_start],
        expected_echo,
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(sampled[CONTROL][-4:], expected_echo, rtol=0.0, atol=0.0)


def test_srre_rzx_preserves_echo_pulse_array_frame_shifts(
    calibration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested X180 frame shifts should affect the second echo and final frame."""
    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", SAMPLING_PERIOD)
    x180 = PulseArray(
        [
            PhaseShift(np.pi / 2),
            Arbitrary([0.8], sampling_period=SAMPLING_PERIOD),
        ]
    )

    schedule = srre_rzx(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        np.pi / 2,
        calibration=calibration,
        x180=x180,
    )

    sampled = schedule.get_sampled_sequences()[CONTROL]

    assert sampled[HALF_SAMPLES] == pytest.approx(0.8j, abs=1e-15)
    assert sampled[-1] == pytest.approx(-0.8, abs=1e-15)
    assert abs(schedule.get_final_frame_shift(CONTROL)) == pytest.approx(
        np.pi, abs=1e-15
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(target_qubit="Q99"), "target_qubit"),
        (lambda data: data["srre_calibration"].update(target="Q99"), "target"),
        (
            lambda data: data["srre_calibration"].update(block_duration=10.0),
            "block_duration",
        ),
        (
            lambda data: data["srre_calibration"].update(sampling_period=2.0),
            "sampling_period",
        ),
    ],
)
def test_builder_rejects_calibration_metadata_mismatch(
    calibration: dict[str, Any], mutate: Any, message: str
) -> None:
    """Calibration metadata inconsistent with the requested gate should fail."""
    mutate(calibration)

    with pytest.raises(ValueError, match=message):
        _build_srre_cross_resonance(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
            echo=False,
            include_srre=True,
        )


def test_builder_rejects_identical_control_and_target_qubits(
    calibration: dict[str, Any],
) -> None:
    """An RZX gate should require two distinct qubits."""
    calibration["control_qubit"] = TARGET

    with pytest.raises(ValueError, match="must be different"):
        srre_rzx(
            cast(Any, _Experiment()),
            TARGET,
            TARGET,
            np.pi / 2,
            calibration=calibration,
        )


def test_builder_rejects_a_failed_calibration_result(
    calibration: dict[str, Any],
) -> None:
    """A partial calibration must not silently become a production gate."""
    calibration["status"] = "failed"

    with pytest.raises(ValueError, match="status must be 'completed'"):
        srre_rzx(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
        )


def test_builder_resolves_missing_measurement_sampling_period(
    calibration: dict[str, Any],
) -> None:
    """A missing measurement value should still resolve and verify hardware dt."""
    exp = _Experiment()
    exp.ctx.measurement.sampling_period = None
    exp.ctx.util.resolve_sampling_period = lambda _value: 2.0

    with pytest.raises(ValueError, match="sampling_period"):
        _build_srre_cross_resonance(
            cast(Any, exp),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
            echo=False,
            include_srre=True,
        )


def test_builder_validates_srre_calibration_when_srre_is_disabled(
    calibration: dict[str, Any],
) -> None:
    """SRRE-off calibration schedules should still reject invalid SRRE metadata."""
    calibration["srre_calibration"]["amplitude"] = 1.1

    with pytest.raises(ValueError, match="absolute amplitude"):
        _build_srre_cross_resonance(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
            echo=False,
            include_srre=False,
        )


def test_builder_rejects_negative_calibrated_srre_amplitude(
    calibration: dict[str, Any],
) -> None:
    """A calibration contract should not encode a negative SRRE root."""
    calibration["srre_calibration"]["amplitude"] = -0.3

    with pytest.raises(ValueError, match="positive"):
        srre_rzx(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
        )


@pytest.mark.parametrize("amplitude", [0.0, -0.2])
def test_builder_rejects_nonpositive_calibrated_cr_amplitude(
    calibration: dict[str, Any], amplitude: float
) -> None:
    """A calibration contract should use a positive reference CR amplitude."""
    calibration["cr_amplitude"] = amplitude

    with pytest.raises(ValueError, match="cr_amplitude must be positive"):
        srre_rzx(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cr_amplitude", 1.1, "CR channel"),
        ("cancel_x", 0.8, "target channel"),
    ],
)
def test_builder_rejects_final_sample_amplitude_overflow(
    calibration: dict[str, Any], field: str, value: float, message: str
) -> None:
    """CR and combined target samples above the hardware limit should fail."""
    calibration[field] = value

    with pytest.raises(ValueError, match=message):
        _build_srre_cross_resonance(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
            echo=False,
            include_srre=True,
        )


def test_builder_rejects_nonfinite_angle_and_incompatible_echo_pulse(
    calibration: dict[str, Any],
) -> None:
    """Non-finite angles and echo pulses on another sampling grid should fail."""
    with pytest.raises(ValueError, match="angle must be finite"):
        srre_rzx(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.nan,
            calibration=calibration,
        )

    with pytest.raises(ValueError, match="sampling period"):
        srre_rzx(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
            x180=Arbitrary([0.8], sampling_period=2.0),
        )

    with pytest.raises(TypeError, match="x180 must be"):
        srre_rzx(
            cast(Any, _Experiment()),
            CONTROL,
            TARGET,
            np.pi / 2,
            calibration=calibration,
            x180=cast(Any, object()),
        )


def test_srre_schedule_is_accepted_as_custom_zx90(
    calibration: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing CNOT builder should consume the schedule as custom ZX90."""
    monkeypatch.setattr(Waveform, "SAMPLING_PERIOD", SAMPLING_PERIOD)
    schedule = srre_rzx(
        cast(Any, _Experiment()),
        CONTROL,
        TARGET,
        np.pi / 2,
        calibration=calibration,
    )
    service = cast(Any, PulseService).__new__(PulseService)
    service._experiment_context = SimpleNamespace(  # noqa: SLF001
        qubits={CONTROL: SimpleNamespace(index=0)}
    )

    cnot = service.cnot(
        CONTROL,
        TARGET,
        zx90=schedule,
        x90=Arbitrary([0.1], sampling_period=SAMPLING_PERIOD),
        only_low_to_high=True,
    )

    assert isinstance(cnot, PulseSchedule)
    for label, expected in schedule.get_sampled_sequences().items():
        assert_allclose(
            cnot.get_sampled_sequences()[label][: expected.size],
            expected,
            rtol=0.0,
            atol=0.0,
        )
