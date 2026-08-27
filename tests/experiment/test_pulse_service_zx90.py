"""Tests for calibrated ZX90 pulse construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.experiment.experiment import Experiment
from qubex.experiment.services import pulse_service as pulse_service_module
from qubex.experiment.services.pulse_service import PulseService
from qubex.pulse import Arbitrary, PulseSchedule, VirtualZ


def _stored_cr_param(**updates: float) -> dict[str, float | str]:
    param: dict[str, float | str] = {
        "target": "Q00-Q01",
        "duration": 96.0,
        "ramptime": 16.0,
        "cr_amplitude": 0.4,
        "cr_phase": 0.1,
        "cr_beta": 0.0,
        "cancel_amplitude": 0.2,
        "cancel_phase": 0.3,
        "cancel_beta": 0.0,
        "rotary_amplitude": 0.05,
        "zx_rotation_rate": 0.002,
    }
    param.update(updates)
    return param


def _make_service(stored: dict[str, float | str] | None) -> PulseService:
    ctx = SimpleNamespace(
        calib_note=SimpleNamespace(get_cr_param=lambda *_args, **_kwargs: stored),
        calibration_valid_days=7,
    )
    return PulseService(experiment_context=cast(Any, ctx))


def _fake_cr_schedule(**kwargs: object) -> PulseSchedule:
    control = cast(str, kwargs["control_qubit"])
    target = cast(str, kwargs["target_qubit"])
    with PulseSchedule([control, f"{control}-{target}", target]) as schedule:
        schedule.add(control, Arbitrary([1.0]))
        schedule.add(f"{control}-{target}", Arbitrary([1.0]))
        schedule.add(target, Arbitrary([1.0]))
    return schedule


def test_zx90_uses_new_stored_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored rotary-Y, detuning, and local-Z values should all affect ZX90 construction."""
    captured: dict[str, object] = {}

    def fake_cross_resonance(**kwargs: object) -> PulseSchedule:
        captured.update(kwargs)
        return _fake_cr_schedule(**kwargs)

    monkeypatch.setattr(
        pulse_service_module,
        "CrossResonance",
        fake_cross_resonance,
    )
    stored = _stored_cr_param(
        rotary_y=-0.07,
        cr_detuning=0.012,
        control_frame_z=0.11,
        target_frame_z=-0.22,
    )
    service = _make_service(stored)

    schedule = service.zx90(
        "Q00",
        "Q01",
        x180=Arbitrary([1.0]),
    )

    expected_cancel = (
        cast(float, stored["cancel_amplitude"])
        * np.exp(1j * cast(float, stored["cancel_phase"]))
        + cast(float, stored["rotary_amplitude"])
        + 1j * cast(float, stored["rotary_y"])
    )
    assert captured["cancel_amplitude"] == pytest.approx(abs(expected_cancel))
    assert captured["cancel_phase"] == pytest.approx(np.angle(expected_cancel))
    assert captured["cr_detuning"] == pytest.approx(0.012)
    assert schedule.get_sequence("Q00").final_frame_shift == pytest.approx(-0.11)
    assert schedule.get_sequence("Q01").final_frame_shift == pytest.approx(0.22)
    assert schedule.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(0.22)
    for label in ("Q00", "Q01", "Q00-Q01"):
        assert isinstance(schedule.get_sequence(label).flattened_elements[-1], VirtualZ)
    repeated = schedule.repeated(3)
    assert repeated.get_sequence("Q00").final_frame_shift == pytest.approx(-0.33)
    assert repeated.get_sequence("Q01").final_frame_shift == pytest.approx(0.66)
    assert repeated.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(0.66)


def test_zx90_defaults_new_parameters_for_legacy_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy CR records should build unchanged with all new parameters defaulted to zero."""
    captured: dict[str, object] = {}

    def fake_cross_resonance(**kwargs: object) -> PulseSchedule:
        captured.update(kwargs)
        return _fake_cr_schedule(**kwargs)

    monkeypatch.setattr(
        pulse_service_module,
        "CrossResonance",
        fake_cross_resonance,
    )
    stored = _stored_cr_param()
    service = _make_service(stored)

    schedule = service.zx90(
        "Q00",
        "Q01",
        x180=Arbitrary([1.0]),
    )

    expected_cancel = cast(float, stored["cancel_amplitude"]) * np.exp(
        1j * cast(float, stored["cancel_phase"])
    ) + cast(float, stored["rotary_amplitude"])
    assert captured["cancel_amplitude"] == pytest.approx(abs(expected_cancel))
    assert captured["cancel_phase"] == pytest.approx(np.angle(expected_cancel))
    assert captured["cr_detuning"] == pytest.approx(0.0)
    assert schedule.get_sequence("Q00").final_frame_shift == pytest.approx(0.0)
    assert schedule.get_sequence("Q01").final_frame_shift == pytest.approx(0.0)
    assert schedule.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(0.0)


def test_cr_pulse_registry_uses_new_stored_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibrated CR registry should apply every extended stored parameter."""
    captured: dict[str, object] = {}

    def fake_cross_resonance(**kwargs: object) -> PulseSchedule:
        captured.update(kwargs)
        return _fake_cr_schedule(**kwargs)

    monkeypatch.setattr(
        pulse_service_module,
        "CrossResonance",
        fake_cross_resonance,
    )
    stored = _stored_cr_param(
        rotary_y=-0.07,
        cr_detuning=0.012,
        control_frame_z=0.11,
        target_frame_z=-0.22,
    )
    ctx = SimpleNamespace(
        cr_targets=("Q00-Q01",),
        cr_pair=lambda _label: ("Q00", "Q01"),
        calib_note=SimpleNamespace(get_cr_param=lambda *_args, **_kwargs: stored),
    )
    service = PulseService(experiment_context=cast(Any, ctx))
    monkeypatch.setattr(service, "x180", lambda _target: Arbitrary([1.0]))

    schedule = service.cr_pulse["Q00-Q01"]

    expected_cancel = (
        cast(float, stored["cancel_amplitude"])
        * np.exp(1j * cast(float, stored["cancel_phase"]))
        + cast(float, stored["rotary_amplitude"])
        + 1j * cast(float, stored["rotary_y"])
    )
    assert captured["cancel_amplitude"] == pytest.approx(abs(expected_cancel))
    assert captured["cancel_phase"] == pytest.approx(np.angle(expected_cancel))
    assert captured["cr_detuning"] == pytest.approx(0.012)
    assert schedule.get_sequence("Q00").final_frame_shift == pytest.approx(-0.11)
    assert schedule.get_sequence("Q01").final_frame_shift == pytest.approx(0.22)
    assert schedule.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(0.22)


def test_zx90_explicit_new_parameters_override_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit new ZX90 parameters should override their stored calibration values."""
    captured: dict[str, object] = {}

    def fake_cross_resonance(**kwargs: object) -> PulseSchedule:
        captured.update(kwargs)
        return _fake_cr_schedule(**kwargs)

    monkeypatch.setattr(
        pulse_service_module,
        "CrossResonance",
        fake_cross_resonance,
    )
    service = _make_service(
        _stored_cr_param(
            rotary_y=0.01,
            cr_detuning=0.002,
            control_frame_z=0.03,
            target_frame_z=0.04,
        )
    )

    schedule = service.zx90(
        "Q00",
        "Q01",
        rotary_y=0.2,
        cr_detuning=-0.005,
        control_frame_z=-0.3,
        target_frame_z=0.4,
        x180=Arbitrary([1.0]),
    )

    expected_cancel = 0.2 * np.exp(0.3j) + 0.05 + 0.2j
    assert captured["cancel_amplitude"] == pytest.approx(abs(expected_cancel))
    assert captured["cancel_phase"] == pytest.approx(np.angle(expected_cancel))
    assert captured["cr_detuning"] == pytest.approx(-0.005)
    assert schedule.get_sequence("Q00").final_frame_shift == pytest.approx(0.3)
    assert schedule.get_sequence("Q01").final_frame_shift == pytest.approx(-0.4)
    assert schedule.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(-0.4)


def test_experiment_zx90_forwards_new_parameters() -> None:
    """Experiment ZX90 should forward every new calibration parameter to PulseService."""
    captured: dict[str, object] = {}
    expected = object()

    def zx90(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    exp = cast(Any, object.__new__(Experiment))
    exp.__dict__["_pulse_service"] = SimpleNamespace(zx90=zx90)

    result = exp.zx90(
        "Q00",
        "Q01",
        rotary_y=0.1,
        cr_detuning=0.002,
        control_frame_z=0.3,
        target_frame_z=-0.4,
    )

    assert result is expected
    assert captured["args"] == ("Q00", "Q01")
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["rotary_y"] == 0.1
    assert kwargs["cr_detuning"] == 0.002
    assert kwargs["control_frame_z"] == 0.3
    assert kwargs["target_frame_z"] == -0.4


def test_zx90_builds_from_complete_explicit_parameters_without_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete explicit pulse parameters should allow ZX90 construction without a stored CR record."""
    captured: dict[str, object] = {}

    def fake_cross_resonance(**kwargs: object) -> PulseSchedule:
        captured.update(kwargs)
        return _fake_cr_schedule(**kwargs)

    monkeypatch.setattr(
        pulse_service_module,
        "CrossResonance",
        fake_cross_resonance,
    )
    service = _make_service(None)

    schedule = service.zx90(
        "Q00",
        "Q01",
        cr_duration=96.0,
        cr_ramptime=16.0,
        cr_amplitude=0.4,
        cr_phase=0.1,
        cr_beta=0.0,
        cancel_amplitude=0.2,
        cancel_phase=0.3,
        cancel_beta=0.0,
        rotary_amplitude=0.05,
        rotary_y=0.06,
        cr_detuning=0.002,
        control_frame_z=0.1,
        target_frame_z=-0.2,
        x180=Arbitrary([1.0]),
    )

    assert captured["cr_amplitude"] == pytest.approx(0.4)
    assert captured["cr_detuning"] == pytest.approx(0.002)
    assert schedule.get_sequence("Q00").final_frame_shift == pytest.approx(-0.1)
    assert schedule.get_sequence("Q01").final_frame_shift == pytest.approx(0.2)
    assert schedule.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(0.2)


def test_zx90_rejects_incomplete_explicit_parameters_without_storage() -> None:
    """Incomplete explicit pulse parameters should not bypass the missing CR calibration error."""
    service = _make_service(None)

    with pytest.raises(ValueError, match="CR parameters for Q00-Q01 are not stored"):
        service.zx90(
            "Q00",
            "Q01",
            cr_duration=96.0,
            cr_amplitude=0.4,
            x180=Arbitrary([1.0]),
        )
