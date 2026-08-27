"""Tests for the contributed fine CR calibration workflow."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.contrib.experiment import fine_cr_calibration as fine_cr
from qubex.experiment.models import CalibrationNote, Result
from qubex.pulse import FlatTop, PulseSchedule


def _params(**updates: float) -> fine_cr.CrGateParameters:
    """Return a valid CR parameter set with optional field replacements."""
    values = {
        "cr_amplitude": 0.4,
        "cr_phase": 0.1,
        "cr_lobe_duration": 160.0,
        "cr_ramptime": 32.0,
        "cr_beta": 0.0,
        "cancel_x": 0.02,
        "cancel_y": -0.01,
        "cancel_beta": 0.0,
        "rotary_x": 0.03,
        "rotary_y": 0.01,
        "cr_detuning": 0.0,
        "control_frame_z": 0.0,
        "target_frame_z": 0.0,
        "zx_rotation_rate": 0.005,
    }
    values.update(updates)
    return fine_cr.CrGateParameters(**values)


def _pulse_experiment() -> Any:
    """Return a minimal experiment that can build explicit CR schedules."""
    return SimpleNamespace(
        x180=lambda _target: FlatTop(duration=32, amplitude=0.2, tau=8)
    )


def _stage_result(
    params: fine_cr.CrGateParameters,
    **metrics: float,
) -> Result:
    """Return a successful characterization result for calibration tests."""
    return Result(
        data={
            "stage": "characterize_ecr_gate",
            "input_params": params,
            "proposed_params": params,
            "converged": True,
            "verified": True,
            "supported": True,
            "status": "success",
            "reason": None,
            "metrics_before": metrics,
            "metrics_after": metrics,
            "uncertainties": {},
            "fit_quality": {"r2": 1.0},
            "sweep": None,
            "raw_results": {},
            "error_angles": metrics,
            "zx_angle": metrics.get("ZX", np.pi / 2),
        }
    )


def _passing_validation(
    exp: Any,
    params: fine_cr.CrGateParameters,
    monkeypatch: pytest.MonkeyPatch,
) -> Result:
    """Return a passing validation bound to the realized candidate waveform."""
    metrics = {
        "ZX": np.pi / 2,
        "ZY": 0.0,
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZZ": 0.0,
    }
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: _stage_result(params, **metrics),
    )
    return fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=False,
        run_irb=False,
        waive_control_local_z=True,
    )


def test_cr_gate_parameters_are_immutable_and_validate_duration() -> None:
    """CR parameters should be immutable and reject an invalid lobe duration."""
    params = _params()

    with pytest.raises(FrozenInstanceError):
        params.cr_amplitude = 0.2  # type: ignore[misc]
    with pytest.raises(ValueError, match="cr_lobe_duration"):
        _params(cr_lobe_duration=32.0, cr_ramptime=32.0)


def test_calibration_options_reject_fractional_repetition_counts() -> None:
    """Repetition counts should not silently truncate fractional values."""
    with pytest.raises(ValueError, match="integers"):
        fine_cr.CrCalibrationOptions(
            repetition_counts=cast(Any, (0, 1.5, 2)),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_shots", 1.5), ("verification_shots", True), ("max_iterations", 2.5)],
)
def test_calibration_options_require_integer_counts(field: str, value: object) -> None:
    """Acquisition counts and iteration limits should be positive integers."""
    with pytest.raises(ValueError, match=field):
        fine_cr.CrCalibrationOptions(**{field: value})  # type: ignore[arg-type]


def test_build_ecr_gate_rejects_boolean_repetitions() -> None:
    """Boolean repetitions should not be accepted as the integer one."""
    with pytest.raises(ValueError, match="repetitions"):
        fine_cr.build_ecr_gate(
            _pulse_experiment(),
            "Q0",
            "Q1",
            params=_params(),
            repetitions=True,
        )


def test_build_ecr_gate_uses_explicit_xy_tones_and_local_frames() -> None:
    """The explicit builder should preserve all channels and supplied parameters."""
    params = _params(control_frame_z=0.02, target_frame_z=-0.03)

    schedule = fine_cr.build_ecr_gate(
        _pulse_experiment(),
        "Q0",
        "Q1",
        params=params,
    )

    assert isinstance(schedule, PulseSchedule)
    assert set(schedule.labels) == {"Q0", "Q1", "Q0-Q1"}
    assert schedule.duration > 2 * params.cr_lobe_duration
    assert schedule.get_final_frame_shift("Q0") == pytest.approx(-0.02, abs=1e-12)
    assert schedule.get_final_frame_shift("Q1") == pytest.approx(0.03, abs=1e-12)
    assert schedule.get_final_frame_shift("Q0-Q1") == pytest.approx(0.03, abs=1e-12)


def test_build_ecr_gate_applies_detuning_only_to_cr_waveforms() -> None:
    """CR detuning should not modulate cancellation, rotary, or X180 waveforms."""
    schedule = fine_cr.build_ecr_gate(
        _pulse_experiment(),
        "Q0",
        "Q1",
        params=_params(cr_detuning=0.002),
    )

    cr_waveforms = schedule.get_sequence("Q0-Q1").get_flattened_waveforms(False)
    target_waveforms = schedule.get_sequence("Q1").get_flattened_waveforms(False)
    control_waveforms = schedule.get_sequence("Q0").get_flattened_waveforms(False)
    assert any(
        pulse.detuning == pytest.approx(0.002, abs=1e-12) for pulse in cr_waveforms
    )
    assert all(
        pulse.detuning == pytest.approx(0.0, abs=1e-12) for pulse in target_waveforms
    )
    assert all(
        pulse.detuning == pytest.approx(0.0, abs=1e-12) for pulse in control_waveforms
    )


def test_characterize_bare_cr_does_not_mutate_calibration_note() -> None:
    """Bare CR characterization should call tomography without touching the note."""
    params = _params(rotary_x=0.0, rotary_y=0.0)
    note = Mock()
    exp: Any = SimpleNamespace(
        ctx=SimpleNamespace(calib_note=note),
        cr_hamiltonian_tomography=Mock(
            return_value=Result(
                data={
                    "coeffs": {
                        "IX": 0.001,
                        "IY": 0.002,
                        "IZ": 0.003,
                        "ZX": 0.020,
                        "ZY": 0.001,
                        "ZZ": 0.002,
                    },
                    "xt_rotation_amplitude_hw": 0.02,
                    "xt_rotation_phase": 0.3,
                    "fit_quality": {"r2": 0.99},
                }
            )
        ),
    )

    result = fine_cr.characterize_bare_cr(
        exp,
        "Q0",
        "Q1",
        params=params,
        time_range=np.linspace(0, 300, 11),
    )

    assert result["converged"] is True
    assert result["coeffs"]["ZX"] == pytest.approx(0.020, abs=1e-12)
    exp.cr_hamiltonian_tomography.assert_called_once()
    note.update_cr_param.assert_not_called()


def test_calibrate_bare_cr_records_rate_when_input_is_already_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already converged bare measurement should refresh the measured ZX rate."""
    params = _params(zx_rotation_rate=cast(Any, None))
    coeffs = {
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZX": 0.02,
        "ZY": 0.0,
        "ZZ": 0.0,
    }
    measurement = Result(
        data={
            "converged": True,
            "fit_quality": {"r2": 0.99},
            "coeffs": coeffs,
        }
    )
    monkeypatch.setattr(
        fine_cr,
        "characterize_bare_cr",
        lambda *_args, **_kwargs: measurement,
    )

    result = fine_cr.calibrate_bare_cr(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        time_range=(0.0, 32.0, 64.0),
    )

    assert result["status"] == "already_converged"
    assert result["proposed_params"].zx_rotation_rate == pytest.approx(
        coeffs["ZX"] / params.cr_amplitude,
        abs=1e-12,
    )


def test_calibrate_bare_cr_rejects_zero_max_iterations() -> None:
    """An explicit zero iteration limit should not fall back to the default."""
    with pytest.raises(ValueError, match="max_iterations"):
        fine_cr.calibrate_bare_cr(
            cast(Any, object()),
            "Q0",
            "Q1",
            initial_params=_params(),
            time_range=(0.0, 32.0, 64.0),
            max_iterations=0,
        )


def test_calibrate_bare_cr_rotates_target_tones_with_cr_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coarse CR phase update should preserve all relative target-tone phases."""
    initial = _params(
        cr_phase=0.1,
        cancel_x=0.02,
        cancel_y=0.03,
        rotary_x=-0.04,
        rotary_y=0.01,
    )
    initial_coeffs = {
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZX": 0.02,
        "ZY": 0.02,
        "ZZ": 0.0,
    }
    corrected_coeffs = {
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZX": np.hypot(0.02, 0.02),
        "ZY": 0.0,
        "ZZ": 0.0,
    }
    measured_params: list[fine_cr.CrGateParameters] = []

    def characterize(
        *_args: object,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        measured_params.append(params)
        coeffs = initial_coeffs if len(measured_params) == 1 else corrected_coeffs
        return Result(
            data={
                "converged": True,
                "fit_quality": {"r2": 0.99},
                "coeffs": coeffs,
                "xt_rotation_amplitude_hw": 0.0,
                "xt_rotation_phase": 0.0,
            }
        )

    monkeypatch.setattr(fine_cr, "characterize_bare_cr", characterize)

    result = fine_cr.calibrate_bare_cr(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=initial,
        time_range=(0.0, 32.0, 64.0),
    )

    phase_delta = -np.angle(complex(initial_coeffs["ZX"], initial_coeffs["ZY"]))
    phase_factor = np.exp(1j * phase_delta)
    proposed = result["proposed_params"]
    assert result["status"] == "success"
    assert proposed.cr_phase == pytest.approx(initial.cr_phase + phase_delta)
    assert proposed.cancel_complex == pytest.approx(
        initial.cancel_complex * phase_factor
    )
    assert proposed.rotary_complex == pytest.approx(
        initial.rotary_complex * phase_factor
    )


def test_scout_cr_operating_points_ranks_only_valid_measurements() -> None:
    """Scouting should rank measured points without committing a candidate."""
    exp: Any = SimpleNamespace(ctx=SimpleNamespace(calib_note=Mock()))
    zx_by_amplitude = {0.2: 0.010, 0.4: 0.025, 0.6: 0.030}

    def characterize(
        *_args: object, params: fine_cr.CrGateParameters, **_kwargs: object
    ) -> Result:
        zx = zx_by_amplitude[params.cr_amplitude]
        return Result(
            data={
                "stage": "characterize_bare_cr",
                "input_params": params,
                "proposed_params": params,
                "converged": True,
                "verified": True,
                "supported": True,
                "status": "success",
                "reason": None,
                "metrics_before": {},
                "metrics_after": {},
                "uncertainties": {},
                "fit_quality": {"r2": 0.99},
                "sweep": None,
                "raw_results": {},
                "coeffs": {
                    "IX": 0.001,
                    "IY": 0.001,
                    "IZ": 0.0,
                    "ZX": zx,
                    "ZY": 0.001,
                    "ZZ": 0.0,
                },
            }
        )

    original = fine_cr.characterize_bare_cr
    fine_cr.characterize_bare_cr = characterize  # type: ignore[assignment]
    try:
        result = fine_cr.scout_cr_operating_points(
            exp,
            "Q0",
            "Q1",
            cr_amplitudes=(0.2, 0.4, 0.6),
            time_range=np.linspace(0, 300, 11),
            initial_params=_params(rotary_x=0.0, rotary_y=0.0),
        )
    finally:
        fine_cr.characterize_bare_cr = original

    assert result["converged"] is True
    assert len(result["candidates"]) == 3
    assert result["recommended_params"].cr_amplitude in zx_by_amplitude
    exp.ctx.calib_note.update_cr_param.assert_not_called()


def test_repeated_ecr_characterization_extracts_conditional_zx_angle() -> None:
    """Repeated tomography should recover an ideal conditional ZX90 angle."""
    counts = np.arange(9, dtype=np.float64)

    def branch(sign: float) -> np.ndarray:
        angles = sign * counts * np.pi / 2
        return np.column_stack(
            [
                np.zeros_like(angles),
                -np.sin(angles),
                np.cos(angles),
            ]
        )

    results = [
        Result(data={"Q1": branch(1.0)}),
        Result(data={"Q1": branch(-1.0)}),
    ]
    exp = _pulse_experiment()
    exp.state_evolution_tomography = Mock(side_effect=results)

    result = fine_cr.characterize_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=_params(),
        repetition_counts=tuple(range(9)),
    )

    assert result["converged"] is True
    assert result["error_angles"]["ZX"] == pytest.approx(np.pi / 2, abs=1e-5)
    assert abs(result["error_angles"]["IX"]) < 1e-5
    assert exp.state_evolution_tomography.call_count == 2


def test_calibrate_ecr_phase_rejects_a_root_outside_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase calibration should not apply an extrapolated zero crossing."""
    params = _params()

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        zy = params.cr_phase + 1.0
        return _stage_result(
            params, ZX=np.pi / 2, ZY=zy, IX=0.0, IY=0.0, IZ=0.0, ZZ=0.0
        )

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_phase(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        phase_offsets=(-0.1, 0.0, 0.1),
    )

    assert result["converged"] is False
    assert result["status"] == "no_interpolated_root"
    assert result["proposed_params"] == params


def test_calibrate_ecr_phase_rotates_all_common_iq_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A common-phase update should rotate CR, cancellation, and rotary together."""
    params = _params(
        cr_phase=0.0, cancel_x=0.02, cancel_y=0.0, rotary_x=0.03, rotary_y=0.0
    )

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        return _stage_result(
            params,
            ZX=np.pi / 2,
            ZY=params.cr_phase - 0.05,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_phase(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        phase_offsets=(-0.1, 0.0, 0.1),
        tolerances=fine_cr.CrCalibrationTolerances(fine_error_angle=1e-6),
    )

    proposed = result["proposed_params"]
    assert proposed.cr_phase == pytest.approx(0.05, abs=1e-8)
    assert proposed.cancel_complex == pytest.approx(0.02 * np.exp(0.05j), abs=1e-8)
    assert proposed.rotary_complex == pytest.approx(0.03 * np.exp(0.05j), abs=1e-8)


def test_calibrate_ecr_phase_requires_the_input_point() -> None:
    """Phase calibration should require zero offset as an observed baseline."""
    with pytest.raises(ValueError, match="must include 0"):
        fine_cr.calibrate_ecr_phase(
            cast(Any, object()),
            "Q0",
            "Q1",
            initial_params=_params(),
            phase_offsets=(-0.2, -0.1, 0.1),
        )


def test_calibrate_ecr_phase_accepts_an_already_converged_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phase sweep with a passing baseline should not require a zero-crossing fit."""
    params = _params()
    characterize = Mock(
        return_value=_stage_result(
            params,
            ZX=np.pi / 2,
            ZY=0.0,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )
    )
    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_phase(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        phase_offsets=(-0.1, 0.0, 0.1),
    )

    assert result["status"] == "already_converged"
    assert result["proposed_params"] == params
    assert characterize.call_count == 3


def test_calibrate_ecr_phase_interpolates_within_a_local_sign_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonlinear sweep should use adjacent bracketing points for its phase update."""
    params = _params(cr_phase=0.0)

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        phase = params.cr_phase
        zy = -1.0 + 101.0 * (phase + 1.0) if phase <= 0.0 else 100.0 - 50.0 * phase
        return _stage_result(
            params,
            ZX=np.pi / 2,
            ZY=zy,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_phase(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        phase_offsets=(-1.0, 0.0, 1.0),
        tolerances=fine_cr.CrCalibrationTolerances(fine_error_angle=1e-6),
    )

    assert result["status"] == "success"
    assert result["phase_correction"] == pytest.approx(-100.0 / 101.0, abs=1e-12)


def test_calibrate_ecr_phase_prefers_the_root_nearest_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple sign brackets should select the smallest verified phase update."""
    params = _params(cr_phase=0.0)
    sampled_phases = np.asarray([-1.0, -0.1, 0.0, 0.1, 1.0])
    sampled_zy = np.asarray([-0.001, 0.001, 0.5, -0.5, -0.6])

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        zy = float(np.interp(params.cr_phase, sampled_phases, sampled_zy))
        return _stage_result(
            params,
            ZX=np.pi / 2,
            ZY=zy,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_phase(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        phase_offsets=sampled_phases,
        tolerances=fine_cr.CrCalibrationTolerances(fine_error_angle=1e-6),
    )

    assert result["status"] == "success"
    assert result["phase_correction"] == pytest.approx(0.05, abs=1e-12)


def test_calibrate_ecr_angle_verifies_the_interpolated_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Angle calibration should verify an in-range amplitude update."""
    params = _params(cr_amplitude=0.4)

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        zx = (params.cr_amplitude / 0.5) * (np.pi / 2)
        return _stage_result(params, ZX=zx, ZY=0.0, IX=0.0, IY=0.0, IZ=0.0, ZZ=0.0)

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_angle(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        amplitude_scales=(0.8, 1.0, 1.2, 1.4),
        tolerances=fine_cr.CrCalibrationTolerances(fine_angle_error=1e-6),
    )

    assert result["converged"] is True
    assert result["verified"] is True
    assert result["proposed_params"].cr_amplitude == pytest.approx(0.5, abs=1e-8)


def test_calibrate_ecr_angle_requires_the_input_scale() -> None:
    """Angle calibration should require unit scale as an observed baseline."""
    with pytest.raises(ValueError, match="must include 1"):
        fine_cr.calibrate_ecr_angle(
            cast(Any, object()),
            "Q0",
            "Q1",
            initial_params=_params(),
            amplitude_scales=(0.8, 0.9, 1.1),
        )


def test_calibrate_ecr_angle_accepts_an_already_converged_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An angle sweep with a passing baseline should not require a zero-crossing fit."""
    params = _params(zx_rotation_rate=cast(Any, None))
    characterize = Mock(
        return_value=_stage_result(
            params,
            ZX=np.pi / 2,
            ZY=0.0,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )
    )
    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_angle(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        amplitude_scales=(0.9, 1.0, 1.1),
    )

    assert result["status"] == "already_converged"
    assert result["proposed_params"].zx_rotation_rate is not None
    assert characterize.call_count == 3


def test_cancellation_calibration_requires_both_input_points() -> None:
    """Cancellation calibration should require zero in both IQ sweeps."""
    with pytest.raises(ValueError, match="cancel_x_offsets must include 0"):
        fine_cr.calibrate_un_echoed_cancellation(
            cast(Any, object()),
            "Q0",
            "Q1",
            initial_params=_params(),
            cancel_x_offsets=(-0.2, -0.1, 0.1),
            cancel_y_offsets=(-0.1, 0.0, 0.1),
        )


def test_cancellation_calibration_accepts_already_nulled_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation axes already within tolerance should not require fitted roots."""
    params = _params()
    monkeypatch.setattr(
        fine_cr,
        "_characterize_un_echoed_gate",
        lambda *_args, **_kwargs: _stage_result(
            params,
            ZX=np.pi / 4,
            ZY=0.0,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        ),
    )

    result = fine_cr.calibrate_un_echoed_cancellation(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        cancel_x_offsets=(-0.1, 0.0, 0.1),
        cancel_y_offsets=(-0.1, 0.0, 0.1),
    )

    assert result["status"] == "success"
    assert result["proposed_params"] == params
    assert result["raw_results"]["x"]["status"] == "already_converged"
    assert result["raw_results"]["y"]["status"] == "already_converged"


def test_optimize_rotary_keeps_input_when_measurement_does_not_improve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rotary optimization should preserve the input when no point improves it."""
    params = _params(rotary_x=0.0, rotary_y=0.0)

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        error = 0.011 + abs(params.rotary_x)
        return _stage_result(
            params, ZX=np.pi / 2, ZY=error, IX=0.0, IY=error, IZ=error, ZZ=error
        )

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.optimize_ecr_rotary(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        rotary_x_values=(0.1, 0.2),
    )

    assert result["converged"] is False
    assert result["proposed_params"] == params
    assert result["heat"]["status"] == "not_run"


def test_optimize_rotary_skips_sweep_when_baseline_is_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified baseline within tolerance should avoid redundant rotary sweeps."""
    params = _params(rotary_x=0.0, rotary_y=0.0)
    characterize = Mock(
        return_value=_stage_result(
            params,
            ZX=np.pi / 2,
            ZY=0.0,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )
    )
    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.optimize_ecr_rotary(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        rotary_x_values=(-0.1, 0.0, 0.1),
        rotary_y_values=(-0.1, 0.0, 0.1),
    )

    assert result["status"] == "already_converged"
    assert result["converged"] is True
    assert result["verified"] is True
    assert result["proposed_params"] == params
    characterize.assert_called_once()


def test_optimize_rotary_rejects_an_all_zero_objective() -> None:
    """Rotary optimization should require at least one positive objective weight."""
    with pytest.raises(ValueError, match="positive"):
        fine_cr.optimize_ecr_rotary(
            cast(Any, object()),
            "Q0",
            "Q1",
            initial_params=_params(),
            rotary_x_values=(-0.1, 0.0, 0.1),
            objective_weights={"IY": 0.0, "ZZ": 0.0},
        )


def test_optimize_rotary_reuses_the_measured_baseline_in_the_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grid containing the input rotary IQ should not measure that point twice."""
    params = _params(rotary_x=0.0, rotary_y=0.0)

    def result_for_candidate(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        error = 0.03 + abs(params.rotary_x)
        return _stage_result(
            params,
            ZX=np.pi / 2,
            ZY=0.0,
            IX=0.0,
            IY=error,
            IZ=error,
            ZZ=error,
        )

    characterize = Mock(side_effect=result_for_candidate)
    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    fine_cr.optimize_ecr_rotary(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
        rotary_x_values=(0.0, 0.1),
    )

    assert characterize.call_count == 2


def test_local_z_calibration_marks_control_phase_as_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-Z calibration should expose the unsupported control phase explicitly."""
    params = _params()

    def characterize(
        _exp: object,
        _control: str,
        _target: str,
        *,
        params: fine_cr.CrGateParameters,
        **_kwargs: object,
    ) -> Result:
        iz = 0.02 + params.target_frame_z
        return _stage_result(
            params, ZX=np.pi / 2, ZY=0.0, IX=0.0, IY=0.0, IZ=iz, ZZ=0.0
        )

    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_local_z(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
    )

    assert result["components"]["target"]["verified"] is True
    assert result["components"]["control"]["status"] == "not_run"
    assert result["status"] == "partial"


def test_local_z_calibration_skips_update_when_target_is_already_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passing target IZ baseline should not be measured a second time."""
    params = _params()
    characterize = Mock(
        return_value=_stage_result(
            params,
            ZX=np.pi / 2,
            ZY=0.0,
            IX=0.0,
            IY=0.0,
            IZ=0.0,
            ZZ=0.0,
        )
    )
    monkeypatch.setattr(fine_cr, "characterize_ecr_gate", characterize)

    result = fine_cr.calibrate_ecr_local_z(
        cast(Any, object()),
        "Q0",
        "Q1",
        initial_params=params,
    )

    assert result["status"] == "partial"
    assert result["proposed_params"] == params
    assert result["components"]["target"]["status"] == "already_converged"
    characterize.assert_called_once()


def test_measure_ecr_leakage_requires_three_state_classifier_data() -> None:
    """Leakage measurement should fail safely when only two states are classified."""
    exp = _pulse_experiment()
    measure_data = SimpleNamespace(
        probabilities=np.array([0.9, 0.1]),
        standard_deviations=np.array([0.01, 0.01]),
    )
    exp.measure = Mock(
        return_value=SimpleNamespace(data={"Q0": measure_data, "Q1": measure_data})
    )

    result = fine_cr.measure_ecr_leakage(
        exp,
        "Q0",
        "Q1",
        params=_params(),
        repetition_counts=(1,),
        initial_states=("00",),
    )

    assert result["converged"] is False
    assert result["supported"] is False
    assert result["status"] == "qutrit_classifier_required"
    measure_kwargs = exp.measure.call_args.kwargs
    assert measure_kwargs["enable_dsp_classification"] is True
    assert "state_classification" not in measure_kwargs


def test_measure_ecr_leakage_rejects_fractional_repetition_counts() -> None:
    """Leakage repetition counts should not silently truncate fractional values."""
    with pytest.raises(ValueError, match="integers"):
        fine_cr.measure_ecr_leakage(
            _pulse_experiment(),
            "Q0",
            "Q1",
            params=_params(),
            repetition_counts=cast(Any, (1.5, 2.5)),
        )


def test_measure_ecr_leakage_rejects_unphysical_probabilities() -> None:
    """Leakage analysis should reject classified populations outside [0, 1]."""
    exp = _pulse_experiment()
    measure_data = SimpleNamespace(
        probabilities=np.array([1.1, -0.1, 0.0]),
        standard_deviations=np.array([0.01, 0.01, 0.01]),
    )
    exp.measure = Mock(
        return_value=SimpleNamespace(data={"Q0": measure_data, "Q1": measure_data})
    )

    result = fine_cr.measure_ecr_leakage(
        exp,
        "Q0",
        "Q1",
        params=_params(),
        repetition_counts=(1,),
        initial_states=("00",),
    )

    assert result["status"] == "measurement_failed"
    assert result["verified"] is False


def test_validate_ecr_gate_is_non_mutating_and_produces_a_commit_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation should be non-mutating and bind success to the exact parameters."""
    params = _params()
    note = Mock()
    exp = _pulse_experiment()
    exp.ctx = SimpleNamespace(calib_note=note)

    result = _passing_validation(exp, params, monkeypatch)

    assert result["passed"] is True
    assert result["committable"] is True
    assert result["validation_token"]
    note.update_cr_param.assert_not_called()


def test_validate_ecr_gate_requires_explicit_control_local_z_waiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation should not hide that target tomography leaves control ZI unmeasured."""
    params = _params()
    exp = _pulse_experiment()
    metrics = {
        "ZX": np.pi / 2,
        "ZY": 0.0,
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZZ": 0.0,
    }
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: _stage_result(params, **metrics),
    )

    result = fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=False,
        run_irb=False,
    )

    assert result["passed"] is False
    assert result["criteria"]["control_local_z"]["status"] == "not_run"


def test_validate_ecr_gate_rejects_low_quality_repeated_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final validation should reject error angles from a low-R² fit."""
    params = _params()
    exp = _pulse_experiment()
    low_quality = _stage_result(
        params,
        ZX=np.pi / 2,
        ZY=0.0,
        IX=0.0,
        IY=0.0,
        IZ=0.0,
        ZZ=0.0,
    )
    low_quality.data["fit_quality"] = {"r2": 0.2}
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: low_quality,
    )

    result = fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=False,
        run_irb=False,
        waive_control_local_z=True,
    )

    assert result["passed"] is False
    assert result["criteria"]["repeated_tomography"]["passed"] is False


def test_validate_ecr_gate_rejects_negative_irb_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative unphysical IRB error estimate should fail validation."""
    params = _params()
    exp = _pulse_experiment()
    label = "Q0-Q1"
    exp.interleaved_randomized_benchmarking = Mock(
        return_value=Result(data={label: {"gate_error": -0.1}})
    )
    metrics = {
        "ZX": np.pi / 2,
        "ZY": 0.0,
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZZ": 0.0,
    }
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: _stage_result(params, **metrics),
    )

    result = fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=False,
        run_irb=True,
        waive_control_local_z=True,
    )

    assert result["passed"] is False
    assert result["criteria"]["irb"]["status"] == "invalid_result"


def test_validate_ecr_gate_rejects_out_of_range_bell_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Bell fidelity outside the physical interval should fail validation."""
    params = _params()
    exp = _pulse_experiment()
    exp.bell_state_tomography = Mock(return_value=Result(data={"fidelity": 1.1}))
    metrics = {
        "ZX": np.pi / 2,
        "ZY": 0.0,
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZZ": 0.0,
    }
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: _stage_result(params, **metrics),
    )

    result = fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=True,
        run_irb=False,
        waive_control_local_z=True,
    )

    assert result["passed"] is False
    assert result["criteria"]["bell_tomography"]["status"] == "invalid_result"


def test_validate_ecr_gate_rejects_missing_bell_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Bell fidelity should fail validation without comparison errors."""
    params = _params()
    exp = _pulse_experiment()
    exp.bell_state_tomography = Mock(return_value=Result(data={}))
    metrics = {
        "ZX": np.pi / 2,
        "ZY": 0.0,
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZZ": 0.0,
    }
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: _stage_result(params, **metrics),
    )

    result = fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=True,
        run_irb=False,
        waive_control_local_z=True,
    )

    assert result["passed"] is False
    assert result["criteria"]["bell_tomography"]["status"] == "invalid_result"
    assert result["criteria"]["bell_tomography"]["fidelity"] is None


def test_validate_ecr_gate_rejects_missing_irb_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing IRB error should fail validation without comparison errors."""
    params = _params()
    exp = _pulse_experiment()
    exp.interleaved_randomized_benchmarking = Mock(return_value=Result(data={}))
    metrics = {
        "ZX": np.pi / 2,
        "ZY": 0.0,
        "IX": 0.0,
        "IY": 0.0,
        "IZ": 0.0,
        "ZZ": 0.0,
    }
    monkeypatch.setattr(
        fine_cr,
        "characterize_ecr_gate",
        lambda *_args, **_kwargs: _stage_result(params, **metrics),
    )

    result = fine_cr.validate_ecr_gate(
        exp,
        "Q0",
        "Q1",
        params=params,
        run_leakage=False,
        run_bell_tomography=False,
        run_irb=True,
        waive_control_local_z=True,
    )

    assert result["passed"] is False
    assert result["criteria"]["irb"]["status"] == "invalid_result"
    assert result["criteria"]["irb"]["gate_error"] is None


def test_commit_is_the_only_stage_that_updates_the_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passing validation result should replace the in-memory entry exactly once."""
    params = _params()
    note = Mock()
    note.get_cr_param.return_value = None
    exp = _pulse_experiment()
    exp.ctx = SimpleNamespace(calib_note=note)
    validation = _passing_validation(exp, params, monkeypatch)

    result = fine_cr.commit_cr_calibration(exp, validation)

    assert result["committed"] is True
    note.remove_cr_param.assert_called_once_with("Q0-Q1")
    note.update_cr_param.assert_called_once()
    stored = note.update_cr_param.call_args.args[1]
    assert "cancel_x" not in stored
    assert "cancel_y" not in stored
    assert "rotary_x" not in stored


def test_commit_rejects_parameters_that_do_not_match_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit should reject a stale or forged parameter fingerprint."""
    params = _params()
    exp = _pulse_experiment()
    exp.ctx = SimpleNamespace(calib_note=Mock())
    valid = _passing_validation(exp, params, monkeypatch)
    validation = Result(
        data={**valid.data, "validated_params": _params(cr_amplitude=0.5)}
    )

    with pytest.raises(ValueError, match="does not match"):
        fine_cr.commit_cr_calibration(exp, validation)


def test_commit_rejects_a_label_that_does_not_match_validated_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit should not write validated parameters under a different CR label."""
    params = _params()
    exp = _pulse_experiment()
    exp.ctx = SimpleNamespace(calib_note=Mock())
    valid = _passing_validation(exp, params, monkeypatch)
    validation = Result(data={**valid.data, "cr_label": "Q9-Q8"})

    with pytest.raises(ValueError, match="does not match"):
        fine_cr.commit_cr_calibration(exp, validation)

    exp.ctx.calib_note.update_cr_param.assert_not_called()


def test_commit_replaces_an_existing_entry_created_in_the_same_second(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Commit should replace rather than timestamp-merge the prior CR entry."""
    params = _params()
    note = CalibrationNote("TEST", file_path=tmp_path / "calibration.json")
    previous = {
        "target": "Q0-Q1",
        "duration": 128.0,
        "ramptime": 24.0,
        "cr_amplitude": 0.25,
        "cr_phase": 0.0,
        "cr_beta": 0.0,
        "cancel_amplitude": 0.01,
        "cancel_phase": 0.0,
        "cancel_beta": 0.0,
        "rotary_amplitude": 0.0,
        "zx_rotation_rate": 0.003,
    }
    note.update_cr_param("Q0-Q1", cast(Any, previous))
    exp = _pulse_experiment()
    exp.ctx = SimpleNamespace(calib_note=note)
    validation = _passing_validation(exp, params, monkeypatch)

    fine_cr.commit_cr_calibration(exp, validation)

    stored = note.get_cr_param("Q0-Q1") or {}
    assert stored["cr_amplitude"] == pytest.approx(params.cr_amplitude, abs=1e-12)
    assert stored["rotary_y"] == pytest.approx(params.rotary_y, abs=1e-12)
