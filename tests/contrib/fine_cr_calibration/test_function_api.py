"""Tests for public fine CR calibration function APIs."""

from __future__ import annotations

from qubex.contrib import (
    CrCalibrationOptions,
    CrCalibrationTolerances,
    CrGateParameters,
    build_ecr_gate,
    calibrate_bare_cr,
    calibrate_ecr_angle,
    calibrate_ecr_local_z,
    calibrate_ecr_phase,
    calibrate_un_echoed_cancellation,
    characterize_bare_cr,
    characterize_ecr_gate,
    check_cr_prerequisites,
    commit_cr_calibration,
    measure_ecr_leakage,
    optimize_ecr_rotary,
    scout_cr_operating_points,
    validate_ecr_gate,
)


def test_fine_cr_calibration_api_is_exported_from_contrib() -> None:
    """Fine CR data models and stages should be available from contrib."""
    assert CrGateParameters is not None
    assert CrCalibrationOptions is not None
    assert CrCalibrationTolerances is not None
    for stage in (
        check_cr_prerequisites,
        scout_cr_operating_points,
        characterize_bare_cr,
        calibrate_bare_cr,
        build_ecr_gate,
        characterize_ecr_gate,
        calibrate_ecr_phase,
        calibrate_ecr_angle,
        calibrate_un_echoed_cancellation,
        optimize_ecr_rotary,
        calibrate_ecr_local_z,
        measure_ecr_leakage,
        validate_ecr_gate,
        commit_cr_calibration,
    ):
        assert callable(stage)
