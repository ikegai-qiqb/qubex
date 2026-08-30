"""Tests for functional SRRE APIs."""

from __future__ import annotations

from qubex.contrib import (
    calculate_srre_moments,
    calibrate_srre,
    calibrate_srre_zx90,
    predict_srre_amplitude,
    srre_rzx,
    srre_waveform,
)
from qubex.contrib.experiment import (
    calculate_srre_moments as experiment_calculate_srre_moments,
    calibrate_srre as experiment_calibrate_srre,
    calibrate_srre_zx90 as experiment_calibrate_srre_zx90,
    predict_srre_amplitude as experiment_predict_srre_amplitude,
    srre_rzx as experiment_srre_rzx,
    srre_waveform as experiment_srre_waveform,
)


def test_all_srre_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all SRRE helpers are available."""
    assert callable(srre_waveform)
    assert callable(calculate_srre_moments)
    assert callable(predict_srre_amplitude)
    assert callable(calibrate_srre)
    assert callable(srre_rzx)
    assert callable(calibrate_srre_zx90)


def test_all_srre_functions_are_exported_from_experiment() -> None:
    """Given experiment package, when imported, then it exposes the same helpers."""
    assert experiment_srre_waveform is srre_waveform
    assert experiment_calculate_srre_moments is calculate_srre_moments
    assert experiment_predict_srre_amplitude is predict_srre_amplitude
    assert experiment_calibrate_srre is calibrate_srre
    assert experiment_srre_rzx is srre_rzx
    assert experiment_calibrate_srre_zx90 is calibrate_srre_zx90
