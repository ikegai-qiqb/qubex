"""Tests for cross-resonance pulse construction."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.pulse import Arbitrary, CrossResonance


def test_cr_detuning_applies_continuous_phase_ramp_only_to_cr_lobes() -> None:
    """CR detuning should phase-ramp CR lobes continuously without changing other tones."""
    detuning = 0.012
    kwargs = {
        "control_qubit": "Q00",
        "target_qubit": "Q01",
        "cr_amplitude": 0.4,
        "cr_duration": 16.0,
        "cr_ramptime": 4.0,
        "cancel_amplitude": 0.2,
        "echo": True,
        "pi_pulse": Arbitrary([1.0, 1.0]),
        "pi_margin": 2.0,
    }
    reference = CrossResonance(**kwargs)
    schedule = CrossResonance(
        **kwargs,
        cr_detuning=detuning,
    )

    reference_cr = reference.values["Q00-Q01"]
    detuned_cr = schedule.values["Q00-Q01"]
    driven = np.abs(reference_cr) > 1e-12
    times = np.arange(len(reference_cr)) * schedule.cr_waveform.sampling_period

    np.testing.assert_allclose(
        detuned_cr[driven] / reference_cr[driven],
        np.exp(-2j * np.pi * detuning * times[driven]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        schedule.values["Q01"],
        reference.values["Q01"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        schedule.values["Q00"],
        reference.values["Q00"],
        rtol=0.0,
        atol=0.0,
    )

    repeated_reference = reference.repeated(2)
    repeated_schedule = schedule.repeated(2)
    repeated_reference_cr = repeated_reference.values["Q00-Q01"]
    repeated_detuned_cr = repeated_schedule.values["Q00-Q01"]
    repeated_driven = np.abs(repeated_reference_cr) > 1e-12
    repeated_times = np.arange(len(repeated_reference_cr)) * (
        schedule.cr_waveform.sampling_period
    )
    np.testing.assert_allclose(
        repeated_detuned_cr[repeated_driven] / repeated_reference_cr[repeated_driven],
        np.exp(-2j * np.pi * detuning * repeated_times[repeated_driven]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        repeated_schedule.values["Q01"], repeated_reference.values["Q01"]
    )
    np.testing.assert_array_equal(
        repeated_schedule.values["Q00"], repeated_reference.values["Q00"]
    )
    assert repeated_schedule.get_sequence("Q00-Q01").final_frame_shift == pytest.approx(
        -2 * np.pi * detuning * schedule.duration * 2
    )


def test_zero_cr_detuning_preserves_existing_waveforms() -> None:
    """An explicit zero detuning should preserve legacy CrossResonance samples."""
    reference = CrossResonance(
        control_qubit="Q00",
        target_qubit="Q01",
        cr_amplitude=0.4,
        cr_duration=16.0,
        cr_ramptime=4.0,
        cancel_amplitude=0.2,
        echo=True,
        pi_pulse=Arbitrary([1.0, 1.0]),
    )
    schedule = CrossResonance(
        control_qubit="Q00",
        target_qubit="Q01",
        cr_amplitude=0.4,
        cr_duration=16.0,
        cr_ramptime=4.0,
        cancel_amplitude=0.2,
        cr_detuning=0.0,
        echo=True,
        pi_pulse=Arbitrary([1.0, 1.0]),
    )

    for label in reference.labels:
        np.testing.assert_array_equal(schedule.values[label], reference.values[label])


def test_un_echoed_cr_detuning_remains_continuous_when_repeated() -> None:
    """Repeated un-echoed CR primitives should preserve one continuous detuning ramp."""
    detuning = -0.009
    kwargs = {
        "control_qubit": "Q00",
        "target_qubit": "Q01",
        "cr_amplitude": 0.4,
        "cr_duration": 16.0,
        "cr_ramptime": 4.0,
        "cancel_amplitude": 0.2,
        "echo": False,
    }
    primitive = CrossResonance(**kwargs, cr_detuning=detuning)
    reference = CrossResonance(**kwargs).repeated(3)
    schedule = primitive.repeated(3)
    reference_cr = reference.values["Q00-Q01"]
    detuned_cr = schedule.values["Q00-Q01"]
    driven = np.abs(reference_cr) > 1e-12
    times = np.arange(len(reference_cr)) * primitive.cr_waveform.sampling_period

    np.testing.assert_allclose(
        detuned_cr[driven] / reference_cr[driven],
        np.exp(-2j * np.pi * detuning * times[driven]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        schedule.values["Q01"],
        reference.values["Q01"],
    )
