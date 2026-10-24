"""Synthetic tests for readout-MIST transition-matrix inference."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.contrib.experiment.readout_mist import (
    TRANSITION_ORDER,
    GefPopulationFitDiagnostic,
    GefPopulationSeries,
    ReadoutMistConditionMeasurements,
    ReadoutMistMeasurements,
    analyze_readout_mist,
)


def _synthetic_measurements() -> tuple[ReadoutMistMeasurements, np.ndarray]:
    """Build noiseless three-state trajectories with finite rank-two covariance."""
    transition = np.array(
        [
            [0.984, 0.012, 0.007],
            [0.010, 0.978, 0.023],
            [0.006, 0.010, 0.970],
        ],
        dtype=np.float64,
    )
    initial: dict[str, np.ndarray] = {
        "initial_g": np.array([0.985, 0.012, 0.003], dtype=np.float64),
        "initial_e": np.array([0.018, 0.971, 0.011], dtype=np.float64),
        "initial_f": np.array([0.012, 0.026, 0.962], dtype=np.float64),
    }
    counts = np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.int64)
    tangent_covariance = 2e-8 * np.array(
        [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]
    )
    protocols = {}
    for name, population_0 in initial.items():
        values = np.asarray(
            [np.linalg.matrix_power(transition, int(n)) @ population_0 for n in counts],
            dtype=np.float64,
        )
        covariance = np.repeat(tangent_covariance[None, :, :], counts.size, axis=0)
        protocols[name] = ReadoutMistConditionMeasurements(
            actual=GefPopulationSeries(
                population=values,
                covariance=covariance,
                standard_error=np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
                population_unconstrained=values.copy(),
            ),
            reference=None,
        )
    return ReadoutMistMeasurements(counts, protocols), transition


def test_joint_fit_recovers_transition_matrix_and_initial_populations() -> None:
    """The full joint GLS fit should recover a known stochastic matrix."""
    measurements, expected = _synthetic_measurements()

    analysis = analyze_readout_mist(measurements, covariance_rcond=1e-12)

    np.testing.assert_allclose(
        analysis.actual.transition_matrix,
        expected,
        rtol=2e-4,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        analysis.actual.transition_matrix.sum(axis=0),
        np.ones(3),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(analysis.actual.transition_matrix >= 0.0)
    assert analysis.actual.fit_quality.value == "accepted"


def test_transition_names_follow_final_by_initial_matrix_convention() -> None:
    """Named probabilities should follow `T[final, initial]`."""
    measurements, expected = _synthetic_measurements()

    fit = analyze_readout_mist(measurements, covariance_rcond=1e-12).actual

    np.testing.assert_allclose(
        [
            fit.transition_probabilities["g_to_e"].value,
            fit.transition_probabilities["e_to_g"].value,
            fit.transition_probabilities["g_to_f"].value,
            fit.transition_probabilities["e_to_f"].value,
            fit.transition_probabilities["f_to_g"].value,
            fit.transition_probabilities["f_to_e"].value,
        ],
        [
            expected[1, 0],
            expected[0, 1],
            expected[2, 0],
            expected[2, 1],
            expected[0, 2],
            expected[1, 2],
        ],
        rtol=1e-12,
        atol=1e-14,
    )
    assert fit.state_order == ("g", "e", "f")
    assert fit.probability_unit == "1/readout_cycle"
    assert tuple(fit.transition_probabilities) == TRANSITION_ORDER
    assert TRANSITION_ORDER == (
        "g_to_e",
        "e_to_g",
        "g_to_f",
        "e_to_f",
        "f_to_g",
        "f_to_e",
    )


def test_population_series_rejects_inconsistent_shapes() -> None:
    """Population records should reject covariance arrays on another grid."""
    with pytest.raises(ValueError, match="covariance"):
        GefPopulationSeries(
            population=np.full((2, 3), 1.0 / 3.0),
            covariance=np.zeros((1, 3, 3)),
            standard_error=np.zeros((2, 3)),
            population_unconstrained=np.full((2, 3), 1.0 / 3.0),
        )


def test_gef_diagnostic_allows_infinite_condition_number() -> None:
    """A singular GEF design should remain representable for warning generation."""
    diagnostic = GefPopulationFitDiagnostic(
        objective=0.0,
        design_rank=1,
        design_condition_number=float("inf"),
        residual=np.zeros(3),
        message="singular design",
    )

    assert np.isinf(diagnostic.design_condition_number)


@pytest.mark.parametrize("value", [True, -1.0, 1.0, float("nan")])
def test_analysis_rejects_invalid_covariance_cutoff(value: object) -> None:
    """The analysis cutoff should be a finite real value in the half-open unit interval."""
    measurements, _ = _synthetic_measurements()

    with pytest.raises((TypeError, ValueError), match="covariance_rcond"):
        analyze_readout_mist(
            measurements,
            covariance_rcond=value,  # type: ignore[arg-type]
        )
