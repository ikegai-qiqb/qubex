"""Numerical helper tests for readout-MIST fitting."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.contrib.experiment.readout_mist import (
    ReadoutMistFitQuality,
    _local_covariance,
    assess_goodness_of_fit,
    fit_full_gls,
)


def test_converged_central_fit_survives_unresolved_local_covariance() -> None:
    """A rank-deficient Jacobian should preserve its converged central estimate."""
    result = fit_full_gls(
        lambda parameters: np.array([parameters[0] - 0.25]),
        (np.array([0.0, 0.0]),),
        parameter_count=2,
    )

    assert result.success
    assert result.parameters[0] == pytest.approx(0.25, rel=0.0, abs=1e-8)
    assert np.all(np.isnan(result.covariance))
    assert "not locally identifiable" in result.message


@pytest.mark.parametrize("parameter_count", [True, 2.5])
def test_parameter_count_must_be_an_integer(parameter_count: object) -> None:
    """Boolean and fractional parameter counts should fail before allocation."""
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        fit_full_gls(
            lambda parameters: parameters,
            (np.zeros(2),),
            parameter_count=parameter_count,  # type: ignore[arg-type]
        )


def test_nonfinite_residual_has_unassessed_goodness_of_fit() -> None:
    """Nonfinite whitened residuals should not receive a statistical classification."""
    goodness = assess_goodness_of_fit(
        np.array([0.0, np.nan, 1.0]),
        n_parameters=1,
    )

    assert goodness.quality is ReadoutMistFitQuality.NOT_ASSESSED
    assert goodness.reason == "No finite whitened residuals were available."
    assert np.isnan(goodness.chi_squared)


def test_nonfinite_jacobian_has_unavailable_local_covariance() -> None:
    """A nonfinite optimizer Jacobian should produce an unavailable covariance."""
    covariance = _local_covariance(
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        reduced_chi_squared=1.0,
    )

    assert covariance.shape == (2, 2)
    assert np.all(np.isnan(covariance))
