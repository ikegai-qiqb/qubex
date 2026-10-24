"""Public API tests for readout-MIST characterization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment.readout_mist as mist_module
from qubex import contrib
from qubex.contrib import experiment as experiment_contrib
from qubex.contrib.experiment import characterize_readout_mist
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, PulseSchedule


def test_readout_mist_is_exported_from_public_contrib_namespaces() -> None:
    """Readout MIST should be callable from both public contrib namespaces."""
    assert mist_module.__all__ == ["characterize_readout_mist"]
    assert contrib.characterize_readout_mist is characterize_readout_mist
    assert experiment_contrib.characterize_readout_mist is characterize_readout_mist


class _Context:
    experiment_system = SimpleNamespace(
        measurement_defaults={"execution": {"shot_interval_ns": 500_000.0}}
    )

    @staticmethod
    def resolve_qubit_label(label: str) -> str:
        return label

    @staticmethod
    def resolve_ge_label(label: str) -> str:
        return label

    @staticmethod
    def resolve_ef_label(label: str) -> str:
        return f"{label}:ef"

    @staticmethod
    def resolve_read_label(label: str) -> str:
        return f"R{label}"


def _experiment() -> Experiment:
    pulse = SimpleNamespace(
        readout=lambda _target: FlatTop(duration=80.0, amplitude=0.2, tau=8.0),
        x180=lambda label: FlatTop(duration=24.0, amplitude=0.1, tau=4.0),
    )
    return cast(Experiment, SimpleNamespace(ctx=_Context(), pulse=pulse))


@pytest.mark.parametrize(
    "counts",
    [
        (0, 1, 2, 4),
        (1, 2, 4, 8, 16),
        (0, 2, 4, 8, 16),
        (0, 1, 1, 4, 8),
        (0, 1, 2, 4, 3),
        (0, 1, 2, 4, 8.0),
    ],
)
def test_invalid_repetition_counts_fail_before_acquisition(counts: Any) -> None:
    """Invalid grids should fail before any hardware acquisition."""
    with pytest.raises((TypeError, ValueError)):
        characterize_readout_mist(
            _experiment(),
            "Q0",
            repetition_counts=counts,
            plot=False,
        )


def test_custom_cycle_cannot_also_request_an_implicit_gap() -> None:
    """A complete cycle override should not accept a second trailing gap."""
    with PulseSchedule() as cycle:
        cycle.add("RQ0", FlatTop(duration=80.0, amplitude=0.2, tau=8.0))

    with pytest.raises(ValueError, match="cycle_gap_ns"):
        characterize_readout_mist(
            _experiment(),
            "Q0",
            readout_cycle=cycle,
            cycle_gap_ns=20.0,
            plot=False,
        )


@pytest.mark.parametrize("value", [True, -1.0, 1.0, float("nan")])
def test_invalid_gef_covariance_cutoff_fails_before_acquisition(value: object) -> None:
    """The GEF covariance cutoff should be validated before hardware acquisition."""
    with pytest.raises((TypeError, ValueError), match="gef_covariance_rcond"):
        characterize_readout_mist(
            _experiment(),
            "Q0",
            gef_covariance_rcond=value,  # type: ignore[arg-type]
            plot=False,
        )


def test_public_defaults_keep_reference_and_disable_second_phase_analysis() -> None:
    """Initial API defaults should enable blank reference without phase-two fits."""
    defaults = characterize_readout_mist.__kwdefaults__

    assert defaults is not None
    assert defaults["measure_reference"] is True
    assert "fit_reduced_models" not in defaults
    assert "gef_bootstrap_n_resamples" not in defaults


def test_workflow_uses_override_for_actual_and_complete_blank_for_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workflow should separate candidate stimulation from diagnostic readout."""
    calls: list[dict[str, Any]] = []
    transition = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.02], [0.01, 0.01, 0.97]])
    initial = {
        "initial_g": np.array([0.98, 0.015, 0.005]),
        "initial_e": np.array([0.015, 0.975, 0.01]),
        "initial_f": np.array([0.01, 0.02, 0.97]),
    }
    covariance = 1e-7 * np.array(
        [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]
    )
    calibration = {"Q0": object()}

    def measure(*_args: Any, **kwargs: Any) -> Result:
        calls.append(kwargs)
        fits = {}
        for name in kwargs["sequences"]:
            protocol = next(value for value in initial if f"/{value}/" in name)
            count = int(name.rsplit("=", 1)[1])
            population = np.linalg.matrix_power(transition, count) @ initial[protocol]
            fits[name] = {
                "Q0": SimpleNamespace(
                    population=population,
                    population_covariance=covariance,
                    population_standard_error=np.sqrt(np.diag(covariance)),
                    population_unconstrained=population,
                    objective=0.0,
                    residual=np.zeros(15),
                    design_rank=2,
                    design_condition_number=2.0,
                    message="synthetic",
                )
            }
        return Result(
            data={
                "fits": fits,
                "calibration": calibration,
                "raw_iq": {},
                "moment_summaries": {},
            }
        )

    monkeypatch.setattr(mist_module, "measure_gef_populations", measure)
    with PulseSchedule() as cycle:
        cycle.add("RQ0", FlatTop(duration=80.0, amplitude=0.25, tau=8.0))
        cycle.add("RQ0", FlatTop(duration=20.0, amplitude=0.05, tau=4.0))

    result = characterize_readout_mist(
        _experiment(),
        target="Q0",
        repetition_counts=(0, 1, 2, 4, 8),
        readout_cycle=cycle,
        plot=False,
        enable_tqdm=False,
    )

    assert len(calls) == 2
    assert calls[0]["calibration_n_shots"] == 8192
    assert calls[0]["capture_index"] == -1
    assert calls[1]["calibration"] is calibration
    assert calls[1]["capture_index"] == -1
    actual_n1 = calls[0]["sequences"]["actual/initial_g/n=1"]
    reference_n1 = calls[1]["sequences"]["reference/initial_g/n=1"]
    assert np.any(actual_n1.values["RQ0"] != 0.0)
    np.testing.assert_allclose(reference_n1.values["RQ0"], 0.0, rtol=0.0, atol=0.0)
    assert actual_n1.duration == reference_n1.duration
    assert result.data["analysis"].actual.state_order == ("g", "e", "f")
    assert result.data["readout_cycle_timing"]["trailing_gap_in_final_cycle"] is True
    assert result.data["measurement_options"]["diagnostic_capture_index"] == -1
