"""Tests for explicit two-qubit RB gate candidates."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
from qxpulse import PulseSchedule

from qubex.clifford.clifford import Clifford
from qubex.experiment.models.result import Result
from qubex.experiment.services.benchmarking_service import BenchmarkingService


class _MeasureResult:
    """Return deterministic computational-state probabilities."""

    @staticmethod
    def get_probabilities(_targets: list[str]) -> dict[str, float]:
        return {"00": 1.0}

    @staticmethod
    def get_mitigated_probabilities(_targets: list[str]) -> dict[str, float]:
        return {"00": 1.0}


def _service(
    *,
    stored_targets: set[str],
    is_cr: bool = True,
) -> tuple[BenchmarkingService, list[PulseSchedule | None], list[object]]:
    """Build a lightweight service and record generated RB inputs."""
    generated_candidates: list[PulseSchedule | None] = []
    measurement_calls: list[object] = []
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        state_centers={"Q00": {0: 0j, 1: 1j}, "Q01": {0: 0j, 1: 1j}},
        experiment_system=SimpleNamespace(
            get_target=lambda _target: SimpleNamespace(is_cr=is_cr)
        ),
        calib_note=SimpleNamespace(cr_params={target: {} for target in stored_targets}),
        cr_pair=lambda _target: ("Q00", "Q01"),
        reset_awg_and_capunits=lambda **_kwargs: None,
    )
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _rb_sequence_2q(
        self: BenchmarkingService,
        *,
        zx90: PulseSchedule | None = None,
        **_kwargs: object,
    ) -> PulseSchedule:
        generated_candidates.append(zx90)
        return PulseSchedule(["Q00", "Q00-Q01", "Q01"])

    service.__dict__["rb_sequence_2q"] = MethodType(_rb_sequence_2q, service)

    def _measure(**kwargs: object) -> _MeasureResult:
        measurement_calls.append(kwargs["sequence"])
        return _MeasureResult()

    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_measure)
    return service, generated_candidates, measurement_calls


def _run_rb(
    service: BenchmarkingService,
    *,
    target: str,
    zx90: dict[str, PulseSchedule] | None = None,
) -> Result:
    """Run the smallest deterministic RB acquisition."""
    return service.rb_experiment_2q(
        targets=target,
        n_cliffords_range=np.array([0]),
        n_trials=1,
        seeds=np.array([7]),
        zx90=zx90,
        mitigate_readout=False,
        plot=False,
        save_image=False,
        reset_awg_and_capunits=False,
    )


def test_rb_2q_accepts_unstored_target_with_explicit_zx90(
    monkeypatch: Any,
) -> None:
    """An explicit candidate makes a valid CR target runnable before commit."""
    monkeypatch.setattr(
        "qubex.experiment.services.benchmarking_service.fitting.fit_rb",
        lambda **_kwargs: {"fig": None},
    )
    service, generated_candidates, measurement_calls = _service(stored_targets=set())
    candidate = PulseSchedule(["Q00", "Q00-Q01", "Q01"])

    result = _run_rb(
        service,
        target="Q00-Q01",
        zx90={"Q00-Q01": candidate},
    )

    assert "Q00-Q01" in result
    assert generated_candidates == [candidate]
    assert len(measurement_calls) == 1


def test_rb_2q_still_accepts_stored_target_without_explicit_zx90(
    monkeypatch: Any,
) -> None:
    """The established stored-parameter path remains backward compatible."""
    monkeypatch.setattr(
        "qubex.experiment.services.benchmarking_service.fitting.fit_rb",
        lambda **_kwargs: {"fig": None},
    )
    service, generated_candidates, measurement_calls = _service(
        stored_targets={"Q00-Q01"}
    )

    result = _run_rb(service, target="Q00-Q01")

    assert "Q00-Q01" in result
    assert generated_candidates == [None]
    assert len(measurement_calls) == 1


def test_rb_2q_keeps_filtering_unstored_target_without_candidate() -> None:
    """Legacy filtering remains unchanged when no gate definition exists."""
    service, generated_candidates, measurement_calls = _service(stored_targets=set())

    result = _run_rb(service, target="Q00-Q01")

    assert not result.data
    assert generated_candidates == []
    assert measurement_calls == []


def test_rb_2q_keeps_filtering_a_null_explicit_candidate() -> None:
    """A null map entry should not bypass the missing stored-gate filter."""
    service, generated_candidates, measurement_calls = _service(stored_targets=set())

    result = _run_rb(
        service,
        target="Q00-Q01",
        zx90=cast(Any, {"Q00-Q01": None}),
    )

    assert not result.data
    assert generated_candidates == []
    assert measurement_calls == []


def test_rb_2q_keeps_filtering_non_cr_target_with_explicit_candidate() -> None:
    """An explicit waveform must not turn a non-CR target into a 2Q target."""
    service, generated_candidates, measurement_calls = _service(
        stored_targets=set(),
        is_cr=False,
    )
    candidate = PulseSchedule(["Q00", "Q00-Q01", "Q01"])

    result = _run_rb(
        service,
        target="Q00-Q01",
        zx90={"Q00-Q01": candidate},
    )

    assert not result.data
    assert generated_candidates == []
    assert measurement_calls == []


def test_irb_2q_uses_explicit_zx90_before_commit(monkeypatch: Any) -> None:
    """Reference RB and the interleaved run both use an uncommitted candidate."""

    def _fit_rb(**_kwargs: object) -> dict[str, float | None]:
        return {
            "A": 0.1,
            "p": 0.99,
            "p_err": 0.01,
            "C": 0.9,
            "avg_gate_error": 0.01,
            "avg_gate_fidelity": 0.99,
            "avg_gate_fidelity_err": 0.001,
            "fig": None,
        }

    monkeypatch.setattr(
        "qubex.experiment.services.benchmarking_service.fitting.fit_rb",
        _fit_rb,
    )
    monkeypatch.setattr(
        "qubex.experiment.services.benchmarking_service.fitting.plot_irb",
        lambda **_kwargs: None,
    )
    service, _, _ = _service(stored_targets=set())
    service.__dict__["_clifford_generator"] = SimpleNamespace(
        cliffords={"ZX90": Clifford.ZX90()}
    )
    generated: list[tuple[PulseSchedule | None, PulseSchedule | None]] = []

    def _rb_sequence_2q(
        self: BenchmarkingService,
        *,
        zx90: PulseSchedule | None = None,
        interleaved_waveform: PulseSchedule | None = None,
        **_kwargs: object,
    ) -> PulseSchedule:
        generated.append((zx90, interleaved_waveform))
        return PulseSchedule(["Q00", "Q00-Q01", "Q01"])

    service.__dict__["rb_sequence_2q"] = MethodType(_rb_sequence_2q, service)
    candidate = PulseSchedule(["Q00", "Q00-Q01", "Q01"])

    result = service.irb_experiment(
        targets="Q00-Q01",
        interleaved_clifford="ZX90",
        interleaved_waveform={"Q00-Q01": candidate},
        zx90={"Q00-Q01": candidate},
        n_cliffords_range=np.array([0]),
        n_trials=1,
        seeds=np.array([7]),
        plot=False,
        save_image=False,
    )

    assert "Q00-Q01" in result
    assert generated == [(candidate, None), (candidate, candidate)]
