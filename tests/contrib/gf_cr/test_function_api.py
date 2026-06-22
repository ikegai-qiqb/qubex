"""Tests for functional APIs in `qubex.contrib.experiment.gf_cr`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment.gf_cr as gf_cr_module
from qubex.contrib import (
    calibrate_gf_zx90,
    gf_bell_state_tomography,
    gf_cr_hamiltonian_tomography,
    gf_zx90_interleaved_randomized_benchmarking,
    measure_gf_bell_state,
    measure_gf_cr_dynamics,
    obtain_gf_cr_params,
    update_gf_cr_params,
)


class _FigureStub:
    def add_trace(self, _trace: object) -> None:
        """Accept added traces."""

    def update_layout(self, **_kwargs: object) -> None:
        """Accept layout updates."""

    def show(self, **_kwargs: object) -> None:
        """Accept show calls."""


class _ScheduleStub:
    duration = 128.0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls: list[tuple[str, object]] = []

    def __enter__(self) -> _ScheduleStub:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def add(self, target: str, pulse: object) -> None:
        self.calls.append(("add", (target, pulse)))

    def barrier(self) -> None:
        self.calls.append(("barrier", None))

    def call(self, schedule: object) -> None:
        self.calls.append(("call", schedule))

    def repeated(self, _n_repetitions: int) -> _ScheduleStub:
        return self

    def plot(self, **_kwargs: object) -> None:
        """Accept plot calls."""


def test_all_gf_cr_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then GF-CR helpers are available."""
    assert callable(measure_gf_cr_dynamics)
    assert callable(gf_cr_hamiltonian_tomography)
    assert callable(update_gf_cr_params)
    assert callable(obtain_gf_cr_params)
    assert callable(calibrate_gf_zx90)
    assert callable(measure_gf_bell_state)
    assert callable(gf_bell_state_tomography)
    assert callable(gf_zx90_interleaved_randomized_benchmarking)


def test_obtain_gf_cr_params_returns_fig_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given iterative GF-CR updates, when obtaining params, then figure history is returned."""
    monkeypatch.setattr(gf_cr_module.viz, "make_figure", lambda: _FigureStub())

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=2.0),
            qubits={
                "Q00": SimpleNamespace(frequency=5.0),
                "Q01": SimpleNamespace(frequency=5.2),
            },
            calib_note=SimpleNamespace(get_cr_param=lambda _label: None),
        ),
        pulse=SimpleNamespace(
            calc_control_amplitude=lambda _target, _rabi_rate: 0.25,
        ),
    )

    update_results = [
        {
            "zx90_duration": 16.0,
            "cr_param": {
                "cr_phase": 0.1,
                "cancel_amplitude": 0.2,
                "cancel_phase": 0.3,
            },
            "coeffs": {"IX": 1.0e-4, "IY": 2.0e-4},
            "fig_c": "fig-c-1",
            "fig_t": "fig-t-1",
            "fig_t_3d": "fig-t-3d-1",
        },
        {
            "zx90_duration": 16.0,
            "cr_param": {
                "cr_phase": 0.4,
                "cancel_amplitude": 0.5,
                "cancel_phase": 0.6,
            },
            "coeffs": {"IX": 0.5e-4, "IY": 1.0e-4},
            "fig_c": "fig-c-2",
            "fig_t": "fig-t-2",
            "fig_t_3d": "fig-t-3d-2",
        },
    ]
    monkeypatch.setattr(
        gf_cr_module,
        "update_gf_cr_params",
        lambda *_args, **_kwargs: update_results.pop(0),
    )

    result = obtain_gf_cr_params(
        cast(Any, exp),
        "Q00",
        "Q01",
        n_iterations=2,
        n_cycles=1,
        n_points_per_cycle=4,
        ramptime=16.0,
        plot=False,
    )

    assert result["figs_history"] == [
        {"fig_c": "fig-c-1", "fig_t": "fig-t-1", "fig_t_3d": "fig-t-3d-1"},
        {"fig_c": "fig-c-2", "fig_t": "fig-t-2", "fig_t_3d": "fig-t-3d-2"},
    ]


def test_calibrate_gf_zx90_stores_drag_beta_and_uses_echo_wrapped_cr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given stored GF-CR params, when calibrating with DRAG, then beta and echo are used."""
    stored_updates: list[tuple[str, dict[str, float]]] = []
    wrapped_calls: list[dict[str, Any]] = []

    cr_param = {
        "cr_amplitude": 0.25,
        "cr_phase": 0.125,
        "cr_beta": 0.012,
        "cancel_amplitude": 0.10,
        "cancel_phase": 0.25,
        "cancel_beta": 0.034,
        "rotary_amplitude": 0.0,
        "zx_rotation_rate": 0.01,
        "duration": 64.0,
        "ramptime": 16.0,
    }

    def _calc_control_amplitude(*_args: object, **kwargs: object) -> float:
        if kwargs.get("target") == "Q01":
            return 0.09
        return 1.0

    def _sweep_parameter(
        sequence_factory: Any,
        *,
        sweep_range: np.ndarray,
        **_kwargs: object,
    ) -> SimpleNamespace:
        sequence_factory(float(sweep_range[0]))
        return SimpleNamespace(
            data={
                "Q01": SimpleNamespace(
                    normalized=np.array([1.0, 0.0, -1.0]),
                    zvalues=np.array([1.0, 0.0, -1.0]),
                )
            }
        )

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            calib_note=SimpleNamespace(
                get_cr_param=lambda _label: cr_param,
                update_cr_param=lambda label, params: stored_updates.append(
                    (label, params)
                ),
            ),
            qubits={
                "Q00": SimpleNamespace(frequency=5.0),
                "Q01": SimpleNamespace(frequency=5.2),
            },
            system_manager=SimpleNamespace(
                config_loader=SimpleNamespace(
                    load_param_data=lambda _name: (_ for _ in ()).throw(KeyError)
                )
            ),
        ),
        pulse=SimpleNamespace(
            calc_control_amplitude=_calc_control_amplitude,
            x180=lambda _target: "x180",
            get_pulse_for_state=lambda _target, _state: "state-pulse",
        ),
        measurement_service=SimpleNamespace(sweep_parameter=_sweep_parameter),
    )

    def _wrapped_cr_sequence(*_args: object, **kwargs: Any) -> _ScheduleStub:
        wrapped_calls.append(kwargs)
        return _ScheduleStub()

    fit_roots = iter([0.3, 0.35, 0.5])
    monkeypatch.setattr(gf_cr_module, "PulseSchedule", _ScheduleStub)
    monkeypatch.setattr(gf_cr_module, "_gf_wrapped_cr_sequence", _wrapped_cr_sequence)
    monkeypatch.setattr(
        gf_cr_module, "_gf_zx90_sequence", lambda *_a, **_k: _ScheduleStub()
    )
    monkeypatch.setattr(
        gf_cr_module.fitting,
        "fit_polynomial",
        lambda **_kwargs: {"root": next(fit_roots)},
    )

    result = calibrate_gf_zx90(
        cast(Any, exp),
        "Q00",
        "Q01",
        amplitude_range=np.array([0.2, 0.3, 0.4]),
        ramptime=16.0,
        use_drag=True,
        x180=cast(Any, {"Q00": "x180"}),
        plot=False,
    )

    assert wrapped_calls
    assert all(call["echo"] is True for call in wrapped_calls)
    assert all(call["cr_beta"] == cr_param["cr_beta"] for call in wrapped_calls)
    assert all(call["cancel_beta"] == cr_param["cancel_beta"] for call in wrapped_calls)

    stored_label, stored_param = stored_updates[0]
    assert stored_label == "Q00-gf-Q01"
    assert stored_param["cr_beta"] == pytest.approx(-1 / (2 * np.pi * (5.0 - 5.2)))
    assert stored_param["cancel_beta"] == 0.0
    assert stored_param["cancel_amplitude"] == pytest.approx(0.2)
    assert stored_param["rotary_amplitude"] == pytest.approx(0.18)
    assert result["stored_cr_label"] == "Q00-gf-Q01"


def test_gf_zx90_irb_fits_once_per_curve_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given mocked RB data, when GF-ZX90 IRB runs, then RB/IRB are fit once and printed."""
    reset_calls: list[set[str]] = []
    curve_calls: list[dict[str, Any]] = []
    fit_calls: list[dict[str, Any]] = []

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            state_centers={"Q00": object(), "Q01": object()},
            cr_pair=lambda _label: ("Q00", "Q01"),
            reset_awg_and_capunits=lambda qubits: reset_calls.append(set(qubits)),
        ),
    )

    def _measure_curve(*_args: object, **kwargs: Any) -> dict[str, np.ndarray]:
        curve_calls.append(kwargs)
        return {
            "n_cliffords": np.array([0, 1, 2]),
            "mean": np.array([1.0, 0.8, 0.7]),
            "std": np.array([0.01, 0.02, 0.03]),
            "trials": np.array([[1.0, 1.0], [0.8, 0.8], [0.7, 0.7]]),
            "seeds": np.array([11, 22]),
        }

    def _fit_rb(**kwargs: Any) -> dict[str, float]:
        fit_calls.append(kwargs)
        p = 0.9 if len(fit_calls) == 1 else 0.8
        return {
            "A": 0.5,
            "p": p,
            "p_err": 0.01,
            "C": 0.25,
            "avg_gate_error": 0.075,
            "avg_gate_fidelity": 0.925,
            "avg_gate_fidelity_err": 0.01,
        }

    monkeypatch.setattr(
        gf_cr_module,
        "_resolve_gf_zx90",
        lambda *_args, **_kwargs: _ScheduleStub(),
    )
    monkeypatch.setattr(gf_cr_module, "_measure_gf_zx90_rb_curve", _measure_curve)
    monkeypatch.setattr(gf_cr_module.fitting, "fit_rb", _fit_rb)
    monkeypatch.setattr(
        gf_cr_module.fitting, "plot_irb", lambda **_kwargs: _FigureStub()
    )
    monkeypatch.setattr(gf_cr_module.viz, "save_figure", lambda *_args, **_kwargs: None)

    result = gf_zx90_interleaved_randomized_benchmarking(
        cast(Any, exp),
        "Q00-gf-Q01",
        n_trials=2,
        seeds=np.array([11, 22]),
        plot=False,
        save_image=False,
    )

    assert len(curve_calls) == 2
    assert curve_calls[0]["interleaved"] is False
    assert curve_calls[1]["interleaved"] is True
    assert curve_calls[0]["reset_awg_and_capunits"] is False
    assert curve_calls[1]["reset_awg_and_capunits"] is False
    assert len(fit_calls) == 2
    assert reset_calls == [{"Q00", "Q01"}]
    assert result["Q00-gf-Q01"]["gate_fidelity"] == pytest.approx(
        1 - 0.08333333333333331
    )

    captured = capsys.readouterr()
    assert "Average gate fidelity (RB)" in captured.out
    assert "Average gate fidelity (IRB)" in captured.out
    assert "Gate fidelity" in captured.out
