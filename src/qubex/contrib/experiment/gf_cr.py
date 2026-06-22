"""Contributed GF-CR calibration helper functions."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import FitResult, fitting, util
from qubex.analysis.state_tomography import (
    mle_fit_density_matrix,
    plot_ghz_state_tomography,
)
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_CR_RAMPTIME,
    DEFAULT_CR_TIME_RANGE,
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_2Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.calibration_note import CrossResonanceParam
from qubex.experiment.models.result import Result
from qubex.pulse import (
    Blank,
    CrossResonance,
    PulseArray,
    PulseSchedule,
    VirtualZ,
    Waveform,
)
from qubex.typing import TargetMap


def _gf_cr_label(control_qubit: str, target_qubit: str) -> str:
    return f"{control_qubit}-gf-{target_qubit}"


def _ramptime(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
) -> float:
    f_ge_control = exp.ctx.qubits[control_qubit].frequency
    f_ef_target = exp.ctx.qubits[target_qubit].control_frequency_ef

    if f_ge_control < f_ef_target:
        return DEFAULT_CR_RAMPTIME
    return DEFAULT_CR_RAMPTIME * 2


def _adiabatic_safe_factor() -> float:
    return 0.75


def _gf_wrapped_cr_sequence(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    duration: float,
    ramptime: float,
    cr_amplitude: float,
    cr_phase: float,
    cr_beta: float,
    cancel_amplitude: float,
    cancel_phase: float,
    cancel_beta: float,
    echo: bool,
    x180: TargetMap[Waveform],
    x180_margin: float,
    ramp_type: Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"],
    ef_x180: Waveform | None,
) -> PulseSchedule:
    control_ef = exp.ctx.resolve_ef_label(control_qubit)
    ef_pi = ef_x180 if ef_x180 is not None else exp.pulse.x180(control_ef)
    cr_label = f"{control_qubit}-{target_qubit}"

    cr = CrossResonance(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        cr_amplitude=cr_amplitude,
        cr_duration=duration,
        cr_ramptime=ramptime,
        cr_phase=cr_phase,
        cr_beta=cr_beta,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cancel_beta=cancel_beta,
        echo=False,
        ramp_type=ramp_type,
    )

    with PulseSchedule([control_qubit, control_ef, cr_label, target_qubit]) as ps:
        ps.add(control_ef, ef_pi)
        ps.barrier()
        ps.call(cr)
        ps.barrier()
        ps.add(control_ef, ef_pi)
        if echo:
            pi_pulse = x180[control_qubit]
            if x180_margin > 0:
                margin = Blank(duration=x180_margin)
                pi_pulse = PulseArray([margin, pi_pulse, margin])
            ps.barrier()
            ps.add(control_qubit, pi_pulse)
            ps.barrier()
            ps.add(control_ef, ef_pi)
            ps.barrier()
            ps.call(cr.scaled(-1))
            ps.barrier()
            ps.add(control_ef, ef_pi)
            ps.barrier()
            ps.add(control_qubit, pi_pulse)
    return ps


def _gf_zx90_sequence(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    cr_param: CrossResonanceParam,
    cr_amplitude: float | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    rotary_amplitude: float | None = None,
    x180: Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float = 0.0,
) -> PulseSchedule:
    if cr_amplitude is None:
        cr_amplitude = cr_param["cr_amplitude"]
    if duration is None:
        duration = cr_param["duration"]
    if ramptime is None:
        ramptime = cr_param["ramptime"]
    if cancel_amplitude is None:
        cancel_amplitude = cr_param["cancel_amplitude"]
    if cancel_phase is None:
        cancel_phase = cr_param["cancel_phase"]
    if rotary_amplitude is None:
        rotary_amplitude = cr_param["rotary_amplitude"]
    if x180 is None:
        x180 = exp.pulse.x180(control_qubit)

    cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude
    return _gf_wrapped_cr_sequence(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        duration=duration,
        ramptime=ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_param["cr_phase"],
        cr_beta=cr_param["cr_beta"],
        cancel_amplitude=np.abs(cancel_pulse),
        cancel_phase=np.angle(cancel_pulse),
        cancel_beta=cr_param["cancel_beta"],
        echo=True,
        x180={control_qubit: x180},
        x180_margin=x180_margin,
        ramp_type="RaisedCosine",
        ef_x180=ef_x180,
    )


def measure_gf_cr_dynamics(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    echo: bool | None = None,
    control_state: str | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180: TargetMap[Waveform] | None = None,
    ef_x180: Waveform | None = None,
    ramp_type: Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"] | None = None,
    x180_margin: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
) -> Result:
    """Measure CR dynamics with the control qubit's e-state parked in f."""
    if echo is None:
        echo = False
    if control_state is None:
        control_state = "0"
    if ramp_type is None:
        ramp_type = "RaisedCosine"
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True

    cr_label = _gf_cr_label(control_qubit, target_qubit)
    if time_range is None:
        time_range = np.array(DEFAULT_CR_TIME_RANGE, dtype=float)
    else:
        time_range = np.array(time_range, dtype=float)
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if cr_phase is None:
        cr_phase = 0.0
    if cancel_amplitude is None:
        cancel_amplitude = 0.0
    if cancel_phase is None:
        cancel_phase = 0.0
    if x180_margin is None:
        x180_margin = 0.0
    if x90 is None:
        x90 = {
            control_qubit: exp.pulse.x90(control_qubit),
            target_qubit: exp.pulse.x90(target_qubit),
        }
    if x180 is None:
        x180 = {
            control_qubit: exp.pulse.x180(control_qubit),
        }

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=[control_qubit, target_qubit])

    control_states = []
    target_states = []
    for duration in time_range:
        sequence = _gf_wrapped_cr_sequence(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            duration=duration + ramptime * 2,
            ramptime=ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cr_beta=0.0,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cancel_beta=0.0,
            echo=echo,
            x180=x180,
            x180_margin=x180_margin,
            ramp_type=ramp_type,
            ef_x180=ef_x180,
        )
        result = exp.measurement_service.state_tomography(
            sequence,
            x90=x90,
            initial_state={control_qubit: control_state},
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=False,
            plot=False,
        )
        control_states.append(np.array(result[control_qubit]))
        target_states.append(np.array(result[target_qubit]))

    control_states = np.array(control_states)
    target_states = np.array(target_states)
    effective_drive_range = time_range + ramptime

    fit_result = fitting.fit_rotation(
        effective_drive_range,
        target_states,
        plot=False,
        title=f"Target qubit dynamics of {cr_label} : |{control_state}>",
        xlabel="Drive time (ns)",
        ylabel=f"Target qubit : {target_qubit}",
    )

    if plot:
        viz.plot_bloch_vectors(
            effective_drive_range,
            control_states,
            title=f"Control qubit dynamics of {cr_label} : |{control_state}>",
            xlabel="Drive time (ns)",
            ylabel=f"Control qubit : {control_qubit}",
        )
        fit_result.get_figure().show()
        fit_result.get_figure("fig3d").show()

    return Result(
        data={
            "time_range": time_range,
            "effective_drive_range": effective_drive_range,
            "control_states": control_states,
            "target_states": target_states,
            "fit_result": fit_result,
            "cr_amplitude": cr_amplitude,
            "ramptime": ramptime,
        }
    )


def gf_cr_hamiltonian_tomography(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180: TargetMap[Waveform] | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
) -> Result:
    """Run Hamiltonian tomography for a GF-CR pulse."""
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if ramptime is None:
        ramptime = _ramptime(exp, control_qubit, target_qubit)

    cr_label = _gf_cr_label(control_qubit, target_qubit)

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=[control_qubit, target_qubit])

    result_0 = measure_gf_cr_dynamics(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        time_range=time_range,
        ramptime=ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        echo=False,
        control_state="0",
        x90=x90,
        x180=x180,
        ef_x180=ef_x180,
        ramp_type="RaisedCosine",
        x180_margin=x180_margin,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=False,
        plot=False,
    )
    result_1 = measure_gf_cr_dynamics(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        time_range=time_range,
        ramptime=ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        echo=False,
        control_state="1",
        x90=x90,
        x180=x180,
        ef_x180=ef_x180,
        ramp_type="RaisedCosine",
        x180_margin=x180_margin,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=False,
        plot=False,
    )

    omega_0 = result_0["fit_result"]["Omega"]
    omega_1 = result_1["fit_result"]["Omega"]
    omega = np.concatenate(
        [
            0.5 * (omega_0 + omega_1),
            0.5 * (omega_0 - omega_1),
        ]
    )
    coeffs = dict(
        zip(
            ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"],
            omega / (2 * np.pi),
            strict=True,
        )
    )

    f_control = exp.ctx.qubits[control_qubit].frequency
    f_target = exp.ctx.qubits[target_qubit].frequency
    f_delta = f_control - f_target

    xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
    xt_rotation_amplitude = np.abs(xt_rotation)
    xt_rotation_amplitude_hw = exp.pulse.calc_control_amplitude(
        target=target_qubit,
        rabi_rate=xt_rotation_amplitude,
    )
    xt_rotation_phase = np.angle(xt_rotation)
    xt_rotation_phase_deg = np.angle(xt_rotation, deg=True)

    cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
    cr_rotation_amplitude = np.abs(cr_rotation)
    cr_rotation_amplitude_hw = exp.pulse.calc_control_amplitude(
        target=target_qubit,
        rabi_rate=cr_rotation_amplitude,
    )
    cr_rotation_phase = np.angle(cr_rotation)
    cr_rotation_phase_deg = np.angle(cr_rotation, deg=True)
    zx90_duration = 1 / (4 * cr_rotation_amplitude)

    cr_rabi_rate = exp.pulse.calc_rabi_rate(control_qubit, cr_amplitude)

    fig_c = _make_control_dynamics_figure(
        result_0=result_0,
        result_1=result_1,
        cr_label=cr_label,
        control_qubit=control_qubit,
        f_delta=f_delta,
        cr_rabi_rate=cr_rabi_rate,
        ramptime=ramptime,
    )
    fig_t, fig_t_3d = _make_target_dynamics_figures(
        result_0=result_0,
        result_1=result_1,
        cr_label=cr_label,
        target_qubit=target_qubit,
        f_delta=f_delta,
        cr_rabi_rate=cr_rabi_rate,
        ramptime=ramptime,
    )

    if plot:
        fig_c.show()
        fig_t.show()
        fig_t_3d.show()

        print("Qubit frequencies:")
        print(f"  omega_c ({control_qubit}) : {f_control * 1e3:.3f} MHz")
        print(f"  omega_t ({target_qubit}) : {f_target * 1e3:.3f} MHz")
        print(f"  Delta ({cr_label}) : {f_delta * 1e3:.3f} MHz")

        print("GF-CR drive:")
        print(f"  Omega : {cr_rabi_rate * 1e3:.3f} MHz ({cr_amplitude:.4f})")

        print("Rotation rates:")
        for key, value in coeffs.items():
            print(f"  {key} : {value * 1e3:+.4f} MHz")

        print("XT (crosstalk) rotation:")
        print(
            f"  rate  : {xt_rotation_amplitude * 1e3:.4f} MHz "
            f"({xt_rotation_amplitude_hw:.6f})"
        )
        print(
            f"  phase : {xt_rotation_phase:.4f} rad ({xt_rotation_phase_deg:.1f} deg)"
        )

        print("GF-CR (cross-resonance) rotation:")
        print(
            f"  rate  : {cr_rotation_amplitude * 1e3:.4f} MHz "
            f"({cr_rotation_amplitude_hw:.6f})"
        )
        print(
            f"  phase : {cr_rotation_phase:.4f} rad ({cr_rotation_phase_deg:.1f} deg)"
        )
        print(f"Estimated ZX90 gate length : {zx90_duration:.1f} ns")

    return Result(
        data={
            "Omega": omega,
            "coeffs": coeffs,
            "cr_rotation_amplitude": cr_rotation_amplitude,
            "cr_rotation_amplitude_hw": cr_rotation_amplitude_hw,
            "cr_rotation_phase": cr_rotation_phase,
            "xt_rotation_amplitude": xt_rotation_amplitude,
            "xt_rotation_amplitude_hw": xt_rotation_amplitude_hw,
            "xt_rotation_phase": xt_rotation_phase,
            "cr_drive_amplitude": cr_rabi_rate,
            "cr_drive_amplitude_hw": cr_amplitude,
            "zx90_duration": zx90_duration,
            "result_0": result_0,
            "result_1": result_1,
            "fig_c": fig_c,
            "fig_t": fig_t,
            "fig_t_3d": fig_t_3d,
        }
    )


def update_gf_cr_params(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    update_cr_phase: bool | None = None,
    update_cancel_pulse: bool | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180: TargetMap[Waveform] | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
    store_params: bool | None = None,
) -> Result:
    """Update GF-CR calibration parameters for a qubit pair."""
    if update_cr_phase is None:
        update_cr_phase = True
    if update_cancel_pulse is None:
        update_cancel_pulse = True
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    if ramptime is None:
        ramptime = _ramptime(exp, control_qubit, target_qubit)
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if cr_phase is None:
        cr_phase = 0.0
    if cancel_amplitude is None:
        cancel_amplitude = 0.0
    if cancel_phase is None:
        cancel_phase = 0.0
    if store_params is None:
        store_params = True

    current_cr_pulse = cr_amplitude * np.exp(1j * cr_phase)
    current_cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase)

    result = gf_cr_hamiltonian_tomography(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        time_range=time_range,
        ramptime=ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        x90=x90,
        x180=x180,
        ef_x180=ef_x180,
        x180_margin=x180_margin,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=reset_awg_and_capunits,
        plot=plot,
    )

    shift = -result["cr_rotation_phase"]
    cancel_pulse = -result["xt_rotation_amplitude_hw"] * np.exp(
        1j * result["xt_rotation_phase"]
    )

    new_cr_pulse = (
        current_cr_pulse * np.exp(1j * shift) if update_cr_phase else current_cr_pulse
    )
    new_cancel_pulse = (
        (current_cancel_pulse + cancel_pulse) * np.exp(1j * shift)
        if update_cancel_pulse
        else current_cancel_pulse
    )

    new_cr_amplitude = np.abs(new_cr_pulse)
    new_cr_phase = np.angle(new_cr_pulse)
    new_cancel_amplitude = np.abs(new_cancel_pulse)
    new_cancel_phase = np.angle(new_cancel_pulse)

    if plot:
        print("Updated GF-CR params:")
        print(
            f"  CR amplitude     : {cr_amplitude:+.4f} -> "
            f"{new_cr_amplitude:+.4f} "
            f"(diff: {new_cr_amplitude - cr_amplitude:+.4f})"
        )
        print(
            f"  CR phase         : {cr_phase:+.4f} -> {new_cr_phase:+.4f} "
            f"(diff: {new_cr_phase - cr_phase:+.4f})"
        )
        print(
            f"  Cancel amplitude : {cancel_amplitude:+.4f} -> "
            f"{new_cancel_amplitude:+.4f} "
            f"(diff: {new_cancel_amplitude - cancel_amplitude:+.4f})"
        )
        print(
            f"  Cancel phase     : {cancel_phase:+.4f} -> "
            f"{new_cancel_phase:+.4f} "
            f"(diff: {new_cancel_phase - cancel_phase:+.4f})"
        )

    cr_label = _gf_cr_label(control_qubit, target_qubit)
    zx_rotation_rate = result["coeffs"]["ZX"] / cr_amplitude
    cr_param: CrossResonanceParam = {
        "target": cr_label,
        "duration": 0.0,
        "ramptime": ramptime,
        "cr_amplitude": new_cr_amplitude,
        "cr_phase": new_cr_phase,
        "cr_beta": 0.0,
        "cancel_amplitude": new_cancel_amplitude,
        "cancel_phase": new_cancel_phase,
        "cancel_beta": 0.0,
        "rotary_amplitude": 0.0,
        "zx_rotation_rate": zx_rotation_rate,
    }
    if store_params:
        exp.ctx.calib_note.update_cr_param(cr_label, cr_param)

    return Result(
        data={
            **result,
            "cr_param": cr_param,
            "stored_cr_label": cr_label if store_params else None,
        }
    )


def obtain_gf_cr_params(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    n_iterations: int | None = None,
    n_cycles: int | None = None,
    n_points_per_cycle: int | None = None,
    use_stored_params: bool | None = None,
    tolerance: float | None = None,
    adiabatic_safe_factor: float | None = None,
    max_amplitude: float | None = None,
    max_time_range: float | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180: TargetMap[Waveform] | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
    store_params: bool | None = None,
) -> Result:
    """Obtain GF-CR parameters for a qubit pair."""
    if n_iterations is None:
        n_iterations = 4
    if n_cycles is None:
        n_cycles = 2
    if n_points_per_cycle is None:
        n_points_per_cycle = 6
    if use_stored_params is None:
        use_stored_params = False
    if tolerance is None:
        tolerance = 0.005e-3
    if max_amplitude is None:
        max_amplitude = 1.0
    if max_time_range is None:
        max_time_range = 4096.0
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    if store_params is None:
        store_params = True
    if ramptime is None:
        ramptime = _ramptime(exp, control_qubit, target_qubit)
    if adiabatic_safe_factor is None:
        adiabatic_safe_factor = _adiabatic_safe_factor()

    sampling_period = exp.ctx.measurement.sampling_period

    def _create_time_range(zx90_duration: float) -> NDArray:
        period = 4 * zx90_duration
        dt = (period / n_points_per_cycle) // sampling_period * sampling_period
        if dt <= 0:
            dt = sampling_period
        duration = min(period * n_cycles, max_time_range)
        return np.arange(0, duration + 1, dt)

    cr_label = _gf_cr_label(control_qubit, target_qubit)

    f_control = exp.ctx.qubits[control_qubit].frequency
    f_target = exp.ctx.qubits[target_qubit].frequency
    f_delta = np.abs(f_target - f_control)
    max_cr_rabi = adiabatic_safe_factor * f_delta
    max_cr_amplitude = exp.pulse.calc_control_amplitude(control_qubit, max_cr_rabi)
    max_cr_amplitude = float(np.clip(max_cr_amplitude, 0.0, max_amplitude))

    current_cr_param = exp.ctx.calib_note.get_cr_param(cr_label)

    if use_stored_params and current_cr_param is not None:
        cr_amplitude = current_cr_param["cr_amplitude"]
        cr_phase = current_cr_param["cr_phase"]
        cancel_amplitude = current_cr_param["cancel_amplitude"]
        cancel_phase = current_cr_param["cancel_phase"]
        zx90_duration = 1 / (4 * cr_amplitude * current_cr_param["zx_rotation_rate"])
        time_range = _create_time_range(zx90_duration)
    else:
        cr_amplitude = cr_amplitude if cr_amplitude is not None else max_cr_amplitude
        cr_phase = 0.0
        cancel_amplitude = 0.0
        cancel_phase = 0.0
        if time_range is None:
            time_range = DEFAULT_CR_TIME_RANGE
        time_range = np.array(time_range, dtype=float)

    params_history = [
        {
            "time_range": time_range,
            "cr_phase": cr_phase,
            "cancel_amplitude": cancel_amplitude,
            "cancel_phase": cancel_phase,
        }
    ]
    coeffs_history = defaultdict(list)
    figs_history = []

    print(f"Conducting GF-CR Hamiltonian tomography for {cr_label}...")
    for i in range(n_iterations):
        print(f"Iteration {i + 1}/{n_iterations}")
        params = params_history[-1]

        result = update_gf_cr_params(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=params["time_range"],
            ramptime=ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=float(params["cr_phase"]),
            cancel_amplitude=float(params["cancel_amplitude"]),
            cancel_phase=float(params["cancel_phase"]),
            x90=x90,
            x180=x180,
            ef_x180=ef_x180,
            x180_margin=x180_margin,
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot=plot,
            store_params=store_params,
        )
        next_time_range = _create_time_range(result["zx90_duration"])
        params_history.append(
            {
                "time_range": next_time_range,
                "cr_phase": result["cr_param"]["cr_phase"],
                "cancel_amplitude": result["cr_param"]["cancel_amplitude"],
                "cancel_phase": result["cr_param"]["cancel_phase"],
            }
        )
        figs_history.append(
            {
                "fig_c": result["fig_c"],
                "fig_t": result["fig_t"],
                "fig_t_3d": result["fig_t_3d"],
            }
        )
        for key, value in result["coeffs"].items():
            coeffs_history[key].append(value)

        if i > 0:
            ix = coeffs_history["IX"][-1]
            iy = coeffs_history["IY"][-1]
            ix_diff = coeffs_history["IX"][-2] - ix
            iy_diff = coeffs_history["IY"][-2] - iy

            if abs(ix) < tolerance and abs(iy) < tolerance:
                print("Convergence reached.")
                print(f"  IX : {ix * 1e3:.4f} MHz")
                print(f"  IY : {iy * 1e3:.4f} MHz")
                break
            if abs(ix_diff) < tolerance and abs(iy_diff) < tolerance:
                print("Convergence reached.")
                print(f"  IX_diff : {ix_diff * 1e3:.4f} MHz")
                print(f"  IY_diff : {iy_diff * 1e3:.4f} MHz")
                break

    hamiltonian_coeffs = {key: np.array(value) for key, value in coeffs_history.items()}

    fig = viz.make_figure()
    for key, value in hamiltonian_coeffs.items():
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(value) + 1),
                y=value * 1e3,
                mode="lines+markers",
                name=f"{key}/2",
            )
        )
    fig.update_layout(
        title=f"GF-CR Hamiltonian coefficients : {cr_label}",
        xaxis_title="Number of steps",
        yaxis_title="Coefficient (MHz)",
    )
    if plot:
        fig.show()

    return Result(
        data={
            "params_history": params_history,
            "coeffs_history": hamiltonian_coeffs,
            "figs_history": figs_history,
            "stored_cr_label": cr_label if store_params else None,
        },
        figure=fig,
    )


def calibrate_gf_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    ramptime: float | None = None,
    duration: float | None = None,
    amplitude_range: ArrayLike | None = None,
    initial_state: str | None = None,
    degree: int | None = None,
    adiabatic_safe_factor: float | None = None,
    max_amplitude: float | None = None,
    rotary_multiple: float | None = None,
    use_drag: bool | None = None,
    duration_unit: float | None = None,
    duration_buffer: float | None = None,
    n_repetitions: int | None = None,
    x180: TargetMap[Waveform] | Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    use_zvalues: bool | None = None,
    store_params: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
) -> Result:
    """Calibrate the ZX90 amplitude for a GF-CR pulse."""
    if initial_state is None:
        initial_state = "0"
    if degree is None:
        degree = 3
    if max_amplitude is None:
        max_amplitude = 1.0
    if rotary_multiple is None:
        rotary_multiple = 9.0
    if use_drag is None:
        use_drag = True
    if duration_unit is None:
        duration_unit = 16.0
    if duration_buffer is None:
        duration_buffer = 1.05
    if n_repetitions is None:
        n_repetitions = 1
    if x180_margin is None:
        x180_margin = 0.0
    if use_zvalues is None:
        use_zvalues = False
    if store_params is None:
        store_params = True
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True

    if ramptime is None:
        ramptime = _ramptime(exp, control_qubit, target_qubit)
    if adiabatic_safe_factor is None:
        adiabatic_safe_factor = _adiabatic_safe_factor()
    if x180 is None:
        control_x180 = exp.pulse.x180(control_qubit)
    elif isinstance(x180, Waveform):
        control_x180 = x180
    else:
        control_x180 = x180[control_qubit]

    cr_label = _gf_cr_label(control_qubit, target_qubit)
    cr_param = exp.ctx.calib_note.get_cr_param(cr_label)
    if cr_param is None:
        raise ValueError(f"GF-CR parameters for {cr_label} are not stored.")

    cr_amplitude = cr_param["cr_amplitude"]
    cr_phase = cr_param["cr_phase"]
    cancel_amplitude = cr_param["cancel_amplitude"]
    cancel_phase = cr_param["cancel_phase"]
    zx_rotation_rate = cr_param["zx_rotation_rate"]
    zx_frequency = zx_rotation_rate * cr_amplitude
    rotary_amplitude = exp.pulse.calc_control_amplitude(
        target=target_qubit,
        rabi_rate=zx_frequency * rotary_multiple,
    )
    cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

    f_control = exp.ctx.qubits[control_qubit].frequency
    f_target = exp.ctx.qubits[target_qubit].frequency
    f_delta = np.abs(f_target - f_control)
    max_cr_rabi = adiabatic_safe_factor * f_delta
    max_cr_amplitude = exp.pulse.calc_control_amplitude(control_qubit, max_cr_rabi)
    max_cr_amplitude = float(np.clip(max_cr_amplitude, 0.0, max_amplitude))

    if duration is None:
        if cr_param["duration"] == 0.0:
            duration = duration_buffer / (8 * zx_frequency) + ramptime
            if duration % duration_unit != 0:
                duration = (duration // duration_unit + 1) * duration_unit
        else:
            duration = cr_param["duration"]

    if duration % duration_unit != 0:
        print(
            f"Warning: Duration {duration} ns is not a multiple of "
            f"duration_unit {duration_unit} ns."
        )

    print(f"duration : {duration} ns")
    print(f"ramptime : {ramptime} ns")

    def ecr_sequence(
        amplitude: float,
        duration: float,
        n_repetitions: int,
    ) -> PulseSchedule:
        scaled_cancel_pulse = amplitude / cr_amplitude * cancel_pulse
        gf_ecr = _gf_wrapped_cr_sequence(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            duration=duration,
            ramptime=ramptime,
            cr_amplitude=amplitude,
            cr_phase=cr_phase,
            cr_beta=cr_param["cr_beta"],
            cancel_amplitude=np.abs(scaled_cancel_pulse),
            cancel_phase=np.angle(scaled_cancel_pulse),
            cancel_beta=cr_param["cancel_beta"],
            echo=True,
            x180={control_qubit: control_x180},
            x180_margin=x180_margin,
            ramp_type="RaisedCosine",
            ef_x180=ef_x180,
        ).repeated(n_repetitions)
        with PulseSchedule() as ps:
            if initial_state != "0":
                ps.add(
                    control_qubit,
                    exp.pulse.get_pulse_for_state(control_qubit, initial_state),
                )
                ps.barrier()
            ps.call(gf_ecr)
        return ps

    def calibrate(
        amplitude_range: ArrayLike,
        duration: float,
        n_repetitions: int,
    ) -> dict:
        amplitude_array = np.asarray(amplitude_range, dtype=float)
        min_amplitude = np.clip(amplitude_array[0], 0.0, max_cr_amplitude)
        max_sweep_amplitude = np.clip(amplitude_array[-1], 0.0, max_cr_amplitude)
        clipped_amplitude_range = np.linspace(
            min_amplitude,
            max_sweep_amplitude,
            len(amplitude_array),
        )
        sweep_result = exp.measurement_service.sweep_parameter(
            lambda x: ecr_sequence(
                amplitude=x,
                duration=duration,
                n_repetitions=n_repetitions,
            ),
            sweep_range=clipped_amplitude_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )

        if use_zvalues:
            signal = sweep_result.data[target_qubit].zvalues
        else:
            signal = sweep_result.data[target_qubit].normalized

        fit_result = fitting.fit_polynomial(
            target=cr_label,
            x=clipped_amplitude_range,
            y=signal,
            degree=degree,
            title=f"GF-ZX90 calibration (n = {n_repetitions})",
            xlabel="Amplitude (arb. units)",
            ylabel="Signal",
        )

        root = fit_result["root"]
        if np.isnan(root):
            root = None

        return {
            "amplitude_range": clipped_amplitude_range,
            "signal": signal,
            "root": root,
            "fit_result": fit_result,
        }

    if amplitude_range is None:
        print(
            f"Estimating GF-CR amplitude of {cr_label} "
            f"(n_repetitions = {n_repetitions})"
        )
        rough_result = calibrate(
            amplitude_range=np.linspace(0.0, cr_amplitude * 2, 20),
            duration=duration,
            n_repetitions=n_repetitions,
        )
        rough_amplitude = rough_result["root"]
        if rough_amplitude is None:
            duration = (duration * duration_buffer // duration_unit + 1) * duration_unit
            print(f"Retrying with duration = {duration} ns")
            rough_result = calibrate(
                amplitude_range=np.linspace(0.0, cr_amplitude * 2, 20),
                duration=duration,
                n_repetitions=n_repetitions,
            )
            rough_amplitude = rough_result["root"]
            if rough_amplitude is None:
                raise ValueError(
                    "Could not find a root for the GF-CR amplitude calibration."
                )
        min_amplitude = float(rough_amplitude * 0.8)
        max_sweep_amplitude = float(rough_amplitude * 1.2)
        amplitude_range = np.linspace(min_amplitude, max_sweep_amplitude, 50)
    else:
        amplitude_range = np.asarray(amplitude_range)

    print(
        f"Calibrating GF-CR amplitude of {cr_label} (n_repetitions = {n_repetitions})"
    )
    result_n1 = calibrate(
        amplitude_range=amplitude_range,
        duration=duration,
        n_repetitions=n_repetitions,
    )
    amplitude_range = np.asarray(result_n1["amplitude_range"])
    signal_n1 = result_n1["signal"]
    fit_result_n1 = result_n1["fit_result"]

    print(
        f"Calibrating GF-CR amplitude of {cr_label} "
        f"(n_repetitions = {n_repetitions + 2})"
    )
    result_n3 = calibrate(
        amplitude_range=amplitude_range,
        duration=duration,
        n_repetitions=n_repetitions + 2,
    )
    signal_n3 = result_n3["signal"]
    fit_result_n3 = result_n3["fit_result"]

    signal = signal_n1 - signal_n3
    fit_result = fitting.fit_polynomial(
        target=cr_label,
        x=amplitude_range,
        y=signal,
        degree=degree,
        title="GF-ZX90 calibration",
        xlabel="Amplitude (arb. units)",
        ylabel="Signal",
    )

    calibrated_cr_amplitude = fit_result["root"]
    if np.isnan(calibrated_cr_amplitude):
        print("Could not find a root for the GF-CR amplitude calibration.")
        calibrated_cr_amplitude = 1.0

    calibrated_cancel_amplitude = (
        calibrated_cr_amplitude / cr_amplitude * cancel_amplitude
    )
    calibrated_rotary_amplitude = (
        calibrated_cr_amplitude / cr_amplitude * rotary_amplitude
    )

    if use_drag:
        delta_ct = 2 * np.pi * (f_control - f_target)
        cr_beta = -1 / delta_ct
        cancel_beta = 0.0
    else:
        cr_beta = 0.0
        cancel_beta = 0.0

    calibrated_cr_param: CrossResonanceParam = {
        "target": cr_label,
        "duration": duration,
        "ramptime": ramptime,
        "cr_amplitude": calibrated_cr_amplitude,
        "cr_phase": cr_phase,
        "cr_beta": cr_beta,
        "cancel_amplitude": calibrated_cancel_amplitude,
        "cancel_phase": cancel_phase,
        "cancel_beta": cancel_beta,
        "rotary_amplitude": calibrated_rotary_amplitude,
        "zx_rotation_rate": zx_rotation_rate,
    }
    if calibrated_cr_amplitude is not None and store_params:
        exp.ctx.calib_note.update_cr_param(cr_label, calibrated_cr_param)

    print()
    print("Calibrated GF-CR parameters:")
    print(f"  CR duration      : {duration:.1f} ns")
    print(f"  CR ramptime      : {ramptime:.1f} ns")
    print(f"  CR amplitude     : {calibrated_cr_amplitude:.6f}")
    print(f"  CR phase         : {cr_phase:.6f}")
    print(f"  CR beta          : {cr_beta:.6f}")
    print(f"  Cancel amplitude : {calibrated_cancel_amplitude:.6f}")
    print(f"  Cancel phase     : {cancel_phase:.6f}")
    print(f"  Cancel beta      : {cancel_beta:.6f}")
    print(f"  Rotary amplitude : {calibrated_rotary_amplitude:.6f}")
    print()

    zx90 = _gf_zx90_sequence(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        cr_param=calibrated_cr_param,
        x180=control_x180,
        ef_x180=ef_x180,
        x180_margin=x180_margin,
    )

    try:
        t1_dict = exp.ctx.system_manager.config_loader.load_param_data("t1")
        t2_dict = exp.ctx.system_manager.config_loader.load_param_data("t2_echo")
        t1 = (t1_dict[control_qubit], t1_dict[target_qubit])
        t2 = (t2_dict[control_qubit], t2_dict[target_qubit])
        coherence_limit = {
            "control_qubit": control_qubit,
            "target_qubit": target_qubit,
            "gate_time": zx90.duration,
            "t1_control": t1[0],
            "t1_target": t1[1],
            "t2_control": t2[0],
            "t2_target": t2[1],
            **util.calc_2q_gate_coherence_limit(
                gate_time=zx90.duration,
                t1=t1,
                t2=t2,
            ),
        }
        print("GF-ZX90 coherence limit:")
        print(f"  Gate time       : {coherence_limit['gate_time']:.0f} ns")
        print(f"  T1 (control)    : {coherence_limit['t1_control'] * 1e-3:.1f} us")
        print(f"  T1 (target)     : {coherence_limit['t1_target'] * 1e-3:.1f} us")
        print(f"  T2 (control)    : {coherence_limit['t2_control'] * 1e-3:.1f} us")
        print(f"  T2 (target)     : {coherence_limit['t2_target'] * 1e-3:.1f} us")
        print(f"  Coherence limit : {coherence_limit['fidelity'] * 100:.2f} %")
        print()
    except KeyError:
        coherence_limit = {}

    if plot:
        zx90.plot(
            title=f"GF-ZX90 sequence : {cr_label}",
            show_physical_pulse=True,
        )

    return Result(
        data={
            "amplitude_range": amplitude_range,
            "signal": signal,
            **fit_result,
            "n1": {
                "signal": signal_n1,
                **fit_result_n1,
            },
            "n3": {
                "signal": signal_n3,
                **fit_result_n3,
            },
            "cr_param": calibrated_cr_param,
            "coherence_limit": coherence_limit,
            "stored_cr_label": cr_label if store_params else None,
        }
    )


def _resolve_gf_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    zx90: PulseSchedule | None = None,
    x180: Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float = 0.0,
) -> PulseSchedule:
    if zx90 is not None:
        return zx90

    cr_label = _gf_cr_label(control_qubit, target_qubit)
    cr_param = exp.ctx.calib_note.get_cr_param(
        cr_label,
        valid_days=exp.ctx.calibration_valid_days,
    )
    if cr_param is None:
        raise ValueError(f"GF-CR parameters for {cr_label} are not stored.")

    return _gf_zx90_sequence(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        cr_param=cr_param,
        x180=x180,
        ef_x180=ef_x180,
        x180_margin=x180_margin,
    )


def _gf_cnot_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    zx90: PulseSchedule | None = None,
    x180: Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float = 0.0,
) -> PulseSchedule:
    gf_zx90 = _resolve_gf_zx90(
        exp,
        control_qubit,
        target_qubit,
        zx90=zx90,
        x180=x180,
        ef_x180=ef_x180,
        x180_margin=x180_margin,
    )
    with PulseSchedule() as ps:
        ps.call(gf_zx90)
        ps.add(control_qubit, VirtualZ(-np.pi / 2))
        ps.add(target_qubit, exp.pulse.x90(target_qubit).scaled(-1))
    return ps


def measure_gf_bell_state(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    control_basis: str | None = None,
    target_basis: str | None = None,
    zx90: PulseSchedule | None = None,
    x180: Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    plot_sequence: bool | None = None,
    plot_raw: bool | None = None,
    plot_mitigated: bool | None = None,
    save_image: bool | None = None,
    reset_awg_and_capunits: bool | None = None,
) -> Result:
    """Measure Bell-state probabilities using a GF-CR entangling pulse."""
    if control_basis is None:
        control_basis = "Z"
    if target_basis is None:
        target_basis = "Z"
    if x180_margin is None:
        x180_margin = 0.0
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if plot_sequence is None:
        plot_sequence = False
    if plot_raw is None:
        plot_raw = True
    if plot_mitigated is None:
        plot_mitigated = True
    if save_image is None:
        save_image = True
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if exp.ctx.state_centers is None:
        exp.measurement_service.build_classifier(plot=False)

    pair = [control_qubit, target_qubit]

    with PulseSchedule() as ps:
        ps.add(control_qubit, exp.pulse.y90(control_qubit))
        ps.call(
            _gf_cnot_sequence(
                exp,
                control_qubit,
                target_qubit,
                zx90=zx90,
                x180=x180,
                ef_x180=ef_x180,
                x180_margin=x180_margin,
            )
        )

        if control_basis == "X":
            ps.add(control_qubit, exp.pulse.y90m(control_qubit))
        elif control_basis == "Y":
            ps.add(control_qubit, exp.pulse.x90(control_qubit))

        if target_basis == "X":
            ps.add(target_qubit, exp.pulse.y90m(target_qubit))
        elif target_basis == "Y":
            ps.add(target_qubit, exp.pulse.x90(target_qubit))

    result = exp.measurement_service.measure(
        ps,
        mode="single",
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=reset_awg_and_capunits,
    )

    basis_labels = result.get_basis_labels(pair)
    prob_dict_raw = result.get_probabilities(pair)
    prob_dict_raw = {label: prob_dict_raw.get(label, 0) for label in basis_labels}
    prob_dict_mitigated = result.get_mitigated_probabilities(pair)

    labels = [f"|{i}>" for i in prob_dict_raw]
    prob_arr_raw = np.array(list(prob_dict_raw.values()))
    prob_arr_mitigated = np.array(list(prob_dict_mitigated.values()))

    cr_label = _gf_cr_label(control_qubit, target_qubit)
    fig = viz.make_figure()
    if plot_raw:
        fig.add_trace(go.Bar(x=labels, y=prob_arr_raw, name="Raw"))
    if plot_mitigated:
        fig.add_trace(go.Bar(x=labels, y=prob_arr_mitigated, name="Mitigated"))
    fig.update_layout(
        title=f"GF Bell state measurement: {cr_label}",
        xaxis_title=f"State ({control_basis}{target_basis} basis)",
        yaxis_title="Probability",
        barmode="group",
        yaxis_range=[0, 1],
    )
    if plot:
        if plot_sequence:
            ps.plot(
                title=f"GF Bell state measurement: {control_basis}{target_basis} basis"
            )
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )
        for label, p_raw, p_mitigated in zip(
            labels,
            prob_arr_raw,
            prob_arr_mitigated,
            strict=True,
        ):
            print(f"{label} : {p_raw:.2%} -> {p_mitigated:.2%}")

    if save_image:
        viz.save_figure(fig, f"gf_bell_state_measurement_{cr_label}")

    return Result(
        data={
            "raw": prob_arr_raw,
            "mitigated": prob_arr_mitigated,
            "result": result,
        },
        figure=fig,
    )


def gf_bell_state_tomography(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    readout_mitigation: bool | None = None,
    zx90: PulseSchedule | None = None,
    x180: Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    mle_fit: bool | None = None,
) -> Result:
    """Perform two-qubit state tomography for a GF-CR Bell state."""
    if readout_mitigation is None:
        readout_mitigation = True
    if x180_margin is None:
        x180_margin = 0.0
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if mle_fit is None:
        mle_fit = True

    n_qubits = 2
    dim = 2**n_qubits
    probabilities = {}
    for control_basis, target_basis in tqdm(
        product(["X", "Y", "Z"], repeat=n_qubits),
        desc="Measuring GF Bell state",
    ):
        result = measure_gf_bell_state(
            exp,
            control_qubit,
            target_qubit,
            control_basis=control_basis,
            target_basis=target_basis,
            zx90=zx90,
            x180=x180,
            ef_x180=ef_x180,
            x180_margin=x180_margin,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )
        basis = f"{control_basis}{target_basis}"
        if readout_mitigation:
            probabilities[basis] = result["mitigated"]
        else:
            probabilities[basis] = result["raw"]

    expected_values = {}
    paulis = {
        "I": np.array([[1, 0], [0, 1]]),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for control_basis, control_pauli in paulis.items():
        for target_basis, target_pauli in paulis.items():
            basis = f"{control_basis}{target_basis}"
            if basis == "II":
                p = probabilities["ZZ"]
                e = p[0b00] + p[0b01] + p[0b10] + p[0b11]
            elif basis in ["IX", "IY", "IZ"]:
                p = probabilities[f"Z{target_basis}"]
                e = p[0b00] - p[0b01] + p[0b10] - p[0b11]
            elif basis in ["XI", "YI", "ZI"]:
                p = probabilities[f"{control_basis}Z"]
                e = p[0b00] + p[0b01] - p[0b10] - p[0b11]
            else:
                p = probabilities[basis]
                e = p[0b00] - p[0b01] - p[0b10] + p[0b11]
            pauli = np.kron(control_pauli, target_pauli)
            rho += e * pauli
            expected_values[basis] = e

    if mle_fit:
        rho = mle_fit_density_matrix(expected_values)
    else:
        rho = rho / dim

    bell_state = np.zeros((dim, 1), dtype=np.complex128)
    bell_state[0, 0] = 1 / np.sqrt(2)
    bell_state[-1, 0] = 1 / np.sqrt(2)
    fidelity = float(np.real(bell_state.T.conj() @ rho @ bell_state))

    cr_label = _gf_cr_label(control_qubit, target_qubit)
    fig = plot_ghz_state_tomography(
        rho=rho,
        qubits=[control_qubit, target_qubit],
        fidelity=fidelity,
        plot=plot,
        save_image=save_image,
        width=600,
        height=366,
        file_name=f"gf_bell_state_tomography_{cr_label}",
    )["figure"]

    return Result(
        data={
            "probabilities": probabilities,
            "expected_values": expected_values,
            "density_matrix": rho,
            "fidelity": fidelity,
        },
        figure=fig,
    )


def _resolve_normal_cr_label(
    exp: Experiment,
    cr_label: str,
) -> tuple[str, str, str]:
    if "-gf-" in cr_label:
        control_qubit, target_qubit = cr_label.split("-gf-", maxsplit=1)
        return control_qubit, target_qubit, f"{control_qubit}-{target_qubit}"

    control_qubit, target_qubit = exp.ctx.cr_pair(cr_label)
    return control_qubit, target_qubit, f"{control_qubit}-{target_qubit}"


def _measure_gf_zx90_rb_curve(
    exp: Experiment,
    *,
    cr_label: str,
    gf_zx90: PulseSchedule,
    interleaved: bool,
    n_cliffords_range: ArrayLike | None,
    n_trials: int,
    seeds: NDArray,
    max_n_cliffords: int,
    x90: TargetMap[Waveform] | None,
    n_shots: int,
    shot_interval: float,
    time_integration: bool,
    mitigate_readout: bool,
    reset_awg_and_capunits: bool,
) -> dict:
    control_qubit, target_qubit = exp.ctx.cr_pair(cr_label)
    if n_cliffords_range is not None:
        sweep_source = np.asarray(n_cliffords_range, dtype=int)
    else:
        sweep_source = None

    interleaved_clifford = None
    interleaved_waveform = None
    if interleaved:
        interleaved_clifford = exp.benchmarking_service.clifford.get("ZX90")
        if interleaved_clifford is None:
            raise ValueError("Invalid Clifford: ZX90")
        interleaved_waveform = gf_zx90

    idx = 0
    sweep_range = []
    mean_data = []
    std_data = []
    trial_matrix_data = []
    while True:
        if sweep_source is None:
            n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
            if n_clifford > max_n_cliffords:
                break
        else:
            if idx >= len(sweep_source):
                break
            n_clifford = int(sweep_source[idx])

        idx += 1
        sweep_range.append(n_clifford)

        trial_data = []
        for seed in seeds:
            sequence = exp.benchmarking_service.rb_sequence_2q(
                target=cr_label,
                n=n_clifford,
                x90=x90,
                zx90=gf_zx90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                seed=int(seed),
            )
            result = exp.measurement_service.measure(
                sequence=sequence,
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                time_integration=time_integration,
                reset_awg_and_capunits=reset_awg_and_capunits,
                plot=False,
            )
            if mitigate_readout:
                probabilities = result.get_mitigated_probabilities(
                    [control_qubit, target_qubit]
                )
            else:
                probabilities = result.get_probabilities([control_qubit, target_qubit])
            trial_data.append(probabilities["00"])

        trial_values = np.asarray(trial_data, dtype=float)
        trial_matrix_data.append(trial_values)
        mean = float(np.mean(trial_values))
        std = float(np.std(trial_values))
        mean_data.append(mean)
        std_data.append(std)

        if sweep_source is None and mean - std * 0.5 < 0.25:
            break

    sweep_range_array = np.asarray(sweep_range, dtype=int)
    mean_array = np.asarray(mean_data, dtype=float)
    std_array = np.asarray(std_data, dtype=float) if n_trials > 1 else None
    trial_matrix = np.vstack(trial_matrix_data)

    return {
        "n_cliffords": sweep_range_array,
        "mean": mean_array,
        "std": std_array,
        "trials": trial_matrix,
        "seeds": np.asarray(seeds, dtype=int),
    }


def gf_zx90_interleaved_randomized_benchmarking(
    exp: Experiment,
    cr_label: str,
    *,
    zx90: PulseSchedule | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180: Waveform | None = None,
    ef_x180: Waveform | None = None,
    x180_margin: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    time_integration: bool | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    reset_awg_and_capunits: bool | None = None,
) -> Result:
    """Run IRB of GF-ZX90 using GF-ZX90 as the two-qubit Clifford primitive."""
    if x180_margin is None:
        x180_margin = 0.0
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q
    if mitigate_readout is None:
        mitigate_readout = True
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if time_integration is None:
        time_integration = True
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True

    control_qubit, target_qubit, normal_cr_label = _resolve_normal_cr_label(
        exp,
        cr_label,
    )
    if exp.ctx.state_centers is None:
        raise ValueError("State classifiers are not built.")

    if seeds is None:
        seeds_array = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seeds_array = np.asarray(seeds, dtype=int)
        if len(seeds_array) != n_trials:
            raise ValueError("The number of seeds must be equal to n_trials.")

    gf_zx90 = _resolve_gf_zx90(
        exp,
        control_qubit,
        target_qubit,
        zx90=zx90,
        x180=x180,
        ef_x180=ef_x180,
        x180_margin=x180_margin,
    )

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits={control_qubit, target_qubit})

    rb_data = _measure_gf_zx90_rb_curve(
        exp,
        cr_label=normal_cr_label,
        gf_zx90=gf_zx90,
        interleaved=False,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds_array,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        n_shots=n_shots,
        shot_interval=shot_interval,
        time_integration=time_integration,
        mitigate_readout=mitigate_readout,
        reset_awg_and_capunits=False,
    )
    rb_n_cliffords = rb_data["n_cliffords"]

    irb_data = _measure_gf_zx90_rb_curve(
        exp,
        cr_label=normal_cr_label,
        gf_zx90=gf_zx90,
        interleaved=True,
        n_cliffords_range=rb_n_cliffords,
        n_trials=n_trials,
        seeds=seeds_array,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        n_shots=n_shots,
        shot_interval=shot_interval,
        time_integration=time_integration,
        mitigate_readout=mitigate_readout,
        reset_awg_and_capunits=False,
    )

    dimension = 4
    rb_fit_result = fitting.fit_rb(
        target=normal_cr_label,
        x=rb_data["n_cliffords"],
        y=rb_data["mean"],
        error_y=rb_data["std"],
        dimension=dimension,
        plot=False,
    )
    irb_fit_result = fitting.fit_rb(
        target=normal_cr_label,
        x=irb_data["n_cliffords"],
        y=irb_data["mean"],
        error_y=irb_data["std"],
        dimension=dimension,
        plot=False,
        title="GF-ZX90 interleaved randomized benchmarking",
    )

    p_rb = rb_fit_result["p"]
    p_irb = irb_fit_result["p"]
    p_rb_err = rb_fit_result["p_err"]
    p_irb_err = irb_fit_result["p_err"]
    avg_gate_error_rb = rb_fit_result["avg_gate_error"]
    avg_gate_fidelity_rb = rb_fit_result["avg_gate_fidelity"]
    avg_gate_fidelity_err_rb = rb_fit_result["avg_gate_fidelity_err"]
    avg_gate_fidelity_irb = irb_fit_result["avg_gate_fidelity"]
    avg_gate_fidelity_err_irb = irb_fit_result["avg_gate_fidelity_err"]
    gate_error = (dimension - 1) * (1 - (p_irb / p_rb)) / dimension
    gate_fidelity = 1 - gate_error
    gate_fidelity_err = (
        (dimension - 1)
        / dimension
        * np.sqrt((p_irb_err / p_rb) ** 2 + (p_rb_err * p_irb / p_rb**2) ** 2)
    )

    fig = fitting.plot_irb(
        target=normal_cr_label,
        x=rb_data["n_cliffords"],
        y_rb=rb_data["mean"],
        y_irb=irb_data["mean"],
        error_y_rb=rb_data["std"],
        error_y_irb=irb_data["std"],
        A_rb=rb_fit_result["A"],
        A_irb=irb_fit_result["A"],
        p_rb=p_rb,
        p_irb=p_irb,
        C_rb=rb_fit_result["C"],
        C_irb=irb_fit_result["C"],
        gate_fidelity=gate_fidelity,
        gate_fidelity_err=gate_fidelity_err,
        plot=plot,
        title="Interleaved randomized benchmarking of GF-ZX90",
        xlabel="Number of Cliffords",
        ylabel="Normalized signal",
    )
    gf_cr_label = _gf_cr_label(control_qubit, target_qubit)

    print()
    print(
        "Average gate fidelity (RB)  : "
        f"{avg_gate_fidelity_rb * 100:.3f} "
        f"+/- {avg_gate_fidelity_err_rb * 100:.3f}%"
    )
    print(
        "Average gate fidelity (IRB) : "
        f"{avg_gate_fidelity_irb * 100:.3f} "
        f"+/- {avg_gate_fidelity_err_irb * 100:.3f}%"
    )
    print()
    print(f"Gate error    : {gate_error * 100:.3f} +/- {gate_fidelity_err * 100:.3f}%")
    print(
        f"Gate fidelity : {gate_fidelity * 100:.3f} +/- {gate_fidelity_err * 100:.3f}%"
    )
    print()
    if gate_error < 0.1 * avg_gate_error_rb:
        print(
            "Warning: Gate error "
            f"({gate_error * 100:.3f}%) is too low compared to the "
            f"average gate error (RB) ({avg_gate_error_rb * 100:.3f}%)."
        )

    if save_image:
        viz.save_figure(
            fig,
            name=f"gf_zx90_interleaved_randomized_benchmarking_{gf_cr_label}",
        )

    return Result(
        data={
            gf_cr_label: {
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "gate_fidelity_err": gate_fidelity_err,
                "rb_fit_result": rb_fit_result,
                "irb_fit_result": irb_fit_result,
                "rb_data": {
                    key: rb_data[key]
                    for key in ("n_cliffords", "mean", "std", "trials", "seeds")
                },
                "irb_data": {
                    key: irb_data[key]
                    for key in ("n_cliffords", "mean", "std", "trials", "seeds")
                },
            }
        },
        figures={gf_cr_label: fig},
    )


def _make_control_dynamics_figure(
    *,
    result_0: Result,
    result_1: Result,
    cr_label: str,
    control_qubit: str,
    f_delta: float,
    cr_rabi_rate: float,
    ramptime: float,
) -> go.Figure:
    fig_c = viz.make_figure()
    fig_c.set_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig_c_0 = viz.make_bloch_vectors_figure(
        result_0["effective_drive_range"],
        result_0["control_states"],
    )
    fig_c_1 = viz.make_bloch_vectors_figure(
        result_1["effective_drive_range"],
        result_1["control_states"],
    )
    for trace in fig_c_0.data:
        data = cast(go.Scatter, trace)
        fig_c.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    for trace in fig_c_1.data:
        data = cast(go.Scatter, trace)
        fig_c.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig_c.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
    fig_c.update_yaxes(
        title_text="Control : |0>",
        range=[-1.1, 1.1],
        row=1,
        col=1,
    )
    fig_c.update_yaxes(
        title_text="Control : |1>",
        range=[-1.1, 1.1],
        row=2,
        col=1,
    )
    fig_c.update_layout(
        title=dict(
            text=f"Control qubit dynamics : {cr_label}",
            subtitle=dict(
                text=(
                    f"Delta = {f_delta * 1e3:.0f} MHz , "
                    f"Omega = {cr_rabi_rate * 1e3:.1f} MHz , "
                    f"tau = {ramptime:.0f} ns"
                )
            ),
        ),
        height=400,
        width=600,
        showlegend=True,
        margin=dict(t=90),
    )
    return fig_c


def _make_target_dynamics_figures(
    *,
    result_0: Result,
    result_1: Result,
    cr_label: str,
    target_qubit: str,
    f_delta: float,
    cr_rabi_rate: float,
    ramptime: float,
) -> tuple[go.Figure, go.Figure]:
    fig_t = viz.make_figure()
    fig_t.set_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    target_fit_result_0 = cast(FitResult, result_0["fit_result"])
    target_fit_result_1 = cast(FitResult, result_1["fit_result"])
    fig_t_0 = target_fit_result_0.get_figure()
    fig_t_1 = target_fit_result_1.get_figure()

    for trace in fig_t_0.data:
        data = cast(go.Scatter, trace)
        fig_t.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    for trace in fig_t_1.data:
        data = cast(go.Scatter, trace)
        fig_t.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig_t.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
    fig_t.update_yaxes(title_text="Control : |0>", range=[-1.1, 1.1], row=1, col=1)
    fig_t.update_yaxes(title_text="Control : |1>", range=[-1.1, 1.1], row=2, col=1)
    fig_t.update_layout(
        title=dict(
            text=f"Target qubit dynamics : {cr_label}",
            subtitle=dict(
                text=(
                    f"Delta = {f_delta * 1e3:.0f} MHz , "
                    f"Omega = {cr_rabi_rate * 1e3:.1f} MHz , "
                    f"tau = {ramptime:.0f} ns"
                )
            ),
        ),
        height=400,
        width=600,
        showlegend=True,
        margin=dict(t=90),
    )

    fig_t_3d = viz.make_figure()
    fig_t_3d.set_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Control : |0>", "Control : |1>"],
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        horizontal_spacing=0.01,
    )
    for trace in target_fit_result_0.get_figure("fig3d").data:
        fig_t_3d.add_trace(trace, row=1, col=1)
    for trace in target_fit_result_1.get_figure("fig3d").data:
        fig_t_3d.add_trace(trace, row=1, col=2)
    fig_t_3d.update_annotations(dict(font=dict(size=13), yshift=-20))
    fig_t_3d.update_layout(
        title=dict(
            text=f"Target qubit dynamics : {cr_label}",
            subtitle=dict(
                text=(
                    f"Delta = {f_delta * 1e3:.0f} MHz , "
                    f"Omega = {cr_rabi_rate * 1e3:.1f} MHz , "
                    f"tau = {ramptime:.0f} ns"
                )
            ),
        ),
        height=400,
        width=600,
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
    )
    return fig_t, fig_t_3d
