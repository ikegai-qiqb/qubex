"""
Characterize measurement-induced state transitions from a readout cycle.

The module constructs repeated candidate and blank-reference schedules, estimates
GEF populations, fits a shared column-stochastic transition matrix, and reports
conditional local uncertainty and model diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import chi2

from qubex.analysis import fitting
from qubex.contrib.experiment.gef_population_estimation import measure_gef_populations
from qubex.experiment import Experiment
from qubex.experiment.models.result import Result
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.pulse import Blank, PulseSchedule

# Data records and public conventions

STATE_ORDER = ("g", "e", "f")
PROTOCOL_ORDER = ("initial_g", "initial_e", "initial_f")
TRANSITION_MATRIX_INDICES = (
    ("g_to_e", 1, 0),
    ("e_to_g", 0, 1),
    ("g_to_f", 2, 0),
    ("e_to_f", 2, 1),
    ("f_to_g", 0, 2),
    ("f_to_e", 1, 2),
)
TRANSITION_ORDER = tuple(name for name, _, _ in TRANSITION_MATRIX_INDICES)


class ReadoutMistProbabilityStatus(str, Enum):
    """Represent whether one transition probability is locally resolved."""

    RESOLVED = "resolved"
    CONSISTENT_WITH_ZERO = "consistent_with_zero"
    PARTIALLY_UNRESOLVED = "partially_unresolved"
    FIT_FAILED = "fit_failed"


class ReadoutMistFitQuality(str, Enum):
    """Represent absolute agreement with shot-noise covariance."""

    ACCEPTED = "accepted"
    STATISTICAL_MISMATCH = "statistical_mismatch"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class GefPopulationFitDiagnostic:
    """Store numerical diagnostics for one GEF population estimate."""

    objective: float
    design_rank: int
    design_condition_number: float
    residual: NDArray[np.float64]
    message: str

    def __post_init__(self) -> None:
        """Normalize the residual and validate scalar diagnostics."""
        residual = np.asarray(self.residual, dtype=np.float64)
        if residual.ndim != 1 or not np.all(np.isfinite(residual)):
            raise ValueError("GEF fit residual must be a finite one-dimensional array.")
        if not np.isfinite(self.objective) or self.objective < 0.0:
            raise ValueError("GEF fit objective must be finite and nonnegative.")
        if self.design_rank < 0:
            raise ValueError("GEF fit design_rank must be nonnegative.")
        if np.isnan(self.design_condition_number) or self.design_condition_number < 0.0:
            raise ValueError(
                "GEF fit design_condition_number must be nonnegative and not NaN."
            )
        object.__setattr__(self, "residual", residual)


@dataclass(frozen=True)
class GefPopulationSeries:
    """Store one processed GEF population trajectory."""

    population: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    population_unconstrained: NDArray[np.float64]
    fit_diagnostics: tuple[GefPopulationFitDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Normalize arrays and require one consistent repetition grid."""
        population = np.asarray(self.population, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        standard_error = np.asarray(self.standard_error, dtype=np.float64)
        unconstrained = np.asarray(self.population_unconstrained, dtype=np.float64)
        if population.ndim != 2 or population.shape[1:] != (3,):
            raise ValueError("population must have shape (n_points, 3).")
        point_count = population.shape[0]
        if covariance.shape != (point_count, 3, 3):
            raise ValueError("covariance must have shape (n_points, 3, 3).")
        if standard_error.shape != population.shape:
            raise ValueError("standard_error must match population shape.")
        if unconstrained.shape != population.shape:
            raise ValueError("population_unconstrained must match population shape.")
        if not all(
            np.all(np.isfinite(values))
            for values in (population, covariance, standard_error, unconstrained)
        ):
            raise ValueError("GEF population series arrays must be finite.")
        if np.any(population < -1e-10) or not np.allclose(
            np.sum(population, axis=1),
            1.0,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("population rows must lie on the probability simplex.")
        if np.any(standard_error < 0.0):
            raise ValueError("standard_error must be nonnegative.")
        if self.fit_diagnostics and len(self.fit_diagnostics) != point_count:
            raise ValueError("fit_diagnostics must be empty or match n_points.")
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "population_unconstrained", unconstrained)


@dataclass(frozen=True)
class ReadoutMistConditionMeasurements:
    """Store actual and optional blank-reference data for one preparation."""

    actual: GefPopulationSeries
    reference: GefPopulationSeries | None


@dataclass(frozen=True)
class ReadoutMistMeasurements:
    """Store G/E/F trajectories on a shared repetition grid."""

    repetition_counts: NDArray[np.int64]
    protocols: dict[str, ReadoutMistConditionMeasurements]

    def __post_init__(self) -> None:
        """Require all three preparations on one shared repetition grid."""
        counts = np.asarray(self.repetition_counts)
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(
                "repetition_counts must be a one-dimensional integer array."
            )
        if set(self.protocols) != set(PROTOCOL_ORDER):
            raise ValueError(
                "protocols must contain exactly initial_g, initial_e, initial_f."
            )
        if (
            counts.size < 2
            or counts[0] != 0
            or 1 not in counts
            or np.any(counts < 0)
            or np.any(np.diff(counts) <= 0)
        ):
            raise ValueError(
                "repetition_counts must be increasing, nonnegative, and contain 0 and 1."
            )
        for protocol in PROTOCOL_ORDER:
            measured = self.protocols[protocol]
            if measured.actual.population.shape[0] != counts.size:
                raise ValueError(
                    f"{protocol} actual data must match repetition_counts."
                )
            if (
                measured.reference is not None
                and measured.reference.population.shape[0] != counts.size
            ):
                raise ValueError(
                    f"{protocol} reference data must match repetition_counts."
                )
        object.__setattr__(
            self, "repetition_counts", counts.astype(np.int64, copy=False)
        )


@dataclass(frozen=True)
class ProbabilityEstimate:
    """Store one probability with conditional local uncertainty."""

    value: float
    standard_error: float | None
    status: ReadoutMistProbabilityStatus
    unit: str = "1/readout_cycle"


@dataclass(frozen=True)
class FitGoodnessOfFit:
    """Store absolute fit diagnostics for whitened residuals."""

    degrees_of_freedom: int
    chi_squared: float
    reduced_chi_squared: float
    chi_squared_p_value: float | None
    maximum_absolute_whitened_residual: float
    quality: ReadoutMistFitQuality
    reason: str | None = None


@dataclass(frozen=True)
class ReadoutMistMatrixFit:
    """Store the full six-transition stochastic-matrix fit."""

    success: bool
    transition_matrix: NDArray[np.float64]
    transition_probabilities: dict[str, ProbabilityEstimate]
    derived_metrics: dict[str, ProbabilityEstimate]
    initial_populations: dict[str, NDArray[np.float64]]
    fitted_populations: dict[str, NDArray[np.float64]]
    parameter_order: tuple[str, ...]
    parameter_covariance: NDArray[np.float64]
    goodness_of_fit: FitGoodnessOfFit
    whitened_residual: NDArray[np.float64]
    message: str
    state_order: tuple[str, str, str] = STATE_ORDER
    probability_unit: str = "1/readout_cycle"

    @property
    def fit_quality(self) -> ReadoutMistFitQuality:
        """Return the nonduplicated absolute fit-quality classification."""
        return self.goodness_of_fit.quality


@dataclass(frozen=True)
class ReadoutMistWarning:
    """Describe one nonfatal limitation detected by the analysis."""

    code: str
    message: str
    affected_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadoutMistAnalysis:
    """
    Store actual/reference fits, diagnostic differences, and warnings.

    `reference_difference` is the signed elementwise difference between the
    independently fitted actual and blank-reference transition probabilities.
    It is diagnostic only and is not a readout-only transition probability.
    """

    actual: ReadoutMistMatrixFit
    reference: ReadoutMistMatrixFit | None
    reference_difference: dict[str, float] | None
    warnings: tuple[ReadoutMistWarning, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# Pulse schedule construction

AcquisitionOrder = Literal["interleaved", "randomized", "grouped"]


@dataclass(frozen=True)
class ReadoutMistSchedules:
    """Store schedules in semantic and submitted acquisition order."""

    by_condition: dict[str, tuple[PulseSchedule, ...]]
    preparations: dict[str, PulseSchedule]
    submitted: dict[str, PulseSchedule]
    names_by_condition: dict[str, tuple[str, ...]]
    acquisition_order: tuple[str, ...]
    cycle_duration_ns: float


def _finite_nonnegative(value: float, name: str) -> float:
    """Validate one finite nonnegative duration."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return normalized


def build_default_readout_cycle(
    exp: Experiment,
    qubit: str,
    *,
    cycle_gap_ns: float,
) -> PulseSchedule:
    """Build the configured readout drive followed by the complete trailing gap."""
    gap = _finite_nonnegative(cycle_gap_ns, "cycle_gap_ns")
    readout_label = exp.ctx.resolve_read_label(qubit)
    readout = exp.pulse.readout(qubit)
    with PulseSchedule([readout_label]) as cycle:
        cycle.add(readout_label, readout)
        if gap > 0.0:
            cycle.add(
                readout_label,
                Blank(gap, sampling_period=readout.sampling_period),
            )
    return cycle


def validate_readout_cycle(cycle: PulseSchedule) -> PulseSchedule:
    """Validate that an override is a finite positive complete cycle."""
    if not isinstance(cycle, PulseSchedule):
        raise TypeError("readout_cycle must be a PulseSchedule or None.")
    if not cycle.is_valid():
        raise ValueError("readout_cycle must be a valid PulseSchedule.")
    if not np.isfinite(cycle.duration) or cycle.duration <= 0.0:
        raise ValueError("readout_cycle must have positive finite duration.")
    return cycle.copy()


def build_reference_cycle(candidate: PulseSchedule) -> PulseSchedule:
    """Zero every waveform while retaining the complete cycle timing and metadata."""
    return candidate.scaled(0.0)


def _preparations(exp: Experiment, qubit: str) -> dict[str, PulseSchedule]:
    """Build G, E, and F preparations using calibrated GE/EF pi pulses."""
    ge_label = exp.ctx.resolve_ge_label(qubit)
    ef_label = exp.ctx.resolve_ef_label(qubit)
    preparations: dict[str, PulseSchedule] = {}
    with PulseSchedule([ge_label, ef_label]) as ground:
        pass
    preparations["initial_g"] = ground
    with PulseSchedule([ge_label, ef_label]) as excited:
        excited.add(ge_label, exp.pulse.x180(ge_label))
        excited.barrier()
    preparations["initial_e"] = excited
    with PulseSchedule([ge_label, ef_label]) as second_excited:
        second_excited.add(ge_label, exp.pulse.x180(ge_label))
        second_excited.barrier()
        second_excited.add(ef_label, exp.pulse.x180(ef_label))
        second_excited.barrier()
    preparations["initial_f"] = second_excited
    return preparations


def _logical_order(
    repetition_counts: Sequence[int],
    order: AcquisitionOrder,
    seed: int | None,
) -> list[tuple[str, int]]:
    """Return deterministic grouped/interleaved or seeded randomized points."""
    if order == "grouped":
        return [
            (protocol, index)
            for protocol in PROTOCOL_ORDER
            for index in range(len(repetition_counts))
        ]
    points = []
    for index in range(len(repetition_counts)):
        offset = index % len(PROTOCOL_ORDER)
        protocols = (*PROTOCOL_ORDER[offset:], *PROTOCOL_ORDER[:offset])
        points.extend((protocol, index) for protocol in protocols)
    if order == "randomized":
        generator = np.random.default_rng(seed)
        generator.shuffle(points)
    elif order != "interleaved":
        raise ValueError(
            "acquisition_order must be 'interleaved', 'randomized', or 'grouped'."
        )
    return points


def build_mist_schedules(
    exp: Experiment,
    qubit: str,
    cycle: PulseSchedule,
    repetition_counts: Sequence[int],
    *,
    prefix: str = "actual",
    acquisition_order: AcquisitionOrder = "interleaved",
    acquisition_seed: int | None = 0,
) -> ReadoutMistSchedules:
    """Build G/E/F schedules while preserving the trailing gap in every cycle."""
    preparations = _preparations(exp, qubit)
    by_condition: dict[str, tuple[PulseSchedule, ...]] = {}
    names_by_condition: dict[str, tuple[str, ...]] = {}
    lookup: dict[tuple[str, int], tuple[str, PulseSchedule]] = {}
    for protocol in PROTOCOL_ORDER:
        schedules = []
        names = []
        for index, count in enumerate(repetition_counts):
            with PulseSchedule() as schedule:
                schedule.call(preparations[protocol], copy=True)
                if count > 0:
                    schedule.barrier()
                    schedule.call(cycle.repeated(int(count)), copy=True)
            name = f"{prefix}/{protocol}/n={int(count)}"
            schedules.append(schedule)
            names.append(name)
            lookup[protocol, index] = (name, schedule)
        by_condition[protocol] = tuple(schedules)
        names_by_condition[protocol] = tuple(names)
    ordered_points = _logical_order(
        repetition_counts,
        acquisition_order,
        acquisition_seed,
    )
    submitted = {lookup[point][0]: lookup[point][1] for point in ordered_points}
    return ReadoutMistSchedules(
        by_condition=by_condition,
        preparations=preparations,
        submitted=submitted,
        names_by_condition=names_by_condition,
        acquisition_order=tuple(submitted),
        cycle_duration_ns=float(cycle.duration),
    )


# Numerical fitting helpers

GOODNESS_OF_FIT_REDUCED_CHI_SQUARED_LIMIT = 5.0
GOODNESS_OF_FIT_MAX_RESIDUAL_LIMIT = 8.0
_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class ResidualBlock:
    """Store one population observation and its covariance whitener."""

    observed: NDArray[np.float64]
    whitener: NDArray[np.float64]


@dataclass(frozen=True)
class FullFitResult:
    """
    Store one nonlinear GLS result.

    `success` records optimizer convergence. A converged central estimate can
    therefore have a NaN covariance when its Jacobian is not locally
    identifiable.
    """

    success: bool
    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64]
    whitened_residual: NDArray[np.float64]
    goodness_of_fit: FitGoodnessOfFit
    message: str


def covariance_whitener(
    covariance: ArrayLike,
    *,
    rcond: float,
) -> NDArray[np.float64]:
    """Return a PSD pseudo-inverse square root for resolved covariance modes."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        raise ValueError("population covariance must be a finite 3x3 matrix.")
    scale = max(float(np.max(np.abs(cov))), _EPS)
    if not np.allclose(
        cov,
        cov.T,
        rtol=1e-7,
        atol=float(max(1e-15, scale * 1e-10)),
    ):
        raise ValueError("population covariance must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), _EPS)
    cutoff = max(rcond, 0.0) * spectral_scale
    if float(np.min(eigenvalues)) < -max(cutoff, 1e-15 * spectral_scale):
        raise ValueError("population covariance must be positive semidefinite.")
    retained = eigenvalues > cutoff
    if not np.any(retained):
        raise ValueError("population covariance has no resolved positive mode.")
    return np.asarray(
        (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])).T,
        dtype=np.float64,
    )


def whiten_predictions(
    predictions: Sequence[ArrayLike],
    blocks: Sequence[ResidualBlock],
) -> NDArray[np.float64]:
    """Concatenate covariance-whitened prediction residuals."""
    if len(predictions) != len(blocks):
        raise ValueError("predictions must match residual blocks.")
    residuals = []
    for prediction, block in zip(predictions, blocks, strict=True):
        predicted = np.asarray(prediction, dtype=np.float64)
        if predicted.shape != (3,):
            raise ValueError("each population prediction must have shape (3,).")
        residuals.append(block.whitener @ (predicted - block.observed))
    return np.concatenate(residuals)


def assess_goodness_of_fit(
    residual: ArrayLike,
    *,
    n_parameters: int,
) -> FitGoodnessOfFit:
    """Classify absolute agreement with shot-noise-whitened observations."""
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    degrees_of_freedom = int(values.size - n_parameters)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return FitGoodnessOfFit(
            degrees_of_freedom=degrees_of_freedom,
            chi_squared=float("nan"),
            reduced_chi_squared=float("nan"),
            chi_squared_p_value=None,
            maximum_absolute_whitened_residual=float("nan"),
            quality=ReadoutMistFitQuality.NOT_ASSESSED,
            reason="No finite whitened residuals were available.",
        )
    chi_squared = float(values @ values)
    maximum = float(np.max(np.abs(values)))
    if degrees_of_freedom <= 0:
        return FitGoodnessOfFit(
            degrees_of_freedom=degrees_of_freedom,
            chi_squared=chi_squared,
            reduced_chi_squared=float("nan"),
            chi_squared_p_value=None,
            maximum_absolute_whitened_residual=maximum,
            quality=ReadoutMistFitQuality.NOT_ASSESSED,
            reason=(
                "Absolute goodness-of-fit requires positive residual degrees of "
                "freedom."
            ),
        )
    reduced = chi_squared / degrees_of_freedom
    p_value = float(chi2.sf(chi_squared, degrees_of_freedom))
    accepted = bool(
        reduced <= GOODNESS_OF_FIT_REDUCED_CHI_SQUARED_LIMIT
        and maximum <= GOODNESS_OF_FIT_MAX_RESIDUAL_LIMIT
    )
    reason = None
    if not accepted:
        reason = (
            "Whitened residuals exceed the conservative absolute fit limits "
            f"(reduced chi-squared={reduced:.3g}, max |r|={maximum:.3g})."
        )
    return FitGoodnessOfFit(
        degrees_of_freedom=degrees_of_freedom,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced,
        chi_squared_p_value=p_value,
        maximum_absolute_whitened_residual=maximum,
        quality=(
            ReadoutMistFitQuality.ACCEPTED
            if accepted
            else ReadoutMistFitQuality.STATISTICAL_MISMATCH
        ),
        reason=reason,
    )


def _local_covariance(
    jacobian: NDArray[np.float64],
    *,
    reduced_chi_squared: float,
) -> NDArray[np.float64]:
    """Return scaled local covariance or NaNs for an unresolved Jacobian."""
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        size = jacobian.shape[-1] if jacobian.ndim > 0 else 0
        return np.full((size, size), np.nan)
    parameter_count = jacobian.shape[1]
    if parameter_count == 0:
        return np.empty((0, 0), dtype=np.float64)
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((parameter_count, parameter_count), np.nan)
    largest = float(np.max(singular_values)) if singular_values.size else 0.0
    tolerance = max(
        1e-12 * largest,
        _EPS * max(jacobian.shape) * largest,
    )
    if (
        singular_values.size < parameter_count
        or not np.all(np.isfinite(singular_values))
        or largest <= 0.0
        or np.count_nonzero(singular_values > tolerance) < parameter_count
    ):
        return np.full((parameter_count, parameter_count), np.nan)
    covariance = (vt.T / np.square(singular_values)) @ vt
    return np.asarray(
        0.5 * (covariance + covariance.T) * max(1.0, reduced_chi_squared),
        dtype=np.float64,
    )


def _run_fit(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial: NDArray[np.float64],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    loss: str,
) -> OptimizeResult | None:
    """Run one bounded least-squares attempt and normalize numerical failure."""
    try:
        return least_squares(
            residual,
            initial,
            bounds=bounds,
            loss=loss,
            f_scale=1.0,
            max_nfev=5000,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def fit_full_gls(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initializations: Sequence[ArrayLike],
    *,
    parameter_count: int,
) -> FullFitResult:
    """
    Fit robust initializations followed by final ordinary GLS.

    The central estimate remains successful when optimization converges but
    local covariance is unavailable. Callers must inspect covariance finiteness
    separately when reporting uncertainty.
    """
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, Integral):
        raise TypeError("parameter_count must be a positive integer.")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer.")
    parameter_count = int(parameter_count)
    starts = tuple(np.asarray(value, dtype=np.float64) for value in initializations)
    if not starts:
        raise ValueError("initializations must contain at least one parameter vector.")
    if any(value.shape != (parameter_count,) for value in starts):
        raise ValueError("each initialization must match parameter_count.")
    if any(not np.all(np.isfinite(value)) for value in starts):
        raise ValueError("initializations must be finite.")
    lower = np.full(parameter_count, -30.0)
    upper = np.full(parameter_count, 30.0)
    solutions: list[tuple[float, OptimizeResult, NDArray[np.float64]]] = []
    for initial in starts:
        robust = _run_fit(residual, initial, (lower, upper), loss="soft_l1")
        final_initial = (
            np.asarray(robust.x, dtype=np.float64)
            if robust is not None and np.all(np.isfinite(robust.x))
            else initial
        )
        final = _run_fit(residual, final_initial, (lower, upper), loss="linear")
        if final is None or not final.success or not np.all(np.isfinite(final.x)):
            continue
        whitened = np.asarray(final.fun, dtype=np.float64).reshape(-1)
        if whitened.size and np.all(np.isfinite(whitened)):
            solutions.append((float(whitened @ whitened), final, whitened))
    if not solutions:
        empty = np.full(parameter_count, np.nan)
        goodness = assess_goodness_of_fit(np.empty(0), n_parameters=parameter_count)
        return FullFitResult(
            success=False,
            parameters=empty,
            covariance=np.full((parameter_count, parameter_count), np.nan),
            whitened_residual=np.empty(0),
            goodness_of_fit=goodness,
            message="All nonlinear GLS initializations failed.",
        )
    _, final, whitened = min(solutions, key=lambda item: item[0])
    jacobian = np.asarray(final.jac, dtype=np.float64)
    dof = max(1, whitened.size - parameter_count)
    covariance = _local_covariance(
        jacobian,
        reduced_chi_squared=float(whitened @ whitened) / dof,
    )
    covariance_available = bool(np.all(np.isfinite(covariance)))
    return FullFitResult(
        success=True,
        parameters=np.asarray(final.x, dtype=np.float64),
        covariance=covariance,
        whitened_residual=whitened,
        goodness_of_fit=assess_goodness_of_fit(
            whitened,
            n_parameters=parameter_count,
        ),
        message=(
            str(final.message)
            if covariance_available
            else (
                f"{final.message} Central parameters converged, but the parameters "
                "are not locally identifiable; local covariance is unavailable."
            )
        ),
    )


# Transition-matrix analysis

_PARAMETER_COUNT = 12
_INITIAL_ANCHORS = {"initial_g": 0, "initial_e": 1, "initial_f": 2}
_GEF_OBJECTIVE_WARNING_THRESHOLD = 100.0
_GEF_CONDITION_NUMBER_WARNING_THRESHOLD = 1e10


def _validate_covariance_rcond(value: float, *, name: str) -> float:
    """Validate a relative covariance cutoff without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be in [0, 1).")
    return normalized


def _simplex_from_logits(
    logits: NDArray[np.float64],
    *,
    anchor: int,
) -> NDArray[np.float64]:
    """Map two log ratios to a three-component simplex with one anchor."""
    destinations = tuple(index for index in range(3) if index != anchor)
    weights = np.ones(3, dtype=np.float64)
    weights[list(destinations)] = np.exp(logits)
    return weights / np.sum(weights)


def _simplex_logits(
    population: NDArray[np.float64],
    *,
    anchor: int,
) -> NDArray[np.float64]:
    """Return clipped log ratios relative to one simplex anchor."""
    clipped = np.clip(np.asarray(population, dtype=np.float64), 1e-10, None)
    clipped /= np.sum(clipped)
    destinations = tuple(index for index in range(3) if index != anchor)
    return np.log(clipped[list(destinations)] / clipped[anchor])


def transition_matrix_from_parameters(
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Decode the first six fit parameters into `T[final, initial]`."""
    matrix = np.zeros((3, 3), dtype=np.float64)
    for initial_state in range(3):
        probabilities = _simplex_from_logits(
            parameters[2 * initial_state : 2 * initial_state + 2],
            anchor=initial_state,
        )
        matrix[:, initial_state] = probabilities
    return matrix


def initial_populations_from_parameters(
    parameters: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Decode the final six fit parameters into three latent populations."""
    return {
        protocol: _simplex_from_logits(
            parameters[6 + 2 * index : 8 + 2 * index],
            anchor=_INITIAL_ANCHORS[protocol],
        )
        for index, protocol in enumerate(PROTOCOL_ORDER)
    }


def _predictions(
    parameters: NDArray[np.float64],
    repetition_counts: NDArray[np.int64],
) -> dict[str, NDArray[np.float64]]:
    """Evaluate all population trajectories in protocol/count order."""
    transition = transition_matrix_from_parameters(parameters)
    initial = initial_populations_from_parameters(parameters)
    return {
        protocol: np.stack(
            [
                np.linalg.matrix_power(transition, int(count)) @ initial[protocol]
                for count in repetition_counts
            ]
        )
        for protocol in PROTOCOL_ORDER
    }


def _project_simplex(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project one finite vector onto the probability simplex."""
    values = np.asarray(vector, dtype=np.float64)
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, values.size + 1)
    active = sorted_values - cumulative / indices > 0.0
    rho = int(np.nonzero(active)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(values - threshold, 0.0)


def _transition_initial_guess(
    populations: Mapping[str, NDArray[np.float64]],
    repetition_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Estimate a one-step matrix from measured zero/one trajectories."""
    n0 = int(np.nonzero(repetition_counts == 0)[0][0])
    n1 = int(np.nonzero(repetition_counts == 1)[0][0])
    p0 = np.column_stack([populations[name][n0] for name in PROTOCOL_ORDER])
    p1 = np.column_stack([populations[name][n1] for name in PROTOCOL_ORDER])
    transition = p1 @ np.linalg.pinv(p0)
    transition = np.column_stack(
        [_project_simplex(transition[:, index]) for index in range(3)]
    )
    values = []
    for initial_state in range(3):
        destinations = tuple(index for index in range(3) if index != initial_state)
        column = np.clip(transition[:, initial_state], 1e-10, None)
        values.extend(np.log(column[list(destinations)] / column[initial_state]))
    for protocol in PROTOCOL_ORDER:
        values.extend(
            _simplex_logits(
                populations[protocol][n0],
                anchor=_INITIAL_ANCHORS[protocol],
            )
        )
    return np.clip(np.asarray(values, dtype=np.float64), -25.0, 25.0)


def _identity_biased_guess(
    populations: Mapping[str, NDArray[np.float64]],
    repetition_counts: NDArray[np.int64],
    transition_probability: float,
) -> NDArray[np.float64]:
    """Build a stable alternative initialization around an identity map."""
    n0 = int(np.nonzero(repetition_counts == 0)[0][0])
    stay = 1.0 - 2.0 * transition_probability
    transition_logits = np.full(
        6,
        np.log(transition_probability / stay),
        dtype=np.float64,
    )
    initial_logits = np.concatenate(
        [
            _simplex_logits(
                populations[protocol][n0],
                anchor=_INITIAL_ANCHORS[protocol],
            )
            for protocol in PROTOCOL_ORDER
        ]
    )
    return np.concatenate((transition_logits, initial_logits))


def _physical_vector(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return six transitions followed by all three initial populations."""
    matrix = transition_matrix_from_parameters(parameters)
    initial = initial_populations_from_parameters(parameters)
    transitions = np.array(
        [matrix[final, initial] for _, final, initial in TRANSITION_MATRIX_INDICES],
        dtype=np.float64,
    )
    return np.concatenate((transitions, *(initial[name] for name in PROTOCOL_ORDER)))


def _numerical_jacobian(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a centered finite-difference Jacobian."""
    baseline = np.asarray(function(parameters), dtype=np.float64)
    jacobian = np.empty((baseline.size, parameters.size), dtype=np.float64)
    for index in range(parameters.size):
        step = 1e-5 * max(1.0, abs(float(parameters[index])))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        jacobian[:, index] = (function(upper) - function(lower)) / (2.0 * step)
    return jacobian


def _estimate(
    value: float,
    variance: float,
) -> ProbabilityEstimate:
    """Build a probability estimate and local resolution status."""
    standard_error = (
        float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else None
    )
    if standard_error is None:
        status = ReadoutMistProbabilityStatus.PARTIALLY_UNRESOLVED
    elif value <= 2.0 * standard_error:
        status = ReadoutMistProbabilityStatus.CONSISTENT_WITH_ZERO
    else:
        status = ReadoutMistProbabilityStatus.RESOLVED
    return ProbabilityEstimate(
        value=float(value),
        standard_error=standard_error,
        status=status,
    )


def _derived_estimate(
    physical: NDArray[np.float64],
    covariance: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> ProbabilityEstimate:
    """Propagate correlated transition uncertainty to a linear metric."""
    return _estimate(
        float(weights @ physical[:6]),
        float(weights @ covariance[:6, :6] @ weights),
    )


def _failed_fit(message: str) -> ReadoutMistMatrixFit:
    """Build a stable unavailable fit result."""
    estimates = {
        name: ProbabilityEstimate(
            value=float("nan"),
            standard_error=None,
            status=ReadoutMistProbabilityStatus.FIT_FAILED,
        )
        for name in TRANSITION_ORDER
    }
    goodness = FitGoodnessOfFit(
        degrees_of_freedom=0,
        chi_squared=float("nan"),
        reduced_chi_squared=float("nan"),
        chi_squared_p_value=None,
        maximum_absolute_whitened_residual=float("nan"),
        quality=ReadoutMistFitQuality.NOT_ASSESSED,
        reason=message,
    )
    return ReadoutMistMatrixFit(
        success=False,
        transition_matrix=np.full((3, 3), np.nan),
        transition_probabilities=estimates,
        derived_metrics={},
        initial_populations={},
        fitted_populations={},
        parameter_order=(),
        parameter_covariance=np.full((15, 15), np.nan),
        goodness_of_fit=goodness,
        whitened_residual=np.empty(0),
        message=message,
    )


def _fit_condition(
    measurements: ReadoutMistMeasurements,
    *,
    reference: bool,
    covariance_rcond: float,
) -> ReadoutMistMatrixFit:
    """Fit actual or reference trajectories with the full 12-parameter model."""
    series = {}
    for protocol in PROTOCOL_ORDER:
        condition = measurements.protocols[protocol]
        value = condition.reference if reference else condition.actual
        if value is None:
            return _failed_fit("Reference populations were not acquired.")
        series[protocol] = value
    blocks = [
        ResidualBlock(
            observed=series[protocol].population[index],
            whitener=covariance_whitener(
                series[protocol].covariance[index],
                rcond=covariance_rcond,
            ),
        )
        for protocol in PROTOCOL_ORDER
        for index in range(measurements.repetition_counts.size)
    ]
    populations = {name: series[name].population for name in PROTOCOL_ORDER}

    def block_predictions(parameters: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        predicted = _predictions(parameters, measurements.repetition_counts)
        return [
            predicted[protocol][index]
            for protocol in PROTOCOL_ORDER
            for index in range(measurements.repetition_counts.size)
        ]

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return whiten_predictions(block_predictions(parameters), blocks)

    initializations = (
        _transition_initial_guess(populations, measurements.repetition_counts),
        _identity_biased_guess(
            populations,
            measurements.repetition_counts,
            1e-3,
        ),
        _identity_biased_guess(
            populations,
            measurements.repetition_counts,
            1e-2,
        ),
    )
    fitted = fit_full_gls(
        residual,
        initializations,
        parameter_count=_PARAMETER_COUNT,
    )
    if not fitted.success:
        return _failed_fit(fitted.message)
    physical = _physical_vector(fitted.parameters)
    transform = _numerical_jacobian(_physical_vector, fitted.parameters)
    covariance = transform @ fitted.covariance @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    transition_estimates = {
        name: _estimate(physical[index], covariance[index, index])
        for index, name in enumerate(TRANSITION_ORDER)
    }
    transition_index = {name: index for index, name in enumerate(TRANSITION_ORDER)}

    def weights(**values: float) -> NDArray[np.float64]:
        """Build a linear derived-metric vector in public transition order."""
        result = np.zeros(len(TRANSITION_ORDER), dtype=np.float64)
        for name, value in values.items():
            result[transition_index[name]] = value
        return result

    derived_weights = {
        "leakage_from_g": weights(g_to_f=1.0),
        "leakage_from_e": weights(e_to_f=1.0),
        "average_computational_leakage": weights(g_to_f=0.5, e_to_f=0.5),
        "seepage_from_f": weights(f_to_g=1.0, f_to_e=1.0),
    }
    derived = {
        name: _derived_estimate(physical, covariance, weights)
        for name, weights in derived_weights.items()
    }
    initial_populations = initial_populations_from_parameters(fitted.parameters)
    predicted = _predictions(fitted.parameters, measurements.repetition_counts)
    physical_order = (
        *TRANSITION_ORDER,
        *(
            f"{protocol}_{state}"
            for protocol in PROTOCOL_ORDER
            for state in STATE_ORDER
        ),
    )
    return ReadoutMistMatrixFit(
        success=True,
        transition_matrix=transition_matrix_from_parameters(fitted.parameters),
        transition_probabilities=transition_estimates,
        derived_metrics=derived,
        initial_populations=initial_populations,
        fitted_populations=predicted,
        parameter_order=physical_order,
        parameter_covariance=covariance,
        goodness_of_fit=fitted.goodness_of_fit,
        whitened_residual=fitted.whitened_residual,
        message=fitted.message,
    )


def _population_model_warnings(
    measurements: ReadoutMistMeasurements,
) -> tuple[ReadoutMistWarning, ...]:
    """Detect GEF boundary, conditioning, and model-mismatch diagnostics."""
    unconstrained_affected = []
    boundary_affected = []
    fit_affected = []
    for protocol, measured in measurements.protocols.items():
        for label, series in (
            ("actual", measured.actual),
            ("reference", measured.reference),
        ):
            if series is None:
                continue
            if np.any(series.population_unconstrained < -0.02) or np.any(
                series.population_unconstrained > 1.02
            ):
                unconstrained_affected.append(f"{label}/{protocol}")
            if np.any(series.population <= 1e-10):
                boundary_affected.append(f"{label}/{protocol}")
            if any(
                diagnostic.design_rank < 2
                or diagnostic.design_condition_number
                > _GEF_CONDITION_NUMBER_WARNING_THRESHOLD
                # The GEF estimator has nominally 15 feature observations and
                # two population degrees of freedom. This deliberately loose
                # threshold flags only gross residual mismatch.
                or diagnostic.objective > _GEF_OBJECTIVE_WARNING_THRESHOLD
                for diagnostic in series.fit_diagnostics
            ):
                fit_affected.append(f"{label}/{protocol}")
    warnings = []
    if unconstrained_affected:
        warnings.append(
            ReadoutMistWarning(
                code="possible_higher_level_leakage",
                message=(
                    "GEF sum-to-one estimates disagree with unconstrained population "
                    "fits; higher-level population or calibration mismatch may be "
                    "present."
                ),
                affected_outputs=tuple(unconstrained_affected),
            )
        )
    if boundary_affected:
        warnings.append(
            ReadoutMistWarning(
                code="population_fit_on_boundary",
                message=(
                    "At least one constrained GEF estimate lies on the simplex boundary."
                ),
                affected_outputs=tuple(boundary_affected),
            )
        )
    if fit_affected:
        warnings.append(
            ReadoutMistWarning(
                code="gef_population_model_mismatch",
                message=(
                    "At least one GEF estimator has poor rank, conditioning, or "
                    "residual objective; inspect the retained raw IQ and residuals."
                ),
                affected_outputs=tuple(fit_affected),
            )
        )
    return tuple(warnings)


def analyze_readout_mist(
    measurements: ReadoutMistMeasurements,
    *,
    covariance_rcond: float = 1e-12,
) -> ReadoutMistAnalysis:
    """Fit actual and optional blank-reference readout-MIST trajectories."""
    covariance_rcond = _validate_covariance_rcond(
        covariance_rcond,
        name="covariance_rcond",
    )
    actual = _fit_condition(
        measurements,
        reference=False,
        covariance_rcond=covariance_rcond,
    )
    has_reference = all(
        measurements.protocols[name].reference is not None for name in PROTOCOL_ORDER
    )
    reference = (
        _fit_condition(
            measurements,
            reference=True,
            covariance_rcond=covariance_rcond,
        )
        if has_reference
        else None
    )
    difference = (
        {
            name: actual.transition_probabilities[name].value
            - reference.transition_probabilities[name].value
            for name in TRANSITION_ORDER
        }
        if reference is not None and reference.success and actual.success
        else None
    )
    warnings = []
    for label, fit in (("actual", actual), ("reference", reference)):
        if fit is None:
            continue
        if not fit.success:
            warnings.append(
                ReadoutMistWarning(
                    code="fit_failed",
                    message=f"The {label} transition-matrix fit failed: {fit.message}",
                )
            )
            continue
        if fit.fit_quality == ReadoutMistFitQuality.STATISTICAL_MISMATCH:
            warnings.append(
                ReadoutMistWarning(
                    code="non_markovian_population_dynamics",
                    message=(
                        f"A single stochastic matrix does not explain the {label} "
                        "population trajectories within shot-noise covariance."
                    ),
                )
            )
        unresolved = tuple(
            name
            for name, estimate in fit.transition_probabilities.items()
            if estimate.status
            in {
                ReadoutMistProbabilityStatus.CONSISTENT_WITH_ZERO,
                ReadoutMistProbabilityStatus.PARTIALLY_UNRESOLVED,
            }
        )
        if unresolved:
            warnings.append(
                ReadoutMistWarning(
                    code="transition_probability_unresolved",
                    message=(
                        f"Some {label} transition probabilities are not resolved from "
                        "zero."
                    ),
                    affected_outputs=unresolved,
                )
            )
    warnings.extend(_population_model_warnings(measurements))
    return ReadoutMistAnalysis(
        actual=actual,
        reference=reference,
        reference_difference=difference,
        warnings=tuple(warnings),
        metadata={
            "model": "discrete_time_column_stochastic_transition_matrix",
            "matrix_convention": "T[final_state, initial_state]",
            "state_order": STATE_ORDER,
            "probability_unit": "1/readout_cycle",
            "initial_condition_treatment": "latent_nuisance_with_n0_observation",
            "uncertainty": "local_gls_conditional_on_gef_calibration",
            "reference_interpretation": "independent_blank_cycle_diagnostic",
            "gef_objective_warning_threshold": _GEF_OBJECTIVE_WARNING_THRESHOLD,
            "gef_condition_number_warning_threshold": (
                _GEF_CONDITION_NUMBER_WARNING_THRESHOLD
            ),
        },
    )


# Acquisition, presentation, and public workflow

DEFAULT_REPETITION_COUNTS = (0, 1, 2, 4, 8, 16, 32)
DEFAULT_GEF_MEASUREMENT_N_SHOTS = 4096
DEFAULT_GEF_CALIBRATION_N_SHOTS = 8192


def _validate_repetition_counts(values: Sequence[int]) -> tuple[int, ...]:
    """Validate the ordered MIST repetition grid."""
    resolved = tuple(values)
    if len(resolved) < 5:
        raise ValueError("repetition_counts must contain at least five points.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in resolved
    ):
        raise TypeError("repetition_counts must contain integers only.")
    normalized = tuple(int(value) for value in resolved)
    if normalized[0] != 0:
        raise ValueError("repetition_counts must start at zero.")
    if 1 not in normalized:
        raise ValueError("repetition_counts must contain one.")
    if any(right <= left for left, right in pairwise(normalized)):
        raise ValueError("repetition_counts must be unique and strictly increasing.")
    return normalized


def _positive_integer(value: int, name: str) -> int:
    """Validate a shot count without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 2:
        raise ValueError(f"{name} must be at least two.")
    return int(value)


def _optional_seed(value: int | None) -> int | None:
    """Validate an optional nonnegative acquisition seed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("acquisition_seed must be a nonnegative integer or None.")
    if value < 0:
        raise ValueError("acquisition_seed must be nonnegative.")
    return int(value)


def _finite_positive(value: float, name: str) -> float:
    """Validate a finite positive real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


def _extract_series(
    result: Result,
    schedules: ReadoutMistSchedules,
    qubit: str,
    protocol: str,
) -> GefPopulationSeries:
    """Convert named GEF fits into one ordered population trajectory."""
    data = cast(Mapping[str, Any], result.data)
    fits = cast(Mapping[str, Mapping[str, Any]], data["fits"])
    selected = [fits[name][qubit] for name in schedules.names_by_condition[protocol]]
    return GefPopulationSeries(
        population=np.stack([fit.population for fit in selected]),
        covariance=np.stack([fit.population_covariance for fit in selected]),
        standard_error=np.stack([fit.population_standard_error for fit in selected]),
        population_unconstrained=np.stack(
            [fit.population_unconstrained for fit in selected]
        ),
        fit_diagnostics=tuple(
            GefPopulationFitDiagnostic(
                objective=float(fit.objective),
                design_rank=int(fit.design_rank),
                design_condition_number=float(fit.design_condition_number),
                residual=np.asarray(fit.residual, dtype=np.float64),
                message=str(fit.message),
            )
            for fit in selected
        ),
    )


def _build_measurements(
    counts: tuple[int, ...],
    qubit: str,
    actual_result: Result,
    actual_schedules: ReadoutMistSchedules,
    reference_result: Result | None,
    reference_schedules: ReadoutMistSchedules | None,
) -> ReadoutMistMeasurements:
    """Build fixed-order measurement records from GEF result mappings."""
    protocols = {
        protocol: ReadoutMistConditionMeasurements(
            actual=_extract_series(actual_result, actual_schedules, qubit, protocol),
            reference=(
                None
                if reference_result is None or reference_schedules is None
                else _extract_series(
                    reference_result,
                    reference_schedules,
                    qubit,
                    protocol,
                )
            ),
        )
        for protocol in PROTOCOL_ORDER
    }
    return ReadoutMistMeasurements(
        repetition_counts=np.asarray(counts, dtype=np.int64),
        protocols=protocols,
    )


def _population_figure(
    measurements: ReadoutMistMeasurements,
    analysis: ReadoutMistAnalysis,
) -> go.Figure:
    """Build three protocol panels with actual/reference data and joint fits."""
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Initial g", "Initial e", "Initial f"),
        vertical_spacing=0.08,
    )
    colors = fitting.COLORS[:3]
    counts = measurements.repetition_counts
    for row, protocol in enumerate(PROTOCOL_ORDER, start=1):
        measured = measurements.protocols[protocol]
        for state_index, state in enumerate(STATE_ORDER):
            figure.add_trace(
                go.Scatter(
                    x=counts,
                    y=measured.actual.population[:, state_index],
                    error_y={
                        "type": "data",
                        "array": measured.actual.standard_error[:, state_index],
                        "visible": True,
                    },
                    mode="markers",
                    marker={"color": colors[state_index]},
                    name=f"actual P{state}",
                    legendgroup=f"actual-{state}",
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
            if analysis.actual.success:
                figure.add_trace(
                    go.Scatter(
                        x=counts,
                        y=analysis.actual.fitted_populations[protocol][:, state_index],
                        mode="lines",
                        line={"color": colors[state_index]},
                        name=f"fit P{state}",
                        legendgroup=f"fit-{state}",
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
            if measured.reference is not None:
                figure.add_trace(
                    go.Scatter(
                        x=counts,
                        y=measured.reference.population[:, state_index],
                        error_y={
                            "type": "data",
                            "array": measured.reference.standard_error[:, state_index],
                            "visible": True,
                        },
                        mode="markers",
                        marker={
                            "color": colors[state_index],
                            "symbol": "circle-open",
                        },
                        name=f"blank P{state}",
                        legendgroup=f"blank-{state}",
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
                if analysis.reference is not None and analysis.reference.success:
                    figure.add_trace(
                        go.Scatter(
                            x=counts,
                            y=analysis.reference.fitted_populations[protocol][
                                :, state_index
                            ],
                            mode="lines",
                            line={"color": colors[state_index], "dash": "dot"},
                            name=f"blank fit P{state}",
                            legendgroup=f"blank-fit-{state}",
                            showlegend=row == 1,
                        ),
                        row=row,
                        col=1,
                    )
        figure.update_yaxes(
            title_text="GEF population", range=[-0.03, 1.03], row=row, col=1
        )
    figure.update_xaxes(title_text="Readout repetitions (n)", row=3, col=1)
    figure.update_layout(
        title="Readout-induced state transitions",
        template="plotly_white",
        height=850,
    )
    return figure


def _matrix_figure(analysis: ReadoutMistAnalysis) -> go.Figure:
    """Build a final-by-initial transition-matrix heatmap."""
    matrix = analysis.actual.transition_matrix
    return go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=STATE_ORDER,
            y=STATE_ORDER,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=np.asarray(
                [[f"{value:.5f}" for value in row] for row in matrix],
                dtype=str,
            ),
            texttemplate="%{text}",
            colorbar={"title": "Probability"},
        ),
        layout={
            "title": "Transition matrix T[final, initial]",
            "xaxis": {"title": "Initial state"},
            "yaxis": {"title": "Final state", "autorange": "reversed"},
            "template": "plotly_white",
        },
    )


def _print_summary(analysis: ReadoutMistAnalysis) -> None:
    """Print transition probabilities and compact fit diagnostics."""
    print("Readout MIST probabilities [1/readout_cycle]")
    if not analysis.actual.success:
        print(f"  fit failed: {analysis.actual.message}")
        if analysis.warnings:
            print(
                "Warnings: " + ", ".join(warning.code for warning in analysis.warnings)
            )
        return
    for name in TRANSITION_ORDER:
        estimate = analysis.actual.transition_probabilities[name]
        uncertainty = (
            "unresolved"
            if estimate.standard_error is None
            else f"{estimate.value:.6g} ± {estimate.standard_error:.2g}"
        )
        print(f"  {name}: {uncertainty} ({estimate.status.value})")
    goodness = analysis.actual.goodness_of_fit
    print(
        "Fit quality: "
        f"{goodness.quality.value}; reduced chi-squared="
        f"{goodness.reduced_chi_squared:.3g}; "
        f"max |whitened residual|={goodness.maximum_absolute_whitened_residual:.3g}"
    )
    if analysis.warnings:
        print("Warnings: " + ", ".join(warning.code for warning in analysis.warnings))


def characterize_readout_mist(
    exp: Experiment,
    qubit: str,
    *,
    repetition_counts: Sequence[int] | None = None,
    readout_cycle: PulseSchedule | None = None,
    cycle_gap_ns: float = 0.0,
    measure_reference: bool = True,
    gef_measurement_n_shots: int = DEFAULT_GEF_MEASUREMENT_N_SHOTS,
    gef_calibration_n_shots: int = DEFAULT_GEF_CALIBRATION_N_SHOTS,
    shot_interval: float | None = None,
    gef_covariance_rcond: float = 1e-12,
    acquisition_order: Literal["interleaved", "randomized", "grouped"] = "interleaved",
    acquisition_seed: int | None = 0,
    enable_tqdm: bool = True,
    plot: bool = True,
) -> Result:
    """
    Characterize effective GEF transitions caused by one complete readout cycle.

    Parameters
    ----------
    exp : Experiment
        Experiment used for calibrated pulse construction and acquisition.
    qubit : str
        Qubit whose GEF transition matrix is measured.
    repetition_counts : Sequence[int] | None, optional
        Strictly increasing counts beginning with zero and containing one.
        At least five points are required. Defaults to `(0, 1, 2, 4, 8, 16, 32)`.
    readout_cycle : PulseSchedule | None, optional
        Complete candidate cycle containing drive, optional active depletion,
        and its trailing gap. When omitted, use the configured readout pulse
        followed by `cycle_gap_ns`.
    cycle_gap_ns : float, optional
        Trailing gap for the automatically built configured cycle, in ns.
        Must remain zero when `readout_cycle` is supplied.
    measure_reference : bool, optional
        Whether to independently fit a duration-matched all-blank cycle.
    gef_measurement_n_shots : int, optional
        Shots for each of the three diagnostic GEF permutations.
    gef_calibration_n_shots : int, optional
        Shots for each of the six GEF calibration configurations.
    shot_interval : float | None, optional
        Shot interval in ns, or the configured measurement default.
    gef_covariance_rcond : float, optional
        Relative covariance eigenvalue cutoff in `[0, 1)`.
    acquisition_order : {"interleaved", "randomized", "grouped"}, optional
        Logical G/E/F and repetition-point submission order.
    acquisition_seed : int | None, optional
        Seed used only for randomized acquisition order.
    enable_tqdm : bool, optional
        Whether acquisition services show progress bars.
    plot : bool, optional
        Whether to construct and show figures and print the summary. If false,
        the returned `figures` mapping is empty.

    Returns
    -------
    Result
        Measurements, full six-transition fits, raw GEF data, timing, and figures.

    Notes
    -----
    This function performs hardware acquisition. Reported probabilities use
    `T[final_state, initial_state]` and units of `1/readout_cycle`. They are
    effective cycle-boundary probabilities rather than microscopic rates.
    Uncertainty is local GLS covariance conditional on the shared GEF calibration.
    The blank reference zeros the entire candidate cycle while preserving its
    timing; its fitted difference is diagnostic and is not subtracted as an
    estimate of readout-only transitions. The final GEF discriminator uses the
    calibrated configured readout, independently of `readout_cycle`. Consequently,
    diagnostic-readout backaction and population outside the closed GEF subspace
    remain model limitations and are surfaced through retained raw data and warnings.
    """
    for name, value in (
        ("measure_reference", measure_reference),
        ("enable_tqdm", enable_tqdm),
        ("plot", plot),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    counts = _validate_repetition_counts(
        DEFAULT_REPETITION_COUNTS if repetition_counts is None else repetition_counts
    )
    measurement_shots = _positive_integer(
        gef_measurement_n_shots,
        "gef_measurement_n_shots",
    )
    calibration_shots = _positive_integer(
        gef_calibration_n_shots,
        "gef_calibration_n_shots",
    )
    covariance_rcond = _validate_covariance_rcond(
        gef_covariance_rcond,
        name="gef_covariance_rcond",
    )
    seed = _optional_seed(acquisition_seed)
    configured_interval = resolve_measurement_defaults(
        exp.ctx.experiment_system.measurement_defaults
    ).execution.shot_interval_ns
    interval = _finite_positive(
        configured_interval if shot_interval is None else shot_interval,
        "shot_interval",
    )
    target = exp.ctx.resolve_qubit_label(qubit)
    if readout_cycle is not None and cycle_gap_ns != 0.0:
        raise ValueError(
            "cycle_gap_ns must be zero when readout_cycle supplies the complete cycle."
        )
    cycle = (
        build_default_readout_cycle(exp, target, cycle_gap_ns=cycle_gap_ns)
        if readout_cycle is None
        else validate_readout_cycle(readout_cycle)
    )
    actual_schedules = build_mist_schedules(
        exp,
        target,
        cycle,
        counts,
        prefix="actual",
        acquisition_order=cast(AcquisitionOrder, acquisition_order),
        acquisition_seed=seed,
    )
    actual_result = measure_gef_populations(
        exp,
        targets=[target],
        sequences=actual_schedules.submitted,
        n_shots=measurement_shots,
        calibration_n_shots=calibration_shots,
        shot_interval=interval,
        covariance_rcond=covariance_rcond,
        n_bootstrap=0,
        enable_tqdm=enable_tqdm,
        progress_label="readout MIST actual",
    )
    actual_data = cast(Mapping[str, Any], actual_result.data)
    calibration = actual_data["calibration"]
    reference_schedules = None
    reference_result = None
    reference_cycle = None
    if measure_reference:
        reference_cycle = build_reference_cycle(cycle)
        reference_schedules = build_mist_schedules(
            exp,
            target,
            reference_cycle,
            counts,
            prefix="reference",
            acquisition_order=cast(AcquisitionOrder, acquisition_order),
            acquisition_seed=seed,
        )
        reference_result = measure_gef_populations(
            exp,
            targets=[target],
            sequences=reference_schedules.submitted,
            calibration=calibration,
            n_shots=measurement_shots,
            shot_interval=interval,
            covariance_rcond=covariance_rcond,
            n_bootstrap=0,
            enable_tqdm=enable_tqdm,
            progress_label="readout MIST blank reference",
        )
    measurements = _build_measurements(
        counts,
        target,
        actual_result,
        actual_schedules,
        reference_result,
        reference_schedules,
    )
    analysis = analyze_readout_mist(
        measurements,
        covariance_rcond=covariance_rcond,
    )
    figures = (
        {"populations": _population_figure(measurements, analysis)} if plot else {}
    )
    if plot and analysis.actual.success:
        figures["transition_matrix"] = _matrix_figure(analysis)
    if plot:
        _print_summary(analysis)
        for figure in figures.values():
            figure.show()
    reference_data = (
        None
        if reference_result is None
        else cast(Mapping[str, Any], reference_result.data)
    )
    measurement_options = {
        "repetition_counts": counts,
        "gef_measurement_n_shots": measurement_shots,
        "gef_calibration_n_shots": calibration_shots,
        "shot_interval_ns": interval,
        "measure_reference": measure_reference,
        "gef_covariance_rcond": covariance_rcond,
        "acquisition_order": acquisition_order,
        "acquisition_seed": seed,
        "enable_tqdm": enable_tqdm,
        "plot": plot,
    }
    return Result(
        data={
            "measurements": measurements,
            "analysis": analysis,
            "raw_data": {
                "gef_calibration": calibration,
                "actual_raw_iq": actual_data["raw_iq"],
                "actual_moment_summaries": actual_data["moment_summaries"],
                "actual_population_fits": actual_data["fits"],
                "reference_raw_iq": (
                    None if reference_data is None else reference_data["raw_iq"]
                ),
                "reference_moment_summaries": (
                    None
                    if reference_data is None
                    else reference_data["moment_summaries"]
                ),
                "reference_population_fits": (
                    None if reference_data is None else reference_data["fits"]
                ),
            },
            "measurement_options": measurement_options,
            "analysis_options": {
                "model": "full_six_transition_discrete_time_matrix",
                "reduced_models": False,
                "profile_upper_bounds": False,
                "calibration_bootstrap": False,
                "uncertainty": "local_gls_conditional_on_gef_calibration",
            },
            "readout_cycle_timing": {
                "duration_ns": float(cycle.duration),
                "labels": tuple(cycle.labels),
                "frequencies": dict(cycle.get_frequencies()),
                "source": "configured" if readout_cycle is None else "override",
                "cycle_gap_ns": float(cycle_gap_ns) if readout_cycle is None else None,
                "trailing_gap_in_final_cycle": True,
                "reference": "complete_duration_matched_blank"
                if measure_reference
                else None,
                "reference_duration_ns": (
                    None if reference_cycle is None else float(reference_cycle.duration)
                ),
            },
            "schedules": {
                "actual": actual_schedules,
                "reference": reference_schedules,
            },
        },
        figure=figures.get("populations"),
        figures=figures,
    )


__all__ = ["characterize_readout_mist"]
