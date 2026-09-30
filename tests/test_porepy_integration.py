"""Tests for the PorePy integration of the PETSc iterative solvers."""

import numpy as np
import porepy as pp
import pytest
from porepy.examples.flow_benchmark_2d_case_4 import solid_constants

from pp_solvers.porepy_integration import (
    IterativeLinearSolver,
    IterativeLinearSolverFailure,
    IterativeLinearSolverNotConverged,
    classify_ksp_outcome,
    relative_residual,
)

KSP_CONVERGED_RTOL = 2
KSP_CONVERGED_ITERATING = 0
KSP_DIVERGED_ITS = -3
KSP_DIVERGED_DTOL = -4
KSP_DIVERGED_BREAKDOWN = -5
KSP_DIVERGED_NANORINF = -9
KSP_DIVERGED_PC_FAILED = -11

FINITE = np.array([1.0, -2.0])
REDUCED = 1e-6
"""A relative residual well below the default tolerance for not converged solves."""
UNREDUCED = 0.5
"""A relative residual well above it."""


@pytest.mark.parametrize(
    "reason, solution, residual_ratio, expected",
    [
        (KSP_CONVERGED_RTOL, FINITE, REDUCED, "success"),
        # Stopped short of the tolerance with a finite solution, after reducing the
        # residual: an inexact solution.
        (KSP_DIVERGED_ITS, FINITE, REDUCED, "not_converged"),
        (KSP_DIVERGED_BREAKDOWN, FINITE, REDUCED, "not_converged"),
        # Stopped short without reducing the residual: no usable step.
        (KSP_DIVERGED_ITS, FINITE, UNREDUCED, "failure"),
        (KSP_DIVERGED_BREAKDOWN, FINITE, UNREDUCED, "failure"),
        # PETSc's own convergence test takes precedence over the relative residual,
        # which it may have met through the absolute tolerance instead.
        (KSP_CONVERGED_RTOL, FINITE, UNREDUCED, "success"),
        # Reasons after which the solution is not to be used, even if finite.
        (KSP_DIVERGED_DTOL, FINITE, REDUCED, "failure"),
        (KSP_DIVERGED_NANORINF, FINITE, REDUCED, "failure"),
        (KSP_DIVERGED_PC_FAILED, FINITE, REDUCED, "failure"),
        (KSP_CONVERGED_ITERATING, FINITE, REDUCED, "failure"),
        # A non-finite solution is a failure whatever PETSc reports.
        (KSP_DIVERGED_ITS, np.array([1.0, np.inf]), REDUCED, "failure"),
        (KSP_CONVERGED_RTOL, np.array([np.nan, 1.0]), REDUCED, "failure"),
    ],
)
def test_classify_ksp_outcome(reason, solution, residual_ratio, expected):
    """A failure here means that a PETSc solve is reported to PorePy under the wrong
    status: an unusable solution passed on as usable, or an inexact but usable one
    discarded as failed. PorePy's Newton solver acts on this distinction.

    """
    assert classify_ksp_outcome(reason, solution, residual_ratio) == expected


@pytest.mark.parametrize(
    "not_converged_rtol, expected", [(1e-2, "not_converged"), (1e-4, "failure")]
)
def test_classify_ksp_outcome_honours_not_converged_rtol(not_converged_rtol, expected):
    """A capped solve with relative residual 1e-3 is usable or not depending on the
    tolerance passed; a failure means the tolerance is ignored or compared the wrong
    way round.

    """
    outcome = classify_ksp_outcome(
        KSP_DIVERGED_ITS, FINITE, 1e-3, not_converged_rtol=not_converged_rtol
    )
    assert outcome == expected


@pytest.mark.parametrize(
    "history, expected",
    [
        (np.array([2.0, 1.0, 1e-3]), 5e-4),
        # The zero initial guess solved the system exactly.
        (np.array([0.0]), 0.0),
        # No residual recorded: no evidence of progress.
        (np.array([]), np.inf),
    ],
)
def test_relative_residual(history, expected):
    assert relative_residual(history) == pytest.approx(expected)


class PoromechanicsModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
):
    pass


CAPPED_GMRES = {"gmres": {"ksp_max_it": 3, "ksp_atol": 1e-30}}
"""GMRES capped far below the iterations the poromechanics model's systems need."""


@pytest.fixture
def poromechanics_model() -> PoromechanicsModel:
    model = PoromechanicsModel(
        {
            "material_constants": {
                "solid": solid_constants,
                "fluid": pp.FluidComponent(compressibility=1e-7),
            },
            "reference_variable_values": pp.ReferenceVariableValues(pressure=1),
            "fracture_indices": [1],
            "cell_size": 0.1,
            "u_north": -0.001,
        }
    )
    model.prepare_simulation()
    return model


def test_capped_solve_short_of_not_converged_rtol_fails(poromechanics_model):
    """A capped solve whose residual has not fallen below `not_converged_rtol` must
    be reported as failed. A failure here most likely means that the solver's
    tolerance does not reach the classification of its solves.

    """
    linear_solver = IterativeLinearSolver(
        solver_options=CAPPED_GMRES, not_converged_rtol=1e-30
    )
    linear_solver.initialize_with_model(poromechanics_model)
    linear_system = poromechanics_model.equation_system.assemble()
    _, status = linear_solver.solve_linear_system(linear_system)
    assert isinstance(status, IterativeLinearSolverFailure)
    assert status.petsc_converged_reason == KSP_DIVERGED_ITS


def test_newton_continues_past_iteration_cap(poromechanics_model):
    """With GMRES capped far below what the system needs, every solve stops at the
    cap. The solutions are inexact, but usable, so Newton must apply them and the
    simulation must still succeed, only with more nonlinear iterations.

    A failure here most likely means that reaching the iteration cap is reported as
    a failure again, which makes PorePy's Newton solver discard the update.

    """
    model = poromechanics_model
    # Some capped solves barely reduce the residual. Accepting them all keeps this test
    # about Newton's handling of the status, apart from the tolerance tested above.
    linear_solver = IterativeLinearSolver(
        solver_options=CAPPED_GMRES, not_converged_rtol=1.0
    )
    statuses = []
    solve = linear_solver.solve_linear_system

    def recording_solve(linear_system):
        solution, status = solve(linear_system)
        statuses.append(status)
        return solution, status

    linear_solver.solve_linear_system = recording_solve
    status = pp.ModelRunner(
        model=model,
        nonlinear_solver=pp.solvers.NewtonSolver(
            params={"nl_max_iterations": 30},
            is_nonlinear_problem=model._is_nonlinear_problem(),
            linear_solver=linear_solver,
        ),
        params={"prepare_simulation": False},
    ).run()

    assert status.is_success()
    assert len(statuses) > 0
    assert all(isinstance(s, IterativeLinearSolverNotConverged) for s in statuses)
