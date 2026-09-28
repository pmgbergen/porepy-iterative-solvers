"""Tests for the PorePy integration of the PETSc iterative solvers."""

import numpy as np
import porepy as pp
import pytest
from porepy.examples.flow_benchmark_2d_case_4 import solid_constants

from pp_solvers.porepy_integration import (
    IterativeLinearSolver,
    IterativeLinearSolverNotConverged,
    classify_ksp_outcome,
)

KSP_CONVERGED_RTOL = 2
KSP_CONVERGED_ITERATING = 0
KSP_DIVERGED_ITS = -3
KSP_DIVERGED_DTOL = -4
KSP_DIVERGED_BREAKDOWN = -5
KSP_DIVERGED_NANORINF = -9
KSP_DIVERGED_PC_FAILED = -11

FINITE = np.array([1.0, -2.0])


@pytest.mark.parametrize(
    "reason, solution, expected",
    [
        (KSP_CONVERGED_RTOL, FINITE, "success"),
        # Stopped short of the tolerance with a finite solution: an inexact solution.
        (KSP_DIVERGED_ITS, FINITE, "not_converged"),
        (KSP_DIVERGED_BREAKDOWN, FINITE, "not_converged"),
        # Reasons after which the solution is not to be used, even if finite.
        (KSP_DIVERGED_DTOL, FINITE, "failure"),
        (KSP_DIVERGED_NANORINF, FINITE, "failure"),
        (KSP_DIVERGED_PC_FAILED, FINITE, "failure"),
        (KSP_CONVERGED_ITERATING, FINITE, "failure"),
        # A non-finite solution is a failure whatever PETSc reports.
        (KSP_DIVERGED_ITS, np.array([1.0, np.inf]), "failure"),
        (KSP_CONVERGED_RTOL, np.array([np.nan, 1.0]), "failure"),
    ],
)
def test_classify_ksp_outcome(reason, solution, expected):
    """A failure here means that a PETSc solve is reported to PorePy under the wrong
    status: an unusable solution passed on as usable, or an inexact but usable one
    discarded as failed. PorePy's Newton solver acts on this distinction.

    """
    assert classify_ksp_outcome(reason, solution) == expected


class PoromechanicsModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
):
    pass


def test_newton_continues_past_iteration_cap():
    """With GMRES capped far below what the system needs, every solve stops at the
    cap. The solutions are inexact, but usable, so Newton must apply them and the
    simulation must still succeed, only with more nonlinear iterations.

    A failure here most likely means that reaching the iteration cap is reported as
    a failure again, which makes PorePy's Newton solver discard the update.

    """
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
    linear_solver = IterativeLinearSolver(
        solver_options={"gmres": {"ksp_max_it": 3, "ksp_atol": 1e-30}}
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
