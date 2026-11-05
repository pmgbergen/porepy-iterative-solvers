"""This file contains tests for the performance of the default preconditioners applied
to simple systems. For a given PorePy model, the tests consist of the following steps:
    1. Solve the model with a direct solver to establish a reference solution.
    2. Solve the model with an iterative solver using the default preconditioner.
    3. Compare the following metrics:
        i) The solutions to the nonlinear system obtained with the direct and
            iterative solver. These should be very close.
        ii) The number of nonlinear iterations required to reach convergence for the
            two solvers. These should be the same.
        iii) The number of linear iterations required to reach convergence for the
            iterative solver. This should be the same as a historical value.
        iv) The PETSc reason for convergence of the iterative solver. This should be
            the same as a historical value.
        Tests number iii) and iv) gives this a regression test character.

For the regression values to be reproducible across platforms, the contact state of
every fracture cell must be well defined at each Newton iterate. If a cell sits at
the switch of the complementarity function (e.g. zero contact traction and zero
normal jump), floating-point round-off (which differs between CPUs and BLAS kernels)
decides the active branch, and with it the Newton path and the Krylov counts.

"""

import numpy as np
import porepy as pp
import pytest
from porepy.examples.flow_benchmark_2d_case_4 import solid_constants
from porepy.models.protocol import PorePyModel
from porepy.time_stepper.time_step_status import (
    TimeStepperStatusFailure,
    TimeStepperStatusSuccess,
)

from pp_solvers.porepy_integration import (
    IterativeLinearSolver,
    IterativeLinearSolverFailure,
    IterativeLinearSolverSuccess,
)


class FluidModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    pp.SinglePhaseFlow,
):
    pass


class MechanicsModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.MomentumBalance,
):
    pass


class PoromechanicsModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
):
    pass


class ThermoporomechanicsModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Thermoporomechanics,
):
    pass


class ThermoporomechanicsTpsaModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.poromechanics.TpsaPoromechanicsMixin,
    pp.Thermoporomechanics,
):
    pass


# Hard-coded expected number of linear iterations for each model. These are used for
# regression testing. With the setup in model_options, these were found to be
# invariant to perturbations of the Newton increments of relative size up to 1e-10
# (a proxy for platform-dependent round-off).
expected_linear_iterations = {
    FluidModel: [3, 3],
    MechanicsModel: [10, 11],
    PoromechanicsModel: [16, 20],
    ThermoporomechanicsModel: [34, 41],
    ThermoporomechanicsTpsaModel: [26, 35],
}


def fetch_linear_iterations_from_statistics(model: PorePyModel) -> list[int]:
    """Collect Krylov iteration counts from all recorded nonlinear solves."""
    linear_iterations: list[int] = []
    for x in model.nonlinear_solver_statistics.simulation_status_history:
        assert isinstance(x, (TimeStepperStatusFailure, TimeStepperStatusSuccess))
        for attempt in x.attempts:
            for y in attempt.nonlinear_solve_status.linear_solver_statuses:
                assert isinstance(
                    y, (IterativeLinearSolverSuccess, IterativeLinearSolverFailure)
                )
                linear_iterations.append(y.num_krylov_iters)
    return linear_iterations


def model_options(pressure_drive: bool = False):
    options = {
        "material_constants": {
            "solid": solid_constants,
            "fluid": pp.FluidComponent(**{"compressibility": 1e-7}),
        },
        # EK note to self: Including both fractures in the model led to severe
        # convergence problem for the non-linear solver, even when a direct solver was
        # used for the linearized system. Use a single fracture for now.
        "fracture_indices": [1],
        # The mesh size must be passed through meshing_arguments, a top-level
        # "cell_size" is silently ignored (and the default 0.5 gives a 4-cell grid).
        "meshing_arguments": {"cell_size": 0.1},
        "u_north": -0.001,  # Used for mechanics problems
    }
    if pressure_drive:
        # A boundary pressure different from the (zero) initial pressure drives the
        # flow. This must not be used in the poromechanical models: There, the fluid
        # pressure pushing the fracture open balances the mechanical compression, so
        # that the fracture ends up exactly at the switch between open and closed
        # (zero contact traction and zero opening). See the module docstring.
        options["reference_variable_values"] = pp.ReferenceVariableValues(
            **{"pressure": 1}
        )
    return options


@pytest.mark.parametrize(
    "model_class",
    [
        FluidModel,
        MechanicsModel,
        PoromechanicsModel,
        ThermoporomechanicsModel,
        ThermoporomechanicsTpsaModel,
    ],
)
def test_model(model_class):
    # EK note to self: I could not go much further down here without running into
    # convergence problems with the nonlinear solver. I suspect this is due to the
    # fracture states changing, possibly because the grid is rather coarse. Leave this
    # for now.
    solver_opts = {"nl_convergence_res_atol": 1e-8, "nl_convergence_inc_atol": 1e-8}
    # Only the pure flow model needs a pressure drive to get a nontrivial problem.
    pressure_drive = model_class is FluidModel

    direct_model = model_class(model_options(pressure_drive))
    try:
        status = pp.ModelRunner(direct_model, solver_opts).run()
        assert status.is_success()
        direct_model_failed = False
    except RuntimeError as e:
        # Some of the models are known to fail. This test ignores it.
        status = e.args[0]
        # But if it fails, we want to know that it is a normal simulation failure and
        # not just some random exception.
        assert status.is_failure()
        direct_model_failed = True

    linear_solver = IterativeLinearSolver(
        solver_options={
            # The iterations will not be printed during pytest (which suppresses
            # output), but will be active during debugging, if the test is run as a
            # python script.
            "gmres": {"ksp_monitor": None},
            # This was the old default, preserving it to correspond to the hard-coded
            # expected iteration count.
            "mechanics_amg": {"pc_hypre_boomeramg_strong_threshold": 0.7},
        },
    )

    iterative_model = model_class(model_options(pressure_drive))
    iterative_model.prepare_simulation()
    try:
        status = pp.ModelRunner(
            model=iterative_model,
            nonlinear_solver=pp.solvers.NewtonSolver(
                params=solver_opts,
                is_nonlinear_problem=iterative_model._is_nonlinear_problem(),
                linear_solver=linear_solver,
            ),
            params={"prepare_simulation": False},
        ).run()
        assert status.is_success()
        iterative_model_failed = False
    except RuntimeError as e:
        # Some of the models are known to fail. This test ignores it.
        status = e.args[0]
        # But if it fails, we want to know that it is a normal simulation failure and
        # not just some random exception.
        assert status.is_failure()
        iterative_model_failed = True

    assert iterative_model_failed == direct_model_failed, (
        "Both models must either succeed or fail regadless of the linear solver."
    )

    # Check that the nonlinear solutions are the same for both models. The tolerance
    # used is not very strict, but is somewhat consistent with the nonlinear tolerances
    # set above.
    direct_solution = direct_model.equation_system.get_variable_values(
        time_step_index=0
    )
    iterative_solution = iterative_model.equation_system.get_variable_values(
        time_step_index=0
    )
    assert np.allclose(direct_solution, iterative_solution, rtol=1e-6, atol=1e-6), (
        "Solutions do not match."
    )

    # Fetch the actual and expected number of iterations.
    linear_iterations = fetch_linear_iterations_from_statistics(iterative_model)
    expected_iterations = expected_linear_iterations[model_class]

    # Experience shows that the number of linear iterations can vary a bit, so we allow
    # for a small tolerance in the number of iterations to avoid false positives.
    iteration_count_tolerance = 2
    assert len(linear_iterations) == len(expected_iterations), (
        "Number of linear solves does not match expected value. Expected: "
        f"{expected_iterations}, actual: {linear_iterations}."
    )
    assert all(
        abs(actual - expected) <= iteration_count_tolerance
        for actual, expected in zip(linear_iterations, expected_iterations)
    ), (
        "Number of linear iterations differs from the expected value by more than "
        f"{iteration_count_tolerance}. Expected: {expected_iterations}, actual: "
        f"{linear_iterations}."
    )


def test_linear_solver_failure():
    """Tests a case when a linear solver fails (due to iterations limit), but nonlinear
    iterations continue until they reach a limit."""
    iterative_model = FluidModel(model_options(pressure_drive=True))
    # Need prepare_simulation to tell if the model is nonlinear.
    iterative_model.prepare_simulation()
    linear_solver = IterativeLinearSolver(
        solver_options={
            "gmres": {
                "ksp_monitor": None,
                # Enforcing a single gmres iteration to ensure non-convergence.
                "ksp_max_it": 1,
            },
        }
    )
    max_nonlinear_iterations = 7
    with pytest.raises(RuntimeError):
        pp.ModelRunner(
            iterative_model,
            {"prepare_simulation": False},
            nonlinear_solver=pp.solvers.NewtonSolver(
                params={"nl_max_iterations": max_nonlinear_iterations},
                is_nonlinear_problem=iterative_model._is_nonlinear_problem(),
                linear_solver=linear_solver,
            ),
        ).run()

    linear_iterations = fetch_linear_iterations_from_statistics(iterative_model)
    assert len(linear_iterations) == max_nonlinear_iterations, (
        f"We did {len(linear_iterations)} Newton iterations and did not converge."
    )
    # No idea why PETSc reports 2 and not 1, but it should not report anything else.
    assert np.all(np.array(linear_iterations) == 2)
