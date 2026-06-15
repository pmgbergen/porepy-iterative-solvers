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


"""

import numpy as np
import porepy as pp
import pytest
from porepy.applications.test_utils.models import add_mixin
from porepy.examples.flow_benchmark_2d_case_4 import solid_constants

import pp_solvers


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
# regression testing. Hopefully the reference values are stable.
expected_linear_iterations = {
    FluidModel: [3, 3],
    MechanicsModel: [5, 6],
    PoromechanicsModel: [8, 12, 10, 12],
    ThermoporomechanicsModel: [10, 15, 14],
    ThermoporomechanicsTpsaModel: [13, 17, 19, 19, 18, 19, 19, 19],
}


def model_options():
    return {
        "material_constants": {
            "solid": solid_constants,
            "fluid": pp.FluidComponent(**{"compressibility": 1e-7}),
        },
        "reference_variable_values": pp.ReferenceVariableValues(**{"pressure": 1}),
        # EK note to self: Including both fractures in the model led to severe
        # convergence problem for the non-linear solver, even when a direct solver was
        # used for the linearized system. Use a single fracture for now.
        "fracture_indices": [1],
        "cell_size": 0.1,
        "u_north": -0.001,  # Used for mechanics problems
    }


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
    opts = model_options()

    # EK note to self: I could not go much further down here without running into
    # convergence problems with the nonlinear solver. I suspect this is due to the
    # fracture states changing, possibly because the grid is rather coarse. Leave this
    # for now.
    solver_opts = {"nl_convergence_tol_res": 1e-8, "nl_convergence_tol": 1e-8}

    direct_model = model_class(opts)
    pp.run_time_dependent_model(direct_model, solver_opts)

    iterative_opts = model_options()
    iterative_opts["linear_solver"] = {
        "options": {
            # The iterations will not be printed during pytest (which surpresses
            # output), but will be active during debugging, if the test is run as a
            # python script.
            "gmres": {"ksp_monitor": None},
            # This was the old default, preserving it to correspond to the hard-coded
            # expected iteration count.
            "mechanics_amg": {"pc_hypre_boomeramg_strong_threshold": 0.7},
        },
    }
    iterative_class = add_mixin(pp_solvers.IterativeSolverMixin, model_class)

    iterative_model = iterative_class(iterative_opts)
    pp.run_time_dependent_model(iterative_model, solver_opts)

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
    linear_iterations = iterative_model.linear_solver_statistics.num_krylov_iters
    expected_iterations = expected_linear_iterations[model_class]
    # The number of non-linear iterations taken may change, e.g., due to updates in
    # PorePy's convergence criteria. To avoid having to update the expected number of
    # iterations, we compare the number of Krylov iterations only for those non-linear
    # iterations that were in common between the historic and current cases. This is in
    # a sense something of a weakening of the test, but it reduces the risk of the known
    # values being updated mindlessly.
    min_length = min(len(linear_iterations), len(expected_iterations))

    np.testing.assert_equal(
        linear_iterations[:min_length],
        expected_iterations[:min_length],
        err_msg="Number of linear iterations does not match expected value.",
    )


def test_linear_solver_failure():
    """Tests a case when a linear solver fails (due to iterations limit), but nonlinear
    iterations continue until they reach a limit."""
    iterative_opts = model_options()
    iterative_opts["linear_solver"] = {
        "options": {
            "gmres": {
                "ksp_monitor": None,
                # Enforcing a single gmres iteration to ensure non-convergence.
                "ksp_max_it": 1,
            },
        },
    }
    iterative_class = add_mixin(pp_solvers.IterativeSolverMixin, FluidModel)
    iterative_model = iterative_class(iterative_opts)
    max_nonlinear_iterations = 7
    with pytest.warns(UserWarning, match="Failed to solve the nonlinear problem"):
        pp.ModelRunner(
            iterative_model, {"nl_max_iterations": max_nonlinear_iterations}
        ).run()
    linear_iterations = iterative_model.linear_solver_statistics.num_krylov_iters
    assert len(linear_iterations) == max_nonlinear_iterations, (
        f"We did {linear_iterations} Newton iterations and did not converge."
    )
    # No idea why PETSc reports 2 and not 1, but it should not report anything else.
    assert np.all(np.array(linear_iterations) == 2)
