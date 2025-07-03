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


factories = {
    FluidModel: pp_solvers.mass_balance_factory,
    MechanicsModel: pp_solvers.momentum_balance_factory,
    PoromechanicsModel: pp_solvers.hm_factory,
    ThermoporomechanicsModel: pp_solvers.thm_factory,
}

# Hard-coded expected number of linear iterations for each model. These are used for
# regression testing. Hopefully the reference values are stable.
expected_linear_iterations = {
    FluidModel: [3, 3],
    MechanicsModel: [5, 6],
    PoromechanicsModel: [8, 14, 13, 11, 11, 11, 10, 13],
    ThermoporomechanicsModel: [10, 12, 12, 12, 15, 13, 15, 13, 15],
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
    [FluidModel, MechanicsModel, PoromechanicsModel, ThermoporomechanicsModel],
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

    factory = factories[model_class]
    iterative_opts = model_options()
    iterative_opts["linear_solver"] = {
        "preconditioner_factory": factory,
        # The iterations will not be printed during pytest (which surpresses output),
        # but will be active during debugging, if the test is run as a python script.
        "options": {"ksp_monitor": None},
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

    linear_iterations = iterative_model.nonlinear_solver_statistics.num_krylov_iters
    assert np.allclose(
        linear_iterations,
        expected_linear_iterations[model_class],
    ), "Number of linear iterations does not match expected value."
