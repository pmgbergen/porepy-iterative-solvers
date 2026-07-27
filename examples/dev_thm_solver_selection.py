"""This example illustrates usage of the solver selector for the THM model, the same
models as in dev_thm.py.

"""

import logging

from examples.dev_thm import FullModel, linear_solver_options, model_params_2d
import porepy as pp
import pp_solvers
from pp_solvers.solver_selection import (
    SolverSpace,
    SolverSelector,
    NumericalChoices,
    CategoricalChoices,
    assemble_default_performance_predictor,
)

# The solver space scheme mirrors the options format that we use without solver
# selection (e.g. see `thm_factory` function docstring):
# {
#     "solver_key": {
#         "petsc_key": "petsc_value"
#     },
# }
# But now we can use CategoricalChoices and NumericalChoices that describe ranges of
# options.

solver_space = SolverSpace(
    {
        "options": {
            "gmres": {
                "ksp_gmres_restart": NumericalChoices([30, 50, 100]),
            },
            "mechanics_amg": {
                "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                    [0.5, 0.6, 0.7, 0.8, 0.9]
                ),
            },
            "cpr0_energy": {
                "pc_type": CategoricalChoices(["pbjacobi", "none"]),
            },
            "cpr0_mass": {
                "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                    [0.5, 0.6, 0.7, 0.8, 0.9]
                ),
            },
            "cpr1": {
                "pc_type": CategoricalChoices(["pbjacobi", "sor", "ilu"]),
            },
        }
    }
)
solver_selector = SolverSelector(
    solver_space=solver_space,
    performance_predictor=assemble_default_performance_predictor(),
)


def main():
    logging.basicConfig(level=logging.INFO)

    model_2d = FullModel(model_params_2d)
    linear_solver = pp_solvers.IterativeLinearSolver(
        solver_selector=solver_selector,
        solver_options=linear_solver_options,
        configuration_factory=pp_solvers.thm_factory,
    )
    pp.ModelRunner(
        model_2d,
        nonlinear_solver=pp.solvers.NewtonSolver(
            params={"nl_convergence_res_atol": 1e-6}, linear_solver=linear_solver
        ),
    ).run()


if __name__ == "__main__":
    main()
