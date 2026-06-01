"""This example illustrates usage of the solver selector for the THM model, the same
models as in dev_thm.py.

"""

from examples.dev_thm import FullModel, model_params_2d
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

model_params_2d = model_params_2d | {
    "linear_solver": pp_solvers.LinearSolverParams(
        # Pass the solver selector to enable it.
        solver_selector=solver_selector,
        # Optionally, pass the manual options. They are merged with the solver
        # selector's output, but the solver selector takes priority on conflicts.
        options={
            "gmres": {
                "ksp_monitor": None,
            }
        },
    )
}


def main():
    model_2d = FullModel(model_params_2d)
    pp.run_time_dependent_model(
        model_2d,
        {
            "nl_convergence_tol_res": 1e-6,
        },
    )


if __name__ == "__main__":
    main()
