from pp_solvers.solver_selection.solver_space import (
    SolverSpace,
    NumericalChoices,
    CategoricalChoices,
    explain_decisions,
)

scheme = {
    "mech": {"strong_threshold": NumericalChoices([0.5, 0.7])},
    "flow": {"strong_threshold": NumericalChoices([0.2, 0.5])},
}

print(explain_decisions(SolverSpace(scheme))[0])
