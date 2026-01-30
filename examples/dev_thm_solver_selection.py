import numpy as np
import porepy as pp
import pp_solvers
from porepy.examples.flow_benchmark_2d_case_1 import (
    # Geometry as Geometry_2d_case_1,
    solid_constants_conductive_fractures as solid_constants_2d_1,
)


class FullModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp_solvers.IterativeSolverMixin,
    pp.Thermoporomechanics,
):
    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual,
        reference_residual: np.ndarray,
        nl_params,
    ) -> tuple[bool, bool]:
        # nonlinear_increment based norm
        nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
            nonlinear_increment
        )
        # Residual based norm
        residual_norm = self.compute_residual_norm(residual, reference_residual)

        print(f"nl_inc: {nonlinear_increment_norm}, res: {residual_norm}")
        return super().check_convergence(
            nonlinear_increment_norm, residual, reference_residual, nl_params
        )


from pp_solvers.solver_selection.selector import SolverSelector
from pp_solvers.solver_selection.solver_space import (
    SolverSpace,
    NumericalChoices,
    CategoricalChoices,
)
from pp_solvers.solver_selection.performance_predictor import (
    assemble_default_performance_predictor,
)


solver_space_scheme = {
    "identity": {"pc_type": CategoricalChoices(["jacobi", "sor", "none"])},
    "mechanics": {
        "pc_hypre_boomeramg_strong_threshold": NumericalChoices([0.5, 0.6, 0.7, 0.8]),
    }
    | CategoricalChoices(
        [
            {
                "pc_hypre_boomeramg_smooth_type": "ilu",
            },
            {
                "pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                    [
                        "Jacobi",
                        "sequential-Gauss-Seidel",
                        "seqboundary-Gauss-Seidel",
                        "SOR/Jacobi",
                        "backward-SOR/Jacobi",
                        "symmetric-SOR/Jacobi",
                        "l1scaled-SOR/Jacobi",
                        "Gaussian-elimination",
                        "l1-Gauss-Seidel",
                        "backward-l1-Gauss-Seidel",
                        "CG",
                        "Chebyshev",
                        "FCF-Jacobi",
                        "l1scaled-Jacobi",
                    ]
                ),
            },
        ]
    ),
}
solver_space = SolverSpace(solver_space_scheme=solver_space_scheme)
solver_selector = SolverSelector(
    solver_space=solver_space,
    performance_predictor=assemble_default_performance_predictor(),
)


model_params_2d = {
    "material_constants": {
        "solid": solid_constants_2d_1,
    },
    "fracture_indices": [0, 1],  # 0, 1],
    "u_north": -0.001,
    "meshing_arguments": {"cell_size": 0.1},
    "linear_solver": {
        "preconditioner_factory": pp_solvers.thm_factory,
        "solver_selector": solver_selector,
        "options": {
            "ksp_monitor": None,
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-12,
            "identity": {"pc_type": "jacobi"},
        },
    },
}
model_2d = FullModel(model_params_2d)
pp.run_time_dependent_model(
    model_2d,
    {
        "nl_convergence_tol_res": 1e-6,
    },
)
