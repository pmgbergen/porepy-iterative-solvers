import numpy as np
import porepy as pp
import scipy.sparse as sps
from petsc4py import PETSc
import pp_solvers

from porepy.examples.flow_benchmark_2d_case_1 import (
    Geometry as Geometry_2d_case_1,
    solid_constants_conductive_fractures as solid_constants_2d_1,
)
from porepy.examples.flow_benchmark_2d_case_4 import (
    Geometry as Geometry_2d_case_4,
    solid_constants as solid_constants_2d,
)
from porepy.examples.flow_benchmark_3d_case_3 import (
    Geometry as Geometry_3d_case_3,
    solid_constants as solid_constants_3d,
)


class FullModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp_solvers.IterativeSolverMixin,
    pp.MomentumBalance,
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


model_params_2d = {
    "material_constants": {
        "solid": solid_constants_2d_1,
    },
    "u_north": [0, 0.001],
    "grid_type": "cartesian",
    "meshing_arguments": {"cell_size": 0.25},
    "fracture_indices": [1],
    # "units": pp.Units(kg=1e2),
    "linear_solver": {
        "preconditioner_factory": pp_solvers.momentum_balance_factory,
        "options": {"gmres": {"ksp_monitor": None}},
    },
}
model_2d = FullModel(model_params_2d)
pp.run_time_dependent_model(
    model_2d,
    {
        "nl_convergence_tol_res": 1e-6,
    },
)
