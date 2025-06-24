import numpy as np
import porepy as pp
import scipy.sparse as sps
import FTHM_Solver.hm_solver
from petsc4py import PETSc
import FTHM_Solver

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
    # Geometry_2d_case_1,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    FTHM_Solver.IterativeSolverMixin,
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


model_params_2d = {
    "material_constants": {
        "solid": solid_constants_2d_1,
    },
    "fracture_indices": [0, 1],  # 0, 1],
    "u_north": -0.001,
    "meshing_arguments": {"cell_size": 0.1},
    "linear_solver": {"preconditioner_factory": FTHM_Solver.thm_factory},
}
model_2d = FullModel(model_params_2d)
pp.run_time_dependent_model(
    model_2d,
    {
        "nl_convergence_tol_res": 1e-6,
    },
)
