import numpy as np
import porepy as pp
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
    # Geometry_2d_case_1,
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
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
        "reference_values": {"pressure": 1},
    },
    "fracture_indices": [0, 1],  # 0, 1],
    "u_north": -0.001,
    "meshing_arguments": {"cell_size": 0.1},
}
model_2d = FullModel(model_params_2d)
linear_solver = pp_solvers.IterativeLinearSolver(
    configuration_factory=pp_solvers.hm_factory,
    solver_options={"gmres": {"ksp_monitor": None}},
)
pp.ModelRunner(
    model_2d,
    nonlinear_solver=pp.NewtonSolver(
        params={"nl_convergence_res_atol": 1e-6}, linear_solver=linear_solver
    ),
).run()
