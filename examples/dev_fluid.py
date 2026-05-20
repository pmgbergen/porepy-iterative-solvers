import numpy as np
import porepy as pp
import scipy.sparse as sps
from petsc4py import PETSc
import pp_solvers


from porepy.examples.flow_benchmark_2d_case_1 import (
    FlowBenchmark2dCase1Model,
    solid_constants_conductive_fractures as solid_constants_2d_1,
)
from porepy.examples.flow_benchmark_2d_case_4 import (
    FlowBenchmark2dCase4Model,
    solid_constants as solid_constants_2d,
)
from porepy.examples.flow_benchmark_3d_case_3 import (
    FlowBenchmark3dCase3Model,
    solid_constants as solid_constants_3d,
)


class FullModel(
    pp_solvers.IterativeSolverMixin,
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    FlowBenchmark2dCase4Model,
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
        "solid": solid_constants_2d,
        "fluid": pp.FluidComponent(**{"compressibility": 1e-7}),
    },
    "reference_variable_values": pp.ReferenceVariableValues(**{"pressure": 1}),
    "fracture_indices": [0, 1],
    "units": pp.Units(m=1e-4),
    "linear_solver": pp_solvers.LinearSolverParams(
        preconditioner_factory=pp_solvers.mass_balance_factory,
        options={
            "gmres": {
                "ksp_monitor": None,
            }
        },
    ),
}
model_2d = FullModel(model_params_2d)
pp.run_time_dependent_model(model_2d)


pressure = model_2d.pressure(model_2d.mdg.subdomains())
# print(model_2d.equation_system.evaluate(pressure))
