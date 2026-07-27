import logging

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
    pp.model_geometries.SquareDomainOrthogonalFractures,
    # Geometry_2d_case_1,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Thermoporomechanics,
):
    pass


model_params_2d = {
    "material_constants": {
        "solid": solid_constants_2d_1,
    },
    "fracture_indices": [0, 1],  # 0, 1],
    "u_north": -0.001,
    "meshing_arguments": {"cell_size": 0.1},
}

linear_solver_options = {"gmres": {"ksp_monitor": None}}


def main():
    logging.basicConfig(level=logging.INFO)
    model_2d = FullModel(model_params_2d)
    linear_solver = pp_solvers.IterativeLinearSolver(
        configuration_factory=pp_solvers.thm_factory,
        solver_options=linear_solver_options,
    )
    pp.ModelRunner(
        model_2d,
        nonlinear_solver=pp.solvers.NewtonSolver(
            params={"nl_convergence_res_atol": 1e-6}, linear_solver=linear_solver
        ),
    ).run()


if __name__ == "__main__":
    main()
