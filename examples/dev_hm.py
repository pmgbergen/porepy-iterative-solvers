import logging

import porepy as pp
from porepy.examples.flow_benchmark_2d_case_1 import Geometry as Geometry_2d_case_1
from porepy.examples.flow_benchmark_2d_case_1 import (
    solid_constants_conductive_fractures as solid_constants_2d_1,
)
from porepy.examples.flow_benchmark_2d_case_4 import Geometry as Geometry_2d_case_4
from porepy.examples.flow_benchmark_2d_case_4 import (
    solid_constants as solid_constants_2d,
)
from porepy.examples.flow_benchmark_3d_case_3 import Geometry as Geometry_3d_case_3
from porepy.examples.flow_benchmark_3d_case_3 import (
    solid_constants as solid_constants_3d,
)

import pp_solvers


class FullModel(
    # Geometry_2d_case_1,
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.Poromechanics,
):
    pass


def main():
    logging.basicConfig(level=logging.INFO)

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
        nonlinear_solver=pp.solvers.NewtonSolver(
            params={"nl_convergence_res_atol": 1e-6}, linear_solver=linear_solver
        ),
    ).run()


if __name__ == "__main__":
    main()
