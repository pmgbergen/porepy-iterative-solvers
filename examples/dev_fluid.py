import logging

import porepy as pp
from porepy.examples.flow_benchmark_2d_case_1 import FlowBenchmark2dCase1Model
from porepy.examples.flow_benchmark_2d_case_1 import (
    solid_constants_conductive_fractures as solid_constants_2d_1,
)
from porepy.examples.flow_benchmark_2d_case_4 import FlowBenchmark2dCase4Model
from porepy.examples.flow_benchmark_2d_case_4 import (
    solid_constants as solid_constants_2d,
)
from porepy.examples.flow_benchmark_3d_case_3 import FlowBenchmark3dCase3Model
from porepy.examples.flow_benchmark_3d_case_3 import (
    solid_constants as solid_constants_3d,
)

import pp_solvers


class FullModel(
    pp.model_geometries.SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    FlowBenchmark2dCase4Model,
):
    pass


def main():
    logging.basicConfig(level=logging.INFO)
    model_params_2d = {
        "material_constants": {
            "solid": solid_constants_2d,
            "fluid": pp.FluidComponent(**{"compressibility": 1e-7}),
        },
        "reference_variable_values": pp.ReferenceVariableValues(**{"pressure": 1}),
        "fracture_indices": [0, 1],
        "units": pp.Units(m=1e-4),
    }
    model_2d = FullModel(model_params_2d)
    linear_solver = pp_solvers.IterativeLinearSolver(
        configuration_factory=pp_solvers.mass_balance_factory,
        solver_options={"gmres": {"ksp_monitor": None}},
    )
    pp.ModelRunner(
        model_2d, nonlinear_solver=pp.solvers.NewtonSolver(linear_solver=linear_solver)
    ).run()

    # pressure = model_2d.pressure(model_2d.mdg.subdomains())
    # print(model_2d.equation_system.evaluate(pressure))


if __name__ == "__main__":
    main()
