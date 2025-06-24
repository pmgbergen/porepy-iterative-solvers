import numpy as np
import porepy as pp
import scipy.sparse as sps
from petsc4py import PETSc
import FTHM_Solver

from functools import cached_property

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
    FTHM_Solver.IterativeSolverMixin,
    pp.model_geometries.SquareDomainOrthogonalFractures,
    FlowBenchmark2dCase4Model,
):
    pass


model_params_2d = {
    "material_constants": {
        "solid": solid_constants_2d,
        "fluid": pp.FluidComponent(**{"compressibility": 1e-7}),
    },
    "fracture_indices": [0, 1],
    "linear_solver": {
        "preconditioner_factory": FTHM_Solver.mass_balance_factory,
        "options": {"ksp_monitor": None},
    },
}
model_2d = FullModel(model_params_2d)
pp.run_time_dependent_model(model_2d)


pressure = model_2d.pressure(model_2d.mdg.subdomains())
# print(model_2d.equation_system.evaluate(pressure))
