import numpy as np
import porepy as pp
import pytest
from porepy.applications.test_utils.models import add_mixin
from scipy.sparse.linalg import spsolve

from porepy.applications.md_grids.model_geometries import (
    TwoEllipticFractures3d,
    TwoWells3d,
)

import pp_solvers
from pp_solvers.solver_mixin import IterativeSolverMixin


@pytest.fixture(
    scope="module", params=["with_fractures", "with_wells", "single_dimension"]
)
def geometry_kind(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=["flow", "mechanics", "TH", "HM", "THM"])
def model_kind(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_kind: str, geometry_kind: str) -> pp.PorePyModel:
    """Instantiate a model for the test suites in this file."""
    match model_kind:
        case "flow":
            model_type = pp.SinglePhaseFlow
        case "mechanics":
            model_type = pp.MomentumBalance
        case "TH":
            model_type = pp.MassAndEnergyBalance
        case "HM":
            model_type = pp.Poromechanics
        case "THM":
            model_type = pp.Thermoporomechanics
        case default:
            raise ValueError(default)

    with_wells = geometry_kind == "with_wells"
    with_fractures = geometry_kind == "with_fractures"
    single_dimension = not with_fractures and not with_wells
    units = pp.Units()

    if with_fractures or single_dimension:
        class TailoredClass(
            pp.model_geometries.SquareDomainOrthogonalFractures,
            pp_solvers.IterativeSolverMixin,
        ):
            """Common base class for all models in this test suite."""

            def meshing_arguments(self):
                return {"cell_size": self.params["cell_size"]}

        params = {
            "linear_solver": {},
            "cell_size": units.convert_units(0.25, 'm'),
            "cartesian": True,
            'units': units,
            "fracture_indices": [0, 1] if with_fractures else [],
        }

    elif with_wells:
        
        # This breaks the generality of the test, but we need a 3D setup for the wells.
        # However, we also cover a 3D setup, which is not a bad thing by itself. The THM
        # model with this geometry has 364 DoFs total. 
        class TailoredClass(
                TwoWells3d,
                TwoEllipticFractures3d,
                pp_solvers.IterativeSolverMixin,
            ):
            pass
        params = {
            "linear_solver": {},
            "grid_type": "simplex",
            'units': units,
            "meshing_arguments": {
                'cell_size': units.convert_units(1.5, 'm'),
            },
            # Fractures
            'fracture_params': {
                'num_points': [3, 3],
            },
        }

    else:
        raise ValueError(f"Unknown geometry_kind: {geometry_kind}.")

    model_class = add_mixin(TailoredClass, model_type)
    model = model_class(params=params)
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    return model


def test_solve_linear_system(model: IterativeSolverMixin):
    model.bmat.rhs[:] = 1
    result = model.solve_linear_system()
    mat, rhs = model.linear_system
    rhs[:] = 1

    expected = spsolve(mat, rhs)
    np.testing.assert_allclose(result, expected, rtol=7e-10, atol=0)
