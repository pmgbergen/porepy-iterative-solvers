import numpy as np
import porepy as pp
import pytest
from porepy.applications.test_utils.models import add_mixin
from scipy.sparse.linalg import spsolve

import pp_solvers


@pytest.fixture(scope="module", params=[False, True])
def with_fractures(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=["flow", "mechanics", "TH", "HM", "THM"])
def model_kind(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_kind, with_fractures) -> pp.PorePyModel:
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

    class TailoredClass(
        pp.model_geometries.SquareDomainOrthogonalFractures,
        pp_solvers.IterativeSolverMixin,
    ):
        """Common base class for all models in this test suite."""

        def meshing_arguments(self):
            return {"cell_size": self.params["cell_size"]}

    params = {
        'linear_solver': {},
        "cell_size": 0.25,
        "cartesian": True,
        "fracture_indices": [0, 1] if with_fractures else [],
    }
    model_class = add_mixin(TailoredClass, model_type)
    model = model_class(params=params)
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    return model


def test_solve_linear_system(model: pp.PorePyModel):
    model.rhs_reordered[:] = 1
    result = model.solve_linear_system()
    mat, rhs = model.linear_system
    rhs[:] = 1

    expected = spsolve(mat, rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=0)
