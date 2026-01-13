import numpy as np
import porepy as pp
import pytest
from porepy.applications.test_utils.models import add_mixin

import pp_solvers
from pp_solvers.dof_manager import DofManager
from pp_solvers.preconditioners import SinglePhysicsPreconditioner

from pp_solvers.fixed_stress import (
    make_fs_analytical_slow_new,
    get_fixed_stress_stabilization,
    get_fs_fractures_analytical,
)


@pytest.fixture(scope="module", params=[False, True])
def with_fractures(request) -> bool:
    return request.param


@pytest.fixture(scope="module")
def model(with_fractures) -> pp.PorePyModel:
    """Instantiate a model for the test suites in this file."""

    class TailoredClass(
        pp_solvers.IterativeSolverMixin,
        pp.model_geometries.SquareDomainOrthogonalFractures,
        pp.Thermoporomechanics,
    ):
        """Common base class for all models in this test suite."""

        def meshing_arguments(self):
            return {"cell_size": self.params["cell_size"]}

    params = {
        "linear_solver": {},
        "cell_size": 0.25,
        "cartesian": True,
        "fracture_indices": [0, 1] if with_fractures else [],
        # A non-zero fluid comressibility is needed for the fracture fixed stress.
        "material_constants": {
            "fluid": pp.FluidComponent(
                compressibility=1,
            ),
        },
    }
    model = TailoredClass(params=params)
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    return model


def test_fixed_stress(model: pp_solvers.IterativeSolverMixin, with_fractures: bool):
    jacobian = model.bmat

    # The fixed stress in fractures requires a non-zero u_intf jump.
    interfaces = model.mdg.interfaces(dim=model.nd - 1)
    u_intf = model.interface_displacement(interfaces)
    u_intf_values = model.equation_system.get_variable_values([u_intf], iterate_index=0)
    u_intf_values[:] = np.arange(u_intf_values.size)
    model.equation_system.set_variable_values(
        values=u_intf_values, variables=[u_intf], iterate_index=0
    )

    # YZ: Is there a better way than hard-coding these numbers?
    num_groups = 14
    p_mat_group = 8
    p_frac_group = 9

    all_groups = list(range(num_groups))
    result = make_fs_analytical_slow_new(
        J=jacobian,
        model=model,
        p_mat_group=p_mat_group,
        p_frac_group=p_frac_group,
        groups=all_groups,
    )

    expected_matrix = get_fixed_stress_stabilization(model).toarray()
    expected_fractures = get_fs_fractures_analytical(model).toarray()

    # We check that the right stabilization submatrices are placed correctly, and there
    # is nothing else.
    for row_group in all_groups:
        for col_group in all_groups:
            submat = result[row_group, col_group].mat
            if row_group == col_group == p_mat_group:
                assert not submat.nnz == 0, submat
                np.testing.assert_equal(submat.toarray(), expected_matrix)
            elif row_group == col_group == p_frac_group:
                if with_fractures:
                    assert not submat.nnz == 0, submat
                else:
                    assert submat.nnz == 0, submat
                np.testing.assert_equal(submat.toarray(), expected_fractures)
            else:
                assert submat.nnz == 0, submat
