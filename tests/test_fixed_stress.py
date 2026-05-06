import numpy as np
import porepy as pp
import pytest

import pp_solvers
from pp_solvers.block_linear_system import BlockLinearSystem
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    MassBalancePressureFracturesGroup,
    MassBalancePressureMatrixGroup,
)
from pp_solvers.fixed_stress import (
    construct_fixed_stress_block_matrix,
    get_fixed_stress_stabilization_fractures,
    get_fixed_stress_stabilization_porous_media,
)
from pp_solvers.petsc_utils import petsc_to_csr
from pp_solvers.preconditioners import FixedStressInverter


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

    # The fixed stress in fractures requires a non-zero u_intf jump.
    interfaces = model.mdg.interfaces(dim=model.nd - 1)
    u_intf = model.interface_displacement(interfaces)
    u_intf_values = model.equation_system.get_variable_values([u_intf], iterate_index=0)
    u_intf_values[:] = np.arange(u_intf_values.size)
    # Setting the *previous time step* (initial condition) value to non-zero. It will be
    # fetched in model.before_nonlinear_loop and passed to the current solution guess.
    model.equation_system.set_variable_values(
        values=u_intf_values, variables=[u_intf], time_step_index=0
    )

    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    return model


def test_fixed_stress(model: pp_solvers.IterativeSolverMixin, with_fractures: bool):
    """The function to compose the fixed stress stabilization for the block matrix is
    `make_fs_analytical_slow_new`. This checks that it does the right things - modifies
    the pressure diagonal blocks and keeps everything else not touched."""

    jacobian = model.bmat

    dof_manager: DofManager = model._solver_factory.dof_manager
    num_groups = len(dof_manager.groups())
    try:
        p_mat_group, p_frac_group = dof_manager.indices_of_groups(
            [MassBalancePressureMatrixGroup(), MassBalancePressureFracturesGroup()]
        )
    except:
        assert False, "These groups should be present."

    all_groups = list(range(num_groups))
    result = construct_fixed_stress_block_matrix(
        indexer=jacobian.indexer,
        model=model,
        p_mat_group=p_mat_group,
        p_frac_group=p_frac_group,
    )

    expected_matrix = get_fixed_stress_stabilization_porous_media(model).toarray()
    expected_fractures = get_fixed_stress_stabilization_fractures(model).toarray()

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


def test_fixed_stress_inverter(model: pp.PorePyModel):
    """Integration test that check that the configuration FixedStressInverter provides a
    correct fixed stress matrix."""
    dof_manager: DofManager = model._solver_factory.dof_manager
    inverter = FixedStressInverter()
    bmat: BlockLinearSystem = model.bmat

    config = inverter.petsc_assembly_config(dof_manager=dof_manager)
    petsc_fs_matrix = petsc_to_csr(config["inverter_additive"](bmat.indexer))

    p_mat_group, p_frac_group = dof_manager.indices_of_groups(
        [MassBalancePressureMatrixGroup(), MassBalancePressureFracturesGroup()]
    )

    expected_matrix = construct_fixed_stress_block_matrix(
        dof_manager.model,
        bmat.indexer,
        p_mat_group=p_mat_group,
        p_frac_group=p_frac_group,
    ).mat

    # These should be identical.
    assert np.all((petsc_fs_matrix - expected_matrix).data == 0)
