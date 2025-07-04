import porepy as pp
import numpy as np
from porepy.applications.test_utils.models import add_mixin
import pp_solvers as pps

import pytest
from pp_solvers.dof_manager import DofManager


class TailoredClass(pp.model_geometries.SquareDomainOrthogonalFractures):
    # Common base class for all models in this test suite.
    def meshing_arguments(self):
        return {"cell_size": self.params["cell_size"]}

    pass


def set_parameters(include_fractures: bool):
    params = {"cell_size": 0.25, "cartesian": True}
    if include_fractures:
        params["fracture_indices"] = [0, 1]
    else:
        params["fracture_indices"] = []

    return params


def create_model(with_fractures: bool, base_model_class):
    """Create a model with or without fractures."""
    params = set_parameters(include_fractures=with_fractures)

    model_class = add_mixin(TailoredClass, base_model_class)
    model = model_class(params=params)
    model.prepare_simulation()
    model.assemble_linear_system()
    return model


def _check_permutation_vector(dof_manager, model, expected_vector=None):
    permutation = dof_manager.eq_rows_permutation(model)
    if expected_vector is None:
        expected_vector = np.arange(permutation.size)
    assert np.all(permutation == expected_vector), (
        "Permutation vector does not match expected values."
    )


def _check_group_identification(dof_manager, model, expected_groups: dict = None):
    if expected_groups is None:
        expected_groups = {}
    truth = {"contact": -1, "u_intf": -1, "energy_balance": [-1]} | expected_groups

    assert dof_manager.identify_contact_group() == truth["contact"]
    assert dof_manager.identify_u_intf_group(model) == truth["u_intf"]
    assert dof_manager.identify_energy_balance_group() == truth["energy_balance"]


@pytest.mark.parametrize("with_fractures", [False, True])
def test_single_phase_flow_model(with_fractures: bool):
    """Test that the model without fractures has the expected number of blocks."""
    model = create_model(with_fractures, pp.SinglePhaseFlow)

    num_subdomains = len(model.mdg.subdomains())
    num_interfaces = len(model.mdg.interfaces())
    num_subdomain_cells = np.sum([sd.num_cells for sd in model.mdg.subdomains()])
    num_interface_cells = np.sum([intf.num_cells for intf in model.mdg.interfaces()])

    block_solvers = pps.mass_balance_factory()
    dof_manager = DofManager(model, block_solvers)

    # By construction of the preconditioner, we expect two blocks:
    interface_solver = block_solvers[0]
    mass_solver = block_solvers[1]

    # Check that the number of blocks is as expected: The preconditioner expects two
    # blocks (mass conservation and interface fluxes).
    assert len(dof_manager.blocks_of_solver(interface_solver)) == 1
    assert len(dof_manager.blocks_of_solver(mass_solver)) == 1

    # The equation groups for the solvers (interface flux and mass conservation) should
    # have length equal to the number of interfaces and subdomains, respectively.
    assert len(dof_manager.equation_groups[0]) == num_interfaces
    assert len(dof_manager.equation_groups[1]) == num_subdomains

    assert len(dof_manager.eq_dofs_by_blocks(model)) == num_interfaces + num_subdomains

    num_dofs_per_equation_group = np.zeros(len(dof_manager.equation_groups), dtype=int)
    for group_ind in range(len(dof_manager.equation_groups)):
        num_dofs_per_equation_group[group_ind] = sum(
            len(dof_manager.eq_dofs_by_blocks(model)[i])
            for i in dof_manager.equation_groups[group_ind]
        )

    assert num_dofs_per_equation_group[0] == num_interface_cells
    assert num_dofs_per_equation_group[1] == num_subdomain_cells

    # Both the permutation vector and group identification should give default values.
    _check_permutation_vector(dof_manager, model)
    _check_group_identification(dof_manager, model)

    # It is possible to test the PETSc IS here, but then we need to give the models
    # the SolverMixin. TODO?
    # dof_manager.petsc_is(block_solvers[0], block_solvers[1:], model.bmat)
