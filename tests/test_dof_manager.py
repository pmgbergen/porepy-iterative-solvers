import porepy as pp
import numpy as np
from porepy.applications.test_utils.models import add_mixin
import pp_solvers as pps

import pytest
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import EquationNames


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


@pytest.mark.parametrize("with_fractures", [False, True])
def test_mechanics_model(with_fractures):
    """Test that the mechanics model is correctly treated by the DofManager."""

    model = create_model(with_fractures, pp.MomentumBalance)

    nd = model.nd

    fracture_subdomains = model.mdg.subdomains(dim=nd - 1)
    interfaces = model.mdg.interfaces(dim=nd - 1)

    num_matrix_subdomains = 1
    num_fracture_subdomains = len(fracture_subdomains)
    num_interfaces = len(interfaces)
    num_matrix_cells = np.sum([sd.num_cells for sd in model.mdg.subdomains(dim=nd)])
    num_fracture_cells = np.sum([sd.num_cells for sd in fracture_subdomains])
    num_interface_cells = np.sum([intf.num_cells for intf in interfaces])

    block_solvers = pps.momentum_balance_factory()
    dof_manager = DofManager(model, block_solvers)

    # By construction of the preconditioner, we expect two blocks:
    contact_solver = block_solvers[0]
    momentum_solver = block_solvers[1]

    # Check that the number of blocks is as expected: The contact solver should have one
    # block (the contact conditions), while the momentum solver should have two groups
    # (momentum balance in the matrix and force continuity at the interfaces).
    assert len(dof_manager.blocks_of_solver(contact_solver)) == 1
    assert len(dof_manager.blocks_of_solver(momentum_solver)) == 2

    # The equation groups for the solvers.
    # Contact conditions, one per fracture subdomain. Note that this implicitly tests
    # that the contact equations in the normal and tangential directions are merged into
    # one equation group per fracture subdomain.
    assert len(dof_manager.equation_groups[0]) == num_fracture_subdomains
    # Matrix momentum balance, one per matrix subdomain.
    assert len(dof_manager.equation_groups[1]) == num_matrix_subdomains
    # Interface force balance, one per interface.
    assert len(dof_manager.equation_groups[2]) == num_interfaces

    # Check that the total number of groups is as expected, and that they together are
    # numbered linearly from 0. This will not be the case if the merging of normal and
    # tangential contact conditions fail.
    all_groups = []
    for group in dof_manager.equation_groups:
        all_groups.extend(group)
    assert np.allclose(np.arange(len(all_groups)), np.sort(all_groups))

    assert (
        len(dof_manager.eq_dofs_by_blocks(model))
        == num_interfaces + num_fracture_subdomains + num_matrix_subdomains
    )

    num_dofs_per_equation_group = np.zeros(len(dof_manager.equation_groups), dtype=int)
    for group_ind in range(len(dof_manager.equation_groups)):
        num_dofs_per_equation_group[group_ind] = sum(
            len(dof_manager.eq_dofs_by_blocks(model)[i])
            for i in dof_manager.equation_groups[group_ind]
        )

    assert num_dofs_per_equation_group[0] == nd * num_fracture_cells
    assert num_dofs_per_equation_group[1] == nd * num_matrix_cells
    assert num_dofs_per_equation_group[2] == nd * num_interface_cells

    # By construction of the EquationGroups that go into the preconditioner, we know
    # that the contact groups is the first one (index 0). The interface groups is 2
    # if there are interfaces, and -1 if there are no interfaces.
    if num_interfaces == 0:
        expected_groups = {"contact": 0, "u_intf": -1}
        # No permutation expected, as there are no interfaces.
        expected_permutation = None
    else:
        expected_groups = {"contact": 0, "u_intf": 2}

        # Construct the expected permutation vector: The momentum and interface force
        # balance equations should be unperturbed, while the contact equations should be
        # permuted so that the normal and tangential equations are grouped together for
        # each fracture cell.

        # The construction of the permutation vector is a bit convoluted: First find the
        # number of degrees of freedom for each equation in the assembled matrix.
        equation_dofs = model.equation_system._equation_image_space_composition
        next_offset = 0
        offsets = {}
        for eq_name, eq in equation_dofs.items():
            offsets[eq_name] = next_offset
            for _, indices in eq.items():
                next_offset += indices.size

        # Build up an expected permutation vector. Assume that the momentum balance and
        # interface force balance equations are grouped first.
        # NOTE: If this test fails, verifying this assumption is a good early step.
        indices = [
            np.arange(num_matrix_cells * nd),  # momentum balance dofs
            np.arange(num_interface_cells * nd)  # interface force balance dofs
            + offsets[EquationNames.INTERFACE_FORCE_BALANCE.value],
        ]

        # Get the indices for the contact equations in the normal and tangential
        # direction. In the PorePy ordering, these are separate; we need to merge them
        # fracture cell for fracture cell.
        normal_key = EquationNames.CONTACT_NORMAL.value
        tangential_key = EquationNames.CONTACT_TANGENTIAL.value
        normal_equations = equation_dofs[normal_key]
        tangential_equations = equation_dofs[tangential_key]

        # Loop over all fracture subdomains. Get the dofs for the normal and tangential
        # direction, with respective offsets. Group them together for each cell (done by
        # the vstack and F-raveling).
        for sd in fracture_subdomains:
            normal_indices = normal_equations[sd] + offsets[normal_key]
            tangential_indices = tangential_equations[sd] + offsets[tangential_key]
            indices.append(np.vstack((normal_indices, tangential_indices)).ravel("F"))

        # Now stack all indices.
        expected_permutation = np.hstack(indices)

    _check_group_identification(dof_manager, model, expected_groups)
    _check_permutation_vector(dof_manager, model, expected_permutation)
