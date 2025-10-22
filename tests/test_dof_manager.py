"""This module tests the integration between the dof manager and porepy models.

The following cases are covered by the parametrized fixtures:

                  |flow|mech| TH | HM |THM |
with fractures    | x  | x  | x  | x  | x  |
without fractures | x  | x  | x  | x  | x  |

All the fixtures, including the dof managers for each model, are created once and reused
for all tests. This means the models and the dof managers should be READ ONLY.

"""

import numpy as np
import porepy as pp
import pytest
from porepy.applications.test_utils.models import add_mixin

import pp_solvers
from pp_solvers.dof_manager import DofManager
from pp_solvers.preconditioners import SinglePhysicsPreconditioner


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

    class TailoredClass(pp.model_geometries.SquareDomainOrthogonalFractures):
        """Common base class for all models in this test suite."""

        def meshing_arguments(self):
            return {"cell_size": self.params["cell_size"]}

    params = {
        "cell_size": 0.25,
        "cartesian": True,
        "fracture_indices": [0, 1] if with_fractures else [],
    }
    model_class = add_mixin(TailoredClass, model_type)
    model = model_class(params=params)
    model.prepare_simulation()
    return model


@pytest.fixture(scope="module")
def solvers(model_kind: str) -> list[SinglePhysicsPreconditioner]:
    match model_kind:
        case "flow":
            return pp_solvers.mass_balance_factory()
        case "mechanics":
            return pp_solvers.momentum_balance_factory()
        case "TH":
            return pp_solvers.th_factory()
        case "HM":
            return pp_solvers.hm_factory()
        case "THM":
            return pp_solvers.thm_factory()
        case default:
            raise ValueError(default)


@pytest.fixture(scope="module")
def dof_manager(
    model: pp.PorePyModel, solvers: list[SinglePhysicsPreconditioner]
) -> DofManager:
    return DofManager(model, solvers)


@pytest.fixture(scope="module")
def expected_composition(
    model: pp.PorePyModel, model_kind: str
) -> dict[str, np.ndarray]:
    """The expected values for the tests."""

    # Constructing relevant sets of domains.
    all_subdomains = model.mdg.subdomains()
    fractures = model.mdg.subdomains(dim=model.nd - 1)
    all_interfaces = model.mdg.interfaces()
    intersections = [
        domain
        for x in range(0, model.nd - 1)
        for domain in (model.mdg.subdomains(dim=x))
    ]
    interfaces_ambient_frac = model.mdg.interfaces(dim=model.nd - 1)
    porous_media_subdomains = model.mdg.subdomains(dim=model.nd)

    # And counting the number of cells in each.
    all_subdomains = np.array([x.num_cells for x in all_subdomains])
    fractures = np.array([x.num_cells for x in fractures])
    all_interfaces = np.array([x.num_cells for x in all_interfaces])
    intersections = np.array([x.num_cells for x in intersections])
    interfaces_ambient_frac = np.array([x.num_cells for x in interfaces_ambient_frac])
    porous_media_subdomains = np.array([x.num_cells for x in porous_media_subdomains])

    nd = model.nd

    # The keys are used for referring to specific groups within this file. They are not
    # meant to be consistent with equation names or anything else.
    match model_kind:
        case "flow":
            return {
                "intf_fluid_flux": all_interfaces,
                "mass_balance_everywhere": all_subdomains,
            }
        case "mechanics":
            return {
                "contact": fractures * nd,
                "momentum_balance": porous_media_subdomains * nd,
                "intf_force_balance": interfaces_ambient_frac * nd,
            }
        case "TH":
            return {
                "intf_fluid_flux": all_interfaces,
                "intf_heat_advection": all_interfaces,
                "intf_heat_diffusion": all_interfaces,
                # Mass balance ambient, fractures, lower (separately).
                "mass_balance_ambient": porous_media_subdomains,
                "mass_balance_fractures": fractures,
                "mass_balance_intersections": intersections,
                # Energy balance ambient, fractures, lower (separately).
                "energy_balance_ambient": porous_media_subdomains,
                "energy_balance_fractures": fractures,
                "energy_balance_intersections": intersections,
            }
        case "HM":
            return {
                "contact": fractures * nd,
                "intf_fluid_flux": all_interfaces,
                "momentum_balance": porous_media_subdomains * nd,
                "intf_force_balance": interfaces_ambient_frac * nd,
                # Mass balance ambient, fractures, lower (separately).
                "mass_balance_ambient": porous_media_subdomains,
                "mass_balance_fractures": fractures,
                "mass_balance_intersections": intersections,
            }
        case "THM":
            return {
                "contact": fractures * nd,
                "intf_fluid_flux": all_interfaces,
                "intf_heat_advection": all_interfaces,
                "intf_heat_diffusion": all_interfaces,
                "momentum_balance": porous_media_subdomains * nd,
                "intf_force_balance": interfaces_ambient_frac * nd,
                # Mass balance ambient, fractures, lower (separately).
                "mass_balance_ambient": porous_media_subdomains,
                "mass_balance_fractures": fractures,
                "mass_balance_intersections": intersections,
                # Energy balance ambient, fractures, lower (separately).
                "energy_balance_ambient": porous_media_subdomains,
                "energy_balance_fractures": fractures,
                "energy_balance_intersections": intersections,
            }
        case default:
            raise ValueError(default)


# Tests begin here.


def test_variable_equation_groups(dof_manager: DofManager, expected_composition: dict):
    """Tests properties `variable_groups` and `equation_groups`."""
    variable_groups = dof_manager.variable_groups
    equation_groups = dof_manager.equation_groups

    # All indices within groups should form a range [0, num_groups).
    flat_variable_groups = [idx for group in variable_groups for idx in group]
    flat_equation_groups = [idx for group in equation_groups for idx in group]
    assert np.all(np.arange(len(flat_equation_groups)) == np.sort(flat_equation_groups))
    assert np.all(np.arange(len(flat_variable_groups)) == np.sort(flat_variable_groups))

    # We check the expected composition.
    assert len(variable_groups) == len(equation_groups) == len(expected_composition)
    for i, expected_domains in enumerate(expected_composition.values()):
        len_expected_domains = len(expected_domains)
        assert len(variable_groups[i]) == len_expected_domains
        assert len(equation_groups[i]) == len_expected_domains


def test_eq_var_dofs_by_blocks(
    dof_manager: DofManager,
    model: pp.PorePyModel,
    expected_composition: dict[str, np.ndarray],
):
    """Tests methods `eq_dofs_by_blocks` and `var_dofs_by_blocks`. This test assumes
    that properties `variable_groups` and `equation_groups` work correctly as tested in
    `test_variable_equation_groups`.

    """
    eq_dofs_by_blocks = dof_manager.eq_dofs_by_blocks(model)
    var_dofs_by_blocks = dof_manager.var_dofs_by_blocks(model)

    # The number of dofs_by_blocks must be as expected in their groups.
    variable_groups = dof_manager.variable_groups
    equation_groups = dof_manager.equation_groups
    flat_variable_groups = [idx for group in variable_groups for idx in group]
    flat_equation_groups = [idx for group in equation_groups for idx in group]
    assert len(eq_dofs_by_blocks) == len(flat_equation_groups)
    assert len(var_dofs_by_blocks) == len(flat_variable_groups)

    # Total number of dofs must be equal to the problem size
    total_num_dofs = model.equation_system.num_dofs()
    flat_eq_dofs = [idx for block in eq_dofs_by_blocks for idx in block]
    flat_var_dofs = [idx for block in var_dofs_by_blocks for idx in block]
    assert np.all(np.arange(total_num_dofs) == np.sort(flat_eq_dofs))
    assert np.all(np.arange(total_num_dofs) == np.sort(flat_var_dofs))

    # We check the expected composition.
    for i, expected_num_dofs in enumerate(expected_composition.values()):
        # We take dofs of all subdomains/interfaces that are in the same group.
        expected_num_dofs = sum(expected_num_dofs)
        num_eq_dofs_in_group = sum(
            dofs.size for blk in equation_groups[i] for dofs in eq_dofs_by_blocks[blk]
        )
        num_var_dofs_in_group = sum(
            dofs.size for blk in variable_groups[i] for dofs in var_dofs_by_blocks[blk]
        )
        assert num_eq_dofs_in_group == expected_num_dofs
        assert num_var_dofs_in_group == expected_num_dofs


def test_blocks_of_solver(
    dof_manager: DofManager,
    solvers: list[SinglePhysicsPreconditioner],
    model_kind: str,
    expected_composition: dict,
):
    """Tests method `blocks_of_solver` against known expected compositions."""
    match model_kind:
        case "flow":
            expected = [
                ["intf_fluid_flux"],
                ["mass_balance_everywhere"],
            ]
        case "mechanics":
            expected = [
                ["contact"],
                [
                    "momentum_balance",
                    "intf_force_balance",
                ],
            ]
        case "TH":
            expected = [
                [
                    "intf_fluid_flux",
                    "intf_heat_advection",
                    "intf_heat_diffusion",
                ],
                [
                    "mass_balance_ambient",
                    "mass_balance_fractures",
                    "mass_balance_intersections",
                    "energy_balance_ambient",
                    "energy_balance_fractures",
                    "energy_balance_intersections",
                ],
            ]
        case "HM":
            expected = [
                ["contact"],
                ["intf_fluid_flux"],
                [
                    "momentum_balance",
                    "intf_force_balance",
                ],
                [
                    "mass_balance_ambient",
                    "mass_balance_fractures",
                    "mass_balance_intersections",
                ],
            ]
        case "THM":
            expected = [
                ["contact"],
                [
                    "intf_fluid_flux",
                    "intf_heat_advection",
                    "intf_heat_diffusion",
                ],
                [
                    "momentum_balance",
                    "intf_force_balance",
                ],
                [
                    "mass_balance_ambient",
                    "mass_balance_fractures",
                    "mass_balance_intersections",
                    "energy_balance_ambient",
                    "energy_balance_fractures",
                    "energy_balance_intersections",
                ],
            ]
        case default:
            raise ValueError(default)

    keys = list(expected_composition.keys())
    index_groups = [[keys.index(k) for k in group] for group in expected]

    assert len(solvers) == len(expected)
    for expected_blocks_of_solver, solver in zip(index_groups, solvers):
        assert dof_manager.blocks_of_solver(solver) == expected_blocks_of_solver


def test_identify_contact_group(dof_manager: DofManager, expected_composition: dict):
    try:
        expected = list(expected_composition.keys()).index("contact")
    except ValueError:
        expected = -1
    assert dof_manager.identify_contact_group() == expected


def test_identify_u_intf_group(
    dof_manager: DofManager,
    model: pp.PorePyModel,
    expected_composition: dict,
    with_fractures: bool,
):
    if with_fractures:
        try:
            expected = list(expected_composition.keys()).index("intf_force_balance")
        except ValueError:
            expected = -1
    else:
        expected = -1

    assert dof_manager.identify_u_intf_group(model) == expected


def test_identify_energy_balance_group(
    dof_manager: DofManager,
    expected_composition: dict,
):
    expected = list(
        i for i, key in enumerate(expected_composition) if "energy_balance" in key
    )
    if len(expected) == 0:
        expected = [-1]

    assert dof_manager.identify_energy_balance_group() == expected


def test_eq_rows_permutation(dof_manager: DofManager, model: pp.PorePyModel):
    """This tests the method `eq_rows_permutation`, assuming that `eq_dofs_by_blocks`,
    `equation_groups` and `identify_contact_group` work correctly, as tested above.

    """
    # Constructing the expected permutation. So far, it is equivalent to
    # np.arange(num_dofs), meaning no permutation. This is the right behavior if contact
    # mechanics is not present in the model. If it is, we will modify the array below.
    eq_dofs_by_blocks = dof_manager.eq_dofs_by_blocks(model)
    expected_permutation = np.concatenate(eq_dofs_by_blocks)

    # Checking if there is contact mechanics in the model.
    contact_group = dof_manager.identify_contact_group()
    if contact_group != -1:
        # Construct the expected permutation vector: The contact equations should be
        # permuted so that the normal and tangential equations are grouped together for
        # each fracture cell. Other equations should be unperturbed.
        blocks_in_contact_group = dof_manager.equation_groups[contact_group]

        # Making a list of contact mechanics dofs. Each element is an array of contact
        # dofs for a single fracture.
        dofs_in_contact_group = [
            eq_dofs_by_blocks[blk] for blk in blocks_in_contact_group
        ]

        # The list can be empty if the contact equation is present, but there are no
        # fractures in the model. In this case, we do nothing.
        if len(dofs_in_contact_group) > 0:
            # Making a flat array of dofs correspoding to all fractures.
            dofs_in_contact_group = np.concatenate(dofs_in_contact_group)

            # First half of the array - normal component for all fractures.
            # Second half of the array - tangential.
            # TODO: Probably, should extend this test for 3D.
            mid = dofs_in_contact_group.size // 2

            # Making the normal and tangential dofs for a each grid cell live together.
            perfuted_contact_dofs = np.stack(
                [dofs_in_contact_group[:mid], dofs_in_contact_group[mid:]]
            ).ravel("F")

            # Setting the permutation array to the contact mechanics location.
            expected_permutation[dofs_in_contact_group] = perfuted_contact_dofs

    result = dof_manager.eq_rows_permutation(model)
    assert np.all(expected_permutation == result)


def test_equation_variable_names(dof_manager: DofManager, model: pp.PorePyModel):
    """This is a simple regression test ensuring that `equation_names` and
    `variable_names` don't raise.

    """
    equation_names = dof_manager.equation_names(model)
    assert all(isinstance(name, str) for name in equation_names)

    variable_names = dof_manager.variable_names(model)
    assert all(isinstance(name, str) for name in variable_names)
