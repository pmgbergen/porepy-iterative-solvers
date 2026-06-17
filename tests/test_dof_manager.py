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

from pp_solvers.block_linear_system import concatenate_dof_indices
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EnergyBalanceTemperatureGroup,
    EquationOnDomains,
    EquationVariableGroup,
    InterfaceForceBalanceGroup,
    MassBalancePressureFracturesGroup,
    MassBalancePressureIntersectionsGroup,
    MassBalancePressureMatrixGroup,
)
from pp_solvers.preconditioners import LinearSolverConfiguration
from pp_solvers.solver_mixin import default_preconditioner_factory


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
def solver_configuration(model: pp.PorePyModel) -> LinearSolverConfiguration:
    return default_preconditioner_factory(model)()


@pytest.fixture(scope="module")
def dof_manager(
    model: pp.PorePyModel, solver_configuration: LinearSolverConfiguration
) -> DofManager:
    return DofManager(model, solver_configuration.solver.groups)


@pytest.fixture(scope="module")
def expected_num_dofs_in_groups(
    model: pp.PorePyModel, model_kind: str
) -> dict[str, int]:
    """The expected values for the tests. Returns a mapping of a group name to the
    number of DoFs of the corresponding groups."""

    # Constructing relevant sets of domains.
    all_subdomains = model.mdg.subdomains()
    fractures = model.mdg.subdomains(dim=model.nd - 1)
    all_interfaces_codim_1 = model.mdg.interfaces(codim=1)
    all_interfaces_codim_2 = model.mdg.interfaces(codim=2)
    intersections = [
        domain
        for x in range(0, model.nd - 1)
        for domain in (model.mdg.subdomains(dim=x))
    ]
    interfaces_ambient_frac = model.mdg.interfaces(dim=model.nd - 1)
    porous_media_subdomains = model.mdg.subdomains(dim=model.nd)

    # And counting the number of cells in each.
    all_subdomains = sum([x.num_cells for x in all_subdomains])
    fractures = sum([x.num_cells for x in fractures])
    all_interfaces_codim_1 = sum([x.num_cells for x in all_interfaces_codim_1])
    all_interfaces_codim_2 = sum([x.num_cells for x in all_interfaces_codim_2])
    intersections = sum([x.num_cells for x in intersections])
    interfaces_ambient_frac = sum([x.num_cells for x in interfaces_ambient_frac])
    porous_media_subdomains = sum([x.num_cells for x in porous_media_subdomains])

    nd = model.nd

    # The keys are used for referring to specific groups within this file. They are not
    # meant to be consistent with equation names or anything else.
    match model_kind:
        case "flow":
            return {
                "intf_fluid_flux": all_interfaces_codim_1,
                "well_flux_equation": all_interfaces_codim_2,
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
                "intf_fluid_flux": all_interfaces_codim_1,
                "intf_heat_advection": all_interfaces_codim_1,
                "intf_heat_diffusion": all_interfaces_codim_1,
                "well_flux_equation": all_interfaces_codim_2,
                "well_enthalpy_flux_equation": all_interfaces_codim_2,
                # Energy balance ambient, fractures, lower (together).
                "energy_balance": all_subdomains,
                # Mass balance ambient, fractures, lower (separately).
                "mass_balance_ambient": porous_media_subdomains,
                "mass_balance_fractures": fractures,
                "mass_balance_intersections": intersections,
            }
        case "HM":
            return {
                "contact": fractures * nd,
                "intf_fluid_flux": all_interfaces_codim_1,
                "well_flux_equation": all_interfaces_codim_2,
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
                # Interfaces.
                "intf_fluid_flux": all_interfaces_codim_1,
                "intf_heat_advection": all_interfaces_codim_1,
                "intf_heat_diffusion": all_interfaces_codim_1,
                "well_flux_equation": all_interfaces_codim_2,
                "well_enthalpy_flux_equation": all_interfaces_codim_2,
                # Elasticity and force balance.
                "momentum_balance": porous_media_subdomains * nd,
                "intf_force_balance": interfaces_ambient_frac * nd,
                # Energy balance ambient, fractures, lower (together).
                "energy_balance": all_subdomains,
                # Mass balance ambient, fractures, lower (separately).
                "mass_balance_ambient": porous_media_subdomains,
                "mass_balance_fractures": fractures,
                "mass_balance_intersections": intersections,
            }
        case default:
            raise ValueError(default)


# MARK: Tests begin here.


def test_eq_var_dofs(
    dof_manager: DofManager,
    expected_num_dofs_in_groups: dict[str, int],
):
    """Tests methods `eq_dofs` and `var_dofs`."""
    eq_dofs = dof_manager.eq_dofs()
    var_dofs = dof_manager.var_dofs()

    # The number of dof arrays must be as the number of expected values.
    assert (
        len(eq_dofs)
        == len(var_dofs)
        == len(expected_num_dofs_in_groups)
        == len(dof_manager.groups())
    )

    # eq_dofs and var_dofs should include all dofs of the problem, no duplicates, no
    # values out of range.
    total_num_dofs = sum([array.size for array in dof_manager.eq_dofs()])
    assert np.all(
        np.arange(total_num_dofs) == np.sort(concatenate_dof_indices(eq_dofs))
    )
    assert np.all(
        np.arange(total_num_dofs) == np.sort(concatenate_dof_indices(var_dofs))
    )

    # We check the number of dofs in each group.
    for i, expected_num_dofs in enumerate(expected_num_dofs_in_groups.values()):
        # We take dofs of all subdomains/interfaces that are in the same group.
        assert eq_dofs[i].size == expected_num_dofs
        assert var_dofs[i].size == expected_num_dofs


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {"keys": ["energy_balance"], "groups": [EnergyBalanceTemperatureGroup()]},
            id="energy_balance",
        ),
        pytest.param(
            {"keys": ["intf_force_balance"], "groups": [InterfaceForceBalanceGroup()]},
            id="u_intf",
        ),
        pytest.param(
            {"keys": ["contact"], "groups": [ContactMechanicsGroup()]}, id="contact"
        ),
        pytest.param(
            {
                "keys": [
                    "mass_balance_fractures",
                    "mass_balance_ambient",
                    "mass_balance_intersections",
                ],
                "groups": [
                    MassBalancePressureFracturesGroup(),
                    MassBalancePressureMatrixGroup(),
                    MassBalancePressureIntersectionsGroup(),
                ],
            },
            id="mass_balance",
        ),
        pytest.param(
            {
                "keys": ["contact", "energy_balance"],
                "groups": [ContactMechanicsGroup(), EnergyBalanceTemperatureGroup()],
            },
            id="contact_and_energy",
        ),
    ],
)
def test_indices_of_groups(
    dof_manager: DofManager, expected_num_dofs_in_groups: dict[str, int], params: dict
):
    """Checks that `DofManager.indices_of_groups` returns expected groups or raises."""
    keys: list[str] = params["keys"]
    groups: list[EquationVariableGroup] = params["groups"]

    try:
        expected_groups = [
            list(expected_num_dofs_in_groups.keys()).index(key) for key in keys
        ]
        should_raise = False
    except ValueError:
        # If at least one not found, we expect the method to raise ValueError.
        expected_groups = []
        should_raise = True

    if should_raise:
        with pytest.raises(ValueError):
            dof_manager.indices_of_groups(groups=groups)
    else:
        assert dof_manager.indices_of_groups(groups=groups) == expected_groups


def test_permute_contact_dofs(dof_manager: DofManager):
    """This tests the method `_permute_contact_dofs`, assuming that `eq_dofs` and
    `indices_of_groups` work correctly, as tested above.

    """
    assert dof_manager.model.nd == 2, "This test assumes a 2D problem."

    # Checking if there is contact mechanics in the model.
    try:
        contact_group = dof_manager.indices_of_groups([ContactMechanicsGroup()])[0]
    except ValueError:
        return  # Skipping this test if no contact group.

    dofs_in_contact_group = dof_manager.eq_dofs()[contact_group]
    if len(dofs_in_contact_group) == 0:
        return  # Skipping this test if the contact group is formally present but empty.

    # DoFs in the contact groups should be already permuted in DofManager.__init__.
    # We check that permuting them back will give an increasing order.
    np.testing.assert_equal(
        np.vstack([dofs_in_contact_group[::2], dofs_in_contact_group[1::2]]).ravel(),
        np.arange(dofs_in_contact_group[0], dofs_in_contact_group[-1] + 1),
    )


def test_equation_variable_names(dof_manager: DofManager):
    """This is a simple regression test ensuring that `equation_names` and
    `variable_names` don't raise.

    """
    equation_names = dof_manager.equation_names()
    assert all(isinstance(name, str) for name in equation_names)

    variable_names = dof_manager.variable_names()
    assert all(isinstance(name, str) for name in variable_names)


class DuplicatingGroup(EquationVariableGroup):
    """This group has the same equation as the `EnergyBalanceTemperatureGroup` group
    and same variable as the `MassBalancePressureMatrixGroup` group.

    """

    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EnergyBalanceTemperatureGroup().equation_group(model=model)

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return MassBalancePressureMatrixGroup().variable_group(model=model)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return "something"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return "something"


def test_duplicating_equations(model: pp.PorePyModel, model_kind: str):
    if model_kind not in ["TH", "THM"]:
        return  # Skip this test for other models.
    groups = [EnergyBalanceTemperatureGroup(), DuplicatingGroup()]
    with pytest.raises(ValueError):
        _ = DofManager(model=model, groups=groups)


def test_duplicating_variables(model: pp.PorePyModel, model_kind: str):
    if model_kind not in ["TH", "THM"]:
        return  # Skip this test for other models.
    groups = [MassBalancePressureMatrixGroup(), DuplicatingGroup()]
    with pytest.raises(ValueError):
        _ = DofManager(model=model, groups=groups)
