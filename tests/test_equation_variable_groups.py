import porepy as pp
import pytest

from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EnergyBalanceTemperatureGroup,
    EquationNames,
    EquationVariableGroup,
    InterfaceDarcyFluxGroup,
    InterfaceEnthalpyFluxGroup,
    InterfaceForceBalanceGroup,
    InterfaceFourierFluxGroup,
    MassBalancePressureFracturesGroup,
    MassBalancePressureGroup,
    MassBalancePressureIntersectionsGroup,
    MassBalancePressureMatrixGroup,
    MechanicsGroup,
    WellEnthalpyFluxGroup,
    WellFluxGroup,
)


@pytest.fixture(scope="module")
def model() -> pp.PorePyModel:
    model = pp.Thermoporomechanics()
    model.prepare_simulation()
    return model


@pytest.mark.parametrize(
    "pp_model, enum_name",
    [
        (pp.SinglePhaseFlow, EquationNames.MASS_BALANCE.value),
        (
            pp.energy_balance.TotalEnergyBalanceEquations,
            EquationNames.ENERGY_BALANCE.value,
        ),
    ],
)
def test_primary_equation_names(pp_model, enum_name):
    """
    Test that the equation names, set in the EquationNames Enum, correspond to the
    primary equations defined in PorePy models.

    A failure of this test would signify that the EquationNames Enum is not in sync with
    the PorePy models.

    TODO: Keep this updated as more PorePy models are assigned primary equations names.
    """
    porepy_name = pp_model.primary_equation_name()

    assert enum_name in porepy_name, f"{enum_name} not found in primary equations"


def test_all_equation_names_are_in_porepy_system(model: pp.PorePyModel):
    """Create a Thermoporomechanics model, which contains all equations of relevance
    for the PorePy models, and check that all equation names in the EquationNames Enum
    are present in the PorePy equation system.

    This test will fail when we introduce solvers for compositional flow.
    """

    porepy_names = model.equation_system.equations.keys()
    for enum_name in EquationNames:
        # Special case: `contact_mechanics_equation` is not a primary equation in PorePy
        # but it is used to denote the combined tangential and normal contact equations.
        # Therefore ignore it in this test.
        if enum_name.value != "contact_mechanics_equation":
            assert enum_name.value in porepy_names, (
                f"{enum_name.value} not found in PorePy equation names"
            )


@pytest.mark.parametrize(
    "group_class",
    [
        ContactMechanicsGroup,
        EnergyBalanceTemperatureGroup,
        InterfaceDarcyFluxGroup,
        InterfaceEnthalpyFluxGroup,
        InterfaceForceBalanceGroup,
        InterfaceFourierFluxGroup,
        MassBalancePressureFracturesGroup,
        MassBalancePressureGroup,
        MassBalancePressureIntersectionsGroup,
        MassBalancePressureMatrixGroup,
        MechanicsGroup,
        WellEnthalpyFluxGroup,
        WellFluxGroup,
    ],
)
def test_equation_variable_group(
    group_class: type[EquationVariableGroup], model: pp.PorePyModel
):
    group = group_class()

    # Different objects of the same group should be treated as equal.
    assert group == group_class()

    # equation_name and variable_name should return some string (not necesserily
    # equivalent to a name in porepy).
    assert isinstance(group.equation_name(model), str)
    assert isinstance(group.variable_name(model), str)

    variable = group.variable_group(model)
    assert isinstance(variable, pp.ad.MixedDimensionalVariable)

    equation_on_domains = group.equation_group(model)

    # Each domain should be either a porepy subdomain or interface grid.
    for domain in equation_on_domains.domains:
        assert isinstance(domain, (pp.Grid, pp.MortarGrid))


    if group_class is ContactMechanicsGroup:
        # ContactMechanicsGroup is the known edge case, handled by a special logic, so
        # we skip the rest of the test.
        return

    # Equation name should be identifiable by porepy equation system.
    assert equation_on_domains.name in model.equation_system.equations
