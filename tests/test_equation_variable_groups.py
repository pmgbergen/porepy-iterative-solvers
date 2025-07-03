import porepy as pp
import pytest
from pp_solvers.equation_variable_groups import EquationNames


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


def test_all_equation_names_are_in_porepy_system():
    """Create a Thermoporomechanics model, which contains all equations of relevance
    for the PorePy models, and check that all equation names in the EquationNames Enum
    are present in the PorePy equation system.

    This test will fail when we introduce solvers for compositional flow.
    """
    model = pp.Thermoporomechanics()
    model.prepare_simulation()
    porepy_names = model.equation_system.equations.keys()
    for enum_name in EquationNames:
        # Special case: `contact_mechanics_equation` is not a primary equation in PorePy
        # but it is used to denote the combined tangential and normal contact equations.
        # Therefore ignore it in this test.
        if enum_name.value != "contact_mechanics_equation":
            assert enum_name.value in porepy_names, (
                f"{enum_name.value} not found in PorePy equation names"
            )
