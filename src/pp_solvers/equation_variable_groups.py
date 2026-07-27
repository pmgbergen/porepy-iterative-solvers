"""Tag-based equation and variable groups for block linear solvers."""

from dataclasses import dataclass, replace

import porepy as pp
from porepy.numerics.solvers import (
    OnAmbientDimension,
    OnFractures,
    OnInterfaces,
    DefaultEquationTags,
    DefaultVariableTags,
    EquationTag,
    VariableTag,
    DomainFilter,
)

__all__ = [
    "DefaultEquationVariableGroups",
    "EquationVariableGroup",
    "CONTACT_MECHANICS_EQUATION_TAG",
]


CONTACT_MECHANICS_EQUATION_TAG = EquationTag("contact_mechanics")
"""TODO YZ"""


@dataclass(frozen=True)
class EquationVariableGroup:
    """Tags defining a submatrix on a diagonal of the matrix.

    The number of DoFs forthe equation and the variable should match.

    This is a dataclass, because it is: (i) comparable, (ii) hashable and (iii)
    immutable.

    """

    equation_tag: EquationTag
    variable_tag: VariableTag

    def restricted(self, defined_on: DomainFilter) -> "EquationVariableGroup":
        equation_tag = replace(self.equation_tag, defined_on=defined_on)
        variable_tag = replace(self.variable_tag, defined_on=defined_on)
        return EquationVariableGroup(equation_tag, variable_tag)


class DefaultEquationVariableGroups:
    """TODO YZ"""

    interface_darcy_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_darcy_flux,
        variable_tag=DefaultVariableTags.interface_darcy_flux,
    )
    interface_enthalpy_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_enthalpy_flux,
        variable_tag=DefaultVariableTags.interface_enthalpy_flux,
    )
    interface_fourier_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_fourier_flux,
        variable_tag=DefaultVariableTags.interface_fourier_flux,
    )
    well_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.well_flux,
        variable_tag=DefaultVariableTags.well_flux,
    )
    well_enthalpy_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.well_enthalpy_flux,
        variable_tag=DefaultVariableTags.well_enthalpy_flux,
    )
    interface_force_balance_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_force_balance,
        variable_tag=DefaultVariableTags.interface_displacement,
    )
    mechanics_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.momentum_balance,
        variable_tag=DefaultVariableTags.displacement,
    )
    contact_mechanics_group = EquationVariableGroup(
        equation_tag=CONTACT_MECHANICS_EQUATION_TAG,
        variable_tag=DefaultVariableTags.contact_traction,
    )
    mass_balance_pressure_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.mass_balance,
        variable_tag=DefaultVariableTags.pressure,
    )
    energy_balance_temperature_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.energy_balance,
        variable_tag=DefaultVariableTags.temperature,
    )
    mass_balance_pressure_matrix_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.mass_balance,
        variable_tag=DefaultVariableTags.pressure,
    ).restricted(OnAmbientDimension())

    mass_balance_pressure_fractures_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.mass_balance,
        variable_tag=DefaultVariableTags.pressure,
    ).restricted(OnFractures())

    mass_balance_pressure_intersections_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.mass_balance,
        variable_tag=DefaultVariableTags.pressure,
    ).restricted(OnInterfaces())
