"""Tag-based equation and variable groups for block linear solvers."""

from dataclasses import dataclass, replace

import porepy as pp
from porepy.numerics.solvers import (
    DefaultEquationTags,
    DefaultVariableTags,
    DomainFilter,
    EquationTag,
    OnAmbientDimension,
    OnFractures,
    OnLowerDimensions,
    VariableTag,
)

__all__ = [
    "DefaultEquationVariableGroups",
    "EquationVariableGroup",
    "CONTACT_MECHANICS_EQUATION_TAG",
]


CONTACT_MECHANICS_EQUATION_TAG = EquationTag("contact_mechanics")
"""Contact mechanics requires special treatment to establish a square block in the
Jacobian matrix, since PorePy defines it by a single variable (vector t_x, t_y, t_z
in 3D, where t is fracture traction), and two equations: normal and tangential
complimentarity conditions. We define this special tag, recognized only within this
pp_solvers package. It expresses a combination of normal and tangential equations as a
single equation. See also the :class:`pp_solvers.dof_manager.DofManager` docstring.

"""


@dataclass(frozen=True)
class EquationVariableGroup:
    """Tags defining a submatrix on a diagonal of the Jacobian matrix.

    The number of DoFs for the equation and the variable should match.

    """

    equation_tag: EquationTag
    variable_tag: VariableTag

    def restricted(self, defined_on: DomainFilter) -> "EquationVariableGroup":
        """Construct a new group that consists of the same equation and variable, but
        with the provided domain of definition.

        Parameters:
            defined_on: The new domain of definition for both the equation and the
                variable. The old domain is discarded.

        Returns:
            The new instance of `EquationVariableGroup`.

        """
        equation_tag = replace(self.equation_tag, defined_on=defined_on)
        variable_tag = replace(self.variable_tag, defined_on=defined_on)
        return EquationVariableGroup(equation_tag, variable_tag)


class DefaultEquationVariableGroups:
    """A namespace for the default equation-variable groups that work with PorePy
    models. It is allowed to define your own groups for custom solvers or models.

    """

    # Mass balance
    mass_balance_pressure_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.mass_balance,
        variable_tag=DefaultVariableTags.pressure,
    )
    interface_darcy_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_darcy_flux,
        variable_tag=DefaultVariableTags.interface_darcy_flux,
    )
    well_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.well_flux,
        variable_tag=DefaultVariableTags.well_flux,
    )
    # Energy balance
    energy_balance_temperature_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.energy_balance,
        variable_tag=DefaultVariableTags.temperature,
    )
    interface_fourier_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_fourier_flux,
        variable_tag=DefaultVariableTags.interface_fourier_flux,
    )
    interface_enthalpy_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_enthalpy_flux,
        variable_tag=DefaultVariableTags.interface_enthalpy_flux,
    )
    well_enthalpy_flux_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.well_enthalpy_flux,
        variable_tag=DefaultVariableTags.well_enthalpy_flux,
    )
    # Momentum balance MPSA
    mechanics_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.momentum_balance,
        variable_tag=DefaultVariableTags.displacement,
    )
    # Momentum balance TPSA
    solid_mass_pressure_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.poromechanics_solid_mass,
        variable_tag=DefaultVariableTags.total_pressure,
    )
    angular_momentum_rotation_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.angular_momentum_balance,
        variable_tag=DefaultVariableTags.rotation_stress,
    )
    # Contact mechanics
    contact_mechanics_group = EquationVariableGroup(
        equation_tag=CONTACT_MECHANICS_EQUATION_TAG,
        variable_tag=DefaultVariableTags.contact_traction,
    )
    interface_force_balance_group = EquationVariableGroup(
        equation_tag=DefaultEquationTags.interface_force_balance,
        variable_tag=DefaultVariableTags.interface_displacement,
    )
    # Some common definitions on the restricted domains.
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
    ).restricted(OnLowerDimensions())
