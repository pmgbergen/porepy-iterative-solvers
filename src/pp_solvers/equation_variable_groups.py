from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import porepy as pp
from porepy.models.energy_balance import TotalEnergyBalanceEquations
from porepy.models.fluid_mass_balance import FluidMassBalanceEquations
from porepy.numerics.ad.operators import MixedDimensionalVariable


class EquationNames(Enum):
    """Enum for the names of the equations in the model."""

    MASS_BALANCE = FluidMassBalanceEquations.primary_equation_name()
    ENERGY_BALANCE = TotalEnergyBalanceEquations.primary_equation_name()
    INTERFACE_DARCY_FLUX = "interface_darcy_flux_equation"

    INTERFACE_ENTHALPY_FLUX = "interface_enthalpy_flux_equation"
    INTERFACE_FOURIER_FLUX = "interface_fourier_flux_equation"

    MECHANICS = "momentum_balance_equation"
    INTERFACE_FORCE_BALANCE = "interface_force_balance_equation"
    CONTACT = "contact_mechanics_equation"
    CONTACT_NORMAL = "normal_fracture_deformation_equation"
    CONTACT_TANGENTIAL = "tangential_fracture_deformation_equation"

    WELL_FLUX = "well_flux_equation"
    WELL_ENTHALPY_FLUX = "well_enthalpy_flux_equation"


@dataclass
class EquationOnDomains:
    """A PorePy equation defined on a collection of domains."""

    name: str
    """Should be identical to the PorePy equation name."""
    domains: list[pp.GridLike]
    """A list of domains of definition for the given equation."""


class EquationVariableGroup(ABC):
    """A base class for all groups. This represents a diagonal submatrix in a block
    matrix, thus the number of DoFs for the equation and the variable should match.

    Despite the group objects are instantiated (e.g. `MassBalancePressureGroup()`), they
    are internally treated as singletones: we treat groups with equal class names as
    equal. Therefore, the group should not have any conditional behavior, based on its
    or model's state.

    Equation and variable names defined should not necessarily match the PorePy equation
    and variable names. They are used for debugging and diagnostics, thus should have
    human readable names, e.g. "mass_balance on fractures" or
    "constrained pressure on injection wells".

    """

    @abstractmethod
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        pass

    @abstractmethod
    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        pass

    @abstractmethod
    def equation_name(self, model: pp.PorePyModel) -> str:
        pass

    @abstractmethod
    def variable_name(self, model: pp.PorePyModel) -> str:
        pass

    def __eq__(self, other: "EquationVariableGroup") -> bool:
        if not isinstance(other, EquationVariableGroup):
            return False
        # Assuming groups are immutable! Probably this must be stated in the class doc.
        return self.__class__ == other.__class__

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __hash__(self) -> int:
        return hash(self.__class__)


class InterfaceDarcyFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_DARCY_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.interface_darcy_flux(model.mdg.interfaces())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.INTERFACE_DARCY_FLUX.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.interface_darcy_flux_variable


class InterfaceEnthalpyFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_ENTHALPY_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.interface_enthalpy_flux(model.mdg.interfaces())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.INTERFACE_ENTHALPY_FLUX.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.interface_enthalpy_flux_variable


class InterfaceFourierFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_FOURIER_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.interface_fourier_flux(model.mdg.interfaces())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.INTERFACE_FOURIER_FLUX.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.interface_fourier_flux_variable


class WellFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.WELL_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.well_flux(model.mdg.interfaces())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.WELL_FLUX.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.well_flux_variable


class WellEnthalpyFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.WELL_ENTHALPY_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.well_enthalpy_flux(model.mdg.interfaces())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.WELL_ENTHALPY_FLUX.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.well_enthalpy_flux_variable


class InterfaceForceBalanceGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_FORCE_BALANCE.value
        return EquationOnDomains(
            name=name, domains=model.mdg.interfaces(dim=model.nd - 1)
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        interfaces = model.mdg.interfaces(dim=model.nd - 1)
        return model.interface_displacement(interfaces)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.INTERFACE_FORCE_BALANCE.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.interface_displacement_variable


class MechanicsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MECHANICS.value,
            domains=model.mdg.subdomains(dim=model.nd),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        subdomains = model.mdg.subdomains(dim=model.nd)
        return model.displacement(subdomains)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MECHANICS.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.displacement_variable


class ContactMechanicsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        # PorePy has no single contact mechanics equation: it has two: normal and
        # tangential. However, we treat them as one in the linear solver context. This
        # is an exception, which is manually treated in the DofManager.
        return EquationOnDomains(
            name=EquationNames.CONTACT.value,
            domains=model.mdg.subdomains(dim=model.nd - 1),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        return model.contact_traction(subdomains)

    def equation_name(self, model) -> str:
        return EquationNames.CONTACT.value

    def variable_name(self, model) -> str:
        return model.contact_traction_variable


class MassBalancePressureMatrixGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value,
            domains=model.mdg.subdomains(dim=model.nd),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.pressure(model.mdg.subdomains(dim=model.nd))

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value + " (matrix)"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable + " (matrix)"


class MassBalancePressureFracturesGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value,
            domains=model.mdg.subdomains(dim=model.nd - 1),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.pressure(model.mdg.subdomains(dim=model.nd - 1))

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value + " (fractures)"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable + " (fractures)"


class MassBalancePressureIntersectionsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        intersections = [
            sd
            for dim in reversed(range(0, model.nd - 1))
            for sd in model.mdg.subdomains(dim=dim)
        ]
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value,
            domains=intersections,
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        intersections = [
            sd
            for dim in reversed(range(0, model.nd - 1))
            for sd in model.mdg.subdomains(dim=dim)
        ]
        return model.pressure(intersections)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value + " (intersections)"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable + " (intersections)"


class MassBalancePressureGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value, domains=model.mdg.subdomains()
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.pressure(model.mdg.subdomains())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable


class EnergyBalanceTemperatureGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.ENERGY_BALANCE.value, domains=model.mdg.subdomains()
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.temperature(model.mdg.subdomains())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.ENERGY_BALANCE.value

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.temperature_variable
