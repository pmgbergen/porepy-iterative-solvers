from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import porepy as pp
from porepy.models.fluid_mass_balance import FluidMassBalanceEquations
from porepy.models.energy_balance import TotalEnergyBalanceEquations
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
    name: str
    domains: list[pp.GridLike]


class EquationVariableGroup(ABC):
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

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        return model.interface_displacement(subdomains)

    def equation_names(self, model) -> list[str]:
        return EquationNames.MECHANICS.value

    def variable_names(self, model) -> list[str]:
        return model.displacement_variable

class ContactMechanicsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.CONTACT.value,
            domains=model.mdg.subdomains(dim=model.nd - 1),
        )

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        return model.contact_traction(subdomains)

    def equation_names(self, model) -> list[str]:
        return EquationNames.CONTACT.value

    def variable_names(self, model) -> list[str]:
        return model.contact_traction_variable


# MARK: Should not be here


class MassBalancePressureGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        production_wells, no_production_wells = model._filter_wells(
            model.mdg.subdomains(), "production"
        )
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE, domains=no_production_wells
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        production_wells, no_production_wells = model._filter_wells(
            model.mdg.subdomains(), "production"
        )
        return model.pressure(no_production_wells)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return "mass_balance"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return "pressure"


class EnergyBalanceEnthalpyGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.ENERGY_BALANCE
        injection_wells, no_injection_wells = model._filter_wells(
            model.mdg.subdomains(), "injection"
        )
        return EquationOnDomains(name=name, domains=no_injection_wells)

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        injection_wells, no_injection_wells = model._filter_wells(
            model.mdg.subdomains(), "injection"
        )
        return model.enthalpy(no_injection_wells)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return "energy_balance"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return "entalpy"


class ProductionPressureConstraintGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = "production_pressure_constraint"
        production_wells, no_production_wells = model._filter_wells(
            model.mdg.subdomains(), "production"
        )
        return EquationOnDomains(name=name, domains=production_wells)

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        production_wells, no_production_wells = model._filter_wells(
            model.mdg.subdomains(), "production"
        )
        return model.pressure(production_wells)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return "production_pressure_constraint"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return "pressure_constraint"


class InjectionTemperatureConstraintGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = "injection_temperature_constraint"
        injection_wells, no_injection_wells = model._filter_wells(
            model.mdg.subdomains(), "injection"
        )
        return EquationOnDomains(name=name, domains=injection_wells)

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        injection_wells, no_injection_wells = model._filter_wells(
            model.mdg.subdomains(), "injection"
        )
        return model.enthalpy(injection_wells)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return "injection_temperature_constraint"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return "enthalpy_constraint"


class ComponentMassBalanceCO2Group(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = "component_mass_balance_equation_CO2"
        return EquationOnDomains(name=name, domains=model.mdg.subdomains())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.fluid.components[1].fraction(model.mdg.subdomains())

    def equation_name(self, model: pp.PorePyModel) -> str:
        return "component_mass_balance_equation_CO2"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return "z_CO2"
