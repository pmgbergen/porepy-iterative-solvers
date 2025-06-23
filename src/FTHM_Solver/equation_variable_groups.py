from abc import ABC, abstractmethod
import porepy as pp
from enum import Enum


class EquationNames(Enum):
    """Enum for the names of the equations in the model."""

    MASS_BALANCE = "mass_balance_equation"
    MASS_BALANCE_MATRIX = "mass_balance_equation"
    MASS_BALANCE_FRACTURES = "mass_balance_equation"
    MASS_BALANCE_INTERSECTIONS = "mass_balance_equation"
    ENERGY_BALANCE = "energy_balance_equation"
    ENERGY_BALANCE_MATRIX = "energy_balance_equation"
    ENERGY_BALANCE_FRACTURES = "energy_balance_equation"
    ENERGY_BALANCE_INTERSECTIONS = "energy_balance_equation"
    INTERFACE_DARCY_FLUX = "interface_darcy_flux_equation"

    INTERFACE_ENTHALPY_FLUX = "interface_enthalpy_flux_equation"
    INTERFACE_FOURIER_FLUX = "interface_fourier_flux_equation"

    MECHANICS = "momentum_balance_equation"
    INTERFACE_FORCE_BALANCE = "interface_force_balance_equation"
    CONTACT = "contact_mechanics_equation"
    CONTACT_NORMAL = "normal_fracture_deformation_equation"
    CONTACT_TANGENTIAL = "tangential_fracture_deformation_equation"


class AbstractGroup(ABC):
    """
    Abstract class for defining a group of equations and variables. This serves two
    purposes:
        1. To define pairs of equations and variables that should be grouped together,
           and thereby define the diagonal blocks of the linear system.
        2. To define groups of equations that will be treated together by the iterative
           solver. The can be used to group equations of the same type (e.g., mass
           balance) on different subdomains, or to group equations of different type,
           but that still should be solved together.

    """

    @abstractmethod
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        pass

    @abstractmethod
    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        pass

    @abstractmethod
    def equation_names(self, model) -> list[str]:
        pass

    @abstractmethod
    def variable_names(self, model) -> list[str]:
        pass


def _split_subdomains_by_dimension(model: pp.PorePyModel):
    matrix_subdomains = model.mdg.subdomains(dim=model.nd)
    fracture_subdomains = model.mdg.subdomains(dim=model.nd - 1)
    intersection_subdomains = [
        sd for sd in model.mdg.subdomains() if sd.dim < model.nd - 1
    ]
    return matrix_subdomains, fracture_subdomains, intersection_subdomains


class MassBalanceGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains()
        return [[(EquationNames.MASS_BALANCE.value, subdomains)]]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [[model.pressure(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.MASS_BALANCE.value]

    def variable_names(self, model) -> list[str]:
        return [model.pressure_variable]


class MassBalanceDimSplitGroup(AbstractGroup):
    """Group for the mass balance equation, with matrix, fractures and intersections
    split into different groups. This is needed for fixed-stress type preconditioners,
    where the stabilization term differs according to the dimension of the subdomains.
    """

    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [(EquationNames.MASS_BALANCE_MATRIX.value, matrix_subdomains)],
            [(EquationNames.MASS_BALANCE_FRACTURES.value, fracture_subdomains)],
            [
                (
                    EquationNames.MASS_BALANCE_INTERSECTIONS.value,
                    intersection_subdomains,
                )
            ],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [model.pressure(matrix_subdomains)],
            [model.pressure(fracture_subdomains)],
            [model.pressure(intersection_subdomains)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.MASS_BALANCE_MATRIX.value,
            EquationNames.MASS_BALANCE_FRACTURES.value,
            EquationNames.MASS_BALANCE_INTERSECTIONS.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.pressure_variable + "_matrix",
            model.pressure_variable + "_fractures",
            model.pressure_variable + "_intersections",
        ]


class EnergyBalanceDimSplitGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [(EquationNames.ENERGY_BALANCE_MATRIX.value, matrix_subdomains)],
            [(EquationNames.ENERGY_BALANCE_FRACTURES.value, fracture_subdomains)],
            [
                (
                    EquationNames.ENERGY_BALANCE_INTERSECTIONS.value,
                    intersection_subdomains,
                )
            ],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [model.temperature(matrix_subdomains)],
            [model.temperature(fracture_subdomains)],
            [model.temperature(intersection_subdomains)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.ENERGY_BALANCE_MATRIX.value,
            EquationNames.ENERGY_BALANCE_FRACTURES.value,
            EquationNames.ENERGY_BALANCE_INTERSECTIONS.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.temperature_variable + "_matrix",
            model.temperature_variable + "_fractures",
            model.temperature_variable + "_intersections",
        ]


class InterfaceDarcyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [[(EquationNames.INTERFACE_DARCY_FLUX.value, interfaces)]]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [[model.interface_darcy_flux(interfaces)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.INTERFACE_DARCY_FLUX.value]

    def variable_names(self, model) -> list[str]:
        return [model.interface_darcy_flux_variable]


class InterfaceEnthalpyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [
            [(EquationNames.INTERFACE_ENTHALPY_FLUX.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_enthalpy_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_ENTHALPY_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_enthalpy_flux_variable,
        ]


class InterfaceFourierFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [
            [(EquationNames.INTERFACE_FOURIER_FLUX.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_fourier_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_FOURIER_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_fourier_flux_variable,
        ]


class InterfaceMassEnergyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [
            [(EquationNames.INTERFACE_ENTHALPY_FLUX.value, interfaces)],
            [(EquationNames.INTERFACE_FOURIER_FLUX.value, interfaces)],
            [(EquationNames.INTERFACE_DARCY_FLUX.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_enthalpy_flux(interfaces)],
            [model.interface_fourier_flux(interfaces)],
            [model.interface_darcy_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_ENTHALPY_FLUX.value,
            EquationNames.INTERFACE_FOURIER_FLUX.value,
            EquationNames.INTERFACE_DARCY_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_enthalpy_flux_variable,
            model.interface_fourier_flux_variable,
            model.interface_darcy_flux_variable,
        ]


class MechanicsGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)

        # Define two groups of equations, one for momentum balance in the matrix and one
        # for force balance on the highest-dimensional interfaces. The mechanics
        # preconditioner will treat these groups jointly.
        return [
            [(EquationNames.MECHANICS.value, subdomains)],
            [(EquationNames.INTERFACE_FORCE_BALANCE.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)

        # Define two groups of variables, one for the displacement in the matrix and one
        # for the interface displacement.
        return [
            [model.displacement(subdomains)],
            [model.interface_displacement(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.MECHANICS.value,
            EquationNames.INTERFACE_FORCE_BALANCE.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [model.displacement_variable, model.interface_displacement_variable]


class ContactGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        # Define a single group of equations to be solved together: The normal and
        # tangential deformation equations for the contact mechanics.
        return [
            [
                (EquationNames.CONTACT_NORMAL.value, subdomains),
                (EquationNames.CONTACT_TANGENTIAL.value, subdomains),
            ]
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        # There is a single group of variables for the contact mechanics, which is the
        # contact traction.
        return [[model.contact_traction(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.CONTACT.value]

    def variable_names(self, model) -> list[str]:
        return [model.contact_traction_variable]
