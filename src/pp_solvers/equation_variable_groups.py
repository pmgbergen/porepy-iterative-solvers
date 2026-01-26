from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

import porepy as pp


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

    WELL_FLUX = "well_flux_equation"
    WELL_ENTHALPY_FLUX = "well_enthalpy_flux_equation"


@dataclass
class EquationGroupItem:
    name: str
    items: list  # or List[Any] for more type safety

    def __iter__(self):
        yield self.name
        yield self.items


@dataclass
class EquationGroup:
    items: List[EquationGroupItem]

    def __iter__(self):
        return iter(self.items)


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
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
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
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains()
        return [
            EquationGroup(
                [
                    EquationGroupItem(EquationNames.MASS_BALANCE.value, subdomains),
                    EquationGroupItem("production_pressure_constraint", subdomains),
                ],
            )
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [[model.pressure(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.MASS_BALANCE.value]

    def variable_names(self, model) -> list[str]:
        return [model.pressure_variable]


class ComponentMassBalanceCO2Group(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains()
        return [
            EquationGroup(
                [EquationGroupItem("component_mass_balance_equation_CO2", subdomains)]
            )
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [[model.fluid.components[1].fraction(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return ["component_mass_balance_equation_CO2"]

    def variable_names(self, model) -> list[str]:
        return ["z_CO2"]


class EnthalpyGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains()
        return [
            EquationGroup([EquationGroupItem("energy_balance_equation", subdomains)])
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [[model.enthalpy(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return ["energy_balance_equation"]

    def variable_names(self, model) -> list[str]:
        return ["enthalpy"]


class MassAndEnthalpyGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains()
        return [
            EquationGroup(
                [
                    EquationGroupItem("mass_balance_equation", subdomains),
                    EquationGroupItem("production_pressure_constraint", subdomains),
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem("energy_balance_equation", subdomains),
                    EquationGroupItem("injection_temperature_constraint", subdomains),
                ]
            ),
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [
            [model.pressure(subdomains)],
            [model.enthalpy(subdomains)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            "mass_balance_equation",
            "energy_balance_equation",
        ]

    def variable_names(self, model) -> list[str]:
        return [
            "pressure",
            "enthalpy",
        ]


class EnthalpyAndComponentGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains()

        return [
            EquationGroup(
                [
                    EquationGroupItem("energy_balance_equation", subdomains),
                    EquationGroupItem("injection_temperature_constraint", subdomains),
                ]
            ),
            EquationGroup(
                [EquationGroupItem("component_mass_balance_equation_CO2", subdomains)]
            ),
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [
            [model.enthalpy(subdomains)],
            [model.fluid.components[1].fraction(subdomains)],
        ]

    def equation_names(self, model) -> list[str]:
        return ["energy_balance_equation", "component_mass_balance_equation_CO2"]

    def variable_names(self, model) -> list[str]:
        return ["enthalpy", "z_CO2"]


class MassBalanceDimSplitGroup(AbstractGroup):
    """Group for the mass balance equation, with matrix, fractures and intersections
    split into different groups. This is needed for fixed-stress type preconditioners,
    where the stabilization term differs according to the dimension of the subdomains.
    """

    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.MASS_BALANCE_MATRIX.value, matrix_subdomains
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.MASS_BALANCE_FRACTURES.value, fracture_subdomains
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.MASS_BALANCE_INTERSECTIONS.value,
                        intersection_subdomains,
                    )
                ]
            ),
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
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.ENERGY_BALANCE_MATRIX.value, matrix_subdomains
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.ENERGY_BALANCE_FRACTURES.value,
                        fracture_subdomains,
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.ENERGY_BALANCE_INTERSECTIONS.value,
                        intersection_subdomains,
                    )
                ]
            ),
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
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        interfaces = model.mdg.interfaces()
        return [
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_DARCY_FLUX.value, interfaces
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(EquationNames.WELL_FLUX.value, interfaces),
                ]
            ),
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_darcy_flux(interfaces)],
            [model.well_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_DARCY_FLUX.value,
            EquationNames.WELL_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_darcy_flux_variable,
            model.well_flux_variable,
        ]


class InterfaceEnthalpyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        interfaces = model.mdg.interfaces()
        return [
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_ENTHALPY_FLUX.value, interfaces
                    )
                ]
            )
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
    # YZ: Not used anywhere.

    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        interfaces = model.mdg.interfaces()
        return [
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_FOURIER_FLUX.value, interfaces
                    )
                ]
            )
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
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        interfaces = model.mdg.interfaces()
        return [
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_ENTHALPY_FLUX.value, interfaces
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_FOURIER_FLUX.value, interfaces
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_DARCY_FLUX.value, interfaces
                    )
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(EquationNames.WELL_FLUX.value, interfaces),
                ]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.WELL_ENTHALPY_FLUX.value, interfaces
                    ),
                ]
            ),
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_enthalpy_flux(interfaces)],
            [model.interface_fourier_flux(interfaces)],
            [model.interface_darcy_flux(interfaces)],
            [model.well_flux(interfaces)],
            [model.well_enthalpy_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_ENTHALPY_FLUX.value,
            EquationNames.INTERFACE_FOURIER_FLUX.value,
            EquationNames.INTERFACE_DARCY_FLUX.value,
            EquationNames.WELL_FLUX.value,
            EquationNames.WELL_ENTHALPY_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_enthalpy_flux_variable,
            model.interface_fourier_flux_variable,
            model.interface_darcy_flux_variable,
            model.well_flux_variable,
            model.well_enthalpy_flux_variable,
        ]


class MechanicsGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)
        return [
            EquationGroup(
                [EquationGroupItem(EquationNames.MECHANICS.value, subdomains)]
            ),
            EquationGroup(
                [
                    EquationGroupItem(
                        EquationNames.INTERFACE_FORCE_BALANCE.value, interfaces
                    )
                ]
            ),
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
    def equation_groups(self, model: pp.PorePyModel) -> list[EquationGroup]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        return [
            EquationGroup(
                [
                    EquationGroupItem(EquationNames.CONTACT_NORMAL.value, subdomains),
                    EquationGroupItem(
                        EquationNames.CONTACT_TANGENTIAL.value, subdomains
                    ),
                ]
            )
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
