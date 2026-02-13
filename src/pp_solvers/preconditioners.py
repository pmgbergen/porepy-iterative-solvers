from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain

import numpy as np

from pp_solvers.block_linear_system import LinearSystemIndexer
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EnergyBalanceTemperatureGroup,
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
from pp_solvers.fixed_stress import make_fs_analytical_slow_new
from pp_solvers.petsc_solvers import PcPythonPermutation
from pp_solvers.petsc_utils import csr_to_petsc

__all__ = [
    # Add all preconditioners here.
    "CompositePreconditioner",
    "GMRES",
    "DiagonalInvertor",
    # Add all the factory functions here.
    "mass_balance_factory",
    "momentum_balance_factory",
    "hm_factory",
    "th_factory",
    "thm_factory",
]


def append_prefix_to_options(prefix: str, options: dict):
    return {f"{prefix}{key}": value for key, value in options.items()}


class PetscInvertor(ABC):
    @abstractmethod
    def petsc_options(self, prefix: str, tag: str, complement_tag: str) -> dict:
        pass

    def petsc_assembly_config(self, prefix: str, dof_manager: DofManager) -> dict:
        return {}


class DiagonalInvertor(PetscInvertor):
    def petsc_options(self, prefix: str, tag: str, complement_tag: str) -> dict:
        return append_prefix_to_options(
            prefix=prefix,
            options={
                "pc_fieldsplit_schur_precondition": "selfp",
            },
        )


class BlockDiagonalInvertor(PetscInvertor):
    def petsc_options(self, prefix: str, tag: str, complement_tag: str) -> dict:
        # YZ: This option "mat_schur_complement_ainv_type" applies to the PETSc object,
        # which represents the non-assembled Schur complement matrix. It tells it to use
        # the block-diagonal approximation when the Schur complement needs to be
        # assembled. This option applies not to the full "fieldsplit" context, but the
        # context of the complement, thus using the complement prefix.
        return append_prefix_to_options(
            prefix=prefix,
            options={
                "pc_fieldsplit_schur_precondition": "selfp",
                f"fieldsplit_{complement_tag}_mat_schur_complement_ainv_type": "blockdiag",
            },
        )


class FixedStressInvertor(PetscInvertor):
    def petsc_options(
        self,
        prefix: str,
        tag: str,
        complement_tag: str,
    ) -> dict:
        return append_prefix_to_options(
            prefix=prefix,
            options={
                "pc_fieldsplit_schur_precondition": "user",
            },
        )

    def petsc_assembly_config(self, prefix: str, dof_manager: DofManager) -> dict:
        flow_mat_group, flow_frac_group = dof_manager.indices_of_groups(
            [MassBalancePressureMatrixGroup(), MassBalancePressureFracturesGroup()]
        )
        try:
            dof_manager.indices_of_groups([MassBalancePressureGroup()])
        except ValueError:
            pass  # It's ok, this group is not present.
        else:
            raise ValueError(
                "Fixed-stress preconditioner requires mass balance equation "
                "with groups splitted by dimensions. Use "
                "`MassBalancePressureMatrixGroup` etc."
            )

        return {
            prefix: {
                "invertor": lambda bmat: csr_to_petsc(
                    make_fs_analytical_slow_new(
                        dof_manager.model,
                        bmat,
                        p_mat_group=flow_mat_group,
                        p_frac_group=flow_frac_group,
                        groups=bmat.enabled_groups_row,
                    ).mat,
                    bsize=1,
                )
            }
        }


class PetscKspPcConfiguration(ABC):
    def __init__(self, groups: list[EquationVariableGroup], key: str) -> None:
        # keys - for the access of user options, must be unique
        # should have semantic meaning, like "mechanics_subsolver"
        self.groups: list[EquationVariableGroup] = groups
        self.key: str = key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(groups={self.groups})"

    @abstractmethod
    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        pass

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        return {}


class ILU(PetscKspPcConfiguration):
    def __init__(self, groups: list[EquationVariableGroup], key: str = "ilu") -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        default_options = {"pc_type": "ilu"}
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )


class AMG(PetscKspPcConfiguration):
    def __init__(self, groups: list[EquationVariableGroup], key: str = "amg") -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        # This is where the default can be model-dependent. E.g. if 2d, strong_th=0.3
        # and 0.7 if 3d.
        default_options = {"pc_type": "hypre", "pc_hypre_type": "boomeramg"}
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )


class Identity(PetscKspPcConfiguration):
    def __init__(
        self, groups: list[EquationVariableGroup], key: str = "identity"
    ) -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        default_options = {"pc_type": "none"}
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )


class GMRES(PetscKspPcConfiguration):
    def __init__(
        self, preconditioner: PetscKspPcConfiguration, key: str = "gmres"
    ) -> None:
        self.preconditioner: PetscKspPcConfiguration = preconditioner
        super().__init__(groups=self.preconditioner.groups, key=key)

    def __repr__(self) -> str:
        return f"GMRES(preconditioner={self.preconditioner})"

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        default_options = {
            "ksp_type": "gmres",
            "ksp_pc_side": "right",
            "ksp_rtol": 1e-12,
            "ksp_max_it": 300,
            "ksp_gmres_restart": 100,
            "ksp_gmres_cgs_refinement_type": "refine_ifneeded",
            "ksp_gmres_classicalgramschmidt": True,  # Not givens rotations??
        }
        pc_options = self.preconditioner.petsc_options(
            user_options=user_options, prefix=prefix
        )
        return append_prefix_to_options(
            prefix=prefix,
            options=pc_options | default_options | user_options.get(self.key, {}),
        )

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        return self.preconditioner.petsc_assembly_config(
            user_options=user_options, prefix=prefix, dof_manager=dof_manager
        )


class CompositePreconditioner(PetscKspPcConfiguration):
    def __init__(
        self, subsolvers: list[PetscKspPcConfiguration], key: str = "composite"
    ) -> None:
        assert len(subsolvers) >= 1
        groups_of_subsolvers = [subsolver.groups for subsolver in subsolvers]
        for groups in groups_of_subsolvers[1:]:
            assert groups == groups_of_subsolvers[0]
        super().__init__(groups_of_subsolvers[0], key=key)
        self.subsolvers: list[PetscKspPcConfiguration] = subsolvers

    def __repr__(self) -> str:
        return f"CompositePreconditioner(subsolvers={self.subsolvers})"

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        result: dict = {
            "pc_type": "composite",
            "pc_composite_type": "multiplicative",
        }
        for i, subsolver in enumerate(self.subsolvers):
            result |= subsolver.petsc_options(
                user_options=user_options, prefix=f"sub_{i}_"
            )
        return append_prefix_to_options(
            prefix=prefix, options=result | user_options.get(self.key, {})
        )

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        config = {
            prefix: {
                "pc_type": "composite",
                "num_stages": len(self.subsolvers),
            },
        }
        for i, subsolver in enumerate(self.subsolvers):
            subsolver_prefix = f"{prefix}sub_{i}_"
            config |= subsolver.petsc_assembly_config(
                user_options=user_options,
                prefix=subsolver_prefix,
                dof_manager=dof_manager,
            )
        return config


class FieldSplit(PetscKspPcConfiguration):
    def __init__(
        self,
        subsolver: PetscKspPcConfiguration,
        complement: PetscKspPcConfiguration,
        approximate_invertor: PetscInvertor,
        petsc_tag: str = "elim",
        petsc_complement_tag: str = "keep",
        key: str = "fieldsplit",
    ) -> None:
        # petsc_tag - internal, for petsc prefix. Must be short, not necessarily unique.
        self.subsolver: PetscKspPcConfiguration = subsolver
        self.complement: PetscKspPcConfiguration = complement
        self.approximate_invertor: PetscInvertor = approximate_invertor
        self.petsc_tag: str = petsc_tag
        self.petsc_complement_tag: str = petsc_complement_tag
        super().__init__(groups=self.subsolver.groups + self.complement.groups, key=key)

        # assert set(self.subsolver.groups).intersection(self.complement.groups) == 0

    def __repr__(self) -> str:
        return (
            f"FieldSplit(subsolver={self.subsolver}, complement={self.complement}, "
            f"approximate_invertor={self.approximate_invertor})"
        )

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        options = (
            {
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_factorization_type": "upper",
                # default values for the children.
                f"fieldsplit_{self.petsc_tag}_ksp_type": "preonly",
                f"fieldsplit_{self.petsc_complement_tag}_ksp_type": "preonly",
            }
            | self.subsolver.petsc_options(
                user_options=user_options, prefix=f"fieldsplit_{self.petsc_tag}_"
            )
            | self.complement.petsc_options(
                user_options=user_options,
                prefix=f"fieldsplit_{self.petsc_complement_tag}_",
            )
            | self.approximate_invertor.petsc_options(
                prefix="",
                tag=self.petsc_tag,
                complement_tag=self.petsc_complement_tag,
            )
        )
        return append_prefix_to_options(
            prefix=prefix, options=options | user_options.get(self.key, {})
        )

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        return (
            {
                prefix: {
                    "pc_type": "fieldsplit",
                    "elim_tag": self.petsc_tag,
                    "keep_tag": self.petsc_complement_tag,
                    "elim_groups": dof_manager.indices_of_groups(
                        groups=self.subsolver.groups
                    ),
                    "keep_groups": dof_manager.indices_of_groups(
                        groups=self.complement.groups
                    ),
                }
            }
            | self.subsolver.petsc_assembly_config(
                user_options=user_options,
                prefix=f"{prefix}fieldsplit_{self.petsc_tag}_",
                dof_manager=dof_manager,
            )
            | self.complement.petsc_assembly_config(
                user_options=user_options,
                prefix=f"{prefix}fieldsplit_{self.petsc_complement_tag}_",
                dof_manager=dof_manager,
            )
            | self.approximate_invertor.petsc_assembly_config(
                prefix=f"{prefix}fieldsplit_",
                dof_manager=dof_manager,
            )
        )


class PythonWrapper(PetscKspPcConfiguration):
    def __init__(
        self,
        python_context: PcPythonPermutation,
        inner_subsolver: PetscKspPcConfiguration,
        key: str = "python_wrapper",
    ) -> None:
        super().__init__(groups=inner_subsolver.groups, key=key)
        self.python_context: PcPythonPermutation = python_context
        self.inner_subsolver: PetscKspPcConfiguration = inner_subsolver

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        options = {"pc_type": "python"} | self.inner_subsolver.petsc_options(
            user_options=user_options, prefix=f"python_"
        )
        return append_prefix_to_options(
            prefix=prefix, options=options | user_options.get(self.key, {})
        )
        # what if user options change pc_type? We assume it is prohibited. Somewhere it
        # should be checked.

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        return {
            prefix: {
                "pc_type": "python",
                "python_context": self.python_context,
            }
        }


class BlockDiagonalPreconditioner(PetscKspPcConfiguration):
    def __init__(
        self, groups: list[EquationVariableGroup], key: str = "block_diagonal"
    ) -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, prefix: str) -> dict:
        default_options = {"pc_type": "pbjacobi"}
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        return {prefix: {"matrix_block_size": dof_manager.model.nd}}


def nested_schur_complements(subsolvers: list[dict]) -> FieldSplit:
    if len(subsolvers) != 2:
        # recursion
        return FieldSplit(
            subsolver=subsolvers[0]["subsolver"],
            approximate_invertor=subsolvers[0]["approximate_invertor"],
            complement=nested_schur_complements(subsolvers=subsolvers[1:]),
        )
    # end of recursion
    return FieldSplit(
        subsolver=subsolvers[0]["subsolver"],
        approximate_invertor=subsolvers[0]["approximate_invertor"],
        complement=subsolvers[1]["subsolver"],
    )


def mass_balance_factory():
    interface_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        WellFluxGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [MassBalancePressureGroup()]

    return GMRES(
        preconditioner=FieldSplit(
            subsolver=ILU(groups=interface_groups, key="interface_flow"),
            complement=AMG(groups=mass_balance_groups, key="mass_balance_amg"),
            approximate_invertor=DiagonalInvertor(),
        )
    )


def momentum_balance_factory():
    contact_groups: list[EquationVariableGroup] = [ContactMechanicsGroup()]
    mechanics_groups: list[EquationVariableGroup] = [
        InterfaceForceBalanceGroup(),
        MechanicsGroup(),
    ]
    return GMRES(
        preconditioner=FieldSplit(
            subsolver=BlockDiagonalPreconditioner(groups=contact_groups, key="contact"),
            complement=AMG(groups=mechanics_groups, key="mechanics_amg"),
            approximate_invertor=BlockDiagonalInvertor(),
        )
    )


def hm_factory():
    contact_groups: list[EquationVariableGroup] = [ContactMechanicsGroup()]
    interface_flux_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        WellFluxGroup(),
    ]
    mechanics_groups: list[EquationVariableGroup] = [
        InterfaceForceBalanceGroup(),
        MechanicsGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        MassBalancePressureMatrixGroup(),
        MassBalancePressureFracturesGroup(),
        MassBalancePressureIntersectionsGroup(),
    ]

    return GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_invertor": BlockDiagonalInvertor(),
                },
                {
                    "subsolver": ILU(
                        groups=interface_flux_groups, key="interface_flow"
                    ),
                    "approximate_invertor": DiagonalInvertor(),
                },
                {
                    "subsolver": AMG(groups=mechanics_groups, key="mechanics_amg"),
                    "approximate_invertor": FixedStressInvertor(),
                },
                {
                    "subsolver": AMG(
                        groups=mass_balance_groups, key="mass_balance_amg"
                    ),
                },
            ]
        )
    )


def th_factory():
    interface_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        MassBalancePressureMatrixGroup(),
        MassBalancePressureFracturesGroup(),
        MassBalancePressureIntersectionsGroup(),
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        EnergyBalanceTemperatureGroup(),
        # EnergyBalanceTemperatureMatrixGroup(),
        # EnergyBalanceTemperatureFracturesGroup(),
        # EnergyBalanceTemperatureIntersectionsGroup(),
    ]

    return GMRES(
        preconditioner=FieldSplit(
            subsolver=ILU(groups=interface_groups, key="interface_flow"),
            approximate_invertor=DiagonalInvertor(),
            complement=CompositePreconditioner(
                subsolvers=[
                    FieldSplit(
                        subsolver=Identity(
                            groups=energy_balance_groups, key="cpr0_energy"
                        ),
                        complement=AMG(groups=mass_balance_groups, key="cpr0_mass"),
                        approximate_invertor=DiagonalInvertor(),
                    ),
                    ILU(groups=energy_balance_groups + mass_balance_groups, key="cpr1"),
                ]
            ),
        )
    )


def thm_factory():
    contact_groups: list[EquationVariableGroup] = [ContactMechanicsGroup()]
    interface_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mechanics_groups: list[EquationVariableGroup] = [
        InterfaceForceBalanceGroup(),
        MechanicsGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        MassBalancePressureMatrixGroup(),
        MassBalancePressureFracturesGroup(),
        MassBalancePressureIntersectionsGroup(),
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        EnergyBalanceTemperatureGroup(),
        # EnergyBalanceTemperatureMatrixGroup(),
        # EnergyBalanceTemperatureFracturesGroup(),
        # EnergyBalanceTemperatureIntersectionsGroup(),
    ]

    return GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_invertor": BlockDiagonalInvertor(),
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_invertor": DiagonalInvertor(),
                },
                {
                    "subsolver": AMG(groups=mechanics_groups, key="mechanics_amg"),
                    "approximate_invertor": FixedStressInvertor(),
                },
                {
                    "subsolver": CompositePreconditioner(
                        subsolvers=[
                            FieldSplit(
                                subsolver=Identity(
                                    groups=energy_balance_groups, key="cpr0_energy"
                                ),
                                complement=AMG(
                                    groups=mass_balance_groups, key="cpr0_mass"
                                ),
                                approximate_invertor=DiagonalInvertor(),
                            ),
                            ILU(
                                groups=energy_balance_groups + mass_balance_groups,
                                key="cpr1",
                            ),
                        ]
                    )
                },
            ]
        )
    )


def _to_cell_ordering(indexer: LinearSystemIndexer, group_lists: list[list[int]]):
    all_groups = list(chain.from_iterable(group_lists))

    indexer = indexer[all_groups]
    rows = [
        np.concatenate([indexer.dofs_row[i] for i in groups]) for groups in group_lists
    ]

    return np.vstack(rows).ravel("F")


def cfle_factory():
    """Factory for a CFLE preconditioner with well equations."""
    from porepy.numerics.ad.operators import MixedDimensionalVariable

    import porepy as pp
    from pp_solvers.equation_variable_groups import EquationOnDomains, EquationNames

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

    class MassBalancePressureGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            production_wells, no_production_wells = model._filter_wells(
                model.mdg.subdomains(), "production"
            )
            return EquationOnDomains(
                name=EquationNames.MASS_BALANCE.value, domains=no_production_wells
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
            name = EquationNames.ENERGY_BALANCE.value
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

    interface_groups = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mass_balance_groups = [
        MassBalancePressureGroup(),
        ProductionPressureConstraintGroup(),
    ]
    energy_balance_groups = [
        EnergyBalanceEnthalpyGroup(),
        InjectionTemperatureConstraintGroup(),
    ]
    component_groups = [ComponentMassBalanceCO2Group()]

    return GMRES(
        preconditioner=FieldSplit(
            subsolver=ILU(groups=interface_groups, key="interface_prec"),
            approximate_invertor=DiagonalInvertor(),
            complement=CompositePreconditioner(
                subsolvers=[
                    FieldSplit(
                        subsolver=Identity(
                            groups=energy_balance_groups + component_groups,
                            key="cpr_stage0_identity",
                        ),
                        approximate_invertor=DiagonalInvertor(),
                        complement=AMG(
                            groups=mass_balance_groups, key="cpr_stage0_amg"
                        ),
                        key="inner_fieldsplit",
                    ),
                    ILU(
                        groups=energy_balance_groups
                        + component_groups
                        + mass_balance_groups,
                        key="cpr_stage1_ilu",
                    ),
                ]
            ),
        )
    )


def cfle_factory_no_well():
    """Factory for a CFLE preconditioner without well equations."""

    from porepy.numerics.ad.operators import MixedDimensionalVariable
    import porepy as pp
    from pp_solvers.equation_variable_groups import EquationOnDomains, EquationNames

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

    class MassBalancePressureGroup(EquationVariableGroup):

        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:

            return EquationOnDomains(

                name=EquationNames.MASS_BALANCE.value, domains=model.mdg.subdomains()

            )

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:

            return model.pressure(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:

            return "mass_balance"

        def variable_name(self, model: pp.PorePyModel) -> str:

            return "pressure"

    class EnergyBalanceEnthalpyGroup(EquationVariableGroup):

        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:

            name = EquationNames.ENERGY_BALANCE.value

            return EquationOnDomains(name=name, domains=model.mdg.subdomains())

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:

            return model.enthalpy(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:

            return "energy_balance"

        def variable_name(self, model: pp.PorePyModel) -> str:

            return "entalpy"

    interface_groups = [

        InterfaceDarcyFluxGroup(),

        InterfaceEnthalpyFluxGroup(),

        InterfaceFourierFluxGroup(),

        WellFluxGroup(),

        WellEnthalpyFluxGroup(),

    ]

    mass_balance_groups = [

        MassBalancePressureGroup(),

    ]

    energy_balance_groups = [

        EnergyBalanceEnthalpyGroup(),

    ]

    component_groups = [ComponentMassBalanceCO2Group()]

    return GMRES(

        preconditioner=FieldSplit(

            subsolver=ILU(groups=interface_groups, key="interface_prec"),

            approximate_invertor=DiagonalInvertor(),

            complement=CompositePreconditioner(

                subsolvers=[

                    FieldSplit(

                        subsolver=Identity(

                            groups=energy_balance_groups + component_groups,

                            key="cpr_stage0_identity",

                        ),

                        approximate_invertor=DiagonalInvertor(),

                        complement=AMG(

                            groups=mass_balance_groups, key="cpr_stage0_amg"

                        ),

                        key="inner_fieldsplit",

                    ),

                    ILU(

                        groups=energy_balance_groups

                        + component_groups

                        + mass_balance_groups,

                        key="cpr_stage1_ilu",

                    ),

                ]

            ),

        )

    )


def cf_factory_no_well():
    """
    Factory for a CF preconditioner without well equations.
    Mike: Preconditioner factory defined for problem involving cf with correlations
    """

    from porepy.numerics.ad.operators import MixedDimensionalVariable
    import porepy as pp
    from pp_solvers.equation_variable_groups import EquationOnDomains, EquationNames

    class ComponentMassBalanceNaClGroup(EquationVariableGroup):

        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:

            name = "component_mass_balance_equation_NaCl"

            return EquationOnDomains(name=name, domains=model.mdg.subdomains())

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:

            return model.fluid.components[1].fraction(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:

            return "component_mass_balance_equation_NaCl"

        def variable_name(self, model: pp.PorePyModel) -> str:

            return "z_NaCl"

    class MassBalancePressureGroup(EquationVariableGroup):

        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:

            return EquationOnDomains(

                name=EquationNames.MASS_BALANCE.value, domains=model.mdg.subdomains()

            )

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:

            return model.pressure(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:

            return "mass_balance"

        def variable_name(self, model: pp.PorePyModel) -> str:

            return "pressure"

    class EnergyBalanceEnthalpyGroup(EquationVariableGroup):

        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:

            name = EquationNames.ENERGY_BALANCE.value

            return EquationOnDomains(name=name, domains=model.mdg.subdomains())

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:

            return model.enthalpy(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:

            return "energy_balance"

        def variable_name(self, model: pp.PorePyModel) -> str:

            return "enthalpy"

    interface_groups = [

        InterfaceDarcyFluxGroup(),

        InterfaceEnthalpyFluxGroup(),

        InterfaceFourierFluxGroup(),

        WellFluxGroup(),

        WellEnthalpyFluxGroup(),

    ]

    mass_balance_groups = [

        MassBalancePressureGroup(),

    ]

    energy_balance_groups = [

        EnergyBalanceEnthalpyGroup(),

    ]

    component_groups = [ComponentMassBalanceNaClGroup()]

    return GMRES(

        preconditioner=FieldSplit(

            subsolver=ILU(groups=interface_groups, key="interface_prec"),

            approximate_invertor=DiagonalInvertor(),

            complement=CompositePreconditioner(

                subsolvers=[

                    FieldSplit(

                        subsolver=Identity(

                            groups=energy_balance_groups + component_groups,

                            key="cpr_stage0_identity",

                        ),

                        approximate_invertor=DiagonalInvertor(),

                        complement=AMG(

                            groups=mass_balance_groups, key="cpr_stage0_amg"

                        ),

                        key="inner_fieldsplit",

                    ),

                    ILU(

                        groups=energy_balance_groups

                        + component_groups

                        + mass_balance_groups,

                        key="cpr_stage1_ilu",

                    ),

                ]

            ),

        )
    )


def cf_factory_well_inj_prod():
    """
    Factory for a CF preconditioner with energy PDE eliminated at the injection well by 
     enthalpy or temperature constraint and the pressure equation eliminated at the producing well
     well equations.
    Mike: Preconditioner factory defined for problem involving cf with correlations
    """
    from porepy.numerics.ad.operators import MixedDimensionalVariable

    import porepy as pp
    from pp_solvers.equation_variable_groups import EquationOnDomains, EquationNames

    class ComponentMassBalanceNaClGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            name = "component_mass_balance_equation_NaCl"
            return EquationOnDomains(name=name, domains=model.mdg.subdomains())

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
            return model.fluid.components[1].fraction(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:
            return "component_mass_balance_equation_NaCl"

        def variable_name(self, model: pp.PorePyModel) -> str:
            return "z_NaCl"

    class MassBalancePressureGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            production_wells, no_production_wells = model._filter_wells(
                model.mdg.subdomains(), "production"
            )
            return EquationOnDomains(
                name=EquationNames.MASS_BALANCE.value, domains=no_production_wells
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
            name = EquationNames.ENERGY_BALANCE.value

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
            return "enthalpy"

    class ProductionPressureConstraintGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            # TODO: I need to check this out, for my case I do not 
            # have production wells, but I have injection wells with temperature constraints, 
            # so I need to check how to handle this case.
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

    interface_groups = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mass_balance_groups = [
        MassBalancePressureGroup(),
        ProductionPressureConstraintGroup(),
    ]
    energy_balance_groups = [
        EnergyBalanceEnthalpyGroup(),
        InjectionTemperatureConstraintGroup(),
    ]
    component_groups = [ComponentMassBalanceNaClGroup()]

    return GMRES(
        preconditioner=FieldSplit(
            subsolver=ILU(groups=interface_groups, key="interface_prec"),
            approximate_invertor=DiagonalInvertor(),
            complement=CompositePreconditioner(
                subsolvers=[
                    FieldSplit(
                        subsolver=Identity(
                            groups=energy_balance_groups + component_groups,
                            key="cpr_stage0_identity",
                        ),
                        approximate_invertor=DiagonalInvertor(),
                        complement=AMG(
                            groups=mass_balance_groups, key="cpr_stage0_amg"
                        ),
                        key="inner_fieldsplit",
                    ),
                    ILU(
                        groups=energy_balance_groups
                        + component_groups
                        + mass_balance_groups,
                        key="cpr_stage1_ilu",
                    ),
                ]
            ),
        )
    )


def cf_factory_well_inj():
    """
    Factory for a CF preconditioner with energy PDE eliminated at the injection grid cell
    is replaced by temperture or enthalpy constraint!!.
    Mike: Preconditioner factory defined for problem involving cf with correlations
    """
    from porepy.numerics.ad.operators import MixedDimensionalVariable

    import porepy as pp
    from pp_solvers.equation_variable_groups import EquationOnDomains, EquationNames

    class ComponentMassBalanceNaClGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            name = "component_mass_balance_equation_NaCl"
            return EquationOnDomains(name=name, domains=model.mdg.subdomains())

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
            return model.fluid.components[1].fraction(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:
            return "component_mass_balance_equation_NaCl"

        def variable_name(self, model: pp.PorePyModel) -> str:
            return "z_NaCl"

    class MassBalancePressureGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            # production_wells, no_production_wells = model._filter_wells(
            #     model.mdg.subdomains(), "production"
            # )
            return EquationOnDomains(
                name=EquationNames.MASS_BALANCE.value, domains=model.mdg.subdomains()
            )

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
            # production_wells, no_production_wells = model._filter_wells(
            #     model.mdg.subdomains(), "production"
            # )
            return model.pressure(model.mdg.subdomains())

        def equation_name(self, model: pp.PorePyModel) -> str:
            return "mass_balance"

        def variable_name(self, model: pp.PorePyModel) -> str:
            return "pressure"

    class EnergyBalanceEnthalpyGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            name = EquationNames.ENERGY_BALANCE.value

            injection_wells, no_injection_wells = model._filter_wells(
                model.mdg.subdomains(), "injection"
            )
            print(f"DEBUG: EnergyBalanceEnthalpyGroup - Injection wells (eq): {len(injection_wells)}, No injection wells (eq): {len(no_injection_wells)}")
            return EquationOnDomains(name=name, domains=no_injection_wells)

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
            injection_wells, no_injection_wells = model._filter_wells(
                model.mdg.subdomains(), "injection"
            )
            print(f"DEBUG: EnergyBalanceEnthalpyGroup - Injection wells (var): {len(injection_wells)}, No injection wells (var): {len(no_injection_wells)}")
            return model.enthalpy(no_injection_wells)

        def equation_name(self, model: pp.PorePyModel) -> str:
            return "energy_balance"

        def variable_name(self, model: pp.PorePyModel) -> str:
            return "enthalpy"

    # class ProductionPressureConstraintGroup(EquationVariableGroup):
    #     def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
    #         # TODO: I need to check this out, for my case I do not
    #         # have production wells, but I have injection wells with temperature constraints,
    #         # so I need to check how to handle this case.
    #         name = "production_pressure_constraint"
    #         production_wells, no_production_wells = model._filter_wells(
    #             model.mdg.subdomains(), "production"
    #         )
    #         print(f"DEBUG: ProductionPressureConstraintGroup - Production wells (eq): {len(production_wells)}, No production wells (eq): {len(no_production_wells)}")
    #         return EquationOnDomains(name=name, domains=production_wells)

    #     def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
    #         production_wells, no_production_wells = model._filter_wells(
    #             model.mdg.subdomains(), "production"
    #         )
    #         print(f"DEBUG: ProductionPressureConstraintGroup - Production wells (var): {len(production_wells)}, No production wells (var): {len(no_production_wells)}")
    #         return model.pressure(production_wells)

    #     def equation_name(self, model: pp.PorePyModel) -> str:
    #         return "production_pressure_constraint"

    #     def variable_name(self, model: pp.PorePyModel) -> str:
    #         return "pressure_constraint"

    class InjectionTemperatureConstraintGroup(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            name = "injection_temperature_constraint"
            injection_wells, no_injection_wells = model._filter_wells(
                model.mdg.subdomains(), "injection"
            )
            print(f"DEBUG: InjectionTemperatureConstraintGroup - Injection wells (eq): {len(injection_wells)}, No injection wells (eq): {len(no_injection_wells)}")
            return EquationOnDomains(name=name, domains=injection_wells)

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
            injection_wells, no_injection_wells = model._filter_wells(
                model.mdg.subdomains(), "injection"
            )
            print(f"DEBUG: InjectionTemperatureConstraintGroup - Injection wells (var): {len(injection_wells)}, No injection wells (var): {len(no_injection_wells)}")
            return model.enthalpy(injection_wells)

        def equation_name(self, model: pp.PorePyModel) -> str:
            return "injection_temperature_constraint"

        def variable_name(self, model: pp.PorePyModel) -> str:
            return "enthalpy_constraint"

    interface_groups = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mass_balance_groups = [
        MassBalancePressureGroup(),
        # ProductionPressureConstraintGroup(),
    ]
    energy_balance_groups = [
        EnergyBalanceEnthalpyGroup(),
        InjectionTemperatureConstraintGroup(),
    ]
    component_groups = [ComponentMassBalanceNaClGroup()]

    return GMRES(
        preconditioner=FieldSplit(
            subsolver=ILU(groups=interface_groups, key="interface_prec"),
            approximate_invertor=DiagonalInvertor(),
            complement=CompositePreconditioner(
                subsolvers=[
                    FieldSplit(
                        subsolver=Identity(
                            groups=energy_balance_groups + component_groups,
                            key="cpr_stage0_identity",
                        ),
                        approximate_invertor=DiagonalInvertor(),
                        complement=AMG(
                            groups=mass_balance_groups, key="cpr_stage0_amg"
                        ),
                        key="inner_fieldsplit",
                    ),
                    ILU(
                        groups=energy_balance_groups
                        + component_groups
                        + mass_balance_groups,
                        key="cpr_stage1_ilu",
                    ),
                ]
            ),
        )
    )
