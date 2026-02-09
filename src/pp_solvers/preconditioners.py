from __future__ import annotations

from typing import Optional
import warnings
from abc import ABC, abstractmethod

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
from pp_solvers.fixed_stress import construct_fixed_stress_block_matrix
from pp_solvers.petsc_solvers import PcPythonPermutation
from pp_solvers.petsc_utils import csr_to_petsc

__all__ = [
    # Add all preconditioners and linear solvers here.
    "PetscKspPcConfiguration",
    "ILU",
    "AMG",
    "Identity",
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

    def petsc_assembly_config(self, dof_manager: DofManager) -> dict:
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
        # YZ: How does it fetch the block size in that matrix?


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

    def petsc_assembly_config(self, dof_manager: DofManager) -> dict:
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
            "invertor_additive": lambda indexer: csr_to_petsc(
                construct_fixed_stress_block_matrix(
                    model=dof_manager.model,
                    indexer=indexer,
                    p_mat_group=flow_mat_group,
                    p_frac_group=flow_frac_group,
                ).mat,
                bsize=1,
            )
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
    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        pass

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        return {}


class ILU(PetscKspPcConfiguration):
    def __init__(self, groups: list[EquationVariableGroup], key: str = "ilu") -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        default_options = {"pc_type": "ilu"}
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )


class AMG(PetscKspPcConfiguration):
    def __init__(
        self,
        groups: list[EquationVariableGroup],
        key: str = "amg",
        vector_problem: bool = False,
    ) -> None:
        self.vector_problem: bool = vector_problem
        super().__init__(groups=groups, key=key)

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        # The default strong threshold is dimension-dependent.
        strong_threshold = 0.7  # if dof_manager.model.nd == 3 else 0.25
        default_options = {
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_strong_threshold": strong_threshold,
        }
        if self.vector_problem:
            default_options["mat_block_size"] = dof_manager.model.nd
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )


class Identity(PetscKspPcConfiguration):
    def __init__(
        self, groups: list[EquationVariableGroup], key: str = "identity"
    ) -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
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

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
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
            user_options=user_options,
            prefix="",
            dof_manager=dof_manager,
        )

        this_user_options = user_options.get(self.key, {})

        # There can be an unfortunate overlap in GMRES and preconditioner options. We
        # print a warning in this case. The current behavior is that the preconditioner
        # options are prioritized.
        intersection_in_options = set(this_user_options).intersection(pc_options)
        if len(intersection_in_options) > 0:
            warnings.warn(
                "Both GMRES and preconditioner override options: "
                f"{intersection_in_options}. Preconditioner options are prioritized."
            )

        return append_prefix_to_options(
            prefix=prefix,
            options=default_options | this_user_options | pc_options,
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
            if groups != groups_of_subsolvers[0]:
                raise ValueError(
                    "CompositePreconditioner subsolvers must operate on identical"
                    " groups."
                )
        super().__init__(groups_of_subsolvers[0], key=key)
        self.subsolvers: list[PetscKspPcConfiguration] = subsolvers

    def __repr__(self) -> str:
        return f"CompositePreconditioner(subsolvers={self.subsolvers})"

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        result: dict = {
            "pc_type": "composite",
            "pc_composite_type": "multiplicative",
        }
        for i, subsolver in enumerate(self.subsolvers):
            result |= subsolver.petsc_options(
                user_options=user_options,
                prefix=f"sub_{i}_",
                dof_manager=dof_manager,
            )
        return append_prefix_to_options(
            prefix=prefix, options=result | user_options.get(self.key, {})
        )

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        config = {
            prefix: {
                "config_type": "composite",
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
        petsc_tag: Optional[str] = None,
        petsc_complement_tag: Optional[str] = None,
        key: str = "fieldsplit",
    ) -> None:
        # petsc_tag - internal, for petsc prefix. Must be short, not necessarily unique.
        if petsc_complement_tag is None and petsc_tag is None:
            petsc_tag = "elim"
            petsc_complement_tag = "elim"
        elif petsc_complement_tag is None:
            petsc_complement_tag = f"{petsc_tag}_cpl"
        self.subsolver: PetscKspPcConfiguration = subsolver
        self.complement: PetscKspPcConfiguration = complement
        self.approximate_invertor: PetscInvertor = approximate_invertor
        self.petsc_tag: str = petsc_tag
        self.petsc_complement_tag: str = petsc_complement_tag
        super().__init__(groups=self.subsolver.groups + self.complement.groups, key=key)

        # This is O(n^2), but we typically have 10 - 20 groups, and this type is not
        # hashable, so why bother?
        for g1 in self.subsolver.groups:
            for g2 in self.complement.groups:
                if g1 == g2:
                    raise ValueError(f"Groups in FielSplit should not overlap: {g1}")

    def __repr__(self) -> str:
        return (
            f"FieldSplit(subsolver={self.subsolver}, complement={self.complement}, "
            f"approximate_invertor={self.approximate_invertor})"
        )

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
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
                user_options=user_options,
                prefix=f"fieldsplit_{self.petsc_tag}_",
                dof_manager=dof_manager,
            )
            | self.complement.petsc_options(
                user_options=user_options,
                prefix=f"fieldsplit_{self.petsc_complement_tag}_",
                dof_manager=dof_manager,
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
                    "config_type": "fieldsplit",
                    "elim_tag": self.petsc_tag,
                    "keep_tag": self.petsc_complement_tag,
                    "elim_groups": dof_manager.indices_of_groups(
                        groups=self.subsolver.groups
                    ),
                    "keep_groups": dof_manager.indices_of_groups(
                        groups=self.complement.groups
                    ),
                }
                | self.approximate_invertor.petsc_assembly_config(
                    dof_manager=dof_manager
                )
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
        )


class PythonPermutationWrapper(PetscKspPcConfiguration):
    def __init__(
        self,
        inner_subsolver: PetscKspPcConfiguration,
        permutation_groups: list[list[EquationVariableGroup]],
        key: str = "python_permutation",
    ) -> None:
        super().__init__(groups=inner_subsolver.groups, key=key)
        self.permutation_groups: list[list[EquationVariableGroup]] = permutation_groups
        self.inner_subsolver: PetscKspPcConfiguration = inner_subsolver

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        options = {"pc_type": "python"} | self.inner_subsolver.petsc_options(
            user_options=user_options, prefix=f"python_", dof_manager=dof_manager
        )
        return append_prefix_to_options(
            prefix=prefix, options=options | user_options.get(self.key, {})
        )
        # what if user options change pc_type? We assume it is prohibited. Somewhere it
        # should be checked.

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        inner_config = self.inner_subsolver.petsc_assembly_config(
            user_options=user_options,
            prefix=f"{prefix}python_",
            dof_manager=dof_manager,
        )
        if len(inner_config) > 0:
            raise NotImplementedError(
                "Nested initialization inside PythonPermutationWrapper is not "
                "implemented."
            )
        return {
            prefix: {
                "config_type": "python_permutation",
                "permutation_groups": [
                    dof_manager.indices_of_groups(g) for g in self.permutation_groups
                ],
            }
        }


class BlockDiagonalPreconditioner(PetscKspPcConfiguration):
    def __init__(
        self, groups: list[EquationVariableGroup], key: str = "block_diagonal"
    ) -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        default_options = {
            "pc_type": "pbjacobi",
            "mat_block_size": dof_manager.model.nd,
        }
        return append_prefix_to_options(
            prefix=prefix, options=default_options | user_options.get(self.key, {})
        )


def nested_schur_complements(subsolvers: list[dict]) -> FieldSplit:
    # Unwrapping parameters.
    kwargs = {
        "subsolver": subsolvers[0]["subsolver"],
        "approximate_invertor": subsolvers[0]["approximate_invertor"],
    }
    if "key" in subsolvers[0]:
        kwargs["key"] = subsolvers[0]["key"]
    if "petsc_tag" in subsolvers[0]:
        kwargs["petsc_tag"] = subsolvers[0]["petsc_tag"]
    if "petsc_complement_tag" in subsolvers[0]:
        kwargs["petsc_complement_tag"] = subsolvers[0]["petsc_complement_tag"]

    if len(subsolvers) > 2:
        # Recursion.
        return FieldSplit(
            complement=nested_schur_complements(subsolvers=subsolvers[1:]), **kwargs
        )
    # End of recursion.
    return FieldSplit(complement=subsolvers[1]["subsolver"], **kwargs)


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
        MechanicsGroup(),
        InterfaceForceBalanceGroup(),
    ]
    return GMRES(
        preconditioner=FieldSplit(
            petsc_tag="contact",
            subsolver=BlockDiagonalPreconditioner(groups=contact_groups, key="contact"),
            complement=AMG(
                groups=mechanics_groups, key="mechanics_amg", vector_problem=True
            ),
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
        MechanicsGroup(),
        InterfaceForceBalanceGroup(),
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
                    "petsc_tag": "contact",
                },
                {
                    "subsolver": ILU(
                        groups=interface_flux_groups, key="interface_flow"
                    ),
                    "approximate_invertor": DiagonalInvertor(),
                    "petsc_tag": "intf_darcy_flux",
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_invertor": FixedStressInvertor(),
                    "petsc_tag": "mechanics",
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
    ]

    return GMRES(
        preconditioner=FieldSplit(
            petsc_tag="intf_mass_energy_flx",
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
        MechanicsGroup(),
        InterfaceForceBalanceGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        MassBalancePressureMatrixGroup(),
        MassBalancePressureFracturesGroup(),
        MassBalancePressureIntersectionsGroup(),
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        EnergyBalanceTemperatureGroup(),
    ]

    return GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_invertor": BlockDiagonalInvertor(),
                    "petsc_tag": "contact",
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_invertor": DiagonalInvertor(),
                    "petsc_tag": "intf_mass_energy_flx",
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_invertor": FixedStressInvertor(),
                    "petsc_tag": "mech",
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
                                petsc_tag="mass_bal",
                                approximate_invertor=DiagonalInvertor(),
                            ),
                            PythonPermutationWrapper(
                                permutation_groups=[
                                    energy_balance_groups,
                                    mass_balance_groups,
                                ],
                                inner_subsolver=ILU(
                                    groups=energy_balance_groups + mass_balance_groups,
                                    key="cpr1",
                                ),
                            ),
                        ]
                    )
                },
            ]
        )
    )
