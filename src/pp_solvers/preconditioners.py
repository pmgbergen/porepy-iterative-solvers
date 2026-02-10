"""This module contains classes that describe the components of the PETSc KSP and PC.
These classes do not produce PETSc options by themselves, they instead generate a dict
of PETSc options, and a dict of instruction, used to assemble PETSc objects in
`options_parser.py`.

This module also defines the default linear solver configurations for PorePy models.

"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Sequence

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
    "PetscInvertor",
    "DiagonalInvertor",
    "BlockDiagonalInvertor",
    "FixedStressInvertor",
    "ILU",
    "AMG",
    "Identity",
    "GMRES",
    "CompositePreconditioner",
    "FieldSplitAdditive",
    "FieldSplitSchur",
    "PythonPermutationWrapper",
    "BlockDiagonalPreconditioner",
    # Add all the factory functions here.
    "mass_balance_factory",
    "momentum_balance_factory",
    "hm_factory",
    "th_factory",
    "thm_factory",
]


def append_prefix_to_options(prefix: str, options: dict):
    return {f"{prefix}{key}": value for key, value in options.items()}


# MARK: Invertors


class PetscInvertor(ABC):
    """The base class for the customizable invertor instruction for the
    `FieldSplitSchur` class.

    """

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
        key = f"fieldsplit_{complement_tag}_mat_schur_complement_ainv_type"
        return append_prefix_to_options(
            prefix=prefix,
            options={
                "pc_fieldsplit_schur_precondition": "selfp",
                key: "blockdiag",
            },
        )
        # The matrix block size should be provided by the subsolver of the group to
        # eliminate.


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


# MARK: Configurations


class PetscKspPcConfiguration(ABC):
    """The base class to define a component of the nested PETSc linear solver
    configuration.

    """

    def __init__(self, groups: list[EquationVariableGroup], key: str) -> None:
        self.groups: list[EquationVariableGroup] = groups
        """The groups this solver operates on. The non-leaf solvers contain groups of
        their children sub-solvers.

        """
        self.key: str = key
        """The key to access the particular subsolver in the nested configuration. Must
        be unique and meaningful for the user, like "mechanics_subsolver", not "amg", as
        there can be several instances of "amg".

        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(groups={self.groups})"

    @abstractmethod
    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        """Builds the options for PETSc command-line format. The non-leaf solvers
        include the options of their children sub-solvers, with the corresponding
        prefices.

        Parameters:
            user_options: A dictionary of the petsc options that the user can pass to
                customize the setup. Expected format:
                ```
                {
                    key: {
                        "petsc_option_1": "value1",
                        "petsc_option_2": "value2",
                    }
                }
                ```
                where `key` corresponds to the unique `PetscKspPcConfiguration.key` of
                the subsolver to apply options. The user should provide options without
                a prefix, e.g. "pc_type", not "fieldsplit_sub_0_pc_type".

            prefix: PETSc prefix to use with the options. If the method is called from
                the user code, the empty prefix should typically be used. Internally,
                used in recursion.

            dof_manager: The `DofManager` for the problem.

        Returns: A flat dictionary of PETSc command-line options.

        """

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        """Builds a configuration for the `assemble_petsc_ksp_pc` function. The non-leaf
        solvers include the configurations of their children sub-solvers, with the
        corresponding prefices.

        Parameters:
            user_options: A dictionary of the petsc options that the user can pass to
                customize the setup. See `PetscKspPcConfiguration.petsc_options` for
                more info.

            prefix: PETSc prefix to use with the options. If the method is called from
                the user code, the empty prefix should typically be used. Internally,
                used in recursion.

            dof_manager: The `DofManager` for the problem.

        """
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
        strong_threshold = 0.7 if dof_manager.model.nd == 3 else 0.25
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


class FieldSplitAdditive(PetscKspPcConfiguration):
    def __init__(
        self,
        subsolvers: Sequence[PetscKspPcConfiguration],
        key: str = "fieldsplit_additive",
    ) -> None:
        self.subsolvers: list[PetscKspPcConfiguration] = list(subsolvers)
        super().__init__(
            groups=[g for subsolver in self.subsolvers for g in subsolver.groups],
            key=key,
        )

        if len(set(self.groups)) < len(self.groups):
            # Non-unique groups are present.
            raise ValueError(
                f"Groups in FielSplitAdditive should not overlap:", self.groups
            )

    def __repr__(self) -> str:
        return f"FieldSplitAdditive(subsolvers={self.subsolvers})"

    def petsc_options(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        options = {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
        } | {
            f"fieldsplit_sub_{i}_ksp_type": "preonly"
            for i in range(len(self.subsolvers))
        }
        for i, subsolver in enumerate(self.subsolvers):
            options.update(
                subsolver.petsc_options(
                    user_options=user_options,
                    prefix=f"fieldsplit_sub_{i}_",
                    dof_manager=dof_manager,
                )
            )

        return append_prefix_to_options(
            prefix=prefix, options=options | user_options.get(self.key, {})
        )

    def petsc_assembly_config(
        self, user_options: dict, prefix: str, dof_manager: DofManager
    ) -> dict:
        result = {
            prefix: {
                "config_type": "fieldsplit_additive",
                "subsolver_groups": [
                    dof_manager.indices_of_groups(subsolver.groups)
                    for subsolver in self.subsolvers
                ],
            }
        }
        for i, subsolver in enumerate(self.subsolvers):
            result.update(
                subsolver.petsc_assembly_config(
                    user_options=user_options,
                    prefix=f"fieldsplit_sub_{i}_",
                    dof_manager=dof_manager,
                )
            )
        return result


class FieldSplitSchur(PetscKspPcConfiguration):
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

        intersection = set(self.subsolver.groups).intersection(self.complement.groups)
        if len(intersection) > 0:
            raise ValueError(f"Groups in FielSplit should not overlap: {intersection}")

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
                    "config_type": "fieldsplit_schur",
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


def nested_schur_complements(subsolvers: list[dict]) -> FieldSplitSchur:
    """A utility function that replaces a deeply nested syntax:
    ```
    configuration = FieldSplitSchur(
        complement=FieldSplitSchur(
            complement=FieldSplitSchur(
                complement=FieldSplitSchur(
                    complement=FieldSplitSchur(
                        complement=...
                    )
                )
            )
        )
    )
    ```
    with a more flat list of dictionary syntax:
    ```
    configuration = nested_schur_complements([
        {'parameter_of_schur_complement_0': ...},
        {'parameter_of_schur_complement_1': ...},
        {'parameter_of_schur_complement_2': ...},
        {'parameter_of_schur_complement_3': ...},
        {'parameter_of_schur_complement_4': ...},
    ])
    ```

    Each dictionary accepts has the keys as the `FieldSplitSchur` constructor
    parameters.

    """
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
        return FieldSplitSchur(
            complement=nested_schur_complements(subsolvers=subsolvers[1:]), **kwargs
        )
    # End of recursion.
    return FieldSplitSchur(complement=subsolvers[1]["subsolver"], **kwargs)


def mass_balance_factory():
    interface_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        WellFluxGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [MassBalancePressureGroup()]

    return GMRES(
        preconditioner=FieldSplitSchur(
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
        preconditioner=FieldSplitSchur(
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
        preconditioner=FieldSplitSchur(
            petsc_tag="intf_mass_energy_flx",
            subsolver=ILU(groups=interface_groups, key="interface_flow"),
            approximate_invertor=DiagonalInvertor(),
            complement=CompositePreconditioner(
                subsolvers=[
                    FieldSplitSchur(
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
                            FieldSplitAdditive(
                                subsolvers=[
                                    Identity(
                                        groups=energy_balance_groups, key="cpr0_energy"
                                    ),
                                    AMG(groups=mass_balance_groups, key="cpr0_mass"),
                                ],
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
