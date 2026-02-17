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
from pp_solvers.petsc_utils import csr_to_petsc

__all__ = [
    # Add all preconditioners and linear solvers here.
    "PetscKspPcConfiguration",
    "PetscInverter",
    "DiagonalInverter",
    "BlockDiagonalInverter",
    "FixedStressInverter",
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


# MARK: Inverters


class PetscInverter(ABC):
    """The base class for the customizable inverter instruction for the
    `FieldSplitSchur` class.

    """

    @abstractmethod
    def petsc_options(self, prefix: str, tag: str, complement_tag: str) -> dict:
        pass

    def petsc_assembly_config(self, dof_manager: DofManager) -> dict:
        return {}


class DiagonalInverter(PetscInverter):
    def petsc_options(self, prefix: str, tag: str, complement_tag: str) -> dict:
        return append_prefix_to_options(
            prefix=prefix,
            options={
                "pc_fieldsplit_schur_precondition": "selfp",
            },
        )


class BlockDiagonalInverter(PetscInverter):
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


class FixedStressInverter(PetscInverter):
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
        # Check that the MassBalancePressureGroup (common for porous media, fractures
        # and interfaces) is not used by mistake. The fixed stress code relies on
        # separate groups for different dimensions.
        try:
            # Not using the return value, just checking that it is present.
            _ = dof_manager.indices_of_groups([MassBalancePressureGroup()])
        except ValueError:
            pass  # It's ok, this group is not present.
        else:
            raise ValueError(
                "Fixed-stress preconditioner requires mass balance equation "
                "with groups splitted by dimensions. Use "
                "`MassBalancePressureMatrixGroup` etc."
            )

        return {
            "inverter_additive": lambda indexer: csr_to_petsc(
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
    configuration. All the components serve as blueprints and can be customized via the
    `user_options` parameter.

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

        Returns: A dictionary of the following structure:
            ```
            {
                petsc_prefix_1: {
                    "config_type": "fieldsplit_schur",
                    ...
                },
                petsc_prefix_2: {
                    "config_type": "composite",
                    ...
                },
                ...
            }
            ```
            where each sub-dictionary corresponds to a sub-solver, which needs to be
            configured via python. `petsc_prefix_x` corresponds to the petsc prefix of
            this sub-solver.

        """
        return {}


class ILU(PetscKspPcConfiguration):
    """PETSc implementation of ILU, see for additional options:
    https://petsc.org/release/manualpages/PC/PCILU/

    """

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
    """HYPRE BoomerAMG, classical AMG. Uses strong threshold of 0.25 for 2D problems and
    0.7 for 3D problem.

    See for additional options:
    https://petsc.org/main/manualpages/PC/PCHYPRE/
    https://mooseframework.inl.gov/releases/moose/2022-06-10/application_development/hypre.html

    """

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
    """A dummy preconditioner that does nothing."""

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
    """PETSc implementation of GMRES. By default, estimates convergence based on the
    unpreconditioned residual norm. See for more options:
    https://petsc.org/release/manualpages/KSP/KSPGMRES/

    """

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
    """A multi-stage preconditioner that applies preconditioners (stages) to the same
    problem. See: https://petsc.org/release/manualpages/PC/PCCOMPOSITE/

    """

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
    """A preconditioner that splits the problem into n sub-problems and treats each
    separately with a sub-solver. See:
    https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/

    """

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
    """A preconditioner that splits the problem into two sub-problems by building a
    Schur complement approximation and treats each separately with a sub-solver. See:
    https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/

    Consider a 2x2 block matrix:
    ```
    [[A, B],
     [C, D]]
    ```
    with the Schur complement `S = D - B * A^-1 * C`.

    **Note**: `petsc_tag` and `petsc_complement_tag` must be short, as PETSc has a limit
    of 127 symbols for a prefix. They may not be equal to the `key` parameter: the tags
    are for internal identification by PETSc, the key is for identification by
    simulation developers.

    Parameters:
        subsolver: A configuration class of a solver that approximates `A^-1`.
        complement_solver: A configuration class of a solver that approximates `S^-1`.
        approximate_inverter: A configuration class to construct the approximate `S`.
        petsc_tag: A string to build a PETSc options prefix that identifies the `A^-1`
            sub-solver. Defaults to `"elim"` (submatrix to eliminate).
        petsc_complemet_tag: A string to build a PETSc options prefix that identifies
            the `S^-1` sub-solver. Defaults to `"keep"` (submatrix to keep) if
            `petsc_tag` is not passed, otherwise to `f"{petsc_tag}_cpl"` (complenent).
        key: A key to pass user options to the configurations.

    """

    def __init__(
        self,
        subsolver: PetscKspPcConfiguration,
        complement_solver: PetscKspPcConfiguration,
        approximate_inverter: PetscInverter,
        petsc_tag: Optional[str] = None,
        petsc_complement_tag: Optional[str] = None,
        key: str = "fieldsplit",
    ) -> None:
        # petsc_tag - internal, for petsc prefix. Must be short, not necessarily unique.
        if petsc_complement_tag is None:
            if petsc_tag is not None:
                petsc_complement_tag = f"{petsc_tag}_cpl"
            else:
                petsc_complement_tag = "keep"
        if petsc_tag is None:
            petsc_tag = "elim"
        self.subsolver: PetscKspPcConfiguration = subsolver
        self.complement_solver: PetscKspPcConfiguration = complement_solver
        self.approximate_inverter: PetscInverter = approximate_inverter
        self.petsc_tag: str = petsc_tag
        self.petsc_complement_tag: str = petsc_complement_tag
        super().__init__(
            groups=self.subsolver.groups + self.complement_solver.groups, key=key
        )

        intersection = set(self.subsolver.groups).intersection(
            self.complement_solver.groups
        )
        if len(intersection) > 0:
            raise ValueError(f"Groups in FielSplit should not overlap: {intersection}")

    def __repr__(self) -> str:
        return (
            f"FieldSplit(subsolver={self.subsolver}, "
            f"complement_solver={self.complement_solver}, "
            f"approximate_inverter={self.approximate_inverter})"
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
            | self.complement_solver.petsc_options(
                user_options=user_options,
                prefix=f"fieldsplit_{self.petsc_complement_tag}_",
                dof_manager=dof_manager,
            )
            | self.approximate_inverter.petsc_options(
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
                        groups=self.complement_solver.groups
                    ),
                }
                | self.approximate_inverter.petsc_assembly_config(
                    dof_manager=dof_manager
                )
            }
            | self.subsolver.petsc_assembly_config(
                user_options=user_options,
                prefix=f"{prefix}fieldsplit_{self.petsc_tag}_",
                dof_manager=dof_manager,
            )
            | self.complement_solver.petsc_assembly_config(
                user_options=user_options,
                prefix=f"{prefix}fieldsplit_{self.petsc_complement_tag}_",
                dof_manager=dof_manager,
            )
        )


class PythonPermutationWrapper(PetscKspPcConfiguration):
    """A pre- and post-processing tool for a preconditioner, that permutes the physical
    quantities in the underlying matrix and then applies the `inner_subsolver`.

    `permutation_groups` denotes the submatrices to permute. For instance,
    ```
    [
        [MassBalanceMatrix(), MassBalanceFractures(), MassBalanceInterfaces()],
        [EnergyBalanceAllSubdomains()]
    ]
    ``` will interleave all values of mass balance and energy balance submatrices.

    """

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
    """PETSc point-block jacobi preconditioner. See:
    https://petsc.org/release/manualpages/PC/PCBJACOBI/

    """

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
        "approximate_inverter": subsolvers[0]["approximate_inverter"],
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
            complement_solver=nested_schur_complements(subsolvers=subsolvers[1:]),
            **kwargs,
        )
    # End of recursion.
    return FieldSplitSchur(complement_solver=subsolvers[1]["subsolver"], **kwargs)


# MARK: Factories


def mass_balance_factory():
    """This configures an iterative linear solver for the single-phase flow model in
    fractured porous media.

    The solver configuration can be customized by passing petsc options without a prefix
    to the PorePy model `params` dictionary as follows:
    ```
    params = {
        "linear_solver": {
            "options": {
                "gmres": {
                    # Customize the Krylov subspace solver. Example:
                    "ksp_gmres_restart": 200,
                },
                "interface_flow": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "mass_balance_amg": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                }
            }
        }
    }
    ```
    Refer to PETSc documentation to see the possible options:
    https://petsc.org/main/manual/ksp/#preconditioners

    """
    interface_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        WellFluxGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [MassBalancePressureGroup()]

    return GMRES(
        preconditioner=FieldSplitSchur(
            subsolver=ILU(groups=interface_groups, key="interface_flow"),
            complement_solver=AMG(groups=mass_balance_groups, key="mass_balance_amg"),
            approximate_inverter=DiagonalInverter(),
        )
    )


def momentum_balance_factory():
    """This configures an iterative linear solver for the contact mechanics model in
    fractured porous media.

    The solver configuration can be customized by passing petsc options without a prefix
    to the PorePy model `params` dictionary as follows:
    ```
    params = {
        "linear_solver": {
            "options": {
                "gmres": {
                    # Customize the Krylov subspace solver. Example:
                    "ksp_gmres_restart": 200,
                },
                "contact": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "mechanics_amg": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                }
            }
        }
    }
    ```
    Refer to PETSc documentation to see the possible options:
    https://petsc.org/main/manual/ksp/#preconditioners

    """
    contact_groups: list[EquationVariableGroup] = [ContactMechanicsGroup()]
    mechanics_groups: list[EquationVariableGroup] = [
        MechanicsGroup(),
        InterfaceForceBalanceGroup(),
    ]
    return GMRES(
        preconditioner=FieldSplitSchur(
            # For clarity, the petsc_tag and key are different concepts.
            petsc_tag="contact",
            subsolver=BlockDiagonalPreconditioner(groups=contact_groups, key="contact"),
            complement_solver=AMG(
                groups=mechanics_groups, key="mechanics_amg", vector_problem=True
            ),
            approximate_inverter=BlockDiagonalInverter(),
        )
    )


def hm_factory():
    """This configures an iterative linear solver for the poromechanics model in
    fractured porous media.

    The solver configuration can be customized by passing petsc options without a prefix
    to the PorePy model `params` dictionary as follows:
    ```
    params = {
        "linear_solver": {
            "options": {
                "gmres": {
                    # Customize the Krylov subspace solver. Example:
                    "ksp_gmres_restart": 200,
                },
                "contact": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "mechanics_amg": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                },
                "interface_flow": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "mass_balance_amg": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                }
            }
        }
    }
    ```
    Refer to PETSc documentation to see the possible options:
    https://petsc.org/main/manual/ksp/#preconditioners

    """
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
                    "approximate_inverter": BlockDiagonalInverter(),
                    "petsc_tag": "contact",
                },
                {
                    "subsolver": ILU(
                        groups=interface_flux_groups, key="interface_flow"
                    ),
                    "approximate_inverter": DiagonalInverter(),
                    "petsc_tag": "intf_darcy_flux",
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_inverter": FixedStressInverter(),
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
    """This configures an iterative linear solver for the poromechanics model in
    fractured porous media.

    The solver configuration can be customized by passing petsc options without a prefix
    to the PorePy model `params` dictionary as follows:
    ```
    params = {
        "linear_solver": {
            "options": {
                "gmres": {
                    # Customize the Krylov subspace solver. Example:
                    "ksp_gmres_restart": 200,
                },
                "interface_flow": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "cpr0_energy": {
                    # customize the energy-balance sub-solver.
                    "pc_type": "pbjacobi",
                },
                "cpr0_mass": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                }
                "cpr1": {
                    # customize the coupled mass-energy sub-solver.
                    "pc_type": "sor",
                }
            }
        }
    }
    ```
    Refer to PETSc documentation to see the possible options:
    https://petsc.org/main/manual/ksp/#preconditioners

    """
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
            approximate_inverter=DiagonalInverter(),
            complement_solver=CompositePreconditioner(
                subsolvers=[
                    FieldSplitSchur(
                        subsolver=Identity(
                            groups=energy_balance_groups, key="cpr0_energy"
                        ),
                        complement_solver=AMG(
                            groups=mass_balance_groups, key="cpr0_mass"
                        ),
                        approximate_inverter=DiagonalInverter(),
                    ),
                    ILU(groups=energy_balance_groups + mass_balance_groups, key="cpr1"),
                ]
            ),
        )
    )


def thm_factory():
    """This configures an iterative linear solver for the poromechanics model in
    fractured porous media.

    The solver configuration can be customized by passing petsc options without a prefix
    to the PorePy model `params` dictionary as follows:
    ```
    params = {
        "linear_solver": {
            "options": {
                "gmres": {
                    # Customize the Krylov subspace solver. Example:
                    "ksp_gmres_restart": 200,
                },
                "contact": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "mechanics_amg": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                },
                "interface_flow": {
                    # customize the interface flow sub-solver.
                    "pc_type": "sor",
                },
                "cpr0_energy": {
                    # customize the energy-balance sub-solver.
                    "pc_type": "pbjacobi",
                },
                "cpr0_mass": {
                    # customize the mass-balance sub-solver.
                    "pc_hypre_boomeramg_strong_threshold": 0.6,
                }
                "cpr1": {
                    # customize the coupled mass-energy sub-solver.
                    "pc_type": "sor",
                }
            }
        }
    }
    ```
    Refer to PETSc documentation to see the possible options:
    https://petsc.org/main/manual/ksp/#preconditioners

    """
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
                    "approximate_inverter": BlockDiagonalInverter(),
                    "petsc_tag": "contact",
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_inverter": DiagonalInverter(),
                    "petsc_tag": "intf_mass_energy_flx",
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_inverter": FixedStressInverter(),
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


def cfle_factory():
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
        preconditioner=FieldSplitSchur(
            subsolver=ILU(groups=interface_groups, key="interface_prec"),
            approximate_inverter=DiagonalInverter(),
            complement_solver=CompositePreconditioner(
                subsolvers=[
                    FieldSplitSchur(
                        subsolver=Identity(
                            groups=energy_balance_groups + component_groups,
                            key="cpr_stage0_identity",
                        ),
                        approximate_inverter=DiagonalInverter(),
                        complement_solver=AMG(
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
