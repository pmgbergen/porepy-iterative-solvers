"""This module contains classes that describe the components of the PETSc KSP and PC.
These classes do not produce PETSc options by themselves, they instead generate a dict
of PETSc options, and a dict of instruction, used to assemble PETSc objects in
`options_parser.py`.

This module also defines the default linear solver configurations for PorePy models.

"""

from __future__ import annotations

from collections import defaultdict
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    CustomEquationVariableGroup,
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
from pp_solvers.transformations import (
    ContactLinearTransformation,
    LinearSystemTransformation,
    ScaleSpecificVolume,
)

logger = logging.getLogger(__name__)

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
    "FieldSplit",
    "FieldSplitSchur",
    "PythonPermutationWrapper",
    "DiagonalPreconditioner",
    "BlockDiagonalPreconditioner",
    # Add all the factory functions here.
    "mass_balance_factory",
    "momentum_balance_factory",
    "hm_factory",
    "th_factory",
    "thm_factory",
    "thm_tpsa_factory",
]

PETSC_OPTIONS_MAX_SYMBOLS = 126


def append_prefix_to_options(prefix: str, options: dict):
    results = {}
    for key, value in options.items():
        new_key = f"{prefix}_{key}"
        if len(new_key) > PETSC_OPTIONS_MAX_SYMBOLS:
            raise ValueError(
                f"PETSc options key {new_key} is larger than "
                f"{PETSC_OPTIONS_MAX_SYMBOLS} symbols."
            )
        results[new_key] = value
    return results


# MARK: Inverters


class PetscInverter(ABC):
    """The base class for the customizable inverter instruction for the
    `FieldSplitSchur` class.

    """

    @abstractmethod
    def petsc_options(
        self, key: str, elim_key: str, complement_key: str, dof_manager: DofManager
    ) -> dict:
        """Return PETSc options for the approximate Schur complement inverter.

        Parameters:
            key: PETSc prefix of the owning `FieldSplitSchur` node. Options such as
                `pc_fieldsplit_schur_precondition` are scoped to this prefix.
            elim_key: PETSc prefix of the eliminated block sub-solver (A00).
            complement_key: PETSc prefix of the Schur complement sub-solver (S11).
                Options that configure the unassembled Schur complement matrix are
                scoped here.
            dof_manager: Used to resolve model properties (e.g. ambient dimension).

        Returns:
            Dict of PETSc CLI options keyed by their fully-prefixed option name.
        """

    def petsc_assembly_config(self, dof_manager: DofManager) -> dict:
        return {}

    def __str__(self) -> str:
        return type(self).__name__


class NoInverter(PetscInverter):
    """Schur complement inverter that does not approximate the inverse of A00.

    Sets ``pc_fieldsplit_schur_precondition`` to ``a11``, so PETSc approximates the
    S11 with A11, ignoring the A10 * inv(A00) * A01 correction.

    """

    def petsc_options(
        self, key: str, elim_key: str, complement_key: str, dof_manager: DofManager
    ) -> dict:
        return append_prefix_to_options(
            prefix=key,
            options={
                "pc_fieldsplit_schur_precondition": "a11",
            },
        )


class DiagonalInverter(PetscInverter):
    """Schur complement inverter that approximates inv(A00) by its inverse diagonal.

    Sets ``pc_fieldsplit_schur_precondition`` to ``selfp``, so PETSc assembles the Schur
    complement approximation S11_approx = A11 - A10 * inv(diag(A00)) * A01. This is the
    cheapest assembled approximation; for block-structured A00 use
    `BlockDiagonalInverter` instead.

    """

    def petsc_options(
        self, key: str, elim_key: str, complement_key: str, dof_manager: DofManager
    ) -> dict:
        return append_prefix_to_options(
            prefix=key,
            options={
                "pc_fieldsplit_schur_precondition": "selfp",
            },
        )


class BlockDiagonalInverter(PetscInverter):
    """Block-diagonal inverter for the eliminated block A00 in a Schur complement split.

    The block size must match that of A00. For example, if the block size is 3,
    the eliminated ILU sub-solver must also be configured with block size 3.

    Parameters:
        block_size: By default, uses the PorePy model ambient dimension.

    """

    def __init__(self, block_size: Optional[int] = None) -> None:
        self.block_size: Optional[int] = block_size

    def petsc_options(
        self, key: str, elim_key: str, complement_key: str, dof_manager: DofManager
    ) -> dict:
        bs = dof_manager.model.nd if self.block_size is None else self.block_size

        # Schur complement produces two matrices: A00 (what we eliminate) and S11 (what
        # we keep). S11 is unassembled, and its approximation needs to be assembled.
        # We pass to S11 the option "mat_schur_complement_ainv_type": "blockdiag", which
        # means S11_approx = A11 - A10 * inv_bdiag(A00) * A01. Matrix A00 inverse is
        # approximated by its block diagonal. We need to pass the block size to A00.

        # Passing this to S11.
        keep_options = {"mat_schur_complement_ainv_type": "blockdiag"}
        # Passing this to A00.
        elim_options = {"mat_block_size": bs}
        # Passing this to the fieldsplit object - a parent of both S11 and A00.
        fieldsplit_options = {"pc_fieldsplit_schur_precondition": "selfp"}

        # Now some hacking happens. When PETSc creates two sub-solvers for the
        # fieldsplit, it initially assigns them prefixes in the format
        # "{fieldsplit_key}_fieldsplit_{subsolver_key}". When nesting multiple
        # fieldsplits, it leads to unreadable prefixes fieldsplit_fieldsplit_fieldsplit_
        # We assign custom, non-nesting prefixes based on our keys. The problem is in
        # the sequence:
        # - default prefixes create in pc.setFieldSplitIS(...)
        # - sub-solvers are initialized and S11 approximation is assembled in pc.setUp()
        # There is no access point to customize a prefix of a sub-solver in the middle
        # of these two actions, neither from Python nor C. PETSc must fetch the options
        # using default prefixes. We provide identical options both with the initial
        # prefixes and the customized ones for completeness. This hack is covered with
        # a unit test, see `test_options_parsers.py/test_block_diagonal_invertor`.
        initial_prefix_keep = f"{key}_fieldsplit_{complement_key}"
        initial_prefix_elim = f"{key}_fieldsplit_{elim_key}"
        return (
            append_prefix_to_options(prefix=key, options=fieldsplit_options)
            | append_prefix_to_options(prefix=complement_key, options=keep_options)
            | append_prefix_to_options(prefix=initial_prefix_keep, options=keep_options)
            | append_prefix_to_options(prefix=elim_key, options=elim_options)
            | append_prefix_to_options(prefix=initial_prefix_elim, options=elim_options)
        )


class FixedStressInverter(PetscInverter):
    def petsc_options(
        self, key: str, elim_key: str, complement_key: str, dof_manager: DofManager
    ) -> dict:
        return append_prefix_to_options(
            prefix=key,
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
        if len(key) > PETSC_OPTIONS_MAX_SYMBOLS:
            raise ValueError(
                f"Key {key} is used as PETSc prefix and must be smaller than "
                f"{PETSC_OPTIONS_MAX_SYMBOLS} symbols."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(groups={self.groups})"

    @abstractmethod
    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        """Return a flat dict of PETSc CLI options for this solver and all sub-solvers.

        Each option key is prefixed with the `key` of the node that owns it, e.g. an
        option ``pc_type`` on a node with ``key="mechanics"`` becomes
        ``"mechanics_pc_type"``. Sub-solver options are included recursively, so the
        returned dict covers the full solver hierarchy.

        Parameters:
            user_options: Per-node overrides keyed by `PetscKspPcConfiguration.key`.
                Format::

                    {
                        "mechanics": {"pc_type": "gamg"},
                        "flow":      {"ksp_type": "preonly"},
                    }

                Options should be provided without a prefix - the correct prefix is
                derived from the node's key automatically.
            dof_manager: DOF manager for the problem; used to resolve group-to-DOF
                mappings and model properties (e.g. ambient dimension).

        Returns:
            Flat dict of fully-prefixed PETSc CLI options ready to pass to
            `insert_petsc_options`.
        """

    def petsc_assembly_config(
        self, user_options: dict, dof_manager: DofManager
    ) -> dict:
        """Return the assembly config consumed by `assemble_petsc_ksp_pc`.

        The config is a flat dict keyed by node `key` values. Each entry describes
        how `assemble_petsc_ksp_pc` should set up the corresponding KSP/PC pair.
        Non-leaf nodes include the configs of their sub-solvers, so the returned
        dict covers the full hierarchy. Format::

            {
                "root_key": {
                    "config_type": "fieldsplit_schur",
                    "elim_key": "mechanics",
                    "keep_key": "flow",
                    ...
                },
                "mechanics": {
                    "config_type": "composite",
                    ...
                },
                ...
            }

        Parameters:
            user_options: Per-node overrides. See `petsc_options` for the format.
            dof_manager: DOF manager for the problem.

        Returns:
            Flat dict mapping each node's key to its assembly sub-config dict.
        """
        return {}

    def get_children(self) -> list[PetscKspPcConfiguration]:
        """Return the direct sub-solver nodes of this configuration node.

        Used to traverse the solver tree, e.g., to validate that all node keys are
        unique. Leaf nodes (ILU, AMG, etc.) return an empty list; composite nodes
        override this to return their sub-solvers.
        """
        return []


class ILU(PetscKspPcConfiguration):
    """PETSc implementation of ILU, see for additional options:
    https://petsc.org/release/manualpages/PC/PCILU/

    """

    def __init__(self, groups: list[EquationVariableGroup], key: str = "ilu") -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        default_options = {"pc_type": "ilu"}
        return append_prefix_to_options(
            prefix=self.key,
            options=default_options | user_options.get(self.key, {}),
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

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
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
            prefix=self.key,
            options=default_options | user_options.get(self.key, {}),
        )


class Identity(PetscKspPcConfiguration):
    """A dummy preconditioner that does nothing."""

    def __init__(
        self, groups: list[EquationVariableGroup], key: str = "identity"
    ) -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        default_options = {"pc_type": "none"}
        return append_prefix_to_options(
            prefix=self.key,
            options=default_options | user_options.get(self.key, {}),
        )


class GMRES(PetscKspPcConfiguration):
    """PETSc implementation of GMRES. By default, estimates convergence based on the
    unpreconditioned residual norm. See for more options:
    https://petsc.org/release/manualpages/KSP/KSPGMRES/

    Implementation note: This class breaks an otherwise convenient assumption that a
    single node (`PetscKspPcConfiguration`) configures both the ksp and pc objects that
    operate on the same matrix and share a key. The convenience is due to:
    - `AMG()` initializes a sub-solver with no KSP and an AMG PC. We don't have to write
        something like `KspNone(pc=AMG())` every time;
    - If we ever need to reinforce the subsolver `AMG()` with a KSP, this is done from
        user options, e.g., `{'amg': {'ksp_type': 'bcgs'}}`.

    GMRES is the edge case: `GMRES(preconditioner=AMG())`. Since the KSP and the PC must
    share the same PETSc prefix, we override the PC key with the KSP key. Therefore,
    this is a mistake:
    `{'gmres': {'gmres_param_1': 'val1'}, 'amg': {'amg_param_1': 'val1'}}`. The correct
    syntax is: `{'gmres': {'gmres_param_1': 'val1', 'amg_param_1': 'val1'}}`. This error
    is not drastic, since we log that the `amg_param_1` is not read by PETSc.

    Benefit of this approach is that it won't lead to unwanted collisions, such as:
    `{'gmres': {'pc_type': 'ilu'}, 'amg': {'ksp_type}}`. YZ does not like this approach,
    but finds it compromisable, unless others find that it brings more chaos than good.

    """

    def __init__(
        self, preconditioner: PetscKspPcConfiguration, key: str = "gmres"
    ) -> None:
        self.preconditioner: PetscKspPcConfiguration = preconditioner
        self.preconditioner.key = key  # Read the class implementation note.
        super().__init__(groups=self.preconditioner.groups, key=key)

    def __repr__(self) -> str:
        return f"GMRES(preconditioner={self.preconditioner})"

    def get_children(self) -> list[PetscKspPcConfiguration]:
        # Read the class implementation note. The KSP and its preconditioner act as a
        # single configuration node, and we reflect it in the children tree.
        return self.preconditioner.get_children()

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
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
            dof_manager=dof_manager,
        )

        this_user_options = user_options.get(self.key, {})

        ksp_options = append_prefix_to_options(
            prefix=self.key, options=default_options | this_user_options
        )

        return ksp_options | pc_options

    def petsc_assembly_config(
        self, user_options: dict, dof_manager: DofManager
    ) -> dict:
        return self.preconditioner.petsc_assembly_config(
            user_options=user_options, dof_manager=dof_manager
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
        _validate_subsolvers_keys_are_unique(
            subsolvers=subsolvers,
            current_node_repr=f"CompositePreconditioner({key = })",
        )

        super().__init__(groups_of_subsolvers[0], key=key)
        self.subsolvers: list[PetscKspPcConfiguration] = subsolvers

    def get_children(self) -> list[PetscKspPcConfiguration]:
        return self.subsolvers

    def __repr__(self) -> str:
        return f"CompositePreconditioner(key={self.key}, subsolvers={self.subsolvers})"

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        own_options = {
            "pc_type": "composite",
            "pc_composite_type": "multiplicative",
        }
        result = append_prefix_to_options(
            prefix=self.key, options=own_options | user_options.get(self.key, {})
        )
        for subsolver in self.subsolvers:
            result[f"{subsolver.key}_ksp_type"] = "preonly"
            result |= subsolver.petsc_options(
                user_options=user_options,
                dof_manager=dof_manager,
            )
        return result

    def petsc_assembly_config(
        self, user_options: dict, dof_manager: DofManager
    ) -> dict:
        config = {
            self.key: {
                "config_type": "composite",
                "subsolver_keys": [subsolver.key for subsolver in self.subsolvers],
            },
        }
        for subsolver in self.subsolvers:
            config |= subsolver.petsc_assembly_config(
                user_options=user_options,
                dof_manager=dof_manager,
            )
        return config


class FieldSplit(PetscKspPcConfiguration):
    """A preconditioner that splits the problem into n sub-problems and treats each
    separately with a sub-solver. See:
    https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/

    """

    def __init__(
        self,
        subsolvers: Sequence[PetscKspPcConfiguration],
        key: Optional[str] = None,
        fieldsplit_type: Literal[
            "additive", "multiplicative", "symmetric_multiplicative"
        ] = "additive",
    ) -> None:
        # PETSc accepts more fieldsplit types than this class supports.
        if fieldsplit_type == "schur":
            raise ValueError("Use class FieldSplitSchur instead.")
        if fieldsplit_type == "gkb":
            logger.warning("FieldSplit type gkb not tested, use on your own risk.")
        self.fieldsplit_type: Literal[
            "additive", "multiplicative", "symmetric_multiplicative"
        ] = fieldsplit_type

        if key is None:
            key = f"fs_{subsolvers[0].key}"
        self.subsolvers: list[PetscKspPcConfiguration] = list(subsolvers)
        super().__init__(
            groups=[g for subsolver in self.subsolvers for g in subsolver.groups],
            key=key,
        )

        if len(set(self.groups)) < len(self.groups):
            # Non-unique groups are present.
            raise ValueError(f"Groups in FieldSplit should not overlap: {self.groups}")

        _validate_subsolvers_keys_are_unique(
            subsolvers=self.subsolvers,
            current_node_repr=f"FieldSplit({key = })",
        )

    def __repr__(self) -> str:
        return (
            f"FieldSplit(fieldsplit_type={self.fieldsplit_type}, key={self.key}, "
            f"subsolvers={self.subsolvers})"
        )

    def get_children(self) -> list[PetscKspPcConfiguration]:
        return self.subsolvers

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        own_options = {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": self.fieldsplit_type,
        }
        result = append_prefix_to_options(
            prefix=self.key, options=own_options | user_options.get(self.key, {})
        )
        for subsolver in self.subsolvers:
            result[f"{subsolver.key}_ksp_type"] = "preonly"
            result |= subsolver.petsc_options(
                user_options=user_options,
                dof_manager=dof_manager,
            )

        return result

    def petsc_assembly_config(
        self, user_options: dict, dof_manager: DofManager
    ) -> dict:
        result = {
            self.key: {
                "config_type": "fieldsplit_common",
                "subsolver_groups": [
                    dof_manager.indices_of_groups(subsolver.groups)
                    for subsolver in self.subsolvers
                ],
                "subsolver_keys": [subsolver.key for subsolver in self.subsolvers],
            }
        }
        for subsolver in self.subsolvers:
            result.update(
                subsolver.petsc_assembly_config(
                    user_options=user_options,
                    dof_manager=dof_manager,
                )
            )
        return result


class FieldSplitSchur(PetscKspPcConfiguration):
    """A preconditioner that applies PETSc PCFIELDSPLIT with a Schur complement
    factorization to a 2x2 block system.  See:
    https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/

    Given the block matrix::

        [[A, B],
         [C, D]]

    the Schur complement is ``S = D - C A^-1 B``. This class sets
    ``pc_fieldsplit_type = schur`` with an upper factorization, so each
    KSP sub-solve uses ``ksp_type = preonly``.

    The PETSc options prefix for each sub-solver is taken from
    ``subsolver.key`` and ``complement_solver.key`` respectively, so
    those keys must be unique within the solver tree and short enough
    to stay within PETSc's 127-character prefix limit.

    Parameters:
        subsolver: Configuration for the solver that approximates ``A^-1``.
            Its ``groups`` define which DOFs are eliminated.
        complement_solver: Configuration for the solver that approximates
            ``S^-1``. Its ``groups`` must be disjoint from ``subsolver.groups``.
        approximate_inverter: Strategy for constructing the approximate Schur complement
            ``S``.
        key: PETSc options prefix for this preconditioner. Defaults to
            ``f"fs_{subsolver.key}"``.

    Raises:
        ValueError: If ``subsolver.groups`` and ``complement_solver.groups``
            overlap, or if the two sub-solvers share a key anywhere in the
            solver tree.
    """

    def __init__(
        self,
        subsolver: PetscKspPcConfiguration,
        complement_solver: PetscKspPcConfiguration,
        approximate_inverter: PetscInverter,
        key: Optional[str] = None,
    ) -> None:
        if key is None:
            key = f"fs_{subsolver.key}"
        self.subsolver: PetscKspPcConfiguration = subsolver
        self.complement_solver: PetscKspPcConfiguration = complement_solver
        self.approximate_inverter: PetscInverter = approximate_inverter
        super().__init__(
            groups=self.subsolver.groups + self.complement_solver.groups, key=key
        )

        intersection = set(self.subsolver.groups).intersection(
            self.complement_solver.groups
        )
        if len(intersection) > 0:
            raise ValueError(f"Groups in FielSplit should not overlap: {intersection}")

        _validate_subsolvers_keys_are_unique(
            subsolvers=[self.subsolver, self.complement_solver],
            current_node_repr=f"FieldSplitSchur(key={self.key})",
        )

    def get_children(self) -> list[PetscKspPcConfiguration]:
        return [self.subsolver, self.complement_solver]

    def __repr__(self) -> str:
        return (
            f"FieldSplit(key={self.key}, subsolver={self.subsolver}, "
            f"complement_solver={self.complement_solver}, "
            f"approximate_inverter={self.approximate_inverter})"
        )

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        subsolver_prefix = f"{self.subsolver.key}_"
        complement_prefix = f"{self.complement_solver.key}_"
        own_options = {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "upper",
        }
        result = append_prefix_to_options(prefix=self.key, options=own_options)
        result[f"{subsolver_prefix}ksp_type"] = "preonly"
        result[f"{complement_prefix}ksp_type"] = "preonly"
        result |= self.subsolver.petsc_options(
            user_options=user_options,
            dof_manager=dof_manager,
        )
        result |= self.complement_solver.petsc_options(
            user_options=user_options,
            dof_manager=dof_manager,
        )
        result |= append_prefix_to_options(
            prefix=self.key, options=user_options.get(self.key, {})
        )

        invertor_results = self.approximate_inverter.petsc_options(
            key=self.key,
            elim_key=self.subsolver.key,
            complement_key=self.complement_solver.key,
            dof_manager=dof_manager,
        )
        intersection = set(result).intersection(invertor_results)
        if len(intersection) > 0:
            for key in intersection:
                from_subsolvers = result[key]
                from_invertor = invertor_results[key]
                if from_subsolvers != from_invertor:
                    raise ValueError(
                        "FieldSplitSchur invertor options override solver options: "
                        f"{intersection}. Value from sub-solvers: {from_subsolvers}, "
                        f"value from invertor: {from_invertor}."
                    )

        result |= invertor_results
        return result

    def petsc_assembly_config(
        self, user_options: dict, dof_manager: DofManager
    ) -> dict:
        return (
            {
                self.key: {
                    "config_type": "fieldsplit_schur",
                    "elim_key": self.subsolver.key,
                    "keep_key": self.complement_solver.key,
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
                dof_manager=dof_manager,
            )
            | self.complement_solver.petsc_assembly_config(
                user_options=user_options,
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

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        result = append_prefix_to_options(
            prefix=self.key,
            options={"pc_type": "python"} | user_options.get(self.key, {}),
        )
        result |= self.inner_subsolver.petsc_options(
            user_options=user_options,
            dof_manager=dof_manager,
        )
        return result
        # what if user options change pc_type? We assume it is prohibited. Somewhere it
        # should be checked.

    def petsc_assembly_config(
        self, user_options: dict, dof_manager: DofManager
    ) -> dict:
        inner_config = self.inner_subsolver.petsc_assembly_config(
            user_options=user_options,
            dof_manager=dof_manager,
        )
        if len(inner_config) > 0:
            raise NotImplementedError(
                "Nested initialization inside PythonPermutationWrapper is not "
                "implemented."
            )
        return {
            self.key: {
                "config_type": "python_permutation",
                "permutation_groups": [
                    dof_manager.indices_of_groups(g) for g in self.permutation_groups
                ],
                "inner_key": self.inner_subsolver.key,
            }
        }


class DiagonalPreconditioner(PetscKspPcConfiguration):
    """PETSc Jacobi (diagonal) preconditioner. See:
    https://petsc.org/release/manualpages/PC/PCJACOBI/

    """

    def __init__(
        self, groups: list[EquationVariableGroup], key: str = "diagonal"
    ) -> None:
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        default_options = {
            "pc_type": "jacobi",
        }
        return append_prefix_to_options(
            prefix=self.key,
            options=default_options | user_options.get(self.key, {}),
        )


class BlockDiagonalPreconditioner(PetscKspPcConfiguration):
    """PETSc point-block jacobi preconditioner. See:
    https://petsc.org/release/manualpages/PC/PCBJACOBI/

    By default, sets ``mat_block_size`` to the model's ambient dimension
    (``model.nd``, e.g., 2 for 2D or 3 for 3D).

    """

    def __init__(
        self,
        groups: list[EquationVariableGroup],
        key: str = "block_diagonal",
        block_size: Optional[int] = None,
    ) -> None:
        self.block_size: Optional[int] = block_size
        super().__init__(groups=groups, key=key)

    def petsc_options(self, user_options: dict, dof_manager: DofManager) -> dict:
        bs = dof_manager.model.nd if self.block_size is None else self.block_size
        default_options = {
            "pc_type": "pbjacobi",
            "mat_block_size": bs,
        }
        return append_prefix_to_options(
            prefix=self.key,
            options=default_options | user_options.get(self.key, {}),
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

    if len(subsolvers) > 2:
        # Recursion.
        return FieldSplitSchur(
            complement_solver=nested_schur_complements(subsolvers=subsolvers[1:]),
            **kwargs,
        )
    # End of recursion.
    return FieldSplitSchur(complement_solver=subsolvers[1]["subsolver"], **kwargs)


@dataclass
class LinearSolverConfiguration:
    solver: PetscKspPcConfiguration
    transformations: list[LinearSystemTransformation] = field(
        default_factory=lambda: []
    )
    groups: list[EquationVariableGroup] = field(default_factory=lambda: [])

    def __post_init__(self):
        if len(self.groups) == 0:
            self.groups = self.solver.groups


# MARK: Validation


def _validate_subsolvers_keys_are_unique(
    subsolvers: list[PetscKspPcConfiguration], current_node_repr: str
):
    """This function is different from `validate_all_keys_are_unique`, since it does not
    do recursion. It is applied at __init__ configurations with children and raises
    early, so the stack trace indicates exactly where the duplicate is."""
    count_subsolver_keys = defaultdict(lambda: 0)
    for subsolver in subsolvers:
        count_subsolver_keys[subsolver.key] += 1
    for subsolver_key, count in count_subsolver_keys.items():
        if count > 1:
            raise ValueError(
                f"{current_node_repr} subsolver key is non-unique: {subsolver_key}"
            )


def validate_all_keys_are_unique(head: PetscKspPcConfiguration):
    keys_nodes: dict[str, list[PetscKspPcConfiguration]] = defaultdict(lambda: [])

    def traverse_subtree(node: PetscKspPcConfiguration):
        keys_nodes[node.key].append(node)
        for child in node.get_children():
            traverse_subtree(child)

    traverse_subtree(head)

    for key, nodes in keys_nodes.items():
        if len(nodes) > 1:
            raise ValueError(
                f"Linear solver configuration {key = } must be unique. Currently used "
                f"in nodes:\n\n{'\n\n'.join(map(str, nodes))}."
            )


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

    solver = GMRES(
        preconditioner=FieldSplitSchur(
            subsolver=ILU(groups=interface_groups, key="interface_flow"),
            complement_solver=AMG(groups=mass_balance_groups, key="mass_balance_amg"),
            approximate_inverter=DiagonalInverter(),
        )
    )
    return LinearSolverConfiguration(solver=solver)


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
    solver = GMRES(
        preconditioner=FieldSplitSchur(
            subsolver=BlockDiagonalPreconditioner(groups=contact_groups, key="contact"),
            complement_solver=AMG(
                groups=mechanics_groups, key="mechanics_amg", vector_problem=True
            ),
            approximate_inverter=BlockDiagonalInverter(),
        )
    )
    return LinearSolverConfiguration(
        solver=solver,
        transformations=[
            ContactLinearTransformation(),
        ],
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

    solver = GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_inverter": BlockDiagonalInverter(),
                },
                {
                    "subsolver": ILU(
                        groups=interface_flux_groups, key="interface_flow"
                    ),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_inverter": FixedStressInverter(),
                },
                {
                    "subsolver": AMG(
                        groups=mass_balance_groups, key="mass_balance_amg"
                    ),
                },
            ]
        )
    )
    return LinearSolverConfiguration(
        transformations=[
            ContactLinearTransformation(),
        ],
        solver=solver,
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

    solver = GMRES(
        preconditioner=FieldSplitSchur(
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
    return LinearSolverConfiguration(
        transformations=[
            ScaleSpecificVolume(groups=[EnergyBalanceTemperatureGroup()]),
        ],
        solver=solver,
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
                    # customize the contact mechanics flow sub-solver.
                    "pc_type": "pbjacobi",
                },
                "mechanics_amg": {
                    # customize the mechanics sub-solver.
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
                },
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

    solver = GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_inverter": BlockDiagonalInverter(),
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_inverter": FixedStressInverter(),
                },
                {
                    "subsolver": CompositePreconditioner(
                        subsolvers=[
                            FieldSplit(
                                subsolvers=[
                                    Identity(
                                        groups=energy_balance_groups,
                                        key="cpr0_energy",
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

    return LinearSolverConfiguration(
        transformations=[
            ContactLinearTransformation(),
            ScaleSpecificVolume(groups=[EnergyBalanceTemperatureGroup()]),
        ],
        solver=solver,
    )


def thm_tpsa_factory():
    """
    Based on https://doi.org/10.1007/s10596-026-10419-4. Differences:
    - It does not split the elastiticy equation and displacement variables into 3
    components (in 3D) and does not solve these 3 subproblems separately. Instead, a
    single AMG instance is applied. Testing on small problems showed no performance
    difference. The difference may become notable for larger or heavily anisotropic
    problems.
    - We do not scale variables in the preconditioner. We rely on PorePy scaling.

    """
    contact_groups: list[EquationVariableGroup] = [ContactMechanicsGroup()]
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

    solid_mass_pressure_group = CustomEquationVariableGroup(
        "Solid_mass_equation_poromechanics", "total_pressure"
    )
    angular_momentum_rotation_group = CustomEquationVariableGroup(
        "angular_momentum_balance_equation", "rotation_stress"
    )

    solver = GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_inverter": BlockDiagonalInverter(),
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": DiagonalPreconditioner(
                        groups=[InterfaceForceBalanceGroup()], key="intf_force_balance"
                    ),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": FieldSplit(
                        key="tpsa_fieldsplit",
                        fieldsplit_type="multiplicative",
                        subsolvers=[
                            AMG(
                                groups=[solid_mass_pressure_group],
                                key="solid_mass_pressure_amg",
                                vector_problem=False,
                            ),
                            DiagonalPreconditioner(
                                groups=[angular_momentum_rotation_group],
                                key="angular_momentum_rotation",
                            ),
                            AMG(
                                groups=[MechanicsGroup()],
                                key="mechanics_amg",
                                vector_problem=True,
                            ),
                        ],
                    ),
                    "approximate_inverter": FixedStressInverter(),
                },
                {
                    "subsolver": CompositePreconditioner(
                        key="mass_energy_cpr",
                        subsolvers=[
                            FieldSplit(
                                subsolvers=[
                                    DiagonalPreconditioner(
                                        groups=energy_balance_groups,
                                        key="cpr0_energy",
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
                                    key="cpr1_ilu",
                                ),
                                key="cpr1_permutation",
                            ),
                        ],
                    )
                },
            ]
        )
    )

    return LinearSolverConfiguration(
        transformations=[
            ContactLinearTransformation(),
            # SchurComplementReduction(primary_groups=solver.groups),
            ScaleSpecificVolume(groups=[EnergyBalanceTemperatureGroup()]),
        ],
        solver=solver,
        groups=solver.groups,
    )
