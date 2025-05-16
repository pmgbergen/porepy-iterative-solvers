"""This module contains schemes, e.g., recepies for constructing a PETSc solver."""

from __future__ import annotations

import numpy as np
from typing import Callable
from dataclasses import dataclass
import porepy as pp
from abc import ABC, abstractmethod

from .block_matrix import BlockMatrixStorage
from .full_petsc_solver import construct_is, PetscKSPScheme, insert_petsc_options

from . import hm_solver
from .iterative_solver import (
    get_equations_group_ids,
    get_variables_group_ids,
)
from .mat_utils import csr_ones, inv_block_diag

from petsc4py import PETSc


__all__ = [
    "MassBalanceGroup",
    "InterfaceFluxGroup",
    "MechanicsGroup",
    "ContactGroup",
    "DofManager",
    "SinglePhysicsPreconditioner",
    "InterfaceDarcyFluxPreconditioner",
    "MassBalancePreconditioner",
    "MechanicsPreconditioner",
    "ContactPreconditioner",
    "MultiPhysicsPreconditioner",
    "IterativeSolverMixin",
    "mass_balance_factory",
]

"""Below are methods that are used to create specific schemes for different equations.
Note that these consider PETSc configurations, and have no responsibility for
taking care of equations etc. (CURRENT IMPLEMENTATION IS NOT RIGHT). This means they are
essentially bearers of options for the solver.
"""


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


class MassBalanceGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains()
        return [[("mass_balance_equation", subdomains)]]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [[model.pressure(subdomains)]]


class InterfaceFluxGroup(AbstractGroup):
    _variables = ["interface_darcy_flux"]
    _equations = ["interface_darcy_flux_equation"]

    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [[("interface_darcy_flux_equation", interfaces)]]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [[model.interface_darcy_flux(interfaces)]]


class MechanicsGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)

        # Define two groups of equations, one for momentum balance in the matrix and one
        # for force balance on the highest-dimensional interfaces. The mechanics
        # preconditioner will treat these groups jointly.
        return [
            [("momentum_balance_equation", subdomains)],
            [("interface_force_balance_equation", interfaces)],
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


class ContactGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        # Define a single group of equations to be solved together: The normal and
        # tangential deformation equations for the contact mechanics.
        return [
            [
                ("normal_fracture_deformation_equation", subdomains),
                ("tangential_fracture_deformation_equation", subdomains),
            ]
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        # There is a single group of variables for the contact mechanics, which is the
        # contact traction.
        return [[model.contact_traction(subdomains)]]


class DofManager:
    """Takes care of translation of blocks and groups (from EquationSystem format) to
    block indices, as well as grouping the fine-scale dofs. Also reordering related to
    the contact problem.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so.
    """

    def __init__(
        self, equation_system: pp.EquationSystem, orderings: list[AbstractGroup]
    ):
        self._equation_system = equation_system
        self._orderings = orderings

        self._group_to_block_ids = {}

    def _group_id(self, group: AbstractGroup) -> int:
        return self._solver_groups[group.__class__]

    def petsc_is(
        self,
        current_group: AbstractGroup,
        other_groups: list[AbstractGroup],
        bmat: BlockMatrixStorage,
    ):
        # Not sure if this belongs here, but it is tempting to put it here and not in
        # the composer.

        # Indices of the block ids
        current_id = self._group_id(current_group.group())
        other_id = []
        for group in other_groups:
            # Get the block id for the group.
            other_id += self._group_id(group.group())

        current_is = construct_is(bmat, current_id)
        other_is = construct_is(bmat, other_id)
        return current_is, other_is

    def variable_groups(self, model):
        var_groups = []
        for group in self._orderings:
            var_groups += group.variable_groups(model)
        return get_variables_group_ids(model, var_groups)

    def identify_contact_group(self, model):
        # Identify the contact group in the equation groups
        for i, group in enumerate(self._orderings):
            if len(group.equation_groups(model)) == 0:
                continue
            for block in group.equation_groups(model):
                if block[0][0] == "normal_fracture_deformation_equation":
                    return i
        return -1

    def identify_u_intf_group(self, model):
        # Identify the interface group in the equation groups
        i = 0
        for group in self._orderings:
            if len(group.variable_groups(model)) == 0:
                continue
            for var in group.variable_groups(model):
                if var[0].name == model.interface_displacement_variable:
                    return i
                else:
                    i += 1
        return -1

    def equation_groups(self, model):
        """Get the equation groups for the model, in the form of a list of
        a list of numbers. If the contact group is present, it will be
        reordered so that the normal and tangential equations are together.
        """
        # Get the equation groups for the model (in name-domain format)

        # Mapping from the ordering (which represents the block solver/preconditioner)
        # to the combination of equation groups that the preconditioner will handle
        # jointly.
        solver_groups = {}

        equation_groups_by_name = []
        counter = 0
        for group in self._orderings:
            groups_loc = group.equation_groups(model)
            equation_groups_by_name += groups_loc
            solver_groups[group.__class__] = [
                i for i in range(counter, counter + len(groups_loc))
            ]

            counter += len(groups_loc)

        # Use the class name of the group as the key for the dictionary..
        # This is a bit of a hack, but it works for now.
        self._solver_groups = solver_groups

        # Convert to numbers (i.e., block ids).
        equation_groups_by_number = get_equations_group_ids(
            model, equation_groups_by_name
        )

        contact_group = self.identify_contact_group(model)
        # If there is no contact group, return the original equation groups.
        if contact_group == -1:
            return equation_groups_by_number

        reordered_groups = self._correct_contact_equations_groups(
            model, equation_groups_by_number, contact_group
        )
        return reordered_groups

    def eq_dofs_by_blocks(self, model):
        """Get the equation dofs for the model, in the form of a list of numbers,
        one per equation-domain pair. If the contact group is present, it will be
        reordered so that the normal and tangential equations for each fracture cell
        form a diganol block.
        """
        eq_dofs: list[np.ndarray] = []
        offset = 0
        for data in model.equation_system._equation_image_space_composition.values():
            local_offset = 0
            for dofs in data.values():
                eq_dofs.append(dofs + offset)
                local_offset += len(dofs)
            offset += local_offset

        contact_group = self.identify_contact_group(model)
        if contact_group > -1:
            # If there is no contact group, return the original equation dofs.
            return self._correct_contact_eq_dofs(model, eq_dofs, contact_group)

        return eq_dofs

    def _correct_contact_eq_dofs(
        self, model, unpermuted_eq_dofs: list[np.ndarray], contact_group: int
    ) -> list[np.ndarray | None]:
        """Rearrange the unknowns (row indices) so that the contact equations are in a
        cell-wise block structure.

        Parameters:
            unpermuted_eq_dofs: The unpermuted equation degrees of freedom.
            contact_group: The group index of the contact mechanics equations.

        Returns:
            The corrected equation degrees of freedom.

        See also:
            _correct_contact_equations_groups for rearrane of the equation blocks
                related to contact (as opposed to the individual dofs handled here).

        """
        # Short cut if no contact mechanics, hence no reordering.
        if len(self.equation_groups(model)[contact_group]) == 0:
            # Ignore mypy error, list[np.ndarray] is a subset of list[np.ndarray |
            # None].
            return unpermuted_eq_dofs  # type: ignore[return-value]

        # We assume that normal equations go first. TODO: Can we make this more robust,
        # or else put an assert here.
        normal_blocks = self.equation_groups(model)[contact_group]
        num_fracs = len(model.mdg.subdomains(dim=model.nd - 1))

        # EK: I believe this is an assumption that the tangential equations are right
        # after the normal equations.
        all_contact_blocks = [
            nb + i * num_fracs for i in range(2) for nb in normal_blocks
        ]

        eq_dofs_corrected: list[np.ndarray | None] = []
        # Add all equations that are not contact equations without any changes.
        for i, x in enumerate(unpermuted_eq_dofs):
            if i not in all_contact_blocks:
                eq_dofs_corrected.append(x)
            elif i in normal_blocks:
                eq_dofs_corrected.append(None)

        offset = unpermuted_eq_dofs[normal_blocks[0]][0]
        for nb in normal_blocks:
            # Create indices for the normal and tangential components of the contact.
            # There will be model.nd equations for each block.
            inds = offset + np.arange(unpermuted_eq_dofs[nb].size * model.nd)
            offset = inds[-1] + 1
            eq_dofs_corrected[nb] = np.array(inds)

        return eq_dofs_corrected

    def _correct_contact_equations_groups(
        self,
        model: pp.PorePyModel,
        equation_groups: list[list[int]],
        contact_group: int,
    ) -> list[list[int]]:
        """The block ordering from PorePy assigns different block indices to the normal
        and tangential components of the contact equations. This method corrects this
        indexing by assigning a single block index for each fracture.

        The method further adjusts the indices of the other equation groups to account
        for the reduced number of blocks.

        Parameters:
            equation_groups: The uncorrected equation groups.
            contact_group: The group index of the contact mechanics equations.

        Returns:
            The corrected equation groups.

        See also:
            _correct_contact_eq_dofs for rearrane of the individual dofs related to
                contact (as opposed to the equation blocks handled here).

        """
        if len(equation_groups[contact_group]) == 0:
            return equation_groups

        # Create a copy of the equation groups to avoid modifying the original.
        eq_groups_corrected = [x.copy() for x in equation_groups]

        num_fracs = len(model.mdg.subdomains(dim=model.nd - 1))
        # Index of the first block after the contact group. This and all subsequent
        # indexes will be reduced by the number of fractures (e.g., the number of
        # block equations that have been removed).
        block_after_contact = max(equation_groups[contact_group]) + 1

        # Change the number of blocks in the contact group to the number of fractures,
        # since we have merged the normal and tangential components.
        eq_groups_corrected[contact_group] = equation_groups[contact_group][:num_fracs]

        # For all other groups with block index after the contact group, reduce the
        # block index by the number of fractures.
        for blocks in eq_groups_corrected:
            for i in range(len(blocks)):
                if blocks[i] >= block_after_contact:
                    blocks[i] -= num_fracs

        return eq_groups_corrected

    def var_dofs_by_blocks(self, model) -> list[np.ndarray]:
        """Variable degrees of freedom (columns of the Jacobian) in the PorePy order
        (how they are arranged in the model).

        Returns:
            List of numpy arrays. Each array contains the global degrees of freedom for
                one variable on one grid and provides the fine-scale (actual column
                indices) of the variable.

        """
        var_dofs: list[np.ndarray] = []
        for var in model.equation_system.variables:
            var_dofs.append(model.equation_system.dofs_of([var]))
        return var_dofs

    def eq_rows_permutation(self, model):
        """Get a permutation vector for the full linear system of equations. This is
        used to reorder the equations so that the contact equations for single fracture
        cells form a diagonal block.

        If no contact group is present, the permutation vector is linear.

        See also eq_dofs_by_blocks, which is used to reorder contact equations within
        the equation block format.
        """
        contact_group = self.identify_contact_group(model)
        # If there is no contact group, return the original equation groups.
        if contact_group == -1:
            return np.arange(model.equation_system.num_dofs())

        return self.make_reorder_contact(model, contact_group)

    def make_reorder_contact(
        self, model: pp.PorePyModel, contact_group: int
    ) -> np.ndarray:
        """Permutate the contact mechanics equations to a cell-wise block structure.

        The PorePy arrangement is:

            [C_n^0, C_n^1, ..., C_n^K, C_y^0, C_z^0, C_y^1, C_z^1, ..., C_z^K, C_z^k],

        where `C_n` is a normal component, `C_y` and `C_z` are two tangential
        components. The superscript corresponds to cell index. We permute it to

            `[C_n^0, C_y^0, C_z^0, ..., C_n^K, C_y^K, C_z^K]`.

        Parameters:
            model: The PorePy model.
            contact_group: The group index of the contact mechanics equations.

        Raises:
            ValueError: If the model dimension is not 2 or 3.

        Returns:


        """
        reorder = np.arange(model.equation_system.num_dofs())

        # Short cut if no contact mechanics, hence no reordering.
        if len(self.equation_groups(model)[contact_group]) == 0:
            return reorder

        # Get the (fine-scale, not block(!)) dofs of the contact mechanics equations.
        dofs_contact = np.concatenate(
            [
                self.eq_dofs_by_blocks(model)[i]
                for i in self.equation_groups(model)[contact_group]
            ]
        )

        # The start and end indices of all contact mechanics equations.
        dofs_contact_start = dofs_contact[0]
        dofs_contact_end = dofs_contact[-1] + 1

        # The number of cells in the contact mechanics equations.
        num_contact_cells = len(dofs_contact) // model.nd

        # 2d and 3d have respectively 1 and 2 tangential components, hence the branch.
        if model.nd == 2:
            # Rearrange the dofs into cell-wise blocks.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells:]
            reorder[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1]
            ).ravel("F")
        elif model.nd == 3:
            # Do the same as in 2d, also for the second tangential component.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells::2]
            dofs_contact_2 = dofs_contact[num_contact_cells + 1 :: 2]
            reorder[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1, dofs_contact_2]
            ).ravel("F")
        else:
            raise ValueError("Model dimension must be 2 or 3.")
        return reorder


class SinglePhysicsPreconditioner(ABC):
    """
    Abstract class for defining a preconditioner.
    """

    @abstractmethod
    def group(self):
        """
        Return the group for the preconditioner.
        """
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        """
        Return the key for the preconditioner.
        """
        pass

    @property
    @abstractmethod
    def tag(self) -> str:
        """
        Return the tag for the preconditioner.
        """
        pass

    @property
    def complement_tag(self) -> str:
        """
        Return the tag for the complement of the preconditioner.
        """
        return self.tag + "_complement"

    @property
    def ksp_keep_use_pmat(self) -> bool:
        """
        Return whether to keep the preconditioner matrix.
        """
        return False

    @property
    def unit_block_size(self) -> bool:
        """
        Return the number of dimensions for the preconditioner.
        """
        return True

    def near_null_space(self, model: pp.PorePyModel) -> np.ndarray | None:
        """
        Return the near null space for the preconditioner.
        """
        return None

    @abstractmethod
    def _default_options(self) -> dict:
        """
        Return the default options for the preconditioner.
        """
        pass

    def _default_fieldsplit_options(self) -> dict:
        """Options for field splits. Provide a separate method for this, since it
        involves some boilerplate options.
        """
        opts = {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "upper",
            "pc_fieldsplit_schur_precondition": "selfp",
            f"fieldsplit_{self.tag}_ksp_type": "preonly",
            f"fieldsplit_{self.complement_tag}_ksp_type": "preonly",
        }
        return opts

    def configure(
        self,
        opts: dict | None = None,
        has_complement: bool = False,
    ) -> dict:
        default_opts = self._default_options()
        user_opts = opts.get(self.key, {})

        local_opts = default_opts | user_opts

        if has_complement:
            fieldsplit_opts = self._default_fieldsplit_options()

            # The local options need to be prefixed with the relevant fieldsplit tag.
            local_fieldsplit_opts = {
                f"fieldsplit_{self.tag}_{k}": v for k, v in local_opts.items()
            }
            return fieldsplit_opts | local_fieldsplit_opts
        else:
            return local_opts


class InterfaceDarcyFluxPreconditioner(SinglePhysicsPreconditioner):
    @property
    def key(self) -> str:
        return "interface_darcy_flux"

    def group(self):
        return InterfaceFluxGroup()

    @property
    def tag(self) -> str:
        return "interface_darcy_flux"

    def _default_options(self) -> dict:
        opts = {"pc_type": "ilu"}
        return opts

    def configure(self, opts: dict | None = None, has_complement: bool = False) -> dict:
        if not has_complement:
            raise ValueError(
                "The interface darcy flux preconditioner requires a complement."
            )
        return super().configure(opts, has_complement)


class MassBalancePreconditioner(SinglePhysicsPreconditioner):
    @property
    def key(self) -> str:
        return "mass_balance"

    @property
    def tag(self) -> str:
        return "mass_balance"

    def group(self):
        return MassBalanceGroup()

    def _default_options(self) -> dict:
        local_opts = {
            "pc_type": "gamg",
            "pc_gamg_threshold": 0.02,
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": 4,
            "mg_levels_pc_type": "sor",
        }
        return local_opts


class MechanicsPreconditioner(SinglePhysicsPreconditioner):
    @property
    def key(self) -> str:
        return "mechanics"

    @property
    def tag(self) -> str:
        return "mechanics"

    def group(self):
        return MechanicsGroup()

    @property
    def unit_block_size(self) -> bool:
        return False

    def _default_options(self) -> dict:
        local_opts = {
            "pc_type": "hypre",
            "hmg_inner_pc_type": "gamg",
            "hmg_inner_pc_gamg_threshold": 0.02,
            # "hmg_inner_pc_hypre_type": "boomeramg",
            # "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": 2,
            "mg_levels_pc_type": "ilu",
        }
        return local_opts

    def near_null_space(self, model):
        return hm_solver.build_mechanics_near_null_space(model)


class ContactPreconditioner(SinglePhysicsPreconditioner):
    @property
    def key(self) -> str:
        return "contact"

    @property
    def tag(self) -> str:
        return "contact"

    @property
    def unit_block_size(self) -> bool:
        return False

    def _default_fieldsplit_options(self):
        opts = super()._default_fieldsplit_options()
        opts.update(
            {
                f"fieldsplit_{self.complement_tag}_mat_schur_complement_ainv_type": "blockdiag"
            }
        )
        return opts

    def _default_options(self) -> dict:
        local_opts = {
            "pc_type": "pbjacobi",
            # TODO: Not sure this will come through in the right way
            # "mat_schur_complement_ainv_type": "blockdiag",
        }
        return local_opts


class MultiPhysicsPreconditioner:
    """Translate a general scheme to a specific PETSc preconditioner, specified as a
    dictionary (really a fully specified petsc options).
    """

    def __init__(
        self,
        components: list[SinglePhysicsPreconditioner],
        dof_manager: DofManager,
        model: pp.PorePyModel,
        options: dict | None = None,
    ):
        """
        Args:
            groups: List of groups of equations and variables.
            schemes: List of schemes for each group.
        """
        self._single_physics_precond = components
        self._dof_manager = dof_manager
        self._nd = model.nd
        self._model = model
        self._options = options if options is not None else {}

    def configure(
        self,
        bmat: BlockMatrixStorage,
        pc,  # PC comes from ksp or similar
        user_options: dict | None = None,
    ) -> dict:
        """
        Populate the PETSc preconditioner based on the groups and schemes. This entails
        making a bridge from the general settings defined in a scheme to the PETSc
        options needed to apply the scheme to a contrete linear system, while also accounting for

        Args:
            model: The model instance specifying the problem to be solved.
        """
        user_options = user_options if user_options is not None else {}
        options = {}

        prefix = ""

        for counter, single_physics_precond in enumerate(self._single_physics_precond):
            # Define a scheme for the group
            has_complement = counter < len(self._single_physics_precond) - 1

            # Generate the actual petsc proconditioner.
            loc_options = single_physics_precond.configure(
                has_complement=has_complement, opts=user_options
            )

            # Get the tag for this group, and prepend it to the options.
            tagged_options = {f"{prefix}{k}": v for k, v in loc_options.items()}

            if not has_complement:
                # If there is no complement, we can use the options directly.
                options |= tagged_options
                insert_petsc_options(options)
                pc.setFromOptions()
                pc.setUp()

                return options

            block_size = 1 if single_physics_precond.unit_block_size else self._nd

            # Get the IS for the group, but only if complement is not None.
            is_this, is_complement = self._dof_manager.petsc_is(
                single_physics_precond,
                self._single_physics_precond[counter + 1 :],
                bmat,
            )
            is_complement.setBlockSize(block_size)
            tag = single_physics_precond.tag
            complement_tag = tag + "_complement"
            insert_petsc_options(tagged_options)
            pc.setFromOptions()
            pc.setFieldSplitIS((tag, is_this), (complement_tag, is_complement))

            pc.setUp()

            ksp_elim = pc.getFieldSplitSubKSP()[0]
            pc_group = ksp_elim.getPC()

            ksp_complement = pc.getFieldSplitSubKSP()[1]
            pc_complement = ksp_complement.getPC()

            if single_physics_precond.ksp_keep_use_pmat:
                _, pmat = ksp_complement.getOperators()
                # TODO: Is it correct to use the same matrix for both arguments?
                ksp_complement.setOperators(pmat, pmat)

            if single_physics_precond.near_null_space(self) is not None:
                null_space_vectors = []
                for b in single_physics_precond.near_null_space:
                    null_space_vec_petsc = PETSc.Vec().create()  # possibly mem leak
                    null_space_vec_petsc.setSizes(b.shape[0], block_size)
                    null_space_vec_petsc.setUp()
                    null_space_vec_petsc.setArray(b)
                    null_space_vectors.append(null_space_vec_petsc)
                # possibly mem leak
                null_space_petsc = PETSc.NullSpace().create(True, null_space_vectors)
                pc_group.getOperators()[1].setNearNullSpace(null_space_petsc)

            # Call on self.complement to configure the PETSc PC object for the complement,
            # and update (override) the options with the options returned by the complement.
            # Note that, due to the tagging system, this may override some options that were
            # set above.
            pc = pc_complement
            prefix = f"{prefix}fieldsplit_{complement_tag}_"

        raise ValueError("Should have reached an empty complement")


def mass_balance_factory():
    return [InterfaceDarcyFluxPreconditioner(), MassBalancePreconditioner()]


def momentum_balance_factory():
    return [ContactPreconditioner(), MechanicsPreconditioner()]


# def hm_factory():


def contact_transform(J, row_group: int, col_group: int, nd: int):
    """Assemble the right linear transformation."""
    # Sorted according to groups. If not done, the matrix can be in porepy order,
    # which does not guarantee that diagonal groups are truly on diagonals.
    Qright = J.empty_container()[:]

    if row_group not in J.active_groups[0]:
        Qright.mat = csr_ones(Qright.shape[0])
        return Qright

    J55 = J[col_group, col_group].mat

    J55_inv = inv_block_diag(J55, nd=nd, lump=False)

    Qright.mat = csr_ones(Qright.shape[0])

    J54 = J[col_group, row_group].mat

    tmp = -J55_inv @ J54
    Qright[col_group, row_group] = tmp
    return Qright


@dataclass
class LinearSolverComponents:
    dof_manager: DofManager
    preconditioner: MultiPhysicsPreconditioner
    ksp_factory: PetscKSPScheme


class IterativeSolverMixin:
    def solve_linear_system(self) -> None:
        # Check for NaN or Inf in the RHS.
        mat, rhs = self.linear_system
        if np.any(np.isnan(rhs) | np.isinf(rhs)):
            raise ValueError("RHS contains NaN or Inf values")

        # By default, print the residual information to screen (ksp_monitor=None).
        solver_options = self.params["linear_solver"].get(
            "options", {"ksp_monitor": None}
        )
        ksp_factory = self._solver_components.ksp_factory
        solver = ksp_factory.make_solver(self.bmat, solver_options)
        try:
            solver = ksp_factory.make_solver(self.bmat, solver_options)
        except Exception as e:
            raise RuntimeError(
                "Failed to create solver with the provided preconditioner."
            ) from e

        rhs_loc = self.bmat.project_rhs_to_local(rhs)
        x_loc = solver.solve(rhs_loc)

        info = solver.ksp.getConvergedReason()
        if info <= 0:
            raise RuntimeError(
                f"Solver did not converge. Reason: {info}. "
                "Check the solver options and the problem setup."
            )

        x = self.bmat.project_solution_to_global(x_loc)

        return np.atleast_1d(x)

    def assemble_linear_system(self):
        super().assemble_linear_system()  # type: ignore[misc]

        dof_manager = self._solver_components.dof_manager
        # Get the linear system from the equation system.

        # TODO: Replace this with a different type of plugin
        mat, rhs = self.linear_system

        # Apply the `contact_permutation`.
        mat = mat[dof_manager.eq_rows_permutation(self)]
        rhs = rhs[dof_manager.eq_rows_permutation(self)]

        bmat = BlockMatrixStorage(
            mat=self.linear_system[0],
            global_dofs_row=dof_manager.eq_dofs_by_blocks(self),
            global_dofs_col=dof_manager.var_dofs_by_blocks(self),
            groups_to_blocks_row=dof_manager.equation_groups(self),
            groups_to_blocks_col=dof_manager.variable_groups(self),
            # group_names_row=self.group_row_names(),
            # group_names_col=self.group_col_names(),
        )

        self.bmat = bmat

    def _initialize_linear_solver(self):
        # Set up preconditioner.
        precond_factory: Callable[[], MultiPhysicsPreconditioner] = self.params[
            "linear_solver"
        ]["preconditioner_factory"]
        if precond_factory is None:
            raise ValueError("Preconditioner factory is not set")
        precond_list: list[SinglePhysicsPreconditioner] = precond_factory()

        ordering_list = [precond.group() for precond in precond_list]

        dof_manager = DofManager(self.equation_system, ordering_list)
        precond = MultiPhysicsPreconditioner(precond_list, dof_manager, self)

        contact_ind = dof_manager.identify_contact_group(self)

        ksp_factory = PetscKSPScheme(preconditioner=precond)
        if contact_ind > -1:
            # If there is a contact group, we need to use a linear solver that takes
            # care of potential singularities in the contact block.
            u_intf_ind = dof_manager.identify_u_intf_group(self)

            block_transform = [
                lambda bmat: contact_transform(bmat, contact_ind, u_intf_ind, self.nd)
            ]

            solver_factory = LinearTransformedScheme(
                nd=self.nd,
                contact_group=contact_ind,
                u_intf_group=u_intf_ind,
                # preconditioner=precond,
                inner=ksp_factory,
                right_transformations=block_transform,
            )

        else:
            # A standard KSP solver will do.
            solver_factory = ksp_factory

        solver_components = LinearSolverComponents(
            dof_manager=dof_manager,
            preconditioner=precond,
            ksp_factory=solver_factory,
        )
        self._solver_components = solver_components
