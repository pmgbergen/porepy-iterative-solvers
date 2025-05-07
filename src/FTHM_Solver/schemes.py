"""This module contains schemes, e.g., recepies for constructing a PETSc solver."""

from __future__ import annotations

import numpy as np
from typing import Type, Callable
from dataclasses import dataclass
import porepy as pp
from abc import ABC, abstractmethod, abstractproperty
import FTHM_Solver.hm_solver
from petsc4py import PETSc
import FTHM_Solver


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
        return [
            [("mass_balance_equation", subdomains)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [
            [model.pressure(subdomains)],
        ]


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

        return [
            [("momentum_balance_equation", subdomains)],
            [("interface_force_balance_equation", interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)
        return [
            [model.displacement(subdomains)],
            [model.interface_force_balance(interfaces)],
        ]


class ContactGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        return [
            [("normal_fracture_deformation_equation", subdomains)],
            [("tangential_fracture_deformation_equation", subdomains)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
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
        return self._equation_groups.index(group)

    def petsc_is(
        self,
        current_group: AbstractGroup,
        other_groups: list[AbstractGroup],
        bmat: FTHM_Solver.BlockMatrixStorage,
    ):
        # Not sure if this belongs here, but it is tempting to put it here and not in
        # the composer.

        # Indices of the block ids
        current_id = self._group_id(current_group)
        other_id = [self._group_id(group) for group in other_groups]

        current_is = FTHM_Solver.construct_is(bmat, current_id)
        other_is = FTHM_Solver.construct_is(bmat, other_id)
        return current_is, other_is

    def variable_groups(self, model):
        groups = [group.variable_groups(model) for group in self._orderings]
        return FTHM_Solver.get_variables_group_ids(groups)

    def _identify_contact_group(self, model):
        # Identify the contact group in the equation groups
        for i, group in enumerate(self._orderings):
            if len(group.equation_groups(model)) == 0:
                continue
            for block in group.equation_groups(model):
                if block[0][0] == "normal_fracture_deformation_equation":
                    return i
        return -1

    def equation_groups(self, model):
        """Get the equation groups for the model, in the form of a list of
        a list of numbers. If the contact group is present, it will be
        reordered so that the normal and tangential equations are together.
        """
        # Get the equation groups for the model (in name-domain format)
        equation_groups_by_name = [
            group.equation_groups(model) for group in self._orderings
        ]
        self._equation_groups = equation_groups_by_name

        # Convert to numbers (i.e., block ids).
        equation_groups_by_number = FTHM_Solver.get_equations_group_ids(
            equation_groups_by_name
        )

        contact_group = self._identify_contact_group(model)
        # If there is no contact group, return the original equation groups.
        if contact_group == -1:
            return equation_groups_by_number

        # Temporary construct to get the correct contact equations groups. To be
        # refactored.
        tmp_solver = FTHM_Solver.hm_solver.IterativeHMSolver()
        reordered_groups = tmp_solver._correct_contact_equations_groups(
            equation_groups_by_number, contact_group
        )
        return

    def eq_dofs_by_blocks(self, model):
        """Get the equation dofs for the model, in the form of a list of numbers,
        one per equation-domain pair. If the contact group is present, it will be
        reordered so that the normal and tangential equations for each fracture cell
        form a diganol block.
        """

        # Temporary construct to get the correct contact equations groups. To be
        # refactored.
        tmp_solver = FTHM_Solver.hm_solver.IterativeHMSolver()
        dofs = tmp_solver.eq_dofs
        return dofs

    def eq_rows_permutation(self, model):
        """Get a permutation vector for the full linear system of equations. This is
        used to reorder the equations so that the contact equations for single fracture
        cells form a diagonal block.

        If no contact group is present, the permutation vector is linear.

        See also eq_dofs_by_blocks, which is used to reorder contact equations within
        the equation block format.
        """
        return FTHM_Solver.hm_solver.make_reorder_contact(
            model, self._identify_contact_group(model)
        )


class SinglePhysicsPreconditioner(ABC):
    """
    Abstract class for defining a preconditioner.
    """

    @abstractmethod
    def __init__(self):
        """
        Args:
            opts: Dictionary of options for the preconditioner.
            complement: Complementary scheme.
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

    def _default_options(self, has_complement: bool) -> dict:
        local_opts = {
            "pc_type": "hypre",
            "ksp_type": "preonly",
        }
        if has_complement:
            local_opts["pc_fieldsplit_schur_precondition"] = "selfp"
            local_opts["pc_fieldsplit_schur_fact"] = "lower"
        return local_opts


class ContactPreconditioner(SinglePhysicsPreconditioner):
    @property
    def key(self) -> str:
        return "contact"

    @property
    def tag(self) -> str:
        return "contact"

    def group(self):
        return ContactGroup()

    def _default_options(self, has_complement: bool) -> dict:
        if not has_complement:
            raise ValueError("The contact preconditioner requires a complement.")
        local_opts = {
            "pc_type": "hypre",
            "ksp_type": "preonly",
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
        options: dict | None = None,
    ):
        """
        Args:
            groups: List of groups of equations and variables.
            schemes: List of schemes for each group.
        """
        self._single_physics_precond = components
        self._dof_manager = dof_manager
        self._options = options if options is not None else {}

    def configure(
        self,
        pc,  # PC comes from ksp or similar
    ) -> dict:
        """
        Populate the PETSc preconditioner based on the groups and schemes. This entails
        making a bridge from the general settings defined in a scheme to the PETSc
        options needed to apply the scheme to a contrete linear system, while also accounting for

        Args:
            model: The model instance specifying the problem to be solved.
        """

        options = {}

        for counter, single_physics_precond in enumerate(self._single_physics_precond):
            # Define a scheme for the group
            has_complement = counter < len(self._single_physics_precond) - 1

            # Generate the actual petsc proconditioner.
            loc_options = single_physics_precond.configure(
                has_complement=has_complement, opts=self._options
            )

            # Get the tag for this group, and prepend it to the options.
            tag = single_physics_precond.tag
            tagged_options = {f"{tag}_{k}": v for k, v in loc_options.items()}

            if not has_complement:
                # If there is no complement, we can use the options directly.
                options |= tagged_options
                return options

            # Get the IS for the group, but only if complement is not None.
            is_this, is_complement = self._dof_manager.petsc_is(
                single_physics_precond.group(),
                self._single_physics_precond[counter + 1 :],
                self.bmat,
            )
            complement_tag = tag + "_complement"

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

            if single_physics_precond.near_null_space is not None:
                null_space_vectors = []
                for b in self.near_null_space:
                    null_space_vec_petsc = PETSc.Vec().create()  # possibly mem leak
                    null_space_vec_petsc.setSizes(b.shape[0], self.block_size)
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

        raise ValueError("Should have reached an empty complement")


def mass_balance_factory():
    return [InterfaceDarcyFluxPreconditioner(), MassBalancePreconditioner()]


# def hm_factory():


class IterativeSolverMixin:
    def solve_linear_system(self) -> None:
        # Check for NaN or Inf in the RHS.
        mat, rhs = self.linear_system
        if np.any(np.isnan(rhs) | np.isinf(rhs)):
            raise ValueError("RHS contains NaN or Inf values")

        precond_factory: Callable[[], MultiPhysicsPreconditioner] = self.params[
            "linear_solver"
        ]["preconditioner_factory"]
        solver_options = self.params["linear_solver"].get("options", {})
        if precond_factory is None:
            raise ValueError("Preconditioner factory is not set")
        precond_list: list[SinglePhysicsPreconditioner] = precond_factory()

        dof_manager = DofManager(self.equation_system, precond_list)
        precond = MultiPhysicsPreconditioner(precond_list, dof_manager)

        ksp_factory = FTHM_Solver.PetscKSPScheme(
            preconditioner=precond, options=solver_options
        )
        try:
            solver = ksp_factory.make_solver(mat)
        except Exception as e:
            raise RuntimeError(
                "Failed to create solver with the provided preconditioner."
            ) from e

        rhs_loc = mat.project_rhs_to_local(rhs)
        x_loc = solver.solve(rhs_loc)

        info = solver.ksp.getConvergedReason()
        if info != 0:
            raise RuntimeError(
                f"Solver did not converge. Reason: {info}. "
                "Check the solver options and the problem setup."
            )

        x = mat.project_solution_to_global(x_loc)

        return np.atleast_1d(x)

    def assemble_linear_system(self):
        super().assemble_linear_system()  # type: ignore[misc]

        # TODO: Replace this with a different type of plugin
        row_permutation = self._linear_solver_scheme_maker.row_indices()
        mat, rhs = self.linear_system

        # Apply the `contact_permutation`.
        mat = mat[row_permutation]
        rhs = rhs[row_permutation]

        scheme_maker = self._linear_solver_scheme_maker

        bmat = FTHM_Solver.BlockMatrixStorage(
            mat=self.linear_system[0],
            global_dofs_row=scheme_maker.eq_dofs,
            global_dofs_col=scheme_maker.var_dofs,
            groups_to_blocks_row=scheme_maker.equation_groups,
            groups_to_blocks_col=scheme_maker.variable_groups,
            group_names_row=self.group_row_names(),  # TODO: Move to the scheme
            group_names_col=self.group_col_names(),
        )

        self.bmat = bmat
