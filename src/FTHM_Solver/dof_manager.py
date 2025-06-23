from __future__ import annotations
import numpy as np
import porepy as pp
from .block_matrix import BlockMatrixStorage
from .full_petsc_solver import construct_is
from .iterative_solver import get_equations_group_ids, get_variables_group_ids
from .preconditioners import SinglePhysicsPreconditioner, CompositePreconditioner

from . import equation_variable_groups as groups
from .equation_variable_groups import EquationNames


class DofManager:
    """Takes care of translation of blocks and groups (from EquationSystem format) to
    block indices, as well as grouping the fine-scale dofs. Also reordering related to
    the contact problem.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so.
    """

    def __init__(
        self,
        equation_system: pp.EquationSystem,
        model: pp.PorePyModel,
        orderings: list[groups.AbstractGroup],
        solvers: list[SinglePhysicsPreconditioner],
    ):
        self._equation_system = equation_system
        self._orderings = orderings
        eq_groups, var_goups, slv_groups, name_to_group_ind = (
            self._process_block_information(model, solvers)
        )
        self._equation_groups = eq_groups
        self._variable_groups = var_goups
        self._solver_groups = slv_groups
        self._name_to_group_indices = name_to_group_ind

    @property
    def groups(self) -> list[groups.AbstractGroup]:
        """Return the groups of equations and variables."""
        return self._orderings

    def petsc_is(
        self,
        current_solver: groups.AbstractGroup,
        other_solver: list[groups.AbstractGroup],
        bmat: BlockMatrixStorage,
    ):
        # Not sure if this belongs here, but it is tempting to put it here and not in
        # the composer.

        # Indices of the block ids
        current_id = self.blocks_of_solver(current_solver)

        other_id = []
        for group in other_solver:
            # Get the block id for the group.
            other_id += self.blocks_of_solver(group)

        current_is = construct_is(bmat, current_id)
        other_is = construct_is(bmat, other_id)
        return current_is, other_is

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        return self._variable_groups

    def equation_groups(self, model: pp.PorePyModel) -> list[list[int]]:
        return self._equation_groups

    def blocks_of_solver(self, solver: SinglePhysicsPreconditioner) -> int:
        return self._solver_groups[solver]

    def _process_block_information(self, model: pp.PorePyModel, solvers):
        """Construct groups of equations, variables and solvers from the orderings and
        solvers.

        This method should be called as part of the DofManager initialization. It
        process information on the orderings and solvers to arrive at the following:

        1. A list of equation groups, where each group defines a set of equations (using
           the block indices of the BlockMatrixStorage) that will be preconditioned
           together. Together, the list spans the full set of equations in the model at
           hand.
        2. A list of variable groups, where each group defines a set of variables that
           will be preconditioned together. Together, the list spans the full set of
           variables in the model at hand.
        3. A dictionary that maps each solver to the indices of the equation groups it
           will solve. This is used to identify which equations are solved by which
           solver, and to construct the appropriate preconditioner for each solver.

        """
        # The construction consists of two main steps: First, iterate over the orderings
        # and gather the equation and variable groups defined by them. This may involve
        # uniquifying the groups (relevant if the preconditioner is a PETSc Composite,
        # which may contain several overlapping groups). In this process, we also
        # construct the map from solvers to block indices. Second, the groups of
        # variables and equations are expanded from a lists of domains (PorePy
        # subdomains and interfaces) into individual variables and equations. For the
        # equations, we also do some reordering needed to merge the contact equations in
        # the normal and tangential directions into a single block, and thereby reveal
        # the underlying block diagonal structure of this equation.

        # Data structures for variables, equations and the solver map.
        var_groups = []
        equations_by_name = []
        solver_indices = {}
        # Counter of block indices, used to assign block indices to the solver map.
        counter = 0

        # Iterate over the orderings, gather the equation an variable groups define
        # there. It is assumed that the orderings define non-intersecting sets of
        # equations and variables that together span the system of equations to be
        # solved. However, an ordering can return a list, corresponding to a multistage
        # preconditioner (a Composite preconditioner in PETSc terminology). This list
        # may contain multiple intersecting groups, that needs to be parsed into a
        # single, non-intersecting set of equations and variables.
        for group, slv in zip(self._orderings, solvers):
            if isinstance(group, list):
                # This is a list of groups, which may contain identical items. First
                # gather them all.
                #
                # First, make sure this is a composite preconditioner; this is a tacit
                # assumption of the below parsing. Dealing with anything else would
                # require a more structured approach to the parsing, but EK has neither
                # the imagination nor the test cases needed to do so now.
                assert isinstance(slv, CompositePreconditioner)

                # Tacitly assump that the group is a list of lists of AbstractGroup. If
                # we hit a third nested level, we will need to do recursion of sorts.
                tmp_var_groups = []
                tmp_equations_by_name = []
                for g in group:
                    tmp_var_groups += g.variable_groups(model)
                    tmp_equations_by_name += g.equation_groups(model)

                # Find the indices of the subsets that will define a unique set of
                # variables and equations. By assumption, the variables and equations in
                # the tmp_x lists match, so that we can use any of them to find indices
                # that define a unique sublist. It is by far simplest to use the
                # variables, since we can rely on their hash values to find the sublist
                # (while the equation list is a confused mess of lists and tuples, which
                # fortunately works).

                # The implementation below assumes that each variable group contains a
                # single variable. Expanding this should be doable, but it has not yet
                # been necessary.
                assert all(len(x) == 1 for x in tmp_var_groups)

                # Find the sorting indices of the unique variable groups.
                hash_values = [hash(item[0]) for item in tmp_var_groups]
                _, sorting_indices = np.unique(hash_values, return_index=True)
                # Sort the indices to ensure a consistent order.
                sorting_indices.sort()

                # Extract unique sublists.
                groups_loc = []
                vars_loc = []
                for ind in sorting_indices:
                    vars_loc.append(tmp_var_groups[ind])
                    groups_loc.append(tmp_equations_by_name[ind])

                for sub_solver in slv.solvers:
                    # If the sub-solver is a list, it is by itself a filedsplit
                    # preconditioner. Treat every sub-solver within the fieldsplit as a
                    # separate solver, and add it to the solver indices.
                    if isinstance(sub_solver, list):
                        for ss in sub_solver:
                            solver_indices[ss] = []
                            # There could be deeper recursion levels here, which we may
                            # need to deal with by some recursive approach, but we
                            # ignore that possibility for now.
                            assert isinstance(ss, SinglePhysicsPreconditioner)
                            ss_vars = ss.group().variable_groups(model)
                            assert isinstance(ss_vars, list)
                            sub_vars = [ss_vars[i][0] for i in range(len(ss_vars))]

                            # Loop over all variables associated with this subsolver.
                            # Find it among the unique variable set (if it is not found,
                            # something is seriously worng), and add the block index to
                            # the list associated with the subsolver.
                            for var in sub_vars:
                                loc_id = int(
                                    np.where(
                                        np.array(hash_values)[sorting_indices]
                                        == hash(var)
                                    )[0][0]
                                )
                                solver_indices[ss].append(loc_id + counter)
                    else:
                        # Assume here that the sub-solver is a
                        # SinglePhysicsPreconditioner. If we at some point need nested
                        # composite preconditioners, something will go wrong here. There
                        # are surely other cases that can break this as well.
                        solver_indices[sub_solver] = [
                            counter + int(i) for i in sorting_indices
                        ]

            else:
                # This is a single group, we can add its variables and equations.
                groups_loc = group.equation_groups(model)
                vars_loc = group.variable_groups(model)

            # Append the groups to the main lists, and update the solver indices.
            equations_by_name += groups_loc
            var_groups += vars_loc
            # Also take note that the solver slv is associated with all indices in the
            # local groups. A composite preconditioner will by this be associated with
            # the entire set of indices *in addition to* the mapping of individual
            # solvers (see the convoluted for-loop above). This double registration is
            # needed for the selection of blocks to work as intended. For pure
            # (non-composite) block preconditioners, we simply map the preconditioner to
            # the registred indices.
            solver_indices[slv] = list(range(counter, counter + len(groups_loc)))
            counter += len(groups_loc)

        # Done with the first step. Next, expand the groups by calling on relevant
        # helper methods.
        var_groups_by_number = get_variables_group_ids(model, var_groups)
        equation_groups_by_number = get_equations_group_ids(model, equations_by_name)

        # Permute the contact equations if present. NOTE: It would have been preferrable
        # to use the name_to_group_indices map, constructed just below, to identify the
        # contact group, but this is not yet available at this point. Refactoring may be
        # a good idea.
        contact_group = self.identify_contact_group(model)
        if contact_group == -1:
            reordered_groups = equation_groups_by_number
        else:
            reordered_groups = self._correct_contact_equations_groups(
                model, equation_groups_by_number, contact_group
            )

        name_to_group_index_map = {}
        for i, item in enumerate(equations_by_name):
            name = item[0][0]
            # Add the equation name to the group.
            if name not in name_to_group_index_map:
                name_to_group_index_map[name] = []
            name_to_group_index_map[name].append(i)

        return (
            reordered_groups,
            var_groups_by_number,
            solver_indices,
            name_to_group_index_map,
        )

    def equation_names(self, model):
        names = []
        for group in self._orderings:
            if isinstance(group, list):
                # If the group is a list, we assume it contains multiple groups.
                # TODO: Unification needed here.
                for g in group:
                    names += g.equation_names(model)
            else:
                names += group.equation_names(model)
        return names

    def variable_names(self, model):
        names = []
        for group in self._orderings:
            if isinstance(group, list):
                # If the group is a list, we assume it contains multiple groups.
                # TODO: Unification needed here.
                for g in group:
                    names += g.variable_names(model)
            else:
                names += group.variable_names(model)
        return names

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
            if isinstance(group, list):
                for sub_group in group:
                    if len(sub_group.variable_groups(model)) == 0:
                        continue
                    for var in sub_group.variable_groups(model):
                        if var[0].name == model.interface_displacement_variable:
                            return i
            else:
                if len(group.variable_groups(model)) == 0:
                    continue
                for var in group.variable_groups(model):
                    if var[0].name == model.interface_displacement_variable:
                        return i
                    else:
                        i += 1
        return -1

    def identify_energy_balance_group(self, model):
        return self._name_to_group_indices[EquationNames.ENERGY_BALANCE.value]

    def eq_dofs_by_blocks(self, model):
        """Get the equation dofs for the model, in the form of a list of numbers,
        one per equation-domain pair. If the contact group is present, it will be
        reordered so that the normal and tangential equations for each fracture cell
        form a digonal block.
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
