from __future__ import annotations

from typing import Sequence

import numpy as np
import porepy as pp

from pp_solvers.equation_variable_groups import (
    AbstractGroup,
    EquationGroup,
    EquationNames,
)
from pp_solvers.preconditioners import (
    CompositePreconditioner,
    SinglePhysicsPreconditioner,
)


class DofManager:
    """Takes care of translation of blocks and groups (from EquationSystem format) to
    block indices, as well as grouping the fine-scale dofs. Also reordering related to
    the contact problem.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so.
    """

    def __init__(
        self,
        model: pp.PorePyModel,
        solvers: list[SinglePhysicsPreconditioner],
    ):
        """Construct groups of equations, variables and solvers from the orderings and
        solvers.

        Information on the equation blocks and solvers to arrive at the following:

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
        self._orderings = [precond.group() for precond in solvers]

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
                # First, make sure this is a composite preconditioner; this is a tacit
                # assumption of the below parsing. Dealing with anything else would
                # require a more structured approach to the parsing, but EK has neither
                # the imagination nor the test cases needed to do so now.
                assert isinstance(slv, CompositePreconditioner)

                # These are the "results" of this conditional branch, used below.
                groups_loc = []
                vars_loc = []

                for sub_solver in slv.solvers:
                    # If the sub-solver is a list, it is by itself a fieldsplit
                    # preconditioner. Treat every sub-solver within the fieldsplit as a
                    # separate solver, and add it to the solver indices.

                    # If it's not a list, making it a list to generalize the code below.
                    if isinstance(sub_solver, SinglePhysicsPreconditioner):
                        sub_solver = [sub_solver]
                    assert isinstance(sub_solver, list)

                    # This is the inner counter to distinguish subsolver indices within
                    # a compositional solver.
                    composite_counter = 0

                    # These lists must be identical for each subsolver of a composite
                    # preconditioner. We check it and then store the results in the
                    # global lists of results "groups_loc" and "vars_loc".
                    groups_loc_composite = []
                    vars_loc_composite = []

                    for ss in sub_solver:
                        # There could be deeper recursion levels here, which we may
                        # need to deal with by some recursive approach, but we
                        # ignore that possibility for now.
                        assert isinstance(ss, SinglePhysicsPreconditioner)

                        ss_groups = ss.group()
                        # It can be either a group or a list of groups.
                        if isinstance(ss_groups, AbstractGroup):
                            ss_groups = [ss_groups]
                        solver_indices[ss] = []

                        for ss_group in ss_groups:
                            ss_vars = ss_group.variable_groups(model)
                            groups_loc_composite.extend(ss_group.equation_groups(model))
                            vars_loc_composite.extend(ss_vars)

                            solver_indices[ss].extend(
                                list(
                                    range(
                                        counter + composite_counter,
                                        len(ss_vars) + counter + composite_counter,
                                    )
                                )
                            )
                            composite_counter += len(ss_vars)

                    # Make sure they are the same for each composite subsolver. This
                    # check is performed starting from the second subsolver.
                    if len(groups_loc) != 0:
                        # It is easy to compare equation groups.
                        assert groups_loc == groups_loc_composite

                        # And more involved to compare variable groups. MDVariables have
                        # different ids, so are technically different objects. We ensure
                        # that the names and domains are the same.
                        for var_loc, var_loc_composite in zip(
                            vars_loc, vars_loc_composite
                        ):
                            for md_var_expected, md_var in zip(
                                var_loc, var_loc_composite
                            ):
                                assert md_var.domains == md_var_expected.domains
                                assert md_var.name == md_var_expected.name
                    else:
                        # This is the first subsolver, just assign them.
                        groups_loc = groups_loc_composite
                        vars_loc = vars_loc_composite

            else:
                # This is not a compositional preconditioner group.
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

        # Done with the first step. The mapping from solvers to block indices can now be
        # stored.
        self._solver_groups = solver_indices

        # Construct the mapping from equation names to the group indices. This must be
        # done before identifying the contact group.
        name_to_group_index_map = {}
        item: EquationGroup
        for i, item in enumerate(equations_by_name):
            for eq_item in item.items:
                name = eq_item.name
                if name not in name_to_group_index_map:
                    name_to_group_index_map[name] = []
                name_to_group_index_map[name].append(i)
        self._name_to_group_indices = name_to_group_index_map

        # Next, expand the groups by calling on relevant helper methods.
        var_groups_by_number = self._variable_block_indices(model, var_groups)
        self._variable_groups = var_groups_by_number

        equation_groups_by_number = self._equation_block_indices(
            model, equations_by_name
        )

        # Permute the contact equations if present. NOTE: It would have been preferrable
        # to use the name_to_group_indices map, constructed just below, to identify the
        # contact group, but this is not yet available at this point. Refactoring may be
        # a good idea.
        contact_group = self.identify_contact_group()
        if contact_group is None:
            self._equation_groups = equation_groups_by_number
        else:
            self._equation_groups = self._correct_contact_equations_groups(
                model, equation_groups_by_number, contact_group
            )

    @property
    def variable_groups(self) -> list[list[int]]:
        """Get the variable groups.

        Returns:
            A list of lists, where each inner list contains MixedDimensionalVariable
            objects representing the variable groups. TODO: Fix return type.

        """
        return self._variable_groups

    @property
    def equation_groups(self) -> list[list[int]]:
        """Get the equation groups.

        Returns:
            A list of lists, where each inner list contains integers representing
            the equation groups.

        """
        return self._equation_groups

    def equation_names(self, model: pp.PorePyModel) -> list[str]:
        """Get the names of equations in the model.

        Parameters:
            model: The PorePy model.

        Returns:
            A list of strings containing the names of equations in the model.

        """
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

    def variable_names(self, model: pp.PorePyModel) -> list[str]:
        """Get the names of variables in the model.

        Parameters:
            model: The PorePy model.

        Returns:
            A list of strings containing the names of variables in the model.

        """

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

    def eq_dofs_by_blocks(self, model) -> list[np.ndarray]:
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

        contact_group = self.identify_contact_group()
        if contact_group == None:
            # If there is no contact group, return the original equation dofs.
            return eq_dofs
        # Short cut if no contact mechanics, hence no reordering.
        if len(self.equation_groups[contact_group]) == 0:
            # Ignore mypy error, list[np.ndarray] is a subset of list[np.ndarray |
            # None].
            return eq_dofs  # type: ignore[return-value]

        # We assume that normal equations go first. TODO: Can we make this more robust,
        # or else put an assert here.
        normal_blocks = self.equation_groups[contact_group]
        num_fracs = len(model.mdg.subdomains(dim=model.nd - 1))

        # EK: I believe this is an assumption that the tangential equations are right
        # after the normal equations.
        all_contact_blocks = [
            nb + i * num_fracs for i in range(2) for nb in normal_blocks
        ]

        eq_dofs_corrected: list[np.ndarray | None] = []
        # Add all equations that are not contact equations without any changes.
        for i, x in enumerate(eq_dofs):
            if i not in all_contact_blocks:
                eq_dofs_corrected.append(x)
            elif i in normal_blocks:
                eq_dofs_corrected.append(None)

        offset = eq_dofs[normal_blocks[0]][0]
        for nb in normal_blocks:
            # Create indices for the normal and tangential components of the contact.
            # There will be model.nd equations for each block.
            inds = offset + np.arange(eq_dofs[nb].size * model.nd)
            offset = inds[-1] + 1
            eq_dofs_corrected[nb] = np.array(inds)

        return eq_dofs_corrected

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

    def blocks_of_solver(self, solver: SinglePhysicsPreconditioner) -> list[int]:
        """Get the block indices associated with a solver.

        # YZ: the consistent name would be groups_of_solvers

        Parameters:
            solver: A SinglePhysicsPreconditioner object.

        Returns:
            The block indices associated with the solver.

        """
        return self._solver_groups[solver]

    def identify_contact_group(self) -> int | None:
        """Identify the contact group in the equation groups.

        It is assumed that there is a single contact group, that is, that the contact
        blocks are not split into several groups.

        Returns:
            The index of the contact group in the equation groups. If no contact group
            is found, returns None.

        """
        # Identify the contact group in the equation groups
        ind = self._name_to_group_indices.get(EquationNames.CONTACT_NORMAL.value, [None])
        return ind[0]

    def identify_energy_balance_groups(self) -> list[int]:
        """Identify the energy balance groups in the equation groups.

        Returns:
            The indices of the energy balance group in the equation groups.

        """
        return self._name_to_group_indices.get(EquationNames.ENERGY_BALANCE.value, [])

    def identify_u_intf_group(self, model) -> int | None:
        """Identify the interface displacement group in the equation groups. It is
        assumed that there is a single group of interface displacements.

        Parameters:
            model: The PorePy model.

        Returns:
            The index of the interface displacement group in the equation groups. If no
            interface displacement group is found, or the group does not have non-empty
            variable, None is returned.

            Note the inconsistency with `identify_contact_group`, which returns a value
            if the model has a contact group, but no equations are defined for it. This
            reflects an asymmetry in PorePy's treatment of equations and variables:
            Equations defined on empty domains are still equations, while variables on
            empty domains have the name `empty_md_variable` and is thereby not
            identifiable as a specific variable.

        """
        if not hasattr(model, "interface_displacement_variable"):
            # The model does not have an interface displacement variable.
            return None

        # Identify the interface group in the equation groups.
        i = 0

        # Note to self: Here we need to loop over the _orderings, since we need to match
        # the ordering of the preconditioner to porepy information. See also the other
        # identify methods.
        for group in self._orderings:
            if isinstance(group, list):
                for sub_group in group:
                    if len(sub_group.variable_groups(model)) == 0:
                        continue
                    for var in sub_group.variable_groups(model):
                        if var[0].name == model.interface_displacement_variable:
                            return i
            else:
                for var in group.variable_groups(model):
                    # Loop over all the variables of the group (variables treated with
                    # this block solver). See if this is the one.
                    if var[0].name == model.interface_displacement_variable:
                        return i
                    else:
                        i += 1
        return None

    def eq_rows_permutation(self, model: pp.PorePyModel):
        """Get a permutation vector for the full linear system of equations.

        This is used to reorder the equations so that the contact equations for single
        fracture cells form a diagonal block.

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
            A numpy array with the permutation indices for the equations. If no contact
            group is present, the permutation is the identity, i.e., no reordering is
            performed.

        """
        permutation = np.arange(model.equation_system.num_dofs())
        contact_group = self.identify_contact_group()

        # If there is no contact group, return the original equation groups.
        if contact_group is None:
            return np.arange(model.equation_system.num_dofs())
        # If contact is formally present, but no equations are defined for it,
        # return the original permutation.
        if len(self.equation_groups[contact_group]) == 0:
            return permutation

        # Get the (fine-scale, not block(!)) dofs of the contact mechanics equations.
        dofs_contact = np.concatenate(
            [
                self.eq_dofs_by_blocks(model)[i]
                for i in self.equation_groups[contact_group]
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
            permutation[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1]
            ).ravel("F")
        elif model.nd == 3:
            # Do the same as in 2d, also for the second tangential component.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells::2]
            dofs_contact_2 = dofs_contact[num_contact_cells + 1 :: 2]
            permutation[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1, dofs_contact_2]
            ).ravel("F")
        else:
            raise ValueError("Model dimension must be 2 or 3.")
        return permutation

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

    def _variable_block_indices(
        self,
        model: pp.PorePyModel,
        md_variables_groups: Sequence[
            Sequence[pp.ad.MixedDimensionalVariable | pp.ad.Variable]
        ],
    ) -> list[list[int]]:
        """Used to assemble the index that will later help accessing the submatrix
        corresponding to a group of variables, which may include one or more variable.

        Example: Group 0 corresponds to the pressure on all the subdomains. It will
        contain indices [0, 1, 2] which point to the pressure variable dofs on sd1, sd2
        and sd3, respectively. Combination of different variables in one group is also
        possible.

        Parameters:
            model: The PorePy model. The model should have the EquationSystem defined.
            md_variables_groups: The order of the groups of variables. Each group is a
                sequence of variables (either MixedDimensionalVariable or Variable).

        Returns:
            List of lists of integers. Each inner list contains the indices of the
                variables in defined in the respective item in md_variables_groups.

        """
        # Create a 0-based index for each variable.
        variable_to_idx = {
            var: i for i, var in enumerate(model.equation_system.variables)
        }
        indices = []
        for md_var_group in md_variables_groups:
            group_idx = []
            for md_var in md_var_group:
                # If we ever get a variable in here, we need to handle it directly, and
                # not call sub_vars.
                assert isinstance(md_var, pp.ad.MixedDimensionalVariable)
                group_idx.extend([variable_to_idx.pop(var) for var in md_var.sub_vars])
            indices.append(group_idx)
        assert len(variable_to_idx) == 0, "Some variables are not used."
        return indices

    def _equation_block_indices(
        self,
        model: pp.PorePyModel,
        equations_group_order: Sequence[Sequence[tuple[str, pp.GridLikeSequence]]],
    ) -> list[list[int]]:
        """Used to assemble the index that will later help accessing the submatrix
        corresponding to a group of equation, which may include one or more equation.

        Parameters:
            model: The PorePy model. The model should have the EquationSystem defined.
            equations_group_order: The order of the groups of equations. Each group is a
                sequence of tuples. Each tuple contains the name of the equation and the
                domain where it is applied.

        Returns:
            List of lists of integers. Each inner list contains the indices of the
                equations in defined in the respective item in equations_group_order.
                The indices refer to the block indices defined in
                model.equation_system._equation_image_space_composition.

        """
        # Assign a unique index to each equation-domain pair.
        equation_to_idx: dict[tuple[str, pp.GridLike], int] = {}
        idx: int = 0
        for (
            eq_name,
            domains,
        ) in model.equation_system._equation_image_space_composition.items():
            for domain in domains:
                equation_to_idx[(eq_name, domain)] = idx
                idx += 1

        indices: list[list[int]] = []
        # The outer loop define different groups of equations (to become blocks in the
        # block matrix).
        for group in equations_group_order:
            group_idx = []
            # Items in the group will contain a single equation defined on one or more
            # domains (subdomains or interfaces). Loop over equations an over all their
            # domains to add the indices to the group.
            for eq_name, domains_of_eq in group:
                for domain in domains_of_eq:
                    if (eq_name, domain) in equation_to_idx:
                        group_idx.append(equation_to_idx.pop((eq_name, domain)))
            indices.append(group_idx)

        # TODO EK: Added this assert just to verify that my understanding of the
        # function is correct. Delete it later.
        assert len(indices) == len(equations_group_order)
        assert len(equation_to_idx) == 0, "Some equations are not used."

        return indices
