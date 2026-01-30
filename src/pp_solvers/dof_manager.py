from __future__ import annotations

from itertools import count

import numpy as np
import porepy as pp

from pp_solvers.equation_variable_groups import (
    EquationOnDomains,
    EquationVariableGroup,
)

from pp_solvers.block_linear_system import concatenate_dof_indices


class DofManager:
    """Takes care of translation of blocks and groups (from EquationSystem format) to
    block indices, as well as grouping the fine-scale dofs. Also reordering related to
    the contact problem.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so.
    """

    def __init__(self, groups: list[EquationVariableGroup]):
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
        self._groups: list[EquationVariableGroup] = groups

    def indices_of_groups(self, groups: list[EquationVariableGroup]):
        return [self._groups.index(x) for x in groups]

    def equation_names(self, model: pp.PorePyModel) -> list[str]:
        """Get the names of equations in the model.

        Parameters:
            model: The PorePy model.

        Returns:
            A list of strings containing the names of equations in the model.

        """
        return [g.equation_name(model) for g in self._groups]

    def variable_names(self, model: pp.PorePyModel) -> list[str]:
        """Get the names of variables in the model.

        Parameters:
            model: The PorePy model.

        Returns:
            A list of strings containing the names of variables in the model.

        """

        return [g.variable_name(model) for g in self._groups]

    def eq_dofs(self, model: pp.PorePyModel) -> list[np.ndarray]:
        equation_groups = [g.equation_group(model) for g in self._groups]

        indices_of_dofs = self._equation_block_indices(model, equation_groups)
        dofs_in_porepy_order = self._eq_dofs_by_blocks(model)
        dofs_row = [
            concatenate_dof_indices([dofs_in_porepy_order[i] for i in dofs_in_group])
            for dofs_in_group in indices_of_dofs
        ]
        return dofs_row

    def var_dofs(self, model: pp.PorePyModel) -> list[np.ndarray]:
        variable_groups = [g.variable_group(model) for g in self._groups]

        variable_indices = self._variable_block_indices(
            model=model, md_variables_groups=variable_groups
        )
        var_dofs_by_blocks = self._var_dofs_by_blocks(model)

        dofs_col = [
            concatenate_dof_indices([var_dofs_by_blocks[i] for i in dofs_in_group])
            for dofs_in_group in variable_indices
        ]
        return dofs_col

    def _eq_dofs_by_blocks(self, model) -> list[np.ndarray]:
        """Get the equation dofs for the model, in the form of a list of numbers,
        one per equation-domain pair. If the contact group is present, it will be
        reordered so that the normal and tangential equations for each fracture cell
        form a digonal block.
        """
        skip_list = {
            "local_component_mass_constraint_CO2",
            "isofugacity_constraint_H2O_G_L",
            "isofugacity_constraint_CO2_G_L",
            "semismooth_complementary_condition_L",
            "semismooth_complementary_condition_G",
            "local_fluid_enthalpy_constraint",
            "local_phase_mass_constraint_G",
        }
        eq_dofs: list[np.ndarray] = []
        offset = 0
        for (
            eq_name,
            data,
        ) in model.equation_system._equation_image_space_composition.items():
            if eq_name in skip_list:
                continue
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
                eq_dofs_corrected.append(None)  # TODO: YZ - Why?

        offset = eq_dofs[normal_blocks[0]][0]
        for nb in normal_blocks:
            # Create indices for the normal and tangential components of the contact.
            # There will be model.nd equations for each block.
            inds = offset + np.arange(eq_dofs[nb].size * model.nd)
            offset = inds[-1] + 1
            eq_dofs_corrected[nb] = np.array(inds)

        return eq_dofs_corrected

    def _var_dofs_by_blocks(self, model) -> list[np.ndarray]:
        """Variable degrees of freedom (columns of the Jacobian) in the PorePy order
        (how they are arranged in the model).

        Returns:
            List of numpy arrays. Each array contains the global degrees of freedom for
                one variable on one grid and provides the fine-scale (actual column
                indices) of the variable.

        """
        skip_list = {
            "temperature",
            "s_G",
            "y_G",
            "x_H2O_L",
            "x_CO2_L",
            "x_H2O_G",
            "x_CO2_G",
        }

        proj = model.equation_system._Schur_complement[3].T.tocsc()

        var_dofs: list[np.ndarray] = []
        for var in model.equation_system.variables:
            if var.name in skip_list:
                continue
            var_dofs.append(
                proj.indices[proj.indptr[model.equation_system.dofs_of([var])]]
            )
        return var_dofs

    def identify_contact_group(self) -> int | None:
        """Identify the contact group in the equation groups.

        It is assumed that there is a single contact group, that is, that the contact
        blocks are not split into several groups.

        Returns:
            The index of the contact group in the equation groups. If no contact group
            is found, returns None.

        """
        return None
        # # Identify the contact group in the equation groups
        # ind = self._name_to_group_indices.get(
        #     EquationNames.CONTACT_NORMAL.value, [None]
        # )
        # return ind[0]

    def identify_energy_balance_groups(self) -> list[int]:
        """Identify the energy balance groups in the equation groups.

        Returns:
            The indices of the energy balance group in the equation groups.

        """
        return []
        # return self._name_to_group_indices.get(EquationNames.ENERGY_BALANCE.value, [])

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
        return None
        # if not hasattr(model, "interface_displacement_variable"):
        #     # The model does not have an interface displacement variable.
        #     return None

        # # Identify the interface group in the equation groups.
        # i = 0

        # # Note to self: Here we need to loop over the _orderings, since we need to match
        # # the ordering of the preconditioner to porepy information. See also the other
        # # identify methods.
        # for group in self._orderings:
        #     if isinstance(group, list):
        #         for sub_group in group:
        #             if len(sub_group.variable_groups(model)) == 0:
        #                 continue
        #             for var in sub_group.variable_groups(model):
        #                 if var[0].name == model.interface_displacement_variable:
        #                     return i
        #     else:
        #         for var in group.variable_groups(model):
        #             # Loop over all the variables of the group (variables treated with
        #             # this block solver). See if this is the one.
        #             if var[0].name == model.interface_displacement_variable:
        #                 return i
        #             else:
        #                 i += 1
        # return None

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
                self._eq_dofs_by_blocks(model)[i]
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
        md_variables_groups: list[pp.ad.MixedDimensionalVariable | pp.ad.Variable],
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
        skip_list = {
            "temperature",
            "s_G",
            "y_G",
            "x_H2O_L",
            "x_CO2_L",
            "x_H2O_G",
            "x_CO2_G",
        }

        counter = count(0)
        variable_to_idx = {
            var: next(counter)
            for var in model.equation_system.variables
            if var.name not in skip_list
        }
        indices = []
        for md_var in md_variables_groups:
            # If we ever get a variable in here, we need to handle it directly, and
            # not call sub_vars.
            assert isinstance(md_var, pp.ad.MixedDimensionalVariable)
            indices.append([variable_to_idx.pop(var) for var in md_var.sub_vars])
        assert len(variable_to_idx) == 0, "Some variables are not used."
        return indices

    def _equation_block_indices(
        self,
        model: pp.PorePyModel,
        equations_group_order: list[EquationOnDomains],
    ) -> list[list[int]]:
        """Used to assemble the index that will later help accessing the submatrix
        corresponding to a group of equation, which may include one or more equation.

        Parameters:
            model: The PorePy model. The model should have the EquationSystem defined.
            equations_group_order: The order of the groups of equations. Each group is a
                sequence of tuples. Each tuple contains the name of the equation and the
                domain where it is applied. (TODO)

        Returns:
            List of lists of integers. Each inner list contains the indices of the
                equations in defined in the respective item in equations_group_order.
                The indices refer to the block indices defined in
                model.equation_system._equation_image_space_composition.

        """
        # Assign a unique index to each equation-domain pair.

        skip_list = {
            "local_component_mass_constraint_CO2",
            "isofugacity_constraint_H2O_G_L",
            "isofugacity_constraint_CO2_G_L",
            "semismooth_complementary_condition_L",
            "semismooth_complementary_condition_G",
            "local_fluid_enthalpy_constraint",
            "local_phase_mass_constraint_G",
        }

        equation_to_idx: dict[tuple[str, pp.GridLike], int] = {}
        idx: int = 0
        for (
            eq_name,
            domains,
        ) in model.equation_system._equation_image_space_composition.items():
            if eq_name in skip_list:
                continue
            for domain in domains:
                equation_to_idx[(eq_name, domain)] = idx
                idx += 1

        indices: list[list[int]] = []
        # The outer loop define different groups of equations (to become blocks in the
        # block matrix).
        for equation_on_domains in equations_group_order:
            eq_name = equation_on_domains.name
            domains = equation_on_domains.domains
            # Items in the group will contain a single equation defined on one or more
            # domains (subdomains or interfaces). Loop over equations an over all their
            # domains to add the indices to the group.
            indices_group: list[int] = []
            for domain in domains:
                if (eq_name, domain) in equation_to_idx:
                    indices_group.append(equation_to_idx.pop((eq_name, domain)))
            indices.append(indices_group)

        # TODO EK: Added this assert just to verify that my understanding of the
        # function is correct. Delete it later.
        assert len(indices) == len(equations_group_order)
        assert len(equation_to_idx) == 0, "Some equations are not used."

        return indices
