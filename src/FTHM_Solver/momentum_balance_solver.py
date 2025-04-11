import numpy as np
import porepy as pp
from functools import cached_property
from .full_petsc_solver import SolverScheme, PreconditionerScheme, PetscFieldSplitScheme
import FTHM_Solver


__all__ = ["MomentumIterativeScheme"]


class MomentumIterativeScheme(SolverScheme):
    def _register_equation_variable_groups(self):
        super()._register_equation_variable_groups()
        dim_max = self.model.mdg.dim_max()
        sd_ambient = self.model.mdg.subdomains(dim=dim_max)
        sd_frac = self.model.mdg.subdomains(dim=dim_max - 1)
        interfaces = self.model.mdg.interfaces(dim=dim_max - 1)

        self._equation_group_keys.append(
            [
                ("normal_fracture_deformation_equation", sd_frac),
                ("tangential_fracture_deformation_equation", sd_frac),
            ]
        )
        self._equation_group_keys.append([("momentum_balance_equation", sd_ambient)])
        self._equation_group_keys.append(
            [("interface_force_balance_equation", interfaces)]
        )

        # Register the groups of variables for this physics
        self._variable_groups_keys.append([self.model.contact_traction(sd_frac)])
        self._variable_groups_keys.append([self.model.displacement(sd_ambient)])
        self._variable_groups_keys.append(
            [self.model.interface_displacement(interfaces)]
        )

    def _reorder_eq_dofs(self) -> None:
        # First call the parent method, potentially setting off a chain of super-calls
        # to other classes.
        super()._reorder_eq_dofs()

        unpermuted_eq_dofs = self._eq_dofs

        contact_group_id = self._group_id_from_name(
            "normal_fracture_deformation_equation"
        )[0]
        # Short cut if no contact mechanics, hence no reordering.
        if len(self.equation_groups[contact_group_id]) == 0:
            # Ignore mypy error, list[np.ndarray] is a subset of list[np.ndarray |
            # None].
            return unpermuted_eq_dofs  # type: ignore[return-value]

        # We assume that normal equations go first. TODO: Can we make this more robust,
        # or else put an assert here.
        normal_blocks = self.equation_groups[contact_group_id]
        num_fracs = len(self.model.mdg.subdomains(dim=self.model.nd - 1))

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
            # There will be self.nd equations for each block.
            inds = offset + np.arange(unpermuted_eq_dofs[nb].size * self.model.nd)
            offset = inds[-1] + 1
            eq_dofs_corrected[nb] = np.array(inds)

        self._eq_dofs = eq_dofs_corrected

    def _reorder_equation_groups(self) -> None:
        # First call the parent method, potentially setting off a chain of super-calls
        # to other classes.
        super()._reorder_equation_groups()
        contact_group_id = self._group_id_from_name(
            "normal_fracture_deformation_equation"
        )[0]

        equation_groups = self._equation_groups

        if len(equation_groups[contact_group_id]) == 0:
            return equation_groups

        # Create a copy of the equation groups to avoid modifying the original.
        eq_groups_corrected = [x.copy() for x in equation_groups]

        num_fracs = len(self.model.mdg.subdomains(dim=self.model.nd - 1))
        # Index of the first block after the contact group. This and all subsequent
        # indexes will be reduced by the number of fractures (e.g., the number of
        # block equations that have been removed).
        block_after_contact = max(equation_groups[contact_group_id]) + 1

        # Change the number of blocks in the contact group to the number of fractures,
        # since we have merged the normal and tangential components.
        eq_groups_corrected[contact_group_id] = equation_groups[contact_group_id][
            :num_fracs
        ]

        # For all other groups with block index after the contact group, reduce the
        # block index by the number of fractures.
        for blocks in eq_groups_corrected:
            for i in range(len(blocks)):
                if blocks[i] >= block_after_contact:
                    blocks[i] -= num_fracs

        self._equation_groups = eq_groups_corrected

    def _reorder_row_indices(self, indices):
        # First call the parent method to invoke reoredring of different groups.
        super()._reorder_row_indices(indices)

        contact_group_id = self._group_id_from_name(
            "normal_fracture_deformation_equation"
        )[0]

        # Get the (fine-scale, not block(!)) dofs of the contact mechanics equations.
        dofs_contact = np.concatenate(
            [self.model.eq_dofs[i] for i in self.equation_groups[contact_group_id]]
        )

        # The start and end indices of all contact mechanics equations.
        dofs_contact_start = dofs_contact[0]
        dofs_contact_end = dofs_contact[-1] + 1

        # The number of cells in the contact mechanics equations.
        num_contact_cells = len(dofs_contact) // self.model.nd

        # 2d and 3d have respectively 1 and 2 tangential components, hence the branch.
        if self.model.nd == 2:
            # Rearrange the dofs into cell-wise blocks.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells:]
            indices[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1]
            ).ravel("F")
        elif self.model.nd == 3:
            # Do the same as in 2d, also for the second tangential component.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells::2]
            dofs_contact_2 = dofs_contact[num_contact_cells + 1 :: 2]
            indices[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1, dofs_contact_2]
            ).ravel("F")
        else:
            raise ValueError("Model dimension must be 2 or 3.")
        return indices

    def get_groups(self):
        raise NotImplementedError("")

    def _eliminate_contact_condition_scheme(
        self, complement, prefix: str = ""
    ) -> PetscFieldSplitScheme:
        contact_group_id = self._group_id_from_name(
            "normal_fracture_deformation_equation"
        )

        fieldsplit_options = {
            "pc_fieldsplit_schur_precondition": "selfp",
        }
        # PETSc's point block Jacobi preconditioner, with the given block
        # size.
        elim_options = {
            "pc_type": "pbjacobi",
        }

        for dct in [elim_options, fieldsplit_options]:
            self._add_prefix(dct, prefix)

        precond_scheme = PetscFieldSplitScheme(
            groups=contact_group_id,
            # The blocks are of size `nd`, the number of contact traction
            # components.
            block_size=self.model.nd,
            elim_options=elim_options,
            fieldsplit_options=fieldsplit_options,
            # TODO: What to do with this one?
            keep_options={
                "mat_schur_complement_ainv_type": "blockdiag",
            },
            complement=complement,
        )
        return precond_scheme

    def _momentum_balance_scheme(
        self, complement, prefix: str = ""
    ) -> PreconditionerScheme:
        # Get the group id of the momentum balance equation
        momentum_group_id = self._group_id_from_name("momentum_balance_equation")
        interface_group_id = self._group_id_from_name(
            "interface_force_balance_equation"
        )
        groups = momentum_group_id + interface_group_id

        opts = {
            "pc_type": "gamg",
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": 1,
            "mg_levels_pc_type": "bjacobi",
            "mg_levels_pc_factor_mat_solver_type": "superlu_dist",
        }

        fieldsplit_options = {
            "pc_fieldsplit_schur_precondition": "selfp",
        }
        # Add the prefix to the options
        for dct in [opts, fieldsplit_options]:
            self._add_prefix(dct, prefix)

        # Set up the preconditioner scheme
        if complement is None:
            precond_scheme = FTHM_Solver.SinglePhysicsPreconditionerScheme(
                groups=groups, opts=opts
            )

        else:
            precond_scheme = FTHM_Solver.PetscFieldSplitScheme(
                groups=groups,
                block_size=self.nd,
                elim_options=opts,
                fieldsplit_options=fieldsplit_options,
                complement=complement,
            )

        return precond_scheme

    def make_solver_scheme(self, opts=None):
        # Create the preconditioner scheme for the contact condition
        precond_scheme = self._eliminate_contact_condition_scheme(
            complement=self._momentum_balance_scheme(complement=None)
        )

        # Create the KSP scheme
        ksp_options = {"ksp_monitor": None}

        ksp_scheme = FTHM_Solver.PetscKSPScheme(
            preconditioner=precond_scheme, petsc_options=ksp_options
        )
        return ksp_scheme
