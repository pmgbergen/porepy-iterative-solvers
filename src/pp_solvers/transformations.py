from abc import ABC, abstractmethod
from logging import getLogger
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix

from porepy.numerics.linalg.matrix_operations import invert_permuted_block_diag_matrix

from pp_solvers.block_linear_system import BlockLinearSystem, concatenate_dof_indices
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EquationVariableGroup,
    InterfaceForceBalanceGroup,
)
from pp_solvers.mat_utils import csr_ones, inv_block_diag

logger = getLogger(__name__)

# TODO: unit tests!


class LinearSystemTransformation(ABC):
    @abstractmethod
    def transform_matrix_rhs(
        self, block_linear_system: BlockLinearSystem, dof_manager: DofManager
    ) -> BlockLinearSystem:
        pass

    @abstractmethod
    def transform_solution(self, sol: np.ndarray) -> np.ndarray:
        pass


def rearrange_matrix_as_array_of_structures(bmat: BlockLinearSystem):
    enabled_groups_row = bmat.indexer.enabled_groups_row
    enabled_groups_col = bmat.indexer.enabled_groups_col
    assert len(enabled_groups_row) == len(enabled_groups_col)

    list_of_dofs_row = []
    list_of_dofs_col = []
    for group_row, group_col in zip(enabled_groups_row, enabled_groups_col):
        dofs_row, dofs_col = bmat.indexer.get_dofs_of_groups((group_row, group_col))
        assert len(dofs_row) == len(dofs_col)
        list_of_dofs_row.append(dofs_row)
        list_of_dofs_col.append(dofs_col)

    row_permutation = np.stack(list_of_dofs_row).ravel(order="F")
    col_permutation = np.stack(list_of_dofs_col).ravel(order="F")

    return row_permutation, col_permutation


class SchurComplementReduction(LinearSystemTransformation):
    def __init__(
        self,
        primary_groups: list[EquationVariableGroup],
        # invertor: Optional[Callable[[csr_matrix], csr_matrix]] = None,
    ):
        # if invertor is None:
        #     invertor = lambda mat: inv_block_diag(mat, nd=1)

        self.primary_groups: list[EquationVariableGroup] = primary_groups

    def transform_matrix_rhs(
        self, block_linear_system: BlockLinearSystem, dof_manager: DofManager
    ) -> BlockLinearSystem:
        secondary_groups = [
            g for g in dof_manager.groups() if g not in self.primary_groups
        ]

        keep_idx = dof_manager.indices_of_groups(self.primary_groups)
        elim_idx = dof_manager.indices_of_groups(secondary_groups)

        # needed to transform solution back
        primary_dofs_col = [block_linear_system.indexer.dofs_col[i] for i in keep_idx]
        self.primary_dofs_col = concatenate_dof_indices(primary_dofs_col)
        secondary_dofs_col = [block_linear_system.indexer.dofs_col[i] for i in elim_idx]
        self.secondary_dofs_col = concatenate_dof_indices(secondary_dofs_col)

        # 0 - elim, 1 - keep
        # A00 A01
        # A10 A11
        A00 = block_linear_system[elim_idx, elim_idx]
        A01 = block_linear_system[elim_idx, keep_idx]
        A10 = block_linear_system[keep_idx, elim_idx]
        A11 = block_linear_system[keep_idx, keep_idx]

        row_perm, col_perm = rearrange_matrix_as_array_of_structures(A00)
        bs = len(secondary_groups)
        block_sizes = np.ones(A00.shape[0] // bs, dtype=np.int64) * bs
        A00_inv = invert_permuted_block_diag_matrix(
            A00.mat,
            row_permutation=row_perm,
            col_permutation=col_perm,
            block_sizes=block_sizes,
        )

        A10_mul_A00_inv = A10.mat @ A00_inv

        # S11 = A11 - A10 * inv(A00) * A01
        S11 = A11.empty_container()
        S11.mat = A11.mat - A10_mul_A00_inv @ A01.mat

        # reduced rhs = b1 - A10 * inv(A00) * b0
        S11.rhs = A11.rhs - A10_mul_A00_inv @ A00.rhs

        self.A00_inv = A00_inv
        self.A01 = A01

        return S11

    def transform_solution(self, sol: np.ndarray) -> np.ndarray:
        # x0 = solve_A00(b0 - A01 @ x1)
        A01 = self.A01
        A00_inv = self.A00_inv

        x0 = A00_inv @ (A01.rhs - A01.mat @ sol)
        
        result = np.zeros(
            self.primary_dofs_col.size + self.secondary_dofs_col.size, dtype=sol.dtype
        )
        result[self.primary_dofs_col] = sol
        result[self.secondary_dofs_col] = x0
        return result


class ContactLinearTransformation(LinearSystemTransformation):
    def transform_matrix_rhs(
        self, block_linear_system: BlockLinearSystem, dof_manager: DofManager
    ) -> BlockLinearSystem:
        """Assemble the right linear transformation."""
        try:
            idx_contact = dof_manager.indices_of_groups([ContactMechanicsGroup()])[0]
        except ValueError:
            logger.warning(
                "You're using ContactLinearTransformation with no contact mechanics"
            )
            return block_linear_system

        try:
            idx_intf_force = dof_manager.indices_of_groups(
                [InterfaceForceBalanceGroup()]
            )
        except ValueError:
            logger.warning(
                "You're using ContactLinearTransformation with no interface force balance equation"
            )
            return block_linear_system

        if len(block_linear_system.indexer.dofs_row[idx_contact]) == 0:
            # If the relevant row group is empty (case without fractures), the
            # transformation is the identity matrix, nothing should be done.
            return block_linear_system

        # Pick out the block matrix corresponding to the interface force balance equation
        # (the row index) and the interface displacement variable (the column index). There
        # is an underlying assumption that the groups in the preconditioner ordering are so
        # that this equtaion-variable pair is on the diagonal of the matrix.
        J55 = block_linear_system[idx_intf_force, idx_intf_force].mat

        # The contribution from the interface displacement variable to the force balance
        # should be diagonally dominant, reflecting that the interface displacement has the
        # strongest influence on the force on its own cell (and less so on the neighboring
        # cell, though, with the MPSA stencil, the latter will not be zero). Note that there
        # is no connection between the two sides of a fracture; this is represented in a
        # different block of the full matrix. Approximate the stencil by a block diagonal,
        # and calculate the inverse cheaply.
        J55_inv = inv_block_diag(J55, nd=dof_manager.model.nd)

        # Extract the block matrix corresponding to the impact of the contact forces on the
        # force balance equation.
        J54 = block_linear_system[idx_intf_force, idx_contact].mat

        # The transformation is given like this, see papers by Zabegaev for the details.
        tmp = -J55_inv @ J54

        diagonal_part = block_linear_system.empty_container()
        diagonal_part.mat = csr_ones(diagonal_part.shape[0])

        # We add non-diagonal values to a matrix in list-of-lists format, then convert
        # it to csr. This prevents scipy performance warning. Real performance benefit
        # not measured.
        nondiagonal_part = block_linear_system.empty_container()
        nondiagonal_part.mat = nondiagonal_part.mat.tolil()  # type: ignore
        nondiagonal_part[idx_intf_force, idx_contact] = tmp
        nondiagonal_part.mat = nondiagonal_part.mat.tocsr()

        transformation_matrix = diagonal_part.mat + nondiagonal_part.mat

        self.transformation_matrix = transformation_matrix

        block_linear_system.mat @= transformation_matrix
        # The rhs remains untouched, since this is a right transfomration that applies
        # only to equations (rows), and not variables (columns): A * Q * Q^-1 x = rhs.
        return block_linear_system

    def transform_solution(self, sol: np.ndarray) -> np.ndarray:
        return self.transformation_matrix @ sol


class ScaleSpecificVolume(LinearSystemTransformation):
    def __init__(self, groups: list[EquationVariableGroup]):
        self.groups: list[EquationVariableGroup] = groups

    def transform_matrix_rhs(
        self, block_linear_system: BlockLinearSystem, dof_manager: DofManager
    ) -> BlockLinearSystem:
        """Assemble the right linear transformation for scaling energy fluxes."""
        try:
            idx_to_scale = dof_manager.indices_of_groups(self.groups)
        except ValueError:
            logger.warning("You're using ScaleSpecificVolume with empty groups.")
            return block_linear_system

        model = dof_manager.model

        subdomains = []
        for group in self.groups:
            equation = group.equation_group(model=model)
            subdomains.extend(equation.domains)

        if len(subdomains) == 0:
            # No subdomains, hence no scaling.
            return block_linear_system

        Q = block_linear_system.empty_container()

        values = 1.0 / model.equation_system.evaluate(model.specific_volume(subdomains))

        Q.mat = csr_ones(Q.shape[0])
        Q.set_diagonal(groups=idx_to_scale, values=values, additive=False)

        block_linear_system.mat = Q.mat @ block_linear_system.mat
        block_linear_system.rhs = Q.mat @ block_linear_system.rhs

        return block_linear_system

    def transform_solution(self, sol: np.ndarray) -> np.ndarray:
        # Only the equations (rows) and not the variables (columns) were reordered:
        # : Q * A * x = Q * rhs.
        return sol
