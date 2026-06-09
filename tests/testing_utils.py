"""This includes:
- submatrices and their indices for various tests.
- mock model and mock dof manager.

"""

from typing import Optional

import numpy as np
import scipy.sparse as sp

from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.equation_variable_groups import EquationVariableGroup


def generate_reference_submatrices_3_groups():
    J00 = [
        [5, 1, 1],
        [2, 5, 1],
        [2, 2, 5],
    ]
    J01 = [
        [1, 3, 0, 0],
        [2, 3, 1, 1],
        [1, 3, 0, 0],
    ]
    J02 = [
        [3, -1],
        [-2, 3],
        [-2, 2],
    ]
    J10 = [
        [-2, 0, 2],
        [2, 0, -2],
        [-2, 1, 2],
        [2, 1, -2],
    ]
    J11 = [
        [6, 1, 5, 1],
        [1, 6, 1, 5],
        [2, 1, 6, 2],
        [1, 2, 2, 6],
    ]
    J12 = [
        [1, 1],
        [0, 1],
        [1, 2],
        [0, 2],
    ]
    J20 = [
        [-3, 1, 1],
        [2, -3, 1],
    ]
    J21 = [
        [-1, -3, 5, 0],
        [-2, -3, 0, 5],
    ]
    J22 = [
        [-3, 1],
        [2, -3],
    ]
    return J00, J01, J02, J10, J11, J12, J20, J21, J22


def generate_reference_matrix_3_groups():
    J00, J01, J02, J10, J11, J12, J20, J21, J22 = (
        generate_reference_submatrices_3_groups()
    )

    # The matrix is intentionally shuffled from the start, as we expect PorePy to
    # generate blocks not in the order we want here.
    return (
        sp.csr_matrix(
            sp.block_array(
                [
                    [J22, J21, J20],
                    [J12, J11, J10],
                    [J02, J01, J00],
                ]
            )
        )
        .astype(float)
        .tocsr()
    )


def generate_reference_rhs_3_groups():
    # The rhs is arranged for groups [2, 1, 0], same as the original matrix.
    return np.array([30, 31, 20, 21, 22, 23, 10, 11, 12], dtype=float)


def generate_reference_dofs_3_groups():
    # Empty 4-th group.
    reference_dofs_row_3_groups = [
        np.array(x, dtype=int) for x in [[6, 7, 8], [2, 3, 4, 5], [0, 1], []]
    ]

    # Empty 4-th group.
    reference_dofs_col_3_groups = [
        np.array(x, dtype=int) for x in [[6, 7, 8], [2, 3, 4, 5], [0, 1], []]
    ]
    return reference_dofs_row_3_groups, reference_dofs_col_3_groups


def generate_block_linear_system(
    num_dofs_per_group: list[int] | None = None,
):
    """Generate a random diagonally-dominant block linear system with a fixed random
    seed.

    Parameters:
        num_dofs_per_group: Number of DOFs in each block group. Groups are assigned
            contiguous DOF indices starting from 0. Defaults to [3, 4, 2, 0], matching
            the 3-group reference system (with an extra empty group).

    """
    if num_dofs_per_group is None:
        num_dofs_per_group = [3, 4, 2, 0]

    total = sum(num_dofs_per_group)
    rng = np.random.default_rng(42)
    A = rng.standard_normal((total, total))
    A = A @ A.T + total * np.eye(total)
    dofs = []
    start = 0
    for n in num_dofs_per_group:
        dofs.append(np.arange(start, start + n, dtype=int))
        start += n
    return BlockLinearSystem(
        mat=sp.csr_matrix(A),
        rhs=rng.standard_normal(total),
        indexer=LinearSystemIndexer(dofs_row=dofs, dofs_col=dofs),
    )


class MockModel:
    nd = 3


class MockDofManager:
    model = MockModel()

    def __init__(self, block_linear_system: Optional[BlockLinearSystem] = None):
        # Some tests need eq_dofs and var_dofs. They can use the dofs of the provided
        # block linear system. Tests that only need indices_of_groups may ignore it.
        self._block_linear_system = block_linear_system

    def indices_of_groups(self, groups: list[EquationVariableGroup]):
        # each mock group is a string "g1", "g2", etc.
        return [int(g[1]) - 1 for g in groups]

    def eq_dofs(self) -> list:
        if self._block_linear_system is None:
            raise ValueError(
                "Pass block_linear_system to MockDofManager to use eq_dofs()"
            )
        return self._block_linear_system.indexer.original_dofs_row

    def var_dofs(self) -> list:
        if self._block_linear_system is None:
            raise ValueError(
                "Pass block_linear_system to MockDofManager to use var_dofs()"
            )
        return self._block_linear_system.indexer.original_dofs_col
