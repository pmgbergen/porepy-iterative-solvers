"""This file contains various tests mainly to examine slicing in the BlockMatrixStorage
class.

"""

import numpy as np
import pytest
import scipy.sparse as sp
from testing_utils import (
    generate_reference_dofs_3_groups,
    generate_reference_matrix_3_groups,
    generate_reference_rhs_3_groups,
    generate_reference_submatrices_3_groups,
)

from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer


@pytest.fixture
def sample_linear_system() -> BlockLinearSystem:
    """The block matrix we use in all the tests. It contains one empty group."""
    reference_dofs_row_3_groups, reference_dofs_col_3_groups = (
        generate_reference_dofs_3_groups()
    )
    return BlockLinearSystem(
        mat=generate_reference_matrix_3_groups(),
        rhs=generate_reference_rhs_3_groups(),
        indexer=LinearSystemIndexer(
            dofs_row=reference_dofs_row_3_groups,
            dofs_col=reference_dofs_col_3_groups,
        ),
    )


J00, J01, J02, J10, J11, J12, J20, J21, J22 = generate_reference_submatrices_3_groups()


@pytest.mark.parametrize(
    "params",
    [
        # Diagonal blocks are not square submatrices.
        {
            "submatrices": [[J10, J11], [J00, J01]],
            "dofs_row": [[0, 1, 2, 3], [4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "raises": True,
        },
        # Too few dofs.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5]],
            "dofs_col": [[0, 1, 2], [3, 4, 5]],
            "raises": True,
        },
        # Too many dofs.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "dofs_col": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "raises": True,
        },
        # Correct shape but garbage values.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[1, 7, 123], [3, 55, 66, 87]],
            "dofs_col": [[643, 12, 42], [312, 2, 32, 52]],
            "raises": True,
        },
        # Too few enabled groups.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "enabled_groups_row": [1],
            "enabled_groups_col": [1],
            "raises": True,
        },
        # Too many enabled groups.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "enabled_groups_row": [0, 1, 2],
            "enabled_groups_col": [0, 1, 2],
            "raises": True,
        },
        # Correct number of enabled groups, but garbage values.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "enabled_groups_row": [0, 5],
            "enabled_groups_col": [0, 5],
            "raises": True,
        },
        # Just a normal matrix creation with all default parameters.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "raises": False,
        },
        # Inconsistent rhs shape.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "rhs": np.ones(8, dtype=float),
            "raises": True,
        },
        # Inconsistent group names - too few.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "group_names_row": ["a"] * 6,
            "group_names_col": ["b"] * 6,
            "raises": True,
        },
        # Inconsistent group names - too many.
        {
            "submatrices": [[J00, J01], [J10, J11]],
            "dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "group_names_row": ["a"] * 8,
            "group_names_col": ["b"] * 8,
            "raises": True,
        },
    ],
)
def test_block_linear_system_creation(params):
    """This test examines invalid arguments passed to the block matrix constructor."""
    submatrices = params["submatrices"]
    dofs_row = params["dofs_row"]
    dofs_col = params["dofs_col"]
    raises = params["raises"]
    rhs = params.get("rhs", [1] * 7)
    enabled_groups_row = params.get("enabled_groups_row", None)
    enabled_groups_col = params.get("enabled_groups_col", None)
    group_names_row = params.get("group_names_row", None)
    group_names_col = params.get("group_names_col", None)
    if raises:
        with pytest.raises(Exception):
            _ = BlockLinearSystem(
                mat=sp.csr_matrix(sp.block_array(submatrices).astype(float)),
                rhs=np.array(rhs, dtype=float),
                indexer=LinearSystemIndexer(
                    dofs_row=[np.array(x) for x in dofs_row],
                    dofs_col=[np.array(x) for x in dofs_col],
                    enabled_groups_row=enabled_groups_row,
                    enabled_groups_col=enabled_groups_col,
                    group_names_row=group_names_row,
                    group_names_col=group_names_col,
                ),
            )
    else:
        _ = BlockLinearSystem(
            mat=sp.csr_matrix(sp.block_array(submatrices).astype(float)),
            rhs=np.ones(7, dtype=float),
            indexer=LinearSystemIndexer(
                dofs_row=[np.array(x) for x in dofs_row],
                dofs_col=[np.array(x) for x in dofs_col],
                enabled_groups_row=enabled_groups_row,
                enabled_groups_col=enabled_groups_col,
                group_names_row=group_names_row,
                group_names_col=group_names_col,
            ),
        )


@pytest.mark.parametrize(
    "params",
    [
        # J[:] (all, sorted).
        {"index": slice(None, None, None), "expected": (9, 9)},
        # J[0] (invalid, single scalar index not allowed).
        {"index": 0, "raises": True},
        # J[[1]] (same as [1, 1]).
        {"index": [1], "expected": (4, 4)},
        # J[[0, 1, 2]] (same as J[:]).
        {"index": [0, 1, 2], "expected": (9, 9)},
        # J[[2, 0]]  (same as J[[2, 0], [2, 0]]).
        {"index": [2, 0], "expected": (5, 5)},
        # J[[-2]] (same as in numpy, indexing from end).
        {"index": [-2], "expected": (2, 2)},
        # J[[3]] (slicing empty group, same as J[3, 3]).
        {"index": [3], "expected": (0, 0)},
        # J[[4]] (invalid, index over the bound).
        {"index": [4], "raises": True},
        # J[[0, 4]] (invalid, includes good and bad index).
        {"index": [0, 4], "raises": True},
        # J[:2] (same as J[[0, 1]]).
        {"index": slice(None, 2), "expected": (7, 7)},
        # J[1:] (same as [[1, 2]]).
        {"index": slice(1, None), "expected": (6, 6)},
        # J[::2] (same as in numpy, only even [0, 2]).
        {"index": slice(None, None, 2), "expected": (5, 5)},
        # J[1::2] (same as in numpy, only odd [1]).
        {"index": slice(1, None, 2), "expected": (4, 4)},
        # J[::-1] (invalid, not implemented).
        {"index": slice(None, None, -1), "raises": True},
    ],
)
def test_shape_simple_index(sample_linear_system: BlockLinearSystem, params):
    """All the possible combinations of simple indexing, e.g. J[x]."""
    index: slice | int | list[int] = params["index"]
    raises: bool = params.get("raises", False)
    expected: tuple[int, int] | None = params.get("expected", None)
    if not raises:
        assert sample_linear_system[index].shape == expected, index
    else:
        with pytest.raises(Exception):
            sample_linear_system[index]


# We test indexing with the Cartesian product of these parameters for rows and cols.
PARAMS_FOR_TEST_SHAPE_TUPLE_INDEX = [
    # J[:, x] (all, sorted).
    {"index": slice(None, None, None), "expected": 9},
    # J[2, x] (group 2 only).
    {"index": 2, "expected": 2},
    # J[[1], x] (same as [1, x]).
    {"index": [1], "expected": 4},
    # J[[0, 1, 2], x] (same as J[:, x]).
    {"index": [0, 1, 2], "expected": 9},
    # J[[2, 0], x]  (same as J[[2, 0], x]).
    {"index": [2, 0], "expected": 5},
    # J[[-2], [-2]] (same as in numpy, indexing from end).
    {"index": [-2], "expected": 2},
    # J[[3], [3]] (slicing empty group, same as J[3, 3]).
    {"index": [3], "expected": 0},
    # J[[4]] (invalid, index over the bound).
    {"index": [4], "raises": True},
    # J[[0, 4], x] (invalid, includes good and bad index).
    {"index": [0, 4], "raises": True},
    # J[:2, x] (same as J[[0, 1], x]).
    {"index": slice(None, 2), "expected": 7},
    # J[1:, x] (same as J[[1, 2], x]).
    {"index": slice(1, None), "expected": 6},
    # J[::2, x] (same as in numpy, only even [0, 2]).
    {"index": slice(None, None, 2), "expected": 5},
    # J[1::2, x] (same as in numpy, only odd [1]).
]


@pytest.mark.parametrize("params_row", PARAMS_FOR_TEST_SHAPE_TUPLE_INDEX)
@pytest.mark.parametrize("params_col", PARAMS_FOR_TEST_SHAPE_TUPLE_INDEX)
def test_shape_tuple_index(
    sample_linear_system: BlockLinearSystem, params_row, params_col
):
    """All the possible combinations of tuple indexing, e.g. J[x, y]."""
    ind_row: slice | int | list[int] = params_row["index"]
    raises_row: bool = params_row.get("raises", False)
    expected_row: int | None = params_row.get("expected", None)
    ind_col: slice | int | list[int] = params_col["index"]
    raises_col: bool = params_col.get("raises", False)
    expected_col: int | None = params_col.get("expected", None)
    if not (raises_row or raises_col):
        assert sample_linear_system[ind_row, ind_col].shape == (
            expected_row,
            expected_col,
        )
    else:
        with pytest.raises(Exception):
            sample_linear_system[ind_row, ind_col]


@pytest.mark.parametrize(
    "params",
    [
        # J[:] (all, sorted).
        {
            "index": slice(None, None, None),
            "expected": [
                [J00, J01, J02],
                [J10, J11, J12],
                [J20, J21, J22],
            ],
        },
        # J[2, 0].
        {"index": (2, 0), "expected": [[J20]]},
        # J[0, 0].
        {"index": (0, 0), "expected": [[J00]]},
        # J[1, 1].
        {"index": (1, 1), "expected": [[J11]]},
        # J[2, 2].
        {"index": (2, 2), "expected": [[J22]]},
        # J[[2, 1, 0], 2].
        {"index": ([2, 1, 0], 2), "expected": [[J22], [J12], [J02]]},
        # J[2, [1, 2, 0]].
        {"index": (2, [1, 2, 0]), "expected": [[J21, J22, J20]]},
        # J[[2, 1], [2, 0]].
        {"index": ([2, 1], [2, 0]), "expected": [[J22, J20], [J12, J10]]},
        # J[[1, 1], [0, 2]] (repeated index).
        {"index": ([1, 1], [0, 2]), "expected": [[J10, J12], [J10, J12]]},
        # J[:, 1] (rows should be sorted, same as [0, 1, 2]).
        {"index": (slice(None, None, None), 1), "expected": [[J01], [J11], [J21]]},
        # J[1, :] (columns should be sorted, same as [0, 1, 2]).
        {"index": (1, slice(None, None, None)), "expected": [[J10, J11, J12]]},
    ],
)
def test_slicing(sample_linear_system: BlockLinearSystem, params):
    """These test ensures that the sliced matrices exactly match the expectation."""
    index = params["index"]
    expected = params["expected"]
    expected = [[sp.csr_array(submat) for submat in row] for row in expected]
    expected = sp.block_array(expected).astype(float).toarray()
    np.testing.assert_array_equal(sample_linear_system[index].mat.toarray(), expected)


@pytest.mark.parametrize(
    "params",
    [
        # J[[2, 1], [1, 0]][1, 0]. Equivalent to J[1, 0].
        {
            "index1": ([2, 1], [1, 0]),
            "index2": (1, 0),
            "expected": (4, 3),
        },
        # J[2, :][2, 0]. Equivalent to J[2, 0].
        {
            "index1": (2, slice(None, None, None)),
            "index2": (2, 0),
            "expected": (2, 3),
        },
        # J[[1, 2]][[0, 1]]. Invalid (0 is not present in [1, 2]).
        {
            "index1": [1, 2],
            "index2": [0, 1],
            "raises": True,
        },
    ],
)
def test_nested_slicing(sample_linear_system: BlockLinearSystem, params):
    """Tests that the slice of a slice works as expected. For some cases, second
    indexing is expected to fail.

    """
    index1 = params["index1"]
    index2 = params["index2"]
    expected = params.get("expected", None)
    raises: bool = params.get("raises", False)

    tmp = sample_linear_system[index1]
    if not raises:
        assert tmp[index2].shape == expected
    else:
        with pytest.raises(IndexError):
            tmp[index2]


@pytest.mark.parametrize(
    "params",
    [
        # J[[2, 0]] = 99
        {
            "index": [2, 0],
            "modify_submatrices": [
                [True, False, True],
                [False, False, False],
                [True, False, True],
            ],
        },
        {
            # J[:, 1] = 99
            "index": (slice(None, None, None), 1),
            "modify_submatrices": [
                [False, True, False],
                [False, True, False],
                [False, True, False],
            ],
        },
        {
            # J[2, 0] = 99
            "index": (2, 0),
            "modify_submatrices": [
                [False, False, False],
                [False, False, False],
                [True, False, False],
            ],
        },
        {
            # Empty group. J[:, 3] = 1. Should do nothing.
            "index": (slice(None, None, None), 3),
            "modify_submatrices": [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
        },
    ],
)
def test_setitem(sample_linear_system: BlockLinearSystem, params):
    """Tests that __setitem__ works as expected."""
    index = params["index"]
    modify_submatrices = params["modify_submatrices"]
    fill_value = 99

    # Constructing modified matrices
    original_matrix = [
        [J00, J01, J02],
        [J10, J11, J12],
        [J20, J21, J22],
    ]
    original_matrix = [
        [sp.csr_array(submat) for submat in row] for row in original_matrix
    ]
    expected = []
    for row, modify_row in zip(original_matrix, modify_submatrices):
        expected_row = []
        expected.append(expected_row)
        for submat, modify_submat in zip(row, modify_row):
            if modify_submat:
                submat[:] = fill_value
            expected_row.append(submat)
    expected = sp.block_array(expected).astype(float).toarray()

    sample_linear_system[index] = fill_value
    np.testing.assert_array_equal(sample_linear_system[:].mat.toarray(), expected)


def test_copy(sample_linear_system: BlockLinearSystem):
    copied_mat = sample_linear_system.copy()
    assert copied_mat.shape == sample_linear_system.shape
    np.testing.assert_array_equal(
        sample_linear_system.mat.toarray(), copied_mat.mat.toarray()
    )
    assert sample_linear_system.mat.data is not copied_mat.mat.data


def test_print(sample_linear_system: BlockLinearSystem):
    print(sample_linear_system)


def test_empty_container(sample_linear_system: BlockLinearSystem):
    empty_matrix = sample_linear_system.empty_container()
    assert empty_matrix.shape == sample_linear_system.shape
    assert empty_matrix.mat.nnz == 0
    assert np.all(empty_matrix.rhs == 0)


# @pytest.mark.parametrize(
#     "params",
#     [
#         # Original matrix, nothing changes.
#         {
#             "index": [2, 1, 0],
#             "expected_permuted": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#             "expected_original": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#         },
#         # Full matrix, sorted groups.
#         {
#             "index": [0, 1, 2],
#             "expected_permuted": [10, 11, 12, 20, 21, 22, 23, 30, 31],
#             "expected_original": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#         },
#         # One row and one column, same as J[1, 1].
#         {
#             "index": [1],
#             "expected_permuted": [20, 21, 22, 23],
#             "expected_original": [0, 0, 20, 21, 22, 23, 0, 0, 0],
#         },
#         # Repeated group index, same as J[[1, 1]].
#         {
#             "index": [1, 1],
#             "expected_permuted": [20, 21, 22, 23, 20, 21, 22, 23],
#             "expected_original": [0, 0, 20, 21, 22, 23, 0, 0, 0],
#         },
#         # All rows, one column, same as J[:, 1]. Rhs should NOT be truncated.
#         {
#             "index": ([0, 1, 2], 1),
#             "expected_permuted": [10, 11, 12, 20, 21, 22, 23, 30, 31],
#             "expected_original": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#         },
#         # All columns, one row, same as J[1, :]. RHS should be truncated.
#         {
#             "index": (1, [0, 1, 2]),
#             "expected_permuted": [20, 21, 22, 23],
#             "expected_original": [0, 0, 20, 21, 22, 23, 0, 0, 0],
#         },
#         # Empty group.
#         {
#             "index": (
#                 3,
#                 [
#                     0,
#                     1,
#                     2,
#                 ],
#             ),
#             "expected_permuted": [],
#             "expected_original": [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         },
#     ],
# )
# def test_permute_left_vector_to_local_and_original(
#     sample_linear_system: BlockLinearSystem, params
# ):
#     """Tests projecting the left vector (rhs) to the permuted arrangement and the
#     reverse transformation from it with `permute_left_vector_to_original`.

#     """
#     # The original rhs is arranged for groups [2, 1, 0], same as the original matrix.
#     # rhs original: [30, 31, 20, 21, 22, 23, 10, 11, 12]
#     index = params["index"]
#     expected_permuted = params["expected_permuted"]
#     expected_original = params["expected_original"]

#     sample_linear_system = sample_linear_system[index]

#     rhs_local = sample_linear_system.rhs
#     np.testing.assert_array_equal(rhs_local, expected_permuted)

#     rhs_back_to_orig = sample_linear_system.permute_left_vector_to_original(rhs_local)
#     np.testing.assert_array_equal(rhs_back_to_orig, expected_original)


# @pytest.mark.parametrize(
#     "params",
#     [
#         # Original matrix, nothing changes.
#         {
#             "index": [2, 1, 0],
#             "solution_permuted": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#             "expected_original": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#         },
#         # Full matrix, sorted groups.
#         {
#             "index": [0, 1, 2],
#             "solution_permuted": [10, 11, 12, 20, 21, 22, 23, 30, 31],
#             "expected_original": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#         },
#         # One row and one column, same as J[1, 1].
#         {
#             "index": [1],
#             "solution_permuted": [20, 21, 22, 23],
#             "expected_original": [0, 0, 20, 21, 22, 23, 0, 0, 0],
#         },
#         # Repeated group index, same as J[[1, 1]].
#         {
#             "index": [1, 1],
#             "solution_permuted": [20, 21, 22, 23, 20, 21, 22, 23],
#             "expected_original": [0, 0, 20, 21, 22, 23, 0, 0, 0],
#         },
#         # All rows, one column, same as J[:, 1]. Soultion should be truncated.
#         {
#             "index": ([0, 1, 2], 1),
#             "solution_permuted": [20, 21, 22, 23],
#             "expected_original": [0, 0, 20, 21, 22, 23, 0, 0, 0],
#         },
#         # All columns, one row, same as J[1, :]. Soultion should NOT be truncated.
#         {
#             "index": (1, [0, 1, 2]),
#             "solution_permuted": [10, 11, 12, 20, 21, 22, 23, 30, 31],
#             "expected_original": [30, 31, 20, 21, 22, 23, 10, 11, 12],
#         },
#         # Empty group.
#         {
#             "index": ([0, 1, 2], 3),
#             "solution_permuted": [],
#             "expected_original": [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         },
#     ],
# )
# def test_permute_right_vector_to_original(
#     sample_linear_system: BlockLinearSystem, params
# ):
#     """Tests `permute_right_vector_to_original`. Results should be different from
#     `permute_left_vector_to_original`, because the solution corresponds to the space
#     defined by columns of the matrix, and the rhs - to the one defined by rows (Ax = b).

#     """
#     index = params["index"]
#     solution_permuted = params["solution_permuted"]
#     expected_original = params["expected_original"]

#     sample_linear_system = sample_linear_system[index]
#     solution_permuted = np.array(solution_permuted)

#     solution_back_to_global = sample_linear_system.permute_right_vector_to_original(
#         solution_permuted
#     )
#     np.testing.assert_array_equal(solution_back_to_global, expected_original)


@pytest.mark.parametrize(
    "params",
    [
        # Fill the whole matrix with zeros.
        {
            "index_row": [2, 0, 1],
            "index_col": [1, 2, 0],
            "modify_submatrices": [
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
        },
        # A single submatrix.
        {
            "index_row": 0,
            "index_col": 2,
            "modify_submatrices": [
                [False, False, True],
                [False, False, False],
                [False, False, False],
            ],
        },
        # Non-contigious groups.
        {
            "index_row": [0, 2],
            "index_col": [2, 0],
            "modify_submatrices": [
                [True, False, True],
                [False, False, False],
                [True, False, True],
            ],
        },
    ],
)
def test_set_zeros(sample_linear_system: BlockLinearSystem, params):
    index_row = params["index_row"]
    index_col = params["index_col"]
    modify_submatrices = params["modify_submatrices"]

    # Constructing modified matrices
    original_matrix = [
        [J00, J01, J02],
        [J10, J11, J12],
        [J20, J21, J22],
    ]
    original_matrix = [
        [sp.csr_array(submat) for submat in row] for row in original_matrix
    ]
    expected = []
    for row, modify_row in zip(original_matrix, modify_submatrices):
        expected_row = []
        expected.append(expected_row)
        for submat, modify_submat in zip(row, modify_row):
            if modify_submat:
                submat[:] = 0
            expected_row.append(submat)
    expected = sp.block_array(expected).astype(float).toarray()

    # Rearranging the matrix a few times to make it more challenging.
    sample_linear_system = sample_linear_system[[1, 2, 0], [2, 0, 1]][[2, 1, 0]][
        [0, 2, 1]
    ]

    sample_linear_system.set_zeros(group_row_idx=index_row, group_col_idx=index_col)
    np.testing.assert_array_equal(sample_linear_system[:].mat.toarray(), expected)


@pytest.mark.parametrize(
    "params",
    [
        # Fill the whole main diagonal, sorted groups.
        {
            "expected_diagonal_change": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "fill_values": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "fill_groups": [0, 1, 2],
        },
        # Fill the whole main diagonal, different groups sorting.
        {
            "expected_diagonal_change": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "fill_values": [20, 21, 22, 23, 10, 11, 12, 30, 31],
            "fill_groups": [1, 0, 2],
        },
        # Fill a single group.
        {
            "expected_diagonal_change": [0, 0, 0, 20, 21, 22, 23, 0, 0],
            "fill_groups": [1],
            "fill_values": [20, 21, 22, 23],
        },
        # Using scalar.
        {
            "expected_diagonal_change": [0, 0, 0, 20, 20, 20, 20, 0, 0],
            "fill_groups": [1],
            "fill_values": 20,
        },
        # Non-contigious groups.
        {
            "expected_diagonal_change": [10, 11, 12, 0, 0, 0, 0, 30, 31],
            "fill_groups": [2, 0],
            "fill_values": [30, 31, 10, 11, 12],
        },
        # Filling empty group.
        {
            "expected_diagonal_change": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "fill_groups": [3],
            "fill_values": [],
        },
    ],
)
@pytest.mark.parametrize("additive", [False, True])
def test_set_diagonal(sample_linear_system: BlockLinearSystem, params, additive: bool):
    expected_diagonal_change = params["expected_diagonal_change"]
    fill_groups = params["fill_groups"]
    fill_values = params["fill_values"]

    # Generating the expected matrix.
    expected_mat = sample_linear_system[:]
    if not additive:
        expected_mat[fill_groups] = (
            expected_mat[fill_groups].mat
            - sp.diags(expected_mat[fill_groups].mat.diagonal()).tocsr()
        )
    expected_mat.mat += sp.diags(np.array(expected_diagonal_change))

    sample_linear_system.set_diagonal(
        groups=fill_groups, values=np.array(fill_values), additive=additive
    )
    np.testing.assert_array_equal(
        sample_linear_system[:].mat.toarray(), expected_mat.mat.toarray()
    )


@pytest.fixture(scope="module", autouse=True)
def matplotlib_use_dummy_backend():
    import matplotlib

    matplotlib.use("template")


# Ignoring a warning that we cannot show matplotlib figures.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*FigureCanvasTemplate is non-interactive.*:UserWarning"
)


@pytest.mark.parametrize("annot", [True, False])
@pytest.mark.parametrize("mean", [True, False])
def test_plot_max(sample_linear_system: BlockLinearSystem, annot: bool, mean: bool):
    sample_linear_system.plot_max(annot=annot, mean=mean)


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("aspect", ["auto", "equal"])
def test_matshow(
    sample_linear_system: BlockLinearSystem, log: bool, show: bool, aspect
):
    sample_linear_system.matshow(log=log, show=show, threshold=1e-5, aspect=aspect)


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("show", [True, False])
def test_matshow_groups(sample_linear_system: BlockLinearSystem, log: bool, show: bool):
    sample_linear_system.matshow_groups(log=log, show=show)


@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("aspect", ["auto", "equal"])
@pytest.mark.parametrize("marker", [".", "+", None])
@pytest.mark.parametrize("color", [True, False])
@pytest.mark.parametrize("hatch", [True, False])
@pytest.mark.parametrize("draw_marker", [True, False])
def test_color_spy(
    sample_linear_system: BlockLinearSystem,
    show: bool,
    aspect,
    marker: str,
    color: bool,
    draw_marker: bool,
    hatch: bool,
):
    sample_linear_system.color_spy(
        show=show,
        aspect=aspect,
        marker=marker,
        color=color,
        hatch=hatch,
        draw_marker=draw_marker,
        alpha=0.4,
    )


@pytest.mark.parametrize("log", [True, False])
def test_plot_solution(sample_linear_system: BlockLinearSystem, log: bool):
    sample_linear_system.plot_solution(sample_linear_system.rhs, log=log)


@pytest.mark.parametrize("log", [True, False])
def test_plot_rhs(sample_linear_system: BlockLinearSystem, log: bool):
    sample_linear_system.plot_rhs(sample_linear_system.rhs, log=log)
