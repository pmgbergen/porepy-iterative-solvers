"""This file contains various tests mainly to examine slicing in the BlockMatrixStorage
class.

"""

import numpy as np
import pytest
import scipy.sparse as sp

from pp_solvers.block_matrix import BlockMatrixStorage


# The submatrices of the tested matrix are defined globally to generate tests.
_J00 = [
    [5, 1, 1],
    [2, 5, 1],
    [2, 2, 5],
]
_J01 = [
    [1, 3, 0, 0],
    [2, 3, 1, 1],
    [1, 3, 0, 0],
]
_J02 = [
    [3, -1],
    [-2, 3],
    [-2, 2],
]
_J10 = [
    [-2, 0, 2],
    [2, 0, -2],
    [-2, 1, 2],
    [2, 1, -2],
]
_J11 = [
    [6, 1, 5, 1],
    [1, 6, 1, 5],
    [2, 1, 6, 2],
    [1, 2, 2, 6],
]
_J12 = [
    [1, 1],
    [0, 1],
    [1, 2],
    [0, 2],
]
_J20 = [
    [-3, 1, 1],
    [2, -3, 1],
]
_J21 = [
    [-1, -3, 5, 0],
    [-2, -3, 0, 5],
]
_J22 = [
    [-3, 1],
    [2, -3],
]


@pytest.fixture
def sample_matrix() -> BlockMatrixStorage:
    """The block matrix we use in all the tests. It contains one empty group."""
    return BlockMatrixStorage(
        mat=sp.block_array(
            [
                [_J22, _J21, _J20],
                [_J12, _J11, _J10],
                [_J02, _J01, _J00],
            ]
        )
        .astype(float)
        .tocsr(),
        global_dofs_row=[np.array(x) for x in [[6, 7, 8], [2, 3, 4, 5], [0, 1], []]],
        global_dofs_col=[np.array(x) for x in [[6, 7, 8], [2, 3, 4, 5], [0, 1], []]],
        groups_to_blocks_row=[[0], [1], [2], []],
        groups_to_blocks_col=[[0], [1], [2], []],
    )


@pytest.mark.parametrize(
    "params",
    [
        # Diagonal blocks are not square submatrices.
        {
            "submatrices": [[_J10, _J11], [_J00, _J01]],
            "global_dofs_row": [[0, 1, 2, 3], [4, 5, 6]],
            "global_dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "groups_to_blocks_row": [[0], [1]],
            "groups_to_blocks_col": [[0], [1]],
        },
        # Block indices are inconsistent - too few.
        {
            "submatrices": [[_J00, _J01], [_J10, _J11]],
            "global_dofs_row": [[0, 1, 2], [3, 4, 5]],
            "global_dofs_col": [[0, 1, 2], [3, 4, 5]],
            "groups_to_blocks_row": [[0], [1]],
            "groups_to_blocks_col": [[0], [1]],
        },
        # Block indices are inconsistent - too many.
        {
            "submatrices": [[_J00, _J01], [_J10, _J11]],
            "global_dofs_row": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "global_dofs_col": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "groups_to_blocks_row": [[0], [1]],
            "groups_to_blocks_col": [[0], [1]],
        },
        # Block indices are inconsistent - correct shape but garbage values.
        {
            "submatrices": [[_J00, _J01], [_J10, _J11]],
            "global_dofs_row": [[1, 7, 123, 3], [3, 55, 66, 87]],
            "global_dofs_col": [[643, 12, 42, 1], [312, 2, 32, 52]],
            "groups_to_blocks_row": [[0], [1]],
            "groups_to_blocks_col": [[0], [1]],
        },
        # Group indices are inconsistent - too few.
        {
            "submatrices": [[_J00, _J01], [_J10, _J11]],
            "global_dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "global_dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "groups_to_blocks_row": [[0]],
            "groups_to_blocks_col": [[0]],
        },
        # Group indices are inconsistent - too many.
        {
            "submatrices": [[_J00, _J01], [_J10, _J11]],
            "global_dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "global_dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "groups_to_blocks_row": [[0, 1, 2]],
            "groups_to_blocks_col": [[0, 1, 2]],
        },
        # Group indices are inconsistent - correct shape but garbage values.
        {
            "submatrices": [[_J00, _J01], [_J10, _J11]],
            "global_dofs_row": [[0, 1, 2], [3, 4, 5, 6]],
            "global_dofs_col": [[0, 1, 2], [3, 4, 5, 6]],
            "groups_to_blocks_row": [[0, 5]],
            "groups_to_blocks_col": [[0, 5]],
        },
    ],
)
def test_bad_block_matrix_creation(params):
    """This test examines invalid arguments passed to the block matrix constructor."""
    submatrices = params["submatrices"]
    global_dofs_row = params["global_dofs_row"]
    global_dofs_col = params["global_dofs_col"]
    groups_to_blocks_row = params["groups_to_blocks_row"]
    groups_to_blocks_col = params["groups_to_blocks_col"]
    with pytest.raises(Exception):
        return BlockMatrixStorage(
            mat=sp.block_array(submatrices).astype(float).tocsr(),
            global_dofs_row=[np.array(x) for x in global_dofs_row],
            global_dofs_col=[np.array(x) for x in global_dofs_col],
            groups_to_blocks_row=groups_to_blocks_row,
            groups_to_blocks_col=groups_to_blocks_col,
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
def test_shape_simple_index(sample_matrix: BlockMatrixStorage, params):
    """All the possible combinations of simple indexing, e.g. J[x]."""
    index: slice | int | list[int] = params["index"]
    raises: bool = params.get("raises", False)
    expected: tuple[int, int] | None = params.get("expected", None)
    if not raises:
        assert sample_matrix[index].shape == expected, index
    else:
        with pytest.raises(Exception):
            sample_matrix[index]


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
def test_shape_tuple_index(sample_matrix: BlockMatrixStorage, params_row, params_col):
    """All the possible combinations of tuple indexing, e.g. J[x, y]."""
    ind_row: slice | int | list[int] = params_row["index"]
    raises_row: bool = params_row.get("raises", False)
    expected_row: int | None = params_row.get("expected", None)
    ind_col: slice | int | list[int] = params_col["index"]
    raises_col: bool = params_col.get("raises", False)
    expected_col: int | None = params_col.get("expected", None)
    if not (raises_row or raises_col):
        assert sample_matrix[ind_row, ind_col].shape == (expected_row, expected_col)
    else:
        with pytest.raises(Exception):
            sample_matrix[ind_row, ind_col]


@pytest.mark.parametrize(
    "params",
    [
        # J[:] (all, sorted).
        {
            "index": slice(None, None, None),
            "expected": [
                [_J00, _J01, _J02],
                [_J10, _J11, _J12],
                [_J20, _J21, _J22],
            ],
        },
        # J[2, 0].
        {"index": (2, 0), "expected": [[_J20]]},
        # J[0, 0].
        {"index": (0, 0), "expected": [[_J00]]},
        # J[1, 1].
        {"index": (1, 1), "expected": [[_J11]]},
        # J[2, 2].
        {"index": (2, 2), "expected": [[_J22]]},
        # J[[2, 1, 0], 2].
        {"index": ([2, 1, 0], 2), "expected": [[_J22], [_J12], [_J02]]},
        # J[2, [1, 2, 0]].
        {"index": (2, [1, 2, 0]), "expected": [[_J21, _J22, _J20]]},
        # J[[2, 1], [2, 0]].
        {"index": ([2, 1], [2, 0]), "expected": [[_J22, _J20], [_J12, _J10]]},
        # J[[1, 1], [0, 2]] (repeated index).
        {"index": ([1, 1], [0, 2]), "expected": [[_J10, _J12], [_J10, _J12]]},
        # J[:, 1] (rows should be sorted, same as [0, 1, 2]).
        {"index": (slice(None, None, None), 1), "expected": [[_J01], [_J11], [_J21]]},
        # J[1, :] (columns should be sorted, same as [0, 1, 2]).
        {"index": (1, slice(None, None, None)), "expected": [[_J10, _J11, _J12]]},
    ],
)
def test_slicing(sample_matrix: BlockMatrixStorage, params):
    """These test ensures that the sliced matrices exactly match the expectation."""
    index = params["index"]
    expected = params["expected"]
    expected = [[sp.csr_array(submat) for submat in row] for row in expected]
    expected = sp.block_array(expected).astype(float).toarray()
    np.testing.assert_array_equal(sample_matrix[index].mat.toarray(), expected)


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
def test_nested_slicing(sample_matrix: BlockMatrixStorage, params):
    """Tests that the slice of a slice works as expected. For some cases, second
    indexing is expected to fail.

    """
    index1 = params["index1"]
    index2 = params["index2"]
    expected = params.get("expected", None)
    raises: bool = params.get("raises", False)

    tmp = sample_matrix[index1]
    if not raises:
        assert tmp[index2].shape == expected
    else:
        with pytest.raises(Exception):
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
            'index': (slice(None, None, None), 3),
            'modify_submatrices': [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        }
    ],
)
def test_setitem(sample_matrix: BlockMatrixStorage, params):
    """Tests that __setitem__ works as expected."""
    index = params["index"]
    modify_submatrices = params["modify_submatrices"]
    fill_value = 99

    # Constructing modified matrices
    original_matrix = [
        [_J00, _J01, _J02],
        [_J10, _J11, _J12],
        [_J20, _J21, _J22],
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

    sample_matrix[index] = fill_value
    np.testing.assert_array_equal(sample_matrix[:].mat.toarray(), expected)


def test_copy(sample_matrix: BlockMatrixStorage):
    copied_mat = sample_matrix.copy()
    assert copied_mat.shape == sample_matrix.shape
    np.testing.assert_array_equal(sample_matrix.mat.toarray(), copied_mat.mat.toarray())
    assert sample_matrix.mat.data is not copied_mat.mat.data


def test_empty_container(sample_matrix: BlockMatrixStorage):
    empty_matrix = sample_matrix.empty_container()
    assert empty_matrix.shape == sample_matrix.shape
    assert empty_matrix.mat.nnz == 0


@pytest.mark.parametrize(
    "params",
    [
        # Original matrix, nothing changes.
        {
            "index": [2, 1, 0],
            "expected_local": [30, 31, 20, 21, 22, 23, 10, 11, 12],
            "expected_global": [30, 31, 20, 21, 22, 23, 10, 11, 12],
        },
        # Full matrix, sorted groups.
        {
            "index": [0, 1, 2],
            "expected_local": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "expected_global": [30, 31, 20, 21, 22, 23, 10, 11, 12],
        },
        # One row and one column, same as J[1, 1].
        {
            "index": [1],
            "expected_local": [20, 21, 22, 23],
            "expected_global": [0, 0, 20, 21, 22, 23, 0, 0, 0],
        },
        # Repeated group index, same as J[[1, 1]].
        {
            "index": [1, 1],
            "expected_local": [20, 21, 22, 23, 20, 21, 22, 23],
            "expected_global": [0, 0, 20, 21, 22, 23, 0, 0, 0],
        },
        # All rows, one column, same as J[:, 1]. Rhs should NOT be truncated.
        {
            "index": ([0, 1, 2], 1),
            "expected_local": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "expected_global": [30, 31, 20, 21, 22, 23, 10, 11, 12],
        },
        # All columns, one row, same as J[1, :]. RHS should be truncated.
        {
            "index": (1, [0, 1, 2]),
            "expected_local": [20, 21, 22, 23],
            "expected_global": [0, 0, 20, 21, 22, 23, 0, 0, 0],
        },
        # Empty group.
        {
            'index': (3, [0, 1, 2,])
        }
    ],
)
def test_project_rhs_to_local_and_global(sample_matrix: BlockMatrixStorage, params):
    """Tests `project_rhs_to_local` and the reverse transformation from it with
    `project_rhs_to_global`.

    """
    # The global rhs is arranged for groups [2, 1, 0], same as the original matrix.
    rhs_global = np.array([30, 31, 20, 21, 22, 23, 10, 11, 12])
    index = params["index"]
    expected_local = params["expected_local"]
    expected_global = params["expected_global"]

    sample_matrix = sample_matrix[index]

    rhs_local = sample_matrix.project_rhs_to_local(rhs_global)
    np.testing.assert_array_equal(rhs_local, expected_local)

    rhs_back_to_global = sample_matrix.project_rhs_to_global(rhs_local)
    np.testing.assert_array_equal(rhs_back_to_global, expected_global)


@pytest.mark.parametrize(
    "params",
    [
        # Original matrix, nothing changes.
        {
            "index": [2, 1, 0],
            "solution_local": [30, 31, 20, 21, 22, 23, 10, 11, 12],
            "expected_global": [30, 31, 20, 21, 22, 23, 10, 11, 12],
        },
        # Full matrix, sorted groups.
        {
            "index": [0, 1, 2],
            "solution_local": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "expected_global": [30, 31, 20, 21, 22, 23, 10, 11, 12],
        },
        # One row and one column, same as J[1, 1].
        {
            "index": [1],
            "solution_local": [20, 21, 22, 23],
            "expected_global": [0, 0, 20, 21, 22, 23, 0, 0, 0],
        },
        # Repeated group index, same as J[[1, 1]].
        {
            "index": [1, 1],
            "solution_local": [20, 21, 22, 23, 20, 21, 22, 23],
            "expected_global": [0, 0, 20, 21, 22, 23, 0, 0, 0],
        },
        # All rows, one column, same as J[:, 1]. Soultion should be truncated.
        {
            "index": ([0, 1, 2], 1),
            "solution_local": [20, 21, 22, 23],
            "expected_global": [0, 0, 20, 21, 22, 23, 0, 0, 0],
        },
        # All columns, one row, same as J[1, :]. Soultion should NOT be truncated.
        {
            "index": (1, [0, 1, 2]),
            "solution_local": [10, 11, 12, 20, 21, 22, 23, 30, 31],
            "expected_global": [30, 31, 20, 21, 22, 23, 10, 11, 12],
        },
    ],
)
def test_project_solution_to_global(sample_matrix: BlockMatrixStorage, params):
    """Tests `project_solution_to_global`. Results should be different from
    `project_rhs_to_global`, because the solution corresponds to the space defined by
    columns of the matrix, and the rhs - to the one defined by rows (Ax = b).

    """
    index = params["index"]
    solution_local = params["solution_local"]
    expected_global = params["expected_global"]

    sample_matrix = sample_matrix[index]
    solution_local = np.array(solution_local)

    solution_back_to_global = sample_matrix.project_solution_to_global(solution_local)
    np.testing.assert_array_equal(solution_back_to_global, expected_global)


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
def test_set_zeros(sample_matrix: BlockMatrixStorage, params):
    index_row = params["index_row"]
    index_col = params["index_col"]
    modify_submatrices = params["modify_submatrices"]

    # Constructing modified matrices
    original_matrix = [
        [_J00, _J01, _J02],
        [_J10, _J11, _J12],
        [_J20, _J21, _J22],
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
    sample_matrix = sample_matrix[[1, 2, 0], [2, 0, 1]][[2, 1, 0]][[0, 2, 1]]

    sample_matrix.set_zeros(group_row_idx=index_row, group_col_idx=index_col)
    np.testing.assert_array_equal(sample_matrix[:].mat.toarray(), expected)


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
            'expected_diagonal_change': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'fill_groups': [3],
            'fill_values': []
        }
    ],
)
@pytest.mark.parametrize("additive", [False, True])
def test_set_diagonal(sample_matrix: BlockMatrixStorage, params, additive: bool):
    expected_diagonal_change = params["expected_diagonal_change"]
    fill_groups = params["fill_groups"]
    fill_values = params["fill_values"]

    # Generating the expected matrix.
    expected_mat = sample_matrix[:]
    if not additive:
        expected_mat[fill_groups] = (
            expected_mat[fill_groups].mat
            - sp.diags(expected_mat[fill_groups].mat.diagonal()).tocsr()
        )
    expected_mat.mat += sp.diags(np.array(expected_diagonal_change))

    sample_matrix.set_diagonal(
        groups=fill_groups, values=np.array(fill_values), additive=additive
    )
    np.testing.assert_array_equal(
        sample_matrix[:].mat.toarray(), expected_mat.mat.toarray()
    )
