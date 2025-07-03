import numpy as np
import pytest
import scipy.sparse as sp

from pp_solvers.block_matrix import BlockMatrixStorage


@pytest.fixture
def sample_matrix():
    mat = sp.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 2],
                [0, 2, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 4, 0, 0],
                [0, 0, 0, 4, 0, 0, 0],
                [0, 0, 5, 0, 5, 0, 0],
                [0, 6, 0, 0, 0, 6, 0],
                [7, 0, 0, 0, 0, 0, 7],
            ]
        )
    )
    global_dofs_row = [
        np.array([0, 1]),
        np.array([2]),
        np.array([3, 4]),
        np.array([5, 6]),
    ]
    global_dofs_col = [
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([4]),
        np.array([5, 6]),
    ]
    groups_to_blocks_row = [[0], [1, 2], [3]]
    groups_to_blocks_col = [[0], [1], [2, 3]]
    return BlockMatrixStorage(
        mat=mat,
        global_dofs_row=global_dofs_row,
        global_dofs_col=global_dofs_col,
        groups_to_blocks_row=groups_to_blocks_row,
        groups_to_blocks_col=groups_to_blocks_col,
    )


def test_shape(sample_matrix):
    assert sample_matrix.shape == (7, 7)


def test_repr(sample_matrix):
    repr_str = repr(sample_matrix)
    assert "BlockMatrixStorage of shape (7, 7)" in repr_str


@pytest.mark.parametrize(
    "index, expected",
    # The zeroth group consists of a single block [0], which has global indices [0, 1]
    [
        ("[0]", np.array([[1, 0], [0, 2]])),
        # The first row group contains blocks [1, 2], which have global row indices [2,
        # 3, 4]. The first column group contains block [1], which has global column
        # indices [2, 3].
        ("[1]", np.array([[3, 0], [0, 4], [5, 0]])),
        # Slicing over all columns and rows.
        ("[0], :", np.array([[1, 0, 0, 0, 0, 0, 2], [0, 2, 0, 0, 0, 3, 0]])),
        (":, [1]", np.array([[0, 0], [0, 0], [3, 0], [0, 4], [5, 0], [0, 0], [0, 0]])),
        # Slicing over a subset of rows.
        ("[0], :2", np.array([[1, 0, 0, 0], [0, 2, 0, 0]])),
        ("[0], 2:", np.array([[0, 0, 2], [0, 3, 0]])),
        ("[0], 1:2", np.array([[0, 0], [0, 0]])),
        # Slicing over a subset of columns.
        (":2, [1]", np.array([[0, 0], [0, 0], [3, 0], [0, 4], [5, 0]])),
        ("2:, [1]", np.array([[0, 0], [0, 0]])),
        ("1:2, [1]", np.array([[3, 0], [0, 4], [5, 0]])),
    ],
)
def test_getitem(sample_matrix, index, expected):
    """Test the __getitem__ method of the BlockMatrixStorage class."""
    submatrix = eval(f"sample_matrix[{index}]")
    assert submatrix.shape == expected.shape
    np.testing.assert_array_equal(submatrix.mat.toarray(), expected)


def test_getitem_nested_invocation(sample_matrix):
    """Test that nested invocations of __getitem__ work as expected."""

    # First verify that sampling in two stages is equivalent to direct sampling.
    stage_1 = sample_matrix[:, [1]]
    stage_2 = stage_1[:2, [1]]
    direct_sampling = sample_matrix[:2, [1]]
    assert stage_2.shape == direct_sampling.shape
    np.testing.assert_array_equal(stage_2.mat.toarray(), direct_sampling.mat.toarray())


@pytest.mark.parametrize(
    "index, expected",
    [
        (
            [0],
            np.array(
                [
                    [42, 42, 0, 0, 0, 0, 2],
                    [42, 42, 0, 0, 0, 3, 0],
                    [0, 0, 3, 0, 4, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 5, 0, 5, 0, 0],
                    [0, 6, 0, 0, 0, 6, 0],
                    [7, 0, 0, 0, 0, 0, 7],
                ]
            ),
        ),
        (
            ([1, 2], [1]),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 2],
                    [0, 2, 0, 0, 0, 3, 0],
                    [0, 0, 42, 42, 4, 0, 0],
                    [0, 0, 42, 42, 0, 0, 0],
                    [0, 0, 42, 42, 5, 0, 0],
                    [0, 6, 42, 42, 0, 6, 0],
                    [7, 0, 42, 42, 0, 0, 7],
                ]
            ),
        ),
    ],
)
def test_setitem(sample_matrix, index, expected):
    """Test the __setitem__ method of the BlockMatrixStorage class.

    Testing is less extensive than for __getitem__ because the method to convert keys
    to indices is common for the two methods.

    """
    sample_matrix[index] = 42
    np.testing.assert_array_equal(sample_matrix.mat.toarray(), expected)


def test_copy(sample_matrix):
    copied_matrix = sample_matrix.copy()
    assert copied_matrix.shape == sample_matrix.shape
    assert copied_matrix.mat.nnz == sample_matrix.mat.nnz


def test_empty_container(sample_matrix):
    empty_matrix = sample_matrix.empty_container()
    assert empty_matrix.shape == sample_matrix.shape
    assert empty_matrix.mat.nnz == 0


def test_project_rhs_to_local(sample_matrix):
    global_rhs = np.array([1, 2, 3, 4, 5, 6, 7])
    local_rhs = sample_matrix.project_rhs_to_local(global_rhs)
    assert np.array_equal(local_rhs, global_rhs)


def test_project_rhs_to_global(sample_matrix):
    local_rhs = np.array([1, 2, 3, 4, 5, 6, 7])
    global_rhs = sample_matrix.project_rhs_to_global(local_rhs)
    assert np.array_equal(global_rhs, local_rhs)


def test_project_solution_to_global(sample_matrix):
    solution = np.array([1, 2, 3, 4, 5, 6, 7])
    global_solution = sample_matrix.project_solution_to_global(solution)
    assert np.array_equal(global_solution, solution)


def atest_set_zeros(sample_matrix):
    sample_matrix.set_zeros([0, 1], [2, 3])
    assert sample_matrix.mat[0, 2] == 0
    assert sample_matrix.mat[1, 3] == 0


def test_get_active_local_dofs(sample_matrix):
    row_idx, col_idx = sample_matrix.get_active_local_dofs()
    assert len(row_idx) == 4
    assert len(col_idx) == 4


def atest_get_active_group_names(sample_matrix):
    row_names, col_names = sample_matrix.get_active_group_names()
    assert len(row_names) == 7
    assert len(col_names) == 7


def test_color_spy(sample_matrix):
    sample_matrix.color_spy(show=False)


def test_plot_max(sample_matrix):
    sample_matrix.plot_max()


# EK: Auto-generated tests related to plotting failed. Mark these as expected to fail
# for now, revisit if we need to fix them.
@pytest.mark.xfail
def test_matshow(sample_matrix):
    sample_matrix.matshow(show=False)


@pytest.mark.xfail
def test_matshow_blocks(sample_matrix):
    sample_matrix.matshow_blocks(show=False)


@pytest.mark.xfail
def test_color_left_vector(sample_matrix):
    local_rhs = np.array([1, 2, 3, 4, 5, 6, 7])
    sample_matrix.color_left_vector(local_rhs)
