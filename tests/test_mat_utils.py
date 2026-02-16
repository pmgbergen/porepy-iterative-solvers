import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix

from pp_solvers import mat_utils


@pytest.mark.parametrize("n", [6, 60])
def test_csr_ones(n):
    mat = mat_utils.csr_ones(n)
    assert isinstance(mat, csr_matrix)  # Will fail when it will become csr_array.
    assert mat.format == "csr"
    np.all(mat.toarray() == np.eye(n))


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_inv_block_diag(block_size):
    blocks = [
        np.eye(block_size) * (block_size + 1.0)
        + 0.1 * np.ones((block_size, block_size))
        + i
        for i in range(5)
    ]
    mat = block_diag(blocks, format="csr")
    expected = np.linalg.inv(mat.toarray())
    mat_inv = mat_utils.inv_block_diag(mat, nd=block_size).toarray()
    np.testing.assert_array_almost_equal(expected, mat_inv)
