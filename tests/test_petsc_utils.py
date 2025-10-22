import numpy as np
import pytest
from pp_solvers import petsc_utils
from scipy.sparse import csr_matrix
from petsc4py import PETSc

@pytest.mark.parametrize('block_size', [1, 2, 3])
def test_csr_to_petsc_to_csr(block_size: int):
    n_rows = 6
    mat = np.arange(n_rows * n_rows).reshape(n_rows, n_rows).astype(float)
    # Adding some sparsity.
    mat[[1, 2, 2, 3, 4], [0, 1, 5, 2, 1]] = 0
    mat = csr_matrix(mat)

    petsc_mat = petsc_utils.csr_to_petsc(mat, bsize=block_size)
    assert petsc_mat.getSize() == mat.shape
    assert petsc_mat.getBlockSize() == block_size
    for i in range(n_rows):
        for j in range(n_rows):
            assert mat[i, j] == petsc_mat[i, j]

    result = petsc_utils.petsc_to_csr(petsc_mat)
    assert np.all(mat.toarray() == result.toarray())


def test_insert_clear_petsc_options():
    for i in range(5):
        options = PETSc.Options()
        petsc_utils.clear_petsc_options()
        assert options.getAll() == {}
        petsc_utils.insert_petsc_options({'aaa': f'{i}{i}{i}'})
        assert options.getAll() == {'aaa': f'{i}{i}{i}'}