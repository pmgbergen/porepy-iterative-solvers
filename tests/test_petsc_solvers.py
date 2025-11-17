import numpy as np
import pytest
import scipy.sparse as sps
from petsc4py import PETSc
from scipy.sparse.linalg import spsolve

import pp_solvers
from pp_solvers import BlockLinearSystem
from pp_solvers.block_matrix import LinearSystemIndexer
from pp_solvers.petsc_solvers import PetscKrylovSolver, LinearSolverWithTransformations


@pytest.fixture
def sample_matrix() -> BlockLinearSystem:
    J00 = [
        [2, -1, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0],
        [0, -1, 2, -1, 0, 0],
        [0, 0, -1, 2, -1, 0],
        [0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, -1, 2],
    ]
    J00 = sps.csr_array(np.array(J00).astype(float))
    J = sps.block_array(
        [
            [J00, J00 * -0.5],
            [J00 * -0.5, J00],
        ]
    )
    dofs = [np.array(x) for x in [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]]
    return BlockLinearSystem(
        mat=J,
        rhs=np.arange(12, dtype=float) + 1,
        indexer=LinearSystemIndexer(
            dofs_row=dofs,
            dofs_col=dofs,
        )
    )


@pytest.fixture
def ksp(sample_matrix: BlockLinearSystem) -> PETSc.KSP:
    ksp = PETSc.KSP().create()
    pp_solvers.insert_petsc_options(
        {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-10,
            "pc_type": "ilu",
            "ksp_gmres_restart": 100,
            "ksp_max_it": 100,
        }
    )
    ksp.setFromOptions()

    petsc_mat = pp_solvers.csr_to_petsc(sample_matrix.mat)
    ksp.setOperators(petsc_mat, petsc_mat)

    yield ksp

    ksp.destroy()
    petsc_mat.destroy()


def test_petsc_krylov_solver(
    ksp: PETSc.KSP,
    sample_matrix: BlockLinearSystem,
):
    rhs = np.arange(12, dtype=float)
    solver = PetscKrylovSolver(ksp=ksp)
    result = solver.solve(rhs)

    expected = spsolve(sample_matrix.mat, rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("right", [True, False])
def test_linear_transformed_solver(
    ksp: PETSc.KSP,
    sample_matrix: BlockLinearSystem,
    left: bool,
    right: bool,
):
    # Generating some transformation matrices.
    Qleft = None
    Qright = None
    transformed_matrix = sample_matrix.copy()
    if left:
        Qleft = sample_matrix.copy()
        transformed_matrix.mat = Qleft.mat @ transformed_matrix.mat
    if right:
        Qright = sample_matrix.copy()
        transformed_matrix.mat = transformed_matrix.mat @ Qright.mat

    # Informing PETSc about the transformed matrix.
    petsc_mat = pp_solvers.csr_to_petsc(transformed_matrix.mat)
    ksp.setOperators(petsc_mat, petsc_mat)

    # Solving the transformed linear system.
    rhs = np.arange(12, dtype=float)
    solver = LinearSolverWithTransformations(
        inner=PetscKrylovSolver(ksp=ksp), Qleft=Qleft, Qright=Qright
    )
    result = solver.solve(rhs)

    # Should return the non-transformed rhs, no matter what transformations we did.
    expected = spsolve(sample_matrix.mat, rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    # Manual teardown.
    petsc_mat.destroy()
