from typing import cast

import numpy as np
import pytest
import scipy.sparse as sps
from petsc4py import PETSc
from scipy.sparse.linalg import spsolve
from testing_utils import MockDofManager, generate_reference_block_linear_system

import pp_solvers
from pp_solvers import BlockLinearSystem
from pp_solvers.block_linear_system import LinearSystemIndexer
from pp_solvers.options_parsers import initialize_petsc_ksp
from pp_solvers.petsc_solvers import PcPythonPermutation, PetscKrylovSolver
from pp_solvers.petsc_utils import petsc_to_csr
from pp_solvers.preconditioners import Identity, PythonPermutationWrapper


@pytest.fixture
def sample_linear_system() -> BlockLinearSystem:
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
        ),
    )


@pytest.fixture
def ksp(sample_linear_system: BlockLinearSystem) -> PETSc.KSP:
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

    petsc_mat = pp_solvers.csr_to_petsc(sample_linear_system.mat)
    ksp.setOperators(petsc_mat, petsc_mat)

    yield ksp

    ksp.destroy()
    petsc_mat.destroy()


def test_petsc_krylov_solver(
    ksp: PETSc.KSP,
    sample_linear_system: BlockLinearSystem,
):
    rhs = np.arange(12, dtype=float)
    solver = PetscKrylovSolver(ksp=ksp)
    result = solver.solve(rhs)

    expected = spsolve(sample_linear_system.mat, rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("num_dofs_per_group", [(4, 4), (4, 3)])
def test_python_permutation(num_dofs_per_group: tuple[int, int]):
    # We create a 2x2 block matrix.
    n, m = num_dofs_per_group
    A = generate_reference_block_linear_system(num_dofs_per_group=[n, m])

    # PythonPermutationWrapper accepts a matrix with same number of dofs in each group.
    petsc_ksp_pc_configuration = PythonPermutationWrapper(
        inner_subsolver=Identity(groups=["g1", "g2"], key="inner"),
        permutation_groups=[["g1"], ["g2"]],
    )

    # We construct a solver and save all the sub-matrices.
    petsc_matrices = {}
    try:
        solver = initialize_petsc_ksp(
            block_linear_system=A,
            dof_manager=MockDofManager(),
            petsc_ksp_pc_configuration=petsc_ksp_pc_configuration,
            user_options={
                "python_permutation": {"ksp_type": "gmres"},
                "delete_matrices": False,
            },
            petsc_matrices=petsc_matrices,
        )
    except ValueError:
        if n != m:
            # It only works if n == m. Otherwise, it should validly crashes here.
            return
        raise
    # Must have two keys: "python_permutation" (operates on unpermuted matrix) and
    # "inner" (operates on permuted matrix).
    assert "python_permutation" in petsc_matrices and "inner" in petsc_matrices
    assert solver.ksp.type == "gmres"  # Checking that the custom option applied.
    pc = solver.ksp.getPC()
    assert pc.type == "python"  # The outer pc is the python interface.
    python_context = cast(PcPythonPermutation, pc.getPythonContext())
    inner_pc = python_context.petsc_pc
    assert inner_pc.type == "none"  # The inner pc is the Identity pc.

    # Constructinge the expected permutation.
    expected_permutation = np.vstack([np.arange(n), np.arange(n, 2 * n)]).ravel("F")
    expected_permuted_mat = A.mat[expected_permutation, :][:, expected_permutation]
    # pmat and amat should be identical in this case.
    for petsc_mat_key in ["petsc_pmat", "petsc_amat"]:
        # "inner" should be permuted.
        actual_permuted_mat = petsc_to_csr(petsc_matrices["inner"][petsc_mat_key])
        np.testing.assert_allclose(
            actual_permuted_mat.toarray(), expected_permuted_mat.toarray(), rtol=1e-10
        )

        # "python_permutation" should be untouched.
        actual_original_mat = petsc_to_csr(
            petsc_matrices["python_permutation"][petsc_mat_key]
        )
        np.testing.assert_allclose(
            actual_original_mat.toarray(), A.mat.toarray(), rtol=1e-10
        )
