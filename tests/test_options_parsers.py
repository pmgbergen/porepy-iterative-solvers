"""TODO"""

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from testing_utils import (
    MockDofManager,
    generate_reference_submatrices_3_groups,
    generate_reference_rhs_3_groups,
    generate_reference_dofs_3_groups,
    generate_reference_matrix_3_groups
)


# integration tests for all the default factories (not here, in preconditioners?)
from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.options_parsers import (
    LinearTransformedScheme,
    PetscKSPScheme,
    assemble_petsc_ksp_pc,
)
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    csr_to_petsc,
    insert_petsc_options,
    petsc_to_csr,
)
from pp_solvers.preconditioners import GMRES, Identity


@pytest.fixture
def block_linear_system() -> BlockLinearSystem:
    dofs_row, dofs_col = generate_reference_dofs_3_groups()
    return BlockLinearSystem(
        mat=generate_reference_matrix_3_groups(),
        rhs=generate_reference_rhs_3_groups(),
        indexer=LinearSystemIndexer(
            dofs_row=dofs_row,
            dofs_col=dofs_col,
        ),
    )


@pytest.fixture
def ksp(block_linear_system: BlockLinearSystem) -> PETSc.KSP:
    petsc_mat = csr_to_petsc(block_linear_system.mat)
    petsc_ksp = PETSc.KSP().create()
    petsc_ksp.setOperators(petsc_mat)
    yield petsc_ksp

    petsc_ksp.destroy()
    petsc_mat.destroy()


class MockPythonContext:
    def setUp(self, pc: PETSc.PC) -> None:
        pass

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        pass

reference_dofs_row, reference_dofs_col = generate_reference_dofs_3_groups()

@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "petsc_options": {"ksp_type": "bcgs", "pc_type": "sor"},
                "assembly_config": {},
            },
            id="monolithic petsc solver",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "pbjacobi",
                    "mat_block_size": 3,
                },
                "assembly_config": {},
            },
            id="matrix block size = 2",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "fgmres",
                    # Use a krylov solver as a preconditioner, see PETSc PCKSP.
                    "pc_type": "ksp",
                    "ksp_ksp_type": "gmres",
                    "ksp_pc_type": "jacobi",
                },
                "assembly_config": {},
            },
            id="fgmres with inner gmres",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                },
                "assembly_config": {},
            },
            id="hypre boomeramg",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "fieldsplit",
                    "fieldsplit_aaa_pc_type": "sor",
                    "fieldsplit_bbb_pc_type": "jacobi",
                    # Custom matrix block sizes within fieldsplit.
                    "fieldsplit_aaa_mat_block_size": 3,
                    "fieldsplit_bbb_mat_block_size": 2,
                },
                "assembly_config": {
                    "": {
                        "pc_type": "fieldsplit",
                        "elim_groups": [0],
                        "keep_groups": [1, 2],
                        "elim_tag": "aaa",
                        "keep_tag": "bbb",
                    }
                },
            },
            id="fieldsplit",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "fieldsplit",
                    "fieldsplit_aaa_pc_type": "fieldsplit",
                    "fieldsplit_aaa_fieldsplit_ccc_pc_type": "sor",
                    "fieldsplit_aaa_fieldsplit_ddd_pc_type": "pbjacobi",
                    "fieldsplit_bbb_pc_type": "jacobi",
                },
                "assembly_config": {
                    "": {
                        "pc_type": "fieldsplit",
                        "elim_groups": [0, 1],
                        "keep_groups": [2],
                        "elim_tag": "aaa",
                        "keep_tag": "bbb",
                    },
                    "fieldsplit_aaa_": {
                        "pc_type": "fieldsplit",
                        "elim_groups": [1],  # Intentionally switching order.
                        "keep_groups": [0],
                        "elim_tag": "ccc",
                        "keep_tag": "ddd",
                    },
                },
            },
            id="nested fieldsplit - elim",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "composite",
                    "sub_0_pc_type": "sor",
                    "sub_1_pc_type": "jacobi",
                    "sub_2_pc_type": "pbjacobi",
                },
                "assembly_config": {
                    "": {
                        "pc_type": "composite",
                        "num_stages": 3,
                    },
                },
            },
            id="nested fieldsplit - keep",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "fieldsplit",
                    "fieldsplit_aaa_pc_type": "sor",
                    "fieldsplit_bbb_pc_type": "jacobi",
                },
                "assembly_config": {
                    "": {
                        "pc_type": "fieldsplit",
                        "elim_groups": [0],
                        "keep_groups": [1, 2],
                        "elim_tag": "aaa",
                        "keep_tag": "bbb",
                    }
                },
            },
            id="composite",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "composite",
                    "mat_block_size": 3,  # custom block size for composite.
                    "sub_0_pc_type": "fieldsplit",
                    "sub_0_fieldsplit_elim_pc_type": "gamg",
                    "sub_0_fieldsplit_keep_pc_type": "none",
                    "sub_1_pc_type": "jacobi",
                },
                "assembly_config": {
                    "": {
                        "pc_type": "composite",
                        "num_stages": 2,
                    },
                    "sub_0_": {
                        "pc_type": "fieldsplit",
                        "elim_groups": [2, 0],  # Intentionally mixed order of groups.
                        "keep_groups": [1],
                        "elim_tag": "elim",
                        "keep_tag": "keep",
                    },
                },
            },
            id="cpr",  # This is not a complete setup of the CPR preconditioner.
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                },
                "assembly_config": {
                    "": {
                        "pc_type": "python",
                        "python_context": MockPythonContext(),
                    }
                },
            },
            id="python context",
        ),
        pytest.param(
            {
                "petsc_options": {
                    "ksp_type": "preonly",
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_precondition": "user",
                },
                "assembly_config": {
                    "": {
                        "pc_type": "fieldsplit",
                        "elim_groups": [2, 0],  # Intentionally mixed order of groups.
                        "keep_groups": [1],
                        "elim_tag": "elim",
                        "keep_tag": "keep",
                        # Constructing an invertor matrix, so S = A - C * D^-1 * B ≈ A.
                        "invertor_additive": lambda _: csr_to_petsc(
                            csr_matrix(
                                (
                                    len(reference_dofs_row[1]),
                                    len(reference_dofs_col[1]),
                                )
                            )
                        ),
                    }
                },
            },
            id="fieldsplit user invertor",
        ),
        # matrix block size
    ],
)
def test_assemble_petsc_ksp_pc(
    params: dict, ksp: PETSc.KSP, block_linear_system: BlockLinearSystem
):
    petsc_options: dict = params["petsc_options"]
    assembly_config: dict = params["assembly_config"]

    # Set up petsc KSP and PC objects.
    options = clear_petsc_options()
    insert_petsc_options(petsc_options)
    assemble_petsc_ksp_pc(
        ksp=ksp,
        pc=ksp.getPC(),
        assembly_config=assembly_config,
        indexer=block_linear_system.indexer,
    )

    # We do not check types of inner preconditioners, but petsc would not let a nested
    # preconditioner (e.g. fieldsplit) to set up, if there is an error in inner solvers.
    assert ksp.type == petsc_options["ksp_type"]
    assert ksp.getPC().type == petsc_options["pc_type"]

    # Check that all PETSc options are applied, which means that inner solvers
    # initialized correctly.
    for key in petsc_options:
        assert options.used(key)

    # Check that the block size initialized properly.
    expected_block_size = petsc_options.get("mat_block_size", 1)
    petsc_mat, petsc_pmat = ksp.getOperators()
    assert petsc_mat.getBlockSize() == expected_block_size
    assert petsc_pmat.getBlockSize() == expected_block_size

    # Solve reference linear system.
    petsc_rhs = petsc_mat.createVecLeft()
    petsc_rhs.setArray(block_linear_system.rhs)
    petsc_sol = petsc_mat.createVecRight()
    ksp.solve(petsc_rhs, petsc_sol)
    petsc_rhs.destroy()
    petsc_sol.destroy()

    # Positive reason means PETSc treats it as success.
    assert ksp.getConvergedReason() > 0


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("right", [True, False])
def test_linear_transformed_scheme(
    block_linear_system: BlockLinearSystem,
    left: bool,
    right: bool,
):
    # This also tests PetscKSPScheme.

    # Sorting the blocks in the matrix, same as it is done in the solver code.
    block_linear_system = block_linear_system[:]
    # Generating some transformation matrices.
    left_transformations = []
    right_transformations = []
    expected_matrix = block_linear_system.mat
    if left:
        Qleft = block_linear_system.copy()
        Qleft2 = block_linear_system.copy()
        Qleft2.mat *= 2
        left_transformations = [lambda _: Qleft, lambda _: Qleft2]
        expected_matrix = Qleft.mat @ Qleft2.mat @ expected_matrix
    if right:
        Qright = block_linear_system.copy()
        Qright2 = block_linear_system.copy()
        Qright2.mat *= 2
        right_transformations = [lambda _: Qright, lambda _: Qright2]
        expected_matrix = expected_matrix @ Qright.mat @ Qright2.mat
    # Initializing the KSP with transformations, without the preconditioner.
    solver_scheme = LinearTransformedScheme(
        inner=PetscKSPScheme(
            petsc_ksp_pc_configuration=GMRES(
                key="custom_key", preconditioner=Identity(groups=["mock_g1"])
            ),
            dof_manager=MockDofManager(),
        ),
        left_transformations=left_transformations,
        right_transformations=right_transformations,
    )
    solver = solver_scheme.make_solver(
        mat_orig=block_linear_system, options={"custom_key": {"ksp_type": "fgmres"}}
    )
    result_mat = petsc_to_csr(solver.ksp.getOperators()[0])
    # They should be exactly equal, numerical error may appear due to different order of
    # matrix multiplication.
    np.testing.assert_allclose(
        result_mat.toarray(), expected_matrix.toarray(), rtol=1e-20, atol=0
    )
    # Check that the custom option applied.
    assert solver.ksp.type == "fgmres"
    solution = solver.solve(block_linear_system.rhs)

    np.testing.assert_allclose(
        block_linear_system.mat @ solution,
        block_linear_system.rhs,
        rtol=0,
        atol=1e-10,
    )
