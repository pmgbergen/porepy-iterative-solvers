"""This module tests that the PETSc KSP and PCs are built correctly based on provided
PETSc options and configuration dictionaries.

"""

from typing import Optional

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from pp_solvers.mat_utils import inv_block_diag
from pp_solvers.preconditioners import (
    BlockDiagonalInverter,
    DiagonalInverter,
    FieldSplit,
    FieldSplitSchur,
    Identity,
    NoInverter,
    PetscInverter,
)
from pp_solvers.transformations import SchurComplementReduction
from testing_utils import (
    MockDofManager,
    generate_block_linear_system,
    generate_reference_dofs_3_groups,
)

# integration tests for all the default factories (not here, in preconditioners?)
from pp_solvers.block_linear_system import BlockLinearSystem
from pp_solvers.options_parsers import assemble_petsc_ksp_pc, initialize_petsc_ksp
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    csr_to_petsc,
    insert_petsc_options,
    petsc_to_csr,
)


@pytest.fixture
def block_linear_system() -> BlockLinearSystem:
    return generate_block_linear_system()


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
        # If broken: we failed to configure the most basice PETSc commands.
        pytest.param(
            {
                "petsc_options": {"root_ksp_type": "bcgs", "root_pc_type": "sor"},
                "assembly_config": {},
            },
            id="monolithic petsc solver",
        ),
        # If broken: we failed to configure the matrix block size. Did you call
        # mat.setFromOptions() ?
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "pbjacobi",
                    "root_mat_block_size": 3,
                },
                "assembly_config": {},
            },
            id="matrix block size = 3",
        ),
        # If broken, recursive setup of a nested ksp does not work. Did you call
        # ksp.setFromOptions and ksp.setUp on the inner ksp?
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "fgmres",
                    # Use a krylov solver as a preconditioner, see PETSc PCKSP.
                    "root_pc_type": "ksp",
                    "root_ksp_ksp_type": "gmres",
                    "root_ksp_pc_type": "jacobi",
                },
                "assembly_config": {},
            },
            id="fgmres with inner gmres",
        ),
        # If broken: Hypre is not installed?
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "hypre",
                    "root_pc_hypre_type": "boomeramg",
                },
                "assembly_config": {},
            },
            id="hypre boomeramg",
        ),
        # This is a basic fieldsplit test. If broken: check
        # _assemble_pc_fieldsplit_schur
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "fieldsplit",
                    # Each sub-solver is addressed by its key, not by a nested prefix.
                    "aaa_pc_type": "sor",
                    "bbb_pc_type": "jacobi",
                    # Custom matrix block sizes within fieldsplit.
                    "aaa_mat_block_size": 3,
                    "bbb_mat_block_size": 2,
                },
                "assembly_config": {
                    "root": {
                        "config_type": "fieldsplit_schur",
                        "elim_groups": [0],
                        "keep_groups": [1, 2],
                        "elim_key": "aaa",
                        "keep_key": "bbb",
                    }
                },
            },
            id="fieldsplit",
        ),
        # This is fieldsplit test, where the subsolver of a group to be eliminated is
        # also an inner fieldsplit (not typical use case, covered for completeness).
        # If broken: do we recursively configure the inner fieldsplit?
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "fieldsplit",
                    "aaa_pc_type": "fieldsplit",
                    "ccc_pc_type": "sor",
                    "ddd_pc_type": "pbjacobi",
                    "bbb_pc_type": "jacobi",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "fieldsplit_schur",
                        "elim_groups": [0, 1],
                        "keep_groups": [2],
                        "elim_key": "aaa",
                        "keep_key": "bbb",
                    },
                    "aaa": {
                        "config_type": "fieldsplit_schur",
                        "elim_groups": [1],  # Intentionally switching order.
                        "keep_groups": [0],
                        "elim_key": "ccc",
                        "keep_key": "ddd",
                    },
                },
            },
            id="nested fieldsplit - elim",
        ),
        # This is fieldsplit test, where the subsolver of Schur complement group is
        # also an inner fieldsplit (typical use case for multiphysics). If broken: do we
        # recursively configure the inner fieldsplit?
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "fieldsplit",
                    "aaa_pc_type": "jacobi",
                    "bbb_pc_type": "fieldsplit",
                    "ccc_pc_type": "sor",
                    "ddd_pc_type": "pbjacobi",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "fieldsplit_schur",
                        "elim_groups": [2],
                        "keep_groups": [0, 1],
                        "elim_key": "aaa",
                        "keep_key": "bbb",
                    },
                    "bbb": {
                        "config_type": "fieldsplit_schur",
                        "elim_groups": [1],  # Intentionally switching order.
                        "keep_groups": [0],
                        "elim_key": "ccc",
                        "keep_key": "ddd",
                    },
                },
            },
            id="nested fieldsplit - keep",
        ),
        # This is a basic test for PCComposite with 3 stages. If broken: check
        # _assemble_pc_composite
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "composite",
                    "stage0_pc_type": "sor",
                    "stage1_pc_type": "jacobi",
                    "stage2_pc_type": "pbjacobi",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "composite",
                        "subsolver_keys": ["stage0", "stage1", "stage2"],
                    },
                },
            },
            id="composite",
        ),
        # This test mimics the structure of the CPR preconditioner: The composite
        # preconditioner, where one of the stages is a fieldsplit. If broken, something
        # fails in the communication between the composite and the fieldsplit parts.
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "composite",
                    "root_mat_block_size": 3,  # custom block size for composite.
                    "stage0_pc_type": "fieldsplit",
                    "elim_pc_type": "gamg",
                    "keep_pc_type": "none",
                    "stage1_pc_type": "jacobi",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "composite",
                        "subsolver_keys": ["stage0", "stage1"],
                    },
                    "stage0": {
                        "config_type": "fieldsplit_schur",
                        "elim_groups": [2, 0],  # Intentionally mixed order of groups.
                        "keep_groups": [1],
                        "elim_key": "elim",
                        "keep_key": "keep",
                    },
                },
            },
            id="cpr",  # This is not a complete setup of the CPR preconditioner.
        ),
        # This test covers the python callback from petsc, which we use to transform the
        # underlying matrix. If broken, check _assemble_pc_python_permutation
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "python",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "python_permutation",
                        "permutation_groups": [[2, 0, 1]],
                        "inner_key": "inner",
                    }
                },
            },
            id="python permutation",
        ),
        # This test covers a user-defined invertor for the Schur complement fieldsplit,
        # such as the fixed-stress approximation.
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "fieldsplit",
                    "root_pc_fieldsplit_type": "schur",
                    "root_pc_fieldsplit_schur_precondition": "user",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "fieldsplit_schur",
                        # The order here matters since PETSc fieldsplit Schur requires
                        # ordered dofs.
                        "elim_groups": [0, 2],
                        "keep_groups": [1],
                        "elim_key": "elim",
                        "keep_key": "keep",
                        # Constructing an inverter matrix, so S = A - C * D^-1 * B ≈ A.
                        "inverter_additive": lambda _: csr_to_petsc(
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
            id="fieldsplit user inverter",
        ),
        # This test covers the non-Schur complement fieldsplit. If broken, check
        # _assemble_pc_fieldsplit_additive
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "fieldsplit",
                    "root_pc_fieldsplit_type": "additive",
                    "s0_pc_type": "sor",
                    "s1_ksp_type": "bcgs",
                    "s2_pc_type": "jacobi",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "fieldsplit_common",
                        "subsolver_groups": [[2], [0], [1]],
                        "subsolver_keys": ["s0", "s1", "s2"],
                    }
                },
            },
            id="fieldsplit additive",
        ),
        # This test covers the nested additive fieldsplits. If broken, do we initialize
        # the inner fieldsplit?
        pytest.param(
            {
                "petsc_options": {
                    "root_ksp_type": "preonly",
                    "root_pc_type": "fieldsplit",
                    "root_pc_fieldsplit_type": "multiplicative",
                    "s0_pc_type": "fieldsplit",
                    "s0_pc_fieldsplit_type": "symmetric_multiplicative",
                    "s01_pc_type": "sor",
                    "s1_ksp_type": "bcgs",
                },
                "assembly_config": {
                    "root": {
                        "config_type": "fieldsplit_common",
                        "subsolver_groups": [[2, 0], [1]],
                        "subsolver_keys": ["s0", "s1"],
                    },
                    "s0": {
                        "config_type": "fieldsplit_common",
                        "subsolver_groups": [[2], [0]],
                        "subsolver_keys": ["s00", "s01"],
                    },
                },
            },
            id="nested fieldsplit common",
        ),
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
        key="root",
    )

    # We do not check types of inner preconditioners, but petsc would not let a nested
    # preconditioner (e.g. fieldsplit) to set up, if there is an error in inner solvers.
    assert ksp.type == petsc_options["root_ksp_type"]
    assert ksp.getPC().type == petsc_options["root_pc_type"]

    # Check that all PETSc options are applied, which means that inner solvers
    # initialized correctly.
    for key in petsc_options:
        assert options.used(key)

    # Check that the block size initialized properly.
    expected_block_size = petsc_options.get("root_mat_block_size", 1)
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


@pytest.mark.parametrize(
    "params",
    [
        # Block size deduced from model.nd = 3. G1 has 3 rows, so it inverts exactly.
        {
            "invertor": BlockDiagonalInverter(),
            "block_size": 3,
            "groups_elim": ["g1"],
            "groups_keep": ["g2", "g3"],
        },
        # Custom block size.
        {
            "invertor": BlockDiagonalInverter(block_size=1),
            "block_size": 1,
            "groups_elim": ["g1"],
            "groups_keep": ["g2", "g3"],
        },
        # Different groups. G2 and g3 have 6 rows, so it inverts approximately.
        {
            "invertor": BlockDiagonalInverter(block_size=2),
            "block_size": 2,
            "groups_elim": ["g2", "g3"],
            "groups_keep": ["g1"],
        },
        # Diagonal invertor.
        {
            "invertor": DiagonalInverter(),
            "block_size": 1,
            "groups_elim": ["g2", "g3"],
            "groups_keep": ["g1"],
        },
    ],
)
def test_petsc_invertors(params: dict):
    invertor: PetscInverter = params["invertor"]
    block_size: int = params["block_size"]
    groups_elim: list[str] = params["groups_elim"]
    groups_keep: list[str] = params["groups_keep"]

    # The block matrix consists of 3 groups: g1, g2, g3.
    A = generate_block_linear_system()[:3]
    dof_manager = MockDofManager(
        groups=sorted(groups_elim + groups_keep), block_linear_system=A
    )
    assert dof_manager.model.nd == 3, "The test assumes a 3D model."

    # Our petsc configuration is a field split with a block-diagonal invertor.
    schur_complement_key = "keep"
    petsc_ksp_pc_configuration = FieldSplitSchur(
        subsolver=Identity(groups=groups_elim, key="elim"),
        complement_solver=Identity(groups=groups_keep, key=schur_complement_key),
        approximate_inverter=invertor,
    )

    # The petsc matrices will be saved here.
    petsc_matrices = {}
    # Initializing the solver and saving petsc matrices.
    _ = initialize_petsc_ksp(
        block_linear_system=A,
        dof_manager=dof_manager,
        petsc_ksp_pc_configuration=petsc_ksp_pc_configuration,
        user_options={"delete_matrices": False},
        petsc_matrices=petsc_matrices,
    )

    # Doing the same procedure from Python to get the expected result.
    reduction = SchurComplementReduction(
        primary_groups=groups_keep,
        secondary_groups=groups_elim,
        invertor=lambda mat: inv_block_diag(mat, nd=block_size),
    )
    S = reduction.transform_matrix_rhs(A, dof_manager)

    expected_mat = S.mat
    # PETSc stores two matrices - one for ksp (amat) and one for pc (pmat). Only pmat is
    # assembled with the block diagonal approximation applied.
    result_pmat = petsc_to_csr(petsc_matrices[schur_complement_key]["petsc_pmat"])
    np.testing.assert_allclose((expected_mat - result_pmat).data, 0, atol=1e-14)
    # Amat is not assembled.
    assert petsc_matrices[schur_complement_key]["petsc_amat"].type == "schurcomplement"


def test_petsc_no_invertor():
    groups_elim = ["g1", "g2"]
    groups_keep = ["g3"]

    # The block matrix consists of 3 groups: g1, g2, g3.
    A = generate_block_linear_system()[:3]
    dof_manager = MockDofManager(
        groups=groups_elim + groups_keep, block_linear_system=A
    )
    assert dof_manager.model.nd == 3, "The test assumes a 3D model."

    # Our petsc configuration is a field split with a block-diagonal invertor.
    schur_complement_key = "keep"
    petsc_ksp_pc_configuration = FieldSplitSchur(
        subsolver=Identity(groups=groups_elim, key="elim"),
        complement_solver=Identity(groups=groups_keep, key=schur_complement_key),
        approximate_inverter=NoInverter(),
    )

    # The petsc matrices will be saved here.
    petsc_matrices = {}
    # Initializing the solver and saving petsc matrices.
    _ = initialize_petsc_ksp(
        block_linear_system=A,
        dof_manager=dof_manager,
        petsc_ksp_pc_configuration=petsc_ksp_pc_configuration,
        user_options={"delete_matrices": False},
        petsc_matrices=petsc_matrices,
    )

    keep_idx = dof_manager.indices_of_groups(groups_keep)
    expected_mat = A[keep_idx].mat
    # PETSc stores two matrices - one for ksp (amat) and one for pc (pmat). Only pmat is
    # assembled with the block diagonal approximation applied.
    result_pmat = petsc_to_csr(petsc_matrices[schur_complement_key]["petsc_pmat"])
    np.testing.assert_allclose((expected_mat - result_pmat).data, 0, atol=1e-14)
    # Amat is not assembled.
    assert petsc_matrices[schur_complement_key]["petsc_amat"].type == "schurcomplement"


@pytest.mark.parametrize(
    "fieldsplit_type", ["additive", "multiplicative", "symmetric_multiplicative"]
)
def test_fieldsplit_common_type(capfd, fieldsplit_type: str):
    # capfd captures the PETSc.PC.view() stdout and stderr.
    configuration = FieldSplit(
        subsolvers=[
            Identity(groups=["g1"], key="i1"),
            Identity(groups=["g2"], key="i2"),
            Identity(groups=["g3"], key="i3"),
        ],
        fieldsplit_type=fieldsplit_type,
    )
    A = generate_block_linear_system(num_dofs_per_group=[2, 3, 4])
    dof_manager = MockDofManager(groups=["g1", "g2", "g3"], block_linear_system=A)
    solver = initialize_petsc_ksp(
        block_linear_system=A,
        dof_manager=dof_manager,
        petsc_ksp_pc_configuration=configuration,
        user_options={},
    )
    pc = solver.ksp.getPC()
    assert pc.type == "fieldsplit"

    pc.view()
    stdout, _ = capfd.readouterr()
    assert (
        f"fieldsplit with {fieldsplit_type} composition: total splits = 3"
        in stdout.lower()
    )
