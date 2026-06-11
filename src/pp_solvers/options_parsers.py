"""This module defines the machinery to parse the configuration of the PETSc linear
solver and build the corresponding PETSc KSP and PC objects."""

import logging
from typing import Optional

import numpy as np
from petsc4py import PETSc

from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.dof_manager import DofManager
from pp_solvers.petsc_solvers import PcPythonPermutation, PetscKrylovSolver
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    construct_is,
    csr_to_petsc,
    insert_petsc_options,
)
from pp_solvers.preconditioners import (
    PETSC_OPTIONS_MAX_SYMBOLS,
    PetscKspPcConfiguration,
)

logger = logging.getLogger(__name__)


def initialize_petsc_ksp(
    block_linear_system: BlockLinearSystem,
    dof_manager: DofManager,
    petsc_ksp_pc_configuration: PetscKspPcConfiguration,
    user_options: dict,
    petsc_matrices: Optional[dict] = None,
):
    """Initialize a PETSc KSP solver from a block linear system and solver config.

    Converts the system matrix to PETSc format, inserts PETSc CLI options derived from
    the configuration, assembles the KSP/PC hierarchy. Warns about any CLI options that
    PETSc did not consume.

    Args:
        block_linear_system: The assembled block linear system containing the matrix and
            index structure.
        dof_manager: Degree-of-freedom manager used to resolve group-to-DOF mappings.
        petsc_ksp_pc_configuration: Solver/preconditioner configuration that produces
            PETSc options and the assembly config.
        user_options: Runtime overrides forwarded to the configuration.
        petsc_matrices: If provided, populated with references to the PETSc Amat/Pmat
            for each sub-solver key (useful for debugging and testing).

    Returns:
        A `PetscKrylovSolver` wrapping the assembled PETSc KSP.
    """
    # We validated that all the solver keys are unique in SolverMixin.

    # Construct a PETSc matrix from the scipy matrix.
    petsc_mat = csr_to_petsc(block_linear_system.mat)
    if user_options.get("delete_matrices", True):
        del block_linear_system.mat  # Delete the scipy matrix to save memory.

    # Clear the PETSc options from a previous solve.
    petsc_options = clear_petsc_options()

    # Produce a flat list of PETSc CLI options
    all_options_dict = petsc_ksp_pc_configuration.petsc_options(
        user_options=user_options, dof_manager=dof_manager
    )
    # Produce Python-specific instructions for solver assembly.
    assembly_config = petsc_ksp_pc_configuration.petsc_assembly_config(
        user_options=user_options, dof_manager=dof_manager
    )

    insert_petsc_options(all_options_dict)

    petsc_ksp = PETSc.KSP().create()

    petsc_ksp.setOperators(petsc_mat)
    assemble_petsc_ksp_pc(
        ksp=petsc_ksp,
        pc=petsc_ksp.getPC(),
        assembly_config=assembly_config,
        indexer=block_linear_system.indexer,
        key=petsc_ksp_pc_configuration.key,
        petsc_matrices=petsc_matrices,
    )

    # Ensure that all PETSc CLI options are acknowledged.
    for key in all_options_dict:
        if not petsc_options.used(key):
            logger.warning(
                f"PETSc option {key}: {all_options_dict[key]} is not used. "
                "Check spelling."
            )

    return PetscKrylovSolver(
        petsc_ksp,
        assembly_config=assembly_config,
        petsc_options=all_options_dict,
    )


def _assemble_pc_fieldsplit_additive(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    key: str,
    petsc_matrices: Optional[dict] = None,
):
    """See the docstring of `assemble_petsc_ksp_pc`."""
    assert pc.type == "fieldsplit"

    prefix_config = assembly_config[key]
    subsolver_groups = prefix_config["subsolver_groups"]
    subsolver_keys = prefix_config["subsolver_keys"]

    for subsolver_key, groups in zip(subsolver_keys, subsolver_groups):
        is_subsolver = construct_is(indexer, groups)
        pc.setFieldSplitIS((subsolver_key, is_subsolver))

    try:
        pc.setUp()
        ksp.setUp()
    except:
        logger.error(f"failed on {key = }")
        raise

    sub_ksp_list = pc.getFieldSplitSubKSP()
    for sub_ksp, groups, subsolver_key in zip(
        sub_ksp_list, subsolver_groups, subsolver_keys
    ):
        assemble_petsc_ksp_pc(
            ksp=sub_ksp,
            pc=sub_ksp.getPC(),
            assembly_config=assembly_config,
            indexer=indexer[groups],
            key=subsolver_key,
            petsc_matrices=petsc_matrices,
        )


def _assemble_pc_fieldsplit_schur(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    key: str,
    petsc_matrices: Optional[dict] = None,
):
    """See the docstring of `assemble_petsc_ksp_pc`."""
    # calls: pc.setUp, ksp.setUp
    assert pc.type == "fieldsplit"

    prefix_config = assembly_config[key]
    elim_groups = prefix_config["elim_groups"]
    keep_groups = prefix_config["keep_groups"]
    elim_key = prefix_config["elim_key"]
    keep_key = prefix_config["keep_key"]

    is_elim = construct_is(indexer, elim_groups)
    is_keep = construct_is(indexer, keep_groups)

    keep_groups_indexer = indexer[keep_groups]

    # We initialize two splitting groups. PETSc gives each group a temporary prefix
    # e.g., {parent_prefix}_fieldsplit_{elim_key}. The right prefix will be set later.
    # Read a detailed explanation of this hack in the `BlockDiagonalInverter` class.
    pc.setFieldSplitIS((elim_key, is_elim), (keep_key, is_keep))

    # For a matrix [[A, B], [C, D]], Schur complement S = D - B * A^-1 * C, here D
    # corresponds to the index set "is_keep". An additive inverter is a matrix X to
    # build the approximat: S = D + X. This is where the fixed-stress approximation for
    # hydromechanics is applied.
    inverter = prefix_config.get("inverter_additive", None)
    if inverter is not None:
        # This copies the submatrix D into S.
        S = pc.getOperators()[1].createSubMatrix(is_keep, is_keep)
        # Extracts the matrix X in petsc format.
        petsc_matrix_inverter = inverter(keep_groups_indexer)
        # S = S + 1 * X
        S.axpy(1, petsc_matrix_inverter)
        # Passing the operator S as a user-defined Schur complement approximation to the
        # preconditioner.
        pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)
        # Destroying a temporary matrix used to construct S.
        petsc_matrix_inverter.destroy()

    try:
        pc.setUp()
        ksp.setUp()
    except:
        logger.error(f"failed on {key = }")
        raise

    sub_ksp_list = pc.getFieldSplitSubKSP()
    if len(sub_ksp_list) != 2:
        raise NotImplementedError
    ksp_elim, ksp_keep = sub_ksp_list

    pc_elim = ksp_elim.getPC()
    assemble_petsc_ksp_pc(
        ksp=ksp_elim,
        pc=pc_elim,
        assembly_config=assembly_config,
        indexer=indexer[elim_groups],
        key=elim_key,
        petsc_matrices=petsc_matrices,
    )

    pc_keep = ksp_keep.getPC()
    assemble_petsc_ksp_pc(
        ksp=ksp_keep,
        pc=pc_keep,
        assembly_config=assembly_config,
        indexer=indexer[keep_groups],
        key=keep_key,
        petsc_matrices=petsc_matrices,
    )


def _assemble_pc_composite(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    key: str,
    petsc_matrices: Optional[dict] = None,
):
    """See the docstring of `assemble_petsc_ksp_pc`."""
    assert pc.type == "composite"
    stage_keys = assembly_config[key]["subsolver_keys"]

    for i, stage_key in enumerate(stage_keys):
        # We need to access each sub-preconditioner. We need to create them using
        # pc.addCompositePCType(type). For each sub-preconditioner, we use the type
        # PCKSP. This creates a structure composite->pc->ksp->pc. The default ksp type
        # is KSPPREONLY, so this is numerically identical to just composite->pc.
        # However, it gives a flexibility to reinforce the CPR parts with GMRES, which
        # is useful for debugging.
        pc.addCompositePCType("ksp")
        # Access the newly created sub-preconditioner. PETSc assigns it a temporary
        # prefix: {parent_prefix}_sub_{i}. The right prefix will be set later.
        child_pc = pc.getCompositePC(i)
        # Each sub-pc of a composite preconditioner works with the same Amat and Pmat.
        child_pc.setOperators(*pc.getOperators())

        sub_ksp = child_pc.getKSP()
        # This may cause problems. We pass the same matrix to the sub-ksp. Below, in
        # assemble_petsc_ksp_pc we change its prefix. It is unclear for YZ whether it
        # can cause problems, but we need to keep it in mind.
        sub_ksp.setOperators(*pc.getOperators())
        sub_pc = sub_ksp.getPC()
        # The actual type of each sub_pc will be fetched here from PETSc options.
        assemble_petsc_ksp_pc(
            ksp=sub_ksp,
            pc=sub_pc,
            assembly_config=assembly_config,
            indexer=indexer,
            key=stage_key,
            petsc_matrices=petsc_matrices,
        )

    try:
        ksp.setUp()
        pc.setUp()
    except:
        logger.error(f"Failed on {key = }")
        raise


def _assemble_pc_python_permutation(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    key: str,
    petsc_matrices: Optional[dict] = None,
):
    """See the docstring of `assemble_petsc_ksp_pc`."""
    config = assembly_config[key]
    permutation_groups: list[list[int]] = config["permutation_groups"]
    inner_key: str = config["inner_key"]

    perm = [indexer.get_dofs_of_groups(g)[0] for g in permutation_groups]
    if np.unique([len(x) for x in perm]).size != 1:
        raise ValueError(
            "PcPythonPermutation accepts groups with equal number of dofs."
        )
    perm = np.vstack(perm).ravel("F")

    python_context = PcPythonPermutation(
        perm=perm, block_size=len(permutation_groups), inner_key=inner_key
    )
    python_context.setFromOptions(pc=pc)

    # This is duplicated here since we don't call assemble_petsc_ksp_pc (see below).
    # This should be removed if the lines below are ever uncommented.
    if petsc_matrices is not None:
        petsc_amat, petsc_pmat = python_context.petsc_pc.getOperators()
        petsc_matrices[inner_key] = {
            "petsc_pmat": petsc_pmat,
            "petsc_amat": petsc_amat,
        }
    # YZ: Nested initialization of python_context.petsc_pc can be here. However, we
    # don't use it now, so I don't cover it with tests and thus not implement it here.
    # assemble_petsc_ksp_pc(
    #     ksp=ksp,
    #     pc=python_context.petsc_pc,
    #     assembly_config=additional_data,
    #     indexer=indexer,
    #     prefix=f"{prefix}_python_",
    # )
    # Another NotImplementedError for this case is raised in PythonPermutationWrapper.
    if python_context.petsc_pc.type in ["fieldsplit", "composite", "python"]:
        raise NotImplementedError(
            "Nested initialization inside PythonPermutationWrapper is not implemented."
        )

    pc.setPythonContext(python_context)
    try:
        ksp.setUp()
        pc.setUp()
    except:
        logger.error(f"Failed on {key = }")
        raise


def assemble_petsc_ksp_pc(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    key: str,
    petsc_matrices: Optional[dict] = None,
) -> None:
    """This is a recursive parser that initializes the PETSc KSP and PC objects based on
    the provided assembly config. The assembly config contains sub-dictionaries, each
    corresponding to a certain PETSc prefix. The empty prefix corresponds to the root
    KSP and PC objects.

    This method **does not** insert command-line options into PETSc.Options(), it
    assumes that it is done beforehand. Method calls `.setFromOptions()` to initialize
    each sub-solver.

    Each sub-dictionary contains a required field "config_type", which determines how to
    parse the rest of the sub-dictionary. Example sub-dictionaries, which list all the
    available keys (using example values) are (be aware that for standard usage, the
    user will not need to list block numbers for the solver groups, as is done below):

    {
        "config_type": "fieldsplit_schur",
        "elim_groups": [0, 1],  # groups to eliminate to build the Schur complement.
        "keep_groups": [2, 3],  # groups to keep to build the Schur complement.
        "elim_key": "tag1",  # key identifying the eliminated sub-solver.
        "keep_key": "tag2",  # key identifying the kept (Schur complement) sub-solver.
    }
    {
        "config_type": "fieldsplit_common",
        "subsolver_groups": [[0, 1], [2, 3], [5, 6]],
        "subsolver_keys": ["key1", "key2", "key3"],  # one key per group entry.
    }
    {
        "config_type": "composite",
        "subsolver_keys": ["key1", "key2", "key3"],  # one key per stage.
    }
    {
        "config_type": "python_permutation",
        "permutation_groups": [[1, 2], [3, 4]],  # Groups to permute.
        "inner_key": "inner",  # key for the inner PC that operates on the permuted mat.
    }

    Parameters:
        ksp: The PETSc KSP object to configure.
        pc: The PETSc PC object to configure.
        assembly_config: Nested dict mapping PETSc prefix keys to sub-solver configs.
        indexer: Maps equation/variable group names to DOF index ranges.
        key: The PETSc prefix key identifying this KSP/PC pair within the config.
        petsc_matrices: Pass an empty dictionary here, and the function will store PETSc
            matrices for each sub-solver key. Used for testing and debugging.

    """
    prefix = f"{key}_"
    if len(prefix) > PETSC_OPTIONS_MAX_SYMBOLS:
        # PETSc has a limit on the prefix length, which seems to be 127
        # characters. If the prefix is too long, we raise a warning.
        msg = "The prefix for the PETSc preconditioner is too long. "
        msg += "Check the configuration of the preconditioner."
        logger.warning(msg)

    # This is where the ksp and pc objects fetch options in PETSc command-line format.
    ksp.setOptionsPrefix(prefix)
    ksp.setFromOptions()
    pc.setOptionsPrefix(prefix)
    pc.setFromOptions()

    # Accessing config for the current key.
    current_config: dict = assembly_config.get(key, {})

    petsc_amat, petsc_pmat = ksp.getOperators()
    # The command-line options for a matrix include mat_block_size (integer) and
    # mat_type including "aij" or "baij", corresponding to csr and bsr sparse formats,
    # respectively. Matrices share the prefix of the ksp and the pc.
    petsc_amat.setOptionsPrefix(prefix)
    petsc_amat.setFromOptions()
    petsc_pmat.setOptionsPrefix(prefix)
    petsc_pmat.setFromOptions()

    # Sanity check that ksp and pc point to the same matrix. If not, it could be that
    # you messed with the prefixes and calling .setFromOptions deleted an old matrix and
    # created a new empty one.
    pc_petsc_amat, pc_petsc_pmat = pc.getOperators()
    assert (
        pc_petsc_amat.prefix
        == petsc_amat.prefix
        == petsc_pmat.prefix
        == pc_petsc_pmat.prefix
    )

    # Store PETSc matriecs for debugging or testing.
    if petsc_matrices is not None:
        petsc_matrices[key] = {
            "petsc_pmat": pc_petsc_pmat,
            "petsc_amat": pc_petsc_amat,
        }

    config_type: str = current_config.get("config_type", "default")

    if config_type == "fieldsplit_schur":
        _assemble_pc_fieldsplit_schur(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            key=key,
            petsc_matrices=petsc_matrices,
        )
    elif config_type == "fieldsplit_common":
        _assemble_pc_fieldsplit_additive(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            key=key,
            petsc_matrices=petsc_matrices,
        )
    elif config_type == "composite":
        _assemble_pc_composite(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            key=key,
            petsc_matrices=petsc_matrices,
        )
    elif config_type == "python_permutation":
        _assemble_pc_python_permutation(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            key=key,
            petsc_matrices=petsc_matrices,
        )
    else:
        # Anything else does not need a special initialization from python.
        try:
            ksp.setUp()
            pc.setUp()
        except:
            logger.error(f"Failed on {key = }")
            raise
