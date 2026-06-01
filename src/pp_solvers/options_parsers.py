"""This module defines the machinery to parse the configuration of the PETSc linear
solver and build the corresponding PETSc KSP and PC objects."""

from warnings import warn

import numpy as np
from petsc4py import PETSc

from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.dof_manager import DofManager
from pp_solvers.petsc_solvers import (
    PcPythonPermutation,
    PetscKrylovSolver,
)
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    construct_is,
    csr_to_petsc,
    insert_petsc_options,
)
from pp_solvers.preconditioners import PetscKspPcConfiguration


def initialize_petsc_ksp(
    block_linear_system: BlockLinearSystem,
    dof_manager: DofManager,
    petsc_ksp_pc_configuration: PetscKspPcConfiguration,
    user_options: dict,
):
    # TODO YZ: Check that the user did not misspell a key in options, e.g. cpr0_mass
    # TODO YZ: Check that all keys in solvers are unique.
    # These two tasks would require a recursive method on PetscKspPcScheme that will
    # gather and count all the keys.

    # Construct a PETSc matrix from the scipy matrix.
    petsc_mat = csr_to_petsc(block_linear_system.mat)
    if user_options.get("delete_matrices", True):
        del block_linear_system.mat  # Delete the scipy matrix to save memory.

    # Clear the PETSc options from a previous solve.
    petsc_options = clear_petsc_options()

    all_options_dict = petsc_ksp_pc_configuration.petsc_options(
        user_options=user_options, prefix="", dof_manager=dof_manager
    )
    assembly_config = petsc_ksp_pc_configuration.petsc_assembly_config(
        user_options=user_options, prefix="", dof_manager=dof_manager
    )

    insert_petsc_options(all_options_dict)

    petsc_ksp = PETSc.KSP().create()
    petsc_ksp.setFromOptions()
    petsc_ksp.setOperators(petsc_mat)
    assemble_petsc_ksp_pc(
        ksp=petsc_ksp,
        pc=petsc_ksp.getPC(),
        assembly_config=assembly_config,
        indexer=block_linear_system.indexer,
        prefix="",
    )

    for key in all_options_dict:
        if not petsc_options.used(key):
            raise ValueError(
                f"PETSc option {key}: {all_options_dict[key]} is not used. "
                "Check spelling."
            )

    return PetscKrylovSolver(petsc_ksp)


def _assemble_pc_fieldsplit_additive(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    assert pc.type == "fieldsplit"

    prefix_config = assembly_config[prefix]
    subsolver_groups = prefix_config["subsolver_groups"]

    for i, groups in enumerate(subsolver_groups):
        is_subsolver = construct_is(indexer, groups)
        pc.setFieldSplitIS((f"sub_{i}", is_subsolver))

    try:
        pc.setUp()
        ksp.setUp()
    except:
        print(f"failed on {prefix = }")
        raise

    sub_ksp_list = pc.getFieldSplitSubKSP()
    for sub_ksp, groups in zip(sub_ksp_list, subsolver_groups):
        assemble_petsc_ksp_pc(
            ksp=sub_ksp,
            pc=sub_ksp.getPC(),
            assembly_config=assembly_config,
            indexer=indexer[groups],
            prefix=sub_ksp.prefix,
        )


def _assemble_pc_fieldsplit_schur(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    # calls: pc.setUp, ksp.setUp
    assert pc.type == "fieldsplit"

    prefix_config = assembly_config[prefix]
    elim_groups = prefix_config["elim_groups"]
    keep_groups = prefix_config["keep_groups"]
    elim_tag = prefix_config["elim_tag"]
    keep_tag = prefix_config["keep_tag"]

    is_elim = construct_is(indexer, elim_groups)
    is_keep = construct_is(indexer, keep_groups)

    keep_groups_indexer = indexer[keep_groups]

    pc.setFieldSplitIS((elim_tag, is_elim))
    pc.setFieldSplitIS((keep_tag, is_keep))

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
        print(f"failed on {prefix = }")
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
        prefix=f"{prefix}fieldsplit_{elim_tag}_",
    )

    pc_keep = ksp_keep.getPC()
    assemble_petsc_ksp_pc(
        ksp=ksp_keep,
        pc=pc_keep,
        assembly_config=assembly_config,
        indexer=indexer[keep_groups],
        prefix=f"{prefix}fieldsplit_{keep_tag}_",
    )


def _assemble_pc_composite(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    assert pc.type == "composite"
    num_stages = assembly_config[prefix]["num_stages"]

    for i in range(num_stages):
        # We need to access each sub-preconditioner. We need to create them using
        # pc.addCompositePCType(type). We do not know the type here, as it is provided
        # in petsc options. So we create them with a placeholder type "none".
        pc_type = assembly_config.get(f"{prefix}sub_{i}_", {}).get("pc_type", "none")
        pc.addCompositePCType(pc_type)
        # Access the newly created sub-preconditioner.
        sub_pc = pc.getCompositePC(i)
        # Each sub-pc of a composite preconditioner works with the same Amat and Pmat.
        sub_pc.setOperators(*pc.getOperators())
        # The actual type of each sub_pc will be fetched here from PETSc options.
        assemble_petsc_ksp_pc(
            ksp=ksp,
            pc=sub_pc,
            assembly_config=assembly_config,
            indexer=indexer,
            prefix=f"{prefix}sub_{i}_",
        )

    try:
        ksp.setUp()
        pc.setUp()
    except:
        print(f"Failed on {prefix = }")
        raise


def _assemble_pc_python_permutation(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    permutation_groups: list[list[int]] = assembly_config[prefix]["permutation_groups"]

    perm = [indexer.get_dofs_of_groups(g)[0] for g in permutation_groups]
    perm = np.vstack(perm).ravel("F")

    python_context = PcPythonPermutation(
        perm=perm, block_size=len(permutation_groups), prefix=prefix
    )
    python_context.setFromOptions(pc=pc)

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
    if python_context.petsc_pc.type in ["fieldsplit", "composite"]:
        raise NotImplementedError(
            "Nested initialization inside PythonPermutationWrapper is not implemented."
        )

    pc.setPythonContext(python_context)
    try:
        ksp.setUp()
        pc.setUp()
    except:
        print(f"Failed on {prefix = }")
        raise


def assemble_petsc_ksp_pc(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    prefix: str = "",
):
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
        "elim_tag": "tag1",  # tag to build the petsc prefix for the eliminated groups.
        "keep_tag": "tag2",  # tag to build the petsc prefix for the kept groups.
    }
    {
        "config_type": "fieldsplit_additive",
        "subsolver_groups": [
            [0, 1],
            [2, 3],
            [5, 6],
        ],  # list of groups to build the non-Schur-complement fieldsplit.
    }
    {
        "config_type": "composite",
        "num_stages": 3,  # number of stages for the composite preconditioner.
    }
    {
        "config_type": "python_permutation",
        "permutation_groups": [[1, 2], [3, 4]],  # Groups to permute.
    }

    """
    if len(prefix) > 126:
        # PETSc has a limit on the prefix length, which seems to be 127
        # characters. If the prefix is too long, we raise a warning.
        msg = "The prefix for the PETSc preconditioner is too long. "
        msg += "Check the configuration of the preconditioner."
        warn(msg)

    # This is where the ksp and pc objects fetch options in PETSc command-line format.
    ksp.setFromOptions()
    pc.setFromOptions()

    current_config: dict = assembly_config.get(prefix, {})

    petsc_amat, petsc_pmat = ksp.getOperators()
    # The command-line options for a matrix include mat_block_size (integer) and
    # mat_type including "aij" or "baij", corresponding to csr and bsr sparse formats,
    # respectively. Matrices share the prefix of the ksp and the pc.
    petsc_amat.setFromOptions()
    petsc_pmat.setFromOptions()

    config_type: str = current_config.get("config_type", "default")

    if config_type == "fieldsplit_schur":
        _assemble_pc_fieldsplit_schur(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    elif config_type == "fieldsplit_additive":
        _assemble_pc_fieldsplit_additive(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    elif config_type == "composite":
        _assemble_pc_composite(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    elif config_type == "python_permutation":
        _assemble_pc_python_permutation(
            ksp=ksp,
            pc=pc,
            assembly_config=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    else:
        # Anything else does not need a special initialization from python.
        try:
            ksp.setUp()
            pc.setUp()
        except:
            print(f"Failed on {prefix = }")
            raise
