from dataclasses import dataclass
from typing import Optional
from warnings import warn
from petsc4py import PETSc
from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.petsc_solvers import LinearSolverWithTransformations, PetscKrylovSolver
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    construct_is,
    csr_to_petsc,
    insert_petsc_options,
)
from pp_solvers.dof_manager import DofManager
from pp_solvers.preconditioners import PetscKspPcConfiguration


# TODO: This class will be refactored (removed), because now it's just a wrapper to
# create a ksp.
@dataclass
class PetscKSPScheme:
    """Scheme for a KSP solver for a multiphysics problem."""

    petsc_ksp_pc_configuration: PetscKspPcConfiguration
    """The factory object to produce the underlying KSP."""

    dof_manager: DofManager

    def make_solver(self, mat_orig: BlockLinearSystem, options: dict):
        # Construct a PETSc matrix from the scipy matrix.
        # TODO: Can we at this point delete the scipy matrix to save memory?
        petsc_mat = csr_to_petsc(mat_orig.mat)

        # Clear the PETSc options from a previous solve.
        petsc_options = clear_petsc_options()

        all_options_dict = self.petsc_ksp_pc_configuration.petsc_options(
            user_options=options, prefix="", dof_manager=self.dof_manager
        )
        assembly_config = self.petsc_ksp_pc_configuration.petsc_assembly_config(
            user_options=options, prefix="", dof_manager=self.dof_manager
        )

        insert_petsc_options(all_options_dict)

        petsc_ksp = PETSc.KSP().create()
        petsc_ksp.setFromOptions()
        petsc_ksp.setOperators(petsc_mat)
        assemble_petsc_ksp_pc(
            ksp=petsc_ksp,
            pc=petsc_ksp.getPC(),
            assembly_config=assembly_config,
            indexer=mat_orig.indexer,
            prefix="",
        )

        for key in all_options_dict:
            if not petsc_options.used(key):
                raise ValueError(
                    f"PETSc option {key}: {all_options_dict[key]} is not used. "
                    "Check spelling."
                )

        return PetscKrylovSolver(petsc_ksp)


# TODO: This class will be refactored.
@dataclass
class LinearTransformedScheme:
    inner: PetscKSPScheme
    """The actual solver, to be applied after the transformations."""

    left_transformations: Optional[list] = None
    right_transformations: Optional[list] = None

    @property
    def dof_manager(self):
        return self.inner.dof_manager

    def make_solver(
        self, mat_orig: BlockLinearSystem, options: dict
    ) -> PetscKrylovSolver | LinearSolverWithTransformations:
        bmat = mat_orig[:]

        if self.left_transformations is None or len(self.left_transformations) == 0:
            Qleft = None
        else:
            # The steps should be roughly the same as for the right transfor (below).
            Qleft = self.left_transformations[0](bmat)
            for tmp in self.left_transformations[1:]:
                tmp = tmp(bmat)
                Qleft.mat @= tmp.mat

        if self.right_transformations is None or len(self.right_transformations) == 0:
            Qright = None
        else:
            Qright = self.right_transformations[0](bmat)
            for tmp in self.right_transformations[1:]:
                tmp = tmp(bmat)
                Qright.mat @= tmp.mat

        bmat_Q = bmat
        if Qleft is not None:
            bmat_Q.mat = Qleft.mat @ bmat_Q.mat
        if Qright is not None:
            bmat_Q.mat = bmat_Q.mat @ Qright.mat

        if self.inner is None:
            raise ValueError("No inner solver provided.")

        # Set up the inner solver.
        solver = self.inner.make_solver(bmat_Q, options=options or {})

        if Qleft is not None or Qright is not None:
            solver = LinearSolverWithTransformations(
                inner=solver, Qright=Qright, Qleft=Qleft
            )

        return solver


def _assemble_pc_fieldsplit(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    additional_data: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    # calls: pc.setUp, ksp.setUp
    prefix_config = additional_data[prefix]
    elim_groups = prefix_config["elim_groups"]
    keep_groups = prefix_config["keep_groups"]
    elim_tag = prefix_config["elim_tag"]
    keep_tag = prefix_config["keep_tag"]

    elim_prefix = f"{prefix}fieldsplit_{elim_tag}_"
    elim_config = additional_data.get(elim_prefix, {})
    # elim_matrix_block_size = elim_config.get("matrix_block_size")
    keep_prefix = f"{prefix}fieldsplit_{keep_tag}_"
    keep_config = additional_data.get(keep_prefix, {})
    # keep_matrix_block_size = keep_config.get("matrix_block_size")

    is_elim = construct_is(indexer, elim_groups)
    is_keep = construct_is(indexer, keep_groups)

    # if elim_matrix_block_size is not None:
    #     is_elim.setBlockSize(elim_matrix_block_size)
    # if keep_matrix_block_size is not None:
    #     is_keep.setBlockSize(keep_matrix_block_size)

    assert pc.type == "fieldsplit"

    pc.setFieldSplitIS((elim_tag, is_elim))
    pc.setFieldSplitIS((keep_tag, is_keep))

    # For a matrix [[A, B], [C, D]], Schur complement S = A - C * D^-1 * B, here A
    # corresponds to the index set "is_keep". An additive invertor is a matrix X to
    # build the approximat: S = A + X. This is where the fixed-stress approximation for
    # hydromechanics is applied.
    invertor = prefix_config.get("invertor_additive", None)
    if invertor is not None:
        # This copies the submatrix A into S.
        S = pc.getOperators()[1].createSubMatrix(is_keep, is_keep)
        # Extracts the matrix X in petsc format.
        petsc_matrix_invertor = invertor(indexer)
        # S = S + 1 * X
        S.axpy(1, petsc_matrix_invertor)
        pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

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
        assembly_config=additional_data,
        indexer=indexer[elim_groups],
        prefix=elim_prefix,
    )

    pc_keep = ksp_keep.getPC()
    assemble_petsc_ksp_pc(
        ksp=ksp_keep,
        pc=pc_keep,
        assembly_config=additional_data,
        indexer=indexer[keep_groups],
        prefix=keep_prefix,
    )


def _assemble_pc_composite(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    additional_data: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    assert pc.type == "composite"
    num_stages = additional_data[prefix]["num_stages"]

    for i in range(num_stages):
        # explaining this
        pc_type = additional_data.get(f"{prefix}sub_{i}_", {}).get("pc_type", "none")
        pc.addCompositePCType(pc_type)
        sub_pc = pc.getCompositePC(i)
        sub_pc.setOperators(*pc.getOperators())
        assemble_petsc_ksp_pc(
            ksp=ksp,
            pc=sub_pc,
            assembly_config=additional_data,
            indexer=indexer,
            prefix=f"{prefix}sub_{i}_",
        )

    pc.setUp()
    ksp.setUp()


def _assemble_pc_python(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    additional_data: dict,
    indexer: LinearSystemIndexer,
    prefix: str,
):
    python_context = additional_data[prefix]["python_context"]
    pc.setPythonContext(python_context)
    pc.setUp()
    ksp.setUp()


def assemble_petsc_ksp_pc(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    assembly_config: dict,
    indexer: LinearSystemIndexer,
    prefix: str = "",
):
    if len(prefix) > 126:
        # PETSc has a limit on the prefix length, which seems to be 127
        # characters. If the prefix is too long, we raise a warning.
        msg = "The prefix for the PETSc preconditioner is too long. "
        msg += "Check the configuration of the preconditioner."
        warn(msg)

    ksp.setFromOptions()
    pc.setFromOptions()

    current_config: dict = assembly_config.get(prefix, {})

    # matrix_block_size: int | None = current_config.get("matrix_block_size")
    # if matrix_block_size is not None:
    petsc_amat, petsc_pmat = ksp.getOperators()
    petsc_amat.setFromOptions()
    petsc_pmat.setFromOptions()

    pc_type: str = current_config.get("pc_type", "other")
    # Sanity check.
    if pc_type != "other":
        assert pc.type == pc_type

    if pc_type == "fieldsplit":
        _assemble_pc_fieldsplit(
            ksp=ksp,
            pc=pc,
            additional_data=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    elif pc_type == "composite":
        _assemble_pc_composite(
            ksp=ksp,
            pc=pc,
            additional_data=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    elif pc_type == "python":
        _assemble_pc_python(
            ksp=ksp,
            pc=pc,
            additional_data=assembly_config,
            indexer=indexer,
            prefix=prefix,
        )
    else:
        try:
            ksp.setUp()
            pc.setUp()
        except:
            print(f"Failed on {prefix = }")
            raise
