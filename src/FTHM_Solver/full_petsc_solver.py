from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from petsc4py import PETSc

from .block_matrix import BlockMatrixStorage
from .mat_utils import csr_to_petsc, сlear_petsc_options

__all__ = [
    "PetscFieldSplitScheme",
    "PetscKSPScheme",
    "insert_petsc_options",
    "petsc_options_as_str",
]


def construct_is(bmat: BlockMatrixStorage, groups: list[int]) -> PETSc.IS:
    """Construct a PETSc IS (index set) from a list of groups.

    Parameters:
        bmat: The block matrix storage.
        groups: The groups to construct the IS from.

    Returns:
        The PETSc IS object representing the groups.

    """
    # TODO: Why is it necessary to create an empty container here, and not just work
    # with information from the bmat object?
    empty_mat = bmat.empty_container()
    dofs = [
        empty_mat.local_dofs_row[x]
        for i in groups
        for x in empty_mat.groups_to_blocks_row[i]
    ]
    if len(dofs) > 0:
        return PETSc.IS().createGeneral(
            np.concatenate(
                dofs,
                dtype=np.int32,  # TODO: What if the size is too large for int32?
            )
        )
    else:
        # Return an empty IS if the group is empty.
        return PETSc.IS().createGeneral(np.array([], dtype=np.int32))


def build_tag(groups: list[int]) -> str:
    """Build a tag from a list of groups."""
    return "-".join([str(x) for x in groups])


def insert_petsc_options(options):
    petsc_options = PETSc.Options()
    for k, v in options.items():
        petsc_options[k] = v


def petsc_options_as_str(stem: str) -> str:
    options = PETSc.Options().getAll()

    s = ""
    if stem.lower().strip() == "ksp":
        known_keys = [
            "ksp_type",
            "ksp_rtol",
            "ksp_max_it",
            "ksp_gmres_cgs_refinement_type",
            "ksp_gmres_classicalgramschmidt",
        ]
        for key in known_keys:
            if key in options:
                s += f"{key}: {options[key]}\n"

    return s


@dataclass
class PetscFieldSplitScheme:
    """WARNING: This documentation is incomplete and may be incorrect.

    Dataclass to define the setup of a PETSc field split preconditioner.

    The preconditioner deals linear systems of the form Ax = b, where A is a 2 x 2 block
    matrix

        A = [[A00, A01],
             [A10, A11]]

    where A00 and A11 are diagonal blocks, and A01 and A10 are off-diagonal blocks. The
    preconditioner is constructed by eliminating A_00, and solving the Schur complement
    system

        S = A_11 - A_10 A_00^{-1} A_01,

    though in practice, an approximation of both A_00^{-1} and S is most often used.

    The details of how the preconditioner operates depend on the specific options set in
    the `elim_options`, `fieldsplit_options`, and `keep_options`.

    See https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/ for more information.



    """

    groups: list[int]
    """The groups that should be eliminated by the preconditioner."""

    complement: Optional[PetscFieldSplitScheme] = None
    """The preconditioner for the complement of the groups."""

    elim_options: dict[str, str | float | int] | None = None
    """Options for the block that is eliminated in this preconditioner. Should contain
    a key `pc_type` that determines the type of preconditioner used, as well as other
    options that are specific to the preconditioner type.
    
    """

    fieldsplit_options: dict = None
    """Options for the field split preconditioner. 
    
    One key parameter is `pc_fieldsplit_schur_precondition`, which determines the
    type of approximation used for the Schur complement. See 
    https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/ for more
    information. 

    Other options may also be possible, but are unknown to EK at the time of writing.

    """

    keep_options: dict[str, str] | None = None
    """Options for the block that is kept in the preconditioner. 

    Possible options are::
      - key 'mat_schur_complement_ainv_type', for more information see
      https://petsc.org/release/manualpages/KSP/MatSchurComplementSetAinvType/
      - Set up a KSP solver for the Schur complement.

    """

    block_size: int = 1
    invert: Callable[[PETSc.Mat], PETSc.Mat] | None = None
    """If not None, a function that inverts the A_00 block (or A_11)???"""

    python_pc: PETSc.PC = None
    # experimental
    near_null_space: list[np.ndarray] | None = None
    """A list of near null space vectors to be used in the preconditioner."""

    ksp_keep_use_pmat: bool = False

    def get_groups(self) -> list[int]:
        groups = [g for g in self.groups]
        if self.complement is not None:
            groups.extend(self.complement.get_groups())
        return groups

    def configure(
        self,
        bmat: BlockMatrixStorage,
        petsc_pc: PETSc.PC,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Configure a PETSc PC object with the given block matrix and options.

        Parameters:
            bmat: The block matrix storage.
            petsc_pc: The PETSc PC object to be configured.
            prefix: A prefix to be used for the PETSc options. Should coincide with the
                prefix used for petsc_pc (the return of petsc_pc.getOptionsPrefix()).

        """
        elim_options = self.elim_options or {}
        fieldsplit_options = self.fieldsplit_options or {}
        keep_options = self.keep_options or {}

        elim = self.groups
        if self.complement is None:
            # There is no inner Schur complement to be treated by a nested fieldsplit.
            # We just need to eliminate the given groups.
            options = (
                {
                    # By default, we use a direct solver for the eliminated block.
                    f"{prefix}ksp_type": "preonly",
                    f"{prefix}pc_type": "lu",
                }
                # Override with options provided for the eliminated block.
                | {f"{prefix}{k}": v for k, v in elim_options.items()}
                # Override with options provided for the fieldsplit. TODO: Why? it makes
                # more sense to EK to use the options from the eliminated block only.
                # This can probably go.
                | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
            )

            if self.python_pc is not None:
                # EK believes this allows us to define a preconditioner in terms of
                # Python code.
                options[f"{prefix}pc_type"] = "python"
                python_pc = self.python_pc(bmat)
                python_pc.petsc_pc.setOptionsPrefix(f"{prefix}python_")
                petsc_pc.setType("python")
                petsc_pc.setPythonContext(python_pc)

            insert_petsc_options(options)
            petsc_pc.setFromOptions()
            petsc_pc.setUp()
            return options

        # If there is a complement, we need to construct a fieldsplit preconditioner.
        keep = self.complement.get_groups()

        # Create tags for the groups to be eliminated and kept. This defines unique
        # identifiers that can be used in the PETSc options.
        elim_tag = build_tag(elim)
        keep_tag = build_tag(keep)
        empty_bmat = bmat.empty_container()[elim + keep]

        if self.invert is not None:
            # The user is obliged to provide a function that inverts the A_00 block used
            # to construct the Schur complement.
            fieldsplit_options["pc_fieldsplit_schur_precondition"] = "user"

        options = (
            {
                # By default, we do a Schur complement preconditioner with a direct
                # solver for the eliminated block.
                #
                # EK note to self: The settings here have proven useful so far, but
                # should not be considered sacrosanct.
                f"{prefix}pc_type": "fieldsplit",
                f"{prefix}pc_fieldsplit_type": "schur",
                f"{prefix}pc_fieldsplit_schur_precondition": "selfp",
                f"{prefix}pc_fieldsplit_schur_fact_type": "upper",
                f"{prefix}fieldsplit_{elim_tag}_ksp_type": "preonly",
                f"{prefix}fieldsplit_{elim_tag}_pc_type": "lu",
                f"{prefix}fieldsplit_{keep_tag}_ksp_type": "preonly",
            }
            # Override with options provided for the eliminated block, the kept block,
            # and the fieldsplit (in increasing order of precedence).
            | {f"{prefix}fieldsplit_{elim_tag}_{k}": v for k, v in elim_options.items()}
            | {f"{prefix}fieldsplit_{keep_tag}_{k}": v for k, v in keep_options.items()}
            | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
        )

        # Insert the new options into the PETSc options singleton.
        insert_petsc_options(options)
        # Set the options for the PETSc PC object.
        petsc_pc.setFromOptions()

        # Construct the PETSc IS objects for the groups to be eliminated and kept.
        petsc_is_keep: PETSc.IS = construct_is(empty_bmat, keep)
        petsc_is_elim: PETSc.IS = construct_is(empty_bmat, elim)
        petsc_is_elim.setBlockSize(self.block_size)
        # Set the IS objects for the fieldsplit.
        petsc_pc.setFieldSplitIS((elim_tag, petsc_is_elim), (keep_tag, petsc_is_keep))

        if self.invert is not None:
            S = petsc_pc.getOperators()[1].createSubMatrix(petsc_is_keep, petsc_is_keep)
            petsc_stab = self.invert(bmat)
            S.axpy(1, petsc_stab)

            petsc_pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

        # Set up the preconditioner. This presumably (EK) constructs sub KSPs for the
        # eliminate and keep blocks.
        petsc_pc.setUp()

        petsc_ksp_elim = petsc_pc.getFieldSplitSubKSP()[0]
        petsc_pc_elim = petsc_ksp_elim.getPC()

        petsc_ksp_keep = petsc_pc.getFieldSplitSubKSP()[1]
        petsc_pc_keep = petsc_ksp_keep.getPC()

        if self.ksp_keep_use_pmat:
            _, pmat = petsc_ksp_keep.getOperators()
            # TODO: Is it correct to use the same matrix for both arguments?
            petsc_ksp_keep.setOperators(pmat, pmat)

        if self.near_null_space is not None:
            null_space_vectors = []
            for b in self.near_null_space:
                null_space_vec_petsc = PETSc.Vec().create()  # possibly mem leak
                null_space_vec_petsc.setSizes(b.shape[0], self.block_size)
                null_space_vec_petsc.setUp()
                null_space_vec_petsc.setArray(b)
                null_space_vectors.append(null_space_vec_petsc)
            # possibly mem leak
            null_space_petsc = PETSc.NullSpace().create(True, null_space_vectors)
            petsc_pc_elim.getOperators()[1].setNearNullSpace(null_space_petsc)

        # Call on self.complement to configure the PETSc PC object for the complement,
        # and update (override) the options with the options returned by the complement.
        # Note that, due to the tagging system, this may override some options that were
        # set above.
        options |= self.complement.configure(
            bmat,
            prefix=f"{prefix}fieldsplit_{keep_tag}_",
            petsc_pc=petsc_pc_keep,
        )

        return options


@dataclass
class PetscKSPScheme:
    """Scheme for a KSP solver for a multiphysics problem."""

    preconditioner: Optional[PetscFieldSplitScheme] = None
    """The preconditioner to be used."""

    petsc_options: Optional[dict] = None
    """Additional options to be passed to PETSc."""

    compute_eigenvalues: bool = False
    """Whether to compute the eigenvalues of the matrix."""

    def get_groups(self) -> list[int]:
        """Return the groups of the preconditioner."""
        return self.preconditioner.get_groups()

    def make_solver(self, mat_orig: BlockMatrixStorage):
        # Construct a PETSc matrix from the scipy matrix.
        # TODO: Can we at this point delete the scipy matrix to save memory?
        petsc_mat = csr_to_petsc(mat_orig.mat)

        # Clear the PETSc options from a previous solve.
        сlear_petsc_options()

        # Hard coded options for the KSP solver. TODO: Figure out how this can be
        # configured from the outside.
        options = {
            # "ksp_monitor": None,
            "ksp_type": "gmres",
            "ksp_pc_side": "right",
            "ksp_rtol": 1e-10,
            "ksp_max_it": 120,
            "ksp_gmres_cgs_refinement_type": "refine_ifneeded",
            "ksp_gmres_classicalgramschmidt": True,  # Not givens rotations??
        } | (self.petsc_options or {})

        # Insert the above options into the PETSc options singleton.
        insert_petsc_options(options)

        # Create the PETSc KSP object, set matrix and preconditioner.
        petsc_ksp = PETSc.KSP().create()
        petsc_ksp.setOperators(petsc_mat)
        petsc_ksp.setFromOptions()
        petsc_pc = petsc_ksp.getPC()
        if self.preconditioner is not None:
            options |= self.preconditioner.configure(
                bmat=mat_orig,
                petsc_pc=petsc_pc,
            )
        if self.compute_eigenvalues:
            petsc_ksp.setComputeEigenvalues(True)

        petsc_ksp.setUp()
        self.options = options
        return PetscKrylovSolver(petsc_ksp)


@dataclass
class LinearTransformedScheme:
    left_transformations: Optional[
        list[Callable[[BlockMatrixStorage], BlockMatrixStorage]]
    ] = None
    right_transformations: Optional[
        list[Callable[[BlockMatrixStorage], BlockMatrixStorage]]
    ] = None
    # This is not optional.
    inner: Optional[PetscKSPScheme] = None
    """The actual solver, to be applied after the transformations.
    
    TODO: Should the typing allow for a more general solver?
    """

    def get_groups(self) -> list[int]:
        return self.inner.get_groups()

    def make_solver(
        self, mat_orig: BlockMatrixStorage
    ) -> PetscKrylovSolver | LinearSolverWithTransformations:
        groups = self.get_groups()
        bmat = mat_orig[groups]

        if self.left_transformations is None or len(self.left_transformations) == 0:
            Qleft = None
        else:
            Qleft = self.left_transformations[0](bmat)[groups]
            for tmp in self.left_transformations[1:]:
                tmp = tmp(bmat)[groups]
                Qleft.mat @= tmp.mat

        if self.right_transformations is None or len(self.right_transformations) == 0:
            Qright = None
        else:
            Qright = self.right_transformations[0](bmat)[groups]
            for tmp in self.right_transformations[1:]:
                tmp = tmp(bmat)[groups]
                Qright.mat @= tmp.mat

        bmat_Q = bmat
        if Qleft is not None:
            bmat_Q.mat = Qleft.mat @ bmat_Q.mat
        if Qright is not None:
            bmat_Q.mat = bmat_Q.mat @ Qright.mat

        if self.inner is None:
            raise ValueError("No inner solver provided.")

        solver: PetscKrylovSolver = self.inner.make_solver(bmat_Q)
        self.options = self.inner.options

        if Qleft is not None or Qright is not None:
            solver = LinearSolverWithTransformations(
                inner=solver, Qright=Qright, Qleft=Qleft
            )

        return solver


class LinearSolverWithTransformations:
    def __init__(
        self,
        inner: PetscKrylovSolver,
        Qleft: Optional[BlockMatrixStorage] = None,
        Qright: Optional[BlockMatrixStorage] = None,
    ):
        self.Qleft: BlockMatrixStorage | None = Qleft
        self.Qright: BlockMatrixStorage | None = Qright
        self.inner: PetscKrylovSolver = inner
        self.ksp = inner.ksp

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Transform the right-hand side, solve the linear system, and transform the
        solution back.

        """
        rhs_Q = rhs
        if self.Qleft is not None:
            rhs_Q = self.Qleft.mat @ rhs_Q

        sol_Q = self.inner.solve(rhs_Q)

        if self.Qright is not None:
            sol = self.Qright.mat @ sol_Q
        else:
            sol = sol_Q

        return sol

    def get_residuals(self):
        return self.inner.get_residuals()


class PetscKrylovSolver:
    """Shallow wrapper around a PETSc KSP object."""

    def __init__(
        self,
        ksp,
    ) -> None:
        """Initialize the solver with a PETSc KSP object.

        Parameters:
            ksp: A PETSc KSP object.

        """
        self.ksp = ksp
        petsc_mat = ksp.getOperators()[0]

        # TODO: Why left here?
        self.petsc_x = petsc_mat.createVecLeft()
        self.petsc_b = petsc_mat.createVecLeft()
        # self.ksp.setComputeEigenvalues(True)
        self.ksp.setConvergenceHistory()

    def __del__(self) -> None:
        """Destroy the PETSc objects."""
        self.ksp.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve the linear system with the given right-hand side.

        Parameters:
            b: The right-hand side of the linear system.

        Returns:
            The solution of the linear system.

        """
        self.petsc_b.setArray(b)
        self.petsc_x.set(0.0)
        self.ksp.solve(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def get_residuals(self):
        return self.ksp.getConvergenceHistory()


@dataclass
class PetscCPRScheme:
    groups: list[int]
    pressure_groups: list[int]
    pressure_options: dict = None
    others_options: dict = None
    cpr_options: dict = None

    def get_groups(self) -> list[int]:
        return self.groups

    def configure(
        self, bmat: BlockMatrixStorage, petsc_pc: PETSc.PC, prefix: str = ""
    ) -> dict:
        bmat = bmat[self.groups]
        cpr_options = self.cpr_options or {}
        flow_options = {"ksp_type": "preonly"} | (self.pressure_options or {})
        others_options = self.others_options or {}
        other_groups = [gr for gr in self.groups if gr not in self.pressure_groups]
        flow_tag = build_tag(self.pressure_groups)
        others_tag = build_tag(other_groups)
        flow_prefix = f"{prefix}sub_0_fieldsplit_{flow_tag}_"
        others_prefix = f"{prefix}sub_0_fieldsplit_{others_tag}_"
        options = (
            {
                f"{prefix}pc_type": "composite",
                f"{prefix}pc_composite_type": "multiplicative",
                f"{prefix}pc_composite_pcs": "fieldsplit,ilu",
                # f"{prefix}sub_0_ksp_type": "preonly",
                f"{prefix}sub_0_pc_fieldsplit_type": "additive",
            }
            | {f"{prefix}{k}": v for k, v in cpr_options.items()}
            | {f"{flow_prefix}{k}": v for k, v in flow_options.items()}
            | {f"{others_prefix}{k}": v for k, v in others_options.items()}
        )
        insert_petsc_options(options)
        petsc_pc.setFromOptions()

        petsc_is_flow = construct_is(bmat, self.pressure_groups)
        petsc_is_others = construct_is(bmat, other_groups)
        fieldsplit = petsc_pc.getCompositePC(0)
        fieldsplit.setFieldSplitIS(
            (flow_tag, petsc_is_flow), (others_tag, petsc_is_others)
        )

        petsc_pc.setUp()
        fieldsplit.setUp()
        return options


@dataclass
class PetscCompositeScheme:
    """Scheme for a composite (2-stage??)  preconditioner."""

    groups: list[int]
    solvers: list[PetscFieldSplitScheme]
    petsc_options: dict = None

    def get_groups(self) -> list[int]:
        return self.groups

    def configure(
        self, bmat: BlockMatrixStorage, petsc_pc: PETSc.PC, prefix: str = ""
    ) -> dict:
        options = {
            f"{prefix}{k}": v
            for k, v in (
                {
                    "pc_type": "composite",
                    "pc_composite_type": "multiplicative",
                    "pc_composite_pcs": ",".join(["none"] * len(self.solvers)),
                }
                | (self.petsc_options or {})
            ).items()
        }
        insert_petsc_options(options)
        petsc_pc.setFromOptions()
        petsc_pc.setUp()
        for i, solver in enumerate(self.solvers):
            sub_pc = petsc_pc.getCompositePC(i)
            sub_options = solver.configure(
                bmat=bmat, petsc_pc=sub_pc, prefix=f"{prefix}sub_{i}_"
            )
            options |= sub_options

        return options


class PcPythonPermutation:
    def __init__(self, perm: np.ndarray, block_size: int):
        self.petsc_pc = PETSc.PC().create()
        self.petsc_is_perm = PETSc.IS().createGeneral(perm.astype(np.int32))
        self.P_perm = PETSc.Mat()
        self.b = PETSc.Vec().create()
        self.bs = block_size
        self.b.setSizes(perm.size)
        self.b.setUp()

    def __del__(self):
        self.petsc_pc.destroy()
        self.petsc_is_perm.destroy()
        self.b.destroy()

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer) -> None:
        self.petsc_pc.view(viewer)

    def setFromOptions(self, pc: PETSc.PC) -> None:
        self.petsc_pc.setFromOptions()

    def setUp(self, pc: PETSc.PC) -> None:
        _, P = pc.getOperators()
        self.P_perm = P.permute(self.petsc_is_perm, self.petsc_is_perm)
        self.P_perm.setBlockSize(self.bs)
        self.petsc_pc.setOperators(self.P_perm, self.P_perm)
        self.petsc_pc.setUp()

    def reset(self, pc: PETSc.PC) -> None:
        self.petsc_pc.reset()
        self.P_perm.destroy()

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        b.copy(self.b)
        self.b.permute(self.petsc_is_perm)
        self.petsc_pc.apply(self.b, x)
        x.permute(self.petsc_is_perm, invert=True)
