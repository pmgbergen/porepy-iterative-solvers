from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import numpy as np
import petsc4py
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numba import njit
from petsc4py import PETSc

if TYPE_CHECKING:
    from block_matrix import BlockMatrixStorage, FieldSplitScheme

petsc4py.init(sys.argv)


__all__ = [
    "csr_to_petsc",
    "petsc_to_csr",
    "сlear_petsc_options",
    "csr_ones",
    "inv_block_diag",
]


def assert_finite(vals: np.ndarray, groups: list[int]) -> None:
    pass
    # if not np.all(np.isfinite(vals)) or np.any(abs(vals).max() > 1e30):
    #     print("Divergence", groups)


def сlear_petsc_options() -> PETSc.Options:
    """Options is a singletone. This ensures that no unwanted options from some previous
    setup reach the current setup."""
    options = PETSc.Options()

    for key in options.getAll():
        options.delValue(key)
    return options


class FieldSplit:
    def __init__(
        self,
        solve_momentum,
        solve_mass,
        C1,
        C2,
        groups_0=None,
        groups_1=None,
        factorization_type: Literal["full", "upper", "lower"] = "full",
    ):
        self.groups_0 = groups_0
        self.groups_1 = groups_1
        self.J00_inv = solve_momentum
        self.S11_inv = solve_mass
        self.J01 = C1
        self.J10 = C2
        self.sep = solve_momentum.shape[0]
        self.factorization_type: Literal["full", "upper", "lower"] = factorization_type
        shape = solve_momentum.shape[0] + solve_mass.shape[0]
        self.shape = shape, shape

    def dot(self, x):
        x_0, x_1 = x[: self.sep], x[self.sep :]
        if self.factorization_type != "upper":
            tmp_0 = self.J00_inv.dot(x_0)
            assert_finite(tmp_0, groups=self.groups_0)  # 1e+32
            tmp_1 = x_1 - self.J01.dot(tmp_0)
        else:
            tmp_0 = x_0
            tmp_1 = x_1
        y_1 = self.S11_inv.dot(tmp_1)
        assert_finite(y_1, groups=self.groups_1)
        y = np.zeros_like(x)
        y[self.sep :] = y_1
        if self.factorization_type != "lower":
            tmp_2 = self.J00_inv.dot(x_0 - self.J10.dot(y_1))
            assert_finite(tmp_2, groups=self.groups_0)
        else:
            tmp_2 = tmp_0
        y[: self.sep] = tmp_2
        return y


class RestrictedOperator:
    def __init__(self, mat: BlockMatrixStorage, solve_scheme: FieldSplitScheme):
        to_groups = solve_scheme.get_groups()
        self.R = self.make_restriction_matrix(mat, to_groups).mat
        _, self.prec = solve_scheme.make_solver(mat[to_groups])
        self.shape = mat.shape

    @staticmethod
    def make_restriction_matrix(
        bmat: "BlockMatrixStorage", to: list[int]
    ) -> "BlockMatrixStorage":
        active_cols = np.concatenate(
            [bmat.local_dofs_col[j] for i in to for j in bmat.groups_to_blocks_col[i]]
        )
        active_rows = np.arange(active_cols.size)
        active_data = np.ones(active_rows.size)
        result = bmat[to, bmat.active_groups[1]].empty_container()
        proj = scipy.sparse.coo_matrix(
            (active_data, (active_rows, active_cols)), result.shape
        )
        result.mat = proj.tocsr()
        return result

    def dot(self, x: np.ndarray) -> np.ndarray:
        x_local = self.R @ x
        y_local = self.prec.dot(x_local)
        return self.R.T @ y_local


class TwoStagePreconditioner:
    def __init__(self, mat: "BlockMatrixStorage", stages: list):
        assert len(stages) == 2
        self.mat: "BlockMatrixStorage" = mat
        self.shape = mat.shape
        self.stages: list = stages

    def dot(self, x: np.ndarray) -> np.ndarray:
        # from matplotlib import pyplot as plt
        y1 = self.stages[0].dot(x)
        r1 = x - self.mat.mat.dot(y1)
        y2 = self.stages[1].dot(r1)
        return y1 + y2


class BlockJacobi:
    def __init__(self, bmat, solve_A, solve_B, groups_0=None, groups_1=None):
        self.bmat = bmat
        self.solve_A = solve_A
        self.solve_B = solve_B
        self.sep = solve_A.shape[0]
        self.shape = bmat.shape
        self.groups_0 = groups_0
        self.groups_1 = groups_1

    def dot(self, x):
        x_0, x_1 = x[: self.sep], x[self.sep :]
        tmp_0 = self.solve_A.dot(x_0)
        assert_finite(tmp_0, groups=self.groups_0)  # 1e+32
        tmp_1 = self.solve_B.dot(x_1)
        assert_finite(tmp_1, groups=self.groups_1)
        return np.concatenate([tmp_0, tmp_1])


class BlockGS:
    def __init__(self, bmat, solve_A, solve_B, groups_0=None, groups_1=None):
        self.bmat = bmat
        self.solve_A = solve_A
        self.solve_B = solve_B
        self.A10 = bmat[groups_1, groups_0].mat
        self.A01 = bmat[groups_0, groups_1].mat
        self.sep = solve_A.shape[0]
        self.shape = bmat.shape
        self.groups_0 = groups_0
        self.groups_1 = groups_1

    def dot(self, x):
        x_0, x_1 = x[: self.sep], x[self.sep :]
        # tmp_0 = self.solve_A.dot(x_0)
        # tmp_1 = self.solve_B.dot(x_1 - self.A10.dot(tmp_0))
        tmp_1 = self.solve_B.dot(x_1)
        tmp_0 = self.solve_A.dot(x_0 - self.A01.dot(tmp_1))
        return np.concatenate([tmp_0, tmp_1])


def cond(mat):
    try:
        mat = mat.todense()
    except AttributeError:
        pass
    return np.linalg.cond(mat)


def eigs(mat):
    try:
        mat = mat.toarray()
    except AttributeError:
        pass
    return np.linalg.eigvals(mat)


def inv(mat):
    return scipy.sparse.csr_matrix(
        scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(mat))
    )


def pinv(mat):
    return scipy.sparse.csr_matrix(np.linalg.pinv(mat.toarray()))


def csr_zeros(n, m=None) -> scipy.sparse.csr_matrix:
    if m is None:
        m = n
    return scipy.sparse.csr_matrix((n, m))


def csr_ones(n) -> scipy.sparse.csr_matrix:
    return scipy.sparse.eye(n, format="csr")


def condest(mat):
    mat = mat.tocsr()
    data = abs(mat.data)
    return data.max() / data.min()


class PetscPC:
    def __init__(
        self,
        mat=None,
        block_size=1,
        null_space: np.ndarray = None,
        petsc_options: dict = None,
        name="",
    ) -> None:
        self.name = name
        self.pc = PETSc.PC().create()
        options = PETSc.Options()

        if petsc_options is None:
            petsc_options = {}
        for k, v in petsc_options.items():
            options[k] = v

        self.petsc_mat = PETSc.Mat()
        self.petsc_x = PETSc.Vec()
        self.petsc_b = PETSc.Vec()
        self.pc.setFromOptions()

        self.null_space_vectors = []
        if null_space is not None:
            for b in null_space:
                null_space_vec_petsc = PETSc.Vec().create()
                null_space_vec_petsc.setSizes(b.shape[0], block_size)
                null_space_vec_petsc.setUp()
                null_space_vec_petsc.setArray(b)
                self.null_space_vectors.append(null_space_vec_petsc)
            self.null_space_petsc = PETSc.NullSpace().create(
                True, self.null_space_vectors
            )
        else:
            self.null_space_petsc = None

        self.block_size = block_size

        self.shape: tuple[int, int]
        if mat is not None:
            self.set_operator(mat)

    def set_operator(self, mat):
        self.shape = mat.shape
        self.petsc_mat.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()
        self.petsc_mat.createAIJ(
            size=mat.shape,
            csr=(mat.indptr, mat.indices, mat.data),
            bsize=self.block_size,
        )
        if self.null_space_petsc is not None:
            self.petsc_mat.setNearNullSpace(self.null_space_petsc)
        self.petsc_b = self.petsc_mat.createVecLeft()
        self.petsc_x = self.petsc_mat.createVecLeft()
        self.pc.setOperators(self.petsc_mat)
        self.pc.setUp()

    def __del__(self):
        self.pc.destroy()
        self.petsc_mat.destroy()
        self.petsc_b.destroy()
        self.petsc_x.destroy()
        for vec in self.null_space_vectors:
            vec.destroy()
        if self.null_space_petsc is not None:
            self.null_space_petsc.destroy()

    def dot(self, b: np.ndarray) -> np.ndarray:
        self.petsc_x.set(0.0)
        self.petsc_b.setArray(b)
        self.pc.apply(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def get_matrix(self):
        return petsc_to_csr(self.petsc_mat)


class PetscAMGVector(PetscPC):
    def __init__(self, dim: int, mat=None) -> None:
        options = сlear_petsc_options()

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = 1
        # options["pc_hypre_boomeramg_cycle_type"] = "W"
        options["pc_hypre_boomeramg_truncfactor"] = 0.3
        super().__init__(mat=mat, block_size=dim)


class PetscAMGMechanics(PetscPC):
    def __init__(
        self,
        dim: int,
        mat=None,
        null_space: np.ndarray = None,
        petsc_options: dict[str, str] = None,
    ) -> None:
        options = сlear_petsc_options()
        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_strong_threshold"] = 0.7
        super().__init__(
            mat=mat, block_size=dim, null_space=null_space, petsc_options=petsc_options
        )


class PetscAMGFlow(PetscPC):
    def __init__(self, mat=None, dim: int = 2) -> None:
        options = сlear_petsc_options()

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = 1
        options["pc_hypre_boomeramg_truncfactor"] = 0.3
        super().__init__(mat=mat, block_size=1)


class PetscLU(PetscPC):
    def __init__(self, mat=None) -> None:
        options = сlear_petsc_options()
        options.setValue("pc_type", "lu")
        super().__init__(mat=mat)


class PetscILU(PetscPC):
    def __init__(self, mat=None, factor_levels: int = 0) -> None:
        options = сlear_petsc_options()
        options.setValue("pc_type", "ilu")
        options.setValue("pc_factor_levels", factor_levels)
        options.setValue("pc_factor_diagonal_fill", None)  # Doesn't affect
        # For some reason, works worse with thermal model and intersections.
        # options.setValue("pc_factor_mat_ordering_type", "rcm")
        options.setValue("pc_factor_nonzeros_along_diagonal", None)
        super().__init__(mat=mat)


class PetscHypreILU(PetscPC):
    def __init__(self, mat=None, factor_levels: int = 0) -> None:
        options = сlear_petsc_options()
        options.setValue("pc_type", "hypre")
        options.setValue("pc_hypre_type", "euclid")
        options.setValue("pc_hypre_euclid_level", factor_levels)
        # options.setValue("pc_hypre_type", "pilut")
        super().__init__(mat=mat)


class PetscPythonPC:
    def __init__(self, pc):
        self.pc = pc

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        """Apply the preconditioner on vector b, return in x."""
        result = self.pc.dot(b.getArray(readonly=True))
        x.setArray(result)


class PetscKrylovSolver:
    def __init__(
        self,
        mat,
        pc: PETSc.PC | None = None,
        tol=1e-10,
        atol=1e-10,
    ) -> None:
        options = PETSc.Options()
        options.setValue("ksp_divtol", 1e10)
        options.setValue("ksp_atol", atol)
        options.setValue("ksp_rtol", tol)

        # If no preconditioner is is explicitly provided and the options do not specify
        # a preconditioner, set it to "none".
        if pc is None and "pc_type" not in options.getAll():
            PETSc.Options().setValue("pc_type", "none")

        self.shape = mat.shape
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()

        self.pc = PETSc.PC()
        if pc is not None:
            self.pc.createPython(PetscPythonPC(pc))
            self.ksp.setPC(self.pc)

        self.ksp.setComputeEigenvalues(True)
        self.ksp.setConvergenceHistory()

        self.petsc_mat = PETSc.Mat().createAIJ(
            size=mat.shape, csr=(mat.indptr, mat.indices, mat.data)
        )
        self.ksp.setOperators(self.petsc_mat)
        self.ksp.setUp()

        self.petsc_x = self.petsc_mat.createVecLeft()
        self.petsc_b = self.petsc_mat.createVecLeft()

    def __del__(self):
        self.ksp.destroy()
        self.pc.destroy()
        self.petsc_mat.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()

    def solve(self, b):
        self.petsc_b.setArray(b)
        self.petsc_x.set(0.0)
        self.ksp.solve(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def dot(self, b):
        return self.solve(b)

    def get_residuals(self):
        return self.ksp.getConvergenceHistory()


class PetscGMRES(PetscKrylovSolver):
    def __init__(
        self,
        mat,
        pc: PETSc.PC | None = None,
        tol=1e-10,
        atol=1e-15,
        restart=30,
        max_it=90,
        pc_side: Literal["left", "right"] = "right",
        petsc_options: dict[str, str] = None,
        name="",
    ) -> None:
        self.name = name
        self.mat = mat

        options = сlear_petsc_options()
        options.setValue("ksp_type", "gmres")
        options.setValue("ksp_max_it", max_it)
        options.setValue("ksp_gmres_restart", restart)
        options.setValue("ksp_gmres_classicalgramschmidt", True)
        options.setValue("ksp_gmres_cgs_refinement_type", "refine_ifneeded")

        if pc_side == "left":
            options.setValue("ksp_pc_side", "left")
            options.setValue("ksp_norm_type", "preconditioned")
        elif pc_side == "right":
            options.setValue("ksp_pc_side", "right")
            options.setValue("ksp_norm_type", "unpreconditioned")
        else:
            raise ValueError(pc_side)

        if petsc_options is None:
            petsc_options = {}
        for k, v in petsc_options.items():
            options.setValue(k, v)

        super().__init__(mat, pc, tol, atol=atol)


class PetscRichardson(PetscKrylovSolver):
    def __init__(
        self,
        mat,
        pc: PETSc.PC | None = None,
        tol=1e-10,
        atol=1e-10,
        pc_side: Literal["left"] = "left",
    ) -> None:
        assert pc_side == "left"

        options = сlear_petsc_options()
        options.setValue("ksp_type", "richardson")
        options.setValue("ksp_max_it", 1000)

        if pc_side == "left":
            options.setValue("ksp_pc_side", "left")
            options.setValue("ksp_norm_type", "preconditioned")
        else:
            raise ValueError(pc_side)

        # Absolute tolerances are different for Richardson and GMRES because the latter
        # checks the unpreconditioned residual.
        super().__init__(mat, pc, tol, atol=atol)


class PetscJacobi(PetscPC):
    def __init__(self, mat=None) -> None:
        options = сlear_petsc_options()
        options["pc_type"] = "jacobi"
        super().__init__(mat=mat)


class PetscSOR(PetscPC):
    def __init__(self, mat=None) -> None:
        options = сlear_petsc_options()
        options["pc_type"] = "sor"
        options["pc_type_symmetric"] = True
        super().__init__(mat=mat)


def extract_diag_inv(mat, eliminate_zeros=False):
    diag = mat.diagonal()
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    if eliminate_zeros:
        diag[abs(diag) < 1e-30] = 1
    diag_inv = 1 / diag
    ones.data[:] = diag_inv
    return ones


def extract_diag(mat, lump=False):
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    if lump:
        ones.data[:] = np.array(abs(mat).sum(axis=1)).ravel()
    else:
        ones.data[:] = mat.diagonal()
    return ones


def inv_block_diag(mat, nd: int, lump: bool = False):
    if lump:
        mat = lump_nd(mat, nd)
    if nd == 1:
        return extract_diag_inv(mat)
    if nd == 2:
        return inv_block_diag_2x2(mat)
    if nd == 3:
        return inv_block_diag_3x3(mat)
    print(f"Using inefficient invert block diag, {nd = }")
    return inv(diag_nd(mat, nd=nd))


def inv_block_diag_2x2(mat):
    ad = mat.diagonal()
    a = ad[::2]
    d = ad[1::2]
    b = mat.diagonal(k=1)[::2]
    c = mat.diagonal(k=-1)[::2]

    det = a * d - b * c

    assert abs(det).min() > 0

    diag = np.zeros_like(ad)
    diag[::2] = d / det
    diag[1::2] = a / det
    lower = np.zeros(ad.size - 1)
    lower[::2] = -c / det
    upper = np.zeros(ad.size - 1)
    upper[::2] = -b / det

    return scipy.sparse.diags([lower, diag, upper], offsets=[-1, 0, 1]).tocsr()


def lump_nd(mat, nd: int):
    result = scipy.sparse.lil_matrix(mat.shape)
    indices = np.arange(0, mat.shape[0], nd)
    for i in range(nd):
        for j in range(nd):
            indices_i = indices + i
            indices_j = indices + j
            ind_i, ind_j = np.meshgrid(
                indices_i, indices_j, copy=False, sparse=True, indexing="ij"
            )
            submat = mat[ind_i, ind_j]
            lump = np.array(submat.sum(axis=1)).ravel()
            result[indices_i, indices_j] = lump
    return result.tocsr()


def diag_nd(mat, nd: int):
    result = scipy.sparse.lil_matrix(mat.shape)
    indices = np.arange(0, mat.shape[0], nd)
    for i in range(nd):
        for j in range(nd):
            indices_i = indices + i
            indices_j = indices + j
            result[indices_i, indices_j] = mat[indices_i, indices_j]
    return result.tocsr()


def extract_rowsum_inv(mat):
    rowsum = np.array((mat).sum(axis=1)).squeeze()
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    diag_inv = 1 / rowsum
    ones.data[:] = diag_inv
    return ones


def reverse_cuthill_mckee(mat):
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    reorder = reverse_cuthill_mckee(mat)
    return mat[reorder][:, reorder]


def pinv_left(A):
    return inv(A.T @ A) @ A.T


def pinv_right(A):
    return A.T @ inv(A @ A.T)


@njit
def inv_list_of_matrices(mats):
    results = np.zeros_like(mats)
    for i, mat in enumerate(mats):
        results[i] = np.linalg.inv(mat)
    return results


def inv_block_diag_3x3(mat):
    assert (mat.shape[0] % 3) == 0
    diag = mat.diagonal()
    a00 = diag[0::3]
    a11 = diag[1::3]
    a22 = diag[2::3]

    diag_m1 = mat.diagonal(k=-1)
    a10 = diag_m1[0::3]
    a21 = diag_m1[1::3]

    diag_m2 = mat.diagonal(k=-2)
    a20 = diag_m2[0::3]

    diag_p1 = mat.diagonal(k=1)
    a01 = diag_p1[0::3]
    a12 = diag_p1[1::3]

    diag_p2 = mat.diagonal(k=2)
    a02 = diag_p2[0::3]

    mats_3x3 = np.array(
        [
            [a00, a01, a02],
            [a10, a11, a12],
            [a20, a21, a22],
        ]
    ).transpose(2, 0, 1)
    mats_3x3_inv = inv_list_of_matrices(mats_3x3)
    return scipy.sparse.block_diag(mats_3x3_inv, format=mat.format)


def make_scaling(
    bmat: "BlockMatrixStorage", scale_groups: list[int] = None
) -> "BlockMatrixStorage":
    if scale_groups is None:
        scale_groups = bmat.active_groups[0]
    R = bmat.empty_container()
    assert R.active_groups[0] == R.active_groups[1]
    mats = []
    for i in R.active_groups[0]:
        tmp = bmat[i, i].mat
        vals = scipy.sparse.eye(*tmp.shape)
        if i in scale_groups:
            vals /= abs(tmp).max()
        mats.append(vals)
    R.mat = scipy.sparse.block_diag(mats, format="csr")
    return R


def make_scaling_1(
    bmat: "BlockMatrixStorage", scale_groups: dict[int, list[int]]
) -> "BlockMatrixStorage":
    R = bmat.empty_container()
    assert R.active_groups[0] == R.active_groups[1]
    mats = []
    for i in R.active_groups[0]:
        tmp = bmat[i, i].mat
        mats.append(scipy.sparse.eye(*tmp.shape))

    for src, targets in scale_groups.items():
        src_nrm = abs(bmat[src, src].mat).max()
        for target in targets:
            target_nrm = abs(bmat[target, target].mat).max()
            mats[target] *= src_nrm / target_nrm
    R.mat = scipy.sparse.block_diag(mats, format="csr")
    return R


class RearrangeAOS:
    def __init__(
        self, bmat: "BlockMatrixStorage", solve, together: list[list[int]] = None
    ):
        if together is None:
            together = [[i] for i in bmat.active_groups[0]]

        row_dofs = []
        col_dofs = []
        for groups in together:
            row_dofs.append(
                np.concatenate(
                    [
                        bmat.local_dofs_row[block]
                        for group in groups
                        for block in bmat.groups_to_blocks_row[group]
                    ]
                )
            )
            col_dofs.append(
                np.concatenate(
                    [
                        bmat.local_dofs_col[block]
                        for group in groups
                        for block in bmat.groups_to_blocks_col[group]
                    ]
                )
            )

        row_transformation = np.stack(row_dofs).ravel(order="F")
        self.Rrow = scipy.sparse.coo_matrix(
            (
                np.ones_like(row_transformation),
                (np.arange(row_transformation.size), row_transformation),
            )
        ).tocsr()
        col_transformation = np.stack(col_dofs).ravel(order="F")
        self.Rcol = scipy.sparse.coo_matrix(
            (
                np.ones_like(col_transformation),
                (col_transformation, np.arange(col_transformation.size)),
            )
        ).tocsr()
        self.solve = solve(self.Rrow @ bmat.mat @ self.Rcol)

    def dot(self, rhs) -> np.ndarray:
        rhs_transformed = self.Rrow @ rhs
        x_transformed = self.solve.dot(rhs_transformed)
        x = self.Rcol @ x_transformed
        return x


class BJacobiILU:
    def __init__(self, bmat: "BlockMatrixStorage"):
        self.bmat: "BlockMatrixStorage" = bmat
        self.shape = bmat.shape
        self.precs = [
            PetscHypreILU(self.bmat[[group]].mat) for group in bmat.active_groups[0]
        ]

    def dot(self, x: np.ndarray) -> np.ndarray:
        solution = np.zeros_like(x)
        for dofs_rhs, dofs_sol, prec in zip(
            *self.bmat.get_active_local_dofs(grouped=True), self.precs
        ):
            solution[dofs_sol] = prec.dot(x[dofs_rhs])
        return solution


def csr_to_petsc(mat: scipy.sparse.csr_matrix, bsize: int = 1) -> PETSc.Mat:
    """Convert a CSR matrix to a PETSc matrix.

    Parameters:
        mat: The matrix to convert.
        bsize: Block size of the matrix.

    Returns:
        The PETSc matrix representation of the given CSR matrix.

    """
    return PETSc.Mat().createAIJ(
        size=mat.shape,
        csr=(mat.indptr, mat.indices, mat.data),
        bsize=bsize,
    )


def petsc_to_csr(petsc_mat: PETSc.Mat) -> scipy.sparse.csr_matrix:
    """Convert a PETSc matrix to a CSR matrix.

    Parameters:
        petsc_mat: The matrix to convert.

    Returns:
        The CSR matrix representation of the given PETSc matrix.

    """
    indptr, indices, data = petsc_mat.getValuesCSR()
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())
