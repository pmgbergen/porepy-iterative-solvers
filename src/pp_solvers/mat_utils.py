"""This module contains utility functions to operate with sparse matrices."""

from __future__ import annotations

import numpy as np
import porepy as pp
import scipy.sparse
from numba import njit

__all__ = [
    "csr_ones",
    "inv_block_diag",
]


def build_mechanics_near_null_space(
    model: pp.PorePyModel, include_sd=True, include_intf=True
):
    # YZ: this function will probably go somewhere else. It is currently not in use, but
    # we will definitely need it at some point.
    cell_center_array = []
    if include_sd:
        cell_center_array.append(model.mdg.subdomains(dim=model.nd)[0].cell_centers)
    if include_intf:
        cell_center_array.extend(
            [intf.cell_centers for intf in model.mdg.interfaces(dim=model.nd - 1)]
        )
    cell_centers = np.concatenate(cell_center_array, axis=1)

    x, y, z = cell_centers
    num_dofs = cell_centers.shape[1]

    null_space = []
    if model.nd == 3:
        vec = np.zeros((3, num_dofs))
        vec[0] = 1
        null_space.append(vec.ravel("F"))
        vec = np.zeros((3, num_dofs))
        vec[1] = 1
        null_space.append(vec.ravel("F"))
        vec = np.zeros((3, num_dofs))
        vec[2] = 1
        null_space.append(vec.ravel("F"))
        # # 0, -z, y
        vec = np.zeros((3, num_dofs))
        vec[1] = -z
        vec[2] = y
        null_space.append(vec.ravel("F"))
        # z, 0, -x
        vec = np.zeros((3, num_dofs))
        vec[0] = z
        vec[2] = -x
        null_space.append(vec.ravel("F"))
        # -y, x, 0
        vec = np.zeros((3, num_dofs))
        vec[0] = -y
        vec[1] = x
        null_space.append(vec.ravel("F"))
    elif model.nd == 2:
        vec = np.zeros((2, num_dofs))
        vec[0] = 1
        null_space.append(vec.ravel("F"))
        vec = np.zeros((2, num_dofs))
        vec[1] = 1
        null_space.append(vec.ravel("F"))
        # -x, y
        vec = np.zeros((2, num_dofs))
        vec[0] = -x
        vec[1] = y
        null_space.append(vec.ravel("F"))
    else:
        raise ValueError

    return np.array(null_space)


def csr_ones(n: int) -> scipy.sparse.csr_matrix:
    """Constructs a square sparse matrix with ones on the diagonal."""
    return scipy.sparse.eye(n, format="csr")


def inv_block_diag(mat, nd: int):
    """Inverses the small block matrices (nd x nd) located on the matrix diagonal.
    Ignores the nonzero entities outside the small block matrices.

    """
    # YZ: The equivalent function in porepy is not truly equivalent, because this one
    # ignores the nonzero entries outside the block diagonal, and PorePy raises an
    # exception. Thus it cannot be easily replaced.
    if nd == 1:
        return _extract_diag_inv(mat)
    if nd == 2:
        return _inv_block_diag_2x2(mat)
    if nd == 3:
        return _inv_block_diag_3x3(mat)
    raise ValueError(f"{nd = } not supported.")


def _extract_diag_inv(mat, eliminate_zeros=False):
    diag = mat.diagonal()
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    if eliminate_zeros:
        diag[abs(diag) < 1e-30] = 1
    diag_inv = 1 / diag
    ones.data[:] = diag_inv
    return ones


def _inv_block_diag_2x2(mat):
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


@njit
def _inv_list_of_matrices(mats):
    results = np.zeros_like(mats)
    for i, mat in enumerate(mats):
        results[i] = np.linalg.inv(mat)
    return results


def _inv_block_diag_3x3(mat):
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
    mats_3x3_inv = _inv_list_of_matrices(mats_3x3)
    return scipy.sparse.block_diag(mats_3x3_inv, format=mat.format)
