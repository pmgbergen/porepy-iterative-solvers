from __future__ import annotations

import sys

import numpy as np
import petsc4py
import scipy.sparse
from petsc4py import PETSc

# This is the place where the user has a change to pass command line options to petsc.
# Before calling init, all petsc objects are unavailable, so this is a reasonable place
# to initialize it.p
petsc4py.init(sys.argv)  


__all__ = [
    "csr_to_petsc",
    "petsc_to_csr",
    "clear_petsc_options",
]


def csr_to_petsc(mat: scipy.sparse.csr_matrix, bsize: int = 1) -> PETSc.Mat:
    """Convert a CSR matrix to a PETSc matrix.

    Parameters:
        mat: The matrix to convert.
        bsize: Block size of the matrix.

    Returns:
        The PETSc matrix representation of the given CSR matrix.

    """
    assert mat.format == 'csr'
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


def clear_petsc_options() -> PETSc.Options:
    """Options is a singletone. This ensures that no unwanted options from some previous
    setup reach the current setup."""
    options = PETSc.Options()

    for key in options.getAll():
        options.delValue(key)
    return options