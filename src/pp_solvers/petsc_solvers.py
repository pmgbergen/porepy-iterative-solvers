"""This module provides shallow wrappers around PETSc's KSP (Krylov Subspace) solver
and related classes.
"""

from typing import Optional

import numpy as np
from petsc4py import PETSc


class PetscKrylovSolver:
    """Shallow wrapper around a PETSc KSP object."""

    def __init__(
        self,
        ksp: PETSc.KSP,
        assembly_config: Optional[dict] = None,
        petsc_options: Optional[dict] = None,
    ) -> None:
        """Initialize the solver with a PETSc KSP object. Optionally, can store
        `assembly_config` and `petsc_options`, which were used to produce this `ksp`.


        Parameters:
            ksp: A PETSc KSP object.
            assembly_config: A dictionary of options used during assembly of this ksp
                from Python code.
            petsc_options: A dictionary of PETSc CLI options used during assembly.

        """
        self.ksp: PETSc.KSP = ksp
        self.assembly_config: Optional[dict] = assembly_config
        """A dictionary of options used during assembly of this ksp from Python code."""
        self.petsc_options: Optional[dict] = petsc_options
        """A dictionary of PETSc CLI options used during assembly."""

        petsc_mat = ksp.getOperators()[0]
        self.petsc_x: PETSc.Vec = petsc_mat.createVecRight()
        """A PETSc vector for the solution approximation."""
        self.petsc_b: PETSc.Vec = petsc_mat.createVecLeft()
        """A PETSc vector for the right-hand side."""
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


class PcPythonPermutation:
    def __init__(self, perm: np.ndarray, block_size: int, inner_key: str):
        self.petsc_pc = PETSc.PC().create()
        self.petsc_pc.setOptionsPrefix(f"{inner_key}_")
        self.petsc_is_perm = PETSc.IS().createGeneral(perm.astype(np.int32))
        self.P_perm = PETSc.Mat()
        self.b = PETSc.Vec().create()
        self.bs = block_size
        self.b.setSizes(perm.size)
        self.b.setUp()

        self.is_set_up: bool = False

    def __del__(self):
        self.petsc_pc.destroy()
        self.petsc_is_perm.destroy()
        self.b.destroy()

    # Methods below are all petsc delegates (follows petsc api). Nothing special here.

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer) -> None:
        self.petsc_pc.view(viewer)

    def setFromOptions(self, pc: PETSc.PC) -> None:
        if self.is_set_up:
            # Set from options should be called only once.
            raise ValueError(
                "This class is not tested for operator reuse and may break your code."
            )
        _, P = pc.getOperators()
        self.P_perm = P.permute(self.petsc_is_perm, self.petsc_is_perm)
        self.P_perm.setBlockSize(self.bs)
        self.petsc_pc.setOperators(self.P_perm, self.P_perm)
        self.petsc_pc.setFromOptions()
        self.is_set_up = True

    def setUp(self, pc: PETSc.PC) -> None:
        self.petsc_pc.setUp()

    def reset(self, pc: PETSc.PC) -> None:
        self.petsc_pc.reset()
        self.P_perm.destroy()

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        b.copy(self.b)
        self.b.permute(self.petsc_is_perm)
        self.petsc_pc.apply(self.b, x)
        x.permute(self.petsc_is_perm, invert=True)
