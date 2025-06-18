"""This module contains schemes, e.g., recepies for constructing a PETSc solver."""

from __future__ import annotations
from itertools import chain

from time import time
from pprint import pprint

from warnings import warn
from pathlib import Path

from enum import Enum
import numpy as np
import scipy.sparse as sps
from typing import Callable
from dataclasses import dataclass
import porepy as pp
from abc import ABC, abstractmethod

from .block_matrix import BlockMatrixStorage
from .full_petsc_solver import (
    construct_is,
    PetscKSPScheme,
    insert_petsc_options,
    LinearTransformedScheme,
    PcPythonPermutation,
)
from .fixed_stress import make_fs_analytical_slow_new
from .thm_solver import make_pt_permutation, get_dofs_of_groups

from . import hm_solver
from .iterative_solver import (
    get_equations_group_ids,
    get_variables_group_ids,
)
from .mat_utils import csr_ones, inv_block_diag, csr_to_petsc

from petsc4py import PETSc
from .dof_manager import DofManager


import equation_variable_groups as groups
from .preconditioners import (
    SinglePhysicsPreconditioner,
    CompositePreconditioner,
)


__all__ = [
    "MultiPhysicsPreconditioner",
    "IterativeSolverMixin",
]

"""Below are methods that are used to create specific schemes for different equations.
Note that these consider PETSc configurations, and have no responsibility for
taking care of equations etc. (CURRENT IMPLEMENTATION IS NOT RIGHT). This means they are
essentially bearers of options for the solver.
"""


class EquationNames(Enum):
    """Enum for the names of the equations in the model."""

    MASS_BALANCE = "mass_balance_equation"
    MASS_BALANCE_MATRIX = "mass_balance_equation"
    MASS_BALANCE_FRACTURES = "mass_balance_equation"
    MASS_BALANCE_INTERSECTIONS = "mass_balance_equation"
    ENERGY_BALANCE = "energy_balance_equation"
    ENERGY_BALANCE_MATRIX = "energy_balance_equation"
    ENERGY_BALANCE_FRACTURES = "energy_balance_equation"
    ENERGY_BALANCE_INTERSECTIONS = "energy_balance_equation"
    INTERFACE_DARCY_FLUX = "interface_darcy_flux_equation"

    INTERFACE_ENTHALPY_FLUX = "interface_enthalpy_flux_equation"
    INTERFACE_FOURIER_FLUX = "interface_fourier_flux_equation"

    MECHANICS = "momentum_balance_equation"
    INTERFACE_FORCE_BALANCE = "interface_force_balance_equation"
    CONTACT = "contact_mechanics_equation"
    CONTACT_NORMAL = "normal_fracture_deformation_equation"
    CONTACT_TANGENTIAL = "tangential_fracture_deformation_equation"


def transform_contact_block(J, row_group: int, col_group: int, nd: int):
    """Assemble the right linear transformation."""
    # Sorted according to groups. If not done, the matrix can be in porepy order,
    # which does not guarantee that diagonal groups are truly on diagonals.
    Qright = J.empty_container()[:]

    if row_group not in J.active_groups[0]:
        Qright.mat = csr_ones(Qright.shape[0])
        return Qright

    J55 = J[col_group, col_group].mat

    J55_inv = inv_block_diag(J55, nd=nd, lump=False)

    Qright.mat = csr_ones(Qright.shape[0])

    J54 = J[col_group, row_group].mat

    tmp = -J55_inv @ J54
    Qright[col_group, row_group] = tmp
    return Qright


def scale_energy_transform(J, row_groups: list[int], model: pp.PorePyModel):
    """Assemble the right linear transformation for scaling energy fluxes."""
    # Sorted according to groups. If not done, the matrix can be in porepy order,
    # which does not guarantee that diagonal groups are truly on diagonals.
    Q = J.empty_container()[:]

    subdomains = model.mdg.subdomains()
    vols = 1.0 / model.equation_system.evaluate(model.specific_volume(subdomains))

    Q.mat = sps.eye(Q.shape[0], format="csr")
    if len(subdomains) == 0:
        # No subdomains, hence no scaling.
        return Q
    Q[row_groups] = sps.diags(vols, format="csr")

    return Q


@dataclass
class LinearSolverComponents:
    dof_manager: DofManager
    preconditioner: MultiPhysicsPreconditioner
    ksp_factory: PetscKSPScheme


class IterativeSolverMixin:
    # Temporary storage for the iterative solver results.
    _petsc_converged_reason = []
    _krylov_iters = []
    _construction_time = []


class LinearSolverComponents:
    dof_manager: DofManager
    preconditioner: MultiPhysicsPreconditioner
    ksp_factory: PetscKSPScheme


class IterativeSolverMixin:
    # Temporary storage for the iterative solver results.
    _petsc_converged_reason = []
    _krylov_iters = []
    _construction_time = []
    _solve_time = []

    def solve_linear_system(self) -> None:
        # Check for NaN or Inf in the RHS.
        mat, rhs = self.linear_system
        if np.any(np.isnan(rhs) | np.isinf(rhs)):
            raise ValueError("RHS contains NaN or Inf values")

        # By default, print the residual information to screen (ksp_monitor=None).
        solver_options = self.params["linear_solver"].get("options", {})
        ksp_factory = self._solver_components.ksp_factory
        # solver = ksp_factory.make_solver(self.bmat, solver_options)
        t0 = time()
        try:
            solver = ksp_factory.make_solver(self.bmat, solver_options)
        except Exception as e:
            raise RuntimeError(
                "Failed to create solver with the provided preconditioner."
            ) from e

        self._construction_time.append(time() - t0)

        rhs_loc = self.bmat.project_rhs_to_local(rhs)
        t0 = time()
        x_loc = solver.solve(rhs_loc)
        self._solve_time.append(time() - t0)

        info = solver.ksp.getConvergedReason()
        if info <= 0:
            raise RuntimeError(
                f"Solver did not converge. Reason: {info}. "
                "Check the solver options and the problem setup."
            )

        x = self.bmat.project_solution_to_global(x_loc)
        self._petsc_converged_reason.append(info)
        self._krylov_iters.append(len(solver.get_residuals()))

        return np.atleast_1d(x)

    def assemble_linear_system(self):
        super().assemble_linear_system()  # type: ignore[misc]

        dof_manager = self._solver_components.dof_manager
        # Get the linear system from the equation system.

        # TODO: Replace this with a different type of plugin
        mat, rhs = self.linear_system

        # Apply the `contact_permutation`.
        mat = mat[dof_manager.eq_rows_permutation(self)]
        rhs = rhs[dof_manager.eq_rows_permutation(self)]

        bmat = BlockMatrixStorage(
            mat=mat,
            global_dofs_row=dof_manager.eq_dofs_by_blocks(self),
            global_dofs_col=dof_manager.var_dofs_by_blocks(self),
            groups_to_blocks_row=dof_manager.equation_groups(self),
            groups_to_blocks_col=dof_manager.variable_groups(self),
            group_names_row=dof_manager.equation_names(self),
            group_names_col=dof_manager.variable_names(self),
        )

        # TODO: Figure out if the [:] is really needed.
        self.bmat = bmat[:]

    def _initialize_linear_solver(self):
        # Set up preconditioner.
        precond_factory: Callable[[], MultiPhysicsPreconditioner] = self.params[
            "linear_solver"
        ]["preconditioner_factory"]
        if precond_factory is None:
            raise ValueError("Preconditioner factory is not set")
        precond_list: list[SinglePhysicsPreconditioner] = precond_factory()

        ordering_list = [precond.group() for precond in precond_list]

        dof_manager = DofManager(
            self.equation_system, self, ordering_list, precond_list
        )
        precond = MultiPhysicsPreconditioner(precond_list, dof_manager, self)

        contact_ind = dof_manager.identify_contact_group(self)

        ksp_factory = PetscKSPScheme(preconditioner=precond)
        contact_transform, thermal_transform = None, None
        if contact_ind > -1:
            # If there is a contact group, we need to use a linear solver that takes
            # care of potential singularities in the contact block.
            u_intf_ind = dof_manager.identify_u_intf_group(self)

            contact_transform = [
                lambda bmat: transform_contact_block(
                    bmat, contact_ind, u_intf_ind, self.nd
                )
            ]
        if any(
            [name.startswith("energy") for name in dof_manager.equation_names(self)]
        ):
            row = dof_manager.identify_energy_balance_group(self)
            thermal_transform = [
                lambda bmat: scale_energy_transform(bmat, row_groups=row, model=self)
            ]

        if contact_transform is not None or thermal_transform is not None:
            solver_factory = LinearTransformedScheme(
                nd=self.nd,
                contact_group=contact_ind,
                u_intf_group=u_intf_ind,
                # preconditioner=precond,
                inner=ksp_factory,
                right_transformations=contact_transform,
                left_transformations=thermal_transform,
            )

        else:
            # A standard KSP solver will do.
            solver_factory = ksp_factory

        solver_components = LinearSolverComponents(
            dof_manager=dof_manager,
            preconditioner=precond,
            ksp_factory=solver_factory,
        )
        self._solver_components = solver_components

    def save_matrix_state(self):
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = "matrix"
        print("Saving matrix", name)
        mat_id = f"{name}.npz"
        rhs_id = f"{name}_rhs.npy"
        sps.save_npz(save_path / mat_id, self.bmat.mat)
        np.save(save_path / rhs_id, rhs)
