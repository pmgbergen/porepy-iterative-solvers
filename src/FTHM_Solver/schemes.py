"""This module contains schemes, e.g., recepies for constructing a PETSc solver."""

from __future__ import annotations

from time import time

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


__all__ = [
    "MassBalanceGroup",
    "InterfaceDarcyFluxGroup",
    "MechanicsGroup",
    "ContactGroup",
    "DofManager",
    "SinglePhysicsPreconditioner",
    "InterfaceDarcyFluxPreconditioner",
    "MassBalancePreconditioner",
    "MechanicsPreconditioner",
    "ContactPreconditioner",
    "MultiPhysicsPreconditioner",
    "IterativeSolverMixin",
    "mass_balance_factory",
    "momentum_balance_factory",
    "hm_factory",
    "thm_factory",
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


class AbstractGroup(ABC):
    """
    Abstract class for defining a group of equations and variables. This serves two
    purposes:
        1. To define pairs of equations and variables that should be grouped together,
           and thereby define the diagonal blocks of the linear system.
        2. To define groups of equations that will be treated together by the iterative
           solver. The can be used to group equations of the same type (e.g., mass
           balance) on different subdomains, or to group equations of different type,
           but that still should be solved together.

    """

    @abstractmethod
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        pass

    @abstractmethod
    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        pass

    @abstractmethod
    def equation_names(self, model) -> list[str]:
        pass

    @abstractmethod
    def variable_names(self, model) -> list[str]:
        pass


class MassBalanceGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains()
        return [[(EquationNames.MASS_BALANCE.value, subdomains)]]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains()
        return [[model.pressure(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.MASS_BALANCE.value]

    def variable_names(self, model) -> list[str]:
        return [model.pressure_variable]


def _split_subdomains_by_dimension(model: pp.PorePyModel):
    matrix_subdomains = model.mdg.subdomains(dim=model.nd)
    fracture_subdomains = model.mdg.subdomains(dim=model.nd - 1)
    intersection_subdomains = [
        sd for sd in model.mdg.subdomains() if sd.dim < model.nd - 1
    ]
    return matrix_subdomains, fracture_subdomains, intersection_subdomains


class MassBalanceDimSplitGroup(AbstractGroup):
    """Group for the mass balance equation, with matrix, fractures and intersections
    split into different groups. This is needed for fixed-stress type preconditioners,
    where the stabilization term differs according to the dimension of the subdomains.
    """

    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [(EquationNames.MASS_BALANCE_MATRIX.value, matrix_subdomains)],
            [(EquationNames.MASS_BALANCE_FRACTURES.value, fracture_subdomains)],
            [
                (
                    EquationNames.MASS_BALANCE_INTERSECTIONS.value,
                    intersection_subdomains,
                )
            ],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [model.pressure(matrix_subdomains)],
            [model.pressure(fracture_subdomains)],
            [model.pressure(intersection_subdomains)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.MASS_BALANCE_MATRIX.value,
            EquationNames.MASS_BALANCE_FRACTURES.value,
            EquationNames.MASS_BALANCE_INTERSECTIONS.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.pressure_variable + "_matrix",
            model.pressure_variable + "_fractures",
            model.pressure_variable + "_intersections",
        ]


class EnergyBalanceDimSplitGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [(EquationNames.ENERGY_BALANCE_MATRIX.value, matrix_subdomains)],
            [(EquationNames.ENERGY_BALANCE_FRACTURES.value, fracture_subdomains)],
            [
                (
                    EquationNames.ENERGY_BALANCE_INTERSECTIONS.value,
                    intersection_subdomains,
                )
            ],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        matrix_subdomains, fracture_subdomains, intersection_subdomains = (
            _split_subdomains_by_dimension(model)
        )
        return [
            [model.temperature(matrix_subdomains)],
            [model.temperature(fracture_subdomains)],
            [model.temperature(intersection_subdomains)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.ENERGY_BALANCE_MATRIX.value,
            EquationNames.ENERGY_BALANCE_FRACTURES.value,
            EquationNames.ENERGY_BALANCE_INTERSECTIONS.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.temperature_variable + "_matrix",
            model.temperature_variable + "_fractures",
            model.temperature_variable + "_intersections",
        ]


class InterfaceDarcyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [[(EquationNames.INTERFACE_DARCY_FLUX.value, interfaces)]]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [[model.interface_darcy_flux(interfaces)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.INTERFACE_DARCY_FLUX.value]

    def variable_names(self, model) -> list[str]:
        return [model.interface_darcy_flux_variable]


class InterfaceEnthalpyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [
            [(EquationNames.INTERFACE_ENTHALPY_FLUX.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_enthalpy_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_ENTHALPY_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_enthalpy_flux_variable,
        ]


class InterfaceFourierFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [
            [(EquationNames.INTERFACE_FOURIER_FLUX.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_fourier_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_FOURIER_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_fourier_flux_variable,
        ]


class InterfaceMassEnergyFluxGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        interfaces = model.mdg.interfaces()
        return [
            [(EquationNames.INTERFACE_ENTHALPY_FLUX.value, interfaces)],
            [(EquationNames.INTERFACE_FOURIER_FLUX.value, interfaces)],
            [(EquationNames.INTERFACE_DARCY_FLUX.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        interfaces = model.mdg.interfaces()
        return [
            [model.interface_enthalpy_flux(interfaces)],
            [model.interface_fourier_flux(interfaces)],
            [model.interface_darcy_flux(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.INTERFACE_ENTHALPY_FLUX.value,
            EquationNames.INTERFACE_FOURIER_FLUX.value,
            EquationNames.INTERFACE_DARCY_FLUX.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [
            model.interface_enthalpy_flux_variable,
            model.interface_fourier_flux_variable,
            model.interface_darcy_flux_variable,
        ]


class MechanicsGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)

        # Define two groups of equations, one for momentum balance in the matrix and one
        # for force balance on the highest-dimensional interfaces. The mechanics
        # preconditioner will treat these groups jointly.
        return [
            [(EquationNames.MECHANICS.value, subdomains)],
            [(EquationNames.INTERFACE_FORCE_BALANCE.value, interfaces)],
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)

        # Define two groups of variables, one for the displacement in the matrix and one
        # for the interface displacement.
        return [
            [model.displacement(subdomains)],
            [model.interface_displacement(interfaces)],
        ]

    def equation_names(self, model) -> list[str]:
        return [
            EquationNames.MECHANICS.value,
            EquationNames.INTERFACE_FORCE_BALANCE.value,
        ]

    def variable_names(self, model) -> list[str]:
        return [model.displacement_variable, model.interface_displacement_variable]


class ContactGroup(AbstractGroup):
    def equation_groups(self, model: pp.PorePyModel) -> list[list[tuple[str, list]]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        # Define a single group of equations to be solved together: The normal and
        # tangential deformation equations for the contact mechanics.
        return [
            [
                (EquationNames.CONTACT_NORMAL.value, subdomains),
                (EquationNames.CONTACT_TANGENTIAL.value, subdomains),
            ]
        ]

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        # There is a single group of variables for the contact mechanics, which is the
        # contact traction.
        return [[model.contact_traction(subdomains)]]

    def equation_names(self, model) -> list[str]:
        return [EquationNames.CONTACT.value]

    def variable_names(self, model) -> list[str]:
        return [model.contact_traction_variable]


class DofManager:
    """Takes care of translation of blocks and groups (from EquationSystem format) to
    block indices, as well as grouping the fine-scale dofs. Also reordering related to
    the contact problem.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so.
    """

    def __init__(
        self,
        equation_system: pp.EquationSystem,
        model: pp.PorePyModel,
        orderings: list[AbstractGroup],
        solvers: list[SinglePhysicsPreconditioner],
    ):
        self._equation_system = equation_system
        self._orderings = orderings
        eq_groups, var_goups, slv_groups, name_to_group_ind = (
            self._process_block_information(model, solvers)
        )
        self._equation_groups = eq_groups
        self._variable_groups = var_goups
        self._solver_groups = slv_groups
        self._name_to_group_indices = name_to_group_ind

    @property
    def groups(self) -> list[AbstractGroup]:
        """Return the groups of equations and variables."""
        return self._orderings

    def petsc_is(
        self,
        current_solver: AbstractGroup,
        other_solver: list[AbstractGroup],
        bmat: BlockMatrixStorage,
    ):
        # Not sure if this belongs here, but it is tempting to put it here and not in
        # the composer.

        # Indices of the block ids
        current_id = self.blocks_of_solver(current_solver)

        other_id = []
        for group in other_solver:
            # Get the block id for the group.
            other_id += self.blocks_of_solver(group)

        current_is = construct_is(bmat, current_id)
        other_is = construct_is(bmat, other_id)
        return current_is, other_is

    def variable_groups(
        self, model: pp.PorePyModel
    ) -> list[list[pp.ad.MixedDimensionalVariable]]:
        return self._variable_groups

    def equation_groups(self, model: pp.PorePyModel) -> list[list[int]]:
        return self._equation_groups

    def blocks_of_solver(self, solver: SinglePhysicsPreconditioner) -> int:
        return self._solver_groups[solver]

    def _process_block_information(self, model: pp.PorePyModel, solvers):
        """Construct groups of equations, variables and solvers from the orderings and
        solvers.

        This method should be called as part of the DofManager initialization. It
        process information on the orderings and solvers to arrive at the following:

        1. A list of equation groups, where each group defines a set of equations (using
           the block indices of the BlockMatrixStorage) that will be preconditioned
           together. Together, the list spans the full set of equations in the model at
           hand.
        2. A list of variable groups, where each group defines a set of variables that
           will be preconditioned together. Together, the list spans the full set of
           variables in the model at hand.
        3. A dictionary that maps each solver to the indices of the equation groups it
           will solve. This is used to identify which equations are solved by which
           solver, and to construct the appropriate preconditioner for each solver.

        """
        # The construction consists of two main steps: First, iterate over the orderings
        # and gather the equation and variable groups defined by them. This may involve
        # uniquifying the groups (relevant if the preconditioner is a PETSc Composite,
        # which may contain several overlapping groups). In this process, we also
        # construct the map from solvers to block indices. Second, the groups of
        # variables and equations are expanded from a lists of domains (PorePy
        # subdomains and interfaces) into individual variables and equations. For the
        # equations, we also do some reordering needed to merge the contact equations in
        # the normal and tangential directions into a single block, and thereby reveal
        # the underlying block diagonal structure of this equation.

        # Data structures for variables, equations and the solver map.
        var_groups = []
        equations_by_name = []
        solver_indices = {}
        # Counter of block indices, used to assign block indices to the solver map.
        counter = 0

        # Iterate over the orderings, gather the equation an variable groups define
        # there. It is assumed that the orderings define non-intersecting sets of
        # equations and variables that together span the system of equations to be
        # solved. However, an ordering can return a list, corresponding to a multistage
        # preconditioner (a Composite preconditioner in PETSc terminology). This list
        # may contain multiple intersecting groups, that needs to be parsed into a
        # single, non-intersecting set of equations and variables.
        for group, slv in zip(self._orderings, solvers):
            if isinstance(group, list):
                # This is a list of groups, which may contain identical items. First
                # gather them all.
                #
                # First, make sure this is a composite preconditioner; this is a tacit
                # assumption of the below parsing. Dealing with anything else would
                # require a more structured approach to the parsing, but EK has neither
                # the imagination nor the test cases needed to do so now.
                assert isinstance(slv, CompositePreconditioner)

                # Tacitly assump that the group is a list of lists of AbstractGroup. If
                # we hit a third nested level, we will need to do recursion of sorts.
                tmp_var_groups = []
                tmp_equations_by_name = []
                for g in group:
                    tmp_var_groups += g.variable_groups(model)
                    tmp_equations_by_name += g.equation_groups(model)

                # Find the indices of the subsets that will define a unique set of
                # variables and equations. By assumption, the variables and equations in
                # the tmp_x lists match, so that we can use any of them to find indices
                # that define a unique sublist. It is by far simplest to use the
                # variables, since we can rely on their hash values to find the sublist
                # (while the equation list is a confused mess of lists and tuples, which
                # fortunately works).

                # The implementation below assumes that each variable group contains a
                # single variable. Expanding this should be doable, but it has not yet
                # been necessary.
                assert all(len(x) == 1 for x in tmp_var_groups)

                # Find the sorting indices of the unique variable groups.
                hash_values = [hash(item[0]) for item in tmp_var_groups]
                _, sorting_indices = np.unique(hash_values, return_index=True)
                # Sort the indices to ensure a consistent order.
                sorting_indices.sort()

                # Extract unique sublists.
                groups_loc = []
                vars_loc = []
                for ind in sorting_indices:
                    vars_loc.append(tmp_var_groups[ind])
                    groups_loc.append(tmp_equations_by_name[ind])

                for sub_solver in slv.solvers:
                    # If the sub-solver is a list, it is by itself a filedsplit
                    # preconditioner. Treat every sub-solver within the fieldsplit as a
                    # separate solver, and add it to the solver indices.
                    if isinstance(sub_solver, list):
                        for ss in sub_solver:
                            solver_indices[ss] = []
                            # There could be deeper recursion levels here, which we may
                            # need to deal with by some recursive approach, but we
                            # ignore that possibility for now.
                            assert isinstance(ss, SinglePhysicsPreconditioner)
                            ss_vars = ss.group().variable_groups(model)
                            assert isinstance(ss_vars, list)
                            sub_vars = [ss_vars[i][0] for i in range(len(ss_vars))]

                            # Loop over all variables associated with this subsolver.
                            # Find it among the unique variable set (if it is not found,
                            # something is seriously worng), and add the block index to
                            # the list associated with the subsolver.
                            for var in sub_vars:
                                loc_id = int(
                                    np.where(
                                        np.array(hash_values)[sorting_indices]
                                        == hash(var)
                                    )[0][0]
                                )
                                solver_indices[ss].append(loc_id + counter)
                    else:
                        # Assume here that the sub-solver is a
                        # SinglePhysicsPreconditioner. If we at some point need nested
                        # composite preconditioners, something will go wrong here. There
                        # are surely other cases that can break this as well.
                        solver_indices[sub_solver] = [
                            counter + int(i) for i in sorting_indices
                        ]

            else:
                # This is a single group, we can add its variables and equations.
                groups_loc = group.equation_groups(model)
                vars_loc = group.variable_groups(model)

            # Append the groups to the main lists, and update the solver indices.
            equations_by_name += groups_loc
            var_groups += vars_loc
            # Also take note that the solver slv is associated with all indices in the
            # local groups. A composite preconditioner will by this be associated with
            # the entire set of indices *in addition to* the mapping of individual
            # solvers (see the convoluted for-loop above). This double registration is
            # needed for the selection of blocks to work as intended. For pure
            # (non-composite) block preconditioners, we simply map the preconditioner to
            # the registred indices.
            solver_indices[slv] = list(range(counter, counter + len(groups_loc)))
            counter += len(groups_loc)

        # Done with the first step. Next, expand the groups by calling on relevant
        # helper methods.
        var_groups_by_number = get_variables_group_ids(model, var_groups)
        equation_groups_by_number = get_equations_group_ids(model, equations_by_name)

        # Permute the contact equations if present. NOTE: It would have been preferrable
        # to use the name_to_group_indices map, constructed just below, to identify the
        # contact group, but this is not yet available at this point. Refactoring may be
        # a good idea.
        contact_group = self.identify_contact_group(model)
        if contact_group == -1:
            reordered_groups = equation_groups_by_number
        else:
            reordered_groups = self._correct_contact_equations_groups(
                model, equation_groups_by_number, contact_group
            )

        name_to_group_index_map = {}
        for i, item in enumerate(equations_by_name):
            name = item[0][0]
            # Add the equation name to the group.
            if name not in name_to_group_index_map:
                name_to_group_index_map[name] = []
            name_to_group_index_map[name].append(i)

        return (
            reordered_groups,
            var_groups_by_number,
            solver_indices,
            name_to_group_index_map,
        )

    def equation_names(self, model):
        names = []
        for group in self._orderings:
            if isinstance(group, list):
                # If the group is a list, we assume it contains multiple groups.
                # TODO: Unification needed here.
                for g in group:
                    names += g.equation_names(model)
            else:
                names += group.equation_names(model)
        return names

    def variable_names(self, model):
        names = []
        for group in self._orderings:
            if isinstance(group, list):
                # If the group is a list, we assume it contains multiple groups.
                # TODO: Unification needed here.
                for g in group:
                    names += g.variable_names(model)
            else:
                names += group.variable_names(model)
        return names

    def identify_contact_group(self, model):
        # Identify the contact group in the equation groups
        for i, group in enumerate(self._orderings):
            if len(group.equation_groups(model)) == 0:
                continue
            for block in group.equation_groups(model):
                if block[0][0] == "normal_fracture_deformation_equation":
                    return i
        return -1

    def identify_u_intf_group(self, model):
        # Identify the interface group in the equation groups
        i = 0
        for group in self._orderings:
            if isinstance(group, list):
                for sub_group in group:
                    if len(sub_group.variable_groups(model)) == 0:
                        continue
                    for var in sub_group.variable_groups(model):
                        if var[0].name == model.interface_displacement_variable:
                            return i
            else:
                if len(group.variable_groups(model)) == 0:
                    continue
                for var in group.variable_groups(model):
                    if var[0].name == model.interface_displacement_variable:
                        return i
                    else:
                        i += 1
        return -1

    def identify_energy_balance_group(self, model):
        return self._name_to_group_indices[EquationNames.ENERGY_BALANCE.value]

    def eq_dofs_by_blocks(self, model):
        """Get the equation dofs for the model, in the form of a list of numbers,
        one per equation-domain pair. If the contact group is present, it will be
        reordered so that the normal and tangential equations for each fracture cell
        form a diganol block.
        """
        eq_dofs: list[np.ndarray] = []
        offset = 0
        for data in model.equation_system._equation_image_space_composition.values():
            local_offset = 0
            for dofs in data.values():
                eq_dofs.append(dofs + offset)
                local_offset += len(dofs)
            offset += local_offset

        contact_group = self.identify_contact_group(model)
        if contact_group > -1:
            # If there is no contact group, return the original equation dofs.
            return self._correct_contact_eq_dofs(model, eq_dofs, contact_group)

        return eq_dofs

    def _correct_contact_eq_dofs(
        self, model, unpermuted_eq_dofs: list[np.ndarray], contact_group: int
    ) -> list[np.ndarray | None]:
        """Rearrange the unknowns (row indices) so that the contact equations are in a
        cell-wise block structure.

        Parameters:
            unpermuted_eq_dofs: The unpermuted equation degrees of freedom.
            contact_group: The group index of the contact mechanics equations.

        Returns:
            The corrected equation degrees of freedom.

        See also:
            _correct_contact_equations_groups for rearrane of the equation blocks
                related to contact (as opposed to the individual dofs handled here).

        """
        # Short cut if no contact mechanics, hence no reordering.
        if len(self.equation_groups(model)[contact_group]) == 0:
            # Ignore mypy error, list[np.ndarray] is a subset of list[np.ndarray |
            # None].
            return unpermuted_eq_dofs  # type: ignore[return-value]

        # We assume that normal equations go first. TODO: Can we make this more robust,
        # or else put an assert here.
        normal_blocks = self.equation_groups(model)[contact_group]
        num_fracs = len(model.mdg.subdomains(dim=model.nd - 1))

        # EK: I believe this is an assumption that the tangential equations are right
        # after the normal equations.
        all_contact_blocks = [
            nb + i * num_fracs for i in range(2) for nb in normal_blocks
        ]

        eq_dofs_corrected: list[np.ndarray | None] = []
        # Add all equations that are not contact equations without any changes.
        for i, x in enumerate(unpermuted_eq_dofs):
            if i not in all_contact_blocks:
                eq_dofs_corrected.append(x)
            elif i in normal_blocks:
                eq_dofs_corrected.append(None)

        offset = unpermuted_eq_dofs[normal_blocks[0]][0]
        for nb in normal_blocks:
            # Create indices for the normal and tangential components of the contact.
            # There will be model.nd equations for each block.
            inds = offset + np.arange(unpermuted_eq_dofs[nb].size * model.nd)
            offset = inds[-1] + 1
            eq_dofs_corrected[nb] = np.array(inds)

        return eq_dofs_corrected

    def _correct_contact_equations_groups(
        self,
        model: pp.PorePyModel,
        equation_groups: list[list[int]],
        contact_group: int,
    ) -> list[list[int]]:
        """The block ordering from PorePy assigns different block indices to the normal
        and tangential components of the contact equations. This method corrects this
        indexing by assigning a single block index for each fracture.

        The method further adjusts the indices of the other equation groups to account
        for the reduced number of blocks.

        Parameters:
            equation_groups: The uncorrected equation groups.
            contact_group: The group index of the contact mechanics equations.

        Returns:
            The corrected equation groups.

        See also:
            _correct_contact_eq_dofs for rearrane of the individual dofs related to
                contact (as opposed to the equation blocks handled here).

        """
        if len(equation_groups[contact_group]) == 0:
            return equation_groups

        # Create a copy of the equation groups to avoid modifying the original.
        eq_groups_corrected = [x.copy() for x in equation_groups]

        num_fracs = len(model.mdg.subdomains(dim=model.nd - 1))
        # Index of the first block after the contact group. This and all subsequent
        # indexes will be reduced by the number of fractures (e.g., the number of
        # block equations that have been removed).
        block_after_contact = max(equation_groups[contact_group]) + 1

        # Change the number of blocks in the contact group to the number of fractures,
        # since we have merged the normal and tangential components.
        eq_groups_corrected[contact_group] = equation_groups[contact_group][:num_fracs]

        # For all other groups with block index after the contact group, reduce the
        # block index by the number of fractures.
        for blocks in eq_groups_corrected:
            for i in range(len(blocks)):
                if blocks[i] >= block_after_contact:
                    blocks[i] -= num_fracs

        return eq_groups_corrected

    def var_dofs_by_blocks(self, model) -> list[np.ndarray]:
        """Variable degrees of freedom (columns of the Jacobian) in the PorePy order
        (how they are arranged in the model).

        Returns:
            List of numpy arrays. Each array contains the global degrees of freedom for
                one variable on one grid and provides the fine-scale (actual column
                indices) of the variable.

        """
        var_dofs: list[np.ndarray] = []
        for var in model.equation_system.variables:
            var_dofs.append(model.equation_system.dofs_of([var]))
        return var_dofs

    def eq_rows_permutation(self, model):
        """Get a permutation vector for the full linear system of equations. This is
        used to reorder the equations so that the contact equations for single fracture
        cells form a diagonal block.

        If no contact group is present, the permutation vector is linear.

        See also eq_dofs_by_blocks, which is used to reorder contact equations within
        the equation block format.
        """
        contact_group = self.identify_contact_group(model)
        # If there is no contact group, return the original equation groups.
        if contact_group == -1:
            return np.arange(model.equation_system.num_dofs())

        return self.make_reorder_contact(model, contact_group)

    def make_reorder_contact(
        self, model: pp.PorePyModel, contact_group: int
    ) -> np.ndarray:
        """Permutate the contact mechanics equations to a cell-wise block structure.

        The PorePy arrangement is:

            [C_n^0, C_n^1, ..., C_n^K, C_y^0, C_z^0, C_y^1, C_z^1, ..., C_z^K, C_z^k],

        where `C_n` is a normal component, `C_y` and `C_z` are two tangential
        components. The superscript corresponds to cell index. We permute it to

            `[C_n^0, C_y^0, C_z^0, ..., C_n^K, C_y^K, C_z^K]`.

        Parameters:
            model: The PorePy model.
            contact_group: The group index of the contact mechanics equations.

        Raises:
            ValueError: If the model dimension is not 2 or 3.

        Returns:


        """
        reorder = np.arange(model.equation_system.num_dofs())

        # Short cut if no contact mechanics, hence no reordering.
        if len(self.equation_groups(model)[contact_group]) == 0:
            return reorder

        # Get the (fine-scale, not block(!)) dofs of the contact mechanics equations.
        dofs_contact = np.concatenate(
            [
                self.eq_dofs_by_blocks(model)[i]
                for i in self.equation_groups(model)[contact_group]
            ]
        )

        # The start and end indices of all contact mechanics equations.
        dofs_contact_start = dofs_contact[0]
        dofs_contact_end = dofs_contact[-1] + 1

        # The number of cells in the contact mechanics equations.
        num_contact_cells = len(dofs_contact) // model.nd

        # 2d and 3d have respectively 1 and 2 tangential components, hence the branch.
        if model.nd == 2:
            # Rearrange the dofs into cell-wise blocks.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells:]
            reorder[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1]
            ).ravel("F")
        elif model.nd == 3:
            # Do the same as in 2d, also for the second tangential component.
            dofs_contact_0 = dofs_contact[:num_contact_cells]
            dofs_contact_1 = dofs_contact[num_contact_cells::2]
            dofs_contact_2 = dofs_contact[num_contact_cells + 1 :: 2]
            reorder[dofs_contact_start:dofs_contact_end] = np.vstack(
                [dofs_contact_0, dofs_contact_1, dofs_contact_2]
            ).ravel("F")
        else:
            raise ValueError("Model dimension must be 2 or 3.")
        return reorder


class SinglePhysicsPreconditioner(ABC):
    """
    Abstract class for defining a preconditioner.
    """

    def group(self):
        """
        Return the group for the preconditioner.
        """
        return self._group

    @property
    @abstractmethod
    def key(self) -> str:
        """
        Return the key for the preconditioner.
        """
        pass

    @property
    @abstractmethod
    def tag(self) -> str:
        """
        Return the tag for the preconditioner.
        """
        pass

    @property
    def complement_tag(self) -> str:
        """
        Return the tag for the complement of the preconditioner.
        """
        return self.tag + "_cpl"

    @property
    def ksp_keep_use_pmat(self) -> bool:
        """
        Return whether to keep the preconditioner matrix.
        """
        return False

    @property
    def unit_block_size(self) -> bool:
        """
        Return the number of dimensions for the preconditioner.
        """
        return True

    def near_null_space(self, model: pp.PorePyModel) -> np.ndarray | None:
        """
        Return the near null space for the preconditioner.
        """
        return None

    @abstractmethod
    def _default_options(self, model: pp.PorePyModel, dof_manager) -> dict:
        """
        Return the default options for the preconditioner.
        """
        pass

    def inverter(
        self,
        model: pp.PorePyModel,
        dof_manager: DofManager,
        groups: list[AbstractGroup],
    ) -> Callable:
        """
        Return the inverter for the preconditioner.
        """
        return None

    def _default_fieldsplit_options(self, model: pp.PorePyModel, dof_manager) -> dict:
        """Options for field splits. Provide a separate method for this, since it
        involves some boilerplate options.
        """
        opts = {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "upper",
            "pc_fieldsplit_schur_precondition": "selfp",
            f"fieldsplit_{self.tag}_ksp_type": "preonly",
            f"fieldsplit_{self.complement_tag}_ksp_type": "preonly",
        }
        return opts

    def configure(
        self,
        model: pp.PorePyModel,
        dof_manager: pp.DofManager,
        opts: dict | None = None,
        has_complement: bool = False,
    ) -> dict:
        default_opts = self._default_options(model, dof_manager)
        user_opts = opts.get(self.key, {})

        local_opts = default_opts | user_opts

        if has_complement:
            fieldsplit_opts = self._default_fieldsplit_options(model, dof_manager)

            # The local options need to be prefixed with the relevant fieldsplit tag.
            local_fieldsplit_opts = {
                f"fieldsplit_{self.tag}_{k}": v for k, v in local_opts.items()
            }
            return fieldsplit_opts | local_fieldsplit_opts
        else:
            return local_opts


class InterfaceDarcyFluxPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = InterfaceDarcyFluxGroup()

    @property
    def key(self) -> str:
        return "interface_darcy_flux"

    @property
    def tag(self) -> str:
        return "intf_darcy_flx"

    def _default_options(self, model, dof_manager) -> dict:
        opts = {"pc_type": "ilu"}
        return opts

    def configure(
        self,
        model: pp.PorePyModel,
        dof_manager: pp.DofManager,
        opts: dict | None = None,
        has_complement: bool = False,
    ) -> dict:
        if not has_complement:
            raise ValueError(
                "The interface darcy flux preconditioner requires a complement."
            )
        return super().configure(model, dof_manager, opts, has_complement)


class InterfaceEnthalpyFluxPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = InterfaceEnthalpyFluxGroup()

    @property
    def key(self) -> str:
        return "interface_energy_flux"

    @property
    def tag(self) -> str:
        return "intf_energy_flx"

    def _default_options(self, model, dof_manager) -> dict:
        opts = {"pc_type": "ilu"}
        return opts

    def configure(
        self,
        model: pp.PorePyModel,
        dof_manager: pp.DofManager,
        opts: dict | None = None,
        has_complement: bool = False,
    ) -> dict:
        if not has_complement:
            raise ValueError(
                "The interface energy flux preconditioner requires a complement."
            )
        return super().configure(model, dof_manager, opts, has_complement)


class InterfaceFourierFluxPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = InterfaceFourierFluxGroup()

    @property
    def key(self) -> str:
        return "interface_energy_flux"

    @property
    def tag(self) -> str:
        return "intf_energy_flx"

    def _default_options(self, model, dof_manager) -> dict:
        opts = {"pc_type": "ilu"}
        return opts

    def configure(
        self,
        model: pp.PorePyModel,
        dof_manager: pp.DofManager,
        opts: dict | None = None,
        has_complement: bool = False,
    ) -> dict:
        if not has_complement:
            raise ValueError(
                "The interface energy flux preconditioner requires a complement."
            )
        return super().configure(model, dof_manager, opts, has_complement)


class InterfaceMassEnergyFluxPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = InterfaceMassEnergyFluxGroup()

    @property
    def key(self) -> str:
        return "interface_mass_energy_flux"

    @property
    def tag(self) -> str:
        return "intf_mass_energy_flx"

    def _default_options(self, model, dof_manager) -> dict:
        opts = {"pc_type": "ilu"}
        return opts

    def configure(
        self,
        model: pp.PorePyModel,
        dof_manager: pp.DofManager,
        opts: dict | None = None,
        has_complement: bool = False,
    ) -> dict:
        if not has_complement:
            raise ValueError(
                "The interface mass energy flux preconditioner requires a complement."
            )
        return super().configure(model, dof_manager, opts, has_complement)


class MassBalancePreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = MassBalanceGroup()

    @property
    def key(self) -> str:
        return "mass_balance"

    @property
    def tag(self) -> str:
        return "mass_bal"

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_strong_threshold": 0.7,
        }
        return local_opts


class MassBalanceDimSplitPreconditioner(MassBalancePreconditioner):
    def __init__(self):
        self._group = MassBalanceDimSplitGroup()


class MassBalanceDimSplitCPRPreconditioner(MassBalanceDimSplitPreconditioner):
    # Special version of the mass balance preconditioner for use in CPR.
    # The key ingredient is that the filedsplit option is set to be additive.
    def _default_fieldsplit_options(self, model, dof_manager):
        inherited_opts = super()._default_fieldsplit_options(model, dof_manager)

        # The following options are not needed for the CPR preconditioner, and will
        # cause issues if they are present.
        keys_to_delete = [
            "pc_fieldsplit_type",
            "pc_fieldsplit_schur_factorization_type",
            "pc_fieldsplit_schur_precondition",
        ]
        for key in keys_to_delete:
            inherited_opts.pop(key, None)

        local_opts = {
            "pc_fieldsplit_type": "additive",
        }
        return inherited_opts | local_opts


class EnergyBalancePreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = MassBalanceGroup()

    @property
    def key(self) -> str:
        return "energy_balance"

    @property
    def tag(self) -> str:
        return "energy_bal"

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_strong_threshold": 0.7,
        }
        return local_opts


class EnergyBalanceDimSplitPreconditioner(EnergyBalancePreconditioner):
    def __init__(self):
        self._group = EnergyBalanceDimSplitGroup()


class MechanicsPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = MechanicsGroup()

    @property
    def key(self) -> str:
        return "mechanics"

    @property
    def tag(self) -> str:
        return "mech"

    @property
    def unit_block_size(self) -> bool:
        return False

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            "ksp_type": "preonly",
            "pc_type": "hmg",
            "hmg_inner_pc_type": "hypre",
            "hmg_inner_pc_hypre_type": "boomeramg",
            "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": 2,
            # 3D model has bad grid
            "mg_levels_pc_type": "ilu" if model.nd == 3 else "sor",
        }
        return local_opts

    # def near_null_space(self, model):
    #     return hm_solver.build_mechanics_near_null_space(model)


class FixedStressPreconditioner(MechanicsPreconditioner):
    def _flow_groups(self, model, dof_manager):
        # Get all groups associated with the flow equations. This will at least include
        # the pressure matrix group, and possibly also the fracture and intersection
        # groups.

        equation_names = dof_manager.equation_names(model)
        target_ind = [
            i
            for i, x in enumerate(equation_names)
            if x == EquationNames.MASS_BALANCE.value
        ]
        return target_ind

    def inverter(
        self, model: pp.PorePyModel, dof_manager: DofManager, groups: list[int]
    ) -> Callable:
        """Get the inverter for the fixed stress preconditioner.

        This class relies on two hard-coded assumptions:
            1. The physics to be stabilized is the mass balance equation. If we ever
               need fixed stress for, say, thermal diffusion, a different approach is
               needed to identify the relevant groups.
            2. The mass balance group is split into first, the matrix group, second, the
                fracture group, and third, the intersection group.
        """

        flow_group = self._flow_groups(model, dof_manager)
        if len(flow_group) == 0:
            raise ValueError(
                "No flow group found in the model. This is required for the fixed stress preconditioner."
            )
        elif len(flow_group) == 1:
            # This is a fixed-dimensional problem.
            raise NotImplementedError(
                "The fixed stress preconditioner is not yet implemented for fixed-dimensional problems."
            )
        else:  # len(flow_group) > 1
            # This is a mixed-dimensional problem, use the md scheme
            return lambda bmat: csr_to_petsc(
                make_fs_analytical_slow_new(
                    model,
                    bmat,
                    p_mat_group=flow_group[0],
                    p_frac_group=flow_group[1],
                    groups=groups,
                ).mat,
                bsize=1,
            )


class ContactPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = ContactGroup()

    @property
    def key(self) -> str:
        return "contact"

    @property
    def tag(self) -> str:
        return "contact"

    @property
    def unit_block_size(self) -> bool:
        return False

    def group(self):
        return ContactGroup()

    def _default_fieldsplit_options(self, model, dof_manager) -> dict:
        opts = super()._default_fieldsplit_options(model, dof_manager)
        opts.update(
            {
                f"fieldsplit_{self.complement_tag}_mat_schur_complement_ainv_type": "blockdiag"
            }
        )
        return opts

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            "pc_type": "pbjacobi",
            # TODO: Not sure this will come through in the right way
            # "mat_schur_complement_ainv_type": "blockdiag",
        }
        return local_opts


class BlockILU(SinglePhysicsPreconditioner):
    def __init__(self, groups):
        self._group = groups

    @property
    def key(self) -> str:
        return "cpr"

    @property
    def tag(self) -> str:
        return "cpr"

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            # "ksp_type": "preonly",
            "python_pc_type": "ilu",
            "pc_type": "python",
            # "pc_cprilu_levels": 2,
            # "pc_cprilu_fill": 0.1,
            # "pc_cprilu_zeropivot": 1e-12,
        }
        return local_opts

    def python_preconditioner(self, bmat, dof_manager: DofManager):
        indices = []
        for g in self._group:
            # Get the indices for the group.
            indices.append(
                dof_manager._name_to_group_indices[g.equation_names(None)[0]]
            )

        # Need to get hold of the groups here.
        return PcPythonPermutation(
            to_cell_ordering(bmat, indices), block_size=len(self._group)
        )


class IdentityPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self, group):
        self._group = group

    @property
    def key(self) -> str:
        return "identity"

    @property
    def tag(self) -> str:
        return "identity"

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            "pc_type": "none",
        }
        return local_opts


class CompositePreconditioner(SinglePhysicsPreconditioner):
    """A class for a composite (e.g., multi-stage) preconditioner for a block."""

    def __init__(self, solvers):
        self.solvers = solvers

        g = []
        for solver in self.solvers:
            if isinstance(solver, SinglePhysicsPreconditioner):
                group = solver.group()
            elif isinstance(solver, list):
                # If the solver is a list, we assume it contains multiple groups.
                group = [slv.group() for slv in solver]
            else:
                raise TypeError(
                    "The solver must be a SinglePhysicsPreconditioner or a list of them."
                )
            if not isinstance(group, list):
                group = [group]
            g += group
        self._group = g

    @property
    def key(self) -> str:
        keys = []
        for slv in self.solvers:
            if isinstance(slv, SinglePhysicsPreconditioner):
                keys.append(slv.key)
            elif isinstance(slv, list):
                keys += [x.key for x in slv]
            else:
                raise TypeError(
                    "The solver must be a SinglePhysicsPreconditioner or a list of them."
                )

        return "composite_" + "_".join([key for key in keys])

    @property
    def tag(self) -> str:
        return "comp"  # + "_".join([g.tag for g in self._groups])

    def _default_options(self, model, dof_manager) -> dict:
        local_opts = {
            "ksp_type": "preonly",
            "pc_type": "composite",
            "pc_composite_type": "multiplicative",
            # "pc_composite_pcs": ",".join(["none"] * len(self.solvers)),
        }
        return local_opts


class MultiPhysicsPreconditioner:
    """Translate a general scheme to a specific PETSc preconditioner, specified as a
    dictionary (really a fully specified petsc options).
    """

    def __init__(
        self,
        components: list[SinglePhysicsPreconditioner],
        dof_manager: DofManager,
        model: pp.PorePyModel,
        options: dict | None = None,
    ):
        """
        Args:
            groups: List of groups of equations and variables.
            schemes: List of schemes for each group.
        """
        self._single_physics_precond = components
        self._dof_manager = dof_manager
        self._nd = model.nd
        self._model = model
        self._options = options if options is not None else {}

    def configure(
        self,
        bmat: BlockMatrixStorage,
        pc,  # PC comes from ksp or similar
        user_options: dict | None = None,
        precond_list: list[SinglePhysicsPreconditioner] | None = None,
        prefix: str | None = None,
    ) -> dict:  # TODO: Return None?
        """
        Populate the PETSc preconditioner based on the groups and schemes. This entails
        making a bridge from the general settings defined in a scheme to the PETSc
        options needed to apply the scheme to a contrete linear system, while also accounting for

        Args:
            model: The model instance specifying the problem to be solved.
        """
        user_options = user_options if user_options is not None else {}

        if precond_list is None:
            precond_list = self._single_physics_precond

        options = {}

        if prefix is None:
            prefix = ""

        dof_manager = self._dof_manager

        for counter, single_physics_precond in enumerate(precond_list):
            # Define a scheme for the group
            has_complement = counter < len(precond_list) - 1

            # Generate the actual petsc proconditioner.
            loc_options = single_physics_precond.configure(
                model=self._model,
                dof_manager=self._dof_manager,
                has_complement=has_complement,
                opts=user_options,
            )
            tagged_options = {f"{prefix}{k}": v for k, v in loc_options.items()}

            if isinstance(single_physics_precond, CompositePreconditioner):
                # Set up the composite preconditioner so that we can get hold of the
                # sub-preconditioners and configure them as well.
                insert_petsc_options(tagged_options)
                pc.setFromOptions()
                pc.setUp()
                for i, sub_solver in enumerate(single_physics_precond.solvers):
                    if isinstance(sub_solver, list):
                        pc.addCompositePCType("fieldsplit")
                        sub_pc = pc.getCompositePC(i)
                        sub_pc.setOperators(*pc.getOperators())
                        loc_options = self.configure(
                            bmat,
                            sub_pc,
                            user_options,
                            sub_solver,
                            prefix=f"{prefix}sub_{i}_",
                        )
                    else:
                        loc_options = sub_solver.configure(
                            model=self._model,
                            dof_manager=self._dof_manager,
                            has_complement=has_complement,
                            opts=user_options,
                        )
                        pc.addCompositePCType(loc_options["pc_type"])
                        sub_pc = pc.getCompositePC(i)
                        sub_pc.setOperators(*pc.getOperators())
                    # Implementation note: This is something of a break with how petsc
                    # options are set in the rest of the package: Instead of defining
                    # the option through PETSc.Options(), we use the Python API
                    # directly. This may be possible to avoid, but turned out to solve
                    # an issue with setting up CompositePC, so it will have to do for
                    # now.
                    # Set the matrix for the sub-preconditioner. This seems to be
                    # necessary for composite preconditioners.
                    if loc_options.get("pc_type") == "python":
                        # EK cannot wrap his head around what this would mean, so we
                        # rule it out for now.
                        assert not has_complement
                        python_pc = sub_solver.python_preconditioner(bmat, dof_manager)
                        python_pc.petsc_pc.setOptionsPrefix(f"{prefix}python_")
                        sub_pc.setType("python")
                        sub_pc.setPythonContext(python_pc)

                    tagged_loc_options = {
                        f"{prefix}sub_{i}_{k}": v for k, v in loc_options.items()
                    }

                    insert_petsc_options(tagged_loc_options)
                    sub_pc.setFromOptions()
                    sub_pc.setUp()
                    tagged_options |= tagged_loc_options

            # Get the tag for this group, and prepend it to the options.

            options |= tagged_options
            if not has_complement:
                # If this is the last preconditioner in the list, we can set it up and
                # return the options database.
                insert_petsc_options(options)
                pc.setFromOptions()
                pc.setUp()

                return options
            else:
                # There are more preconditioners to process, using a fieldsplit style
                # preconditioner. We need to parse the fieldsplit options and set up the
                # preconditioners of the next group.
                pc, prefix = self._parse_fieldsplit_pc(
                    precond_list[counter:],
                    bmat,
                    pc,
                    prefix,
                    tagged_options=tagged_options,
                )

        raise ValueError("Should have reached an empty complement")

    def _parse_fieldsplit_pc(
        self,
        precond_list,
        bmat: BlockMatrixStorage,
        pc: PETSc.PC,
        prefix: str,
        tagged_options: dict | None = None,
    ):
        dof_manager = self._dof_manager

        elim_precond = precond_list[0]

        elim_group = dof_manager.blocks_of_solver(elim_precond)
        keep_group = []
        for i in range(1, len(precond_list)):
            keep_group += dof_manager.blocks_of_solver(precond_list[i])

        empty_bmat = bmat.empty_container()[elim_group + keep_group]

        block_size = 1 if elim_precond.unit_block_size else self._nd

        # Get the IS for the group, but only if complement is not None.
        is_elim, is_keep = self._dof_manager.petsc_is(
            elim_precond, precond_list[1:], empty_bmat
        )
        is_elim.setBlockSize(block_size)
        elim_tag = elim_precond.tag
        keep_tag = elim_precond.complement_tag

        insert_petsc_options(tagged_options)
        pc.setFromOptions()
        pc.setFieldSplitIS((elim_tag, is_elim), (keep_tag, is_keep))

        # Invoke the inverter, if any. This is where the fixed-stress approximation
        # for hydromechanical problems is applied. Note to self: Need to send in
        # all remaining groups to the inverter to make sure the returned matrix is
        # correct.
        inverter = elim_precond.inverter(self._model, dof_manager, keep_group)
        if inverter is not None:
            S = pc.getOperators()[1].createSubMatrix(is_keep, is_keep)
            petsc_stab = inverter(bmat)
            S.axpy(1, petsc_stab)
            pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

        pc.setUp()

        elim_ksp = pc.getFieldSplitSubKSP()[0]
        elim_pc = elim_ksp.getPC()

        keep_ksp = pc.getFieldSplitSubKSP()[1]
        keep_pc = keep_ksp.getPC()

        if len(keep_pc.getOptionsPrefix()) > 126:
            # PETSc has a limit on the prefix length, which seems to be 127
            # characters. If the prefix is too long, we raise a warning.
            msg = "The prefix for the PETSc preconditioner is too long. "
            msg += "Check the configuration of the preconditioner."
            warn(msg)

        if elim_precond.ksp_keep_use_pmat:
            _, pmat = keep_ksp.getOperators()
            # TODO: Is it correct to use the same matrix for both arguments?
            keep_ksp.setOperators(pmat, pmat)

        if elim_precond.near_null_space(self._model) is not None:
            null_space_vectors = []
            for b in elim_precond.near_null_space(self._model):
                null_space_vec_petsc = PETSc.Vec().create()  # possibly mem leak
                null_space_vec_petsc.setSizes(b.shape[0], block_size)
                null_space_vec_petsc.setUp()
                null_space_vec_petsc.setArray(b)
                null_space_vectors.append(null_space_vec_petsc)
            # possibly mem leak
            null_space_petsc = PETSc.NullSpace().create(True, null_space_vectors)
            elim_pc.getOperators()[1].setNearNullSpace(null_space_petsc)

        # Call on self.complement to configure the PETSc PC object for the complement,
        # and update (override) the options with the options returned by the complement.
        # Note that, due to the tagging system, this may override some options that were
        # set above.

        pc = keep_pc
        prefix = f"{prefix}fieldsplit_{keep_tag}_"
        return pc, prefix


def mass_balance_factory():
    return [InterfaceDarcyFluxPreconditioner(), MassBalancePreconditioner()]


def momentum_balance_factory():
    return [ContactPreconditioner(), MechanicsPreconditioner()]


def hm_factory():
    return [
        ContactPreconditioner(),
        InterfaceDarcyFluxPreconditioner(),
        FixedStressPreconditioner(),
        MassBalanceDimSplitPreconditioner(),
    ]


def thm_factory():
    # Stage 1 of CPR is a
    cpr_1 = [
        MassBalanceDimSplitCPRPreconditioner(),
        IdentityPreconditioner(EnergyBalanceDimSplitGroup()),
    ]

    # Stage 2 is a BlockILU preconditioner, which will also include a permutation to a
    # cell-wise ordering of the unknowns.
    cpr_2 = BlockILU([MassBalanceDimSplitGroup(), EnergyBalanceDimSplitGroup()])

    cpr = CompositePreconditioner(solvers=[cpr_1, cpr_2])

    return [
        ContactPreconditioner(),
        InterfaceMassEnergyFluxPreconditioner(),
        # InterfaceDarcyFluxPreconditioner(),
        # InterfaceEnthalpyFluxPreconditioner(),
        # InterfaceFourierFluxPreconditioner(),
        FixedStressPreconditioner(),
        cpr,
        # MassBalanceDimSplitPreconditioner(),
        # EnergyBalanceDimSplitPreconditioner(),
    ]


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


def to_cell_ordering(J, group_lists: list[list[int]]):
    all_groups = list(chain.from_iterable(group_lists))
    J = J[all_groups]

    rows = [
        get_dofs_of_groups(J.groups_to_blocks_row, J.local_dofs_row, group)
        for group in group_lists
    ]
    return np.row_stack(rows).ravel("F")


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
