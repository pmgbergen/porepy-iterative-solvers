from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, Callable

import numpy as np
import porepy as pp

from . import equation_variable_groups as groups
from .equation_variable_groups import EquationNames
from .fixed_stress import make_fs_analytical_slow_new
from .mat_utils import csr_to_petsc
from .petsc_solvers import PcPythonPermutation

if TYPE_CHECKING:
    from .dof_manager import DofManager


__all__ = [
    # Add all preconditioners here.
    "SinglePhysicsPreconditioner",
    "InterfaceDarcyFluxPreconditioner",
    "InterfaceEnthalpyFluxPreconditioner",
    "InterfaceFourierFluxPreconditioner",
    "InterfaceMassEnergyFluxPreconditioner",
    "MassBalancePreconditioner",
    "MassBalanceDimSplitPreconditioner",
    "MassBalanceDimSplitCPRPreconditioner",
    "EnergyBalancePreconditioner",
    "EnergyBalanceDimSplitPreconditioner",
    "MechanicsPreconditioner",
    "FixedStressPreconditioner",
    "ContactPreconditioner",
    "BlockILU",
    "IdentityPreconditioner",
    "CompositePreconditioner",
    # Add all the factory functions here.
    "mass_balance_factory",
    "momentum_balance_factory",
    "hm_factory",
    "thm_factory",
]


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
        groups: list[groups.AbstractGroup],
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
        self._group = groups.InterfaceDarcyFluxGroup()

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
        self._group = groups.InterfaceEnthalpyFluxGroup()

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
        self._group = groups.InterfaceFourierFluxGroup()

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
        self._group = groups.InterfaceMassEnergyFluxGroup()

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
        self._group = groups.MassBalanceGroup()

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
        self._group = groups.MassBalanceDimSplitGroup()


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
        self._group = groups.MassBalanceGroup()

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
        self._group = groups.EnergyBalanceDimSplitGroup()


class MechanicsPreconditioner(SinglePhysicsPreconditioner):
    def __init__(self):
        self._group = groups.MechanicsGroup()

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
                "Mass balance group not found in the model. "
                "This is required for the fixed-stress preconditioner."
            )
        elif len(flow_group) == 1:
            # This is a fixed-dimensional problem.
            raise NotImplementedError(
                "Have not yet implemented the fixed stress preconditioner "
                "for fixed-dimensional problems."
            )
        else:
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
        self._group = groups.ContactGroup()

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
        return groups.ContactGroup()

    def _default_fieldsplit_options(self, model, dof_manager) -> dict:
        opts = super()._default_fieldsplit_options(model, dof_manager)
        key = f"fieldsplit_{self.tag}_mat_schur_complement_ainv_type"
        opts.update({key: "blockdiag"})
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
            _to_cell_ordering(bmat, indices), block_size=len(self._group)
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
                    "The solver must be a SinglePhysicsPreconditioner"
                    " or a list of them."
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
                    "The solver must be a SinglePhysicsPreconditioner"
                    " or a list of them."
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
        IdentityPreconditioner(groups.EnergyBalanceDimSplitGroup()),
    ]

    # Stage 2 is a BlockILU preconditioner, which will also include a permutation to a
    # cell-wise ordering of the unknowns.
    cpr_2 = BlockILU(
        [groups.MassBalanceDimSplitGroup(), groups.EnergyBalanceDimSplitGroup()]
    )

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


def _to_cell_ordering(J, group_lists: list[list[int]]):
    all_groups = list(chain.from_iterable(group_lists))
    J = J[all_groups]

    rows = [
        get_dofs_of_groups(J.groups_to_blocks_row, J.local_dofs_row, group)
        for group in group_lists
    ]
    return np.row_stack(rows).ravel("F")


def get_dofs_of_groups(
    groups_to_block: list[list[int]], dofs: list[np.ndarray], groups: list[int]
) -> np.ndarray:
    blocks = [blk for g in groups for blk in groups_to_block[g]]
    return np.concatenate([dofs[blk] for blk in blocks])
