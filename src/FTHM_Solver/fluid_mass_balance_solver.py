import porepy as pp
from functools import cached_property
from .full_petsc_solver import SolverScheme, PreconditionerScheme, PetscFieldSplitScheme
import FTHM_Solver


__all__ = ["FluidIterativeScheme"]


class FluidIterativeScheme(SolverScheme):  # Use SolverScheme directly
    def __init__(self, model: pp.PorePyModel, opts=None):
        self.model = model
        self.opts = opts
        self._equation_group_counter = 0
        self._equation_group_keys = []
        self._variable_groups_keys = []

        self._register_equation_variable_groups()

    def _register_equation_variable_groups(self) -> None:
        super()._register_equation_variable_groups()

        # If this is broken, we will not get the expected block-diagonal structure
        # of the equations and variables.
        assert len(self._equation_group_keys) == len(self._variable_groups_keys)

        subdomains = self.model.mdg.subdomains()
        interfaces = self.model.mdg.interfaces()

        # Register the groups of equations for this physics
        local_equation_keys = [
            "mass_balance_equation",
            "interface_darcy_flux_equation",
        ]

        # Fluid pressure
        for item in self._equation_group_keys:
            if item[0] in local_equation_keys:
                # The key has been added before. Raise an error.
                raise ValueError(
                    f"Key {item[0]} has already been added to the equation group."
                )
        # Register the groups of equations for this physics
        self._equation_group_keys.append([("mass_balance_equation", subdomains)])
        self._equation_group_keys.append(
            [("interface_darcy_flux_equation", interfaces)]
        )

        # Register the groups of variables for this physics
        self._variable_groups_keys.append([self.model.pressure(subdomains)])
        self._variable_groups_keys.append([self.model.interface_darcy_flux(interfaces)])

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        """Assign the following groups to the variables:

        0: Fluid pressure on all subdomains.
        1: Interface darcy fluxes on all interfaces.

        """
        return FTHM_Solver.get_variables_group_ids(
            model=self.model,
            md_variables_groups=self._variable_groups_keys,
        )

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        """Define the groups of equation in the specific order, that we will use in
        the block Jacobian to access the submatrices.

        Returns:
            List of lists of integers. Each list contains the indices of the equations
                in the group.

        """
        return FTHM_Solver.get_equations_group_ids(
            model=self.model,
            equations_group_order=self._equation_group_keys,
        )

    def get_groups(self) -> list[list[int]]:
        return [1]

    def _group_id_from_name(self, name: str) -> int:
        """Get the group id from the name of the group.

        Args:
            name: Name of the group.

        Returns:
            Group id.

        """
        # This is rough, does not allow for duplicates.
        for i, group in enumerate(self._equation_group_keys):
            if group[0][0] == name:
                return [i]
        raise ValueError(f"Group {name} not found.")

    def _add_prefix(self, dct: dict, prefix: str):
        for key in dct.keys():
            if prefix:
                dct[key] = f"{prefix}_{dct[key]}"
            else:
                dct[key] = dct[key]

    def _eliminate_interface_darcy_flux_scheme(
        self, complement: PreconditionerScheme, prefix: str = ""
    ) -> PetscFieldSplitScheme:
        elim_group = self._group_id_from_name("interface_darcy_flux_equation")

        elim_options = {
            "ksp_type": "preonly",
            "pc_type": "ilu",
        }

        fieldsplit_options = {
            "pc_fieldsplit_schur_precondition": "selfp",
        }

        for dct in [elim_options, fieldsplit_options]:
            self._add_prefix(dct, prefix)

        scheme = FTHM_Solver.PetscFieldSplitScheme(
            groups=elim_group,
            elim_options=elim_options,
            fieldsplit_options=fieldsplit_options,
            complement=complement,
        )
        return scheme

    def _fluid_mass_balance_scheme(
        self, complement: PreconditionerScheme | None, prefix: str = ""
    ) -> PreconditionerScheme:
        group = self._group_id_from_name("mass_balance_equation")
        opts = {
            "pc_type": "gamg",
            "pc_gamg_threshold": 0.02,
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": 4,
            "mg_levels_pc_type": "sor",
        }

        field_split_options = {
            "pc_fieldsplit_schur_precondition": "selfp",
        }
        for dct in [opts, field_split_options]:
            self._add_prefix(dct, prefix)

        if complement is None:
            return FTHM_Solver.SinglePhysicsPreconditionerScheme(
                groups=group, opts=opts
            )
        else:
            return FTHM_Solver.PetscFieldSplitScheme(
                groups=group,
                elim_options=opts,
                fieldsplit_options=field_split_options,
                complement=complement,
            )

    def make_solver_scheme(self, opts=None):
        precond_scheme = self._eliminate_interface_darcy_flux_scheme(
            complement=self._fluid_mass_balance_scheme(complement=None)
        )

        ksp_options = {"ksp_monitor": None}

        ksp_scheme = FTHM_Solver.PetscKSPScheme(
            preconditioner=precond_scheme, petsc_options=ksp_options
        )
        return ksp_scheme
