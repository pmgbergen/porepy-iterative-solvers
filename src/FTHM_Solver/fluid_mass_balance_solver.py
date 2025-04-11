import porepy as pp

from .full_petsc_solver import SolverScheme, PreconditionerScheme, PetscFieldSplitScheme
import FTHM_Solver


__all__ = ["FluidIterativeScheme"]


class FluidIterativeScheme(SolverScheme):  # Use SolverScheme directly
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

    def get_groups(self) -> list[list[int]]:
        return [1]

    def _eliminate_interface_darcy_flux_scheme(
        self, complement: PreconditionerScheme, prefix: str = ""
    ) -> PetscFieldSplitScheme:
        elim_group = self._group_id_from_name("interface_darcy_flux_equation")
        loc_group = self.equation_groups[elim_group[0]]

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
            groups=loc_group,
            elim_options=elim_options,
            fieldsplit_options=fieldsplit_options,
            complement=complement,
        )
        return scheme

    def _fluid_mass_balance_scheme(
        self, complement: PreconditionerScheme | None, prefix: str = ""
    ) -> PreconditionerScheme:
        mb_group = self._group_id_from_name("mass_balance_equation")
        loc_group = self.equation_groups[mb_group[0]]
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
                groups=loc_group, opts=opts
            )
        else:
            return FTHM_Solver.PetscFieldSplitScheme(
                groups=loc_group,
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
