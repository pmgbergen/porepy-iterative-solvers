from functools import cached_property
from typing import Callable

import numpy as np
import porepy as pp
import scipy.sparse

from .block_matrix import BlockMatrixStorage, FieldSplitScheme
from .fixed_stress import make_fs_analytical_slow_new
from .full_petsc_solver import (LinearTransformedScheme, PcPythonPermutation,
                                PetscCompositeScheme, PetscFieldSplitScheme,
                                PetscKSPScheme)
from .hm_solver import IterativeHMSolver
from .iterative_solver import get_equations_group_ids, get_variables_group_ids
from .mat_utils import csr_to_petsc


class THMSolver(IterativeHMSolver):
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable]

    interface_fourier_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    interface_enthalpy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]

    well_enthalpy_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]

    def simulation_name(self) -> str:
        name = "stats_thermoporomechanics"
        # setup = self.params["linear_solver_config"]
        # name = f"{name}_geo{setup['geometry']}x{setup['grid_refinement']}"
        # name = f"{name}_sol{setup['solver']}"
        # if (bc := setup.get("thermal_diffusion_bc")) not in ("dir", None):
        #     name = f"{name}_bc{bc}"
        return name

    CONTACT_GROUP = 0

    def group_row_names(self) -> list[str]:
        return [
            "Contact frac.",
            "Flow intf.",
            "Energy intf.",
            "Force mat.",
            "Force intf.",
            "Flow mat.",
            "Flow frac.",
            "Flow lower",
            "Energy mat.",
            "Energy frac.",
            "Energy lower",
        ]

    def group_col_names(self) -> list[str]:
        return [
            r"$\lambda_{frac}$",
            r"$v_{intf}$",
            "$T_{intf}$",
            r"$u_{3D}$",
            r"$u_{intf}$",
            r"$p_{3D}$",
            r"$p_{frac}$",
            "$p_{lower}$",
            "$T_{3D}$",
            "$T_{frac}$",
            "$T_{lower}$",
        ]

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        """Prepares the groups of variables in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[x, 0]` - contact traction variable;
        `J[x, 1]` - interface Darcy flux;
        `J[x, 2]` - interface fluxes temperature;
        `J[x, 3]` - displacement in ambient dimension;
        `J[x, 4]` - displacement on interfaces;
        `J[x, 5]` - pressure in ambient dimension;
        `J[x, 6]` - pressure in fractures;
        `J[x, 7]` - pressure in interfaces;
        `J[x, 8]` - temperature in ambient dimension;
        `J[x, 9]` - temperature in fractures;
        `J[x, 10]` - temperature in interfaces;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed.

        """
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        sd_lower = [
            k
            for i in reversed(range(0, dim_max - 1))
            for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()
        intf_frac = self.mdg.interfaces(dim=dim_max - 1)

        return get_variables_group_ids(
            model=self,
            md_variables_groups=[
                [self.contact_traction(sd_frac)],  # 0
                [self.interface_darcy_flux(intf)],  # 1
                [  # 2
                    self.interface_fourier_flux(intf),
                    self.interface_enthalpy_flux(intf),
                    self.well_enthalpy_flux(intf),
                ],
                [self.displacement(sd_ambient)],  # 3
                [self.interface_displacement(intf_frac)],  # 4
                [self.pressure(sd_ambient)],  # 5
                [self.pressure(sd_frac)],  # 6
                [self.pressure(sd_lower)],  # 7
                [self.temperature(sd_ambient)],  # 8
                [self.temperature(sd_frac)],  # 9
                [self.temperature(sd_lower)],  # 10
            ],
        )

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        sd_lower = [
            k
            for i in reversed(range(0, dim_max - 1))
            for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()

        return self._correct_contact_equations_groups(
            get_equations_group_ids(
                model=self,
                equations_group_order=[
                    [  # 0
                        ("normal_fracture_deformation_equation", sd_frac),
                        ("tangential_fracture_deformation_equation", sd_frac),
                    ],
                    [("interface_darcy_flux_equation", intf)],  # 1
                    [  # 2
                        ("interface_fourier_flux_equation", intf),
                        ("interface_enthalpy_flux_equation", intf),
                        ("well_enthalpy_flux_equation", intf),  # ???
                    ],
                    [("momentum_balance_equation", sd_ambient)],  # 3
                    [("interface_force_balance_equation", intf)],  # 4
                    [("mass_balance_equation", sd_ambient)],  # 5
                    [("mass_balance_equation", sd_frac)],  # 6
                    [("mass_balance_equation", sd_lower)],  # 7
                    [("energy_balance_equation", sd_ambient)],  # 8
                    [("energy_balance_equation", sd_frac)],  # 9
                    [("energy_balance_equation", sd_lower)],  # 10
                ],
            ),
            contact_group=self.CONTACT_GROUP,
        )

    def scale_energy_balance(self, bmat: BlockMatrixStorage):
        res = bmat.empty_container()
        subdomains = [
            sd for d in reversed(range(self.nd)) for sd in self.mdg.subdomains(dim=d)
        ]
        diag = 1 / self.specific_volume(subdomains).value(self.equation_system)
        res.mat = scipy.sparse.eye(res.shape[0], format="csr")
        if len(subdomains) == 0:
            return res
        res[[9, 10]] = scipy.sparse.diags(diag)
        return res

    def make_solver_scheme(self) -> FieldSplitScheme:
        config: dict = self.params.get("linear_solver_config", {})
        solver_type: str = config.get("solver", "SAMG")
        do_linear_transformation: bool = config.get("treat_singularity_contact", True)

        if solver_type == "FGMRES":
            return self.make_solver_scheme_fgmres()

        # Groups of equations. See `equation_groups` property.
        contact = [0]  # Fracture deformation equations
        intf = [1, 2]  # Interface equations (Darcy flux, energy balance)
        mech = [3, 4]  # Momentum balance, interface force balance
        flow = [5, 6, 7]  # Mass balance in ambient, fracture, intersections
        temp = [8, 9, 10]  # Energy balance in ambient, fracture, intersections

        pt_solver_cpr_global = PetscFieldSplitScheme(
            groups=flow,
            fieldsplit_options={
                "pc_fieldsplit_type": "additive",
            },
            elim_options={
                # "pc_type": "hypre",
                # "pc_hypre_type": "boomeramg",
                # "pc_hypre_boomeramg_strong_threshold": 0.7,
                "pc_type": "gamg",
                "pc_gamg_threshold": 0.02,
            },
            complement=PetscFieldSplitScheme(
                groups=temp,
                elim_options={
                    "pc_type": "none",
                },
            ),
        )

        def make_2stage_scheme(global_prec):
            return PetscCompositeScheme(
                groups=flow + temp,
                solvers=[
                    global_prec,
                    PetscFieldSplitScheme(
                        groups=flow + temp,
                        python_pc=lambda bmat: PcPythonPermutation(
                            make_pt_permutation(bmat, p_groups=flow, t_groups=temp),
                            block_size=2,
                        ),
                        elim_options={
                            "python_pc_type": "ilu",
                            # "python_pc_type": "hypre",
                            # "python_pc_hypre_type": "Euclid",
                        },
                    ),
                ],
            )

        pt_solver_SAMG = PetscFieldSplitScheme(
            groups=flow + temp,
            python_pc=lambda bmat: PcPythonPermutation(
                make_pt_permutation(bmat, p_groups=flow, t_groups=temp),
                block_size=2,
            ),
            elim_options={
                # "python_pc_type": "hypre",
                # "python_pc_hypre_type": "boomeramg",
                # "python_pc_hypre_boomeramg_strong_threshold": 0.7,
                # "python_pc_hypre_boomeramg_P_max": 16,
                "python_pc_type": "gamg",
                "python_pc_gamg_threshold": 0.02,
                "python_mg_levels_ksp_type": "richardson",
                "python_mg_levels_ksp_max_it": 4,
                "python_mg_levels_pc_type": "sor",
            },
        )

        pt_solver_S4_diag = PetscFieldSplitScheme(
            groups=flow,
            elim_options={
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "hypre_boomeramg_strong_threshold": 0.7,
            },
            fieldsplit_options={
                "pc_fieldsplit_schur_precondition": "selfp",
            },
            complement=PetscFieldSplitScheme(
                groups=temp,
                elim_options={
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "hypre_boomeramg_strong_threshold": 0.7,
                },
            ),
        )

        pt_solver_AAMG = PetscFieldSplitScheme(
            groups=flow,
            fieldsplit_options={
                "pc_fieldsplit_type": "additive",
            },
            elim_options={
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "pc_hypre_boomeramg_strong_threshold": 0.7,
            },
            complement=PetscFieldSplitScheme(
                groups=temp,
                elim_options={
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "pc_hypre_boomeramg_strong_threshold": 0.7,
                },
            ),
        )

        if solver_type == "SAMG":
            pressure_temperature_solver = pt_solver_SAMG
        elif solver_type == "S4_diag":
            pressure_temperature_solver = pt_solver_S4_diag
        elif solver_type == "S4_Roy":
            raise NotImplementedError("S4_Roy")
        elif solver_type == "CPR" or (solver_type.find("+ILU") != -1):
            if solver_type == "CPR":
                global_prec = pt_solver_cpr_global
            elif solver_type.find("SAMG") != -1:
                global_prec = pt_solver_SAMG
            elif solver_type.find("S4_diag") != -1:
                global_prec = pt_solver_S4_diag
            elif solver_type.find("S4_Roy") != -1:
                raise NotImplementedError("S4_Roy")
            elif solver_type.find("AAMG") != -1:
                global_prec = pt_solver_AAMG
            else:
                raise ValueError(solver_type)
            pressure_temperature_solver = make_2stage_scheme(global_prec)
        else:
            raise ValueError(solver_type)

        return LinearTransformedScheme(
            right_transformations=[
                lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
            ]
            if do_linear_transformation
            else [],
            left_transformations=[
                lambda bmat: self.scale_energy_balance(bmat),
            ],
            # The inner solver is a KSP solver with a nested preconditioner.
            inner=PetscKSPScheme(
                petsc_options=(
                    {
                        # "ksp_type": "fgmres",
                        "ksp_rtol": 1e-8,
                    }
                    | {"ksp_monitor": None}
                    if config.get("ksp_monitor", True)
                    else {}
                ),
                compute_eigenvalues=False,
                # Nested field split preconditioner. The outermost layer treats the contact
                # equations.
                preconditioner=PetscFieldSplitScheme(
                    # The contact mechanics equations.
                    groups=contact,
                    # The blocks are of size `nd`, the number of contact traction
                    # components.
                    block_size=self.nd,
                    # Use a diagonal approximation when constructing the Schur
                    # complement, e.g. S_A = D - C diag(A)^-1 B. See the PETSc manual
                    # for details.
                    fieldsplit_options={
                        "pc_fieldsplit_schur_precondition": "selfp",
                    },
                    # PETSc's point block Jacobi preconditioner, with the given block
                    # size.
                    elim_options={
                        "pc_type": "pbjacobi",
                    },
                    keep_options={
                        "mat_schur_complement_ainv_type": "blockdiag",
                    },
                    complement=PetscFieldSplitScheme(
                        # Eliminate flow and transport interface equations.
                        groups=intf,
                        elim_options={
                            "pc_type": "ilu",
                        },
                        fieldsplit_options={
                            "pc_fieldsplit_schur_precondition": "selfp",
                        },
                        # Nested field split preconditioner.
                        complement=PetscFieldSplitScheme(
                            # Eliminate the mechanics equations (momentum balance and interface
                            # force balance) with a hmg preconditioner based on hypre.
                            groups=mech,
                            elim_options=(
                                {
                                    # "pc_type": "hypre",
                                    # "pc_hypre_type": "boomeramg",
                                    # "pc_hypre_boomeramg_strong_threshold": 0.7,
                                    # #
                                    # 'pc_hypre_boomeramg_max_row_sum': 1.0,
                                    # # "pc_hypre_boomeramg_smooth_type": "Euclid",
                                    "pc_type": "hmg",
                                    "hmg_inner_pc_type": "gamg",
                                    "hmg_inner_pc_gamg_threshold": 0.02,
                                    # "hmg_inner_pc_hypre_type": "boomeramg",
                                    # "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
                                    "mg_levels_ksp_type": "richardson",
                                    "mg_levels_ksp_max_it": 2,
                                    "mg_levels_pc_type": "ilu",
                                }
                            ),
                            keep_options={},
                            block_size=self.nd,
                            invert=lambda bmat: csr_to_petsc(
                                make_fs_analytical_slow_new(
                                    self,
                                    bmat,
                                    p_mat_group=5,
                                    p_frac_group=6,
                                    groups=flow + temp,
                                ).mat,
                                bsize=1,
                            ),
                            complement=pressure_temperature_solver,
                        ),
                    ),
                ),
            ),
        )

    def make_solver_scheme_fgmres(self):
        config: dict = self.params.get("linear_solver_config", {})
        ksp_monitor = {"ksp_monitor": None} if config.get("ksp_monitor", True) else {}

        do_linear_transformation: bool = config.get("treat_singularity_contact", True)
        contact = [0]
        intf = [1, 2]
        mech = [3, 4]
        flow = [5, 6, 7]
        temp = [8, 9, 10]
        inner_rtol = 1e-5
        return LinearTransformedScheme(
            right_transformations=[
                lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
            ]
            if do_linear_transformation
            else [],
            left_transformations=[
                lambda bmat: self.scale_energy_balance(bmat),
            ],
            inner=PetscKSPScheme(
                petsc_options=(
                    {
                        "ksp_type": "fgmres",
                        "ksp_rtol": 1e-8,
                    }
                    | ksp_monitor
                ),
                compute_eigenvalues=False,
                preconditioner=PetscFieldSplitScheme(
                    groups=contact,
                    block_size=self.nd,
                    fieldsplit_options={
                        "pc_fieldsplit_schur_precondition": "selfp",
                    },
                    elim_options={
                        "pc_type": "pbjacobi",
                    },
                    keep_options={
                        "mat_schur_complement_ainv_type": "blockdiag",
                    },
                    complement=PetscFieldSplitScheme(
                        groups=intf,
                        elim_options={
                            "ksp_type": "gmres",
                            "ksp_rtol": inner_rtol,
                            "ksp_pc_side": "right",
                            #
                            "pc_type": "ilu",
                        }
                        | ksp_monitor,
                        fieldsplit_options={
                            "pc_fieldsplit_schur_precondition": "selfp",
                        },
                        complement=PetscFieldSplitScheme(
                            groups=mech,
                            elim_options=(
                                {
                                    "ksp_type": "gmres",
                                    "ksp_rtol": inner_rtol,
                                    "ksp_pc_side": "right",
                                    #
                                    "pc_type": "hmg",
                                    "hmg_inner_pc_type": "gamg",
                                    "hmg_inner_pc_gamg_threshold": 0.02,
                                    # "hmg_inner_pc_type": "hypre",
                                    # "hmg_inner_pc_hypre_type": "boomeramg",
                                    # "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
                                    "mg_levels_ksp_type": "richardson",
                                    "mg_levels_ksp_max_it": 2,
                                    # 3D model has bad grid
                                    "mg_levels_pc_type": "ilu",
                                }
                                | ksp_monitor
                            ),
                            keep_options={
                                "ksp_type": "gmres",
                                "ksp_rtol": inner_rtol,
                                "ksp_pc_side": "right",
                            }
                            | ksp_monitor,
                            ksp_keep_use_pmat=True,
                            block_size=self.nd,
                            invert=lambda bmat: csr_to_petsc(
                                make_fs_analytical_slow_new(
                                    self,
                                    bmat,
                                    p_mat_group=5,
                                    p_frac_group=6,
                                    groups=flow + temp,
                                ).mat,
                                bsize=1,
                            ),
                            complement=PetscCompositeScheme(
                                groups=flow + temp,
                                solvers=[
                                    PetscFieldSplitScheme(
                                        groups=flow,
                                        fieldsplit_options={
                                            "pc_fieldsplit_type": "additive",
                                        },
                                        elim_options={
                                            "pc_type": "gamg",
                                            "pc_gamg_threshold": 0.02,
                                            # "pc_type": "hypre",
                                            # "pc_hypre_type": "boomeramg",
                                            # "pc_hypre_boomeramg_strong_threshold": 0.7,
                                        },
                                        complement=PetscFieldSplitScheme(
                                            groups=temp,
                                            elim_options={
                                                "pc_type": "none",
                                            },
                                        ),
                                    ),
                                    PetscFieldSplitScheme(
                                        groups=flow + temp,
                                        python_pc=lambda bmat: PcPythonPermutation(
                                            make_pt_permutation(
                                                bmat, p_groups=flow, t_groups=temp
                                            ),
                                            block_size=2,
                                        ),
                                        elim_options={
                                            "python_pc_type": "ilu",
                                            # "python_pc_type": "hypre",
                                            # "python_pc_hypre_type": "Euclid",
                                        },
                                    ),
                                ],
                            ),
                        ),
                    ),
                ),
            ),
        )


def get_dofs_of_groups(
    groups_to_block: list[list[int]], dofs: list[np.ndarray], groups: list[int]
) -> np.ndarray:
    blocks = [blk for g in groups for blk in groups_to_block[g]]
    return np.concatenate([dofs[blk] for blk in blocks])


def make_pt_permutation(
    J: BlockMatrixStorage, p_groups: list[int], t_groups: list[int]
):
    J = J[p_groups + t_groups]
    t_dofs = get_dofs_of_groups(
        groups_to_block=J.groups_to_blocks_row, dofs=J.local_dofs_row, groups=t_groups
    )
    p_dofs = get_dofs_of_groups(
        groups_to_block=J.groups_to_blocks_row, dofs=J.local_dofs_row, groups=p_groups
    )
    return np.vstack([p_dofs, t_dofs]).ravel("F")
