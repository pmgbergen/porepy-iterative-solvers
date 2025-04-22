from functools import cached_property
from typing import Callable

import numpy as np
import porepy as pp

from .block_matrix import BlockMatrixStorage, KSPScheme
from .fixed_stress import make_fs_analytical_slow_new
from .full_petsc_solver import (
    LinearTransformedScheme,
    PetscFieldSplitScheme,
    PetscKSPScheme,
)
from .iterative_solver import (
    IterativeLinearSolver,
    get_equations_group_ids,
    get_variables_group_ids,
)
from .mat_utils import csr_ones, csr_to_petsc, inv_block_diag


class IterativeHMSolver(IterativeLinearSolver):
    """Iterative solver mixin for coupled hydro-mechanical problems.

    The solver is intended used for problems with hydromechanics.
    """

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Variable]
    interface_darcy_flux: Callable[[list[pp.MortarGrid]], pp.ad.Variable]
    displacement: Callable[[list[pp.Grid]], pp.ad.Variable]
    interface_displacement: Callable[[list[pp.MortarGrid]], pp.ad.Variable]
    pressure: Callable[[list[pp.Grid]], pp.ad.Variable]

    CONTACT_GROUP: int = 0

    def group_row_names(self) -> list[str]:
        return [
            "Contact frac.",
            "Flow intf.",
            "Force mat.",
            "Force intf.",
            "Flow mat.",
            "Flow frac.",
            "Flow lower.",
        ]

    def group_col_names(self) -> list[str]:
        return [
            r"$\lambda_{frac}$",
            r"$v_{intf}$",
            r"$u_{3D}$",
            r"$u_{intf}$",
            r"$p_{3D}$",
            r"$p_{frac}$",
            r"$p_{lower}$",
        ]

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        """Prepares the groups of variables in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[x, 0]` - contact traction variable;
        `J[x, 1]` - interface Darcy flux variable;
        `J[x, 2]` - matrix displacement variable;
        `J[x, 3]` - interface displacement variable;
        `J[x, 4]` - matrix pressure variable;
        `J[x, 5]` - lower-dim pressure variable;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed.

        Returns:
            List of lists of integers. Each list contains the indices (on the block
                level) of the variables in the group (defined above).

        """
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_intersec = [
            k
            for i in reversed(range(0, dim_max - 1))
            for k in self.mdg.subdomains(dim=i)
        ]
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        intf = self.mdg.interfaces()
        intf_frac = self.mdg.interfaces(dim=dim_max - 1)

        return get_variables_group_ids(
            model=self,
            md_variables_groups=[
                [self.contact_traction(sd_frac)],  # 0
                [self.interface_darcy_flux(intf)],  # 1
                [self.displacement(sd_ambient)],  # 2
                [self.interface_displacement(intf_frac)],  # 3
                [self.pressure(sd_ambient)],  # 4
                [self.pressure(sd_frac)],  # 5
                [self.pressure(sd_intersec)],  # 6
            ],
        )

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        """Prepares the groups of equation in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[0, x]` - contact traction equations;
        `J[1, x]` - interface Darcy flux equation;
        `J[2, x]` - matrix momentum balance equation;
        `J[3, x]` - interface force balance equation;
        `J[4, x]` - matrix mass balance equation;
        `J[5, x]` - lower-dim mass balance equation;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed. Encorporates the permutation `contact_permutation` which
        rearranges the conctact conditions into a cell-wise block structure.

        Returns:
            List of lists of integers. Each list contains the indices (in terms of the
                blocks defined above) of the equations in the group, as defined by the
                EquationSystem of the PorePy model.

        """
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        sd_intersec = [
            k
            for i in reversed(range(0, dim_max - 1))
            for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()

        return self._correct_contact_equations_groups(
            equation_groups=get_equations_group_ids(
                model=self,
                equations_group_order=[
                    [  # 0
                        ("normal_fracture_deformation_equation", sd_frac),
                        ("tangential_fracture_deformation_equation", sd_frac),
                    ],
                    [("interface_darcy_flux_equation", intf)],  # 1
                    [("momentum_balance_equation", sd_ambient)],  # 2
                    [("interface_force_balance_equation", intf)],  # 3
                    [("mass_balance_equation", sd_ambient)],  # 4
                    [("mass_balance_equation", sd_frac)],  # 5
                    [("mass_balance_equation", sd_intersec)],  # 6
                ],
            ),
            contact_group=self.CONTACT_GROUP,
        )

    @cached_property
    def contact_permutation(self) -> np.ndarray:
        """Permutation of the contact mechanics equations. Must be applied to the
        Jacobian.

        The PorePy arrangement is:

            `[[C0_norm], [C1_norm], [C0_tang], [C1_tang]]`,

        where `C0` and `C1` correspond to the contact equation on fractures 0 and 1.
        We permute it to:

            `[[f0_norm, f0_tang], [f1_norm, f1_tang]]`

        Returns:


        """
        return make_reorder_contact(self, contact_group=self.CONTACT_GROUP)

    @cached_property
    def eq_dofs(self) -> list[np.ndarray | None]:
        """Equation indices (rows of the Jacobian) in the order defined by the PorePy
        EquationSystem.

        Compared to the parent class, this method corrects the contact equations
        permutation. See method contact_permutation for details.

        Returns:
            List of numpy arrays. Each list entry correspond to one equation on one
                grid. The arrays provide the fine-scale (actual row indices) of the
                equation.

        """
        unpermuted_eq_dofs = super().eq_dofs
        return self._correct_contact_eq_dofs(
            unpermuted_eq_dofs, contact_group=self.CONTACT_GROUP
        )

    def _correct_contact_eq_dofs(
        self, unpermuted_eq_dofs: list[np.ndarray], contact_group: int
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
        if len(self.equation_groups[contact_group]) == 0:
            # Ignore mypy error, list[np.ndarray] is a subset of list[np.ndarray |
            # None].
            return unpermuted_eq_dofs  # type: ignore[return-value]

        # We assume that normal equations go first. TODO: Can we make this more robust,
        # or else put an assert here.
        normal_blocks = self.equation_groups[contact_group]
        num_fracs = len(self.mdg.subdomains(dim=self.nd - 1))

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
            # There will be self.nd equations for each block.
            inds = offset + np.arange(unpermuted_eq_dofs[nb].size * self.nd)
            offset = inds[-1] + 1
            eq_dofs_corrected[nb] = np.array(inds)

        return eq_dofs_corrected

    def _correct_contact_equations_groups(
        self, equation_groups: list[list[int]], contact_group: int
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

        num_fracs = len(self.mdg.subdomains(dim=self.nd - 1))
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

    def Qright(self, contact_group: int, u_intf_group: int) -> BlockMatrixStorage:
        """Assemble the right linear transformation."""
        J = self.bmat
        # Sorted according to groups. If not done, the matrix can be in porepy order,
        # which does not guarantee that diagonal groups are truly on diagonals.
        Qright = J.empty_container()[:]

        if contact_group not in J.active_groups[0]:
            Qright.mat = csr_ones(Qright.shape[0])
            return Qright

        J55 = J[u_intf_group, u_intf_group].mat

        J55_inv = inv_block_diag(J55, nd=self.nd, lump=False)

        Qright.mat = csr_ones(Qright.shape[0])

        J54 = J[u_intf_group, contact_group].mat

        tmp = -J55_inv @ J54
        Qright[u_intf_group, contact_group] = tmp
        return Qright

    def Qleft(self, contact_group: int, u_intf_group: int) -> BlockMatrixStorage:
        """Assemble the left linear transformation."""
        J = self.bmat
        # Sorted according to groups. If not done, the matrix can be in porepy order,
        # which does not guarantee that diagonal groups are truly on diagonals.
        Qleft = J.empty_container()[:]

        if contact_group not in J.active_groups[0]:
            Qleft.mat = csr_ones(Qleft.shape[0])
            return Qleft

        J55_inv = inv_block_diag(
            J[u_intf_group, u_intf_group].mat, nd=self.nd, lump=False
        )
        # J55_inv = inv(J[u_intf_group, u_intf_group].mat)
        Qleft.mat = csr_ones(Qleft.shape[0])
        Qleft[contact_group, u_intf_group] = (
            -J[contact_group, u_intf_group].mat @ J55_inv
        )
        return Qleft

    def sticking_sliding_open(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        opening = self.opening_indicator(fractures).value(self.equation_system) < 0
        closed = np.logical_not(opening)
        sliding = np.logical_and(
            closed, self.sliding_indicator(fractures).value(self.equation_system) > 0
        )
        sticking = np.logical_not(opening | sliding)

        return sticking, sliding, opening

    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        mat, rhs = self.linear_system

        # Apply the `contact_permutation`.
        mat = mat[self.contact_permutation]
        rhs = rhs[self.contact_permutation]
        self.bmat.mat = mat
        self.linear_system = mat, rhs

    def make_solver_scheme(self) -> KSPScheme | LinearTransformedScheme:
        contact = [0]
        intf = [1]
        mech = [2, 3]
        flow = [4, 5, 6]
        config = self.params.get("linear_solver_config", {})

        do_linear_transformation: bool = config.get("treat_singularity_contact", True)
        ksp_monitor_options = (
            {"ksp_monitor": None} if config.get("ksp_monitor", True) else {}
        )
        inner_ksp_monitor_options = ksp_monitor_options

        return LinearTransformedScheme(
            right_transformations=[
                lambda bmat: self.Qright(contact_group=0, u_intf_group=3)
            ]
            if do_linear_transformation
            else [],
            inner=PetscKSPScheme(
                petsc_options={
                    "ksp_rtol": config.get("ksp_rtol", 1e-10),
                    "ksp_atol": config.get("ksp_atol", 1e-15),
                    "ksp_max_it": config.get("ksp_max_it", 90),
                    "ksp_gmres_restart": config.get("ksp_gmres_restart", 30),
                }
                | ksp_monitor_options,
                preconditioner=PetscFieldSplitScheme(
                    groups=contact,
                    block_size=self.nd,
                    fieldsplit_options={
                        "pc_fieldsplit_schur_precondition": "selfp",
                    },
                    elim_options={
                        "pc_type": "pbjacobi",
                        # "ksp_type": "gmres",
                        # "ksp_rtol": 1e-5,
                    }
                    | inner_ksp_monitor_options,
                    keep_options={
                        "mat_schur_complement_ainv_type": "blockdiag",
                    },
                    complement=PetscFieldSplitScheme(
                        groups=intf,
                        fieldsplit_options={
                            "pc_fieldsplit_schur_precondition": "selfp",
                        },
                        elim_options={
                            # "ksp_type": "gmres",
                            # "ksp_rtol": 1e-3,
                            "pc_type": "ilu",
                            "pc_factor_levels": 1,
                        }
                        | inner_ksp_monitor_options,
                        complement=PetscFieldSplitScheme(
                            groups=mech,
                            block_size=self.nd,
                            invert=lambda bmat: csr_to_petsc(
                                make_fs_analytical_slow_new(
                                    self,
                                    bmat,
                                    p_mat_group=4,
                                    p_frac_group=5,
                                    groups=flow,
                                ).mat,
                                bsize=1,
                            ),
                            elim_options={
                                "pc_type": "gamg",
                                "mg_levels_ksp_type": "richardson",
                                "mg_levels_ksp_max_it": 4,
                                "mg_levels_pc_type": "ilu",
                                "mg_levels_pc_factor_levels": 1,
                            }
                            | inner_ksp_monitor_options,
                            keep_options={
                                # "ksp_type": "gmres",
                                # "ksp_rtol": 1e-3,
                            }
                            | inner_ksp_monitor_options,
                            near_null_space=build_mechanics_near_null_space(self),
                            ksp_keep_use_pmat=True,
                            complement=PetscFieldSplitScheme(
                                groups=flow,
                                elim_options={
                                    # "pc_type": "lu"
                                    "pc_type": "gamg",
                                    "pc_gamg_threshold": 0.02,
                                    "mg_levels_ksp_type": "richardson",
                                    "mg_levels_ksp_max_it": 4,
                                    "mg_levels_pc_type": "sor",
                                },
                            ),
                        ),
                    ),
                ),
            ),
        )


def make_reorder_contact(model: IterativeHMSolver, contact_group: int) -> np.ndarray:
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
    if len(model.equation_groups[contact_group]) == 0:
        return reorder

    # Get the (fine-scale, not block(!)) dofs of the contact mechanics equations.
    dofs_contact = np.concatenate(
        [model.eq_dofs[i] for i in model.equation_groups[contact_group]]
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


def build_mechanics_near_null_space(
    model: IterativeHMSolver, include_sd=True, include_intf=True
):
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
