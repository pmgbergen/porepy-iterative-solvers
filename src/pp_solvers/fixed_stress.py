import numpy as np
import porepy as pp
import scipy.sparse as sps

from .block_matrix import BlockMatrixStorage


def get_fixed_stress_stabilization(model, l_factor: float = 0.6) -> sps.spmatrix:
    """Define the fixed stress stabilization matrix."""

    mu_lame = model.solid.shear_modulus
    lambda_lame = model.solid.lame_lambda
    alpha_biot = model.solid.biot_coefficient
    dim = model.nd

    subdomains = model.mdg.subdomains(dim=dim)
    cell_volumes = subdomains[0].cell_volumes
    if alpha_biot == 0:
        return sps.diags(0 * cell_volumes)

    # Stabilization value determined by physical reasoning.
    l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
    # Stabilization value determined by theoretical reasoning.
    l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

    # TODO: Where does this formula come from?
    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    diagonal_approx *= cell_volumes

    density = model.equation_system.evaluate(model.fluid.density(subdomains))
    diagonal_approx *= density

    dt = model.time_manager.dt
    diagonal_approx /= dt

    return sps.diags(diagonal_approx)


def get_fs_fractures_analytical(model: pp.PorePyModel) -> sps.spmatrix:
    alpha_biot = model.solid.biot_coefficient  # [-]
    lame_lambda = model.solid.lame_lambda  # [Pa]
    M = 1 / model.solid.specific_storage  # [Pa]
    compressibility = model.fluid.components[0].compressibility  # [1 / Pa]
    porosity = model.solid.porosity

    fractures = model.mdg.subdomains(dim=model.nd - 1)

    if len(fractures) == 0:
        return sps.csr_matrix((0, 0))

    nd_vec_to_normal = model.normal_component(fractures)
    # The normal component of the contact traction and the displacement jump.
    u_n_operator = nd_vec_to_normal @ model.displacement_jump(fractures)
    u_n = model.equation_system.evaluate(u_n_operator)

    if compressibility != 0:
        val = (
            alpha_biot**2
            * u_n  # / resid_aperture# ** 3
            / (lame_lambda / (compressibility * M) + porosity * lame_lambda)
        )
    else:
        val = 0

    cell_volumes = np.concatenate([f.cell_volumes for f in fractures])
    val *= cell_volumes

    density = model.equation_system.evaluate(model.fluid.density(fractures))
    val *= density

    dt = model.time_manager.dt
    val /= dt

    return sps.diags(val)


def make_fs_analytical_slow_new(
    model, J: BlockMatrixStorage, p_mat_group: int, p_frac_group: int, groups: list[int]
) -> BlockMatrixStorage:
    index = J[groups].empty_container()
    diagonals = []
    for group in groups:
        if group == p_mat_group:
            diagonals.append(get_fixed_stress_stabilization(model))
        elif group == p_frac_group:
            diagonals.append(get_fs_fractures_analytical(model))
        else:
            diagonals.append(index[[group]].mat)
    index.mat = sps.block_diag(diagonals, format="csr")
    return index
