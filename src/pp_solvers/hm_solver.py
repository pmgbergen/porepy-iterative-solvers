import numpy as np
import porepy as pp


def build_mechanics_near_null_space(
    model: pp.PorePyModel, include_sd=True, include_intf=True
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
