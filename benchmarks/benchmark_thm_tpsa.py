import logging

import numpy as np
import porepy as pp

from pathlib import Path

from porepy.applications.material_values.fluid_values import water
from porepy.applications.material_values.solid_values import granite
from porepy.applications.test_utils.models import add_mixin
from porepy.applications.md_grids.domains import nd_cube_domain


class ThmModel(
    pp.constitutive_laws.CubicLawPermeability,
    pp.poromechanics.TpsaPoromechanicsMixin,
    pp.Thermoporomechanics,
):
    # MARK: Geometry

    @property
    def domain_size(self) -> pp.number:
        return self.params["domain_size"]

    def set_fractures(self) -> None:
        """Assigns 0 to 3 fractures."""
        fracture_indices = self.params.get("fracture_indices", [0])
        all_fractures = orthogonal_fractures_3d(self.domain_size)
        self._fractures = [all_fractures[i] for i in fracture_indices]

    def set_domain(self) -> None:
        """Set the cube domain."""
        self._domain = nd_cube_domain(3, self.domain_size)

    def grid_type(self):
        return "cartesian"

    # MARK: BCs

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))

        t_n = self.units.convert_units(-1e7, "Pa")
        t_n = np.tile(t_n, (bg.num_cells, 1)).T
        values[:, domain_sides.north] = t_n[:, domain_sides.north]
        return values.ravel("F")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        sd = bg.parent

        # This somehow accounts for the reference pressure inside, so we don't need to
        # subtract it.
        values = np.zeros(bg.num_cells)

        if self.is_well_grid(sd):
            well = self.well_network.wells[sd.tags["parent_well_index"]]
            well_tag = well.tags["well_name"]
            # Find indices of the well boundary sides.
            domain_sides = self.domain_boundary_sides(bg)
            # The top of the domain is '.top' in 3d, '.north' in 2d.
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            values[inds] = self.get_well_pressure(well_tag)
        return values

    # MARK: Wells

    injection_well_name = "injection_well"
    production_well_name = "production_well"

    def set_well_network(self) -> None:
        # We assume a cube
        dx = dy = dz = self.domain_size
        well_1 = pp.Well(
            np.array(
                [
                    [0.5 * dx, 0.5 * dx],
                    [0.4 * dy, 0.4 * dy],
                    [0.5 * dz, 1.0 * dz],
                ]
            ),
            tags={"well_name": self.injection_well_name},
        )
        well_2 = pp.Well(
            np.array(
                [
                    [0.5 * dx, 0.5 * dx],
                    [0.6 * dy, 0.6 * dy],
                    [0.6 * dz, 1.0 * dz],
                ]
            ),
            tags={"well_name": self.production_well_name},
        )
        self._wells = [well_1, well_2]

        mesh_size = self.params.get("well_mesh_size", {"mesh_size": 0.1 * dz})
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=self._wells, parameters=mesh_size
        )

    def get_well_pressure(self, well_tag: str) -> float:
        # The units should be already converted here.
        if well_tag == self.injection_well_name:
            return self.params["well_pressure_injection"]
        elif well_tag == self.production_well_name:
            return self.params["well_pressure_production"]
        else:
            raise ValueError(well_tag)

    # MARK: Initial conditions

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.reference_variable_values.pressure


def orthogonal_fractures_3d(size: pp.number) -> list[pp.PlaneFracture]:
    # They are not touching the boundary.
    coords_a = [0.5, 0.5, 0.5, 0.5]
    coords_b = [0.1, 0.1, 0.9, 0.9]
    coords_c = [0.1, 0.9, 0.9, 0.1]
    pts = []
    pts.append(np.array([coords_a, coords_b, coords_c]) * size)
    pts.append(np.array([coords_b, coords_a, coords_c]) * size)
    pts.append(np.array([coords_b, coords_c, coords_a]) * size)
    return [pp.PlaneFracture(pts[i]) for i in range(3)]


# MARK: Runscript


def prepare_thm_tpsa_model(
    direct_solver: bool = False,
    radial_return: bool = False,
    constant_dt=False,
    grid_refinement_level: int = 0,
):
    model_class = ThmModel
    if not direct_solver:
        import pp_solvers

        model_class = add_mixin(pp_solvers.IterativeSolverMixin, model_class)

    if radial_return:
        from porepy.models.contact_mechanics import RadialReturnFormulation

        model_class = add_mixin(RadialReturnFormulation, model_class)

    units = pp.Units(kg=1e10)
    time_manager = pp.TimeManager(
        schedule=[0, 1 * pp.MINUTE if constant_dt else 1 * pp.HOUR],
        dt_init=1 * pp.SECOND,
        constant_dt=constant_dt,
        iter_max=12,
        iter_optimal_range=(8, 12),
        print_info=True,
    )

    model_params = {
        # common
        "time_manager": time_manager,
        "units": units,
        # materials
        "material_constants": {
            "solid": pp.SolidConstants(units=units, **granite),
            "fluid": pp.FluidComponent(units=units, **water),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            units=units,
            pressure=units.convert_units(0, "Pa"),
        ),
        # boundary conditions
        "well_pressure_injection": units.convert_units(6e7, "Pa"),
        "well_pressure_production": units.convert_units(-1e7, "Pa"),
        # geometry
        "fracture_indices": [1, 2],
        "domain_size": units.convert_units(1000, "m"),
        "meshing_arguments": {
            "cell_size": units.convert_units(200, "m") / (2**grid_refinement_level),
        },
        "grid_refinement_level": grid_refinement_level,
        # output
        "folder_name": Path(__file__).parent.parent
        / "visualization"
        / f"{'radial_return' if radial_return else 'standard'}_{'direct' if direct_solver else 'iterative'}_{'constant_dt' if constant_dt else 'adaptive_dt'}_grid_x{grid_refinement_level}",
        "solver_statistics_file_name": "solver_statistics_hm.json",
    }
    if not direct_solver:
        model_params["linear_solver"] = pp_solvers.LinearSolverParams(
            preconditioner_factory=pp_solvers.thm_tpsa_factory,
            options={
                # "gmres": {"ksp_monitor": None},
                # "contact": {"ksp_monitor": None},
                # "interface_flow": {"ksp_monitor": None},
                # "intf_force_balance": {"ksp_monitor": None},
                # "tpsa_fieldsplit": {"ksp_monitor": None},
                # "solid_mass_pressure_amg": {"ksp_monitor": None},
                # "angular_momentum_rotation": {"ksp_monitor": None},
                # "mechanics_amg": {"ksp_monitor": None},
                # "cpr0_energy": {"ksp_monitor": None},
                # "cpr0_mass": {"ksp_monitor": None},
                # "cpr1": {"ksp_monitor": None},
            },
        )

    return model_class(model_params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = prepare_thm_tpsa_model(direct_solver=False, grid_refinement_level=0)
    pp.ModelRunner(
        model,
        {
            "nl_convergence_tol_res": 1e-6,
            "nl_max_iterations": 20,
        },
    ).run()
