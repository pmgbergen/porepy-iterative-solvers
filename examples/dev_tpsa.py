"""The code below is based on the setup used for unit tests in porepy, see
porepy/tests/models/test_thermoporomechanics.py. It is tested and works against porepy
develop branch, commit befcea47c93b3e863cf9b96a5557c4c233ccfdb5

"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import porepy as pp
from porepy.applications.test_utils.models import Thermoporomechanics, add_mixin

import pp_solvers


class NonzeroFractureGapPoromechanics(pp.PorePyModel):
    """Adjust bc values and initial condition."""

    pressure_variable: str
    displacement_variable: str
    interface_displacement_variable: str
    fracture_stress: Callable[[list[pp.MortarGrid]], pp.ad.Operator]

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        # Initial pressure equals reference pressure.
        return self.reference_variable_values.pressure * np.ones(sd.num_cells)

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        # Set initial displacement compatible with fracture gap for matrix subdomain.
        if len(self.mdg.subdomains()) > 1:
            top_cells = sd.cell_centers[1] > self.units.convert_units(0.5, "m")
            vals = np.zeros((self.nd, sd.num_cells))
            vals[1, top_cells] = self.solid.fracture_gap
            return vals.ravel("F")
        else:
            # Call super to return expected trivial values, because this class is used
            # in other test cases as well.
            return super().ic_values_displacement(sd)

    def ic_values_interface_displacement(self, intf: pp.MortarGrid) -> np.ndarray:
        # Set initial displacement compatible with fracture gap for matrix-fracture
        # interface.
        if len(self.mdg.subdomains()) > 1:
            sd = self.mdg.subdomains()[0]
            faces_primary = intf.primary_to_mortar_int().tocsr().indices
            switcher = pp.grid_utils.switch_sign_if_inwards_normal(
                sd,
                self.nd,
                faces_primary,
            )

            normals = (switcher * sd.face_normals[: sd.dim].ravel("F")).reshape(
                sd.dim, -1, order="F"
            )
            intf_normals = normals[:, faces_primary]
            top_cells = intf_normals[1, :] < 0

            # Set mortar displacement to zero on bottom and fracture gap value on top
            vals = np.zeros((self.nd, intf.num_cells))
            vals[1, top_cells] = (
                self.solid.fracture_gap + self.solid.maximum_elastic_fracture_opening
            )
            return vals.ravel("F")
        else:
            # Call super to return expected trivial values, because this class is used
            # in other test cases as well.
            return super().ic_values_interface_displacement(intf)

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        internal_boundaries = super().fluid_source(subdomains)
        if "fracture_source_value" not in self.params:
            return internal_boundaries

        vals = []
        for sd in subdomains:
            if sd.dim == self.nd:
                vals.append(np.zeros(sd.num_cells))
            else:
                val = self.units.convert_units(
                    self.params["fracture_source_value"], "kg * s ^ -1"
                )
                # Distribute source term over cells based on cell volumes.
                vals.append(val * sd.cell_volumes / np.sum(sd.cell_volumes))
        fracture_source = pp.wrap_as_dense_ad_array(
            np.hstack(vals), name="fracture_fluid_source"
        )
        return internal_boundaries + fracture_source


class TailoredThermoporomechanics(
    NonzeroFractureGapPoromechanics,
    pp.model_boundary_conditions.TimeDependentMechanicalBCsDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsEnergyDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    Thermoporomechanics,
):
    pass


class TailoredThermoporomechanicsTpsa(
    pp.poromechanics.TpsaPoromechanicsMixin, TailoredThermoporomechanics
):
    pass


def create_fractured_model(
    solid_vals: dict, fluid_vals: dict, params: dict, model_class: type
) -> TailoredThermoporomechanics:
    """Create a model for a 2d problem with a single fracture.

    Parameters:
        solid_vals: Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        fluid_vals: Dictionary with keys as those in :class:`pp.FluidComponent`
            and corresponding values.
        params: Dictionary with keys as those in params of
            :class:`TailoredThermoporomechanics`.

    Returns:
        model: Model object for the problem.

    """
    # Instantiate constants and store in params.
    solid_vals["fracture_gap"] = 0.042
    solid_vals["residual_aperture"] = 1e-10
    solid_vals["biot_coefficient"] = 1.0
    solid_vals["thermal_expansion"] = 1e-1
    fluid_vals["compressibility"] = 1
    fluid_vals["thermal_expansion"] = 1e-1
    solid = pp.SolidConstants(**solid_vals)
    fluid = pp.FluidComponent(**fluid_vals)

    default = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "nl_max_iterations": 20,
        "nl_convergence_inc_atol": 1e-6,
    }
    default.update(params)
    if issubclass(model_class, TailoredThermoporomechanicsTpsa):
        # Tpsa is only consistent with Cartesian grids.
        default["cartesian"] = True

    model = model_class(default)
    return model


def main():
    logging.basicConfig(level=logging.INFO)
    model_class = add_mixin(
        pp_solvers.IterativeSolverMixin, TailoredThermoporomechanicsTpsa
    )
    model_params = {
        "u_north": [0.0, 0.001],
        "linear_solver": pp_solvers.LinearSolverParams(
            preconditioner_factory=make_linear_solver,
            options={
                "tpsa_fieldsplit": {
                    "pc_fieldsplit_type": "multiplicative",
                }
            },
        ),
    }
    model = create_fractured_model({}, {}, model_params, model_class)
    pp.ModelRunner(
        model,
    ).run()


def make_linear_solver():
    from pp_solvers.equation_variable_groups import (
        ContactMechanicsGroup,
        CustomEquationVariableGroup,
        EnergyBalanceTemperatureGroup,
        EquationVariableGroup,
        InterfaceDarcyFluxGroup,
        InterfaceEnthalpyFluxGroup,
        InterfaceForceBalanceGroup,
        InterfaceFourierFluxGroup,
        MassBalancePressureFracturesGroup,
        MassBalancePressureIntersectionsGroup,
        MassBalancePressureMatrixGroup,
        MechanicsGroup,
        WellEnthalpyFluxGroup,
        WellFluxGroup,
    )
    from pp_solvers.fixed_stress import construct_fixed_stress_block_matrix
    from pp_solvers.petsc_utils import csr_to_petsc
    from pp_solvers.preconditioners import (
        AMG,
        GMRES,
        ILU,
        BlockDiagonalInverter,
        BlockDiagonalPreconditioner,
        CompositePreconditioner,
        DiagonalInverter,
        FieldSplitAdditive,
        FixedStressInverter,
        Identity,
        NoInverter,
        LinearSolverConfiguration,
        PythonPermutationWrapper,
        nested_schur_complements,
    )
    from pp_solvers.transformations import (
        ContactLinearTransformation,
        LinearSystemTransformation,
        ScaleSpecificVolume,
    )

    contact_groups: list[EquationVariableGroup] = [ContactMechanicsGroup()]
    interface_groups: list[EquationVariableGroup] = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        MassBalancePressureMatrixGroup(),
        MassBalancePressureFracturesGroup(),
        MassBalancePressureIntersectionsGroup(),
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        EnergyBalanceTemperatureGroup(),
    ]

    solid_mass_pressure_group = CustomEquationVariableGroup(
        "Solid_mass_equation_poromechanics", "total_pressure"
    )
    angular_momentum_rotation_group = CustomEquationVariableGroup(
        "angular_momentum_balance_equation", "rotation_stress"
    )

    solver = GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_inverter": BlockDiagonalInverter(),
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=[InterfaceForceBalanceGroup()], key="intf_force_balance"
                    ),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": FieldSplitAdditive(
                        key="tpsa_fieldsplit",
                        subsolvers=[
                            AMG(
                                groups=[solid_mass_pressure_group],
                                key="solid_mass_pressure_amg",
                                vector_problem=True,
                            ),
                            BlockDiagonalPreconditioner(
                                groups=[angular_momentum_rotation_group],
                                key="angular_momentum_rotation",
                            ),
                            AMG(
                                groups=[MechanicsGroup()],
                                key="mechanics_amg",
                                vector_problem=True,
                            ),
                        ],
                    ),
                    "approximate_inverter": FixedStressInverter(),
                },
                {
                    "subsolver": CompositePreconditioner(
                        subsolvers=[
                            FieldSplitAdditive(
                                subsolvers=[
                                    Identity(
                                        groups=energy_balance_groups,
                                        key="cpr0_energy",
                                    ),
                                    AMG(groups=mass_balance_groups, key="cpr0_mass"),
                                ],
                            ),
                            PythonPermutationWrapper(
                                permutation_groups=[
                                    energy_balance_groups,
                                    mass_balance_groups,
                                ],
                                inner_subsolver=ILU(
                                    groups=energy_balance_groups + mass_balance_groups,
                                    key="cpr1",
                                ),
                            ),
                        ]
                    )
                },
            ]
        )
    )

    return LinearSolverConfiguration(
        transformations=[
            ContactLinearTransformation(),
            ScaleSpecificVolume(groups=[EnergyBalanceTemperatureGroup()]),
        ],
        solver=solver,
    )


if __name__ == "__main__":
    main()
