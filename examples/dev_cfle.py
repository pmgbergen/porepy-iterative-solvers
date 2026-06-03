"""YZ: 2D model for injection cold water-CO2 mixture into hot domain. Based on the code
by Veljko Lipovac in porepy. This script runs consistently with porepy (nonlinear
divergence) on branch composite-flow, commit 7b13a5c9a8b386d32360d7e9d4827ef41a9b23f3

You need to disable the line os.environ["NUMBA_DISABLE_JIT"] = "0" in run_f2d.py

"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import porepy as pp
from porepy.applications.test_utils.models import add_mixin
from porepy.examples.cold_injection.config import (
    get_default_convergence_criteria,
)
from porepy.examples.cold_injection.run_f2d import (
    BUOYANCY_ON,
    ModelClass,
    max_iterations,
    model_params,
    newton_tol_inc,
    newton_tol_res,
    newton_tol_res_isofug,
    solver_params,
    set_schur_complement,
)

import pp_solvers


def cfle_factory():
    from porepy.numerics.ad.operators import MixedDimensionalVariable
    from dataclasses import dataclass

    from pp_solvers.equation_variable_groups import (
        EquationOnDomains,
        EquationNames,
        CustomEquationVariableGroup,
        OnWell,
        NotOnWell,
        EquationVariableGroup,
        InterfaceDarcyFluxGroup,
        InterfaceEnthalpyFluxGroup,
        InterfaceFourierFluxGroup,
        WellEnthalpyFluxGroup,
        WellFluxGroup,
    )
    from pp_solvers.preconditioners import (
        GMRES,
        FieldSplitSchur,
        ILU,
        DiagonalInverter,
        AMG,
        LinearSolverConfiguration,
        SchurComplementReduction,
        CompositePreconditioner,
        Identity,
    )

    @dataclass(frozen=True)
    class ComponentMassBalanceCO2Group(EquationVariableGroup):
        def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
            name = "component_mass_balance_equation_CO2"
            return EquationOnDomains(name=name, domains=model.mdg.subdomains())

        def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
            return model.fluid.components[1].fraction(model.mdg.subdomains())

    interface_groups = [
        InterfaceDarcyFluxGroup(),
        InterfaceEnthalpyFluxGroup(),
        InterfaceFourierFluxGroup(),
        WellFluxGroup(),
        WellEnthalpyFluxGroup(),
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        CustomEquationVariableGroup(
            EquationNames.MASS_BALANCE.value,
            "pressure",
            defined_on=NotOnWell("production"),
        ),
        CustomEquationVariableGroup(
            "production_pressure_constraint",
            "pressure",
            defined_on=OnWell("production"),
        ),
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        CustomEquationVariableGroup(
            EquationNames.ENERGY_BALANCE.value,
            "enthalpy",
            defined_on=NotOnWell("injection"),
        ),
        CustomEquationVariableGroup(
            "injection_temperature_constraint",
            "enthalpy",
            defined_on=OnWell("injection"),
        ),
    ]
    component_groups: list[EquationVariableGroup] = [ComponentMassBalanceCO2Group()]

    secondary_groups: list[EquationVariableGroup] = [
        CustomEquationVariableGroup("local_component_mass_constraint_CO2", "x_CO2_L"),
        CustomEquationVariableGroup("isofugacity_constraint_H2O_G_L", "x_H2O_G"),
        CustomEquationVariableGroup("isofugacity_constraint_CO2_G_L", "x_CO2_G"),
        CustomEquationVariableGroup("semismooth_complementary_condition_L", "y_G"),
        CustomEquationVariableGroup("semismooth_complementary_condition_G", "s_G"),
        CustomEquationVariableGroup("local_fluid_enthalpy_constraint", "temperature"),
        CustomEquationVariableGroup("local_phase_mass_constraint_G", "x_H2O_L"),
    ]

    solver = GMRES(
        preconditioner=FieldSplitSchur(
            subsolver=ILU(groups=interface_groups, key="interface_prec"),
            approximate_inverter=DiagonalInverter(),
            complement_solver=CompositePreconditioner(
                subsolvers=[
                    FieldSplitSchur(
                        subsolver=Identity(
                            groups=energy_balance_groups + component_groups,
                            key="cpr_stage0_identity",
                        ),
                        approximate_inverter=DiagonalInverter(),
                        complement_solver=AMG(
                            groups=mass_balance_groups, key="cpr_stage0_amg"
                        ),
                        key="inner_fieldsplit",
                    ),
                    ILU(
                        groups=energy_balance_groups
                        + component_groups
                        + mass_balance_groups,
                        key="cpr_stage1_ilu",
                    ),
                ]
            ),
        )
    )

    return LinearSolverConfiguration(
        transformations=[SchurComplementReduction(primary_groups=solver.groups)],
        solver=solver,
        groups=secondary_groups + solver.groups,
    )


BUOYANCY_ON = True

if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%I-%M-%S")
    sub_folder = f"f2d_{timestamp}_BUOY_{BUOYANCY_ON}"
    model_params["folder_name"] = f"visualization/{sub_folder}"

    ModelClass = add_mixin(pp_solvers.IterativeSolverMixin, ModelClass)
    model_params["linear_solver"] = pp_solvers.LinearSolverParams(
        preconditioner_factory=cfle_factory
    )
    model_params["apply_schur_complement_reduction"] = False

    model = ModelClass(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    # Defining sub system for Schur complement reduction.
    # set_schur_complement(model)
    solver_params.update(
        get_default_convergence_criteria(
            model, max_iterations, newton_tol_res, newton_tol_inc, newton_tol_res_isofug
        )
    )

    t_0 = time.time()
    pp.run_time_dependent_model(model, solver_params)
    sim_time = time.time() - t_0

    print(f"Simulation prepared after {prep_sim_time:.2f} (s).")
    print(f"Simulation finished after {sim_time / 60.0:.2f} (m).")
