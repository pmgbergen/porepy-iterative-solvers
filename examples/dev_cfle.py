"""2D model for injection cold water-CO2 mixture into hot domain. Based on the code by
Veljko Lipovac in porepy run_f2d.py"""

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
from pp_solvers.preconditioners import cfle_factory

BUOYANCY_ON = True

if __name__ == "__main__":
    timestamp = datetime.today().strftime("%d%B%Y_%I-%M-%S")
    sub_folder = f"f2d_{timestamp}_BUOY_{BUOYANCY_ON}"
    model_params["folder_name"] = f"visualization/{sub_folder}"

    # ModelClass = add_mixin(pp_solvers.IterativeSolverMixin, ModelClass)
    # model_params["linear_solver"] = pp_solvers.LinearSolverParams(
    #     preconditioner_factory=cfle_factory
    # )

    model = ModelClass(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    # Defining sub system for Schur complement reduction.
    set_schur_complement(model)
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
