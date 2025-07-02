from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import porepy as pp
import scipy.sparse
from porepy.models.solution_strategy import ContactIndicators


@dataclass
class LinearSolveStats:
    simulation_dt: float = -1
    krylov_iters: int = -1
    petsc_converged_reason: int = -100
    error_matrix_contribution: float = -1
    num_sticking: int = -1
    num_sliding: int = -1
    num_open: int = -1
    # Assumptions
    coulomb_mismatch: float = -1
    sticking_u_mismatch: float = -1
    lambdan_max: float = -1
    lambdat_max: float = -1
    un_max: float = -1
    ut_max: float = -1
    # Matrix saving
    matrix_id: str = ""
    rhs_id: str = ""
    state_id: str = ""
    iterate_id: str = ""
    # Thermal
    temp_min: float = -1
    temp_max: float = -1
    cfl: float = -1
    enthalpy_max: float = -1
    enthalpy_mean: float = -1
    fourier_max: float = -1
    fourier_mean: float = -1
    # TO BE REMOVED:
    peclet_max: float = -1
    peclet_mean: float = -1


@dataclass
class TimeStepStats:
    linear_solves: list[LinearSolveStats] = field(default_factory=list)
    nonlinear_convergence_status: int = 1  # 1 converged -1 diverged

    @classmethod
    def from_json(cls, json: str) -> TimeStepStats:
        data = cls(**json)
        tmp = []
        for x in data.linear_solves:
            payload = {
                k: v for k, v in x.items() if k in LinearSolveStats.__dataclass_fields__
            }
            tmp.append(LinearSolveStats(**payload))
        data.linear_solves = tmp
        return data


def dump_json(name: str, data: list[TimeStepStats]) -> None:
    save_path = Path("./stats")
    save_path.mkdir(exist_ok=True)
    try:
        dict_data = [asdict(x) for x in data]
    except TypeError:
        dict_data = data
    json_data = json.dumps(dict_data)
    with open(save_path / name, "w") as file:
        file.write(json_data)


class StatisticsSavingMixin(ContactIndicators):
    _linear_solve_stats: LinearSolveStats
    _time_step_stats: TimeStepStats

    @cached_property
    def statistics(self) -> list[TimeStepStats]:
        return []

    def simulation_name(self) -> str:
        name = "stats_linear_solver"
        return name

    def before_nonlinear_loop(self) -> None:
        self._time_step_stats = TimeStepStats()
        self.statistics.append(self._time_step_stats)
        config = self.params.get("linear_solver_config", {})
        if config.get("logging", False):
            print()
            DAY = 24 * 60 * 60
            time = self.time_manager.time / DAY
            dt = self.time_manager.dt / DAY
            print(f"Sim time: {time:.2e}, Dt: {dt:.2e} (days)")
        super().before_nonlinear_loop()

    def after_nonlinear_convergence(self) -> None:
        config = self.params.get("linear_solver_config", {})
        if config.get("save_statistics", False):
            dump_json(self.simulation_name() + ".json", self.statistics)
        super().after_nonlinear_convergence()

    def after_nonlinear_failure(self) -> None:
        self._time_step_stats.nonlinear_convergence_status = -1
        config = self.params.get("linear_solver_config", {})
        if config.get("save_statistics", False):
            dump_json(self.simulation_name() + ".json", self.statistics)
        if config.get("logging", False):
            print("Time step did not converge")
        super().after_nonlinear_failure()

    def before_nonlinear_iteration(self) -> None:
        self._linear_solve_stats = LinearSolveStats()
        super().before_nonlinear_iteration()
        # self.collect_stats_sticking_sliding_open()
        # self.collect_stats_ut_mismatch()
        # self.collect_stats_coulomb_mismatch()
        # self.collect_stats_u_lambda_max()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        config = self.params.get("linear_solver_config", {})
        if config.get("logging", False):
            print(
                f"Newton iter: {len(self._time_step_stats.linear_solves)}, "
                f"Krylov iters: {self._linear_solve_stats.krylov_iters}"
            )
        self._linear_solve_stats.simulation_dt = self.time_manager.dt
        self._time_step_stats.linear_solves.append(self._linear_solve_stats)
        # if self.params["linear_solver_config"].get("save_matrix", False):
        #     self.save_matrix_state()
        config = self.params.get("linear_solver_config", {})
        if config.get("save_statistics", False):
            from FTHM_Solver.plot_utils import write_dofs_info

            dump_json(self.simulation_name() + ".json", self.statistics)
            write_dofs_info(self)

        super().after_nonlinear_iteration(solution_vector)

    def sticking_sliding_open(self):
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        opening = self.opening_indicator(fractures).value(self.equation_system) < 0
        closed = np.logical_not(opening)
        sliding = np.logical_and(
            closed, self.sliding_indicator(fractures).value(self.equation_system) > 0
        )
        sticking = np.logical_not(opening | sliding)

        return sticking, sliding, opening

    def collect_stats_sticking_sliding_open(self):
        data = self.sticking_sliding_open()
        self._linear_solve_stats.num_sticking = int(sum(data[0]))
        self._linear_solve_stats.num_sliding = int(sum(data[1]))
        self._linear_solve_stats.num_open = int(sum(data[2]))
        print(
            f"sticking: {self._linear_solve_stats.num_sticking}, "
            f"sliding: {self._linear_solve_stats.num_sliding}, "
            f"open: {self._linear_solve_stats.num_open}"
        )

    def collect_stats_ut_mismatch(self):
        sticking, _, _ = self.sticking_sliding_open()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(fractures)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(fractures)
        u_t_increment = pp.ad.time_increment(u_t).value(self.equation_system)

        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            fractures, dim=self.nd - 1
        )
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        ).value(self.equation_system)
        sticking = (scalar_to_tangential @ sticking).astype(bool)

        u_t_sticking = u_t_increment[sticking]
        try:
            self._linear_solve_stats.sticking_u_mismatch = abs(u_t_sticking).max()
        except ValueError:
            self._linear_solve_stats.sticking_u_mismatch = 0

    def collect_stats_coulomb_mismatch(self):
        _, sliding, _ = self.sticking_sliding_open()

        fractures = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(fractures)
        t_t = (nd_vec_to_tangential @ self.contact_traction(fractures)).value(
            self.equation_system
        )
        b = self.friction_bound(fractures).value(self.equation_system)
        tangential_basis = self.basis(fractures, dim=self.nd - 1)
        t_t_nrm = np.sqrt(sum(comp._mat.T @ t_t**2 for comp in tangential_basis))

        diff = (-t_t_nrm + b)[sliding]
        try:
            self._linear_solve_stats.coulomb_mismatch = abs(diff).max()
        except ValueError:
            self._linear_solve_stats.coulomb_mismatch = 0

    def collect_stats_u_lambda_max(self):
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(fractures)
        nd_vec_to_normal = self.normal_component(fractures)

        t = self.contact_traction(fractures)
        u = self.displacement_jump(fractures)

        t_n = (nd_vec_to_normal @ t).value(self.equation_system)
        t_t = (nd_vec_to_tangential @ t).value(self.equation_system)
        u_n = (nd_vec_to_normal @ u).value(self.equation_system)
        u_t = (nd_vec_to_tangential @ u).value(self.equation_system)

        try:
            self._linear_solve_stats.lambdan_max = abs(t_n).max()
            self._linear_solve_stats.lambdat_max = abs(t_t).max()
            self._linear_solve_stats.un_max = abs(u_n).max()
            self._linear_solve_stats.ut_max = abs(u_t).max()
        except ValueError:
            pass

    def save_matrix_state(self):
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = f"{self.simulation_name()}_{int(time.time() * 1000)}"
        print("Saving matrix", name)
        mat_id = f"{name}.npz"
        rhs_id = f"{name}_rhs.npy"
        state_id = f"{name}_state.npy"
        iterate_id = f"{name}_iterate.npy"
        scipy.sparse.save_npz(save_path / mat_id, self.bmat.mat)
        np.save(save_path / rhs_id, rhs)
        np.save(
            save_path / state_id,
            self.equation_system.get_variable_values(time_step_index=0),
        )
        np.save(
            save_path / iterate_id,
            self.equation_system.get_variable_values(iterate_index=0),
        )
        self._linear_solve_stats.iterate_id = iterate_id
        self._linear_solve_stats.state_id = state_id
        self._linear_solve_stats.matrix_id = mat_id
        self._linear_solve_stats.rhs_id = rhs_id
