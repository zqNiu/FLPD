"""Canonical FLPD v2 model: cumulative drone payload + round-trip energy.

This module is the single source of truth for the mathematical model used by
all v2 scripts.  The customer-allocation variable x[i,j,k] is the fraction of
customer i's demand served by drone k based at field j.

The retained constraints are:

    sum_j y[j] <= P
    z[j,k] <= y[j]
    sum_j z[j,k] <= 1
    sum_jk x[i,j,k] == 1
    sum_i b[j,i] w[i] x[i,j,k] <= B[k] z[j,k]
    sum_i w[i] x[i,j,k] <= C[k] z[j,k]
    sum_ik w[i] x[i,j,k] <= U[j] y[j]

Here b[j,i] = 2 * alpha * d[j,i] is the round-trip energy coefficient per
unit of demand.  Demand is divisible over the strategic planning horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import math
import time

import numpy as np


MODEL_TAG = "FLPD-v2-cumulative-payload-roundtrip"


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be scalar or shape ({length},), got {arr.shape}")
    return arr


@dataclass
class ModelData:
    demand: np.ndarray
    transport_cost_ji: np.ndarray
    energy_ji: np.ndarray
    num_drones: int
    max_facilities: int
    setup_cost_j: Any
    facility_capacity_j: Any
    battery_budget_k: Any
    payload_capacity_k: Any
    distance_ji: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.demand = np.asarray(self.demand, dtype=float)
        self.transport_cost_ji = np.asarray(self.transport_cost_ji, dtype=float)
        self.energy_ji = np.asarray(self.energy_ji, dtype=float)
        if self.demand.ndim != 1 or self.demand.size == 0:
            raise ValueError("demand must be a nonempty one-dimensional array")
        if self.num_drones <= 0:
            raise ValueError("num_drones must be positive")
        if self.transport_cost_ji.ndim != 2:
            raise ValueError("transport_cost_ji must have shape (J, I)")
        j_count, i_count = self.transport_cost_ji.shape
        if i_count != self.demand.size:
            raise ValueError("transport_cost_ji and demand have inconsistent I")
        if self.energy_ji.shape != (j_count, i_count):
            raise ValueError("energy_ji must have the same shape as transport_cost_ji")
        if self.distance_ji is not None:
            self.distance_ji = np.asarray(self.distance_ji, dtype=float)
            if self.distance_ji.shape != (j_count, i_count):
                raise ValueError("distance_ji must have shape (J, I)")
        if not 1 <= self.max_facilities <= j_count:
            raise ValueError("max_facilities must be between 1 and J")

        self.setup_cost_j = _vector(self.setup_cost_j, j_count, "setup_cost_j")
        self.facility_capacity_j = _vector(
            self.facility_capacity_j, j_count, "facility_capacity_j"
        )
        self.battery_budget_k = _vector(
            self.battery_budget_k, self.num_drones, "battery_budget_k"
        )
        self.payload_capacity_k = _vector(
            self.payload_capacity_k, self.num_drones, "payload_capacity_k"
        )

        for name, arr, strictly_positive in (
            ("demand", self.demand, True),
            ("transport_cost_ji", self.transport_cost_ji, False),
            ("energy_ji", self.energy_ji, False),
            ("setup_cost_j", self.setup_cost_j, False),
            ("facility_capacity_j", self.facility_capacity_j, True),
            ("battery_budget_k", self.battery_budget_k, True),
            ("payload_capacity_k", self.payload_capacity_k, True),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            if strictly_positive and np.any(arr <= 0):
                raise ValueError(f"{name} must be strictly positive")
            if not strictly_positive and np.any(arr < 0):
                raise ValueError(f"{name} must be nonnegative")

        self.metadata = dict(self.metadata)
        self.metadata.setdefault("model_tag", MODEL_TAG)

    @property
    def num_customers(self) -> int:
        return int(self.demand.size)

    @property
    def num_candidates(self) -> int:
        return int(self.transport_cost_ji.shape[0])

    @property
    def c_ji(self) -> np.ndarray:
        return self.transport_cost_ji

    @property
    def b_ji(self) -> np.ndarray:
        return self.energy_ji

    @classmethod
    def from_geographic_csv(
        cls,
        csv_path: str | Path,
        *,
        num_drones: int,
        max_facilities: int,
        setup_cost: float,
        transport_cost_per_km: float,
        alpha: float,
        facility_capacity: float,
        battery_budget: float,
        payload_capacity: float,
    ) -> "ModelData":
        """Build a geographic instance using Haversine distance in kilometres.

        The energy coefficient is always round trip: b = 2 * alpha * distance.
        """
        import pandas as pd

        csv_path = Path(csv_path)
        frame = pd.read_csv(csv_path)
        required = {"type", "lat", "lon", "demand"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        customers = frame[frame["type"] == "demand_point"].reset_index(drop=True)
        fields = frame[frame["type"] == "depot"].reset_index(drop=True)
        if customers.empty or fields.empty:
            raise ValueError("CSV must contain demand_point and depot rows")

        radius_km = 6371.0
        lat_j = np.radians(fields["lat"].to_numpy(dtype=float))
        lon_j = np.radians(fields["lon"].to_numpy(dtype=float))
        lat_i = np.radians(customers["lat"].to_numpy(dtype=float))
        lon_i = np.radians(customers["lon"].to_numpy(dtype=float))
        dlat = lat_i[None, :] - lat_j[:, None]
        dlon = lon_i[None, :] - lon_j[:, None]
        hav = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_j[:, None])
            * np.cos(lat_i[None, :])
            * np.sin(dlon / 2.0) ** 2
        )
        hav = np.clip(hav, 0.0, 1.0)
        distance = radius_km * 2.0 * np.arctan2(np.sqrt(hav), np.sqrt(1.0 - hav))
        demand = customers["demand"].to_numpy(dtype=float)

        return cls(
            demand=demand,
            transport_cost_ji=transport_cost_per_km * distance,
            energy_ji=2.0 * alpha * distance,
            distance_ji=distance,
            num_drones=num_drones,
            max_facilities=max_facilities,
            setup_cost_j=setup_cost,
            facility_capacity_j=facility_capacity,
            battery_budget_k=battery_budget,
            payload_capacity_k=payload_capacity,
            metadata={
                "source_csv": str(csv_path),
                "energy_definition": "2*alpha*distance",
                "alpha": float(alpha),
                "transport_cost_per_km": float(transport_cost_per_km),
            },
        )


@dataclass
class Variables:
    y_j: np.ndarray
    z_jk: np.ndarray
    x_ijk: np.ndarray

    @classmethod
    def zeros(cls, data: ModelData) -> "Variables":
        return cls(
            y_j=np.zeros(data.num_candidates, dtype=float),
            z_jk=np.zeros((data.num_candidates, data.num_drones), dtype=float),
            x_ijk=np.zeros(
                (data.num_customers, data.num_candidates, data.num_drones),
                dtype=float,
            ),
        )

    def copy(self) -> "Variables":
        return Variables(self.y_j.copy(), self.z_jk.copy(), self.x_ijk.copy())


@dataclass
class LagrangianMultipliers:
    a_jk: np.ndarray
    beta_jk: np.ndarray
    delta_jk: np.ndarray
    gamma_j: np.ndarray

    @classmethod
    def zeros(cls, data: ModelData) -> "LagrangianMultipliers":
        shape = (data.num_candidates, data.num_drones)
        return cls(
            a_jk=np.zeros(shape),
            beta_jk=np.zeros(shape),
            delta_jk=np.zeros(shape),
            gamma_j=np.zeros(data.num_candidates),
        )


@dataclass
class FeasibilityReport:
    is_feasible: bool
    objective: float
    max_violation: dict[str, float]


@dataclass
class LagrangianResult:
    best_upper_bound: float
    best_lower_bound: float
    best_variables: Optional[Variables]
    history: list[dict[str, float]]
    iterations: int
    elapsed_seconds: float
    termination: str


def objective_value(data: ModelData, variables: Variables) -> float:
    fixed = float(np.dot(data.setup_cost_j, variables.y_j))
    transport = float(
        np.einsum(
            "ji,i,ijk->",
            data.transport_cost_ji,
            data.demand,
            variables.x_ijk,
        )
    )
    return fixed + transport


def validate_solution(
    data: ModelData, variables: Variables, tolerance: float = 1e-7
) -> FeasibilityReport:
    """Check every constraint of the canonical v2 model.

    A solution is never accepted as an upper bound without passing this check.
    """
    y = np.asarray(variables.y_j, dtype=float)
    z = np.asarray(variables.z_jk, dtype=float)
    x = np.asarray(variables.x_ijk, dtype=float)
    expected = (
        (data.num_candidates,),
        (data.num_candidates, data.num_drones),
        (data.num_customers, data.num_candidates, data.num_drones),
    )
    if y.shape != expected[0] or z.shape != expected[1] or x.shape != expected[2]:
        raise ValueError(
            f"solution shapes must be {expected}, got {(y.shape, z.shape, x.shape)}"
        )

    w = data.demand
    battery_load = np.einsum("ji,i,ijk->jk", data.energy_ji, w, x)
    payload_load = np.einsum("i,ijk->jk", w, x)
    facility_load = np.einsum("i,ijk->j", w, x)

    violations = {
        "field_count": max(0.0, float(y.sum() - data.max_facilities)),
        "drone_field_link": max(0.0, float(np.max(z - y[:, None]))),
        "one_field_per_drone": max(0.0, float(np.max(z.sum(axis=0) - 1.0))),
        "demand_balance": float(np.max(np.abs(x.sum(axis=(1, 2)) - 1.0))),
        "battery": max(
            0.0,
            float(np.max(battery_load - data.battery_budget_k[None, :] * z)),
        ),
        "cumulative_payload": max(
            0.0,
            float(np.max(payload_load - data.payload_capacity_k[None, :] * z)),
        ),
        "facility_capacity": max(
            0.0,
            float(np.max(facility_load - data.facility_capacity_j * y)),
        ),
        "x_lower_bound": max(0.0, float(-np.min(x))),
        "x_upper_bound": max(0.0, float(np.max(x - 1.0))),
        "y_binary": float(np.max(np.abs(y - np.rint(y)))),
        "z_binary": float(np.max(np.abs(z - np.rint(z)))),
    }
    feasible = all(value <= tolerance for value in violations.values())
    return FeasibilityReport(feasible, objective_value(data, variables), violations)


def solve_subproblem_x(
    data: ModelData, multipliers: LagrangianMultipliers
) -> tuple[np.ndarray, float]:
    """Exact SP1 solution for the demand-simplex constraints.

    Because cumulative payload is dualized, SP1 has no per-arc payload cap.
    Each customer is assigned to a minimum reduced-cost (j,k) pair.
    """
    reduced = (
        data.transport_cost_ji[:, :, None]
        + multipliers.beta_jk[:, None, :] * data.energy_ji[:, :, None]
        + multipliers.delta_jk[:, None, :]
        + multipliers.gamma_j[:, None, None]
    )
    by_customer = reduced.transpose(1, 0, 2).reshape(data.num_customers, -1)
    best = np.argmin(by_customer, axis=1)
    best_j, best_k = np.divmod(best, data.num_drones)
    x = np.zeros(
        (data.num_customers, data.num_candidates, data.num_drones), dtype=float
    )
    x[np.arange(data.num_customers), best_j, best_k] = 1.0
    value = float(np.dot(data.demand, by_customer[np.arange(data.num_customers), best]))
    return x, value


def solve_subproblem_y(
    data: ModelData, multipliers: LagrangianMultipliers
) -> tuple[np.ndarray, float]:
    coefficient = (
        data.setup_cost_j
        - multipliers.a_jk.sum(axis=1)
        - multipliers.gamma_j * data.facility_capacity_j
    )
    order = np.argsort(coefficient)
    selected = [
        int(j)
        for j in order[: data.max_facilities]
        if coefficient[j] < -1e-12
    ]
    y = np.zeros(data.num_candidates)
    y[selected] = 1.0
    return y, float(coefficient[selected].sum()) if selected else 0.0


def solve_subproblem_z(
    data: ModelData, multipliers: LagrangianMultipliers
) -> tuple[np.ndarray, float]:
    coefficient = (
        multipliers.a_jk
        - multipliers.beta_jk * data.battery_budget_k[None, :]
        - multipliers.delta_jk * data.payload_capacity_k[None, :]
    )
    best_j = np.argmin(coefficient, axis=0)
    best_value = coefficient[best_j, np.arange(data.num_drones)]
    use = best_value < -1e-12
    z = np.zeros((data.num_candidates, data.num_drones))
    z[best_j[use], np.arange(data.num_drones)[use]] = 1.0
    return z, float(best_value[use].sum())


def lagrangian_value_and_subgradients(
    data: ModelData,
    variables: Variables,
    multipliers: LagrangianMultipliers,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    y, z, x = variables.y_j, variables.z_jk, variables.x_ijk
    w = data.demand
    g_a = z - y[:, None]
    g_beta = (
        np.einsum("ji,i,ijk->jk", data.energy_ji, w, x)
        - data.battery_budget_k[None, :] * z
    )
    g_delta = (
        np.einsum("i,ijk->jk", w, x)
        - data.payload_capacity_k[None, :] * z
    )
    g_gamma = (
        np.einsum("i,ijk->j", w, x) - data.facility_capacity_j * y
    )
    value = objective_value(data, variables)
    value += float(np.sum(multipliers.a_jk * g_a))
    value += float(np.sum(multipliers.beta_jk * g_beta))
    value += float(np.sum(multipliers.delta_jk * g_delta))
    value += float(np.sum(multipliers.gamma_j * g_gamma))
    return value, (g_a, g_beta, g_delta, g_gamma)


def _facility_transport_lp(
    data: ModelData, open_fields: Sequence[int]
) -> Optional[tuple[np.ndarray, float]]:
    """Optimal customer-to-field transport for a fixed open set.

    q[i,j] is an actual demand amount, not a fraction.  Consequently energy is
    computed as b[j,i] * q[i,j] and demand must not be multiplied a second time.
    """
    from scipy.optimize import linprog

    open_fields = sorted({int(j) for j in open_fields})
    if not open_fields:
        return None
    i_count = data.num_customers
    pairs = [(i, j) for i in range(i_count) for j in open_fields]
    index = {pair: p for p, pair in enumerate(pairs)}
    objective = np.array([data.transport_cost_ji[j, i] for i, j in pairs])
    a_eq = np.zeros((i_count, len(pairs)))
    for i in range(i_count):
        for j in open_fields:
            a_eq[i, index[(i, j)]] = 1.0
    a_ub = np.zeros((len(open_fields), len(pairs)))
    for row, j in enumerate(open_fields):
        for i in range(i_count):
            a_ub[row, index[(i, j)]] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=data.facility_capacity_j[open_fields],
        A_eq=a_eq,
        b_eq=data.demand,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        return None
    q = np.zeros((i_count, data.num_candidates))
    for pair, p in index.items():
        q[pair] = result.x[p]
    return q, float(result.fun)


def _allocate_one_field_to_drones(
    data: ModelData,
    field_j: int,
    customer_amount_i: np.ndarray,
    available_drones: Sequence[int],
    tolerance: float,
) -> Optional[tuple[list[int], np.ndarray]]:
    """Allocate divisible field demand to drones with payload and battery limits.

    The allocation LP is exact for a fixed selected drone set.  Drones are added
    until the LP is feasible; a returned allocation therefore satisfies both
    cumulative resource constraints.
    """
    from scipy.optimize import linprog

    active_i = np.flatnonzero(customer_amount_i > tolerance)
    if active_i.size == 0:
        return [], np.zeros((data.num_customers, data.num_drones))
    available = list(map(int, available_drones))
    if not available:
        return None

    total_payload = float(customer_amount_i.sum())
    total_energy = float(np.dot(data.energy_ji[field_j], customer_amount_i))
    payload_scale = max(total_payload, tolerance)
    energy_scale = max(total_energy, tolerance)
    available.sort(
        key=lambda k: min(
            data.payload_capacity_k[k] / payload_scale,
            data.battery_budget_k[k] / energy_scale,
        ),
        reverse=True,
    )

    selected: list[int] = []
    for k in available:
        selected.append(k)
        if (
            data.payload_capacity_k[selected].sum() + tolerance < total_payload
            or data.battery_budget_k[selected].sum() + tolerance < total_energy
        ):
            continue

        pairs = [(int(i), int(drone)) for i in active_i for drone in selected]
        index = {pair: p for p, pair in enumerate(pairs)}
        a_eq = np.zeros((active_i.size, len(pairs)))
        for row, i in enumerate(active_i):
            for drone in selected:
                a_eq[row, index[(int(i), drone)]] = 1.0
        a_ub = np.zeros((2 * len(selected), len(pairs)))
        b_ub = np.zeros(2 * len(selected))
        for col, drone in enumerate(selected):
            for i in active_i:
                p = index[(int(i), drone)]
                a_ub[col, p] = 1.0
                a_ub[len(selected) + col, p] = data.energy_ji[field_j, i]
            b_ub[col] = data.payload_capacity_k[drone]
            b_ub[len(selected) + col] = data.battery_budget_k[drone]

        result = linprog(
            np.zeros(len(pairs)),
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=customer_amount_i[active_i],
            bounds=(0.0, None),
            method="highs",
        )
        if result.success:
            q_ik = np.zeros((data.num_customers, data.num_drones))
            for pair, p in index.items():
                q_ik[pair] = result.x[p]
            return selected, q_ik
    return None


def construct_feasible_from_open_fields(
    data: ModelData,
    open_fields: Iterable[int],
    tolerance: float = 1e-7,
) -> Optional[tuple[Variables, float]]:
    open_fields = sorted({int(j) for j in open_fields})
    if not open_fields or len(open_fields) > data.max_facilities:
        return None
    if min(open_fields) < 0 or max(open_fields) >= data.num_candidates:
        raise ValueError("open field index out of range")

    assignment = _facility_transport_lp(data, open_fields)
    if assignment is None:
        return None
    q_ij, _ = assignment

    field_difficulty = []
    for j in open_fields:
        payload = float(q_ij[:, j].sum())
        energy = float(np.dot(data.energy_ji[j], q_ij[:, j]))
        field_difficulty.append((max(payload, energy), j))
    field_difficulty.sort(reverse=True)

    variables = Variables.zeros(data)
    variables.y_j[open_fields] = 1.0
    available = list(range(data.num_drones))
    for _, j in field_difficulty:
        allocation = _allocate_one_field_to_drones(
            data, j, q_ij[:, j], available, tolerance
        )
        if allocation is None:
            return None
        selected, q_ik = allocation
        variables.z_jk[j, selected] = 1.0
        for i in np.flatnonzero(q_ij[:, j] > tolerance):
            variables.x_ijk[i, j, :] = q_ik[i, :] / data.demand[i]
        selected_set = set(selected)
        available = [k for k in available if k not in selected_set]

    report = validate_solution(data, variables, tolerance=max(tolerance, 1e-6))
    if not report.is_feasible:
        return None
    return variables, report.objective


def _rank_fields(data: ModelData, relaxed: Optional[Variables]) -> list[int]:
    weighted_mean_cost = (
        data.transport_cost_ji @ data.demand / max(float(data.demand.sum()), 1e-12)
    )
    if relaxed is None:
        support = np.zeros(data.num_candidates)
    else:
        routed = np.einsum("i,ijk->j", data.demand, relaxed.x_ijk)
        support = relaxed.y_j + relaxed.z_jk.sum(axis=1) + routed / data.demand.sum()
    return sorted(
        range(data.num_candidates),
        key=lambda j: (-float(support[j]), float(weighted_mean_cost[j])),
    )


def _minimum_field_count_by_capacity(data: ModelData) -> Optional[int]:
    cumulative = np.cumsum(np.sort(data.facility_capacity_j)[::-1])
    positions = np.flatnonzero(cumulative + 1e-9 >= data.demand.sum())
    return int(positions[0] + 1) if positions.size else None


def refine_open_fields(
    data: ModelData,
    initial_open_fields: Iterable[int],
    *,
    candidate_limit: int = 25,
    max_passes: int = 8,
) -> tuple[set[int], Optional[Variables], float]:
    current = set(map(int, initial_open_fields))
    built = construct_feasible_from_open_fields(data, current)
    if built is None:
        return current, None, math.inf
    best_variables, best_cost = built
    proximity_order = _rank_fields(data, None)

    for _ in range(max_passes):
        improved = False
        unused = [j for j in proximity_order if j not in current][:candidate_limit]
        for remove_j in sorted(current):
            trials = [current - {remove_j}]
            trials.extend((current - {remove_j}) | {add_j} for add_j in unused)
            for trial in trials:
                if not trial:
                    continue
                candidate = construct_feasible_from_open_fields(data, trial)
                if candidate is None:
                    continue
                variables, cost = candidate
                if cost < best_cost - 1e-7:
                    current = set(trial)
                    best_variables = variables
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current, best_variables, best_cost


def primal_heuristic(
    data: ModelData,
    relaxed: Optional[Variables] = None,
    *,
    max_extra_fields: int = 8,
    run_local_search: bool = False,
) -> tuple[Optional[Variables], float]:
    minimum = _minimum_field_count_by_capacity(data)
    if minimum is None or minimum > data.max_facilities:
        return None, math.inf
    order = _rank_fields(data, relaxed)
    final_n = min(data.max_facilities, minimum + max_extra_fields)
    best_variables: Optional[Variables] = None
    best_cost = math.inf
    best_open: set[int] = set()
    for n_open in range(minimum, final_n + 1):
        open_fields = set(order[:n_open])
        candidate = construct_feasible_from_open_fields(data, open_fields)
        if candidate is None:
            continue
        variables, cost = candidate
        if cost < best_cost:
            best_variables, best_cost, best_open = variables, cost, open_fields
    if best_variables is None:
        return None, math.inf
    if run_local_search:
        _, refined, refined_cost = refine_open_fields(data, best_open)
        if refined is not None and refined_cost < best_cost:
            best_variables, best_cost = refined, refined_cost
    return best_variables, best_cost


def solve_lagrangian_relaxation(
    data: ModelData,
    *,
    max_iterations: int = 1000,
    time_limit: float = 600.0,
    relative_gap_tolerance: float = 1e-3,
    repair_every: int = 25,
    checkpoint_every: int = 25,
    initial_step_scale: float = 1.0,
    initial_multipliers: Optional[LagrangianMultipliers] = None,
    no_guidance: bool = False,
    post_local_search: bool = True,
    final_recovery_only: bool = False,
    expansion_factor: float = 1.3,
    contraction_factor: float = 0.5,
    initial_solution: Optional[Variables] = None,
    verbose: bool = True,
) -> LagrangianResult:
    multipliers = initial_multipliers or LagrangianMultipliers.zeros(data)
    best_lb = -math.inf
    best_ub = math.inf
    best_variables: Optional[Variables] = None
    history: list[dict[str, float]] = []
    step_scale = float(initial_step_scale)
    checkpoint_lb = -math.inf
    start = time.time()
    termination = "iteration_limit"
    if initial_solution is not None:
        report = validate_solution(data, initial_solution, tolerance=1e-6)
        if not report.is_feasible:
            raise ValueError('Initial solution is infeasible')
        best_variables = initial_solution.copy()
        best_ub = report.objective

    if no_guidance:
        candidate, best_ub = primal_heuristic(data, None, run_local_search=post_local_search)
        return LagrangianResult(best_ub, 0.0, candidate, [], 0, time.time()-start, 'no_guidance')
    last_relaxed = None
    for iteration in range(max_iterations):
        elapsed = time.time() - start
        if elapsed >= time_limit:
            termination = "time_limit"
            break

        x, _ = solve_subproblem_x(data, multipliers)
        y, _ = solve_subproblem_y(data, multipliers)
        z, _ = solve_subproblem_z(data, multipliers)
        relaxed = Variables(y, z, x)
        last_relaxed = relaxed
        current_l, gradients = lagrangian_value_and_subgradients(
            data, relaxed, multipliers
        )
        best_lb = max(best_lb, current_l)

        should_repair = (not final_recovery_only) and (iteration == 0 or iteration % max(1, repair_every) == 0)
        if should_repair:
            candidate, candidate_ub = primal_heuristic(data, relaxed)
            if candidate is not None:
                report = validate_solution(data, candidate, tolerance=1e-6)
                if not report.is_feasible:
                    raise RuntimeError(
                        "primal heuristic returned an infeasible solution: "
                        + repr(report.max_violation)
                    )
                if candidate_ub < best_ub:
                    best_ub = candidate_ub
                    best_variables = candidate.copy()

        duality_tol = 1e-6 * max(1.0, abs(best_ub))
        if math.isfinite(best_ub) and best_lb > best_ub + duality_tol:
            raise RuntimeError(
                f"weak-duality violation: best LB {best_lb} exceeds feasible UB {best_ub}"
            )
        gap = (
            max(0.0, (best_ub - best_lb) / best_ub)
            if math.isfinite(best_ub) and best_ub > 0
            else math.inf
        )
        history.append(
            {
                "iteration": float(iteration + 1),
                "current_lagrangian": float(current_l),
                "best_lower_bound": float(best_lb),
                "best_upper_bound": float(best_ub),
                "relative_gap": float(gap),
                "step_scale": float(step_scale),
                "elapsed_seconds": float(time.time() - start),
            }
        )
        if verbose and (iteration < 5 or (iteration + 1) % 25 == 0):
            print(
                f"iter={iteration + 1:4d} L={current_l:12.3f} "
                f"bestLB={best_lb:12.3f} bestUB={best_ub:12.3f} "
                f"gap={100.0 * gap:7.3f}%"
            )
        if gap <= relative_gap_tolerance:
            termination = "relative_gap"
            break

        if (iteration + 1) % max(1, checkpoint_every) == 0:
            if best_lb > checkpoint_lb + 1e-7:
                step_scale = min(2.0, expansion_factor * step_scale)
            else:
                step_scale = max(0.05, contraction_factor * step_scale)
            checkpoint_lb = best_lb

        norm_squared = sum(float(np.sum(g * g)) for g in gradients)
        if norm_squared <= 1e-20:
            termination = "zero_subgradient"
            break
        if math.isfinite(best_ub):
            numerator = best_ub - current_l
            if numerator < -duality_tol:
                raise RuntimeError("negative Polyak numerator beyond numerical tolerance")
            theta = step_scale * max(0.0, numerator) / norm_squared
        else:
            theta = step_scale / math.sqrt(iteration + 1.0)

        g_a, g_beta, g_delta, g_gamma = gradients
        multipliers.a_jk = np.maximum(0.0, multipliers.a_jk + theta * g_a)
        multipliers.beta_jk = np.maximum(
            0.0, multipliers.beta_jk + theta * g_beta
        )
        multipliers.delta_jk = np.maximum(
            0.0, multipliers.delta_jk + theta * g_delta
        )
        multipliers.gamma_j = np.maximum(
            0.0, multipliers.gamma_j + theta * g_gamma
        )
    else:
        iteration = max_iterations - 1

    if final_recovery_only and last_relaxed is not None:
        candidate, candidate_ub = primal_heuristic(data, last_relaxed)
        if candidate is not None and candidate_ub < best_ub:
            best_variables, best_ub = candidate, candidate_ub
    if best_variables is not None and post_local_search:
        open_fields = np.flatnonzero(best_variables.y_j > 0.5)
        _, refined, refined_cost = refine_open_fields(data, open_fields)
        if refined is not None and refined_cost < best_ub:
            report = validate_solution(data, refined, tolerance=1e-6)
            if not report.is_feasible:
                raise RuntimeError("refined solution failed final feasibility audit")
            best_variables, best_ub = refined, refined_cost

    elapsed = time.time() - start
    iterations = len(history)
    return LagrangianResult(
        best_upper_bound=float(best_ub),
        best_lower_bound=float(best_lb),
        best_variables=best_variables,
        history=history,
        iterations=iterations,
        elapsed_seconds=float(elapsed),
        termination=termination,
    )


def solve_full_milp_gurobi(
    data: ModelData,
    *,
    time_limit: float = 180.0,
    mip_gap: Optional[float] = None,
    return_solution: bool = False,
) -> dict[str, Any]:
    """Solve the exact canonical model and report incumbent and dual bound separately."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is required for the exact model") from exc

    i_count, j_count, k_count = (
        data.num_customers,
        data.num_candidates,
        data.num_drones,
    )
    model = gp.Model("flpd_v2_cumulative_payload_roundtrip")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    x = model.addVars(i_count, j_count, k_count, lb=0.0, ub=1.0, name="x")
    y = model.addVars(j_count, vtype=GRB.BINARY, name="y")
    z = model.addVars(j_count, k_count, vtype=GRB.BINARY, name="z")
    model.setObjective(
        gp.quicksum(data.setup_cost_j[j] * y[j] for j in range(j_count))
        + gp.quicksum(
            data.transport_cost_ji[j, i] * data.demand[i] * x[i, j, k]
            for i in range(i_count)
            for j in range(j_count)
            for k in range(k_count)
        ),
        GRB.MINIMIZE,
    )
    model.addConstr(gp.quicksum(y[j] for j in range(j_count)) <= data.max_facilities)
    model.addConstrs(
        (z[j, k] <= y[j] for j in range(j_count) for k in range(k_count)),
        name="field_link",
    )
    model.addConstrs(
        (gp.quicksum(z[j, k] for j in range(j_count)) <= 1 for k in range(k_count)),
        name="one_field_per_drone",
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k] for j in range(j_count) for k in range(k_count))
            == 1
            for i in range(i_count)
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(
                data.energy_ji[j, i] * data.demand[i] * x[i, j, k]
                for i in range(i_count)
            )
            <= data.battery_budget_k[k] * z[j, k]
            for j in range(j_count)
            for k in range(k_count)
        ),
        name="battery",
    )
    model.addConstrs(
        (
            gp.quicksum(data.demand[i] * x[i, j, k] for i in range(i_count))
            <= data.payload_capacity_k[k] * z[j, k]
            for j in range(j_count)
            for k in range(k_count)
        ),
        name="cumulative_payload",
    )
    model.addConstrs(
        (
            gp.quicksum(
                data.demand[i] * x[i, j, k]
                for i in range(i_count)
                for k in range(k_count)
            )
            <= data.facility_capacity_j[j] * y[j]
            for j in range(j_count)
        ),
        name="facility_capacity",
    )
    model.optimize()

    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.INTERRUPTED: "interrupted",
    }
    has_incumbent = model.SolCount > 0
    result: dict[str, Any] = {
        "model_tag": MODEL_TAG,
        "status": status_names.get(model.Status, f"status_{model.Status}"),
        "incumbent_objective": float(model.ObjVal) if has_incumbent else None,
        "best_bound": float(model.ObjBound) if np.isfinite(model.ObjBound) else None,
        "mip_gap": float(model.MIPGap) if has_incumbent else None,
        "runtime_seconds": float(model.Runtime),
        "node_count": float(model.NodeCount),
        "field_count": (
            int(round(sum(y[j].X for j in range(j_count)))) if has_incumbent else None
        ),
        "is_proven_optimal": model.Status == GRB.OPTIMAL,
    }
    if return_solution and has_incumbent:
        variables = Variables.zeros(data)
        variables.y_j = np.array([y[j].X for j in range(j_count)])
        variables.z_jk = np.array(
            [[z[j, k].X for k in range(k_count)] for j in range(j_count)]
        )
        variables.x_ijk = np.array(
            [
                [[x[i, j, k].X for k in range(k_count)] for j in range(j_count)]
                for i in range(i_count)
            ]
        )
        report = validate_solution(data, variables, tolerance=1e-5)
        result["variables"] = variables
        result["feasibility_report"] = report
    return result


def aggregate_relaxation_bound(
    data: ModelData, *, time_limit: float = 300.0
) -> dict[str, Any]:
    """Drone-aggregated capacitated-facility lower bound."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is required for the aggregate bound") from exc

    i_count, j_count = data.num_customers, data.num_candidates
    model = gp.Model("flpd_v2_aggregate_relaxation")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    x = model.addVars(i_count, j_count, lb=0.0, ub=1.0, name="xbar")
    y = model.addVars(j_count, vtype=GRB.BINARY, name="y")
    model.setObjective(
        gp.quicksum(data.setup_cost_j[j] * y[j] for j in range(j_count))
        + gp.quicksum(
            data.transport_cost_ji[j, i] * data.demand[i] * x[i, j]
            for i in range(i_count)
            for j in range(j_count)
        ),
        GRB.MINIMIZE,
    )
    model.addConstr(gp.quicksum(y[j] for j in range(j_count)) <= data.max_facilities)
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in range(j_count)) == 1 for i in range(i_count))
    )
    model.addConstrs(
        (
            gp.quicksum(data.demand[i] * x[i, j] for i in range(i_count))
            <= data.facility_capacity_j[j] * y[j]
            for j in range(j_count)
        )
    )
    model.optimize()
    has_incumbent = model.SolCount > 0
    bound = float(model.ObjBound) if np.isfinite(model.ObjBound) else None
    status = "optimal" if model.Status == GRB.OPTIMAL else "bound"
    return {
        "model_tag": MODEL_TAG,
        "status": status if bound is not None else "unavailable",
        "lower_bound": bound,
        "incumbent_objective": float(model.ObjVal) if has_incumbent else None,
        "runtime_seconds": float(model.Runtime),
    }


def certified_gap(upper_bound: float, lower_bound: float, tolerance: float = 1e-7) -> float:
    if not (np.isfinite(upper_bound) and np.isfinite(lower_bound) and upper_bound > 0):
        raise ValueError("finite positive upper bound and finite lower bound are required")
    violation_tolerance = tolerance * max(1.0, abs(upper_bound))
    if lower_bound > upper_bound + violation_tolerance:
        raise RuntimeError(
            f"invalid certificate: lower bound {lower_bound} exceeds upper bound {upper_bound}"
        )
    return max(0.0, (upper_bound - lower_bound) / upper_bound)


def hybrid_lower_bound(lr_lower_bound: float, aggregate_result: dict[str, Any]) -> float:
    values = [float(lr_lower_bound)]
    aggregate = aggregate_result.get("lower_bound")
    if aggregate is not None:
        values.append(float(aggregate))
    return max(values)
