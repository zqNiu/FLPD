"""Deterministic instance builders for the canonical FLPD v2 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flpd_core_v2 import MODEL_TAG, ModelData, Variables, validate_solution


def _balanced_candidate_grid(count: int, area: float) -> np.ndarray:
    """Return count points distributed across the full rectangular service area.

    Unlike truncating the first count points of a square mesh, the evenly spaced
    index selection does not systematically remove the upper rows of the grid.
    """
    nx = int(np.ceil(np.sqrt(count)))
    ny = int(np.ceil(count / nx))
    xs = np.linspace(0.0, area, nx)
    ys = np.linspace(0.0, area, ny)
    gx, gy = np.meshgrid(xs, ys)
    all_points = np.column_stack([gx.ravel(), gy.ravel()])
    if all_points.shape[0] == count:
        return all_points
    indices = np.rint(np.linspace(0, all_points.shape[0] - 1, count)).astype(int)
    if np.unique(indices).size != count:
        raise RuntimeError("candidate-grid selection produced duplicate indices")
    return all_points[indices]


def generate_synthetic_geometry(
    num_customers: int,
    geometry: str,
    *,
    area: float = 10.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    geometry = geometry.upper()
    if geometry == "R":
        customer_xy = rng.uniform(0.0, area, size=(num_customers, 2))
    elif geometry == "C":
        centers = rng.uniform(0.0, area, size=(4, 2))
        choices = rng.integers(0, len(centers), size=num_customers)
        customer_xy = centers[choices] + rng.normal(
            0.0, area * 0.07, size=(num_customers, 2)
        )
        customer_xy = np.clip(customer_xy, 0.0, area)
    elif geometry == "RC":
        clustered_count = num_customers // 2
        centers = rng.uniform(0.0, area, size=(3, 2))
        choices = rng.integers(0, len(centers), size=clustered_count)
        clustered = centers[choices] + rng.normal(
            0.0, area * 0.07, size=(clustered_count, 2)
        )
        random = rng.uniform(0.0, area, size=(num_customers - clustered_count, 2))
        customer_xy = np.clip(np.vstack([clustered, random]), 0.0, area)
    else:
        raise ValueError("geometry must be one of C, R, or RC")

    demand = rng.uniform(1.0, 3.0, size=num_customers)
    field_xy = _balanced_candidate_grid(3 * num_customers, area)
    return customer_xy, demand, field_xy


def build_synthetic_model(
    num_customers: int,
    geometry: str,
    *,
    seed: int,
    area: float = 10.0,
    num_drones: int | None = None,
    max_facilities: int | None = None,
    setup_cost: float = 350.0,
    transport_cost_per_km: float = 5.0,
    alpha: float = 1.43,
    battery_budget: float = 50.0,
    payload_capacity: float = 5.0,
) -> tuple[ModelData, np.ndarray, np.ndarray]:
    customer_xy, demand, field_xy = generate_synthetic_geometry(
        num_customers, geometry, area=area, seed=seed
    )
    distance_ij = np.linalg.norm(
        customer_xy[:, None, :] - field_xy[None, :, :], axis=2
    )
    distance_ji = distance_ij.T.copy()
    data = ModelData(
        demand=demand,
        transport_cost_ji=transport_cost_per_km * distance_ji,
        energy_ji=2.0 * alpha * distance_ji,
        distance_ji=distance_ji,
        num_drones=num_drones if num_drones is not None else 3 * num_customers,
        max_facilities=(
            max_facilities if max_facilities is not None else num_customers
        ),
        setup_cost_j=setup_cost,
        facility_capacity_j=15.0 * float(np.median(demand)),
        battery_budget_k=battery_budget,
        payload_capacity_k=payload_capacity,
        metadata={
            "model_tag": MODEL_TAG,
            "instance_family": "deterministic_synthetic",
            "geometry": geometry.upper(),
            "seed": int(seed),
            "area_km": float(area),
            "alpha": float(alpha),
            "energy_definition": "2*alpha*distance",
        },
    )
    return data, customer_xy, field_xy


def build_shanghai_model(
    csv_path: str | Path,
    *,
    num_drones: int = 310,
    max_facilities: int = 30,
    setup_cost: float = 500_000.0,
    transport_cost_per_km: float = 8.0,
    alpha: float = 0.01,
    facility_capacity: float = 2500.0,
    battery_budget: float = 20.0,
    payload_capacity: float = 200.0,
) -> ModelData:
    return ModelData.from_geographic_csv(
        csv_path,
        num_drones=num_drones,
        max_facilities=max_facilities,
        setup_cost=setup_cost,
        transport_cost_per_km=transport_cost_per_km,
        alpha=alpha,
        facility_capacity=facility_capacity,
        battery_budget=battery_budget,
        payload_capacity=payload_capacity,
    )


def primary_field_by_customer(variables: Variables) -> np.ndarray:
    """Field receiving the largest allocation share for plotting only."""
    return np.argmax(variables.x_ijk.sum(axis=2), axis=1)


def solution_summary(
    data: ModelData,
    variables: Variables,
    *,
    customer_xy: np.ndarray | None = None,
    field_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    report = validate_solution(data, variables, tolerance=1e-6)
    if not report.is_feasible:
        raise RuntimeError(f"cannot serialize infeasible solution: {report.max_violation}")
    payload_load = np.einsum("i,ijk->jk", data.demand, variables.x_ijk)
    battery_load = np.einsum(
        "ji,i,ijk->jk", data.energy_ji, data.demand, variables.x_ijk
    )
    opened = np.flatnonzero(variables.y_j > 0.5)
    used_pairs = np.argwhere(variables.z_jk > 0.5)
    result: dict[str, Any] = {
        "objective": float(report.objective),
        "opened_fields": opened.astype(int).tolist(),
        "field_count": int(opened.size),
        "used_drones": int(used_pairs.shape[0]),
        "max_payload_load": float(payload_load.max(initial=0.0)),
        "max_battery_load": float(battery_load.max(initial=0.0)),
        "max_violation": report.max_violation,
        "primary_field_by_customer": primary_field_by_customer(variables)
        .astype(int)
        .tolist(),
    }
    if customer_xy is not None:
        result["customer_xy"] = np.asarray(customer_xy, dtype=float).tolist()
    if field_xy is not None:
        result["field_xy"] = np.asarray(field_xy, dtype=float).tolist()
    return result

