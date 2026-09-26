"""
Process Delft3D-FM / D-Flow FM NetCDF outputs for DUALFloodGNN / Cluster-DUALFloodGNN and mSWE-compatible evaluation targets.

What this script does
---------------------
1. Reads Delft3D-FM map NetCDF files.
2. Builds a cell-centroid graph:
   - nodes = mesh faces / cells
   - directed internal edges = mesh2d_edge_faces[:, 0] -> mesh2d_edge_faces[:, 1]
   - signed edge flow = mesh2d_q1, so positive q is assumed along the stored link direction.
3. Computes:
   DUALFloodGNN targets:
   - node water volume [m3] = water depth * cell area
   - signed edge discharge [m3/s] = mesh2d_q1 on directed internal links

   mSWE-compatible targets/evaluation arrays:
   - node water depth [m] = mesh2d_waterdepth
   - velocity components [m/s] = mesh2d_ucx, mesh2d_ucy when available
   - node unit-discharge magnitude [m2/s] = h * sqrt(ucx^2 + ucy^2)

   Additional edge diagnostic:
   - edge unit discharge [m2/s] = signed discharge / physical face-edge length
4. Checks DEM-to-face alignment.
5. Checks mass balance to confirm the Delft3D flow sign convention.
6. Uses mesh2d_edge_type when available:
   - 1 = normal internal edge
   - 2 = boundary-condition edge
   - 3 = other boundary / wall edge
7. Optionally detects the inflow boundary-condition edge and creates a ghost-node edge:
   - ghost -> real cell for inflow
   Wall/no-flow boundary edges are not converted into ghost nodes.
7. Saves processed arrays as .npz and optional shapefiles/csv diagnostics.

Important convention
--------------------
For internal edges, this script uses:

    edge_index = [face_L, face_R]
    q1 > 0 means flow from face_L to face_R
    q1 < 0 means flow from face_R to face_L

This should be verified using the mass-balance report produced by the script.

Required packages
-----------------
xarray, numpy, pandas
Optional: shapely/geopandas for shapefiles
Optional: scipy for nearest-neighbour DEM matching.

Example
-------
python process_delft3d_for_dualfloodgnn.py \
  --map_nc "raw_datasets_mesh/Simulations/output_2_map.nc" \
  --dem_xyz "raw_datasets_mesh/DEM/DEM_2.xyz" \
  --hydrograph "raw_datasets_mesh/Hydrograph/Hydrograph_2.txt" \
  --out_dir "processed/M02" \
  --manning 0.023 \
  --save_shapefiles

Batch example
-------------
python process_delft3d_for_dualfloodgnn.py \
  --root "raw_datasets_mesh" \
  --sim_ids 1 2 3 4 5 \
  --out_dir "processed" \
  --manning 0.023
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
except Exception:  # pragma: no cover
    gpd = None
    Point = LineString = Polygon = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Delft3DVariables:
    face_x: str = "mesh2d_face_x"
    face_y: str = "mesh2d_face_y"
    edge_x: str = "mesh2d_edge_x"
    edge_y: str = "mesh2d_edge_y"
    node_x: str = "mesh2d_node_x"
    node_y: str = "mesh2d_node_y"
    face_nodes: str = "mesh2d_face_nodes"
    edge_faces: str = "mesh2d_edge_faces"
    edge_nodes: str = "mesh2d_edge_nodes"
    edge_type: str = "mesh2d_edge_type"
    waterdepth: str = "mesh2d_waterdepth"
    discharge: str = "mesh2d_q1"
    ucx: str = "mesh2d_ucx"
    ucy: str = "mesh2d_ucy"
    time: str = "time"


# -----------------------------
# General helpers
# -----------------------------

def require_var(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        available = ", ".join(list(ds.variables)[:60])
        raise KeyError(f"Variable '{name}' not found in NetCDF. First available variables: {available}")
    return ds[name]


def as_numpy(ds: xr.Dataset, name: str) -> np.ndarray:
    return require_var(ds, name).values


def as_numpy_optional(ds: xr.Dataset, name: str, default_shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
    """Return variable values if present; otherwise return None or a NaN array of default_shape."""
    if name not in ds:
        if default_shape is None:
            return None
        print(f"[WARNING] Optional variable '{name}' not found; filling with NaN.")
        return np.full(default_shape, np.nan, dtype=float)
    return ds[name].values


def to_2d_time_major(arr: np.ndarray, n_time: int, n_space: int, var_name: str) -> np.ndarray:
    """Return array as shape (time, space)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{var_name} must be 2D, got shape {arr.shape}")
    if arr.shape == (n_time, n_space):
        return arr
    if arr.shape == (n_space, n_time):
        return arr.T
    raise ValueError(f"Cannot interpret {var_name} shape {arr.shape}; expected {(n_time, n_space)} or {(n_space, n_time)}")


def normalize_connectivity(conn: np.ndarray, expected_rows: Optional[int] = None) -> np.ndarray:
    """
    Normalize UGRID connectivity to shape (n_items, item_width).
    Delft3D stores connectivity as 1-based indices, with 0/NaN/fill for missing.
    """
    conn = np.asarray(conn)
    if conn.ndim != 2:
        raise ValueError(f"Connectivity must be 2D, got shape {conn.shape}")

    # If expected_rows is known, orient accordingly.
    if expected_rows is not None:
        if conn.shape[0] == expected_rows:
            return conn
        if conn.shape[1] == expected_rows:
            return conn.T

    # Heuristic: connectivity width is usually small (2 for edges, 3/4/5+ for faces).
    if conn.shape[0] <= 8 and conn.shape[1] > conn.shape[0]:
        return conn.T
    return conn


def valid_1based_to_0based(v: Any) -> Optional[int]:
    """Convert Delft3D 1-based connectivity value to 0-based Python index; return None for missing."""
    try:
        if pd.isna(v):
            return None
        iv = int(v)
    except Exception:
        return None
    if iv <= 0:
        return None
    return iv - 1


def time_to_seconds(time_values: np.ndarray) -> np.ndarray:
    """Convert xarray time coordinate values to seconds from start."""
    t = np.asarray(time_values)

    if np.issubdtype(t.dtype, np.datetime64):
        return ((t - t[0]) / np.timedelta64(1, "s")).astype(float)

    if np.issubdtype(t.dtype, np.timedelta64):
        return (t / np.timedelta64(1, "s")).astype(float)

    # Numeric time. Delft3D map files often store seconds.
    return t.astype(float)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def mae(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else float("nan")


# -----------------------------
# DEM and hydrograph
# -----------------------------

def read_dem_xyz(dem_xyz: Path) -> pd.DataFrame:
    dem = pd.read_csv(dem_xyz, sep=r"\s+", header=None, names=["x", "y", "z"])
    if not {"x", "y", "z"}.issubset(dem.columns):
        raise ValueError("DEM xyz file must have three columns: x y z")
    return dem


def match_dem_to_faces(
    dem: pd.DataFrame,
    face_x: np.ndarray,
    face_y: np.ndarray,
    xy_tol: float = 1e-6,
    allow_nearest: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return elevation array aligned to face order.

    First tries exact row-order alignment.
    If that fails and scipy is available, falls back to nearest-neighbour matching.
    """
    report: Dict[str, Any] = {
        "method": None,
        "row_count_match": len(dem) == len(face_x),
        "max_abs_dx": None,
        "max_abs_dy": None,
        "max_nearest_distance": None,
    }

    dem_x = dem["x"].to_numpy(float)
    dem_y = dem["y"].to_numpy(float)
    dem_z = dem["z"].to_numpy(float)

    if len(dem) == len(face_x):
        dx = np.abs(dem_x - face_x)
        dy = np.abs(dem_y - face_y)
        report["max_abs_dx"] = float(np.nanmax(dx))
        report["max_abs_dy"] = float(np.nanmax(dy))

        if np.nanmax(dx) <= xy_tol and np.nanmax(dy) <= xy_tol:
            report["method"] = "row_order_exact"
            return dem_z.copy(), report

    if not allow_nearest:
        raise ValueError(
            "DEM rows do not align with face centroids and nearest-neighbour matching is disabled. "
            f"Row count match: {report['row_count_match']}; "
            f"max dx: {report['max_abs_dx']}; max dy: {report['max_abs_dy']}"
        )

    if cKDTree is None:
        raise ImportError(
            "DEM rows do not align with face centroids and scipy is not installed for nearest-neighbour matching."
        )

    tree = cKDTree(np.column_stack([dem_x, dem_y]))
    distances, idx = tree.query(np.column_stack([face_x, face_y]), k=1)
    report["method"] = "nearest_neighbour"
    report["max_nearest_distance"] = float(np.nanmax(distances))
    report["mean_nearest_distance"] = float(np.nanmean(distances))

    if np.nanmax(distances) > max(1e-3, xy_tol * 100):
        print(
            "[WARNING] DEM nearest-neighbour matching has relatively large maximum distance: "
            f"{np.nanmax(distances):.6g}. Please inspect CRS/coordinates."
        )

    return dem_z[idx], report


def read_hydrograph(hydrograph_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if hydrograph_path is None:
        return None
    if not Path(hydrograph_path).exists():
        raise FileNotFoundError(hydrograph_path)

    hydro = pd.read_csv(hydrograph_path, sep=r"\s+", header=None)
    if hydro.shape[1] < 2:
        raise ValueError("Hydrograph file must have at least two columns: time_seconds inflow_m3s")

    hydro = hydro.iloc[:, :2].copy()
    hydro.columns = ["time_seconds", "inflow_m3s"]
    hydro["time_seconds"] = hydro["time_seconds"].astype(float)
    hydro["inflow_m3s"] = hydro["inflow_m3s"].astype(float)
    return hydro


def interpolate_hydrograph_to_model_time(hydro: pd.DataFrame, model_time_seconds: np.ndarray) -> np.ndarray:
    return np.interp(
        model_time_seconds,
        hydro["time_seconds"].to_numpy(float),
        hydro["inflow_m3s"].to_numpy(float),
    )


# -----------------------------
# Geometry processing
# -----------------------------

def build_face_polygons(
    face_nodes_raw: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> Tuple[List[Any], np.ndarray]:
    """
    Build cell polygons and cell area.

    Face IDs are returned in original 0-based order. If a face has invalid polygon,
    area is NaN and polygon is None.
    """
    if Polygon is None:
        raise ImportError("geopandas/shapely is required for polygon construction.")

    face_nodes = normalize_connectivity(face_nodes_raw, expected_rows=None)
    n_faces = face_nodes.shape[0]

    polygons: List[Any] = []
    areas = np.full(n_faces, np.nan, dtype=float)

    for i in range(n_faces):
        ids = [valid_1based_to_0based(v) for v in face_nodes[i, :]]
        ids = [idx for idx in ids if idx is not None]

        if len(ids) < 3:
            polygons.append(None)
            continue

        coords = [(float(node_x[j]), float(node_y[j])) for j in ids]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons.append(poly)
        areas[i] = float(poly.area)

    return polygons, areas


def compute_physical_edge_lengths(
    ds: xr.Dataset,
    v: Delft3DVariables,
    n_edges: int,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    Physical face-edge length used for converting discharge [m3/s] to unit discharge [m2/s].

    Priority:
    1. mesh2d_edge_nodes geometry, if available
    2. existing edge length variable candidates
    3. NaN fallback
    """
    # Try edge-node connectivity: actual mesh face-edge endpoints.
    if v.edge_nodes in ds:
        edge_nodes_raw = normalize_connectivity(as_numpy(ds, v.edge_nodes), expected_rows=n_edges)
        lengths = np.full(n_edges, np.nan, dtype=float)
        for e, pair in enumerate(edge_nodes_raw):
            if len(pair) < 2:
                continue
            n1 = valid_1based_to_0based(pair[0])
            n2 = valid_1based_to_0based(pair[1])
            if n1 is None or n2 is None:
                continue
            lengths[e] = math.hypot(float(node_x[n2] - node_x[n1]), float(node_y[n2] - node_y[n1]))
        if np.isfinite(lengths).any():
            return lengths, "mesh2d_edge_nodes"

    # Try common variable names.
    for candidate in ["mesh2d_edge_length", "mesh2d_edge_lengths", "mesh2d_flowlink_length", "mesh2d_flowelem_edge_length"]:
        if candidate in ds:
            arr = np.asarray(ds[candidate].values).astype(float).reshape(-1)
            if len(arr) == n_edges:
                return arr, candidate

    return np.full(n_edges, np.nan, dtype=float), "not_available"


def build_edges(
    edge_faces_raw: np.ndarray,
    face_x: np.ndarray,
    face_y: np.ndarray,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    elevation: np.ndarray,
    physical_edge_length: np.ndarray,
    edge_type: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build internal and boundary edge tables.

    edge_faces are assumed 1-based from Delft3D:
    - first valid face = L
    - second valid face = R
    Internal directed graph edge is L -> R.
    """
    n_edges = edge_faces_raw.shape[0]
    if edge_type is None:
        edge_type = np.full(n_edges, -1, dtype=int)
    else:
        edge_type = np.asarray(edge_type).reshape(-1).astype(int)
        if edge_type.shape[0] != n_edges:
            raise ValueError(f"edge_type length {edge_type.shape[0]} does not match n_edges {n_edges}")
    records_internal = []
    records_boundary = []

    # Count how many boundary edges touch each cell.
    n_faces = len(face_x)
    boundary_touch_count = np.zeros(n_faces, dtype=int)

    for e in range(n_edges):
        f1 = valid_1based_to_0based(edge_faces_raw[e, 0])
        f2 = valid_1based_to_0based(edge_faces_raw[e, 1])

        if f1 is None and f2 is None:
            continue

        if f1 is not None and f2 is not None:
            x1, y1 = float(face_x[f1]), float(face_y[f1])
            x2, y2 = float(face_x[f2]), float(face_y[f2])
            dual_len = math.hypot(x2 - x1, y2 - y1)
            dz = float(elevation[f2] - elevation[f1])
            slope = dz / dual_len if dual_len > 0 else np.nan

            records_internal.append({
                "edge_id_0based": e,
                "edge_id_1based": e + 1,
                "edge_type": int(edge_type[e]),
                "from_node_0based": f1,
                "to_node_0based": f2,
                "from_node_1based": f1 + 1,
                "to_node_1based": f2 + 1,
                "from_elevation_m": float(elevation[f1]),
                "to_elevation_m": float(elevation[f2]),
                "dual_length_m": dual_len,
                "physical_edge_length_m": float(physical_edge_length[e]) if np.isfinite(physical_edge_length[e]) else np.nan,
                "slope_to_from": slope,
                "is_boundary": False,
            })
        else:
            real = f1 if f1 is not None else f2
            missing_side = "right" if f2 is None else "left"
            boundary_touch_count[real] += 1

            x1, y1 = float(face_x[real]), float(face_y[real])
            x2, y2 = float(edge_x[e]), float(edge_y[e])
            dual_len = math.hypot(x2 - x1, y2 - y1)

            records_boundary.append({
                "edge_id_0based": e,
                "edge_id_1based": e + 1,
                "edge_type": int(edge_type[e]),
                "edge_type_label": "bc_edge" if int(edge_type[e]) == 2 else ("wall_or_other_boundary" if int(edge_type[e]) == 3 else "boundary_unknown"),
                "real_cell_0based": real,
                "real_cell_1based": real + 1,
                "missing_side": missing_side,
                "real_cell_elevation_m": float(elevation[real]),
                "dual_length_m": dual_len,
                "physical_edge_length_m": float(physical_edge_length[e]) if np.isfinite(physical_edge_length[e]) else np.nan,
                "is_boundary": True,
            })

    internal_edges = pd.DataFrame(records_internal)
    boundary_edges = pd.DataFrame(records_boundary)
    return internal_edges, boundary_edges, boundary_touch_count, np.array([r["edge_id_0based"] for r in records_internal], dtype=int), np.array([r["edge_id_0based"] for r in records_boundary], dtype=int)


# -----------------------------
# Flow and mass balance
# -----------------------------

def boundary_q_into_domain(q: np.ndarray, missing_side: str, internal_sign: int = +1) -> np.ndarray:
    """
    Convert boundary q1 to positive-inflow-into-domain time series.

    Assumption:
    - q positive follows Delft3D link direction L -> R.
    - if missing side is 'left', outside is L and real cell is R, so positive q enters domain.
    - if missing side is 'right', real cell is L and outside is R, so positive q leaves domain.
    """
    q = internal_sign * np.asarray(q, dtype=float)
    if missing_side == "left":
        return q
    if missing_side == "right":
        return -q
    raise ValueError(f"Unknown missing_side: {missing_side}")


def detect_inflow_boundary_edge(
    boundary_edges: pd.DataFrame,
    q_all: np.ndarray,
    time_seconds: np.ndarray,
    hydrograph: Optional[pd.DataFrame],
    internal_sign: int = +1,
) -> Optional[Dict[str, Any]]:
    """
    Identify which boundary edge best matches the inflow hydrograph.

    Returns None if no hydrograph or no boundary edges.
    """
    if hydrograph is None or boundary_edges.empty:
        return None

    hydro_q = interpolate_hydrograph_to_model_time(hydrograph, time_seconds)
    denom = np.nanmax(np.abs(hydro_q))
    if denom <= 0:
        denom = 1.0

    candidates = []
    for _, row in boundary_edges.iterrows():
        e = int(row["edge_id_0based"])
        qin = boundary_q_into_domain(q_all[:, e], row["missing_side"], internal_sign=internal_sign)

        # Test direct and reversed sign, because boundary sign convention can be confusing.
        rmse_direct = rmse(qin - hydro_q) / denom
        rmse_reversed = rmse((-qin) - hydro_q) / denom

        if rmse_reversed < rmse_direct:
            best_rmse = rmse_reversed
            sign_multiplier = -1
        else:
            best_rmse = rmse_direct
            sign_multiplier = +1

        corr = np.corrcoef(sign_multiplier * qin, hydro_q)[0, 1] if np.std(qin) > 0 and np.std(hydro_q) > 0 else np.nan

        candidates.append({
            "edge_id_0based": e,
            "edge_id_1based": e + 1,
            "real_cell_0based": int(row["real_cell_0based"]),
            "real_cell_1based": int(row["real_cell_1based"]),
            "missing_side": row["missing_side"],
            "sign_multiplier": sign_multiplier,
            "nrmse_to_hydrograph": float(best_rmse),
            "corr_to_hydrograph": float(corr) if np.isfinite(corr) else np.nan,
        })

    best = sorted(candidates, key=lambda d: d["nrmse_to_hydrograph"])[0]
    return best


def compute_flux_volume_change(
    q_at_edges: np.ndarray,
    edge_faces_raw: np.ndarray,
    dt: float,
    n_faces: int,
    internal_sign: int = +1,
    include_boundary_q1: bool = True,
) -> np.ndarray:
    """
    Compute expected volume change in each cell from q1 over one interval.

    q_at_edges: q1 values at one time step, shape (n_edges,)
    edge_faces_raw: shape (n_edges, 2), 1-based with missing as 0/NaN
    dt: seconds
    """
    expected_dv = np.zeros(n_faces, dtype=float)
    q_signed = internal_sign * np.asarray(q_at_edges, dtype=float)

    for e, q in enumerate(q_signed):
        f1 = valid_1based_to_0based(edge_faces_raw[e, 0])
        f2 = valid_1based_to_0based(edge_faces_raw[e, 1])

        if f1 is not None and f2 is not None:
            # positive q leaves L/f1 and enters R/f2
            expected_dv[f1] -= q * dt
            expected_dv[f2] += q * dt
        elif include_boundary_q1:
            # Boundary: missing side is outside-domain.
            if f1 is not None and f2 is None:
                # positive q leaves real cell to outside
                expected_dv[f1] -= q * dt
            elif f1 is None and f2 is not None:
                # positive q enters real cell from outside
                expected_dv[f2] += q * dt

    return expected_dv


def mass_balance_report(
    volume: np.ndarray,
    q_all: np.ndarray,
    edge_faces_raw: np.ndarray,
    time_seconds: np.ndarray,
    boundary_touch_count: np.ndarray,
    include_boundary_q1: bool = True,
    flux_time_options: Sequence[str] = ("left", "right", "average"),
) -> pd.DataFrame:
    """
    Compare dV from waterdepth*area against flux-based dV from q1.

    Returns diagnostics for different sign and time alignment assumptions.
    The best case should normally be internal_sign = +1 if q1 follows edge_faces[:,0] -> edge_faces[:,1].
    """
    n_time, n_faces = volume.shape
    dt_all = np.diff(time_seconds)
    if len(dt_all) != n_time - 1:
        raise ValueError("time_seconds length must match volume time dimension")

    interior_cell_mask = boundary_touch_count == 0
    wet_or_active_mask = np.nanmax(np.abs(np.diff(volume, axis=0)), axis=0) > 1e-6

    rows = []
    for internal_sign in (+1, -1):
        for flux_time in flux_time_options:
            residuals_all = []
            residuals_interior = []
            residuals_active = []

            for t in range(n_time - 1):
                dt = float(dt_all[t])
                dV = volume[t + 1] - volume[t]

                if flux_time == "left":
                    q_use = q_all[t]
                elif flux_time == "right":
                    q_use = q_all[t + 1]
                elif flux_time == "average":
                    q_use = 0.5 * (q_all[t] + q_all[t + 1])
                else:
                    raise ValueError(f"Unknown flux_time option: {flux_time}")

                expected = compute_flux_volume_change(
                    q_use,
                    edge_faces_raw=edge_faces_raw,
                    dt=dt,
                    n_faces=n_faces,
                    internal_sign=internal_sign,
                    include_boundary_q1=include_boundary_q1,
                )
                residual = dV - expected

                residuals_all.append(residual)
                residuals_interior.append(residual[interior_cell_mask])
                residuals_active.append(residual[wet_or_active_mask])

            residuals_all = np.concatenate([r.reshape(1, -1) for r in residuals_all], axis=0)
            residuals_interior = np.concatenate([r.reshape(1, -1) for r in residuals_interior], axis=0) if interior_cell_mask.any() else np.array([])
            residuals_active = np.concatenate([r.reshape(1, -1) for r in residuals_active], axis=0) if wet_or_active_mask.any() else np.array([])

            dV_all = np.diff(volume, axis=0)
            scale = np.nanmean(np.abs(dV_all))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

            rows.append({
                "internal_sign": internal_sign,
                "q_direction_assumption": "q>0 L->R" if internal_sign == +1 else "q>0 R->L",
                "flux_time": flux_time,
                "include_boundary_q1": include_boundary_q1,
                "all_cells_mae_m3": mae(residuals_all),
                "all_cells_rmse_m3": rmse(residuals_all),
                "all_cells_rel_rmse": rmse(residuals_all) / scale,
                "interior_cells_mae_m3": mae(residuals_interior) if residuals_interior.size else np.nan,
                "interior_cells_rmse_m3": rmse(residuals_interior) if residuals_interior.size else np.nan,
                "active_cells_mae_m3": mae(residuals_active) if residuals_active.size else np.nan,
                "active_cells_rmse_m3": rmse(residuals_active) if residuals_active.size else np.nan,
            })

    report = pd.DataFrame(rows)
    report = report.sort_values(["interior_cells_rmse_m3", "all_cells_rmse_m3"], na_position="last").reset_index(drop=True)
    return report


# -----------------------------
# Optional ghost-edge construction
# -----------------------------

def build_inflow_ghost_edge(
    n_faces: int,
    best_boundary: Optional[Dict[str, Any]],
    q_all: np.ndarray,
    boundary_edges: pd.DataFrame,
    time_seconds: np.ndarray,
    hydrograph: Optional[pd.DataFrame],
    use_hydrograph_values: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Create one ghost node and one directed edge ghost -> real cell for the detected inflow.

    This is for model input construction. The ghost edge flow is positive into the real cell.
    """
    if best_boundary is None:
        return None

    real_cell = int(best_boundary["real_cell_0based"])
    ghost_node = n_faces
    ghost_edge_index = np.array([[ghost_node], [real_cell]], dtype=np.int64)

    if use_hydrograph_values and hydrograph is not None:
        qin = interpolate_hydrograph_to_model_time(hydrograph, time_seconds)
    else:
        row = boundary_edges.loc[boundary_edges["edge_id_0based"] == best_boundary["edge_id_0based"]].iloc[0]
        qin = boundary_q_into_domain(q_all[:, int(row["edge_id_0based"])], row["missing_side"])
        qin = best_boundary.get("sign_multiplier", 1) * qin

    return {
        "ghost_node_id_0based": np.array([ghost_node], dtype=np.int64),
        "ghost_edge_index": ghost_edge_index,
        "ghost_edge_flow_m3s": np.asarray(qin, dtype=float).reshape(-1, 1),
        "ghost_edge_real_cell_0based": np.array([real_cell], dtype=np.int64),
    }


# -----------------------------
# Shapefile saving
# -----------------------------

def save_shapefiles(
    out_dir: Path,
    crs: Optional[str],
    face_x: np.ndarray,
    face_y: np.ndarray,
    elevation: np.ndarray,
    area: np.ndarray,
    polygons: Sequence[Any],
    internal_edges: pd.DataFrame,
    boundary_edges: pd.DataFrame,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
) -> None:
    if gpd is None:
        raise ImportError("geopandas/shapely are required to save shapefiles.")

    shp_dir = out_dir / "shapefiles"
    safe_mkdir(shp_dir)

    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": np.arange(1, len(face_x) + 1),
            "node0": np.arange(len(face_x)),
            "x": face_x,
            "y": face_y,
            "elev_m": elevation,
            "area_m2": area,
        },
        geometry=[Point(float(x), float(y)) for x, y in zip(face_x, face_y)],
        crs=crs,
    )
    nodes_gdf.to_file(shp_dir / "nodes.shp")

    valid_polygons = [(i, p) for i, p in enumerate(polygons) if p is not None]
    cells_gdf = gpd.GeoDataFrame(
        {
            "cell_id": [i + 1 for i, _ in valid_polygons],
            "cell0": [i for i, _ in valid_polygons],
            "elev_m": [float(elevation[i]) for i, _ in valid_polygons],
            "area_m2": [float(area[i]) for i, _ in valid_polygons],
        },
        geometry=[p for _, p in valid_polygons],
        crs=crs,
    )
    cells_gdf.to_file(shp_dir / "cells.shp")

    # Internal centroid-to-centroid links.
    lines = []
    for _, row in internal_edges.iterrows():
        f1 = int(row["from_node_0based"])
        f2 = int(row["to_node_0based"])
        lines.append(LineString([(float(face_x[f1]), float(face_y[f1])), (float(face_x[f2]), float(face_y[f2]))]))

    if len(internal_edges) > 0:
        gdf_internal = gpd.GeoDataFrame(internal_edges.copy(), geometry=lines, crs=crs)
        gdf_internal.to_file(shp_dir / "internal_links.shp")

    # Boundary cell-centroid to edge-midpoint links.
    blines = []
    for _, row in boundary_edges.iterrows():
        f = int(row["real_cell_0based"])
        e = int(row["edge_id_0based"])
        blines.append(LineString([(float(face_x[f]), float(face_y[f])), (float(edge_x[e]), float(edge_y[e]))]))

    if len(boundary_edges) > 0:
        gdf_boundary = gpd.GeoDataFrame(boundary_edges.copy(), geometry=blines, crs=crs)
        gdf_boundary.to_file(shp_dir / "boundary_links.shp")


# -----------------------------
# Main processing function
# -----------------------------

def process_one_simulation(
    map_nc: Path,
    dem_xyz: Path,
    out_dir: Path,
    hydrograph_path: Optional[Path] = None,
    manning: float = 0.023,
    crs: Optional[str] = None,
    xy_tol: float = 1e-6,
    save_shapefiles_flag: bool = False,
    include_boundary_q1_in_mass_balance: bool = True,
) -> Dict[str, Any]:
    safe_mkdir(out_dir)

    v = Delft3DVariables()

    hydrograph = read_hydrograph(hydrograph_path) if hydrograph_path else None

    with xr.open_dataset(map_nc) as ds:
        face_x = as_numpy(ds, v.face_x).astype(float).reshape(-1)
        face_y = as_numpy(ds, v.face_y).astype(float).reshape(-1)
        edge_x = as_numpy(ds, v.edge_x).astype(float).reshape(-1)
        edge_y = as_numpy(ds, v.edge_y).astype(float).reshape(-1)
        node_x = as_numpy(ds, v.node_x).astype(float).reshape(-1)
        node_y = as_numpy(ds, v.node_y).astype(float).reshape(-1)

        n_faces = len(face_x)
        n_edges = len(edge_x)

        time_seconds = time_to_seconds(as_numpy(ds, v.time))
        n_time = len(time_seconds)

        edge_faces_raw = normalize_connectivity(as_numpy(ds, v.edge_faces), expected_rows=n_edges)
        if edge_faces_raw.shape[1] != 2:
            raise ValueError(f"Expected edge_faces to have width 2; got shape {edge_faces_raw.shape}")

        q_all = to_2d_time_major(as_numpy(ds, v.discharge), n_time=n_time, n_space=n_edges, var_name=v.discharge).astype(float)
        depth = to_2d_time_major(as_numpy(ds, v.waterdepth), n_time=n_time, n_space=n_faces, var_name=v.waterdepth).astype(float)

        # mSWE-GNN uses cell-centred velocity components from Delft3D/D-Hydro outputs.
        # These are optional here because DUALFloodGNN itself uses edge q1 as the flow target.
        ucx_raw = as_numpy_optional(ds, v.ucx)
        ucy_raw = as_numpy_optional(ds, v.ucy)
        if ucx_raw is None or ucy_raw is None:
            ucx = np.full_like(depth, np.nan, dtype=float)
            ucy = np.full_like(depth, np.nan, dtype=float)
        else:
            ucx = to_2d_time_major(ucx_raw, n_time=n_time, n_space=n_faces, var_name=v.ucx).astype(float)
            ucy = to_2d_time_major(ucy_raw, n_time=n_time, n_space=n_faces, var_name=v.ucy).astype(float)

        # Delft3D/D-Hydro edge type: 1 normal edge, 2 BC edge, 3 other boundary edge.
        if v.edge_type in ds:
            edge_type = np.asarray(ds[v.edge_type].values).reshape(-1).astype(int)
        else:
            print(f"[WARNING] Optional variable '{v.edge_type}' not found; boundary edges will be inferred from missing faces.")
            edge_type = np.full(n_edges, -1, dtype=int)

        dem = read_dem_xyz(dem_xyz)
        elevation, dem_report = match_dem_to_faces(dem, face_x, face_y, xy_tol=xy_tol, allow_nearest=True)

        polygons, area = build_face_polygons(
            face_nodes_raw=as_numpy(ds, v.face_nodes),
            node_x=node_x,
            node_y=node_y,
        )

        # Check for invalid/zero area cells.
        if np.any(~np.isfinite(area)) or np.any(area <= 0):
            bad = int(np.sum((~np.isfinite(area)) | (area <= 0)))
            print(f"[WARNING] {bad} cells have invalid or non-positive area. Inspect cell polygons.")

        physical_edge_length, physical_edge_length_source = compute_physical_edge_lengths(
            ds=ds,
            v=v,
            n_edges=n_edges,
            node_x=node_x,
            node_y=node_y,
        )

        internal_edges, boundary_edges, boundary_touch_count, internal_edge_ids, boundary_edge_ids = build_edges(
            edge_faces_raw=edge_faces_raw,
            face_x=face_x,
            face_y=face_y,
            edge_x=edge_x,
            edge_y=edge_y,
            elevation=elevation,
            physical_edge_length=physical_edge_length,
            edge_type=edge_type,
        )

        if save_shapefiles_flag:
            save_shapefiles(
                out_dir=out_dir,
                crs=crs,
                face_x=face_x,
                face_y=face_y,
                elevation=elevation,
                area=area,
                polygons=polygons,
                internal_edges=internal_edges,
                boundary_edges=boundary_edges,
                edge_x=edge_x,
                edge_y=edge_y,
            )

    # Do calculations after closing dataset safely.
    # DUALFloodGNN node target: volume [m3].
    volume = depth * area.reshape(1, -1)

    # mSWE-compatible node targets/evaluation variables.
    # Official mSWE-GNN processing uses water depth and cell-centred velocity components;
    # unit-discharge magnitude is h * sqrt(ucx^2 + ucy^2).
    mswe_velocity_magnitude_ms = np.sqrt(ucx ** 2 + ucy ** 2)
    mswe_unit_discharge_x_m2s = depth * ucx
    mswe_unit_discharge_y_m2s = depth * ucy
    mswe_unit_discharge_magnitude_m2s = depth * mswe_velocity_magnitude_ms
    mswe_water_level_m = elevation.reshape(1, -1) + depth

    # Internal graph arrays.
    edge_index = np.vstack([
        internal_edges["from_node_0based"].to_numpy(np.int64),
        internal_edges["to_node_0based"].to_numpy(np.int64),
    ])
    edge_flow_m3s = q_all[:, internal_edge_ids]

    edge_face_length = internal_edges["physical_edge_length_m"].to_numpy(float)
    edge_dual_length = internal_edges["dual_length_m"].to_numpy(float)
    edge_slope = internal_edges["slope_to_from"].to_numpy(float)

    # Unit discharge using physical face-edge length. Fall back to dual length if physical length is missing.
    denom = edge_face_length.copy()
    missing_len = (~np.isfinite(denom)) | (denom <= 0)
    if np.any(missing_len):
        print(
            f"[WARNING] {int(np.sum(missing_len))}/{len(denom)} internal edges have missing physical edge length. "
            "Falling back to centroid-to-centroid dual length for those edges."
        )
        denom[missing_len] = edge_dual_length[missing_len]

    edge_unit_discharge_m2s = edge_flow_m3s / denom.reshape(1, -1)

    # Mass balance reports.
    mb_with_boundary = mass_balance_report(
        volume=volume,
        q_all=q_all,
        edge_faces_raw=edge_faces_raw,
        time_seconds=time_seconds,
        boundary_touch_count=boundary_touch_count,
        include_boundary_q1=include_boundary_q1_in_mass_balance,
    )
    mb_internal_only = mass_balance_report(
        volume=volume,
        q_all=q_all,
        edge_faces_raw=edge_faces_raw,
        time_seconds=time_seconds,
        boundary_touch_count=boundary_touch_count,
        include_boundary_q1=False,
    )

    mb_with_boundary.to_csv(out_dir / "mass_balance_report_with_boundary_q1.csv", index=False)
    mb_internal_only.to_csv(out_dir / "mass_balance_report_internal_cells_focus.csv", index=False)

    # Boundary inflow detection using q1 and hydrograph.
    # Prefer Delft3D/D-Hydro boundary-condition edges (edge_type == 2).
    # If edge_type is absent or no BC edge is found, fall back to all boundary edges.
    boundary_edges_for_detection = boundary_edges
    if "edge_type" in boundary_edges.columns and (boundary_edges["edge_type"] == 2).any():
        boundary_edges_for_detection = boundary_edges.loc[boundary_edges["edge_type"] == 2].copy()

    best_boundary = detect_inflow_boundary_edge(
        boundary_edges=boundary_edges_for_detection,
        q_all=q_all,
        time_seconds=time_seconds,
        hydrograph=hydrograph,
        internal_sign=+1,
    )
    ghost = build_inflow_ghost_edge(
        n_faces=n_faces,
        best_boundary=best_boundary,
        q_all=q_all,
        boundary_edges=boundary_edges,
        time_seconds=time_seconds,
        hydrograph=hydrograph,
        use_hydrograph_values=True,
    )

    # Save tables.
    nodes = pd.DataFrame({
        "node_id_0based": np.arange(n_faces),
        "node_id_1based": np.arange(1, n_faces + 1),
        "x": face_x,
        "y": face_y,
        "elevation_m": elevation,
        "area_m2": area,
        "manning": manning,
        "is_boundary_cell": boundary_touch_count > 0,
        "n_boundary_edges": boundary_touch_count,
    })

    nodes.to_csv(out_dir / "nodes.csv", index=False)
    internal_edges.to_csv(out_dir / "internal_edges.csv", index=False)
    boundary_edges.to_csv(out_dir / "boundary_edges.csv", index=False)

    if hydrograph is not None:
        hydrograph.to_csv(out_dir / "hydrograph.csv", index=False)

    if best_boundary is not None:
        with open(out_dir / "detected_inflow_boundary.json", "w", encoding="utf-8") as f:
            json.dump(best_boundary, f, indent=2)

    # Save arrays for model training.
    npz_payload = {
        "time_seconds": time_seconds,
        "node_xy": np.column_stack([face_x, face_y]),
        "node_area_m2": area.astype(float),
        "node_elevation_m": elevation.astype(float),
        "node_manning": np.full(n_faces, manning, dtype=float),
        "node_is_boundary": (boundary_touch_count > 0),
        "edge_index": edge_index.astype(np.int64),
        "edge_original_id_0based": internal_edge_ids.astype(np.int64),
        "edge_original_id_1based": (internal_edge_ids + 1).astype(np.int64),
        "edge_dual_length_m": edge_dual_length.astype(float),
        "edge_physical_length_m": edge_face_length.astype(float),
        "edge_slope_to_from": edge_slope.astype(float),
        "edge_type_internal": internal_edges["edge_type"].to_numpy(np.int64) if "edge_type" in internal_edges.columns else np.full(len(internal_edges), -1, dtype=np.int64),

        # Shared water-depth target for water-depth-only comparison against mSWE-GNN.
        "water_depth_m": depth.astype(float),

        # DUALFloodGNN targets.
        "dualflood_node_target_volume_m3": volume.astype(float),
        "dualflood_edge_target_flow_m3s": edge_flow_m3s.astype(float),

        # Backward-compatible aliases for existing DUALFloodGNN scripts.
        "water_volume_m3": volume.astype(float),
        "edge_flow_m3s": edge_flow_m3s.astype(float),

        # mSWE-compatible node targets/evaluation arrays.
        "mswe_water_depth_m": depth.astype(float),
        "mswe_velocity_x_ms": ucx.astype(float),
        "mswe_velocity_y_ms": ucy.astype(float),
        "mswe_velocity_magnitude_ms": mswe_velocity_magnitude_ms.astype(float),
        "mswe_unit_discharge_x_m2s": mswe_unit_discharge_x_m2s.astype(float),
        "mswe_unit_discharge_y_m2s": mswe_unit_discharge_y_m2s.astype(float),
        "mswe_unit_discharge_magnitude_m2s": mswe_unit_discharge_magnitude_m2s.astype(float),
        "mswe_water_level_m": mswe_water_level_m.astype(float),

        # Edge diagnostic, not identical to mSWE node target.
        "edge_unit_discharge_m2s": edge_unit_discharge_m2s.astype(float),

        "boundary_edge_original_id_0based": boundary_edge_ids.astype(np.int64),
        "boundary_edge_type": boundary_edges["edge_type"].to_numpy(np.int64) if "edge_type" in boundary_edges.columns else np.full(len(boundary_edges), -1, dtype=np.int64),
        "bc_boundary_edge_original_id_0based": boundary_edges.loc[boundary_edges["edge_type"] == 2, "edge_id_0based"].to_numpy(np.int64) if "edge_type" in boundary_edges.columns else np.array([], dtype=np.int64),
        "wall_boundary_edge_original_id_0based": boundary_edges.loc[boundary_edges["edge_type"] == 3, "edge_id_0based"].to_numpy(np.int64) if "edge_type" in boundary_edges.columns else np.array([], dtype=np.int64),
        "boundary_touch_count": boundary_touch_count.astype(np.int64),
    }

    if hydrograph is not None:
        npz_payload["hydrograph_time_seconds"] = hydrograph["time_seconds"].to_numpy(float)
        npz_payload["hydrograph_inflow_m3s"] = hydrograph["inflow_m3s"].to_numpy(float)
        npz_payload["hydrograph_inflow_interp_m3s"] = interpolate_hydrograph_to_model_time(hydrograph, time_seconds)

    if ghost is not None:
        npz_payload.update(ghost)

    np.savez_compressed(out_dir / "processed_dualfloodgnn.npz", **npz_payload)

    metadata = {
        "map_nc": str(map_nc),
        "dem_xyz": str(dem_xyz),
        "hydrograph_path": str(hydrograph_path) if hydrograph_path else None,
        "n_faces": int(n_faces),
        "n_edges_total": int(n_edges),
        "n_internal_edges": int(len(internal_edges)),
        "n_boundary_edges": int(len(boundary_edges)),
        "n_bc_boundary_edges": int((boundary_edges["edge_type"] == 2).sum()) if "edge_type" in boundary_edges.columns else None,
        "n_wall_or_other_boundary_edges": int((boundary_edges["edge_type"] == 3).sum()) if "edge_type" in boundary_edges.columns else None,
        "n_time": int(n_time),
        "manning": float(manning),
        "crs": crs,
        "dem_alignment": dem_report,
        "physical_edge_length_source": physical_edge_length_source,
        "best_mass_balance_with_boundary": mb_with_boundary.iloc[0].to_dict() if len(mb_with_boundary) else None,
        "best_mass_balance_internal_only": mb_internal_only.iloc[0].to_dict() if len(mb_internal_only) else None,
        "detected_inflow_boundary": best_boundary,
        "notes": {
            "internal_edge_convention": "edge_index = edge_faces[:,0] -> edge_faces[:,1], using 0-based Python indices",
            "positive_q": "positive mesh2d_q1 is assumed to follow edge_index direction; verify using mass_balance_report",
            "dual_targets": "dualflood_node_target_volume_m3 and dualflood_edge_target_flow_m3s",
            "water_depth_comparison": "use water_depth_m or mswe_water_depth_m; convert DUAL predictions by predicted_volume / node_area_m2",
            "mswe_compatible_targets": "mswe_water_depth_m and mswe_unit_discharge_magnitude_m2s = h * sqrt(ucx^2 + ucy^2)",
            "edge_unit_discharge": "edge_unit_discharge_m2s = edge_flow_m3s / physical mesh face-edge length; diagnostic only, not identical to mSWE node unit-discharge target",
            "edge_type": "mesh2d_edge_type is saved when available: 1 normal/internal, 2 BC edge, 3 wall/other boundary",
            "ghost_edge": "if detected, ghost_edge_index is directed ghost -> real cell and ghost_edge_flow_m3s is positive into domain; detected only from BC edges when edge_type==2 is available",
        },
    }

    with open(out_dir / "processing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Processing completed ===")
    print(f"Output directory: {out_dir}")
    print(f"Faces/nodes: {n_faces}")
    print(f"Internal edges: {len(internal_edges)}")
    print(f"Boundary edges: {len(boundary_edges)}")
    if "edge_type" in boundary_edges.columns:
        print(f"  BC boundary edges (edge_type==2): {int((boundary_edges['edge_type'] == 2).sum())}")
        print(f"  Wall/other boundary edges (edge_type==3): {int((boundary_edges['edge_type'] == 3).sum())}")
    print(f"DEM alignment: {dem_report}")
    print(f"Physical edge length source: {physical_edge_length_source}")
    print("\nBest mass-balance assumption including boundary q1:")
    print(mb_with_boundary.head(3).to_string(index=False))
    print("\nBest mass-balance assumption focusing on internal cells:")
    print(mb_internal_only.head(3).to_string(index=False))

    if best_boundary:
        print("\nDetected inflow boundary candidate:")
        print(json.dumps(best_boundary, indent=2))

    return metadata


# -----------------------------
# Batch processing
# -----------------------------

def process_batch(
    root: Path,
    sim_ids: Sequence[int],
    out_dir: Path,
    manning: float = 0.023,
    crs: Optional[str] = None,
    save_shapefiles_flag: bool = False,
) -> None:
    """
    Process multiple mSWE-GNN-style simulations.

    Expected layout:
      root/Simulations/output_{id}_map.nc
      root/DEM/DEM_{id}.xyz
      root/Hydrograph/Hydrograph_{id}.txt
    """
    safe_mkdir(out_dir)
    summary = []

    for sid in sim_ids:
        sim_out = out_dir / f"M{sid:03d}"
        map_nc = root / "Simulations" / f"output_{sid}_map.nc"
        dem_xyz = root / "DEM" / f"DEM_{sid}.xyz"
        hydro = root / "Hydrograph" / f"Hydrograph_{sid}.txt"

        print(f"\n\n### Processing simulation {sid} ###")
        try:
            meta = process_one_simulation(
                map_nc=map_nc,
                dem_xyz=dem_xyz,
                hydrograph_path=hydro if hydro.exists() else None,
                out_dir=sim_out,
                manning=manning,
                crs=crs,
                save_shapefiles_flag=save_shapefiles_flag,
            )
            summary.append({
                "sim_id": sid,
                "status": "ok",
                "n_faces": meta["n_faces"],
                "n_internal_edges": meta["n_internal_edges"],
                "n_boundary_edges": meta["n_boundary_edges"],
                "best_mb_rmse_with_boundary": meta["best_mass_balance_with_boundary"]["all_cells_rmse_m3"] if meta["best_mass_balance_with_boundary"] else np.nan,
                "best_mb_rmse_internal": meta["best_mass_balance_internal_only"]["interior_cells_rmse_m3"] if meta["best_mass_balance_internal_only"] else np.nan,
            })
        except Exception as e:
            print(f"[ERROR] Simulation {sid} failed: {e}")
            summary.append({
                "sim_id": sid,
                "status": "failed",
                "error": str(e),
            })

    pd.DataFrame(summary).to_csv(out_dir / "batch_processing_summary.csv", index=False)
    print(f"\nBatch summary saved to: {out_dir / 'batch_processing_summary.csv'}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Delft3D-FM NetCDF outputs for DUALFloodGNN and mSWE-compatible targets.")

    parser.add_argument("--map_nc", type=Path, default=None, help="Path to one Delft3D-FM map NetCDF file.")
    parser.add_argument("--dem_xyz", type=Path, default=None, help="Path to corresponding DEM xyz file.")
    parser.add_argument("--hydrograph", type=Path, default=None, help="Path to corresponding hydrograph txt file.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")

    parser.add_argument("--root", type=Path, default=None, help="Root dataset folder for batch mode.")
    parser.add_argument("--sim_ids", type=int, nargs="*", default=None, help="Simulation IDs for batch mode, e.g. --sim_ids 1 2 3.")

    parser.add_argument("--manning", type=float, default=0.023, help="Uniform Manning roughness value.")
    parser.add_argument("--crs", type=str, default=None, help="Optional CRS, e.g. EPSG:28992.")
    parser.add_argument("--xy_tol", type=float, default=1e-6, help="Tolerance for DEM row-order coordinate alignment.")
    parser.add_argument("--save_shapefiles", action="store_true", help="Save nodes/cells/links shapefiles.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.root is not None:
        if not args.sim_ids:
            raise ValueError("Batch mode requires --sim_ids.")
        process_batch(
            root=args.root,
            sim_ids=args.sim_ids,
            out_dir=args.out_dir,
            manning=args.manning,
            crs=args.crs,
            save_shapefiles_flag=args.save_shapefiles,
        )
    else:
        if args.map_nc is None or args.dem_xyz is None:
            raise ValueError("Single-file mode requires --map_nc and --dem_xyz.")
        process_one_simulation(
            map_nc=args.map_nc,
            dem_xyz=args.dem_xyz,
            hydrograph_path=args.hydrograph,
            out_dir=args.out_dir,
            manning=args.manning,
            crs=args.crs,
            xy_tol=args.xy_tol,
            save_shapefiles_flag=args.save_shapefiles,
        )


if __name__ == "__main__":
    main()
