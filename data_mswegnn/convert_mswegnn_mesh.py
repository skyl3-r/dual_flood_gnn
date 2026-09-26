#!/usr/bin/env python3
"""Convert Delft3D NetCDF meshes for the mSWE-GNN loader.

The DEM matching, boundary classification, inflow detection, and physics
diagnostics are adapted from Prof. Viraj's reference code.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data_mswegnn" / "datasets" / "raw"
DEFAULT_SIMULATION_DIR = DEFAULT_RAW_DIR / "Simulations"
DEFAULT_DEM_DIR = DEFAULT_RAW_DIR / "DEM"
DEFAULT_HYDROGRAPH_DIR = DEFAULT_RAW_DIR / "Hydrograph"
DEFAULT_OUTPUT_DIR = DEFAULT_RAW_DIR / "Geometry"


def _normalise_connectivity(values, expected_rows, name):
    """Return UGRID connectivity as (item, connectivity-width).

    Delft3D files encountered by the reference processor use either
    (n_items, width) or (width, n_items), depending on the exporter.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {values.shape}")
    if values.shape[0] == expected_rows:
        return values
    if values.shape[1] == expected_rows:
        return values.T
    raise ValueError(
        f"{name} has shape {values.shape}; expected one dimension to equal "
        f"the number of items ({expected_rows})"
    )


def _as_zero_based_connectivity(values, name):
    """Convert 1-based UGRID indices while retaining missing entries as -1."""
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values) | (values <= 0)
    result = np.full(values.shape, -1, dtype=np.int64)
    result[~missing] = values[~missing].astype(np.int64) - 1
    return result


def verify_conversion(ds, nodes_gdf, cells_gdf, edges_gdf, n_faces, n_ghost):
    """Validate the training-facing geometry and NetCDF edge mapping."""
    required = {
        "nodes": {"X", "Y", "Elevation1", "node_type", "face_id"},
        "cells": {"cell_id", "face_id", "area_m2"},
        "edges": {"from_node", "to_node", "length", "slope", "edge_type", "fc_length", "nc_edge_id"},
    }
    frames = {"nodes": nodes_gdf, "cells": cells_gdf, "edges": edges_gdf}
    for name, columns in required.items():
        missing = columns - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} shapefile is missing columns: {sorted(missing)}")

    if len(nodes_gdf) != n_faces + n_ghost or len(cells_gdf) != len(nodes_gdf):
        raise ValueError("Node and cell counts do not match the real-face/ghost-node layout")
    if (nodes_gdf["node_type"].to_numpy()[:n_faces] != 1).any():
        raise ValueError("The first n_faces node records must be normal nodes")
    if n_ghost and not np.isin(nodes_gdf["node_type"].to_numpy()[n_faces:], [2, 3]).all():
        raise ValueError("Ghost nodes must have node_type=2 or node_type=3")

    node_count = len(nodes_gdf)
    endpoints = edges_gdf[["from_node", "to_node"]].to_numpy()
    if len(endpoints) and ((endpoints < 0).any() or (endpoints >= node_count).any()):
        raise ValueError("Edge endpoints are outside the exported node range")

    nc_edge_ids = edges_gdf["nc_edge_id"].to_numpy(dtype=np.int64)
    n_nc_edges = int(ds["mesh2d_edge_faces"].size // 2)
    if len(nc_edge_ids) and ((nc_edge_ids < 0).any() or (nc_edge_ids >= n_nc_edges).any()):
        raise ValueError("Exported nc_edge_id values do not map to mesh2d_q1 edges")
    if "mesh2d_q1" in ds and ds["mesh2d_q1"].ndim != 2:
        raise ValueError("mesh2d_q1 must be two-dimensional")
    if "mesh2d_q1" in ds:
        q_dimensions = [size for size in ds["mesh2d_q1"].shape if size == n_nc_edges]
        if not q_dimensions:
            raise ValueError("mesh2d_q1 has no dimension matching mesh2d_edge_faces")
        q_edge_count = q_dimensions[0]
        if len(nc_edge_ids) and nc_edge_ids.max() >= q_edge_count:
            raise ValueError("Exported nc_edge_id exceeds the mesh2d_q1 edge dimension")

    if not np.isfinite(cells_gdf["area_m2"].to_numpy()).all():
        raise ValueError("Cell areas contain non-finite values")
    if (cells_gdf["area_m2"].to_numpy() < 0).any():
        raise ValueError("Cell areas cannot be negative")
    print(
        f"Verification passed: {len(nodes_gdf)} nodes, {len(cells_gdf)} cells, "
        f"{len(edges_gdf)} edges ({n_ghost} ghost nodes)."
    )


def _reference_tools():
    """Load the validated Delft3D processing helpers only when requested."""
    reference_path = Path(__file__).resolve().with_name("delft3d_reference.py")
    spec = importlib.util.spec_from_file_location("dualflood_reference_processor", reference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reference processor: {reference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {
        "build_edges": module.build_edges,
        "detect_inflow_boundary_edge": module.detect_inflow_boundary_edge,
        "mass_balance_report": module.mass_balance_report,
        "match_dem_to_faces": module.match_dem_to_faces,
        "read_dem_xyz": module.read_dem_xyz,
        "read_hydrograph": module.read_hydrograph,
        "time_to_seconds": module.time_to_seconds,
    }


def convert_batch(
    input_dir,
    output_root,
    start_run_id,
    end_run_id,
    expected_inflow_edges=1,
    dem_dir=None,
    hydrograph_dir=None,
):
    """
    Batch-convert all output_{run_id}_map.nc files in a directory range.

    Example:
        python convert_mswegnn_mesh.py \
            --input-dir datasets/raw/Simulations \
            --output datasets/raw/Geometry \
            --start-run-id 1 \
            --end-run-id 100
    """

    input_dir = Path(input_dir)
    output_root = Path(output_root)

    if start_run_id > end_run_id:
        raise ValueError("--start-run-id must be <= --end-run-id")

    for run_id in range(start_run_id, end_run_id + 1):
        nc_file = input_dir / f"output_{run_id}_map.nc"

        if not nc_file.exists():
            print(f"Skipping missing file: {nc_file}")
            continue

        print(f"\n=== Converting run_id {run_id} ===")
        convert_mesh(
            nc_file,
            output_root,
            run_id,
            expected_inflow_edges=expected_inflow_edges,
            dem_xyz=(Path(dem_dir) / f"DEM_{run_id}.xyz") if dem_dir else None,
            hydrograph=(Path(hydrograph_dir) / f"Hydrograph_{run_id}.txt") if hydrograph_dir else None,
        )


def convert_mesh(
    nc_file,
    output_root,
    run_id,
    expected_inflow_edges=1,
    dem_xyz=None,
    hydrograph=None,
):
    """
    Convert an mSWE-GNN D-Hydro map NetCDF into:

        Nodes/nodes_{run_id}.shp
        Edges/edges_{run_id}.shp
        Cells/cells_{run_id}.shp

    The graph construction follows mSWE-GNN's graph_creation.py:
      - GNN nodes = mesh faces/cells
      - GNN edges = dual mesh edges (face-to-face adjacency)
      - boundary dual edges are removed
      - dual graph is undirected
    """

    nc_file = Path(nc_file)
    output_root = Path(output_root)
    if dem_xyz is None or hydrograph is None:
        raise ValueError("dem_xyz and hydrograph are required for the validated conversion workflow")

    nodes_dir = output_root / "Nodes"
    edges_dir = output_root / "Edges"
    cells_dir = output_root / "Cells"

    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {nc_file}")

    ds = xr.open_dataset(nc_file)

    tools = _reference_tools()

    # ============================================================
    # 1. GNN NODES = mesh faces/cells
    # ============================================================

    face_x = ds["mesh2d_face_x"].values.astype(np.float64)
    face_y = ds["mesh2d_face_y"].values.astype(np.float64)
    node_x = ds["mesh2d_node_x"].values.astype(np.float64)
    node_y = ds["mesh2d_node_y"].values.astype(np.float64)
    node_z = ds["mesh2d_node_z"].values.astype(np.float64)

    n_faces = len(face_x)

    print(f"Number of GNN nodes / cells: {n_faces}")

    # The dataset applies boundary conditions by removing ghost nodes and
    # appending them again after processing.  D-Hydro has one flow boundary
    # in these simulations, so represent each type-2 boundary face by one
    # ghost node at the end of the node table.
    edge_faces_raw = _normalise_connectivity(
        ds["mesh2d_edge_faces"].values, len(ds["mesh2d_edge_x"]), "mesh2d_edge_faces"
    )
    edge_faces = _as_zero_based_connectivity(edge_faces_raw, "mesh2d_edge_faces")
    edge_types_nc = ds["mesh2d_edge_type"].values.astype(np.int64)
    face_nodes = _normalise_connectivity(
        ds["mesh2d_face_nodes"].values, n_faces, "mesh2d_face_nodes"
    )
    edge_nodes = _normalise_connectivity(
        ds["mesh2d_edge_nodes"].values, len(edge_faces), "mesh2d_edge_nodes"
    )
    edge_nodes = _as_zero_based_connectivity(edge_nodes, "mesh2d_edge_nodes")
    if edge_faces.shape[1] != 2 or edge_nodes.shape[1] != 2:
        raise ValueError("Expected edge_faces and edge_nodes to have width 2")
    if len(edge_types_nc) != len(edge_faces):
        raise ValueError("mesh2d_edge_type does not match the number of edges")
    if np.any(edge_nodes < 0):
        raise ValueError("mesh2d_edge_nodes contains missing or invalid node indices")
    if np.any(edge_nodes >= len(node_x)):
        raise ValueError("mesh2d_edge_nodes contains out-of-range node indices")
    mesh_edge_length = np.hypot(
        node_x[edge_nodes[:, 1]] - node_x[edge_nodes[:, 0]],
        node_y[edge_nodes[:, 1]] - node_y[edge_nodes[:, 0]],
    )
    boundary_mask = np.any(edge_faces < 0, axis=1)
    type2_boundary_mask = (edge_types_nc == 2) & boundary_mask
    type3_boundary_mask = (edge_types_nc == 3) & boundary_mask
    unexpected_boundary_mask = boundary_mask & ~np.isin(edge_types_nc, [2, 3])
    inflow_edge_ids = np.flatnonzero(type2_boundary_mask)

    # Use the external DEM when supplied, with the reference script's
    # coordinate matching and diagnostics. Otherwise retain the NetCDF-node
    # elevation fallback used by the original converter.
    elevation_report = None
    if dem_xyz is not None:
        dem_path = Path(dem_xyz)
        if not dem_path.exists():
            raise FileNotFoundError(dem_path)
        dem = tools["read_dem_xyz"](dem_path)
        face_elevation, elevation_report = tools["match_dem_to_faces"](dem, face_x, face_y)
    else:
        face_elevation = None

    # With a hydrograph, use the reference script to identify the actual
    # inflow edge. Wall boundaries never become graph/ghost edges.
    detected_inflow = None
    hydro = None
    boundary_touch_count = np.zeros(n_faces, dtype=int)
    if hydrograph is not None:
        hydro = tools["read_hydrograph"](Path(hydrograph))
    if hydro is not None:
        q_all = np.asarray(ds["mesh2d_q1"].values, dtype=float)
        n_edges = len(edge_faces_raw)
        if q_all.ndim != 2 or n_edges not in q_all.shape:
            raise ValueError("mesh2d_q1 must be 2D with one dimension matching mesh2d_edge_faces")
        if q_all.shape[1] == n_edges:
            q_all = q_all
        else:
            q_all = q_all.T
        time_seconds = tools["time_to_seconds"](ds["time"].values)
        edge_x = ds["mesh2d_edge_x"].values
        edge_y = ds["mesh2d_edge_y"].values
        if face_elevation is None:
            face_elevation = np.empty(n_faces, dtype=float)
            for face_id, face_node_ids in enumerate(face_nodes):
                valid = _as_zero_based_connectivity(face_node_ids, "mesh2d_face_nodes")
                valid = valid[valid >= 0]
                if len(valid) < 3 or np.any(valid >= len(node_z)):
                    raise ValueError(f"Invalid mesh2d_face_nodes connectivity for face {face_id}")
                face_elevation[face_id] = np.mean(node_z[valid])
        physical_lengths = mesh_edge_length
        _, boundary_edges, boundary_touch_count, _, _ = tools["build_edges"](
            edge_faces_raw,
            face_x,
            face_y,
            edge_x,
            edge_y,
            face_elevation,
            physical_lengths,
            edge_types_nc,
        )
        type2_boundary_edges = boundary_edges[boundary_edges["edge_type"] == 2]
        detected_inflow = tools["detect_inflow_boundary_edge"](
            type2_boundary_edges,
            q_all,
            time_seconds,
            hydro,
        )
        if detected_inflow is None:
            raise ValueError("Could not identify a type-2 inflow edge from the hydrograph")
        inflow_edge_ids = np.array([detected_inflow["edge_id_0based"]], dtype=int)

    wall_edge_ids = np.flatnonzero(type3_boundary_mask)
    ghost_edge_ids = np.r_[inflow_edge_ids, wall_edge_ids]
    n_ghost = len(ghost_edge_ids)
    print(
        f"Boundary classification: {len(inflow_edge_ids)} type-2 inflow edge(s), "
        f"{int(type3_boundary_mask.sum())} type-3 wall edge(s), "
        f"{int(unexpected_boundary_mask.sum())} other boundary edge(s)"
    )
    if expected_inflow_edges is not None and len(inflow_edge_ids) != expected_inflow_edges:
        raise ValueError(
            f"Expected {expected_inflow_edges} type-2 boundary/inflow edge(s), "
            f"but found {len(inflow_edge_ids)}. Use --expected-inflow-edges to override "
            "this dataset-specific validation."
        )

    node_geometries = [
        Point(float(x), float(y))
        for x, y in zip(face_x, face_y)
    ]
    edge_x = ds["mesh2d_edge_x"].values.astype(np.float64)
    edge_y = ds["mesh2d_edge_y"].values.astype(np.float64)
    ghost_face_ids = []
    for edge_id in ghost_edge_ids:
        valid_faces = edge_faces[edge_id][edge_faces[edge_id] >= 0]
        if len(valid_faces) != 1:
            raise ValueError(
                f"Boundary edge {edge_id} must have exactly one real face; "
                f"got {edge_faces[edge_id].tolist()}"
            )
        ghost_face_ids.append(valid_faces[0])
    ghost_face_ids = np.asarray(ghost_face_ids, dtype=np.int64)
    for edge_id in ghost_edge_ids:
        # Keep ghost-node coordinates at the adjacent real face centre. Ghost
        # nodes are removed before final training features are used, and this
        # keeps temporary DEM sampling inside the mesh/DEM footprint. The
        # exported boundary edge geometry still uses the physical edge
        # midpoint below.
        face_id = ghost_face_ids[len(node_geometries) - n_faces]
        node_geometries.append(Point(float(face_x[face_id]), float(face_y[face_id])))

    if face_elevation is None:
        face_elevation = np.empty(n_faces, dtype=np.float64)
    for face_id, face_node_ids in enumerate(face_nodes):
        valid = _as_zero_based_connectivity(face_node_ids, "mesh2d_face_nodes")
        valid = valid[valid >= 0]
        if len(valid) < 3 or np.any(valid >= len(node_z)):
            raise ValueError(f"Invalid mesh2d_face_nodes connectivity for face {face_id}")
        if dem_xyz is None:
            face_elevation[face_id] = np.mean(node_z[valid])

    nodes_gdf = gpd.GeoDataFrame(
        {
            "X": np.r_[face_x, face_x[ghost_face_ids]],
            "Y": np.r_[face_y, face_y[ghost_face_ids]],
            "Elevation1": np.r_[face_elevation, face_elevation[ghost_face_ids]],
            "node_type": np.r_[
                np.ones(n_faces, dtype=np.int64),
                np.r_[
                    np.full(len(inflow_edge_ids), 2, dtype=np.int64),
                    np.full(len(wall_edge_ids), 3, dtype=np.int64),
                ],
            ],
            "face_id": np.r_[np.arange(n_faces, dtype=np.int64), ghost_face_ids],
        },
        geometry=node_geometries,
        crs=None,
    )

    nodes_file = nodes_dir / f"nodes_{run_id}.shp"
    nodes_gdf.to_file(nodes_file)

    print(f"Written: {nodes_file}")

    # ============================================================
    # 2. CELLS = mesh faces
    # ============================================================
    #
    # mSWE-GNN reads:
    #
    #     mesh2d_face_nodes - 1
    #
    # to get zero-based mesh-node IDs.
    #
    # Each face may have 3 or 4 vertices.
    #
    # The NetCDF uses NaN for unused entries in a face.
    # ============================================================

    # Convert to zero-based indexing exactly as mSWE-GNN does.
    #
    # We handle NaNs before converting to int.
    cell_geometries = []

    for face_id, nodes in enumerate(face_nodes):

        valid_nodes = _as_zero_based_connectivity(nodes, "mesh2d_face_nodes")
        valid_nodes = valid_nodes[valid_nodes >= 0]
        if len(valid_nodes) < 3 or np.any(valid_nodes >= len(node_x)):
            raise ValueError(f"Invalid mesh2d_face_nodes connectivity for face {face_id}")

        # Get the actual mesh vertex coordinates.
        polygon_node_x = node_x[valid_nodes]
        polygon_node_y = node_y[valid_nodes]

        coords = list(zip(polygon_node_x, polygon_node_y))

        # Close polygon if necessary.
        polygon = Polygon(coords)

        # Some mesh formats can technically produce invalid
        # polygons. We don't silently change them here.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        cell_geometries.append(polygon)

    cell_area_m2 = np.array([geom.area for geom in cell_geometries], dtype=np.float64)

    cells_gdf = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(n_faces + n_ghost, dtype=np.int64),
            "face_id": np.r_[np.arange(n_faces, dtype=np.int64), ghost_face_ids],
            "area_m2": np.r_[cell_area_m2, np.zeros(n_ghost, dtype=np.float64)],
        },
        # Ghost cells are removed before training, but the raw feature
        # construction needs one zero-area record per ghost node.
        geometry=cell_geometries + [cell_geometries[int(face_id)]
                                    for face_id in ghost_face_ids],
        crs=None,
    )

    cells_file = cells_dir / f"cells_{run_id}.shp"
    cells_gdf.to_file(cells_file)

    print(f"Written: {cells_file}")

    # ============================================================
    # 3. GNN EDGES = DUAL MESH EDGES
    # ============================================================
    #
    # mSWE-GNN does:
    #
    #     dual_edge_index =
    #         mesh2d_edge_faces.T.astype(int) - 1
    #
    # Then removes entries where either face is -1.
    #
    # Finally:
    #
    #     to_undirected(...)
    #
    # ============================================================

    # Keep internal edges and type-2 inflow edges. Closed boundaries are not
    # graph edges; type-2 edges are connected to the appended ghost node.
    internal_edge_ids = np.flatnonzero((edge_types_nc == 1) & np.all(edge_faces >= 0, axis=1))
    # Preserve the exact NetCDF/q1 edge order for the historical loader.
    selected_edge_ids = np.arange(len(edge_faces), dtype=np.int64)
    graph_faces = edge_faces[selected_edge_ids].copy()
    ghost_lookup = {int(edge_id): n_faces + offset for offset, edge_id in enumerate(ghost_edge_ids)}
    for edge_id in ghost_edge_ids:
        row = np.flatnonzero(selected_edge_ids == edge_id)[0]
        face_id = graph_faces[row, graph_faces[row] >= 0][0]
        # The original boundary-condition pipeline preserves a boundary edge
        # when the boundary ghost is already the source. If it is the target,
        # it reverses the edge and flips its dynamic flow. Store the verified
        # inflow edge in the final physical orientation directly.
        ghost_node = ghost_lookup[int(edge_id)]
        if edge_id in inflow_edge_ids:
            graph_faces[row] = [ghost_node, face_id]
        else:
            graph_faces[row] = [face_id, ghost_node]

    print(f"Raw mesh edges: {len(edge_faces)}")
    print(f"Internal cell-cell edges: {len(internal_edge_ids)}")

    # ============================================================
    # mSWE-GNN converts the dual graph to an undirected graph.
    #
    # Store each pair as (min_id, max_id), which removes the
    # distinction between:
    #
    #     [12, 31]
    #     [31, 12]
    #
    # ============================================================

    edge_records = []

    for edge_id, (face_a, face_b) in zip(selected_edge_ids, graph_faces):

        if face_a == face_b:
            continue

        # Keep one row per NetCDF edge: q1 is indexed by the primal edge.
        edge_records.append((int(face_a), int(face_b), int(edge_id)))

    undirected_edges = edge_records

    print(f"Unique undirected GNN edges: {len(undirected_edges)}")

    edge_geometries = []

    source_ids = []
    target_ids = []

    for source, target, nc_edge_id in undirected_edges:

        if source >= n_faces:
            source_point = Point(
                float(ds["mesh2d_edge_x"].values[nc_edge_id]),
                float(ds["mesh2d_edge_y"].values[nc_edge_id]),
            )
        else:
            source_point = Point(float(face_x[source]), float(face_y[source]))

        if target < n_faces:
            target_point = Point(float(face_x[target]), float(face_y[target]))
        else:
            target_point = Point(float(ds["mesh2d_edge_x"].values[nc_edge_id]),
                                 float(ds["mesh2d_edge_y"].values[nc_edge_id]))

        edge_geometries.append(
            LineString([source_point, target_point])
        )

        source_ids.append(source)
        target_ids.append(target)

    dual_edge_lengths = np.array([g.length for g in edge_geometries], dtype=np.float64)
    primal_edge_lengths = np.asarray([mesh_edge_length[nc_id]
                                      for _, _, nc_id in undirected_edges], dtype=np.float64)
    edge_source_elevation = np.asarray([
        face_elevation[target] if source >= n_faces else face_elevation[source]
        for source, target, _ in undirected_edges
    ])
    edge_target_elevation = np.asarray([
        face_elevation[target] if target < n_faces else face_elevation[source]
        for source, target, _ in undirected_edges
    ])

    edges_gdf = gpd.GeoDataFrame(
        {
            "from_node": np.asarray(source_ids, dtype=np.int64),
            "to_node": np.asarray(target_ids, dtype=np.int64),
            # `length` is the dual (face-centre) distance. `fc_length` is
            # the primal mesh-edge/interface length, matching mSWE-GNN.
            "length": dual_edge_lengths,
            "slope": np.divide(edge_source_elevation - edge_target_elevation,
                                dual_edge_lengths,
                                out=np.zeros_like(dual_edge_lengths),
                                where=dual_edge_lengths > 0),
            "fc_length": primal_edge_lengths,
            "edge_type": np.asarray([edge_types_nc[nc_id]
                                      for _, _, nc_id in undirected_edges], dtype=np.int64),
            "nc_edge_id": np.asarray([nc_id for _, _, nc_id in undirected_edges], dtype=np.int64),
        },
        geometry=edge_geometries,
        crs=None,
    )

    edges_file = edges_dir / f"edges_{run_id}.shp"
    edges_gdf.to_file(edges_file)

    print(f"Written: {edges_file}")

    verify_conversion(ds, nodes_gdf, cells_gdf, edges_gdf, n_faces, n_ghost)

    if hydro is None:
        raise ValueError("Physics verification requires a hydrograph")
    if "mesh2d_waterdepth" not in ds or "time" not in ds:
        raise ValueError("Physics verification requires mesh2d_waterdepth and time")
    depth = np.asarray(ds["mesh2d_waterdepth"].values, dtype=float)
    if depth.ndim != 2 or depth.shape[1] != n_faces:
        if depth.ndim == 2 and depth.shape[0] == n_faces:
            depth = depth.T
        else:
            raise ValueError("mesh2d_waterdepth does not match face/time dimensions")
    q_all = np.asarray(ds["mesh2d_q1"].values, dtype=float)
    if q_all.shape[1] != len(edge_faces_raw):
        q_all = q_all.T
    polygons_area = cells_gdf["area_m2"].to_numpy()[:n_faces]
    volume = depth * polygons_area.reshape(1, -1)
    report = tools["mass_balance_report"](
        volume,
        q_all,
        edge_faces_raw,
        tools["time_to_seconds"](ds["time"].values),
        boundary_touch_count,
    )
    report_path = output_root / f"verification_{run_id}.csv"
    report.to_csv(report_path, index=False)
    print(f"Physics verification written: {report_path}")
    if detected_inflow is not None:
        inflow_path = output_root / f"detected_inflow_{run_id}.json"
        inflow_path.write_text(json.dumps(detected_inflow, indent=2), encoding="utf-8")
        print(f"Detected inflow written: {inflow_path}")
    if elevation_report is not None:
        print(f"DEM alignment: {elevation_report}")

    ds.close()

    print("\nDone.")
    print(f"  Nodes: {nodes_file}")
    print(f"  Edges: {edges_file}")
    print(f"  Cells: {cells_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert mSWE-GNN runs using the historical-loader workflow."
    )

    parser.add_argument("--input-dir", type=Path, default=DEFAULT_SIMULATION_DIR)
    parser.add_argument("--dem-dir", type=Path, default=DEFAULT_DEM_DIR)
    parser.add_argument("--hydrograph-dir", type=Path, default=DEFAULT_HYDROGRAPH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--run-id",
        type=int,
        help="Simulation/run ID for a single input file"
    )

    parser.add_argument(
        "--start-run-id",
        type=int,
        help="First Simulation/run ID for batch conversion"
    )

    parser.add_argument(
        "--end-run-id",
        type=int,
        help="Last Simulation/run ID for batch conversion"
    )

    parser.add_argument(
        "--expected-inflow-edges",
        type=int,
        default=1,
        help="Expected number of type-2 boundary edges per run (default: 1; use -1 to disable).",
    )

    args = parser.parse_args()

    expected = None if args.expected_inflow_edges < 0 else args.expected_inflow_edges
    if args.run_id is not None:
        if args.start_run_id is not None or args.end_run_id is not None:
            parser.error("use either --run-id or --start-run-id/--end-run-id")
        convert_mesh(
            args.input_dir / f"output_{args.run_id}_map.nc",
            args.output,
            args.run_id,
            expected_inflow_edges=expected,
            dem_xyz=args.dem_dir / f"DEM_{args.run_id}.xyz",
            hydrograph=args.hydrograph_dir / f"Hydrograph_{args.run_id}.txt",
        )
        return

    if args.start_run_id is None or args.end_run_id is None:
        parser.error("provide --run-id or both --start-run-id and --end-run-id")
    convert_batch(
        args.input_dir,
        args.output,
        args.start_run_id,
        args.end_run_id,
        expected_inflow_edges=expected,
        dem_dir=args.dem_dir,
        hydrograph_dir=args.hydrograph_dir,
    )


if __name__ == "__main__":
    main()
