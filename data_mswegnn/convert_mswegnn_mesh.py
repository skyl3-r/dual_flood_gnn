#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def convert_batch(input_dir, output_root, start_run_id, end_run_id):
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
        convert_mesh(nc_file, output_root, run_id)


def convert_mesh(nc_file, output_root, run_id):
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

    nodes_dir = output_root / "Nodes"
    edges_dir = output_root / "Edges"
    cells_dir = output_root / "Cells"

    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {nc_file}")

    ds = xr.open_dataset(nc_file)

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
    edge_faces = ds["mesh2d_edge_faces"].values.astype(np.int64) - 1
    edge_types_nc = ds["mesh2d_edge_type"].values.astype(np.int64)
    edge_nodes = ds["mesh2d_edge_nodes"].values.astype(np.int64) - 1
    mesh_edge_length = np.hypot(
        node_x[edge_nodes[:, 1]] - node_x[edge_nodes[:, 0]],
        node_y[edge_nodes[:, 1]] - node_y[edge_nodes[:, 0]],
    )
    bc_edge_ids = np.flatnonzero((edge_types_nc == 2) & np.any(edge_faces < 0, axis=1))
    n_ghost = len(bc_edge_ids)

    node_geometries = [
        Point(float(x), float(y))
        for x, y in zip(face_x, face_y)
    ]
    ghost_face_ids = edge_faces[bc_edge_ids, 1].copy()
    for edge_id, face_id in zip(bc_edge_ids, ghost_face_ids):
        node_geometries.append(Point(float(face_x[face_id]), float(face_y[face_id])))

    face_elevation = np.empty(n_faces, dtype=np.float64)
    face_nodes = ds["mesh2d_face_nodes"].values
    for face_id, face_node_ids in enumerate(face_nodes):
        valid = face_node_ids[np.isfinite(face_node_ids)].astype(np.int64) - 1
        face_elevation[face_id] = np.mean(node_z[valid])

    nodes_gdf = gpd.GeoDataFrame(
        {
            "X": np.r_[face_x, face_x[ghost_face_ids]],
            "Y": np.r_[face_y, face_y[ghost_face_ids]],
            "Elevation1": np.r_[face_elevation, face_elevation[ghost_face_ids]],
            "node_type": np.r_[np.ones(n_faces, dtype=np.int64),
                                np.full(n_ghost, 2, dtype=np.int64)],
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

        valid_nodes = nodes[~np.isnan(nodes)].astype(np.int64) - 1

        # Get the actual mesh vertex coordinates.
        polygon_node_x = node_x[valid_nodes]
        polygon_node_y = node_y[valid_nodes]

        coords = list(zip(polygon_node_x, polygon_node_y))

        # Close polygon if necessary.
        if coords[0] != coords[-1]:
            coords.append(coords[0])

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
    selected_edge_ids = np.r_[internal_edge_ids, bc_edge_ids]
    graph_faces = edge_faces[selected_edge_ids].copy()
    for ghost_offset, edge_id in enumerate(bc_edge_ids):
        row = np.flatnonzero(selected_edge_ids == edge_id)[0]
        face_id = graph_faces[row, graph_faces[row] >= 0][0]
        graph_faces[row] = [face_id, n_faces + ghost_offset]

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

    undirected_edges = set()

    for edge_id, (face_a, face_b) in zip(selected_edge_ids, graph_faces):

        if face_a == face_b:
            continue

        # Keep one row per NetCDF edge: q1 is indexed by the primal edge.
        undirected_edges.add((int(face_a), int(face_b), int(edge_id)))

    undirected_edges = sorted(undirected_edges)

    print(f"Unique undirected GNN edges: {len(undirected_edges)}")

    edge_geometries = []

    source_ids = []
    target_ids = []

    for source, target, nc_edge_id in undirected_edges:

        source_point = Point(
            float(face_x[source]),
            float(face_y[source])
        )

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
    edge_source_elevation = np.asarray([face_elevation[source] for source, _, _ in undirected_edges])
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
            "edge_type": np.asarray([2 if edge_types_nc[nc_id] == 2 else 1
                                      for _, _, nc_id in undirected_edges], dtype=np.int64),
            "nc_edge_id": np.asarray([nc_id for _, _, nc_id in undirected_edges], dtype=np.int64),
        },
        geometry=edge_geometries,
        crs=None,
    )

    edges_file = edges_dir / f"edges_{run_id}.shp"
    edges_gdf.to_file(edges_file)

    print(f"Written: {edges_file}")

    ds.close()

    print("\nDone.")
    print(f"  Nodes: {nodes_file}")
    print(f"  Edges: {edges_file}")
    print(f"  Cells: {cells_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Convert mSWE-GNN D-Hydro NetCDF mesh to shapefiles."
    )

    parser.add_argument(
        "--input",
        help="Path to a single output_{run_id}_map.nc"
    )

    parser.add_argument(
        "--input-dir",
        help="Directory containing output_{run_id}_map.nc files for batch conversion"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )

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

    args = parser.parse_args()

    # Single-file mode: --input + --run-id
    if args.input:
        if Path(args.input).is_dir():
            parser.error(
                "--input must point to one output_<run_id>_map.nc file; "
                "use --input-dir for a directory"
            )
        if args.run_id is None:
            parser.error("--run-id is required when using --input")
        if args.input_dir:
            parser.error("use either --input or --input-dir, not both")
        if args.start_run_id is not None or args.end_run_id is not None:
            parser.error("batch range flags are not allowed with --input")

        convert_mesh(
            args.input,
            args.output,
            args.run_id
        )
        return

    # Batch mode: --input-dir + --start-run-id + --end-run-id
    if args.input_dir:
        if args.run_id is not None:
            parser.error("--run-id is not used in batch mode")
        if args.start_run_id is None or args.end_run_id is None:
            parser.error("--start-run-id and --end-run-id are required when using --input-dir")

        convert_batch(
            args.input_dir,
            args.output,
            args.start_run_id,
            args.end_run_id,
        )
        return

    parser.error("provide either --input for one file or --input-dir with --start-run-id/--end-run-id for a batch")


if __name__ == "__main__":
    main()
