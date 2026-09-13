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

    face_x = ds["mesh2d_face_x"].values
    face_y = ds["mesh2d_face_y"].values

    n_faces = len(face_x)

    print(f"Number of GNN nodes / cells: {n_faces}")

    node_geometries = [
        Point(float(x), float(y))
        for x, y in zip(face_x, face_y)
    ]

    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": np.arange(n_faces, dtype=np.int64),
            "face_id": np.arange(n_faces, dtype=np.int64),
            "x": face_x,
            "y": face_y,
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

    face_nodes = ds["mesh2d_face_nodes"].values

    # Convert to zero-based indexing exactly as mSWE-GNN does.
    #
    # We handle NaNs before converting to int.
    cell_geometries = []

    for face_id, nodes in enumerate(face_nodes):

        valid_nodes = nodes[~np.isnan(nodes)].astype(np.int64) - 1

        # Get the actual mesh vertex coordinates.
        node_x = ds["mesh2d_node_x"].values[valid_nodes]
        node_y = ds["mesh2d_node_y"].values[valid_nodes]

        coords = list(zip(node_x, node_y))

        # Close polygon if necessary.
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        polygon = Polygon(coords)

        # Some mesh formats can technically produce invalid
        # polygons. We don't silently change them here.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        cell_geometries.append(polygon)

    cells_gdf = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(n_faces, dtype=np.int64),
            "face_id": np.arange(n_faces, dtype=np.int64),
        },
        geometry=cell_geometries,
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

    edge_faces = ds["mesh2d_edge_faces"].values

    # Convert D-Hydro indexing to zero-based indexing.
    edge_faces = edge_faces.astype(np.int64) - 1

    # Remove boundary edges.
    #
    # A boundary mesh edge has one face and -1 for the other.
    valid_mask = np.all(edge_faces >= 0, axis=1)

    internal_edges = edge_faces[valid_mask]

    print(f"Raw mesh edges: {len(edge_faces)}")
    print(f"Internal cell-cell edges: {len(internal_edges)}")

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

    for face_a, face_b in internal_edges:

        if face_a == face_b:
            continue

        a = int(min(face_a, face_b))
        b = int(max(face_a, face_b))

        undirected_edges.add((a, b))

    undirected_edges = sorted(undirected_edges)

    print(f"Unique undirected GNN edges: {len(undirected_edges)}")

    edge_geometries = []

    source_ids = []
    target_ids = []

    for source, target in undirected_edges:

        source_point = Point(
            float(face_x[source]),
            float(face_y[source])
        )

        target_point = Point(
            float(face_x[target]),
            float(face_y[target])
        )

        edge_geometries.append(
            LineString([source_point, target_point])
        )

        source_ids.append(source)
        target_ids.append(target)

    edges_gdf = gpd.GeoDataFrame(
        {
            "edge_id": np.arange(
                len(undirected_edges),
                dtype=np.int64
            ),
            "source": np.asarray(source_ids, dtype=np.int64),
            "target": np.asarray(target_ids, dtype=np.int64),
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