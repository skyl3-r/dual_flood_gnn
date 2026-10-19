"""Export a cached hierarchical graph for inspection in QGIS.

The hierarchy is built after mSWE-GNN ghost/boundary-node processing.  This
script therefore uses raw node rows of type 1 (normal) followed by type 2
(boundary), which is the ordering used by the processed graph.

Example:
    python visualize_hierarchy.py --event 1 --out visualization/event_1.gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString


def export_event(event: int, out_path: Path, root: Path) -> None:
    processed = root / "processed"
    raw = root / "raw"
    static_path = processed / f"static_values_event_{event}.npz"
    if not static_path.exists():
        raise FileNotFoundError(f"Cached hierarchy not found: {static_path}")

    values = np.load(static_path)
    cluster = values["cluster"].astype(np.int64)
    edge_index = values["edge_index"].astype(np.int64)
    coarse_edge_index = values["coarse_edge_index"].astype(np.int64)
    num_supernodes = int(values["num_supernodes"])

    nodes_path = raw / "Geometry" / "Nodes" / f"nodes_{event}.shp"
    cells_path = raw / "Geometry" / "Cells" / f"cells_{event}.shp"
    nodes = gpd.read_file(nodes_path)
    cells = gpd.read_file(cells_path)

    # BoundaryCondition.remove() drops type 3 ghost nodes and apply() appends
    # type 2 boundary nodes. This exactly reconstructs processed graph order.
    retained = nodes[ nodes["node_type"].isin([1, 2]) ].copy()
    retained = retained.reset_index(drop=False).rename(columns={"index": "raw_idx"})
    if len(retained) != len(cluster):
        raise ValueError(
            f"Processed graph has {len(cluster)} nodes, but raw type 1/2 nodes "
            f"provide {len(retained)}. Check the dataset cache and event id."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    retained["node_id"] = np.arange(len(retained), dtype=np.int64)
    retained["cluster"] = cluster
    retained["is_boundary"] = retained["node_type"].eq(2)
    retained.to_file(out_path, layer="fine_nodes", driver="GPKG")

    # The cell layer has the same raw ordering as the node layer.
    retained_raw_indices = retained["raw_idx"].to_numpy()
    retained_cells = cells.iloc[retained_raw_indices].copy().reset_index(drop=True)
    retained_cells["node_id"] = np.arange(len(retained_cells), dtype=np.int64)
    retained_cells["cluster"] = cluster
    retained_cells.to_file(out_path, layer="cluster_cells", driver="GPKG")

    # Dissolved polygons are especially useful for seeing the METIS partition.
    clusters = retained_cells[["cluster", "geometry"]].dissolve(by="cluster", as_index=False)
    clusters["num_fine_nodes"] = np.bincount(cluster, minlength=num_supernodes)
    clusters.to_file(out_path, layer="supernode_regions", driver="GPKG")

    # Area-weighted centroids, matching data/hierarchy.py. Raw cell area is
    # used here because the cached static node features may be normalized.
    areas = retained_cells.geometry.area.to_numpy(dtype=float)
    # Prefer the dataset's area attribute when available; geometry area may be
    # zero for an invalid/no-CRS mesh.
    if "area_m2" in retained_cells:
        areas = retained_cells["area_m2"].to_numpy(dtype=float)
    areas = np.abs(areas)
    areas[areas == 0] = 1.0
    xy = np.column_stack([retained.geometry.x, retained.geometry.y])
    centroids = np.zeros((num_supernodes, 2), dtype=float)
    weights = np.bincount(cluster, weights=areas, minlength=num_supernodes)
    np.add.at(centroids, cluster, xy * areas[:, None])
    centroids /= np.maximum(weights[:, None], 1e-12)

    supernodes = gpd.GeoDataFrame(
        {
            "supernode": np.arange(num_supernodes, dtype=np.int64),
            "num_fine": np.bincount(cluster, minlength=num_supernodes),
            "area": weights,
        },
        geometry=gpd.points_from_xy(centroids[:, 0], centroids[:, 1]),
        crs=nodes.crs,
    )
    supernodes.to_file(out_path, layer="supernodes", driver="GPKG")

    coarse_lines = []
    for source, target in coarse_edge_index.T:
        a = centroids[source]
        b = centroids[target]
        coarse_lines.append(LineString([a, b]))
    coarse_edges = gpd.GeoDataFrame(
        {
            "source": coarse_edge_index[0],
            "target": coarse_edge_index[1],
        },
        geometry=coarse_lines,
        crs=nodes.crs,
    )
    coarse_edges.to_file(out_path, layer="supernode_edges", driver="GPKG")

    # Fine graph edges are optional but useful as a faint background layer.
    fine_lines = [
        LineString([xy[source], xy[target]])
        for source, target in edge_index.T
    ]
    fine_edges = gpd.GeoDataFrame(
        {"source": edge_index[0], "target": edge_index[1]},
        geometry=fine_lines,
        crs=nodes.crs,
    )
    fine_edges.to_file(out_path, layer="fine_edges", driver="GPKG")

    print(f"Wrote {out_path}")
    print(f"  fine nodes: {len(retained)}")
    print(f"  supernodes: {num_supernodes}")
    print(f"  supernode edges: {len(coarse_edges)}")
    print("  layers: fine_nodes, cluster_cells, supernode_regions, supernodes, supernode_edges, fine_edges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=int, required=True, help="Event/run number, e.g. 1")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_mswegnn/datasets"),
        help="Dataset root containing raw/ and processed/",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .gpkg path")
    args = parser.parse_args()
    export_event(args.event, args.out, args.root)


if __name__ == "__main__":
    main()
